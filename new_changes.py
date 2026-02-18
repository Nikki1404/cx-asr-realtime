import argparse
import asyncio
import csv
import json
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import jiwer
import websockets

TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

@dataclass
class BenchResult:
    subset: str
    file: str
    backend: str

    latency_ms: int
    audio_sec_sent: float

    ref_text: str
    hyp_text: str
    wer: Optional[float]

    error: Optional[str] = None

def get_reference_text(wav_path: Path, wav_root: Path, raw_root: Path) -> str:
    utt_id = wav_path.stem
    rel = wav_path.relative_to(wav_root)
    subset, spk, chap = rel.parts[:3]

    trans = raw_root / subset / spk / chap / f"{spk}-{chap}.trans.txt"
    if not trans.exists():
        return ""

    with open(trans, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(utt_id + " "):
                return line.strip().split(" ", 1)[1]
    return ""


def iter_wav_chunks(path: Path):
    with wave.open(str(path), "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            yield data


def silence_bytes(sec: float) -> bytes:
    return b"\x00\x00" * int(TARGET_SR * sec)


def parse_pause_spec(spec: str) -> List[Tuple[float, float]]:
    """
    Example:
    "2.0:0.5,5.0:1.0"
    """
    if not spec:
        return []

    out = []
    for p in spec.split(","):
        at, dur = p.split(":")
        out.append((float(at), float(dur)))
    return sorted(out)

async def transcribe_ws(
    *,
    url: str,
    backend: str,
    wav_path: Path,
    pause_plan: List[Tuple[float, float]],
) -> Tuple[str, float, int]:

    async with websockets.connect(url, max_size=None) as ws:

        # Send config
        await ws.send(json.dumps({
            "type": "config",
            "backend": backend,
            "sampling_rate": TARGET_SR,
            "chunk_ms": CHUNK_MS,
        }))

        finals = []
        final_received = asyncio.Event()

        async def receiver():
            async for msg in ws:
                if isinstance(msg, str):
                    obj = json.loads(msg)
                    if obj.get("type") == "final":
                        if obj.get("text"):
                            finals.append(obj["text"].strip())
                        final_received.set()

        recv_task = asyncio.create_task(receiver())

        frames_sent = 0
        audio_sec_sent = 0.0
        pause_idx = 0

        t_start = time.time()

        for chunk in iter_wav_chunks(wav_path):

            # Inject pauses
            while pause_idx < len(pause_plan) and audio_sec_sent >= pause_plan[pause_idx][0]:
                dur = pause_plan[pause_idx][1]
                await ws.send(silence_bytes(dur))
                frames_sent += int(TARGET_SR * dur)
                audio_sec_sent = frames_sent / TARGET_SR
                pause_idx += 1

            await ws.send(chunk)

            frames_sent += len(chunk) // 2
            audio_sec_sent = frames_sent / TARGET_SR

        # Final flush
        await ws.send(silence_bytes(0.6))
        await ws.send(b"")

        await asyncio.wait_for(final_received.wait(), timeout=30)

        latency_ms = int((time.time() - t_start) * 1000)

        await ws.close()
        recv_task.cancel()

        return " ".join(finals), audio_sec_sent, latency_ms


async def process_one(
    *,
    wav_path: Path,
    backend: str,
    url: str,
    wav_root: Path,
    raw_root: Path,
    pause_plan: List[Tuple[float, float]],
) -> BenchResult:

    subset = wav_path.relative_to(wav_root).parts[0]
    ref = get_reference_text(wav_path, wav_root, raw_root)

    try:
        hyp, audio_sec, latency_ms = await transcribe_ws(
            url=url,
            backend=backend,
            wav_path=wav_path,
            pause_plan=pause_plan,
        )

        wer = None
        if ref and hyp:
            wer = jiwer.wer(ref, hyp, transform, transform)
            wer = round(float(wer), 4)

        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,
            latency_ms=latency_ms,
            audio_sec_sent=round(audio_sec, 3),
            ref_text=ref,
            hyp_text=hyp,
            wer=wer,
        )

    except Exception as e:
        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,
            latency_ms=0,
            audio_sec_sent=0.0,
            ref_text=ref,
            hyp_text="",
            wer=None,
            error=str(e),
        )


async def process_one_triple(**kwargs):
    return await asyncio.gather(
        process_one(backend="nemotron", **kwargs),
        process_one(backend="whisper", **kwargs),
        process_one(backend="google", **kwargs),
    )

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")
    p.add_argument("--data-wav-root", required=True)
    p.add_argument("--raw-librispeech-root", required=True)
    p.add_argument("--max-files", type=int, default=20)
    p.add_argument("--inject-pause", default="")
    args = p.parse_args()

    wav_root = Path(args.data_wav_root)
    raw_root = Path(args.raw_librispeech_root)
    wavs = sorted(wav_root.rglob("*.wav"))[: args.max_files]

    pause_plan = parse_pause_spec(args.inject_pause)
    results: List[BenchResult] = []

    for wav in wavs:
        triple = await process_one_triple(
            wav_path=wav,
            url=args.url,
            wav_root=wav_root,
            raw_root=raw_root,
            pause_plan=pause_plan,
        )

        results.extend(triple)

        for r in triple:
            print(
                f"{r.backend:9s} {r.file} "
                f"latency={r.latency_ms}ms wer={r.wer}"
            )

    out_csv = f"bench_realtime_triple_{uuid.uuid4().hex[:8]}.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BenchResult.__dataclass_fields__.keys())
        for r in results:
            w.writerow(r.__dict__.values())

    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    asyncio.run(main())

nemotron	2263	4.815	NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER	nor is mister Quilter's manner less interesting than his matter	0	
whisper	4352	4.815	NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER	Nor is Mr. Quilter's manner less interesting than his matter.	0.1	
google	4349	4.815	NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER	Nor is Mr. Quilter's mannerless interesting than his matter.	0.3	


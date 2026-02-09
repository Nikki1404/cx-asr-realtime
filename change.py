"""
Client-side realtime ASR benchmarking (DUAL backend)

Features:
- Parallel Nemotron + Whisper per WAV
- Fixed truncations: 6,7,8,9,10 seconds
- Optional pause injection
- LibriSpeech-compatible WER
- Realtime or FAST mode
- CSV output

LibriSpeech layout:
datasets/data/wav/{dev-clean,dev-other,test-clean,test-other}/<spk>/<chap>/<utt>.wav
datasets/data/raw/LibriSpeech/{subset}/{spk}/{chap}/{spk}-{chap}.trans.txt
"""

import argparse
import asyncio
import csv
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import jiwer
import numpy as np
import websockets

# =========================
# AUDIO CONFIG
# =========================
TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

TRUNCATION_LIST = [6, 7, 8, 9, 10]

# =========================
# TEXT NORMALIZATION
# =========================
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

# =========================
# RESULT SCHEMA
# =========================
@dataclass
class BenchResult:
    subset: str
    file: str
    backend: str
    trunc_sec: int

    audio_sec_sent: float
    ref_text: str
    hyp_text: str
    wer: Optional[float]

    client_connect_ms: int
    client_ttft_ms: Optional[int]
    client_ttf_ms: Optional[int]

    server_reason: Optional[str]
    server_ttf_ms: Optional[int]
    server_ttft_ms: Optional[int]
    server_audio_ms: Optional[int]
    server_rtf: Optional[float]

    error: Optional[str] = None

# =========================
# REFERENCE LOOKUP
# =========================
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

# =========================
# WAV HELPERS
# =========================
def read_wav_info(path: Path):
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()

def iter_wav_chunks(path: Path):
    with wave.open(str(path), "rb") as wf:
        while True:
            d = wf.readframes(CHUNK_FRAMES)
            if not d:
                break
            yield d

def silence_bytes(sec: float) -> bytes:
    return b"\x00\x00" * int(TARGET_SR * sec)

# =========================
# PAUSE PARSER
# =========================
def parse_pause_spec(spec: str) -> List[Tuple[float, float]]:
    if not spec:
        return []
    out = []
    for p in spec.split(","):
        at, dur = p.split(":")
        out.append((float(at), float(dur)))
    return sorted(out)

# =========================
# WS TRANSCRIPTION
# =========================
async def transcribe_ws(
    *,
    url: str,
    backend: str,
    wav_path: Path,
    truncate_sec: int,
    pause_plan: List[Tuple[float, float]],
    realtime: bool,
) -> Dict:

    t_conn0 = time.time()
    async with websockets.connect(url, max_size=None) as ws:
        connect_ms = int((time.time() - t_conn0) * 1000)

        # send config (matches your client)
        await ws.send(json.dumps({
            "type": "config",
            "backend": backend,
            "sampling_rate": TARGET_SR,
            "chunk_ms": CHUNK_MS,
        }))

        final_obj = None
        finals = []
        t_first_partial = None
        t_final = None

        async def receiver():
            nonlocal final_obj, t_first_partial, t_final
            while True:
                try:
                    msg = await ws.recv()
                except:
                    return
                obj = json.loads(msg)
                if obj.get("type") == "partial" and t_first_partial is None:
                    t_first_partial = time.time()
                elif obj.get("type") == "final":
                    final_obj = obj
                    if t_final is None:
                        t_final = time.time()
                    if obj.get("text"):
                        finals.append(obj["text"].strip())

        recv_task = asyncio.create_task(receiver())

        sr, ch, sw = read_wav_info(wav_path)
        assert sr == 16000 and ch == 1 and sw == 2

        frames_sent = 0
        audio_sec_sent = 0.0
        pause_idx = 0
        t_stream = time.time()

        for chunk in iter_wav_chunks(wav_path):
            if audio_sec_sent >= truncate_sec:
                break

            while pause_idx < len(pause_plan) and audio_sec_sent >= pause_plan[pause_idx][0]:
                dur = pause_plan[pause_idx][1]
                await ws.send(silence_bytes(dur))
                if realtime:
                    await asyncio.sleep(dur)
                frames_sent += int(TARGET_SR * dur)
                audio_sec_sent = frames_sent / TARGET_SR
                pause_idx += 1

            await ws.send(chunk)
            frames_sent += len(chunk) // 2
            audio_sec_sent = frames_sent / TARGET_SR

            if realtime:
                await asyncio.sleep(SLEEP_SEC)

        await ws.send(silence_bytes(0.8))
        if realtime:
            await asyncio.sleep(0.8)
        await ws.send(b"")

        t_wait = time.time()
        while final_obj is None and time.time() - t_wait < 30:
            await asyncio.sleep(0.05)

        await ws.close()
        try:
            await asyncio.wait_for(recv_task, 1.0)
        except:
            pass

        return {
            "audio_sec": audio_sec_sent,
            "connect_ms": connect_ms,
            "client_ttft_ms": int((t_first_partial - t_stream) * 1000) if t_first_partial else None,
            "client_ttf_ms": int((t_final - t_stream) * 1000) if t_final else None,
            "final_text": " ".join(finals),
            "server": final_obj,
        }

# =========================
# SINGLE BACKEND
# =========================
async def process_one(
    *,
    wav_path: Path,
    backend: str,
    truncate_sec: int,
    url: str,
    wav_root: Path,
    raw_root: Path,
    pause_plan: List[Tuple[float, float]],
    realtime: bool,
) -> BenchResult:

    subset = wav_path.relative_to(wav_root).parts[0]
    ref = get_reference_text(wav_path, wav_root, raw_root)

    try:
        out = await transcribe_ws(
            url=url,
            backend=backend,
            wav_path=wav_path,
            truncate_sec=truncate_sec,
            pause_plan=pause_plan,
            realtime=realtime,
        )

        hyp = out["final_text"]
        wer = jiwer.wer(ref, hyp, transform, transform) if ref and hyp else None
        s = out["server"] or {}

        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,
            trunc_sec=truncate_sec,
            audio_sec_sent=round(out["audio_sec"], 3),
            ref_text=ref,
            hyp_text=hyp,
            wer=round(float(wer), 4) if wer is not None else None,
            client_connect_ms=out["connect_ms"],
            client_ttft_ms=out["client_ttft_ms"],
            client_ttf_ms=out["client_ttf_ms"],
            server_reason=s.get("reason"),
            server_ttf_ms=s.get("ttf_ms"),
            server_ttft_ms=s.get("ttft_ms"),
            server_audio_ms=s.get("audio_ms"),
            server_rtf=s.get("rtf"),
        )

    except Exception as e:
        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,
            trunc_sec=truncate_sec,
            audio_sec_sent=0.0,
            ref_text=ref,
            hyp_text="",
            wer=None,
            client_connect_ms=0,
            client_ttft_ms=None,
            client_ttf_ms=None,
            server_reason=None,
            server_ttf_ms=None,
            server_ttft_ms=None,
            server_audio_ms=None,
            server_rtf=None,
            error=str(e),
        )

# =========================
# DUAL BACKEND
# =========================
async def process_one_dual(**kwargs):
    n, w = await asyncio.gather(
        process_one(backend="nemotron", **kwargs),
        process_one(backend="whisper", **kwargs),
    )
    return [n, w]

# =========================
# MAIN
# =========================
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")
    p.add_argument("--data-wav-root", required=True)
    p.add_argument("--raw-librispeech-root", required=True)
    p.add_argument("--max-files", type=int, default=30)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--inject-pause", default="")
    p.add_argument("--out", default="bench_dual_trunc_pause.csv")
    args = p.parse_args()

    wav_root = Path(args.data_wav_root)
    raw_root = Path(args.raw_librispeech_root)
    wavs = sorted(wav_root.rglob("*.wav"))[: args.max_files]

    pause_plan = parse_pause_spec(args.inject_pause)
    sem = asyncio.Semaphore(args.workers)
    results: List[BenchResult] = []

    async def run(wav, trunc):
        async with sem:
            return await process_one_dual(
                wav_path=wav,
                truncate_sec=trunc,
                url=args.url,
                wav_root=wav_root,
                raw_root=raw_root,
                pause_plan=pause_plan,
                realtime=not args.fast,
            )

    for trunc in TRUNCATION_LIST:
        print(f"\n=== Truncation {trunc}s | pause={pause_plan} ===")
        tasks = [asyncio.create_task(run(w, trunc)) for w in wavs]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            results.extend(res)
            for r in res:
                print(f"{r.backend:9s} {r.file} "
                      f"ttf={r.client_ttf_ms}ms wer={r.wer}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BenchResult.__dataclass_fields__.keys())
        for r in results:
            w.writerow(r.__dict__.values())

    print(f"\nSaved → {args.out}")

if __name__ == "__main__":
    asyncio.run(main())


python benchmarking.py ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\path\to\datasets\data\wav" ^
  --raw-librispeech-root "C:\path\to\datasets\data\raw\LibriSpeech" ^
  --max-files 30

python benchmarking.py ^
  --url ws://127.0.0.1:8002/asr/realtime-custom-vad ^
  --data-wav-root "C:\path\to\datasets\data\wav" ^
  --raw-librispeech-root "C:\path\to\datasets\data\raw\LibriSpeech" ^
  --inject-pause "2.0:0.6,5.0:1.0" ^
  --max-files 20


python benchmarking.py ^
  --url ws://127.0.0.1:8002/asr/realtime-custom-vad ^
  --data-wav-root "C:\path\to\datasets\data\wav" ^
  --raw-librispeech-root "C:\path\to\datasets\data\raw\LibriSpeech" ^
  --fast ^
  --max-files 30


python benchmarking.py ^
  --url ws://127.0.0.1:8002/asr/realtime-custom-vad ^
  --data-wav-root "C:\path\to\datasets\data\wav" ^
  --raw-librispeech-root "C:\path\to\datasets\data\raw\LibriSpeech" ^
  --workers 4 ^
  --max-files 20


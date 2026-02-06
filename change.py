# asr_realtime_benchmark.py
"""
Client-side realtime benchmarking for ASR-Realtime WebSocket server.

✅ Uses your existing LibriSpeech layout:
  - WAVs:  datasets/data/wav/{dev-clean,dev-other,test-clean,test-other}/<spk>/<chap>/<utt>.wav
  - Refs:  datasets/data/raw/LibriSpeech/{subset}/{spk}/{chap}/{spk}-{chap}.trans.txt

✅ Does NOT touch server code.
✅ Streams WAV as PCM16 chunks over WebSocket (mic-like).
✅ Sends first WS message: {"backend": "nemotron" | "whisper"}  (matches your server)
✅ Collects:
  - final transcript
  - server metrics (ttf_ms, ttft_ms, rtf, reason, etc.)
  - client wall timings (connect, t_first_partial, t_final)
✅ Computes WER using jiwer (same normalization style you use).
✅ Supports:
  - realtime pacing (default) or fast send
  - optional pause injection (silence) at fixed offsets
  - optional truncation tests (e.g., 1s..20s) without new files
  - concurrency (limited) without stressing server too much

Run examples:
  python asr_realtime_benchmark.py --backend nemotron --url ws://127.0.0.1:8002/ws/asr
  python asr_realtime_benchmark.py --backend whisper --fast
  python asr_realtime_benchmark.py --backend nemotron --max-files 50 --workers 2
  python asr_realtime_benchmark.py --backend nemotron --truncate-sec 1,2,5,10,20
  python asr_realtime_benchmark.py --backend nemotron --inject-pause "2.0:0.6,5.0:1.0"
"""

import argparse
import asyncio
import csv
import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import jiwer
import numpy as np
import websockets

# -------------------------
# Defaults (edit if needed)
# -------------------------
TARGET_SR = 16000
CHUNK_MS = 80  # 80ms chunks (good for Nemotron streaming stability)
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# -------------------------
# Text normalization (same spirit as yours)
# -------------------------
transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)

# -------------------------
# Data structures
# -------------------------
@dataclass
class BenchResult:
    subset: str
    file: str
    backend: str

    audio_sec_sent: float
    ref_text: str
    hyp_text: str

    wer: Optional[float]

    # Client wall timings
    client_connect_ms: int
    client_ttft_ms: Optional[int]   # time to first partial (client-side), Nemotron only
    client_ttf_ms: Optional[int]    # time to final (client-side)

    # Server metrics (if present)
    server_reason: Optional[str]
    server_ttf_ms: Optional[int]
    server_ttft_ms: Optional[int]
    server_audio_ms: Optional[int]
    server_rtf: Optional[float]
    server_chunks: Optional[int]
    server_preproc_ms: Optional[int]
    server_infer_ms: Optional[int]
    server_flush_ms: Optional[int]

    error: Optional[str] = None


# -------------------------
# LibriSpeech ref lookup (your logic, generalized to Paths)
# -------------------------
def get_reference_text(wav_path: Path, data_wav_root: Path, raw_librispeech_root: Path) -> str:
    """
    Extract reference transcription from LibriSpeech chapter-level *.trans.txt
    """
    utt_id = wav_path.stem

    rel = wav_path.relative_to(data_wav_root)
    # rel parts: subset / speaker / chapter / file.wav
    subset = rel.parts[0]
    speaker_id = rel.parts[1]
    chapter_id = rel.parts[2]

    trans_file = (
        raw_librispeech_root
        / subset
        / speaker_id
        / chapter_id
        / f"{speaker_id}-{chapter_id}.trans.txt"
    )

    if not trans_file.exists():
        return ""

    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(utt_id + " "):
                return line.strip().split(" ", 1)[1]

    return ""


# -------------------------
# WAV utilities
# -------------------------
def read_wav_pcm16_mono_16k(wav_path: Path) -> Tuple[int, int, int]:
    """
    Validate WAV is 16kHz mono PCM16.
    Returns (sr, channels, sampwidth_bytes).
    """
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    return sr, ch, sw


def iter_wav_chunks_pcm16(wav_path: Path, frames_per_chunk: int):
    """
    Yields PCM16 bytes chunks from WAV.
    """
    with wave.open(str(wav_path), "rb") as wf:
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            yield data


def pcm16_silence_bytes(duration_sec: float, sr: int = TARGET_SR) -> bytes:
    frames = int(sr * duration_sec)
    return b"\x00\x00" * frames


# -------------------------
# Pause injection parsing
# -------------------------
def parse_pause_spec(spec: str) -> List[Tuple[float, float]]:
    """
    "2.0:0.6,5.0:1.0" -> [(2.0,0.6),(5.0,1.0)]
    meaning: at t=2.0s of audio sent, inject 0.6s silence
    """
    if not spec:
        return []
    out = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        left, right = p.split(":")
        out.append((float(left), float(right)))
    out.sort(key=lambda x: x[0])
    return out


# -------------------------
# WebSocket run for 1 file
# -------------------------
async def transcribe_realtime_ws(
    url: str,
    backend: str,
    wav_path: Path,
    *,
    realtime: bool,
    truncate_sec: Optional[float],
    pause_plan: List[Tuple[float, float]],
    connect_timeout: float = 15.0,
) -> Dict:
    """
    Returns:
      {
        "final_text": str,
        "partials_seen": bool,
        "client_ttft_ms": Optional[int],
        "client_ttf_ms": Optional[int],
        "server_final_obj": Optional[dict],
        "audio_sec_sent": float
      }
    """
    t_connect0 = time.time()
    async with websockets.connect(url, max_size=None, open_timeout=connect_timeout) as ws:
        client_connect_ms = int((time.time() - t_connect0) * 1000)

        # Send init config (required by your server)
        await ws.send(json.dumps({"backend": backend}))

        # Receiver task
        final_obj = None
        partials_seen = False
        t_first_partial = None
        t_final = None
        finals: List[str] = []

        async def receiver():
            nonlocal final_obj, partials_seen, t_first_partial, t_final
            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    return
                obj = json.loads(msg)
                typ = obj.get("type")
                if typ == "partial":
                    partials_seen = True
                    if t_first_partial is None:
                        t_first_partial = time.time()
                elif typ == "final":
                    final_obj = obj
                    if t_final is None:
                        t_final = time.time()
                    txt = (obj.get("text") or "").strip()
                    if txt:
                        finals.append(txt)

        recv_task = asyncio.create_task(receiver())

        # Stream audio
        sr, ch, sw = read_wav_pcm16_mono_16k(wav_path)
        if not (sr == TARGET_SR and ch == 1 and sw == 2):
            raise ValueError(
                f"WAV must be 16kHz mono PCM16 for mic-like streaming. Got sr={sr}, ch={ch}, sw={sw}"
            )

        # plan pauses
        pause_plan = list(pause_plan)
        pause_idx = 0

        audio_frames_sent = 0
        audio_sec_sent = 0.0

        t_stream_start = time.time()

        for chunk in iter_wav_chunks_pcm16(wav_path, CHUNK_FRAMES):
            # Truncate: stop after N seconds of audio have been sent
            if truncate_sec is not None and audio_sec_sent >= truncate_sec:
                break

            # Inject pauses at requested audio offsets (based on audio_sec_sent)
            while pause_idx < len(pause_plan) and audio_sec_sent >= pause_plan[pause_idx][0]:
                pause_dur = pause_plan[pause_idx][1]
                silence = pcm16_silence_bytes(pause_dur, sr=TARGET_SR)
                await ws.send(silence)
                if realtime:
                    await asyncio.sleep(pause_dur)
                # update sent counters for silence too (important!)
                audio_frames_sent += int(TARGET_SR * pause_dur)
                audio_sec_sent = audio_frames_sent / TARGET_SR
                pause_idx += 1

            await ws.send(chunk)

            frames_in_chunk = len(chunk) // 2  # PCM16 mono
            audio_frames_sent += frames_in_chunk
            audio_sec_sent = audio_frames_sent / TARGET_SR

            if realtime:
                await asyncio.sleep(SLEEP_SEC)

        # send trailing silence + EOS to force finalize
        tail_sil = pcm16_silence_bytes(0.8, sr=TARGET_SR)
        await ws.send(tail_sil)
        if realtime:
            await asyncio.sleep(0.8)
        await ws.send(b"")  # EOS

        # Wait for final (with timeout)
        t_wait0 = time.time()
        while final_obj is None and (time.time() - t_wait0) < 60.0:
            await asyncio.sleep(0.05)

        # close
        try:
            await ws.close()
        except Exception:
            pass

        try:
            await asyncio.wait_for(recv_task, timeout=2.0)
        except Exception:
            pass

        client_ttft_ms = None
        client_ttf_ms = None
        if t_first_partial is not None:
            client_ttft_ms = int((t_first_partial - t_stream_start) * 1000)
        if t_final is not None:
            client_ttf_ms = int((t_final - t_stream_start) * 1000)

        final_text = " ".join([t for t in finals if t.strip()]).strip()

        return {
            "client_connect_ms": client_connect_ms,
            "final_text": final_text,
            "partials_seen": partials_seen,
            "client_ttft_ms": client_ttft_ms,
            "client_ttf_ms": client_ttf_ms,
            "server_final_obj": final_obj,
            "audio_sec_sent": float(audio_sec_sent),
        }


# -------------------------
# Process one wav end-to-end
# -------------------------
async def process_one(
    wav_path: Path,
    *,
    backend: str,
    url: str,
    data_wav_root: Path,
    raw_librispeech_root: Path,
    realtime: bool,
    truncate_sec: Optional[float],
    pause_plan: List[Tuple[float, float]],
) -> BenchResult:
    subset = wav_path.relative_to(data_wav_root).parts[0]
    ref = get_reference_text(wav_path, data_wav_root, raw_librispeech_root)

    try:
        ws_out = await transcribe_realtime_ws(
            url=url,
            backend=backend,
            wav_path=wav_path,
            realtime=realtime,
            truncate_sec=truncate_sec,
            pause_plan=pause_plan,
        )

        hyp = ws_out["final_text"]

        wer = None
        if ref and hyp:
            wer = jiwer.wer(
                ref,
                hyp,
                reference_transform=transform,
                hypothesis_transform=transform,
            )
            wer = round(float(wer), 4)

        final_obj = ws_out.get("server_final_obj") or {}

        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,

            audio_sec_sent=round(ws_out["audio_sec_sent"], 3),
            ref_text=ref,
            hyp_text=hyp,

            wer=wer,

            client_connect_ms=int(ws_out["client_connect_ms"]),
            client_ttft_ms=ws_out["client_ttft_ms"],
            client_ttf_ms=ws_out["client_ttf_ms"],

            server_reason=final_obj.get("reason"),
            server_ttf_ms=final_obj.get("ttf_ms"),
            server_ttft_ms=final_obj.get("ttft_ms"),
            server_audio_ms=final_obj.get("audio_ms"),
            server_rtf=final_obj.get("rtf"),
            server_chunks=final_obj.get("chunks"),
            server_preproc_ms=final_obj.get("model_preproc_ms"),
            server_infer_ms=final_obj.get("model_infer_ms"),
            server_flush_ms=final_obj.get("model_flush_ms"),

            error=None,
        )

    except Exception as e:
        return BenchResult(
            subset=subset,
            file=wav_path.name,
            backend=backend,

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
            server_chunks=None,
            server_preproc_ms=None,
            server_infer_ms=None,
            server_flush_ms=None,

            error=str(e),
        )


# -------------------------
# Main orchestration
# -------------------------
def discover_wavs(data_wav_root: Path) -> List[Path]:
    return sorted(list(data_wav_root.rglob("*.wav")))


async def run_benchmark(
    wavs: List[Path],
    *,
    backend: str,
    url: str,
    data_wav_root: Path,
    raw_librispeech_root: Path,
    workers: int,
    realtime: bool,
    truncate_sec: Optional[float],
    pause_plan: List[Tuple[float, float]],
) -> List[BenchResult]:
    sem = asyncio.Semaphore(workers)
    results: List[BenchResult] = []

    async def bound_task(w: Path):
        async with sem:
            r = await process_one(
                w,
                backend=backend,
                url=url,
                data_wav_root=data_wav_root,
                raw_librispeech_root=raw_librispeech_root,
                realtime=realtime,
                truncate_sec=truncate_sec,
                pause_plan=pause_plan,
            )
            return r

    tasks = [asyncio.create_task(bound_task(w)) for w in wavs]

    done = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        done += 1
        status = "[OK]" if not r.error else "[ERR]"
        print(f"{status} {done}/{len(wavs)} {r.backend} {r.file}  "
              f"audio_sent={r.audio_sec_sent}s  "
              f"WER={r.wer}  "
              f"client_ttf_ms={r.client_ttf_ms}  "
              f"server_ttf_ms={r.server_ttf_ms}  "
              f"err={r.error}")

    return results


def write_csv(results: List[BenchResult], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "subset",
            "file",
            "backend",
            "audio_sec_sent",
            "reference_text",
            "predicted_text",
            "wer",
            "client_connect_ms",
            "client_ttft_ms",
            "client_ttf_ms",
            "server_reason",
            "server_ttf_ms",
            "server_ttft_ms",
            "server_audio_ms",
            "server_rtf",
            "server_chunks",
            "server_preproc_ms",
            "server_infer_ms",
            "server_flush_ms",
            "error",
        ])
        for r in results:
            w.writerow([
                r.subset,
                r.file,
                r.backend,
                r.audio_sec_sent,
                r.ref_text,
                r.hyp_text,
                r.wer,
                r.client_connect_ms,
                r.client_ttft_ms,
                r.client_ttf_ms,
                r.server_reason,
                r.server_ttf_ms,
                r.server_ttft_ms,
                r.server_audio_ms,
                r.server_rtf,
                r.server_chunks,
                r.server_preproc_ms,
                r.server_infer_ms,
                r.server_flush_ms,
                r.error,
            ])


def parse_truncate_list(spec: str) -> List[Optional[float]]:
    """
    "1,2,5,10,20" -> [1.0,2.0,5.0,10.0,20.0]
    """
    if not spec:
        return [None]
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


async def main_async():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nemotron", "whisper"], required=True)
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")

    p.add_argument("--data-wav-root", required=True,
                   help="Path to datasets/data/wav (the root that contains dev-clean, test-clean etc.)")
    p.add_argument("--raw-librispeech-root", required=True,
                   help="Path to datasets/data/raw/LibriSpeech")

    p.add_argument("--out-dir", default=str(Path.cwd() / "bench_results"))

    p.add_argument("--max-files", type=int, default=50)
    p.add_argument("--workers", type=int, default=1, help="Keep low for realtime WS to avoid stressing server")
    p.add_argument("--fast", action="store_true", help="Disable realtime pacing (send as fast as possible)")

    p.add_argument("--truncate-sec", default="",
                   help="Comma list (seconds) to stop streaming early: e.g. 1,2,5,10,20. "
                        "If empty, stream full wav.")
    p.add_argument("--inject-pause", default="",
                   help='Inject silence while streaming. Format "atSec:durSec,atSec:durSec" e.g. "2.0:0.6,5.0:1.0"')

    args = p.parse_args()

    data_wav_root = Path(args.data_wav_root).resolve()
    raw_root = Path(args.raw_librispeech_root).resolve()

    if not data_wav_root.exists():
        raise FileNotFoundError(f"data-wav-root not found: {data_wav_root}")
    if not raw_root.exists():
        raise FileNotFoundError(f"raw-librispeech-root not found: {raw_root}")

    wavs = discover_wavs(data_wav_root)
    if args.max_files is not None:
        wavs = wavs[: args.max_files]

    pause_plan = parse_pause_spec(args.inject_pause)
    trunc_list = parse_truncate_list(args.truncate_sec)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run one pass per truncate setting (so you can compare latency vs audio length)
    all_results: List[BenchResult] = []
    for trunc in trunc_list:
        label = "full" if trunc is None else f"trunc{int(trunc)}s"
        out_csv = out_dir / f"asr_realtime_{args.backend}_{label}.csv"

        print("\n==============================")
        print(f"Backend: {args.backend}")
        print(f"URL: {args.url}")
        print(f"Mode: {'FAST' if args.fast else 'REALTIME'}")
        print(f"Truncate: {trunc}")
        print(f"Pause plan: {pause_plan}")
        print(f"Files: {len(wavs)} | workers={args.workers}")
        print(f"Output: {out_csv}")
        print("==============================\n")

        t0 = time.time()
        results = await run_benchmark(
            wavs,
            backend=args.backend,
            url=args.url,
            data_wav_root=data_wav_root,
            raw_librispeech_root=raw_root,
            workers=max(1, int(args.workers)),
            realtime=(not args.fast),
            truncate_sec=trunc,
            pause_plan=pause_plan,
        )
        dt = time.time() - t0

        write_csv(results, out_csv)
        all_results.extend(results)

        # quick summary
        ok = [r for r in results if not r.error]
        wers = [r.wer for r in ok if r.wer is not None]
        avg_wer = round(float(np.mean(wers)), 4) if wers else None
        p95_ttf = int(np.percentile([r.client_ttf_ms for r in ok if r.client_ttf_ms is not None], 95)) if ok else None

        print("\nSUMMARY")
        print(f"  Completed: {len(results)} in {dt:.2f}s")
        print(f"  Success:   {len(ok)}")
        print(f"  Avg WER:   {avg_wer}")
        print(f"  P95 client TTF(ms): {p95_ttf}")
        print(f"  Saved CSV: {out_csv}\n")

    # combined CSV (optional)
    combined = out_dir / f"asr_realtime_{args.backend}_combined.csv"
    write_csv(all_results, combined)
    print(f"Combined CSV saved: {combined}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()



''''''
python asr_realtime_benchmark.py ^
  --backend nemotron ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --max-files 50 ^
  --workers 1

python asr_realtime_benchmark.py ^
  --backend whisper ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --max-files 50 ^
  --workers 1 ^
  --fast

python asr_realtime_benchmark.py ^
  --backend nemotron ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --max-files 30 ^
  --truncate-sec 1,2,5,10,20

python asr_realtime_benchmark.py ^
  --backend nemotron ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --max-files 20 ^
  --inject-pause "2.0:0.6,5.0:1.0"

python asr_realtime_benchmark.py ^
  --backend whisper ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --inject-pause "2.0:0.6,5.0:1.0" ^
  --max-files 20

python asr_realtime_benchmark.py ^
  --backend whisper ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav" ^
  --raw-librispeech-root "C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech" ^
  --truncate-sec 1,2,5,10,20 ^
  --max-files 30


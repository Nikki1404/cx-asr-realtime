import asyncio
import argparse
import json
import os
import sys
import time
import wave
import tempfile

import numpy as np
import soundfile as sf
import resampy
import websockets

try:
    import sounddevice as sd
    HAS_MIC = True
except Exception:
    HAS_MIC = False

TARGET_SR = 16000

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# ---- Whisper UX state ----
last_audio_time = 0.0
is_whisper = False


# =========================
# 🔹 BACKEND SELECTION UI
# =========================
def choose_backend() -> str:
    print("\nChoose ASR backend:")
    print("  1) Nemotron (true streaming, partials)")
    print("  2) Whisper Turbo (batch, no partials)")
    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice == "1":
            return "nemotron"
        if choice == "2":
            return "whisper"
        print("Invalid choice. Please enter 1 or 2.")


def resample_to_16k(wav_path: str) -> str:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr, TARGET_SR)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, audio, TARGET_SR, subtype="PCM_16")
    return tmp.name


async def receiver(ws):
    finals = []
    while True:
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            return finals

        obj = json.loads(msg)
        typ = obj.get("type")

        if typ == "partial":
            txt = obj.get("text", "").replace("\n", " ")
            sys.stdout.write("\r[PARTIAL] " + txt[:160] + " " * 20)
            sys.stdout.flush()

        elif typ == "final":
            txt = (obj.get("text") or "").strip()
            print("\n[FINAL]", txt)
            finals.append(txt)

            print(
                "[SERVER_METRICS]",
                f"reason={obj.get('reason')}",
                f"ttf_ms={obj.get('ttf_ms')}",
                f"audio_ms={obj.get('audio_ms')}",
                f"rtf={obj.get('rtf')}",
                f"chunks={obj.get('chunks')}",
            )


async def whisper_status_printer():
    while True:
        await asyncio.sleep(0.4)
        if time.time() - last_audio_time < 1.0:
            sys.stdout.write("\r⏳ Whisper is transcribing…   ")
            sys.stdout.flush()


async def run_mic(ws):
    global last_audio_time

    if not HAS_MIC:
        raise RuntimeError("sounddevice not installed. Install: pip install sounddevice")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def cb(indata, frames, t, status):
        global last_audio_time
        last_audio_time = time.time()
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
        callback=cb,
    )
    stream.start()

    print("\n🎤 Speak freely. Pause to end sentences. Ctrl+C to exit.")

    async def sender():
        while True:
            blk = await q.get()
            if blk is None:
                return
            try:
                await ws.send(blk.tobytes())
            except websockets.exceptions.ConnectionClosed:
                return

    send_task = asyncio.create_task(sender())

    try:
        while True:
            await asyncio.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

        try:
            await ws.send(b"\x00\x00" * int(TARGET_SR * 0.8))
            await asyncio.sleep(0.8)
            await ws.send(b"")
        except Exception:
            pass

        await q.put(None)
        await send_task


async def main():
    global is_whisper

    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8002/ws/asr")
    p.add_argument("--mic", action="store_true", help="Use live microphone")
    args = p.parse_args()

    backend = choose_backend()
    is_whisper = backend == "whisper"

    async with websockets.connect(args.url, max_size=None) as ws:
        print(f"\n[INFO] Connected to {args.url}")
        print(f"[INFO] Selected backend: {backend}")

        # 🔑 Send backend selection FIRST
        await ws.send(json.dumps({
            "type": "config",
            "backend": backend
        }))

        recv_task = asyncio.create_task(receiver(ws))

        status_task = None
        if is_whisper:
            status_task = asyncio.create_task(whisper_status_printer())

        if args.mic:
            await run_mic(ws)

        finals = await recv_task

        if status_task:
            status_task.cancel()

    print("\nFULL TRANSCRIPT:")
    print(" ".join([t for t in finals if t.strip()]))


if __name__ == "__main__":
    asyncio.run(main())

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


# =========================
# Client-side state
# =========================
class ClientState:
    def __init__(self):
        self.received_partial = False
        self.received_final = False
        self.last_audio_ts = time.time()


# =========================
# Utilities
# =========================
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


# =========================
# Receiver
# =========================
async def receiver(ws, state: ClientState):
    finals = []

    while True:
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            return finals

        obj = json.loads(msg)
        typ = obj.get("type")

        if typ == "partial":
            state.received_partial = True
            txt = obj.get("text", "").replace("\n", " ")
            sys.stdout.write("\r[PARTIAL] " + txt[:160] + " " * 20)
            sys.stdout.flush()

        elif typ == "final":
            state.received_final = True
            txt = (obj.get("text") or "").strip()
            print("\n[FINAL]", txt)
            finals.append(txt)

            print("[SERVER_METRICS]",
                  f"reason={obj.get('reason')}",
                  f"ttft_ms={obj.get('ttft_ms')}",
                  f"ttf_ms={obj.get('ttf_ms')}",
                  f"audio_ms={obj.get('audio_ms')}",
                  f"rtf={obj.get('rtf')}",
                  f"chunks={obj.get('chunks')}",
                  f"preproc_ms={obj.get('model_preproc_ms')}",
                  f"infer_ms={obj.get('model_infer_ms')}",
                  f"flush_ms={obj.get('model_flush_ms')}",
                  )


# =========================
# Whisper UX status loop
# =========================
async def whisper_status_loop(state: ClientState):
    """
    Shows 'Transcribing…' ONLY when:
    - No partials are coming (Whisper)
    - Audio has stopped recently
    - Final has not arrived yet
    """
    while not state.received_final:
        await asyncio.sleep(0.3)

        if not state.received_partial:
            idle = time.time() - state.last_audio_ts
            if idle > 0.6:
                sys.stdout.write("\r[WHISPER] ⏳ Transcribing… please wait   ")
                sys.stdout.flush()


# =========================
# WAV sender
# =========================
async def run_wav(ws, wav_path: str, realtime: bool, state: ClientState):
    with wave.open(wav_path, "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            await ws.send(data)
            state.last_audio_ts = time.time()
            if realtime:
                await asyncio.sleep(SLEEP_SEC)

    await ws.send(b"\x00\x00" * int(TARGET_SR * 0.8))
    await asyncio.sleep(0.8)
    await ws.send(b"")


# =========================
# Mic sender
# =========================
async def run_mic(ws, state: ClientState):
    if not HAS_MIC:
        raise RuntimeError("sounddevice not installed. pip install sounddevice")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    def cb(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=TARGET_SR,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
        callback=cb,
    )
    stream.start()

    print("🎤 Speak freely. Pause to end sentences. Ctrl+C to exit.")

    async def sender():
        while True:
            blk = await q.get()
            if blk is None:
                return
            try:
                await ws.send(blk.tobytes())
                state.last_audio_ts = time.time()
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


# =========================
# Main
# =========================
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/asr")
    p.add_argument("--wav", help="Path to wav file")
    p.add_argument("--mic", action="store_true")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    state = ClientState()
    start = time.time()

    async with websockets.connect(args.url, max_size=None) as ws:
        print(f"[INFO] Connected to {args.url}")

        recv_task = asyncio.create_task(receiver(ws, state))
        status_task = asyncio.create_task(whisper_status_loop(state))

        if args.mic:
            await run_mic(ws, state)
        else:
            if not args.wav:
                raise ValueError("--wav required unless --mic is set")

            wav = args.wav
            cleanup = None
            with wave.open(wav, "rb") as wf:
                sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            if sr != TARGET_SR or ch != 1 or sw != 2:
                print(f"[INFO] Resampling WAV → 16kHz mono PCM16")
                wav = resample_to_16k(wav)
                cleanup = wav

            await run_wav(ws, wav, realtime=not args.fast, state=state)

            if cleanup:
                os.unlink(cleanup)

        finals = await recv_task
        status_task.cancel()

    print("\nFULL TRANSCRIPT:")
    print(" ".join([t for t in finals if t.strip()]))

    print("\nCLIENT METRICS:")
    print(f"Total wall time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

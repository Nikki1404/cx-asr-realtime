import asyncio
import websockets
import json
import time
import sys
import numpy as np
import pyaudio

# =========================
# CONFIG
# =========================
WS_URL = "ws://127.0.0.1:8002/ws/asr"   # change if needed
BACKEND = "whisper"                    # "whisper" or "nemotron"

TARGET_SR = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# Whisper flush tuning
WHISPER_FLUSH_SILENCE_SEC = 0.35   # 350ms silence
WHISPER_MAX_FLUSH_INTERVAL = 2.0   # force flush every 2s

# =========================
# STATE
# =========================
last_audio_time = 0.0
last_flush_time = 0.0
is_running = True

# =========================
# AUDIO CONFIG (ONLY WHAT SERVER NEEDS)
# =========================
def build_audio_config():
    return {
        "type": "config",
        "backend": BACKEND,
        "sampling_rate": TARGET_SR,
        "chunk_ms": CHUNK_MS,
    }

# =========================
# RECEIVER
# =========================
async def receive_data(ws):
    while True:
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            print("\n[INFO] WebSocket closed by server")
            return

        if isinstance(msg, str):
            obj = json.loads(msg)
            typ = obj.get("type")

            if typ == "partial":
                txt = obj.get("text", "").replace("\n", " ")
                sys.stdout.write("\r[PARTIAL] " + txt[:160] + " " * 10)
                sys.stdout.flush()

            elif typ == "final":
                txt = (obj.get("text") or "").strip()
                print("\n[FINAL]", txt)

                print(
                    "[SERVER]",
                    f"ttf_ms={obj.get('ttf_ms')}",
                    f"audio_ms={obj.get('audio_ms')}",
                    f"rtf={obj.get('rtf')}",
                    f"chunks={obj.get('chunks')}",
                )

# =========================
# MIC STREAM
# =========================
async def stream_microphone(ws):
    global last_audio_time, last_flush_time, is_running

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=TARGET_SR,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )

    print(f"\n🎤 Mic started | backend={BACKEND}")
    print("Speak freely. Ctrl+C to stop.\n")

    try:
        while is_running:
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            now = time.time()
            await ws.send(pcm.tobytes())
            last_audio_time = now

            # =========================
            # 🔥 WHISPER MICRO-FLUSH
            # =========================
            if BACKEND == "whisper":
                silence_gap = now - last_audio_time
                since_flush = now - last_flush_time

                if silence_gap > 0.25 or since_flush > WHISPER_MAX_FLUSH_INTERVAL:
                    print("\n⏱️ Whisper micro-flush")

                    silence = b"\x00\x00" * int(TARGET_SR * WHISPER_FLUSH_SILENCE_SEC)
                    await ws.send(silence)
                    await asyncio.sleep(WHISPER_FLUSH_SILENCE_SEC)

                    await ws.send(b"")  # EOS
                    last_flush_time = now

            await asyncio.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        is_running = False

        try:
            await ws.send(b"\x00\x00" * int(TARGET_SR * 0.8))
            await asyncio.sleep(0.8)
            await ws.send(b"")
        except Exception:
            pass

        stream.stop_stream()
        stream.close()
        pa.terminate()

# =========================
# MAIN
# =========================
async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("[INFO] Connected:", WS_URL)

        # Send config FIRST
        cfg = build_audio_config()
        await ws.send(json.dumps(cfg))
        print("[INFO] Sent config:", cfg)

        recv_task = asyncio.create_task(receive_data(ws))
        await stream_microphone(ws)

        recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())

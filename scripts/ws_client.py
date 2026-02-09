import asyncio
import websockets
import json
import pyaudio
import numpy as np
import time

# =========================
# CONFIG
# =========================
WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/ws/asr"

# ASR backend: "nemotron" or "whisper"
ASR_BACKEND = "nemotron"

TARGET_SR = 16000
CHANNELS = 1

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

FORMAT = pyaudio.paInt16

# =========================
# GLOBAL STATE
# =========================
websocket = None
stream = None
is_recording = False
last_audio_time = 0.0


# =========================
# RECEIVE DATA
# =========================
async def receive_data():
    global websocket
    try:
        while True:
            data = await websocket.recv()
            if isinstance(data, str):
                json_data = json.loads(data)

                msg_type = json_data.get("type")

                if msg_type == "partial":
                    txt = json_data.get("text", "").replace("\n", " ")
                    print(f"\r[PARTIAL] {txt[:160]}", end="", flush=True)

                elif msg_type == "final":
                    print("\n[FINAL]", json_data.get("text", ""))
                    print(
                        "[SERVER_METRICS]",
                        f"reason={json_data.get('reason')}",
                        f"ttf_ms={json_data.get('ttf_ms')}",
                        f"ttft_ms={json_data.get('ttft_ms')}",
                        f"audio_ms={json_data.get('audio_ms')}",
                        f"rtf={json_data.get('rtf')}",
                        f"chunks={json_data.get('chunks')}",
                    )

                else:
                    # Config / status / misc
                    print("[SERVER]", json_data)

    except websockets.exceptions.ConnectionClosed:
        print("\n[INFO] WebSocket connection closed")


# =========================
# CONNECT
# =========================
async def connect_websocket():
    global websocket
    websocket = await websockets.connect(WEBSOCKET_ADDRESS, max_size=None)
    print("[INFO] WebSocket connection established")


# =========================
# SEND AUDIO CONFIG (EXACT test.py STYLE)
# =========================
async def send_audio_config():
    """
    Everything goes as JSON config — exactly like test.py
    """
    audio_config = {
        "service": "asr",
        "asrPipeline": ASR_BACKEND,     # <-- key difference
        "sampling_rate": TARGET_SR,
        "channels": CHANNELS,

        # Streaming behavior
        "chunk_offset_seconds": CHUNK_MS / 1000.0,
        "chunk_length_seconds": CHUNK_MS / 1000.0,

        # Optional flags (safe to ignore server-side)
        "user_speaking": True,
        "realtime": True,
    }

    await websocket.send(json.dumps(audio_config))
    print(f"[INFO] Sent audio_config: {audio_config}")


# =========================
# AUDIO STREAM
# =========================
async def start_recording():
    global stream, is_recording
    is_recording = True

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=TARGET_SR,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )

    print("🎤 Recording started (Ctrl+C to stop)")


async def stop_recording():
    global stream, is_recording
    is_recording = False

    if stream:
        stream.stop_stream()
        stream.close()

    print("\n🛑 Recording stopped")


# =========================
# SEND AUDIO LOOP
# =========================
async def process_audio():
    global last_audio_time
    try:
        while is_recording:
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            last_audio_time = time.time()
            await websocket.send(data)

            # Mic-like pacing
            await asyncio.sleep(SLEEP_SEC)

    except websockets.exceptions.ConnectionClosed:
        print("\n[ERROR] WebSocket closed while sending audio")


# =========================
# MAIN
# =========================
async def main():
    global websocket

    await connect_websocket()
    await send_audio_config()
    await start_recording()

    receive_task = asyncio.create_task(receive_data())
    send_task = asyncio.create_task(process_audio())

    try:
        while True:
            await asyncio.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received")

    finally:
        # EOS handling (important)
        try:
            silence = b"\x00\x00" * int(TARGET_SR * 0.8)
            await websocket.send(silence)
            await asyncio.sleep(0.8)
            await websocket.send(b"")  # EOS
        except Exception:
            pass

        await stop_recording()

        receive_task.cancel()
        send_task.cancel()

        await websocket.close()
        print("[INFO] Connection closed")


if __name__ == "__main__":
    asyncio.run(main())

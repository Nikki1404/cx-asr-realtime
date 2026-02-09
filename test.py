import asyncio
import websockets
import json
import pyaudio
import numpy as np

# =========================
# WebSocket address
# =========================
# websocket_address = "wss://cx-asr.exlservice.com/asr/realtime"
websocket_address = "ws://127.0.0.1:8002/ws/asr"

# =========================
# Audio configuration
# =========================
sample_rate = 16000
channels = 1

CHUNK_MS = 80
CHUNK_FRAMES = int(sample_rate * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# =========================
# Global state
# =========================
websocket = None
stream = None
is_recording = False


# =========================
# Receive messages
# =========================
async def receive_data():
    global websocket
    while True:
        try:
            data = await websocket.recv()
            if isinstance(data, str):
                json_data = json.loads(data)
                print("Data received from server:", json_data)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            return


# =========================
# Connect WebSocket
# =========================
async def connect_websocket():
    global websocket
    websocket = await websockets.connect(websocket_address, max_size=None)
    print("WebSocket connection established")


# =========================
# Send audio_config (EXACT STYLE)
# =========================
async def send_audio_config():
    audio_config = {
        "service": "asr",
        "backend": "nemotron",   # 🔁 change to "whisper" to test Whisper
        "user_speaking": True
    }

    await websocket.send(json.dumps(audio_config))
    print("Sent audio_config:", audio_config)


# =========================
# Start recording
# =========================
async def start_recording():
    global stream, is_recording
    is_recording = True

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )

    print("Recording started (Ctrl+C to stop)")


# =========================
# Stop recording + EOS
# =========================
async def stop_recording():
    global stream, is_recording
    is_recording = False

    try:
        # trailing silence + EOS
        await websocket.send(b"\x00\x00" * int(sample_rate * 0.8))
        await asyncio.sleep(0.8)
        await websocket.send(b"")
    except Exception:
        pass

    if stream:
        stream.stop_stream()
        stream.close()

    print("Recording stopped")


# =========================
# Send audio
# =========================
async def process_audio(sample_data):
    try:
        await websocket.send(sample_data.tobytes())
    except Exception as e:
        print(f"Error sending audio: {e}")


# =========================
# Main
# =========================
async def main():
    await connect_websocket()
    await send_audio_config()
    await start_recording()

    receive_task = asyncio.create_task(receive_data())

    try:
        while is_recording:
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            sample_data = np.frombuffer(data, dtype=np.int16)
            await process_audio(sample_data)
            await asyncio.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")

    await stop_recording()
    receive_task.cancel()


asyncio.run(main())

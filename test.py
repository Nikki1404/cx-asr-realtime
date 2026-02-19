import websockets
from pathlib import Path
from typing import Optional, List, Tuple
import json
import asyncio
import wave
import time

import jiwer
import uuid
from config import models
import pandas as pd
from tqdm import tqdm

TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)

def iter_wav_chunks(path: Path):
    with wave.open(str(path), "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            yield data

def silence_bytes(sec: float) -> bytes:
    return b"\x00\x00" * int(TARGET_SR * sec)

async def transcribe_ws(
    *,
    url: str,
    backend: str,
    wav_path: Path,
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

        t_start = time.time()

        for chunk in iter_wav_chunks(wav_path):
            await ws.send(chunk)
            frames_sent += len(chunk) // 2
            audio_sec_sent = frames_sent / TARGET_SR

        # finalize
        await ws.send(silence_bytes(0.6))
        await ws.send(b"")

        await asyncio.wait_for(final_received.wait(), timeout=30)

        latency_ms = int((time.time() - t_start) * 1000)

        await ws.close()
        recv_task.cancel()

        return " ".join(finals), audio_sec_sent, latency_ms

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
# REFERENCE LOOKUP
# =========================
def get_reference_text(txt_path: Path) -> str:
    with open(txt_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


# =========================
# SINGLE BACKEND
# =========================
async def get_benchmark_result(
    *,
    wav_path: Path,
    url: str,
    ref: str,
    backend: str,
) -> dict:

    try:
        hyp, audio_sec, latency_ms = await transcribe_ws(
            url=url,
            backend=backend,
            wav_path=wav_path,
        )

        wer = None
        if ref and hyp:
            wer = jiwer.wer(
                reference=ref,
                hypothesis=hyp,
                reference_transform=transform,
                hypothesis_transform=transform
            )
            wer = round(float(wer), 4)

        return wer, latency_ms, hyp

    except Exception as e:
        print("error: ", e)
        return None
    

async def process_models(
        filename: str,
        txt_path: str,
        **kwargs):

    ref = get_reference_text(txt_path)

    bench_result = {
        "file": filename,
        "ref_text": ref
    }

    for model in models:
        wer, latency, transcript = await get_benchmark_result(backend=model, ref=ref, **kwargs)
        bench_result[f"wer_{model}"] = wer
        bench_result[f"latency_{model}"] = latency
        bench_result[f"transcript_{model}"] = transcript

    return bench_result

# =========================
# MAIN
# =========================
async def main():
    """
        - test-cases <------ root_path
            - test1
                - audio1.wav
                - ref1.txt
            - test2
                - audio2.wav
                - ref2.txt
            .
            .
            .
    """
    url = "wss://cx-asr.exlservice.com/asr/realtime-custom-vad"

    root_path = Path(input("Enter you test folder path: "))

    num_models = len(models)
    columns = ['']*(3*num_models+2)
    columns[0] = 'file'
    columns[2*num_models+1] = 'ref_text'

    for idx, model in enumerate(models):
        columns[idx+1] = f'wer_{model}'
        columns[idx+1+num_models] = f'latency_{model}'
        columns[idx+2+2*num_models] = f'transcript_{model}'

    df = pd.DataFrame(columns=columns)

    for subfolder in tqdm(root_path.iterdir(), total=len(list(root_path.iterdir()))):
        if subfolder.is_dir():
            print(subfolder)
            wav_path = next(subfolder.glob("*.wav"), None)
            txt_path = next(subfolder.glob("*.txt"), None)

            if wav_path and txt_path:
                try:
                    bench_result = await process_models(
                        filename=subfolder.relative_to(root_path),
                        wav_path=wav_path,
                        url=url,
                        txt_path=txt_path,
                    )
                    df.loc[len(df)] =  bench_result
                except Exception as e:
                    print(e)

    save_filename = f"bench_realtime_dual_{uuid.uuid4().hex[:8]}.xlsx"
    df.to_excel(save_filename, index=False)

    print(f"\nSaved → {save_filename}")

if __name__ == "__main__":
    asyncio.run(main())

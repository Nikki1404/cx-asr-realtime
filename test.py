import argparse
import asyncio
import io
import json
import time
import uuid
import wave
from pathlib import Path

import jiwer
import pandas as pd
import websockets
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer


# CONFIG
TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_FRAMES * 2

# ⭐ DYNAMIC MODELS
models = ["google", "nemotron", "whisper"]

normalizer = EnglishTextNormalizer()

raw_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

norm_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])


def normalize(txt):
    return normalizer(txt or "")


# AUDIO
def wav_to_pcm(wav_blob):
    with wave.open(io.BytesIO(wav_blob), "rb") as wf:
        return wf.readframes(wf.getnframes())


def iter_chunks(pcm):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


def silence(sec):
    return b"\x00\x00" * int(TARGET_SR * sec)


# TRANSCRIBE
async def transcribe_ws(url, backend, pcm):

    async with websockets.connect(url, max_size=None) as ws:

        await ws.send(json.dumps({
            "type": "config",
            "backend": backend,
            "sampling_rate": TARGET_SR,
            "chunk_ms": CHUNK_MS,
        }))

        finals = []
        done = asyncio.Event()

        async def receiver():
            async for msg in ws:
                if not isinstance(msg, str):
                    continue

                obj = json.loads(msg)

                if obj.get("type") == "final":
                    txt = (obj.get("text") or "").strip()
                    if txt:
                        finals.append(txt)
                    done.set()

        recv_task = asyncio.create_task(receiver())

        t0 = time.time()

        # ⭐ REALTIME STREAMING
        for c in iter_chunks(pcm):
            await ws.send(c)
            await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(silence(1.0))
        await ws.send(b"")

        await asyncio.wait_for(done.wait(), timeout=120)

        latency = int((time.time() - t0) * 1000)

        recv_task.cancel()

        return " ".join(finals).strip(), latency


# PROCESS MODELS (DYNAMIC)
async def process_models(filename, wav_path, txt_path, url):

    ref = txt_path.read_text().strip()
    wav_blob = wav_path.read_bytes()
    pcm = wav_to_pcm(wav_blob)

    results = await asyncio.gather(
        *[transcribe_ws(url, m, pcm) for m in models]
    )

    row = {}
    row["filename"] = str(filename)

    ref_n = normalize(ref)

    row["reference_text"] = ref
    row["normalized_ref_text"] = ref_n

    for model, (hyp, lat) in zip(models, results):

        hyp_n = normalize(hyp)

        row[f"latency_{model}"] = lat
        row[f"transcript_{model}"] = hyp

        row[f"wer_{model}"] = jiwer.wer(
            ref, hyp,
            raw_transform,
            raw_transform
        )

        row[f"normalized_transcript_{model}"] = hyp_n

        row[f"normalized_wer_{model}"] = jiwer.wer(
            ref_n,
            hyp_n,
            norm_transform,
            norm_transform
        )

    return row


# MAIN (DYNAMIC DATAFRAME STRUCTURE)

async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    url = args.url

    # USER INPUT ROOT PATH
    root_path = Path(input("Enter your test folder path: "))

    # DYNAMIC COLUMN CREATION
    num_models = len(models)

    columns = [''] * (3 * num_models + 4)

    columns[0] = "filename"
    columns[1] = "reference_text"

    idx = 2

    for m in models:
        columns[idx] = f"wer_{m}"
        idx += 1

    for m in models:
        columns[idx] = f"latency_{m}"
        idx += 1

    for m in models:
        columns[idx] = f"transcript_{m}"
        idx += 1

    columns[idx] = "normalized_ref_text"
    idx += 1

    for m in models:
        columns.append(f"normalized_transcript_{m}")
        columns.append(f"normalized_wer_{m}")

    df = pd.DataFrame(columns=columns)

    # PROCESS
    folders = list(root_path.iterdir())

    for subfolder in tqdm(folders, total=len(folders)):

        if not subfolder.is_dir():
            continue

        wav_path = next(subfolder.glob("*.wav"), None)
        txt_path = next(subfolder.glob("*.txt"), None)

        if wav_path and txt_path:

            try:
                bench_result = await process_models(
                    filename=subfolder.relative_to(root_path),
                    wav_path=wav_path,
                    txt_path=txt_path,
                    url=url,
                )

                df.loc[len(df)] = bench_result

            except Exception as e:
                print("ERROR:", e)

    save_filename = f"bench_realtime_dynamic_{uuid.uuid4().hex[:8]}.xlsx"

    df.to_excel(save_filename, index=False)

    print("\nSaved →", save_filename)


if __name__ == "__main__":
    asyncio.run(main())

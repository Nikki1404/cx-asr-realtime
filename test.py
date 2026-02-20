TARGET_SR = 16000
CHUNK_MS = 80

MODELS = [
    "google",
    "nemotron",
    "whisper",
]


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

import config


# =====================================================
# CONFIG
# =====================================================

TARGET_SR = config.TARGET_SR
CHUNK_MS = config.CHUNK_MS
MODELS = config.MODELS

CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_FRAMES * 2

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


# =====================================================
# AUDIO
# =====================================================

def wav_to_pcm(wav_blob):
    with wave.open(io.BytesIO(wav_blob), "rb") as wf:
        return wf.readframes(wf.getnframes())


def iter_chunks(pcm):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


def silence(sec):
    return b"\x00\x00" * int(TARGET_SR * sec)


# =====================================================
# TRANSCRIBE
# =====================================================

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

        for c in iter_chunks(pcm):
            await ws.send(c)
            await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(silence(1.0))
        await ws.send(b"")

        await asyncio.wait_for(done.wait(), timeout=120)

        latency = int((time.time() - t0) * 1000)

        recv_task.cancel()

        return " ".join(finals).strip(), latency


# =====================================================
# MAIN
# =====================================================

async def main():

    url = input("Enter websocket URL: ")
    root_path = Path(input("Enter your test folder path: "))

    rows = []

    for folder in tqdm(list(root_path.iterdir())):

        if not folder.is_dir():
            continue

        wav_path = next(folder.glob("*.wav"), None)
        txt_path = next(folder.glob("*.txt"), None)

        if not wav_path or not txt_path:
            continue

        try:
            ref = txt_path.read_text().strip()
            pcm = wav_to_pcm(wav_path.read_bytes())

            results = await asyncio.gather(
                *[transcribe_ws(url, m, pcm) for m in MODELS]
            )

            ref_n = normalize(ref)

            row = {}

            # ----------------------------------
            # 1️⃣ filename
            # ----------------------------------
            row["filename"] = folder.name

            # ----------------------------------
            # 2️⃣ latency_{model}
            # ----------------------------------
            for model, (_, latency) in zip(MODELS, results):
                row[f"latency_{model}"] = latency

            # ----------------------------------
            # 3️⃣ reference_text
            # ----------------------------------
            row["reference_text"] = ref

            # ----------------------------------
            # 4️⃣ transcript_{model}
            # ----------------------------------
            for model, (hyp, _) in zip(MODELS, results):
                row[f"transcript_{model}"] = hyp

            # ----------------------------------
            # 5️⃣ wer_{model}
            # ----------------------------------
            for model, (hyp, _) in zip(MODELS, results):
                row[f"wer_{model}"] = jiwer.wer(
                    ref,
                    hyp,
                    raw_transform,
                    raw_transform
                )

            # ----------------------------------
            # 6️⃣ normalized_ref_text
            # ----------------------------------
            row["normalized_ref_text"] = ref_n

            # ----------------------------------
            # 7️⃣ normalized_transcript_{model}
            # ----------------------------------
            for model, (hyp, _) in zip(MODELS, results):
                row[f"normalized_transcript_{model}"] = normalize(hyp)

            # ----------------------------------
            # 8️⃣ normalized_wer_{model}
            # ----------------------------------
            for model, (hyp, _) in zip(MODELS, results):
                row[f"normalized_wer_{model}"] = jiwer.wer(
                    ref_n,
                    normalize(hyp),
                    norm_transform,
                    norm_transform
                )

            rows.append(row)

        except Exception as e:
            print("ERROR:", folder.name, e)

    # -------------------------------------------------
    # FORCE COLUMN ORDER EXACTLY AS REQUESTED
    # -------------------------------------------------

    ordered_columns = ["filename"]

    ordered_columns += [f"latency_{m}" for m in MODELS]
    ordered_columns += ["reference_text"]
    ordered_columns += [f"transcript_{m}" for m in MODELS]
    ordered_columns += [f"wer_{m}" for m in MODELS]
    ordered_columns += ["normalized_ref_text"]
    ordered_columns += [f"normalized_transcript_{m}" for m in MODELS]
    ordered_columns += [f"normalized_wer_{m}" for m in MODELS]

    df = pd.DataFrame(rows)
    df = df[ordered_columns]

    save_filename = f"bench_realtime_full_{uuid.uuid4().hex[:8]}.xlsx"
    df.to_excel(save_filename, index=False)

    print("\nSaved →", save_filename)


if __name__ == "__main__":
    asyncio.run(main())

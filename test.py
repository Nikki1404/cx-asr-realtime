TARGET_SR = 16000
CHUNK_MS = 80

MODELS = [
    "google",
    "nemotron",
    "whisper",
]

import argparse
import asyncio
import io
import json
import time
import uuid
import wave

import boto3
import jiwer
import pandas as pd
import websockets
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
# S3 HELPERS
# =====================================================

def s3_client(region):
    return boto3.client("s3", region_name=region)


def list_folders(s3, bucket, prefix):
    resp = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/"
    )
    return [x["Prefix"] for x in resp.get("CommonPrefixes", [])]


def list_objects(s3, bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [x["Key"] for x in resp.get("Contents", [])]


def read_text(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().strip()


def read_bytes(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="cx-speech")
    parser.add_argument("--prefix", default="asr-realtime/benchmarking-data-3/")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--url", required=True)
    parser.add_argument("--max-files", type=int, default=0)

    args = parser.parse_args()

    s3 = s3_client(args.region)

    folders = list_folders(s3, args.bucket, args.prefix)

    if args.max_files > 0:
        folders = folders[:args.max_files]

    N = len(MODELS)

    columns = [''] * (5 * N + 3)

    columns[0] = "file"
    columns[2 * N + 1] = "ref_text"
    columns[3 * N + 2] = "normalized_ref_text"

    for idx, model in enumerate(MODELS):
        columns[idx + 1] = f"wer_{model}"
        columns[idx + 1 + N] = f"latency_ms_{model}"
        columns[idx + 2 + 2 * N] = f"transcript_{model}"
        columns[idx + 3 + 3 * N] = f"normalized_transcript_{model}"
        columns[idx + 3 + 4 * N] = f"normalized_wer_{model}"

    df = pd.DataFrame(columns=columns)

    rows = []

    for folder in folders:

        try:
            keys = list_objects(s3, args.bucket, folder)

            wav_key = [k for k in keys if k.endswith(".wav")][0]
            txt_key = [k for k in keys if k.endswith("transcript.txt")][0]

            ref = read_text(s3, args.bucket, txt_key)
            wav_blob = read_bytes(s3, args.bucket, wav_key)
            pcm = wav_to_pcm(wav_blob)

            (g, lg), (n, ln), (w, lw) = await asyncio.gather(
                transcribe_ws(args.url, "google", pcm),
                transcribe_ws(args.url, "nemotron", pcm),
                transcribe_ws(args.url, "whisper", pcm),
            )

            ref_n = normalize(ref)
            g_n = normalize(g)
            n_n = normalize(n)
            w_n = normalize(w)

            row = [''] * (5 * N + 3)

            row[0] = folder
            row[2 * N + 1] = ref
            row[3 * N + 2] = ref_n

            # google
            row[1] = jiwer.wer(ref, g, raw_transform, raw_transform)
            row[1 + N] = lg
            row[2 + 2 * N] = g
            row[3 + 3 * N] = g_n
            row[3 + 4 * N] = jiwer.wer(ref_n, g_n, norm_transform, norm_transform)

            # nemotron
            row[2] = jiwer.wer(ref, n, raw_transform, raw_transform)
            row[2 + N] = ln
            row[3 + 2 * N] = n
            row[4 + 3 * N] = n_n
            row[4 + 4 * N] = jiwer.wer(ref_n, n_n, norm_transform, norm_transform)

            # whisper
            row[3] = jiwer.wer(ref, w, raw_transform, raw_transform)
            row[3 + N] = lw
            row[4 + 2 * N] = w
            row[5 + 3 * N] = w_n
            row[5 + 4 * N] = jiwer.wer(ref_n, w_n, norm_transform, norm_transform)

            rows.append(row)

        except Exception as e:
            print("ERROR:", folder, e)

    df = pd.DataFrame(rows, columns=columns)

    out = f"bench_s3_dynamic_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(out, index=False)

    print("\nSaved →", out)


if __name__ == "__main__":
    asyncio.run(main())

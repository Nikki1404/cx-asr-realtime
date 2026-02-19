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


# =====================================================
# CONFIG
# =====================================================

TARGET_SR = 16000
CHUNK_MS = 80
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

token_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])


# =====================================================
# S3 HELPERS
# =====================================================

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
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8").strip()


def read_bytes(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


# =====================================================
# AUDIO
# =====================================================

def wav_to_pcm(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    return pcm


def iter_chunks(pcm):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


def silence(sec):
    return b"\x00\x00" * int(TARGET_SR * sec)


# =====================================================
# NORMALIZATION
# =====================================================

def norm(text):
    return normalizer(text or "")


# =====================================================
# WEBSOCKET TRANSCRIBE
# =====================================================

async def transcribe(url, backend, pcm):

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
                obj = json.loads(msg)
                if obj.get("type") == "final":
                    txt = obj.get("text", "").strip()
                    if txt:
                        finals.append(txt)
                    done.set()

        recv_task = asyncio.create_task(receiver())

        start = time.time()

        for c in iter_chunks(pcm):
            await ws.send(c)

        await ws.send(silence(0.6))
        await ws.send(b"")

        await asyncio.wait_for(done.wait(), timeout=30)

        latency = int((time.time() - start) * 1000)

        recv_task.cancel()

        return " ".join(finals), latency


# =====================================================
# PROCESS SINGLE FOLDER
# =====================================================

async def process_folder(s3, url, bucket, folder):

    keys = list_objects(s3, bucket, folder)

    wav_key = [k for k in keys if k.endswith(".wav")][0]
    txt_key = [k for k in keys if k.endswith("transcript.txt")][0]

    ref = read_text(s3, bucket, txt_key)
    wav_bytes = read_bytes(s3, bucket, wav_key)

    pcm = wav_to_pcm(wav_bytes)

    results = await asyncio.gather(
        transcribe(url, "google", pcm),
        transcribe(url, "nemotron", pcm),
        transcribe(url, "whisper", pcm),
    )

    g_txt, g_lat = results[0]
    n_txt, n_lat = results[1]
    w_txt, w_lat = results[2]

    ref_n = norm(ref)
    g_n = norm(g_txt)
    n_n = norm(n_txt)
    w_n = norm(w_txt)

    return {
        "filename": folder,
        "latency_google": g_lat,
        "latency_nemotron": n_lat,
        "latency_whisper": w_lat,
        "reference_text": ref,
        "transcript_google": g_txt,
        "transcript_nemotron": n_txt,
        "transcript_whisper": w_txt,
        "wer_google": jiwer.wer(ref, g_txt, raw_transform, raw_transform),
        "wer_nemotron": jiwer.wer(ref, n_txt, raw_transform, raw_transform),
        "wer_whisper": jiwer.wer(ref, w_txt, raw_transform, raw_transform),
        "normalized_transcript_nemotron": n_n,
        "normalized_transcript_whisper": w_n,
        "normalized_transcript_google": g_n,
        "normalized_ref_text": ref_n,
        "normalized_wer_nemotron": jiwer.wer(ref_n, n_n, token_transform, token_transform),
        "normalized_wer_whisper": jiwer.wer(ref_n, w_n, token_transform, token_transform),
        "normalized_wer_google": jiwer.wer(ref_n, g_n, token_transform, token_transform),
    }


# =====================================================
# MAIN
# =====================================================

async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", default="cx-speech")
    parser.add_argument("--prefix", default="asr-realtime/benchmarking-data-3/")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--url", required=True)
    parser.add_argument("--max-folders", type=int, default=None)

    args = parser.parse_args()

    # REGION EXPLICITLY SET
    s3 = boto3.client("s3", region_name=args.region)

    folders = list_folders(s3, args.bucket, args.prefix)

    if args.max_folders:
        folders = folders[:args.max_folders]

    print(f"Benchmarking {len(folders)} folders...")

    rows = []

    for f in folders:
        print("Running:", f)
        row = await process_folder(s3, args.url, args.bucket, f)
        rows.append(row)

    df = pd.DataFrame(rows)

    out = f"benchmark_report_{uuid.uuid4().hex[:6]}.csv"
    df.to_csv(out, index=False)

    print("Saved:", out)


if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import io
import json
import time
import uuid
import wave
from typing import List

import boto3
import jiwer
import pandas as pd
import websockets
from whisper_normalizer.english import EnglishTextNormalizer


# CONFIG
TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_FRAMES * 2

whisper_norm = EnglishTextNormalizer()

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
    return whisper_norm(txt or "")


# S3
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


# AUDIO
def wav_to_pcm(wav_blob):
    with wave.open(io.BytesIO(wav_blob), "rb") as wf:
        return wf.readframes(wf.getnframes())


def iter_chunks(pcm):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i+CHUNK_BYTES]


def silence(sec):
    return b"\x00\x00" * int(TARGET_SR * sec)


# WEBSOCKET TRANSCRIBE 
async def transcribe_ws(url, backend, pcm, timeout_sec=120):

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

        #  REALTIME FIX
        for c in iter_chunks(pcm):
            await ws.send(c)
            await asyncio.sleep(CHUNK_MS / 1000)  

        # end speech
        await ws.send(silence(0.8))
        await ws.send(b"")

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            print(f"[WARN] timeout ({backend}) — using partial transcript")

        latency = int((time.time() - t0) * 1000)

        recv_task.cancel()

        return " ".join(finals).strip(), latency


# MAIN
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

            rows.append({
                "filename": folder,
                "latency_google": lg,
                "latency_nemotron": ln,
                "latency_whisper": lw,
                "reference_text": ref,
                "transcript_google": g,
                "transcript_nemotron": n,
                "transcript_whisper": w,
                "wer_google": jiwer.wer(ref, g, raw_transform, raw_transform),
                "wer_nemotron": jiwer.wer(ref, n, raw_transform, raw_transform),
                "wer_whisper": jiwer.wer(ref, w, raw_transform, raw_transform),
                "normalized_wer_google": jiwer.wer(ref_n, g_n, norm_transform, norm_transform),
                "normalized_wer_nemotron": jiwer.wer(ref_n, n_n, norm_transform, norm_transform),
                "normalized_wer_whisper": jiwer.wer(ref_n, w_n, norm_transform, norm_transform),
                "error": "",
            })

            print("DONE:", folder)

        except Exception as e:
            print(f"ERROR: {folder} -> {e}")
            rows.append({"filename": folder, "error": str(e)})

    df = pd.DataFrame(rows)

    out = f"bench_s3_wide_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(out, index=False)

    print("\nSaved →", out)


if __name__ == "__main__":
    asyncio.run(main())

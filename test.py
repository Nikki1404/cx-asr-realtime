import argparse
import asyncio
import io
import json
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional, List, Tuple

import boto3
import jiwer
import pandas as pd
import websockets
from whisper_normalizer.english import EnglishTextNormalizer


# -----------------------------
# Audio / streaming params
# -----------------------------
TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
CHUNK_BYTES = CHUNK_FRAMES * 2


# -----------------------------
# Normalization
# -----------------------------
whisper_norm = EnglishTextNormalizer()

raw_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

norm_token_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])


def normalize_with_whisper(text: str) -> str:
    return whisper_norm(text or "")


# -----------------------------
# S3 helpers
# -----------------------------
def s3_client(region):
    return boto3.client("s3", region_name=region)


def list_subprefixes(s3, bucket, prefix):
    resp = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/"
    )
    return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]


def list_objects_under(s3, bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [x["Key"] for x in resp.get("Contents", [])]


def read_s3_text(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().strip()


def read_s3_bytes(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


# -----------------------------
# WAV → PCM (NO FLAC now)
# -----------------------------
def wav_bytes_to_pcm(wav_blob: bytes):

    with wave.open(io.BytesIO(wav_blob), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    return pcm


def iter_pcm_chunks(pcm):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


def silence_bytes(sec: float):
    return b"\x00\x00" * int(TARGET_SR * sec)


# -----------------------------
# Websocket transcription
# -----------------------------
async def transcribe_ws(url, backend, pcm_bytes):

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
                    txt = (obj.get("text") or "").strip()
                    if txt:
                        finals.append(txt)
                    done.set()

        recv_task = asyncio.create_task(receiver())

        t0 = time.time()

        for c in iter_pcm_chunks(pcm_bytes):
            await ws.send(c)

        await ws.send(silence_bytes(0.6))
        await ws.send(b"")

        await asyncio.wait_for(done.wait(), timeout=30)

        recv_task.cancel()

        return " ".join(finals), int((time.time() - t0) * 1000)


# -----------------------------
# MAIN
# -----------------------------
async def main():

    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="cx-speech")
    p.add_argument("--prefix", default="asr-realtime/benchmarking-data-3/")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--url", required=True)
    p.add_argument("--max-files", type=int, default=0)
    args = p.parse_args()

    s3 = s3_client(args.region)

    folders = list_subprefixes(s3, args.bucket, args.prefix)

    if args.max_files > 0:
        folders = folders[:args.max_files]

    rows = []

    for folder in folders:
        try:

            keys = list_objects_under(s3, args.bucket, folder)

            wav_key = [k for k in keys if k.endswith(".wav")][0]
            txt_key = [k for k in keys if k.endswith("transcript.txt")][0]

            ref = read_s3_text(s3, args.bucket, txt_key)
            wav_blob = read_s3_bytes(s3, args.bucket, wav_key)

            pcm = wav_bytes_to_pcm(wav_blob)

            (hyp_g, lat_g), (hyp_n, lat_n), (hyp_w, lat_w) = await asyncio.gather(
                transcribe_ws(args.url, "google", pcm),
                transcribe_ws(args.url, "nemotron", pcm),
                transcribe_ws(args.url, "whisper", pcm),
            )

            # RAW WER
            wer_g = jiwer.wer(ref, hyp_g, raw_transform, raw_transform)
            wer_n = jiwer.wer(ref, hyp_n, raw_transform, raw_transform)
            wer_w = jiwer.wer(ref, hyp_w, raw_transform, raw_transform)

            # NORMALIZED
            ref_n = normalize_with_whisper(ref)
            g_n = normalize_with_whisper(hyp_g)
            n_n = normalize_with_whisper(hyp_n)
            w_n = normalize_with_whisper(hyp_w)

            nwer_g = jiwer.wer(ref_n, g_n, norm_token_transform, norm_token_transform)
            nwer_n = jiwer.wer(ref_n, n_n, norm_token_transform, norm_token_transform)
            nwer_w = jiwer.wer(ref_n, w_n, norm_token_transform, norm_token_transform)

            rows.append({
                "filename": folder,
                "latency_google": lat_g,
                "latency_nemotron": lat_n,
                "latency_whisper": lat_w,
                "reference_text": ref,
                "transcript_google": hyp_g,
                "transcript_nemotron": hyp_n,
                "transcript_whisper": hyp_w,
                "wer_google": wer_g,
                "wer_nemotron": wer_n,
                "wer_whisper": wer_w,
                "normalized_transcript_nemotron": n_n,
                "normalized_transcript_whisper": w_n,
                "normalized_transcript_google": g_n,
                "normalized_ref_text": ref_n,
                "normalized_wer_nemotron": nwer_n,
                "normalized_wer_whisper": nwer_w,
                "normalized_wer_google": nwer_g,
                "error": "",
            })

            print("DONE:", folder)

        except Exception as e:
            rows.append({
                "filename": folder,
                "error": str(e),
            })
            print("ERROR:", folder, e)

    df = pd.DataFrame(rows)

    out_csv = f"bench_s3_wide_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved →", out_csv)


if __name__ == "__main__":
    asyncio.run(main())

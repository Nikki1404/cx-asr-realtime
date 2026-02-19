import argparse
import asyncio
import csv
import io
import json
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional, List, Tuple, Dict

import boto3
import jiwer
import librosa
import numpy as np
import soundfile as sf
import websockets
from whisper_normalizer.english import EnglishTextNormalizer


# -----------------------------
# Audio / streaming params
# -----------------------------
TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)  # frames per chunk
CHUNK_BYTES = CHUNK_FRAMES * 2  # int16 => 2 bytes


# -----------------------------
# Normalization / WER transforms
# -----------------------------
whisper_norm = EnglishTextNormalizer()

# "Raw WER" transform: light & standard (no fancy number expansion)
raw_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

# "Normalized WER" transform: WhisperNormalizer first, then tokenize
# We'll run whisper_norm(text) ourselves and then just tokenize with this:
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
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def list_subprefixes(s3, bucket: str, prefix: str) -> List[str]:
    """
    Returns child "folders" (CommonPrefixes) directly under prefix.
    """
    out = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        out.extend([cp["Prefix"] for cp in resp.get("CommonPrefixes", [])])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def list_objects_under(s3, bucket: str, prefix: str) -> List[str]:
    """
    List ALL object keys under a prefix (non-recursive delimiter).
    """
    out = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def read_s3_text(s3, bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8", errors="replace").strip()


def read_s3_bytes(s3, bucket: str, key: str) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


# -----------------------------
# FLAC -> PCM16 bytes (16 kHz) in memory
# -----------------------------
def flac_bytes_to_pcm16_16k(flac_blob: bytes) -> bytes:
    """
    Returns mono PCM16 little-endian at 16kHz as raw bytes.
    """
    data, sr = sf.read(io.BytesIO(flac_blob), dtype="float32", always_2d=False)

    # If stereo, average to mono
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    # Clip then int16
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def iter_pcm_chunks(pcm_bytes: bytes):
    for i in range(0, len(pcm_bytes), CHUNK_BYTES):
        yield pcm_bytes[i:i + CHUNK_BYTES]


def silence_bytes(sec: float) -> bytes:
    return b"\x00\x00" * int(TARGET_SR * sec)


# -----------------------------
# Websocket transcription
# -----------------------------
async def transcribe_ws(
    *,
    url: str,
    backend: str,
    pcm_bytes: bytes,
    final_timeout_sec: int = 30,
) -> Tuple[str, int]:
    """
    Returns (final_text, latency_ms)
    """
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "backend": backend,
            "sampling_rate": TARGET_SR,
            "chunk_ms": CHUNK_MS,
        }))

        finals: List[str] = []
        final_received = asyncio.Event()

        async def receiver():
            async for msg in ws:
                if not isinstance(msg, str):
                    continue
                obj = json.loads(msg)
                if obj.get("type") == "final":
                    txt = (obj.get("text") or "").strip()
                    if txt:
                        finals.append(txt)
                    final_received.set()

        recv_task = asyncio.create_task(receiver())
        t0 = time.time()

        for chunk in iter_pcm_chunks(pcm_bytes):
            await ws.send(chunk)

        # flush
        await ws.send(silence_bytes(0.6))
        await ws.send(b"")

        await asyncio.wait_for(final_received.wait(), timeout=final_timeout_sec)

        latency_ms = int((time.time() - t0) * 1000)
        recv_task.cancel()

        return " ".join(finals).strip(), latency_ms


# -----------------------------
# Row format (wide CSV)
# -----------------------------
@dataclass
class WideRow:
    filename: str

    latency_google: Optional[int]
    latency_nemotron: Optional[int]
    latency_whisper: Optional[int]

    reference_text: str
    transcript_google: str
    transcript_nemotron: str
    transcript_whisper: str

    wer_google: Optional[float]
    wer_nemotron: Optional[float]
    wer_whisper: Optional[float]

    normalized_transcript_nemotron: str
    normalized_transcript_whisper: str
    normalized_transcript_google: str
    normalized_ref_text: str

    normalized_wer_nemotron: Optional[float]
    normalized_wer_whisper: Optional[float]
    normalized_wer_google: Optional[float]

    error: str = ""


def make_filename_label(flac_key: str, subset_label: str) -> str:
    """
    Convert flac key name like:
      .../1272_1281041272-128104-0001/1272-128104-0001.flac
    into:
      librispeech:dev-clean/1272/128104/128104-0001
    """
    base = PurePosixPath(flac_key).name.replace(".flac", "")
    parts = base.split("-")
    if len(parts) == 3:
        spk, chap, utt = parts
        return f"librispeech:{subset_label}/{spk}/{chap}/{chap}-{utt}"
    return f"librispeech:{subset_label}/{base}"


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="cx-speech")
    p.add_argument("--prefix", default="asr-realtime/benchmarking-data-2/")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--url", required=True)
    p.add_argument("--max-files", type=int, default=0, help="0 = all")
    p.add_argument("--subset-label", default="dev-clean", help="Used only in output filename label")
    args = p.parse_args()

    s3 = s3_client(args.region)

    # 1) list sample folders
    folders = list_subprefixes(s3, args.bucket, args.prefix)
    if args.max_files and args.max_files > 0:
        folders = folders[:args.max_files]

    out_csv = f"bench_s3_wide_{uuid.uuid4().hex[:8]}.csv"

    headers = [
        "filename",
        "latency_google",
        "latency_nemotron",
        "latency_whisper",
        "reference_text",
        "transcript_google",
        "transcript_nemotron",
        "transcript_whisper",
        "wer_google",
        "wer_nemotron",
        "wer_whisper",
        "normalized_transcript_nemotron",
        "normalized_transcript_whisper",
        "normalized_transcript_google",
        "normalized_ref_text",
        "normalized_wer_nemotron",
        "normalized_wer_whisper",
        "normalized_wer_google",
        "error",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for folder_prefix in folders:
            try:
                # 2) discover the flac + transcript in this folder
                keys = list_objects_under(s3, args.bucket, folder_prefix)
                flac_keys = [k for k in keys if k.lower().endswith(".flac")]
                txt_keys = [k for k in keys if k.lower().endswith("transcript.txt")]

                if not flac_keys or not txt_keys:
                    row = WideRow(
                        filename=folder_prefix,
                        latency_google=None, latency_nemotron=None, latency_whisper=None,
                        reference_text="",
                        transcript_google="", transcript_nemotron="", transcript_whisper="",
                        wer_google=None, wer_nemotron=None, wer_whisper=None,
                        normalized_transcript_nemotron="",
                        normalized_transcript_whisper="",
                        normalized_transcript_google="",
                        normalized_ref_text="",
                        normalized_wer_nemotron=None,
                        normalized_wer_whisper=None,
                        normalized_wer_google=None,
                        error="Missing .flac or transcript.txt in folder",
                    )
                    w.writerow(list(row.__dict__.values()))
                    continue

                flac_key = flac_keys[0]
                txt_key = txt_keys[0]

                # 3) load reference + audio from s3
                ref = read_s3_text(s3, args.bucket, txt_key)
                flac_blob = read_s3_bytes(s3, args.bucket, flac_key)
                pcm_bytes = flac_bytes_to_pcm16_16k(flac_blob)

                # 4) transcribe concurrently
                (hyp_g, lat_g), (hyp_n, lat_n), (hyp_w, lat_w) = await asyncio.gather(
                    transcribe_ws(url=args.url, backend="google", pcm_bytes=pcm_bytes),
                    transcribe_ws(url=args.url, backend="nemotron", pcm_bytes=pcm_bytes),
                    transcribe_ws(url=args.url, backend="whisper", pcm_bytes=pcm_bytes),
                )

                # 5) raw WER (light normalization)
                wer_g = wer_n = wer_w = None
                if ref.strip() and hyp_g.strip():
                    wer_g = round(float(jiwer.wer(ref, hyp_g, raw_transform, raw_transform)), 4)
                if ref.strip() and hyp_n.strip():
                    wer_n = round(float(jiwer.wer(ref, hyp_n, raw_transform, raw_transform)), 4)
                if ref.strip() and hyp_w.strip():
                    wer_w = round(float(jiwer.wer(ref, hyp_w, raw_transform, raw_transform)), 4)

                # 6) whisper-normalized strings + normalized WER
                ref_norm = normalize_with_whisper(ref)
                g_norm = normalize_with_whisper(hyp_g)
                n_norm = normalize_with_whisper(hyp_n)
                w_norm = normalize_with_whisper(hyp_w)

                nwer_g = nwer_n = nwer_w = None
                if ref_norm.strip() and g_norm.strip():
                    nwer_g = round(float(jiwer.wer(ref_norm, g_norm, norm_token_transform, norm_token_transform)), 4)
                if ref_norm.strip() and n_norm.strip():
                    nwer_n = round(float(jiwer.wer(ref_norm, n_norm, norm_token_transform, norm_token_transform)), 4)
                if ref_norm.strip() and w_norm.strip():
                    nwer_w = round(float(jiwer.wer(ref_norm, w_norm, norm_token_transform, norm_token_transform)), 4)

                filename_label = make_filename_label(flac_key, args.subset_label)

                row = WideRow(
                    filename=filename_label,
                    latency_google=lat_g,
                    latency_nemotron=lat_n,
                    latency_whisper=lat_w,
                    reference_text=ref,
                    transcript_google=hyp_g,
                    transcript_nemotron=hyp_n,
                    transcript_whisper=hyp_w,
                    wer_google=wer_g,
                    wer_nemotron=wer_n,
                    wer_whisper=wer_w,
                    normalized_transcript_nemotron=n_norm,
                    normalized_transcript_whisper=w_norm,
                    normalized_transcript_google=g_norm,
                    normalized_ref_text=ref_norm,
                    normalized_wer_nemotron=nwer_n,
                    normalized_wer_whisper=nwer_w,
                    normalized_wer_google=nwer_g,
                    error="",
                )

                w.writerow(list(row.__dict__.values()))
                print(f"DONE: {filename_label} | rawWER g/n/w={wer_g}/{wer_n}/{wer_w} | normWER g/n/w={nwer_g}/{nwer_n}/{nwer_w}")

            except Exception as e:
                row = WideRow(
                    filename=folder_prefix,
                    latency_google=None, latency_nemotron=None, latency_whisper=None,
                    reference_text="",
                    transcript_google="", transcript_nemotron="", transcript_whisper="",
                    wer_google=None, wer_nemotron=None, wer_whisper=None,
                    normalized_transcript_nemotron="",
                    normalized_transcript_whisper="",
                    normalized_transcript_google="",
                    normalized_ref_text="",
                    normalized_wer_nemotron=None,
                    normalized_wer_whisper=None,
                    normalized_wer_google=None,
                    error=str(e),
                )
                w.writerow(list(row.__dict__.values()))
                print(f"ERROR: {folder_prefix} -> {e}")

    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    asyncio.run(main())

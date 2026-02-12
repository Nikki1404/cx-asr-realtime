# ASR Realtime Benchmarking (Dual Backend)

## Overview

This benchmarking setup evaluates **Nemotron (streaming)** and **Whisper (batch)** ASR models in a **true realtime simulation environment** using LibriSpeech WAV files.

The script:
- Streams WAV audio incrementally (chunk-based)
- Runs **Nemotron + Whisper in parallel**
- Computes:
  - Latency (client-side final transcription time)
  - WER (Word Error Rate)
- Supports pause injection
- Saves CSV with unique UUID filename per run

---

## Data Used for Benchmarking

We use **LibriSpeech dataset structure**:

### WAV files
datasets/data/wav/
├── dev-clean/
├── dev-other/
├── test-clean/
└── test-other/
    └── <speaker>/<chapter>/<utterance>.wav

### Reference transcripts
datasets/data/raw/LibriSpeech/
├── dev-clean/
├── dev-other/
├── test-clean/
└── test-other/
    └── <speaker>/<chapter>/<speaker>-<chapter>.trans.txt

The script:
- Matches WAV file to transcript automatically
- Computes WER using normalized comparison

---

## Realtime Simulation Logic

- WAV files are streamed in 80ms chunks
- Optional silence injection simulates speech pauses
- Nemotron + Whisper run simultaneously per file
- No microphone required

---

## Installation (Client Only)

pip install websockets jiwer numpy

---

## Start ASR Server

Example:

python scripts/run_server.py --host 0.0.0.0 --port 8002

WebSocket endpoint:
ws://127.0.0.1:8002/ws/asr

---

## CLI Usage

### Basic Benchmark (Windows)

python asr_realtime_benchmark_dual.py ^
  --url ws://127.0.0.1:8002/ws/asr ^
  --data-wav-root "C:\\path\\to\\datasets\\data\\wav" ^
  --raw-librispeech-root "C:\\path\\to\\datasets\\data\\raw\\LibriSpeech" ^
  --max-files 20

### Basic Benchmark (Linux/Mac)

python asr_realtime_benchmark_dual.py \\
  --url ws://127.0.0.1:8002/ws/asr \\
  --data-wav-root /path/to/datasets/data/wav \\
  --raw-librispeech-root /path/to/datasets/data/raw/LibriSpeech \\
  --max-files 20

---

### Inject Pauses

python asr_realtime_benchmark_dual.py \\
  --url ws://127.0.0.1:8002/ws/asr \\
  --data-wav-root /path/to/datasets/data/wav \\
  --raw-librispeech-root /path/to/datasets/data/raw/LibriSpeech \\
  --inject-pause "2.0:0.5,5.0:1.0" \\
  --max-files 20

---

### Fast Mode (No pacing)

python asr_realtime_benchmark_dual.py \\
  --url ws://127.0.0.1:8002/ws/asr \\
  --data-wav-root /path/to/datasets/data/wav \\
  --raw-librispeech-root /path/to/datasets/data/raw/LibriSpeech \\
  --fast \\
  --max-files 20

---

### Concurrency Testing

python asr_realtime_benchmark_dual.py \\
  --workers 4 \\
  --max-files 30

---

## CSV Output

Each run generates:

bench_realtime_dual_<uuid>.csv

Columns:
- subset
- file
- backend
- latency_ms
- audio_sec_sent
- ref_text
- hyp_text
- wer
- error

---

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

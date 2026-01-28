# Realtime ASR Benchmarking & Streaming Service

This repository provides a **production-ready realtime ASR service** with pluggable backends, designed to **compare true streaming ASR vs batch ASR** under identical VAD, endpointing, and metrics instrumentation.

Currently supported ASR engines:
- **Nemotron Streaming ASR** (true streaming, low TTFT, partials)
- **Whisper Turbo Large v3** (batch ASR, high accuracy, no partials)

---

## High-Level Architecture

Client (Mic / PCM16)
→ WebSocket (/ws/asr)
→ VAD + Endpointing
→ ASR Engine (Nemotron or Whisper)

---

## Key Design Principles

### 1. Single Unified Pipeline
- WebSocket API
- VAD + endpointing
- metrics
- client behavior

Only the ASR engine implementation differs.

### 2. Correct Model-Specific Behavior

We do **not** force batch models into fake streaming.

| Capability | Nemotron | Whisper |
|----------|---------|--------|
| True streaming | Yes | No |
| Partials | Yes | No |
| TTFT meaningful | Yes | No |
| Decoder state | Incremental | Full-context |

---

## ASR Engines

### Nemotron Streaming ASR
- Uses NeMo conformer_stream_step
- Stateful streaming
- Partial transcripts + TTFT
- Best for realtime use cases

### Whisper Turbo Large v3
- Batch decode using generate()
- Endpointed transcription
- No partials, no TTFT
- Best for offline / post-call ASR

---

## Metrics

Metrics endpoint:
GET /metrics

All ASR metrics are labeled with:
- backend
- model

TTFT is recorded **only** for Nemotron.

---

## Example Result (Whisper Turbo)

[FINAL] Hello, this is Mike testing for ASR Whisper Turbo Large V3  
reason=silence  
ttft_ms=None  
ttf_ms=12328  
audio_ms=12160  
rtf=0.065  

Interpretation:
- Very low compute cost
- Latency dominated by audio duration
- Expected batch ASR behavior

---

## Running the Service

Docker (GPU):

docker run --gpus all -p 8000:8000 \
  -e ASR_BACKEND=whisper \
  -e MODEL_NAME=openai/whisper-large-v3-turbo \
  bu_digital_cx_asr_realtime

---

## WebSocket API

Endpoint:
ws://<host>:8000/ws/asr

Audio format:
- PCM16
- 16kHz
- Mono
- Binary frames
- Empty frame signals EOS

---

## When to Use Which Engine

Use Nemotron when:
- Realtime interaction needed
- Partial transcripts required

Use Whisper when:
- Accuracy > latency
- Offline / batch transcription

---

## Final Note

This system prioritizes:
- Correct ASR behavior
- Clean metrics
- Production safety
- Future extensibility

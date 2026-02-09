FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# ---- Proxy control (build-time) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "Enabling proxy for build stage"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for build stage"; \
    fi

# ---------------------------
# System dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Python tooling
# ---------------------------
RUN python3.10 -m pip install -U \
    pip \
    setuptools \
    wheel

# ---------------------------
# App dependencies FIRST
# ---------------------------
COPY requirements.txt /srv/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt

# ============================
# NeMo (same behavior as working image)
# ============================
RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# ============================
# Torch LAST (lock CUDA ABI)
# ============================
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1


# ===========================
# --- STAGE 2: RUNTIME ---
# ===========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Proxy control (runtime) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Persisting runtime proxy"; \
        echo "http_proxy=${HTTP_PROXY}" >> /etc/environment; \
        echo "https_proxy=${HTTPS_PROXY}" >> /etc/environment; \
    else \
        echo "🌐 Runtime proxy disabled"; \
    fi

# ---------------------------
# Runtime environment
# ---------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# ---------------------------
# Runtime system deps
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Copy Python environment from builder
# ---------------------------
COPY --from=builder /usr/local /usr/local

# ---------------------------
# Application code
# ---------------------------
COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8002

CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]


docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t cx_asr_realtime .

docker run --gpus all -p 8002:8002 \
  -e USE_PROXY=true \
  -e http_proxy=http://163.116.128.80:8080 \
  -e https_proxy=http://163.116.128.80:8080 \
  -e HTTP_PROXY=http://163.116.128.80:8080 \
  -e HTTPS_PROXY=http://163.116.128.80:8080 \
  cx_asr_realtime

[NeMo I 2026-02-09 11:03:56 rnnt_models:83] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-09 11:03:56 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-09 11:03:56 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-09 11:04:16 modelPT:501] Model EncDecRNNTBPEModel was successfully restored from /srv/hf_cache/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/c0acae9cc4163ab0d45cd403fbecbcb0635ee685/nemotron-speech-streaming-en-0.6b.nemo.
ERROR:asr_server:❌ Failed to preload nemotron: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 22.09 GiB of which 12.19 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 3.25 GiB is allocated by PyTorch, and 144.92 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
🎉 All engines preloaded! Client requests will be INSTANT.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)

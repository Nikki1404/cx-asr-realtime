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
# 🔥 CUDA MEMORY FIX (CRITICAL)
# ---------------------------
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_P2P_DISABLE=1

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local

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

How to use
EC2 (with proxy)
docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t cx_asr_realtime .

docker run --gpus all -p 8002:8002 \
  -e USE_PROXY=true \
  -e HTTP_PROXY=http://163.116.128.80:8080 \
  -e HTTPS_PROXY=http://163.116.128.80:8080 \
  cx_asr_realtime

Kubernetes (no proxy)

Just deploy the image. Nothing extra needed.

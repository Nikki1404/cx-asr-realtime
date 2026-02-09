FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ============================
# Build-time proxy support
# ============================
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Enabling proxy for build"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled"; \
    fi

# ============================
# Runtime environment
# ============================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# ============================
# System dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Python tooling
# ============================
RUN python3.10 -m pip install --no-cache-dir -U \
    pip \
    setuptools \
    wheel \
    "Cython<3.0"

# ============================
# Torch + Torchaudio (CUDA 12.4)
# ============================
RUN python3.10 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

# ============================
# ABI-safe pins (CRITICAL)
# ============================
RUN python3.10 -m pip install --no-cache-dir \
    "numpy<2.0" \
    "pyarrow<15" \
    "pandas<2.2" \
    "huggingface_hub<0.21"

# ============================
# App dependencies
# ============================
COPY requirements.txt /srv/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt

# ============================
# NeMo (stable tag)
# ============================
RUN python3.10 -m pip install --no-cache-dir --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

# ============================
# Application code
# ============================
COPY app /srv/app
COPY scripts /srv/scripts

# ============================
# Clear proxy for runtime
# ============================
ENV http_proxy=""
ENV https_proxy=""

EXPOSE 8002

CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t cx_asr_realtime .

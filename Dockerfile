FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /srv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# Torch (CUDA 12.4)
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

COPY requirements.txt /srv/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /srv/requirements.txt

# NeMo for Nemotron
RUN python3 -m pip install --no-cache-dir \
    "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8080

# Backend set at runtime (ENV or CLI)
CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

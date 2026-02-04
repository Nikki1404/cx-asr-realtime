FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"

WORKDIR /srv

# 1. Set Environment Variables
# LD_LIBRARY_PATH ensures torchaudio finds the CUDA libraries at runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 2. Install System Dependencies
# libsox-dev is critical for torchaudio's streaming backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# 3. Install Torch & Torchaudio (Pinned to CUDA 12.4)
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

# 4. Install requirements.txt
# Ensure torch/torchaudio are NOT listed in your requirements.txt
COPY requirements.txt /srv/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /srv/requirements.txt

# 5. Install NeMo (ASR)
RUN python3 -m pip install --no-cache-dir \
    "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

# 6. Copy application code
COPY app /srv/app
COPY scripts /srv/scripts

# 7. MODEL PRE-CACHING (Warmup)
# This downloads Whisper Large Turbo and Nemotron 0.6B during build.
# This makes the image large (~8-10GB) but startup will be instant.
RUN python3 -c "import torch; \
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; \
import nemo.collections.asr as nemo_asr; \
print('Pre-loading Whisper Turbo...'); \
AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo'); \
AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo'); \
print('Pre-loading Nemotron...'); \
nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/nemotron-speech-streaming-en-0.6b')"

EXPOSE 8002

# Using the provided run_server.py which initializes app.main:app
CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]

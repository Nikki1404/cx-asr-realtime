# Use NVIDIA's official PyTorch image - has torch+torchaudio+CUDA pre-built
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /srv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache

# Install system dependencies for audio (torchaudio + NeMo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

# NeMo for Nemotron (will use pre-installed torch/torchaudio)
RUN pip install --no-cache-dir \
    "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8002

CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]

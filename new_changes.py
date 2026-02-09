FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# Proxy (set ONCE if needed)
ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Enabling proxy for builder stage"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for builder stage"; \
    fi

# System deps
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

# Python tooling
RUN python3.10 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir "Cython<3.0"

# Torch + Torchaudio (LOCKED FIRST)
RUN python3.10 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

# App dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# NeMo (PINNED, ABI SAFE)
RUN python3.10 -m pip install --no-cache-dir --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

#  MODEL PRELOAD + CACHE
ENV HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache

RUN python3.10 - << 'PY'
import os
print("📦 HF_HOME:", os.environ["HF_HOME"])

# Whisper
print("⬇️  Preloading Whisper Turbo...")
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

# Nemotron
print("⬇️  Preloading Nemotron...")
import nemo.collections.asr as nemo_asr
nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

print("✅ Model preload complete")
PY

# --- STAGE 2: RUNTIME ---
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# Runtime-only deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python env + models
COPY --from=builder /usr/local /usr/local
COPY --from=builder /srv/hf_cache /srv/hf_cache

# Security: non-root
RUN useradd -m appuser && chown -R appuser:appuser /srv
USER appuser

# App code (last → keeps cache!)
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8002
CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

#docker build --build-arg USE_PROXY=true --build-arg HTTP_PROXY="http://163.116.128.80:8080" --build-arg HTTPS_PROXY="http://163.116.128.80:8080" -t cx_asr_realtime .


# ===========================
# --- STAGE 1: BUILDER ---
# ===========================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# ---------------------------
# Proxy (ONCE)
# ---------------------------
RUN if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
        echo "Proxy enabled"; \
    else \
        echo "Proxy disabled"; \
    fi

# ---------------------------
# System deps
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
RUN python3.10 -m pip install -U pip setuptools wheel && \
    python3.10 -m pip install "Cython<3.0"

# ---------------------------
# Torch (LOCK FIRST)
# ---------------------------
RUN python3.10 -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchaudio==2.5.1

# ---------------------------
# 🔴 CRITICAL PINS (FIX)
# ---------------------------
RUN python3.10 -m pip install \
    huggingface_hub==0.19.4 \
    transformers==4.39.3 \
    accelerate==0.27.2

# ---------------------------
# App deps
# ---------------------------
COPY requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

# ---------------------------
# NeMo (NOW SAFE)
# ---------------------------
RUN python3.10 -m pip install --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

# ---------------------------
# 🔥 MODEL PRELOAD + CACHE
# ---------------------------
ENV HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache

RUN python3.10 - << 'PY'
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import nemo.collections.asr as nemo_asr

print("⬇️ Preloading Whisper...")
AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

print("⬇️ Preloading Nemotron...")
nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

print("✅ Models cached successfully")
PY

# ===========================
# --- STAGE 2: RUNTIME ---
# ===========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache \
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

# Copy python + models
COPY --from=builder /usr/local /usr/local
COPY --from=builder /srv/hf_cache /srv/hf_cache

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /srv
USER appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8002
CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

 => ERROR [builder 11/11] RUN python3.10 - << 'PY'                                                                11.3s
------
 > [builder 11/11] RUN python3.10 - << 'PY':
1.298 /usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
1.298   import pynvml  # type: ignore[import]
2.163 /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
2.163   warnings.warn(
10.15 Traceback (most recent call last):
10.15   File "<stdin>", line 2, in <module>
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
10.15     from nemo.collections.asr import data, losses, models, modules
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 18, in <module>
10.15     from nemo.collections.asr.models.classification_models import EncDecClassificationModel, EncDecFrameClassificationModel
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/classification_models.py", line 29, in <module>
10.15     from nemo.collections.asr.data import audio_to_label_dataset, feature_to_label_dataset
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_label_dataset.py", line 19, in <module>
10.15     from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_text_dataset.py", line 28, in <module>
10.15     from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text_dataset.py", line 17, in <module>
10.15     from nemo.collections.asr.data.huggingface.hf_audio_to_text import (
10.15   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text.py", line 17, in <module>
10.15     import datasets as hf_datasets
10.15   File "/usr/local/lib/python3.10/dist-packages/datasets/__init__.py", line 22, in <module>
10.15     from .arrow_dataset import Dataset
10.15   File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 67, in <module>
10.15     from .arrow_writer import ArrowWriter, OptimizedTypedSequence
10.15   File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_writer.py", line 27, in <module>
10.15     from .features import Features, Image, Value
10.15   File "/usr/local/lib/python3.10/dist-packages/datasets/features/__init__.py", line 18, in <module>
10.15     from .features import Array2D, Array3D, Array4D, Array5D, ClassLabel, Features, Sequence, Value
10.15   File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 634, in <module>
10.15     class _ArrayXDExtensionType(pa.PyExtensionType):
10.15 AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?
------
Dockerfile:77
--------------------

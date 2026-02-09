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

# App dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Torch + Torchaudio (LOCKED FIRST)
RUN python3.10 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1


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
# App deps
# ---------------------------
COPY requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

# ---------------------------
# 🔴 CRITICAL PINS (FIX)
# ---------------------------
RUN python3.10 -m pip install \
    huggingface_hub==0.19.4 \
    transformers==4.39.3 \
    accelerate==0.27.2 \
    datasets==2.18.0 \
    pyarrow==12.0.1



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

 => ERROR [builder 11/11] RUN python3.10 - << 'PY'                                                                10.1s
------
 > [builder 11/11] RUN python3.10 - << 'PY':
1.239 /usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
1.239   import pynvml  # type: ignore[import]
2.093 /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
2.093   warnings.warn(
6.302
6.302 A module that was compiled using NumPy 1.x cannot be run in
6.302 NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
6.302 versions of NumPy, modules must be compiled with NumPy 2.0.
6.302 Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
6.302
6.302 If you are a user of the module, the easiest solution will be to
6.302 downgrade to 'numpy<2' or try to upgrade the affected module.
6.302 We expect that some modules will need time to support NumPy 2.
6.302
6.302 Traceback (most recent call last):  File "<stdin>", line 2, in <module>
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
6.302     from nemo.collections.asr import data, losses, models, modules
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/__init__.py", line 16, in <module>
6.302     from nemo.collections.asr.losses.audio_losses import SDRLoss
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/audio_losses.py", line 21, in <module>
6.302     from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/__init__.py", line 16, in <module>
6.302     from nemo.collections.asr.parts.preprocessing.features import FeaturizerFactory, FilterbankFeatures, WaveformFeaturizer
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/features.py", line 44, in <module>
6.302     from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/perturb.py", line 50, in <module>
6.302     from nemo.collections.common.parts.preprocessing import collections, parsers
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/__init__.py", line 16, in <module>
6.302     from nemo.collections.common import data, losses, parts, tokenizers
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/__init__.py", line 20, in <module>
6.302     from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
6.302   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/regex_tokenizer.py", line 20, in <module>
6.302     import pandas as pd
6.302   File "/usr/local/lib/python3.10/dist-packages/pandas/__init__.py", line 26, in <module>
6.302     from pandas.compat import (
6.302   File "/usr/local/lib/python3.10/dist-packages/pandas/compat/__init__.py", line 29, in <module>
6.302     from pandas.compat.pyarrow import (
6.302   File "/usr/local/lib/python3.10/dist-packages/pandas/compat/pyarrow.py", line 8, in <module>
6.302     import pyarrow as pa
6.302   File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
6.302     import pyarrow.lib as _lib
6.302 AttributeError: _ARRAY_API not found
6.493
6.493 A module that was compiled using NumPy 1.x cannot be run in
6.493 NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
6.493 versions of NumPy, modules must be compiled with NumPy 2.0.
6.493 Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
6.493
6.493 If you are a user of the module, the easiest solution will be to
6.493 downgrade to 'numpy<2' or try to upgrade the affected module.
6.493 We expect that some modules will need time to support NumPy 2.
6.493
6.493 Traceback (most recent call last):  File "<stdin>", line 2, in <module>
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
6.493     from nemo.collections.asr import data, losses, models, modules
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/__init__.py", line 16, in <module>
6.493     from nemo.collections.asr.losses.audio_losses import SDRLoss
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/audio_losses.py", line 21, in <module>
6.493     from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/__init__.py", line 16, in <module>
6.493     from nemo.collections.asr.parts.preprocessing.features import FeaturizerFactory, FilterbankFeatures, WaveformFeaturizer
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/features.py", line 44, in <module>
6.493     from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/perturb.py", line 50, in <module>
6.493     from nemo.collections.common.parts.preprocessing import collections, parsers
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/__init__.py", line 16, in <module>
6.493     from nemo.collections.common import data, losses, parts, tokenizers
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/__init__.py", line 20, in <module>
6.493     from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
6.493   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/regex_tokenizer.py", line 20, in <module>
6.493     import pandas as pd
6.493   File "/usr/local/lib/python3.10/dist-packages/pandas/__init__.py", line 49, in <module>
6.493     from pandas.core.api import (
6.493   File "/usr/local/lib/python3.10/dist-packages/pandas/core/api.py", line 9, in <module>
6.493     from pandas.core.dtypes.dtypes import (
6.493   File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/dtypes.py", line 24, in <module>
6.493     from pandas._libs import (
6.493   File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
6.493     import pyarrow.lib as _lib
6.494 AttributeError: _ARRAY_API not found
9.095
9.095 A module that was compiled using NumPy 1.x cannot be run in
9.095 NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
9.095 versions of NumPy, modules must be compiled with NumPy 2.0.
9.095 Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
9.095
9.095 If you are a user of the module, the easiest solution will be to
9.095 downgrade to 'numpy<2' or try to upgrade the affected module.
9.095 We expect that some modules will need time to support NumPy 2.
9.095
9.095 Traceback (most recent call last):  File "<stdin>", line 2, in <module>
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
9.095     from nemo.collections.asr import data, losses, models, modules
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 18, in <module>
9.095     from nemo.collections.asr.models.classification_models import EncDecClassificationModel, EncDecFrameClassificationModel
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/classification_models.py", line 29, in <module>
9.095     from nemo.collections.asr.data import audio_to_label_dataset, feature_to_label_dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_label_dataset.py", line 19, in <module>
9.095     from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_text_dataset.py", line 28, in <module>
9.095     from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text_dataset.py", line 17, in <module>
9.095     from nemo.collections.asr.data.huggingface.hf_audio_to_text import (
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text.py", line 17, in <module>
9.095     import datasets as hf_datasets
9.095   File "/usr/local/lib/python3.10/dist-packages/datasets/__init__.py", line 18, in <module>
9.095     from .arrow_dataset import Dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 60, in <module>
9.095     import pyarrow as pa
9.095   File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
9.095     import pyarrow.lib as _lib
9.095 AttributeError: _ARRAY_API not found
9.095 Traceback (most recent call last):
9.095   File "<stdin>", line 2, in <module>
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
9.095     from nemo.collections.asr import data, losses, models, modules
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 18, in <module>
9.095     from nemo.collections.asr.models.classification_models import EncDecClassificationModel, EncDecFrameClassificationModel
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/classification_models.py", line 29, in <module>
9.095     from nemo.collections.asr.data import audio_to_label_dataset, feature_to_label_dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_label_dataset.py", line 19, in <module>
9.095     from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_text_dataset.py", line 28, in <module>
9.095     from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text_dataset.py", line 17, in <module>
9.095     from nemo.collections.asr.data.huggingface.hf_audio_to_text import (
9.095   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text.py", line 17, in <module>
9.095     import datasets as hf_datasets
9.095   File "/usr/local/lib/python3.10/dist-packages/datasets/__init__.py", line 18, in <module>
9.095     from .arrow_dataset import Dataset
9.095   File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 60, in <module>
9.095     import pyarrow as pa
9.095   File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
9.095     import pyarrow.lib as _lib
9.095   File "pyarrow/lib.pyx", line 36, in init pyarrow.lib
9.095 ImportError: numpy.core.multiarray failed to import
------
Dockerfile:81
--------------------
  80 |
  81 | >>> RUN python3.10 - << 'PY'
  82 | >>> from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
  83 | >>> import nemo.collections.asr as nemo_asr
  84 | >>>
  85 | >>> print("⬇️ Preloading Whisper...")
  86 | >>> AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
  87 | >>> AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")
  88 | >>>
  89 | >>> print("⬇️ Preloading Nemotron...")
  90 | >>> nemo_asr.models.ASRModel.from_pretrained(
  91 | >>>     "nvidia/nemotron-speech-streaming-en-0.6b"
  92 | >>> )
  93 | >>>
  94 | >>> print("✅ Models cached successfully")
  95 | >>> PY
  96 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3.10 - << 'PY'\nfrom transformers import AutoProcessor, AutoModelForSpeechSeq2Seq\nimport nemo.collections.asr as nemo_asr\n\nprint(\"⬇️ Preloading Whisper...\")\nAutoProcessor.from_pretrained(\"openai/whisper-large-v3-turbo\")\nAutoModelForSpeechSeq2Seq.from_pretrained(\"openai/whisper-large-v3-turbo\")\n\nprint(\"⬇️ Preloading Nemotron...\")\nnemo_asr.models.ASRModel.from_pretrained(\n    \"nvidia/nemotron-speech-streaming-en-0.6b\"\n)\n\nprint(\"✅ Models cached successfully\")\nPY" did not complete successfully: exit code: 1

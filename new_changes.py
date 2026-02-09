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

RUN if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
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
# Torch (CUDA 12.4)
# ---------------------------
RUN python3.10 -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchaudio==2.5.1

# ---------------------------
# 🔴 ABI LOCKS (CRITICAL)
# ---------------------------
RUN python3.10 -m pip install --force-reinstall \
    numpy==1.26.4 \
    pyarrow==12.0.1 \
    pandas==1.5.3 \
    datasets==2.18.0 \
    huggingface_hub==0.19.4 \
    transformers==4.39.3 \
    accelerate==0.27.2

# ---------------------------
# App deps
# ---------------------------
COPY requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

# ---------------------------
# NeMo (now safe)
# ---------------------------
RUN python3.10 -m pip install --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

# ---------------------------
# 🔥 Model preload + cache
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

COPY --from=builder /usr/local /usr/local
COPY --from=builder /srv/hf_cache /srv/hf_cache

RUN useradd -m appuser && chown -R appuser:appuser /srv
USER appuser


(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime_updated#  docker run --gpus all -p 8002:8002 cx_asr_realtime

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
DEBUG: Startup cfg.model_name='nvidia/nemotron-speech-streaming-en-0.6b' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
🚀 Preloading ASR engines (this happens once at startup)...
   Loading whisper (openai/whisper-large-v3-turbo)...
ERROR:asr_server:❌ Failed to preload whisper: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like openai/whisper-large-v3-turbo is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
   Loading nemotron (nvidia/nemotron-speech-streaming-en-0.6b)...
INFO:matplotlib.font_manager:generated new fontManager

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/srv/scripts/run_server.py", line 22, in <module>
    main()
  File "/srv/scripts/run_server.py", line 13, in main
    uvicorn.run(
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/main.py", line 594, in run
    server.run()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 67, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/_compat.py", line 60, in asyncio_run
    return loop.run_until_complete(main)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/lifespan/on.py", line 86, in main
    await app(scope, self.receive, self.send)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 29, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1138, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 151, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 49, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 725, in app
    await self.lifespan(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 228, in __aenter__
    await self._router._startup()
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 4557, in _startup
    await handler()
  File "/srv/app/main.py", line 55, in startup_event
    await preload_engines()
  File "/srv/app/main.py", line 42, in preload_engines
    load_sec = engine.load()
  File "/srv/app/asr_engines/nemotron_asr.py", line 93, in load
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/__init__.py", line 16, in <module>
    from nemo.collections.asr.losses.audio_losses import SDRLoss
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/audio_losses.py", line 21, in <module>
    from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/__init__.py", line 16, in <module>
    from nemo.collections.asr.parts.preprocessing.features import FeaturizerFactory, FilterbankFeatures, WaveformFeaturizer
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/features.py", line 44, in <module>
    from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/perturb.py", line 50, in <module>
    from nemo.collections.common.parts.preprocessing import collections, parsers
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/__init__.py", line 16, in <module>
    from nemo.collections.common import data, losses, parts, tokenizers
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/__init__.py", line 20, in <module>
    from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/regex_tokenizer.py", line 20, in <module>
    import pandas as pd
  File "/usr/local/lib/python3.10/dist-packages/pandas/__init__.py", line 26, in <module>
    from pandas.compat import (
  File "/usr/local/lib/python3.10/dist-packages/pandas/compat/__init__.py", line 29, in <module>
    from pandas.compat.pyarrow import (
  File "/usr/local/lib/python3.10/dist-packages/pandas/compat/pyarrow.py", line 8, in <module>
    import pyarrow as pa
  File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
    import pyarrow.lib as _lib
AttributeError: _ARRAY_API not found

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/srv/scripts/run_server.py", line 22, in <module>
    main()
  File "/srv/scripts/run_server.py", line 13, in main
    uvicorn.run(
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/main.py", line 594, in run
    server.run()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 67, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/_compat.py", line 60, in asyncio_run
    return loop.run_until_complete(main)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/lifespan/on.py", line 86, in main
    await app(scope, self.receive, self.send)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 29, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1138, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 151, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 49, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 725, in app
    await self.lifespan(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 228, in __aenter__
    await self._router._startup()
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 4557, in _startup
    await handler()
  File "/srv/app/main.py", line 55, in startup_event
    await preload_engines()
  File "/srv/app/main.py", line 42, in preload_engines
    load_sec = engine.load()
  File "/srv/app/asr_engines/nemotron_asr.py", line 93, in load
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/__init__.py", line 16, in <module>
    from nemo.collections.asr.losses.audio_losses import SDRLoss
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/losses/audio_losses.py", line 21, in <module>
    from nemo.collections.asr.parts.preprocessing.features import make_seq_mask_like
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/__init__.py", line 16, in <module>
    from nemo.collections.asr.parts.preprocessing.features import FeaturizerFactory, FilterbankFeatures, WaveformFeaturizer
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/features.py", line 44, in <module>
    from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/preprocessing/perturb.py", line 50, in <module>
    from nemo.collections.common.parts.preprocessing import collections, parsers
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/__init__.py", line 16, in <module>
    from nemo.collections.common import data, losses, parts, tokenizers
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/__init__.py", line 20, in <module>
    from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/common/tokenizers/regex_tokenizer.py", line 20, in <module>
    import pandas as pd
  File "/usr/local/lib/python3.10/dist-packages/pandas/__init__.py", line 49, in <module>
    from pandas.core.api import (
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/api.py", line 9, in <module>
    from pandas.core.dtypes.dtypes import (
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/dtypes.py", line 24, in <module>
    from pandas._libs import (
  File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
    import pyarrow.lib as _lib
AttributeError: _ARRAY_API not found

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/srv/scripts/run_server.py", line 22, in <module>
    main()
  File "/srv/scripts/run_server.py", line 13, in main
    uvicorn.run(
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/main.py", line 594, in run
    server.run()
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/server.py", line 67, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/_compat.py", line 60, in asyncio_run
    return loop.run_until_complete(main)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/lifespan/on.py", line 86, in main
    await app(scope, self.receive, self.send)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 29, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1138, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 151, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 49, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 725, in app
    await self.lifespan(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 228, in __aenter__
    await self._router._startup()
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 4557, in _startup
    await handler()
  File "/srv/app/main.py", line 55, in startup_event
    await preload_engines()
  File "/srv/app/main.py", line 42, in preload_engines
    load_sec = engine.load()
  File "/srv/app/asr_engines/nemotron_asr.py", line 93, in load
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 18, in <module>
    from nemo.collections.asr.models.classification_models import EncDecClassificationModel, EncDecFrameClassificationModel
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/classification_models.py", line 29, in <module>
    from nemo.collections.asr.data import audio_to_label_dataset, feature_to_label_dataset
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_label_dataset.py", line 19, in <module>
    from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/audio_to_text_dataset.py", line 28, in <module>
    from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text_dataset.py", line 17, in <module>
    from nemo.collections.asr.data.huggingface.hf_audio_to_text import (
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/data/huggingface/hf_audio_to_text.py", line 17, in <module>
    import datasets as hf_datasets
  File "/usr/local/lib/python3.10/dist-packages/datasets/__init__.py", line 22, in <module>
    from .arrow_dataset import Dataset
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 59, in <module>
    import pyarrow as pa
  File "/usr/local/lib/python3.10/dist-packages/pyarrow/__init__.py", line 65, in <module>
    import pyarrow.lib as _lib
AttributeError: _ARRAY_API not found
ERROR:asr_server:❌ Failed to preload nemotron: numpy.core.multiarray failed to import
🎉 All engines preloaded! Client requests will be INSTANT.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)


COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8002
CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

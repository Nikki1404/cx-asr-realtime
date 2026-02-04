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
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

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


getting this 
 => [10/11] COPY scripts /srv/scripts                                                                                                                                  0.1s
 => ERROR [11/11] RUN python3 -c "import torch; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; import nemo.collections.asr as nemo_asr; print('P  14.5s
------
 > [11/11] RUN python3 -c "import torch; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; import nemo.collections.asr as nemo_asr; print('Pre-loading Whisper Turbo...'); AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo'); AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo'); print('Pre-loading Nemotron...'); nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/nemotron-speech-streaming-en-0.6b')":
2.440 /usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
2.440   import pynvml  # type: ignore[import]
4.528 /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
4.528   warnings.warn(
10.47 [NeMo W 2026-02-04 16:55:33 <frozen importlib:241] Megatron num_microbatches_calculator not found, using Apex version.
10.94 OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
10.95 No exporters were provided. This means that no telemetry data will be collected.
12.46 Traceback (most recent call last):
12.46   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1442, in load_library
12.46     ctypes.CDLL(path)
12.46   File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
12.46     self._handle = _dlopen(self._name, mode)
12.46 OSError: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
12.46
12.46 The above exception was the direct cause of the following exception:
12.46
12.46 Traceback (most recent call last):
12.46   File "<string>", line 1, in <module>
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
12.46     from nemo.collections.asr import data, losses, models, modules
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 15, in <module>
12.46     from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/aed_multitask_models.py", line 32, in <module>
12.46     from nemo.collections.asr.metrics import MultiTaskMetric
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/metrics/__init__.py", line 15, in <module>
12.46     from nemo.collections.asr.metrics.bleu import BLEU
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/metrics/bleu.py", line 24, in <module>
12.46     from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/multitask_decoding.py", line 22, in <module>
12.46     from nemo.collections.asr.parts.submodules.multitask_beam_decoding import (
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/multitask_beam_decoding.py", line 22, in <module>
12.46     from nemo.collections.asr.modules.transformer import (
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/__init__.py", line 15, in <module>
12.46     from nemo.collections.asr.modules.audio_preprocessing import (
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/audio_preprocessing.py", line 26, in <module>
12.46     from nemo.collections.audio.parts.utils.transforms import MFCC
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/__init__.py", line 15, in <module>
12.46     from nemo.collections.audio import data, losses, metrics, models, modules
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/__init__.py", line 15, in <module>
12.46     from nemo.collections.audio.metrics.audio import AudioMetricWrapper
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/audio.py", line 24, in <module>
12.46     from nemo.collections.audio.metrics.squim import SquimMOSMetric, SquimObjectiveMetric
12.46   File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/squim.py", line 22, in <module>
12.46     import torchaudio
12.46   File "/usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py", line 2, in <module>
12.46     from . import _extension  # noqa  # usort: skip
12.46   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py", line 38, in <module>
12.46     _load_lib("libtorchaudio")
12.46   File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/utils.py", line 60, in _load_lib
12.46     torch.ops.load_library(path)
12.46   File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1444, in load_library
12.46     raise OSError(f"Could not load this library: {path}") from e
12.46 OSError: Could not load this library: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so
------
Dockerfile:56
--------------------
  55 |     # This makes the image large (~8-10GB) but startup will be instant.
  56 | >>> RUN python3 -c "import torch; \
  57 | >>> from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; \
  58 | >>> import nemo.collections.asr as nemo_asr; \
  59 | >>> print('Pre-loading Whisper Turbo...'); \
  60 | >>> AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo'); \
  61 | >>> AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo'); \
  62 | >>> print('Pre-loading Nemotron...'); \
  63 | >>> nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/nemotron-speech-streaming-en-0.6b')"
  64 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -c \"import torch; from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; import nemo.collections.asr as nemo_asr; print('Pre-loading Whisper Turbo...'); AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3-turbo'); AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo'); print('Pre-loading Nemotron...'); nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/nemotron-speech-streaming-en-0.6b')\"" did not complete successfully: exit code: 1


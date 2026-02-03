(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# docker run --gpus all -p 8002:8000 -e ASR_
BACKEND=nemotron -e MODEL_NAME=nvidia/nemotron-speech-streaming-en-0.6b bu_digital_cx_asr_realtime

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:matplotlib.font_manager:generated new fontManager
[NeMo W 2026-02-03 10:32:24 <frozen importlib:241] Megatron num_microbatches_calculator not found, using Apex version.
WARNING:nv_one_logger.api.config:OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
INFO:nv_one_logger.exporter.export_config_manager:Final configuration contains 0 exporter(s)
WARNING:nv_one_logger.training_telemetry.api.training_telemetry_provider:No exporters were provided. This means that no telemetry data will be collected.
ERROR:    Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1442, in load_library
    ctypes.CDLL(path)
  File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 571, in __aenter__
    await self._router.startup()
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 671, in startup
    await handler()
  File "/srv/app/main.py", line 28, in startup
    load_sec = engine.load()
  File "/srv/app/asr_engines/nemotron_asr.py", line 93, in load
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/__init__.py", line 15, in <module>
    from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/models/aed_multitask_models.py", line 32, in <module>
    from nemo.collections.asr.metrics import MultiTaskMetric
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/metrics/__init__.py", line 15, in <module>
    from nemo.collections.asr.metrics.bleu import BLEU
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/metrics/bleu.py", line 24, in <module>
    from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/multitask_decoding.py", line 22, in <module>
    from nemo.collections.asr.parts.submodules.multitask_beam_decoding import (
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/parts/submodules/multitask_beam_decoding.py", line 22, in <module>
    from nemo.collections.asr.modules.transformer import (
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/__init__.py", line 15, in <module>
    from nemo.collections.asr.modules.audio_preprocessing import (
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/asr/modules/audio_preprocessing.py", line 26, in <module>
    from nemo.collections.audio.parts.utils.transforms import MFCC
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/__init__.py", line 15, in <module>
    from nemo.collections.audio import data, losses, metrics, models, modules
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/__init__.py", line 15, in <module>
    from nemo.collections.audio.metrics.audio import AudioMetricWrapper
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/audio.py", line 24, in <module>
    from nemo.collections.audio.metrics.squim import SquimMOSMetric, SquimObjectiveMetric
  File "/usr/local/lib/python3.10/dist-packages/nemo/collections/audio/metrics/squim.py", line 22, in <module>
    import torchaudio
  File "/usr/local/lib/python3.10/dist-packages/torchaudio/__init__.py", line 2, in <module>
    from . import _extension  # noqa  # usort: skip
  File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/__init__.py", line 38, in <module>
    _load_lib("libtorchaudio")
  File "/usr/local/lib/python3.10/dist-packages/torchaudio/_extension/utils.py", line 60, in _load_lib
    torch.ops.load_library(path)
  File "/usr/local/lib/python3.10/dist-packages/torch/_ops.py", line 1444, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: /usr/local/lib/python3.10/dist-packages/torchaudio/lib/libtorchaudio.so

ERROR:    Application startup failed. Exiting


dockerifle -

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"

WORKDIR /srv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
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

EXPOSE 8000

# Backend set at runtime (ENV or CLI)
CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

.

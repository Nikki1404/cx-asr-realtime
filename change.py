FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV https_proxy="http://163.116.128.80:8080"
ENV http_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

COPY requirements.txt /srv/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /srv/requirements.txt

RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

RUN python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchaudio==2.4.1


COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8002

CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime_updated# docker build -t cx_asr_realtime .
[+] Building 1436.4s (16/16) FINISHED                                                                    docker:default
 => [internal] load build definition from Dockerfile                                                               0.0s
 => => transferring dockerfile: 1.25kB                                                                             0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                            0.2s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                         0.0s
 => [internal] load .dockerignore                                                                                  0.0s
 => => transferring context: 2B                                                                                    0.0s
 => [ 1/10] FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd  0.0s
 => [internal] load build context                                                                                  0.0s
 => => transferring context: 673B                                                                                  0.0s
 => CACHED [ 2/10] WORKDIR /srv                                                                                    0.0s
 => [ 3/10] RUN apt-get update && apt-get install -y --no-install-recommends     python3     python3-pip     py  172.3s
 => [ 4/10] RUN python3 -m pip install -U pip setuptools wheel                                                     3.5s
 => [ 5/10] COPY requirements.txt /srv/requirements.txt                                                            0.1s
 => [ 6/10] RUN python3 -m pip install --no-cache-dir -r /srv/requirements.txt                                   582.7s
 => [ 7/10] RUN python3 -m pip install --no-cache-dir     "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/Ne  128.7s
 => [ 8/10] RUN python3 -m pip install --no-cache-dir --force-reinstall     --index-url https://download.pytorc  487.6s
 => [ 9/10] COPY app /srv/app                                                                                      0.1s
 => [10/10] COPY scripts /srv/scripts                                                                              0.1s
 => exporting to image                                                                                            61.0s
 => => exporting layers                                                                                           60.9s
 => => writing image sha256:c9be36916645b7e3c3544f4d4b9370cd32eb2d4785d28786f20a699be3771379                       0.0s
 => => naming to docker.io/library/cx_asr_realtime                                                                 0.0s
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime_updated# docker run --gpus all -p 8002:8002 cx_asr_realtime

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:58: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
DEBUG: Startup cfg.model_name='nvidia/nemotron-speech-streaming-en-0.6b' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
🚀 Preloading ASR engines (this happens once at startup)...
   Loading whisper (openai/whisper-large-v3-turbo)...
`torch_dtype` is deprecated! Use `dtype` instead!
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
INFO:asr_server:✅ Preloaded whisper (openai/whisper-large-v3-turbo) in 42.07s
   Loading nemotron (nvidia/nemotron-speech-streaming-en-0.6b)...
INFO:matplotlib.font_manager:generated new fontManager
[NeMo W 2026-02-04 22:12:20 <frozen importlib:241] Megatron num_microbatches_calculator not found, using Apex version.
WARNING:nv_one_logger.api.config:OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
INFO:nv_one_logger.exporter.export_config_manager:Final configuration contains 0 exporter(s)
WARNING:nv_one_logger.training_telemetry.api.training_telemetry_provider:No exporters were provided. This means that no telemetry data will be collected.
ERROR:asr_server:❌ Failed to preload nemotron: module 'torch.nn' has no attribute 'Buffer'
🎉 All engines preloaded! Client requests will be INSTANT.

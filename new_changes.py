FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# ---- Proxy control (build-time) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "Enabling proxy for build stage"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for build stage"; \
    fi

# ---------------------------
# System dependencies
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
RUN python3.10 -m pip install -U \
    pip \
    setuptools \
    wheel

# ---------------------------
# App dependencies FIRST
# ---------------------------
COPY requirements.txt /srv/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt

# ============================
# NeMo (same behavior as working image)
# ============================
RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# ============================
# Torch LAST (lock CUDA ABI)
# ============================
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1


# ===========================
# --- STAGE 2: RUNTIME ---
# ===========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Proxy control (runtime) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Persisting runtime proxy"; \
        echo "http_proxy=${HTTP_PROXY}" >> /etc/environment; \
        echo "https_proxy=${HTTPS_PROXY}" >> /etc/environment; \
    else \
        echo "🌐 Runtime proxy disabled"; \
    fi

# ---------------------------
# Runtime environment
# ---------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# ---------------------------
# Runtime system deps
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Copy Python environment from builder
# ---------------------------
COPY --from=builder /usr/local /usr/local

# ---------------------------
# Application code
# ---------------------------
COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8002

CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]


docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t cx_asr_realtime .

docker run --gpus all -p 8002:8002 \
  -e USE_PROXY=true \
  -e HTTP_PROXY=http://163.116.128.80:8080 \
  -e HTTPS_PROXY=http://163.116.128.80:8080 \
  cx_asr_realtime


(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t bu_digital_cx_asr_realtime .
[+] Building 3374.7s (23/23) FINISHED                                                                    docker:default
 => [internal] load build definition from Dockerfile                                                               0.0s
 => => transferring dockerfile: 3.17kB                                                                             0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                            0.2s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04                              0.5s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                         0.0s
 => [internal] load .dockerignore                                                                                  0.0s
 => => transferring context: 2B                                                                                    0.0s
 => [builder 1/9] FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c  88.2s
 => => resolve docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc  0.0s
 => => sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42 743B / 743B                         0.0s
 => => sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4 2.84kB / 2.84kB                     0.0s
 => => sha256:edd3b6bf59a6acc4d56fdcdfade4d1bc9aa206359a6823a1a43a162c3021334d 19.68kB / 19.68kB                   0.0s
 => => sha256:312a542960e3345001fc709156a5139ff8a1d8cc21a51a50f83e87ec2982f579 88.86kB / 88.86kB                   0.1s
 => => sha256:ae033ce9621d2cceaef2769ead17429ae8b29f098fb0350bdd4e0f55a36996db 670.18MB / 670.18MB                12.5s
 => => sha256:8e79813a7b9d5784bb880ca2909887465549de5183411b24f6de72fab0802bcd 2.65GB / 2.65GB                    30.0s
 => => extracting sha256:8e79813a7b9d5784bb880ca2909887465549de5183411b24f6de72fab0802bcd                         36.4s
 => => extracting sha256:312a542960e3345001fc709156a5139ff8a1d8cc21a51a50f83e87ec2982f579                          0.0s
 => => extracting sha256:ae033ce9621d2cceaef2769ead17429ae8b29f098fb0350bdd4e0f55a36996db                         18.6s
 => CACHED [stage-1 1/7] FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50  0.0s
 => [internal] load build context                                                                                  0.1s
 => => transferring context: 42.42kB                                                                               0.0s
 => [stage-1 2/7] RUN if [ "true" = "true" ]; then         echo " Enabling proxy for runtime";         export htt  0.4s
 => [stage-1 3/7] WORKDIR /srv                                                                                     0.1s
 => [stage-1 4/7] RUN apt-get update && apt-get install -y --no-install-recommends     python3.10     python3-p  126.0s
 => [builder 2/9] WORKDIR /build                                                                                   2.4s
 => [builder 3/9] RUN if [ "true" = "true" ]; then         echo "Enabling proxy for build stage";         export   0.6s
 => [builder 4/9] RUN apt-get update && apt-get install -y --no-install-recommends     python3.10     python3-p  109.0s
 => [builder 5/9] RUN python3.10 -m pip install -U     pip     setuptools     wheel                                4.1s
 => [builder 6/9] COPY requirements.txt /srv/requirements.txt                                                      0.1s
 => [builder 7/9] RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt                         1417.5s
 => [builder 8/9] RUN python3 -m pip install --no-cache-dir     "nemo_toolkit[asr] @ git+https://github.com/NVI  186.7s
 => [builder 9/9] RUN python3 -m pip install --no-cache-dir --force-reinstall     --index-url https://download  1043.0s
 => [stage-1 5/7] COPY --from=builder /usr/local /usr/local                                                      239.3s
 => [stage-1 6/7] COPY app /srv/app                                                                                0.1s
 => [stage-1 7/7] COPY scripts /srv/scripts                                                                        0.1s
 => exporting to image                                                                                           143.7s
 => => exporting layers                                                                                          143.6s
 => => writing image sha256:fbe8efa5a966efbab30d2643fe1b55439633982c117f58a2a3c92d0cce772b4e                       0.0s
 => => naming to docker.io/library/bu_digital_cx_asr_realtime                                                      0.0s
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# docker run --gpus all -p 8000:8002 \
  -e USE_PROXY=true \
  -e HTTP_PROXY=http://163.116.128.80:8080 \
  -e HTTPS_PROXY=http://163.116.128.80:8080 \
  bu_digital_cx_asr_realtime

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
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
DEBUG: Startup cfg.model_name='nvidia/nemotron-speech-streaming-en-0.6b' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
🚀 Preloading ASR engines (this happens once at startup)...
   Loading whisper (openai/whisper-large-v3-turbo)...
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 4a17019f-58e0-4a1d-9b9e-bbf43e1e99d0)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 4a17019f-58e0-4a1d-9b9e-bbf43e1e99d0)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
Retrying in 1s [Retry 1/5].
WARNING:huggingface_hub.utils._http:Retrying in 1s [Retry 1/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 75bdf54c-c3d8-4e69-a956-a25461c15721)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 75bdf54c-c3d8-4e69-a956-a25461c15721)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
Retrying in 2s [Retry 2/5].
WARNING:huggingface_hub.utils._http:Retrying in 2s [Retry 2/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 6533889f-e402-490f-9ec0-d62053e75c4d)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 6533889f-e402-490f-9ec0-d62053e75c4d)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
Retrying in 4s [Retry 3/5].
WARNING:huggingface_hub.utils._http:Retrying in 4s [Retry 3/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 9f04bdfd-be25-4c19-b7f2-7d0c6a4cad27)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 9f04bdfd-be25-4c19-b7f2-7d0c6a4cad27)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
Retrying in 8s [Retry 4/5].
WARNING:huggingface_hub.utils._http:Retrying in 8s [Retry 4/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: b7c2c915-715d-410f-a22e-e8831f987983)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: b7c2c915-715d-410f-a22e-e8831f987983)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
Retrying in 8s [Retry 5/5].
WARNING:huggingface_hub.utils._http:Retrying in 8s [Retry 5/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 7914c6d6-f706-4798-9593-c7cc0bcce9c4)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 7914c6d6-f706-4798-9593-c7cc0bcce9c4)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 898dd728-69d9-44d5-b4dd-b49c3530141a)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 898dd728-69d9-44d5-b4dd-b49c3530141a)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
Retrying in 1s [Retry 1/5].
WARNING:huggingface_hub.utils._http:Retrying in 1s [Retry 1/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 6e3f9a43-a57f-4ffc-a146-4638279ba0fe)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 6e3f9a43-a57f-4ffc-a146-4638279ba0fe)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
Retrying in 2s [Retry 2/5].
WARNING:huggingface_hub.utils._http:Retrying in 2s [Retry 2/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 64fcdb44-3975-4fbd-b7dc-33918e756f76)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 64fcdb44-3975-4fbd-b7dc-33918e756f76)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
Retrying in 4s [Retry 3/5].
WARNING:huggingface_hub.utils._http:Retrying in 4s [Retry 3/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 28d7d2e9-3708-4614-ba4e-28370c1438f3)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 28d7d2e9-3708-4614-ba4e-28370c1438f3)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
Retrying in 8s [Retry 4/5].
WARNING:huggingface_hub.utils._http:Retrying in 8s [Retry 4/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 20811185-3a97-4a21-92b3-9cb16b36dcd1)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 20811185-3a97-4a21-92b3-9cb16b36dcd1)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
Retrying in 8s [Retry 5/5].
WARNING:huggingface_hub.utils._http:Retrying in 8s [Retry 5/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 557ce5c6-cbab-45f4-9802-a59d695ad4d6)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 557ce5c6-cbab-45f4-9802-a59d695ad4d6)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: be2686fd-5ae4-4f77-a661-770b1de65639)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: be2686fd-5ae4-4f77-a661-770b1de65639)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
Retrying in 1s [Retry 1/5].
WARNING:huggingface_hub.utils._http:Retrying in 1s [Retry 1/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: eaa4932c-e6d1-4013-a91b-9b7cdad90e60)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: eaa4932c-e6d1-4013-a91b-9b7cdad90e60)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
Retrying in 2s [Retry 2/5].
WARNING:huggingface_hub.utils._http:Retrying in 2s [Retry 2/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: be96cbdc-dac2-4de4-9844-332134e46768)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
WARNING:huggingface_hub.utils._http:'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: be96cbdc-dac2-4de4-9844-332134e46768)')' thrown while requesting HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/video_preprocessor_config.json
Retrying in 4s [Retry 3/5].
WARNING:huggingface_hub.utils._http:Retrying in 4s [Retry 3/5].
'(ProtocolError('Connection aborted.', ConnectionResetError(104, 'Con

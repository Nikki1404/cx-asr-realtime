(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# docker build -t bu_digital_cx_asr_realtime .
[+] Building 1.1s (13/15)                                                                                docker:default
 => [internal] load build definition from Dockerfile                                                               0.0s
 => => transferring dockerfile: 1.23kB                                                                             0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                            0.2s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                         0.0s
 => [internal] load .dockerignore                                                                                  0.0s
 => => transferring context: 2B                                                                                    0.0s
 => [ 1/10] FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd  0.0s
 => [internal] load build context                                                                                  0.0s
 => => transferring context: 673B                                                                                  0.0s
 => CACHED [ 2/10] WORKDIR /srv                                                                                    0.0s
 => CACHED [ 3/10] RUN apt-get update && apt-get install -y --no-install-recommends     python3     python3-pip    0.0s
 => CACHED [ 4/10] RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel                               0.0s
 => CACHED [ 5/10] RUN python3 -m pip install --no-cache-dir     --index-url https://download.pytorch.org/whl/cu1  0.0s
 => CACHED [ 6/10] COPY requirements.txt /srv/requirements.txt                                                     0.0s
 => CACHED [ 7/10] RUN python3 -m pip install --no-cache-dir -r /srv/requirements.txt                              0.0s
 => ERROR [ 8/10] RUN python3 -m pip install --no-cache-dir     "git+https://github.com/NVIDIA/NeMo.git@main#egg=  0.8s
------
 > [ 8/10] RUN python3 -m pip install --no-cache-dir     "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]":
0.673 error: invalid-egg-fragment
0.673
0.673 × The 'nemo_toolkit[asr]' egg fragment is invalid
0.673 ╰─> from 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'
0.673
0.673 hint: Try using the Direct URL requirement syntax: 'name[extra] @ URL'
------
Dockerfile:35
--------------------
  34 |
  35 | >>> RUN python3 -m pip install --no-cache-dir \
  36 | >>>     "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
  37 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c python3 -m pip install --no-cache-dir     \"git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]\"" did not complete successfully: exit code: 1

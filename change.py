(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime#  docker run --gpus all -p 8000:8000 -e ASR_BACKEND=whisper -e MODEL_NAME=openai/whisper-large-v3-turbo bu_digital_cx_asr_realtime

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
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
ERROR:    Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 571, in __aenter__
    await self._router.startup()
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 671, in startup
    await handler()
  File "/srv/app/main.py", line 28, in startup
    load_sec = engine.load()
  File "/srv/app/asr_engines/whisper_asr.py", line 46, in load
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
    return model_class.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4881, in from_pretrained
    hf_quantizer, config, dtype, device_map = get_hf_quantizer(
  File "/usr/local/lib/python3.10/dist-packages/transformers/quantizers/auto.py", line 319, in get_hf_quantizer
    hf_quantizer.validate_environment(
  File "/usr/local/lib/python3.10/dist-packages/transformers/quantizers/quantizer_bnb_8bit.py", line 73, in validate_environment
    raise ImportError(
ImportError: Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`

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
`torch_dtype` is deprecated! Use `dtype` instead!
ERROR:    Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 694, in lifespan
    async with self.lifespan_context(app) as maybe_state:
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 571, in __aenter__
    await self._router.startup()
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 671, in startup
    await handler()
  File "/srv/app/main.py", line 28, in startup
    load_sec = engine.load()
  File "/srv/app/asr_engines/whisper_asr.py", line 50, in load
    self.model = self.model.cuda()
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4276, in cuda
    return super().cuda(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1093, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 933, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 933, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 933, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 964, in _apply
    param_applied = fn(param)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1093, in <lambda>
    return self._apply(lambda t: t.cuda(device))
torch.AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

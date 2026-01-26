 => => naming to docker.io/library/bu_digital_cx_asr_realtime                                                                                                                                                                                                         0.0s
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime#   docker run --gpus all -p 8000:8000 -e ASR_BACKEND=whisper -e MODEL_NAME=openai/whisper-large-v3-turbo bu_digital_cx_asr_realtime

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
INFO:     Started server process [1]
INFO:     Waiting for application startup.
`torch_dtype` is deprecated! Use `dtype` instead!
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
INFO:asr_server:ASR backend=whisper model=openai/whisper-large-v3-turbo loaded in 30.88s
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.17.0.1:45076 - "WebSocket /ws/asr" [accepted]
INFO:asr_server:WS connected: Address(host='172.17.0.1', port=45076)
INFO:     connection open
INFO:asr_server:WS disconnected
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/websockets/websockets_impl.py", line 244, in run_asgi
    result = await self.app(self.scope, self.asgi_receive, self.asgi_send)  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1135, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 151, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 364, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 141, in app
    await wrap_app_handling_exceptions(app, session)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 138, in app
    await func(session)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 438, in app
    await dependant.call(**solved_result.values)
  File "/srv/app/main.py", line 162, in ws_asr
    final = session.finalize(cfg.post_speech_pad_ms)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/srv/app/asr_engines/whisper_asr.py", line 149, in finalize
    generated_ids = self.engine.model.generate(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/generation_whisper.py", line 704, in generate
    init_tokens = self._retrieve_init_tokens(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/generation_whisper.py", line 1572, in _retrieve_init_tokens
    lang_ids = self.detect_language(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/generation_whisper.py", line 1676, in detect_language
    logits = self(**inputs, decoder_input_ids=decoder_input_ids, use_cache=False).logits[:, -1]
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1289, in forward
    outputs = self.model(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 1114, in forward
    encoder_outputs = self.encoder(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py", line 679, in forward
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 375, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py", line 370, in _conv_forward
    return F.conv1d(
RuntimeError: Input type (float) and bias type (c10::Half) should be the same

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime_updated# docker run --gpus all -p 8002:8002 bu_digital_cx_asr_realtime_new

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
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
=INFO:     172.17.0.1:52476 - "WebSocket /ws/asr" [accepted]
INFO:     connection open
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", line 479, in cached_files
    hf_hub_download(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 106, in _inner_fn
    validate_repo_id(arg_value)
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 160, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.

During handling of the above exception, another exception occurred:

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
  File "/srv/app/main.py", line 63, in ws_asr
    engine = get_engine(backend)
  File "/srv/app/main.py", line 38, in get_engine
    load_sec = engine.load()
  File "/srv/app/asr_engines/whisper_asr.py", line 42, in load
    self.processor = AutoProcessor.from_pretrained(self.model_name)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/processing_auto.py", line 303, in from_pretrained
    processor_config_file = cached_file(pretrained_model_name_or_path, PROCESSOR_NAME, **cached_file_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", line 531, in cached_files
    resolved_files = [
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", line 532, in <listcomp>
    _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision, repo_type)
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", line 143, in _get_cache_file_to_return
    resolved_file = try_to_load_from_cache(
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 106, in _inner_fn
    validate_repo_id(arg_value)
  File "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_validators.py", line 160, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.

#main.py
import asyncio
import json
import time
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config, Config
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.factory import build_engine
from app.asr_engines.base import ASREngine

cfg = load_config()
logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("asr_server")

app = FastAPI()

# cache engines per backend
ENGINE_CACHE: dict[str, ASREngine] = {}


def get_engine(backend: str) -> ASREngine:
    if backend in ENGINE_CACHE:
        return ENGINE_CACHE[backend]

    tmp_cfg = Config()
    tmp_cfg.asr_backend = backend
    tmp_cfg.model_name = ""
    tmp_cfg.device = cfg.device
    tmp_cfg.sample_rate = cfg.sample_rate
    tmp_cfg.context_right = cfg.context_right

    engine = build_engine(tmp_cfg)
    load_sec = engine.load()
    log.info(f"Loaded backend={backend} in {load_sec:.2f}s")

    ENGINE_CACHE[backend] = engine
    return engine


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    # 🔑 FIRST MESSAGE MUST BE CONFIG
    init = await ws.receive_text()
    init_obj = json.loads(init)

    backend = init_obj.get("backend")
    if backend not in ("nemotron", "whisper"):
        await ws.close(code=4000)
        return

    engine = get_engine(backend)

    labels = (backend, engine.model_name)

    active_streams = ACTIVE_STREAMS.labels(*labels)
    partials_total = PARTIALS_TOTAL.labels(*labels)
    finals_total = FINALS_TOTAL.labels(*labels)
    utterances_total = UTTERANCES_TOTAL.labels(*labels)

    ttft_wall = TTFT_WALL.labels(*labels)
    ttf_wall = TTF_WALL.labels(*labels)

    infer_sec = INFER_SEC.labels(*labels)
    preproc_sec = PREPROC_SEC.labels(*labels)
    flush_sec = FLUSH_SEC.labels(*labels)

    audio_sec_hist = AUDIO_SEC.labels(*labels)
    rtf_hist = RTF.labels(*labels)
    backlog_ms_gauge = BACKLOG_MS.labels(*labels)

    active_streams.inc()
    log.info(f"WS connected ({backend}) {ws.client}")

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

    frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
    raw_buf = bytearray()

    utt_started = False
    utt_audio_ms = 0
    t_utt_start = None
    t_first_partial = None
    silence_ms = 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                continue

            if data == b"":
                if utt_started:
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "eos",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
                break

            raw_buf.extend(data)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)
                silence_ms = 0 if is_speech else silence_ms + cfg.vad_frame_ms

                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_utt_start = time.time()
                    t_first_partial = None
                    silence_ms = 0
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                if engine.caps.partials:
                    text = session.step_if_ready()
                    if text:
                        partials_total.inc()
                        if t_first_partial is None:
                            t_first_partial = time.time()
                            if engine.caps.ttft_meaningful:
                                ttft_wall.observe(t_first_partial - t_utt_start)
                        await ws.send_text(json.dumps({"type": "partial", "text": text}))

                if (
                    not is_speech
                    and utt_audio_ms >= cfg.min_utt_ms
                    and silence_ms >= cfg.end_silence_ms
                ):
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "silence",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

    finally:
        active_streams.dec()
        await ws.close()
        log.info("WS disconnected")


async def _emit_final(
    ws,
    session,
    final_text,
    audio_ms,
    t_start,
    t_first_partial,
    reason,
    utterances_total,
    finals_total,
    ttf_wall,
    audio_sec_hist,
    rtf_hist,
    engine,
):
    if not final_text:
        return

    utterances_total.inc()
    finals_total.inc()

    audio_sec = audio_ms / 1000.0
    ttf = time.time() - t_start

    ttf_wall.observe(ttf)
    audio_sec_hist.observe(audio_sec)

    compute_sec = session.utt_preproc + session.utt_infer + session.utt_flush
    if audio_sec > 0:
        rtf_hist.observe(compute_sec / audio_sec)

    await ws.send_text(json.dumps({
        "type": "final",
        "text": final_text,
        "reason": reason,
        "audio_ms": audio_ms,
        "ttf_ms": int(ttf * 1000),
        "ttft_ms": (
            int((t_first_partial - t_start) * 1000)
            if t_first_partial and engine.caps.ttft_meaningful
            else None
        ),
        "chunks": session.chunks,
        "model_preproc_ms": int(session.utt_preproc * 1000),
        "model_infer_ms": int(session.utt_infer * 1000),
        "model_flush_ms": int(session.utt_flush * 1000),
        "rtf": (compute_sec / audio_sec) if audio_sec > 0 else None,
    }))


#whisper_asr.py-
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.asr_engines.base import ASREngine, EngineCaps


class WhisperTurboASR(ASREngine):
    """
    Chunked (non-streaming) ASR using Whisper Turbo.

    IMPORTANT:
    - Whisper is NOT a true streaming ASR.
    - No partials.
    - No meaningful TTFT.
    - Final transcription only.
    """

    caps = EngineCaps(
        streaming=False,
        partials=False,
        ttft_meaningful=False,
    )

    def __init__(self, model_name: str, device: str, sample_rate: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate

        self.model = None
        self.processor = None

    def load(self) -> float:
        """
        Load Whisper model + processor.
        """
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model.eval()

        # Warmup (important to avoid first-request latency spike)
        self._warmup()

        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        """
        Warm up with ~1s of silence.
        """
        try:
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            inputs = self.processor(
                silence,
                sampling_rate=self.sr,
                return_tensors="pt",
            )

            #  FIX (only change): match model dtype
            inputs = {
                k: v.to(
                    device=self.model.device,
                    dtype=self.model.dtype
                )
                for k, v in inputs.items()
            }

            _ = self.model.generate(**inputs)
        except Exception:
            # Warmup must never crash startup
            pass

    def new_session(self, max_buffer_ms: int):
        return WhisperSession(self, max_buffer_ms=max_buffer_ms)


class WhisperSession:
    """
    Per-utterance session for Whisper.

    Behavior:
    - Buffers audio until finalize()
    - No partial outputs
    - Single forward pass on finalize
    """

    def __init__(self, engine: WhisperTurboASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        # audio buffer (float32)
        self.audio = np.array([], dtype=np.float32)

        # timing accumulators (kept for consistency with metrics)
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

    def accept_pcm16(self, pcm16: bytes):
        """
        Append PCM16 audio to buffer.
        """
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])

        # bound buffer
        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]

    def step_if_ready(self) -> Optional[str]:
        """
        Whisper does NOT support partials.
        Always return None.
        """
        return None

    @torch.inference_mode()
    def finalize(self, pad_ms: int) -> str:
        """
        Run full Whisper transcription.
        """
        if len(self.audio) == 0:
            return ""

        # pad to avoid clipping last word
        pad = np.zeros(int(self.engine.sr * (pad_ms / 1000.0)), dtype=np.float32)
        audio = np.concatenate([self.audio, pad])

        # preprocess
        t0 = time.perf_counter()
        inputs = self.engine.processor(
            audio,
            sampling_rate=self.engine.sr,
            return_tensors="pt",
        )
        self.utt_preproc += (time.perf_counter() - t0)

        #  FIX (only change): match model dtype
        inputs = {
            k: v.to(
                device=self.engine.model.device,
                dtype=self.engine.model.dtype
            )
            for k, v in inputs.items()
        }

        # inference
        t1 = time.perf_counter()
        generated_ids = self.engine.model.generate(
            **inputs,
            max_new_tokens=444,
        )
        self.utt_infer += (time.perf_counter() - t1)

        self.chunks += 1

        # decode
        text = self.engine.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        # reset buffer for next utterance
        self.audio = np.array([], dtype=np.float32)

        return text

import os
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.asr_engines.base import ASREngine, EngineCaps


class WhisperTurboASR(ASREngine):
    """
    Production notes:
    - GPU-only
    - OOM-safe model loading using device_map="cuda"
    - Optional quantization (8bit/4bit) via env vars
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
        Load Whisper model + processor (GPU-only, OOM-safe).
        """
        t0 = time.time()

        if self.device != "cuda":
            raise RuntimeError("WhisperTurboASR is configured as GPU-only. Set DEVICE=cuda.")

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Optional quantization knobs
        # Set one of:
        #   WHISPER_LOAD_IN_8BIT=1
        #   WHISPER_LOAD_IN_4BIT=1
        # Requires bitsandbytes installed in image.
        load_in_8bit = os.getenv("WHISPER_LOAD_IN_8BIT", "0") == "1"
        load_in_4bit = os.getenv("WHISPER_LOAD_IN_4BIT", "0") == "1"

        # Default dtype when not quantized
        dtype = torch.float16

        # Build kwargs for from_pretrained
        kwargs = dict(
            low_cpu_mem_usage=True,
        )

        # OOM-safe GPU loading (prevents .cuda() allocation spike)
        # device_map="cuda" keeps it on GPU but loads safely
        kwargs["device_map"] = "cuda"

        # If quantizing, pass flags (Transformers may warn it's deprecated, but works depending on version)
        # Better: BitsAndBytesConfig (requires transformers+bnb versions aligned). We keep this minimal & robust.
        if load_in_4bit and load_in_8bit:
            raise RuntimeError("Set only one of WHISPER_LOAD_IN_4BIT or WHISPER_LOAD_IN_8BIT.")

        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            # dtype still relevant for compute
            kwargs["torch_dtype"] = dtype
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs["torch_dtype"] = dtype
        else:
            # Standard FP16 load
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            **kwargs,
        )

        self.model.eval()

        # Warmup (avoid first-request latency spike)
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

            #  Ensure inputs match model dtype/device
            inputs = {
                k: v.to(device=self.model.device, dtype=self.model.dtype)
                for k, v in inputs.items()
            }

            # Keep tokens safe (no overflow)
            _ = self.model.generate(**inputs, max_new_tokens=64)
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

        # Match model dtype/device
        inputs = {
            k: v.to(device=self.engine.model.device, dtype=self.engine.model.dtype)
            for k, v in inputs.items()
        }

        # inference
        t1 = time.perf_counter()

        #  FIX: avoid max_new_tokens overflow for Whisper
        # decoder_input_ids already consumes some space; 444 is safe for max_target_positions=448.
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

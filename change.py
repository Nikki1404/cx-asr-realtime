import os
import time
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    BitsAndBytesConfig,
)

from app.asr_engines.base import ASREngine, EngineCaps


class WhisperTurboASR(ASREngine):
    """
    Chunked (non-streaming) ASR using Whisper Turbo (Large V3).

    IMPORTANT:
    - Whisper is NOT a true streaming ASR
    - No partials
    - No meaningful TTFT
    - Final transcription only

    Production guarantees:
    - GPU-only
    - OOM-safe model loading
    - Optional 4-bit / 8-bit quantization
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

    # MODEL LOAD (OOM-SAFE + QUANTIZED)
    def load(self) -> float:
        """
        Load Whisper model + processor safely on GPU.
        """
        t0 = time.time()

        if self.device != "cuda":
            raise RuntimeError(
                "WhisperTurboASR is GPU-only. Set DEVICE=cuda."
            )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Quantization selection (env-driven)
        load_in_4bit = os.getenv("WHISPER_LOAD_IN_4BIT", "0") == "1"
        load_in_8bit = os.getenv("WHISPER_LOAD_IN_8BIT", "0") == "1"

        if load_in_4bit and load_in_8bit:
            raise RuntimeError(
                "Only one of WHISPER_LOAD_IN_4BIT or WHISPER_LOAD_IN_8BIT may be set"
            )

        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # CRITICAL: device_map="cuda" prevents load-time OOM
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        # Warmup to avoid first-request latency
        self._warmup()

        return time.time() - t0


    # WARMUP
    @torch.inference_mode()
    def _warmup(self):
        """
        Run a short silent inference to warm CUDA kernels.
        """
        try:
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            inputs = self.processor(
                silence,
                sampling_rate=self.sr,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(
                    device=self.model.device,
                    dtype=self.model.dtype
                )
                for k, v in inputs.items()
            }

            self.model.generate(**inputs, max_new_tokens=64)
        except Exception:
            # Warmup must NEVER break startup
            pass

    # SESSION
    def new_session(self, max_buffer_ms: int):
        return WhisperSession(self, max_buffer_ms=max_buffer_ms)


class WhisperSession:
    """
    Per-utterance session for Whisper.

    Behavior:
    - Buffers audio until finalize()
    - No partials
    - Single forward pass
    """

    def __init__(self, engine: WhisperTurboASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        self.audio = np.array([], dtype=np.float32)

        # Metrics compatibility
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

    # AUDIO INGEST
    def accept_pcm16(self, pcm16: bytes):
        x = (
            np.frombuffer(pcm16, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

        self.audio = np.concatenate([self.audio, x])

        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples :]

    # NO PARTIALS
    def step_if_ready(self) -> Optional[str]:
        return None

    # FINAL TRANSCRIPTION
    @torch.inference_mode()
    def finalize(self, pad_ms: int) -> str:
        if len(self.audio) == 0:
            return ""

        pad = np.zeros(
            int(self.engine.sr * (pad_ms / 1000.0)),
            dtype=np.float32,
        )
        audio = np.concatenate([self.audio, pad])

        # Preprocess
        t0 = time.perf_counter()
        inputs = self.engine.processor(
            audio,
            sampling_rate=self.engine.sr,
            return_tensors="pt",
        )
        self.utt_preproc += time.perf_counter() - t0

        inputs = {
            k: v.to(
                device=self.engine.model.device,
                dtype=self.engine.model.dtype
            )
            for k, v in inputs.items()
        }

        # Inference
        t1 = time.perf_counter()

        # SAFE TOKEN LIMIT (Whisper max_target_positions=448)
        generated_ids = self.engine.model.generate(
            **inputs,
            max_new_tokens=444,
        )

        self.utt_infer += time.perf_counter() - t1
        self.chunks += 1

        # Decode
        text = self.engine.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        self.audio = np.array([], dtype=np.float32)
        return text

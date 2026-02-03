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

            # ✅ FIX (only change): match model dtype
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

        # ✅ FIX (only change): match model dtype
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

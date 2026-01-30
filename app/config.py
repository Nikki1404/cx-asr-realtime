from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BackendParams:
    end_silence_ms: int
    short_pause_flush_ms: int   # Whisper-only (used indirectly)
    min_utt_ms: int
    finalize_pad_ms: int


@dataclass
class Config:
    # backend
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")

    # model
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # nemotron streaming
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))

    # VAD
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    # utterance bounds (USED BY main.py)
    min_utt_ms: int = 250
    end_silence_ms: int = 900
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
    post_speech_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    backend_params: BackendParams | None = None


def load_config() -> Config:
    cfg = Config()

    # Default model selection
    if not cfg.model_name:
        cfg.model_name = (
            "openai/whisper-large-v3-turbo"
            if cfg.asr_backend == "whisper"
            else "nvidia/nemotron-speech-streaming-en-0.6b"
        )

    # Backend-specific tuning
    if cfg.asr_backend == "whisper":
        cfg.backend_params = BackendParams(
            end_silence_ms=int(os.getenv("WHISPER_END_SILENCE_MS", "900")),
            short_pause_flush_ms=int(os.getenv("WHISPER_SHORT_PAUSE_FLUSH_MS", "350")),
            min_utt_ms=int(os.getenv("WHISPER_MIN_UTT_MS", "250")),
            finalize_pad_ms=cfg.post_speech_pad_ms,
        )
    else:
        cfg.backend_params = BackendParams(
            end_silence_ms=int(os.getenv("NEMO_END_SILENCE_MS", "900")),
            short_pause_flush_ms=0,
            min_utt_ms=int(os.getenv("NEMO_MIN_UTT_MS", "250")),
            finalize_pad_ms=cfg.post_speech_pad_ms,
        )

    # IMPORTANT: expose backend params to main.py (NO CODE CHANGE THERE)
    cfg.end_silence_ms = cfg.backend_params.end_silence_ms
    cfg.min_utt_ms = cfg.backend_params.min_utt_ms
    cfg.post_speech_pad_ms = cfg.backend_params.finalize_pad_ms

    return cfg

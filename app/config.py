from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class BackendParams:
    end_silence_ms: int
    short_pause_flush_ms: int
    min_utt_ms: int
    finalize_pad_ms: int


@dataclass(frozen=True)
class Config:
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))
    min_utt_ms: int = 250
    end_silence_ms: int = 900
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
    post_speech_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    backend_params: BackendParams | None = None


# MODEL MAPPING - Backend → Model
MODEL_MAP = {
    "whisper": "openai/whisper-large-v3-turbo",
    "nemotron": "nvidia/nemotron-speech-streaming-en-0.6b",
    # for google we label model name for metrics; actual model is in env GOOGLE_MODEL
    "google": "google-stt-v2-streaming",
}


def load_config() -> Config:
    cfg = Config()

    # Default model for startup (nemotron)
    if not cfg.model_name:
        cfg = replace(cfg, model_name=MODEL_MAP["nemotron"])

    # Backend-specific tuning defaults
    backend_params = BackendParams(
        end_silence_ms=int(os.getenv("NEMO_END_SILENCE_MS", "900")),
        short_pause_flush_ms=0,
        min_utt_ms=int(os.getenv("NEMO_MIN_UTT_MS", "250")),
        finalize_pad_ms=cfg.post_speech_pad_ms,
    )

    if cfg.asr_backend == "whisper":
        backend_params = BackendParams(
            end_silence_ms=int(os.getenv("WHISPER_END_SILENCE_MS", "900")),
            short_pause_flush_ms=int(os.getenv("WHISPER_SHORT_PAUSE_FLUSH_MS", "350")),
            min_utt_ms=int(os.getenv("WHISPER_MIN_UTT_MS", "250")),
            finalize_pad_ms=cfg.post_speech_pad_ms,
        )

    if cfg.asr_backend == "google":
        # Google streaming can be tuned more aggressively if you want.
        backend_params = BackendParams(
            end_silence_ms=int(os.getenv("GOOGLE_END_SILENCE_MS", "700")),  # lower than default
            short_pause_flush_ms=0,
            min_utt_ms=int(os.getenv("GOOGLE_MIN_UTT_MS", "200")),
            finalize_pad_ms=int(os.getenv("GOOGLE_FINALIZE_PAD_MS", str(cfg.post_speech_pad_ms))),
        )

        # label model for metrics; actual model is env GOOGLE_MODEL (default in engine)
        cfg = replace(cfg, model_name=MODEL_MAP["google"])

    cfg = replace(cfg, backend_params=backend_params)
    cfg = replace(
        cfg,
        end_silence_ms=backend_params.end_silence_ms,
        min_utt_ms=backend_params.min_utt_ms,
        post_speech_pad_ms=backend_params.finalize_pad_ms,
    )

    print(f"DEBUG: Startup cfg.model_name='{cfg.model_name}' cfg.asr_backend='{cfg.asr_backend}'")
    return cfg

from app.config import Config
from app.asr_engines.nemotron_asr import NemotronStreamingASR
from app.asr_engines.whisper_asr import WhisperTurboASR
from app.asr_engines.google_streaming_asr import GoogleStreamingASR


def build_engine(cfg: Config):

    # -------------------------
    # NEMOTRON
    # -------------------------
    if cfg.asr_backend == "nemotron":
        return NemotronStreamingASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
            context_right=cfg.context_right,
        )

    # -------------------------
    # WHISPER
    # -------------------------
    if cfg.asr_backend == "whisper":
        return WhisperTurboASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )

    # -------------------------
    # GOOGLE STREAMING
    # -------------------------
    if cfg.asr_backend == "google":
        if not cfg.google_recognizer:
            raise ValueError("GOOGLE_RECOGNIZER is required for google backend")

        return GoogleStreamingASR(
            recognizer=cfg.google_recognizer,
            region=cfg.google_region,
            sample_rate=cfg.sample_rate,
            language_code=cfg.google_language,
            model=cfg.google_model,
            interim_results=cfg.google_interim,
            explicit_decoding=cfg.google_explicit_decoding,
        )

    # -------------------------
    # Unsupported
    # -------------------------
    raise ValueError(f"Unsupported ASR_BACKEND={cfg.asr_backend}")

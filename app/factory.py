from app.config import Config
from app.asr_engines.nemotron_asr import NemotronStreamingASR
from app.asr_engines.whisper_asr import WhisperTurboASR
from app.asr_engines.google_streaming_asr import GoogleStreamingASR


def build_engine(cfg: Config):

    if cfg.asr_backend == "nemotron":
        return NemotronStreamingASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
            context_right=cfg.context_right,
        )

    if cfg.asr_backend == "whisper":
        return WhisperTurboASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )
    if cfg.asr_backend == "google":
        # Google engine reads everything from environment variables
        return GoogleStreamingASR(
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(f"Unsupported ASR_BACKEND={cfg.asr_backend}")

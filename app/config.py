from dataclasses import dataclass
import os


@dataclass
class Config:
    # backend selection
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")  # nemotron|whisper

    # model
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")  # cuda/cpu
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # nemotron streaming knob
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))

    # VAD + endpointing
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    # Endpointing (pause triggers final)
    end_silence_ms: int = int(os.getenv("END_SILENCE_MS", "900"))
    min_utt_ms: int = int(os.getenv("MIN_UTT_MS", "250"))
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    # When finalizing, we add extra zero padding (ms) to flush last words
    finalize_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))
    post_speech_pad_ms: int = finalize_pad_ms

    # Keep ring-buffer bounded
    max_buffer_ms: int = int(os.getenv("MAX_BUFFER_MS", "12000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def load_config() -> Config:
    cfg = Config()

    # default model by backend if MODEL_NAME not provided
    if not cfg.model_name:
        if cfg.asr_backend == "whisper":
            cfg.model_name = "openai/whisper-large-v3-turbo"
        else:
            cfg.model_name = "nvidia/nemotron-speech-streaming-en-0.6b"

    return cfg

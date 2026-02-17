/usr/local/lib/python3.10/dist-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.12) which Google will stop supporting in new releases of google.cloud.speech_v2 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.speech_v2 past that date.
  warnings.warn(message, FutureWarning)
DEBUG: Startup cfg.model_name='nvidia/nemotron-speech-streaming-en-0.6b' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
   Loading whisper (openai/whisper-large-v3-turbo)...
2026-02-17 10:20:25,445 - INFO -  Preloading ASR engines (this happens once at startup)...
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
2026-02-17 10:20:26,978 - ERROR -  Failed to preload whisper: Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`

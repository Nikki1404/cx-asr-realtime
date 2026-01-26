from prometheus_client import Counter, Histogram, Gauge

LABELS = ["backend", "model"]

ACTIVE_STREAMS = Gauge("asr_active_streams", "Active websocket streams", LABELS)

PARTIALS_TOTAL = Counter("asr_partials_total", "Partial messages sent", LABELS)
FINALS_TOTAL = Counter("asr_finals_total", "Final messages sent", LABELS)
UTTERANCES_TOTAL = Counter("asr_utterances_total", "Utterances finalized", LABELS)

# NOTE: TTFT only recorded when engine.caps.ttft_meaningful == True (Nemotron)
TTFT_WALL = Histogram("asr_ttft_wall_sec", "Wall TTFT seconds (streaming only)", LABELS)
TTF_WALL  = Histogram("asr_ttf_wall_sec", "Wall TTF seconds", LABELS)

INFER_SEC = Histogram("asr_infer_sec", "Model inference seconds", LABELS)
PREPROC_SEC = Histogram("asr_preproc_sec", "Model preproc seconds", LABELS)
FLUSH_SEC = Histogram("asr_flush_sec", "Finalize/flush wall seconds", LABELS)

AUDIO_SEC = Histogram("asr_audio_sec", "Audio seconds per utterance", LABELS)
RTF = Histogram("asr_rtf", "Real-time factor (infer/audio)", LABELS)

BACKLOG_MS = Gauge("asr_backlog_ms", "Buffered audio backlog (ms)", LABELS)


GPU_UTIL = Gauge("asr_gpu_util", "GPU utilization percent")
GPU_MEM_USED_MB = Gauge("asr_gpu_mem_used_mb", "GPU memory used MB")
GPU_MEM_TOTAL_MB = Gauge("asr_gpu_mem_total_mb", "GPU memory total MB")

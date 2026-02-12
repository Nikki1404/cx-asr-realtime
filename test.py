(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# docker run --gpus all -p 8000:8002   -e USE_PROXY=true   -e HTTP_PROXY=http://163.116.128.80:8080   -e HTTPS_PROXY=http://163.116.128.80:8080 -e http_proxy=http://163.116.128.80:8080   -e https_proxy=http://163.116.128.80:8080 bu_digital_asr_realtime

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.12) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/usr/local/lib/python3.10/dist-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.12) which Google will stop supporting in new releases of google.cloud.speech_v2 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.speech_v2 past that date.
  warnings.warn(message, FutureWarning)
DEBUG: Startup cfg.model_name='nvidia/nemotron-speech-streaming-en-0.6b' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
🚀 Preloading ASR engines (this happens once at startup)...
   Loading whisper (openai/whisper-large-v3-turbo)...
`torch_dtype` is deprecated! Use `dtype` instead!
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
INFO:asr_server:✅ Preloaded whisper (openai/whisper-large-v3-turbo) in 42.78s
   Loading nemotron (nvidia/nemotron-speech-streaming-en-0.6b)...
INFO:matplotlib.font_manager:generated new fontManager
[NeMo W 2026-02-12 13:00:31 <frozen importlib:241] Megatron num_microbatches_calculator not found, using Apex version.
WARNING:nv_one_logger.api.config:OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
INFO:nv_one_logger.exporter.export_config_manager:Final configuration contains 0 exporter(s)
WARNING:nv_one_logger.training_telemetry.api.training_telemetry_provider:No exporters were provided. This means that no telemetry data will be collected.
[NeMo I 2026-02-12 13:01:48 mixins:69] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo W 2026-02-12 13:01:49 rnnt_models:63] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    use_lhotse: true
    skip_missing_manifest_entries: true
    input_cfg: null
    tarred_audio_filepaths: null
    manifest_filepath: null
    sample_rate: 16000
    shuffle: true
    num_workers: 2
    pin_memory: true
    max_duration: 40.0
    min_duration: 0.1
    text_field: answer
    batch_duration: null
    use_bucketing: true
    max_tps: null
    bucket_duration_bins: null
    bucket_batch_size: null
    num_buckets: null
    bucket_buffer_size: null
    shuffle_buffer_size: null
    augmentor: null

[NeMo W 2026-02-12 13:01:49 rnnt_models:63] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    use_lhotse: true
    manifest_filepath: /data/ASR/en/librispeech/test-other.json
    sample_rate: 16000
    batch_size: 32
    shuffle: false
    max_duration: 40.0
    min_duration: 0.1
    num_workers: 2
    pin_memory: true
    text_field: answer
    tarred_audio_filepaths: null

[NeMo I 2026-02-12 13:01:54 rnnt_models:83] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-12 13:01:54 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-12 13:01:54 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-12 13:01:57 modelPT:502] Model EncDecRNNTBPEModel was successfully restored from /srv/hf_cache/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/c0acae9cc4163ab0d45cd403fbecbcb0635ee685/nemotron-speech-streaming-en-0.6b.nemo.
[NeMo I 2026-02-12 13:01:57 rnnt_models:231] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.005, 'clamp': -1.0}
[NeMo I 2026-02-12 13:01:57 nemotron_asr:117] Changed decoding strategy to
    model_type: rnnt
    strategy: greedy
    compute_hypothesis_token_set: false
    preserve_alignments: null
    tdt_include_token_duration: null
    confidence_cfg:
      preserve_frame_confidence: false
      preserve_token_confidence: false
      preserve_word_confidence: false
      exclude_blank: true
      aggregation: min
      tdt_include_duration: false
      method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
    fused_batch_size: null
    compute_timestamps: null
    compute_langs: false
    word_seperator: ' '
    segment_seperators:
    - .
    - '!'
    - '?'
    segment_gap_threshold: null
    rnnt_timestamp_type: all
    greedy:
      max_symbols_per_step: 10
      preserve_alignments: false
      preserve_frame_confidence: false
      tdt_include_token_duration: false
      tdt_include_duration_confidence: false
      confidence_method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
      loop_labels: false
      use_cuda_graph_decoder: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
        key_phrases_file: null
        key_phrases_list: null
        key_phrase_items_list: null
        context_score: 1.0
        depth_scaling: 2.0
        unk_score: 0.0
        final_eos_score: 1.0
        score_per_phrase: 0.0
        source_lang: en
        use_triton: true
        uniform_weights: false
        use_bpe_dropout: false
        num_of_transcriptions: 5
        bpe_alpha: 0.3
      boosting_tree_alpha: 0.0
      enable_per_stream_biasing: false
      max_symbols: 10
    beam:
      beam_size: 4
      search_type: default
      score_norm: true
      return_best_hypothesis: true
      tsd_max_sym_exp_per_step: 50
      alsd_max_target_len: 1.0
      nsc_max_timesteps_expansion: 1
      nsc_prefix_alpha: 1
      maes_num_steps: 2
      maes_prefix_alpha: 1
      maes_expansion_gamma: 2.3
      maes_expansion_beta: 2
      language_model: null
      softmax_temperature: 1.0
      preserve_alignments: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
        key_phrases_file: null
        key_phrases_list: null
        key_phrase_items_list: null
        context_score: 1.0
        depth_scaling: 2.0
        unk_score: 0.0
        final_eos_score: 1.0
        score_per_phrase: 0.0
        source_lang: en
        use_triton: true
        uniform_weights: false
        use_bpe_dropout: false
        num_of_transcriptions: 5
        bpe_alpha: 0.3
      boosting_tree_alpha: 0.0
      hat_subtract_ilm: false
      hat_ilm_weight: 0.0
      max_symbols_per_step: 10
      blank_lm_score_mode: LM_WEIGHTED_FULL
      pruning_mode: LATE
      allow_cuda_graphs: true
    temperature: 1.0
    durations: []
    big_blank_durations: []

INFO:asr_server:✅ Preloaded nemotron (nvidia/nemotron-speech-streaming-en-0.6b) in 82.02s
ERROR:asr_server:❌ Failed to preload google: GoogleStreamingASR.__init__() got an unexpected keyword argument 'recognizer'
   Loading google (google-stt-v2-streaming)...
🎉 All engines preloaded! Client requests will be INSTANT.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)

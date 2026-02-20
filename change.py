Nemotron-Speech-Streaming-En-0.6b is a transcription model that belongs to the Nemotron Speech open-model family.  This model was announced on January 2026 and it is engineered to deliver high-quality English transcription across both low-latency streaming and high-throughput batch workloads. 

Unlike traditional "buffered" streaming, the Native Streaming architecture of this model enables continuous transcription by processing only new audio chunks while reusing cached encoder context. This significantly improves computational efficiency and minimizes end-to-end delay without sacrificing accuracy.  So, it is suitable for low-latency voice agent applications interaction. 

Additonally, compared to traditional buffered streaming approaches, this model also allows for a higher number of parallel streams within the same GPU memory constraints. 

We deployed this model and our findings also suggest this model to be a better fit (compared to google.cloud.speech_v2 and whisper) for our asr-realtime needs because of the following reasons: 
  - It faired better when we tested it on some audio files from librispeech (used for huggingface leaderboard too)  
  - The latency for this model was computed to be better 
  - On top of that, it supports partial transcripts which further improves user experience.  We will look for google models supporting partial transcripts next. 
    - p.s. Whisper does not support native streaming as per this documentation that says it works on 30ms chunks (https://openai.com/index/whisper/) 
  - This model is a bit easier to deploy compared to Nvidia's parakeet model (which requires exposing a service of it's own) 

INFO:asr_server:🔥 Using cached google engine (0ms latency!)
INFO:asr_server:WS connected (google) Address(host='172.17.0.1', port=33972)
ERROR:grpc._channel:Exception iterating requests!
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/grpc/_channel.py", line 276, in consume_request_iterator
    request = next(request_iterator)
  File "/srv/app/asr_engines/google_streaming_asr.py", line 154, in _request_gen
    streaming_config = cloud_speech.StreamingRecognitionConfig(
  File "/usr/local/lib/python3.10/dist-packages/proto/message.py", line 673, in __init__
    raise ValueError(
ValueError: Unknown field for StreamingRecognitionConfig: interim_results

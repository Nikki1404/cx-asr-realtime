docker build --build-arg USE_PROXY=true --build-arg HTTP_PROXY=http://163.116.128.80:8080 --build-arg HTTPS_PROXY=http://163.116.128.80:8080 -t bu_digital_asr_realtime

(client_env) PS C:\Users\re_nikitav\Desktop\cx-asr-realtime\scripts> python .\ws_client.py google
🔗 Connected to ws://127.0.0.1:8002/asr/realtime-custom-vad
📤 Sent backend config: google
🎤 Recording started (Ctrl+C to stop)

[FINAL] Hello. This is ASR testing and I'm testing Google and I want to check if it's working fine or not.
[SERVER] reason=silence ttf_ms=7435 audio_ms=6180 rtf=1.4637283211902483 chunks=0

[FINAL] Google.
[SERVER] reason=silence ttf_ms=5000 audio_ms=4080 rtf=4.400537484816239 chunks=0

[FINAL] Hello, I want to test the name of Ron.
[SERVER] reason=silence ttf_ms=3955 audio_ms=3220 rtf=6.989845436649189 chunks=0
🛑 Recording stopped
Traceback (most recent call last):

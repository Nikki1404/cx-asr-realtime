import os
import time
import queue
import threading
from typing import Optional

from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from app.asr_engines.base import ASREngine, EngineCaps


class GoogleStreamingASR(ASREngine):

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.client = None
        self.recognizer = os.getenv("GOOGLE_RECOGNIZER")
        self.region = os.getenv("GOOGLE_REGION", "us-central1")

    def load(self) -> float:
        t0 = time.time()

        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.region}-speech.googleapis.com"
            )
        )

        return time.time() - t0

    def new_session(self, max_buffer_ms: int):
        return GoogleStreamingSession(self)


class GoogleStreamingSession:

    def __init__(self, engine: GoogleStreamingASR):
        self.engine = engine
        self.audio_queue = queue.Queue()
        self.closed = False
        self.latest_partial = ""
        self.final_text = ""

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._start_stream()

    def _start_stream(self):

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="latest_short",
        )

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )

        def request_gen():
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=self.engine.recognizer,
                streaming_config=streaming_config,
            )

            while not self.closed:
                chunk = self.audio_queue.get()
                if chunk is None:
                    return
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

        def run():
            responses = self.engine.client.streaming_recognize(requests=request_gen())
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        self.final_text += " " + transcript
                    else:
                        self.latest_partial = transcript

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def accept_pcm16(self, pcm16: bytes):
        self.audio_queue.put(pcm16)

    def step_if_ready(self) -> Optional[str]:
        if self.latest_partial:
            return self.latest_partial
        return None

    def finalize(self, pad_ms: int) -> str:
        self.audio_queue.put(None)
        self.closed = True
        time.sleep(0.2)
        return self.final_text.strip()

import os
import time
import queue
import threading
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from app.asr_engines.base import ASREngine, EngineCaps


class GoogleStreamingASR(ASREngine):
    """
    Google Cloud Speech-to-Text v2 Streaming Engine
    SDK-safe version (no interim flags)

    ENV required:
      GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
      GOOGLE_RECOGNIZER=projects/<PROJECT>/locations/<REGION>/recognizers/<ID>
      GOOGLE_REGION=us-central1
    """

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.client: Optional[speech_v2.SpeechClient] = None

        self.region = os.getenv("GOOGLE_REGION", "us-central1")
        self.recognizer = os.getenv("GOOGLE_RECOGNIZER", "").strip()

        self.model = os.getenv("GOOGLE_MODEL", "latest_short")
        self.language_code = os.getenv("GOOGLE_LANG", "en-US")

    @property
    def model_name(self) -> str:
        return f"google:{self.model}"

    def load(self) -> float:
        """
        Preload Google client (lightweight).
        """
        t0 = time.time()

        endpoint = f"{self.region}-speech.googleapis.com"
        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )

        if not self.recognizer:
            raise ValueError(
                "GOOGLE_RECOGNIZER env not set.\n"
                "Example:\n"
                "projects/<PROJECT_ID>/locations/<REGION>/recognizers/<ID>"
            )

        return time.time() - t0

    def new_session(self, max_buffer_ms: int):
        return GoogleStreamingSession(self)

# SESSION

class GoogleStreamingSession:
    def __init__(self, engine: GoogleStreamingASR):
        self.engine = engine

        if self.engine.client is None:
            raise RuntimeError("GoogleStreamingASR not loaded")

        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

        self._latest_partial: str = ""
        self._final_accum: list[str] = []

        # Metrics parity
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._thread = threading.Thread(
            target=self._run_streaming,
            daemon=True
        )
        self._thread.start()

    def accept_pcm16(self, pcm16: bytes) -> None:
        if not self._closed:
            self._audio_q.put(pcm16)
            self.chunks += 1

    def step_if_ready(self) -> Optional[str]:
        if self._latest_partial:
            return self._latest_partial
        return None

    def finalize(self, pad_ms: int) -> str:
        if not self._closed:
            self._closed = True
            self._audio_q.put(None)

        t0 = time.perf_counter()
        self._thread.join(timeout=2.0)
        self.utt_flush += (time.perf_counter() - t0)

        return " ".join(self._final_accum).strip()
    # STREAMING
    def _request_gen(self):
        """
        SDK-safe streaming config.
        No interim flags.
        """

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.engine.language_code],
            model=self.engine.model,
        )

        # IMPORTANT: no interim_results, no streaming_features
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=config
        )

        # First message = config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.engine.recognizer,
            streaming_config=streaming_config,
        )

        # Then audio chunks
        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                return

            yield cloud_speech.StreamingRecognizeRequest(
                audio=chunk
            )

    def _run_streaming(self):
        try:
            t0 = time.perf_counter()

            responses = self.engine.client.streaming_recognize(
                requests=self._request_gen()
            )

            for resp in responses:
                self.utt_infer += (time.perf_counter() - t0)
                t0 = time.perf_counter()

                for result in resp.results:
                    if not result.alternatives:
                        continue

                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue

                    if result.is_final:
                        self._final_accum.append(text)
                        self._latest_partial = ""
                    else:
                        self._latest_partial = text

        except Exception:
            # Never crash thread
            self._latest_partial = ""
            return

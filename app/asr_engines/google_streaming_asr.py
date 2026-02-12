import os
import time
import json
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Any

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from app.asr_engines.base import ASREngine, EngineCaps


@dataclass
class GoogleStreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class GoogleStreamingASR(ASREngine):
    """
    Google Cloud Speech-to-Text v2 streaming engine.

    ✅ True streaming: partials + final
    ✅ TTFT meaningful (network dependent)
    ✅ CPU-side; inference happens in Google Cloud

    ENV required:
      - GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
      - GOOGLE_RECOGNIZER=projects/<PROJECT_ID>/locations/<REGION>/recognizers/<RECOGNIZER_ID>
      - GOOGLE_REGION=us-central1 (or your region)
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

        # Tuning knobs (can be overridden via env)
        self.model = os.getenv("GOOGLE_MODEL", "latest_short")
        self.language_code = os.getenv("GOOGLE_LANG", "en-US")

        if not self.recognizer:
            # keep load() from crashing hard; server preload will log error
            pass

    @property
    def model_name(self) -> str:
        # for metrics labeling
        return f"google:{self.model}"

    def load(self) -> float:
        t0 = time.time()

        # Endpoint form: <region>-speech.googleapis.com
        endpoint = f"{self.region}-speech.googleapis.com"
        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )

        # sanity check recognizer env
        if not self.recognizer:
            raise ValueError(
                "GOOGLE_RECOGNIZER env not set. Example: "
                "projects/<PROJECT_ID>/locations/<REGION>/recognizers/<RECOGNIZER_ID>"
            )

        return time.time() - t0

    def new_session(self, max_buffer_ms: int):
        return GoogleStreamingSession(self)


class GoogleStreamingSession:
    """
    Per-websocket session.

    - accept_pcm16(): enqueue audio bytes
    - step_if_ready(): returns latest partial (if any)
    - finalize(): ends stream, returns accumulated final
    """

    def __init__(self, engine: GoogleStreamingASR):
        self.engine = engine
        if self.engine.client is None:
            raise RuntimeError("GoogleStreamingASR not loaded (client is None)")

        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

        self._latest_partial: str = ""
        self._final_accum: list[str] = []

        # timings for parity with metrics
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._thread = threading.Thread(target=self._run_streaming, daemon=True)
        self._thread.start()

    def accept_pcm16(self, pcm16: bytes) -> None:
        if self._closed:
            return
        self._audio_q.put(pcm16)
        self.chunks += 1

    def step_if_ready(self) -> Optional[str]:
        # return partial if present
        if self._latest_partial:
            return self._latest_partial
        return None

    def finalize(self, pad_ms: int) -> str:
        # end stream
        if not self._closed:
            self._closed = True
            self._audio_q.put(None)

        # small wait to allow thread to flush final results
        t0 = time.perf_counter()
        self._thread.join(timeout=1.5)
        self.utt_flush += (time.perf_counter() - t0)

        return " ".join([t for t in self._final_accum if t.strip()]).strip()

    # -------------------------
    # internal
    # -------------------------
    def _request_gen(self):
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.engine.language_code],
            model=self.engine.model,
        )

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                enable_partial_results=True
            ),
        )

        # FIRST message MUST contain config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.engine.recognizer,
            streaming_config=streaming_config,
        )

        # Then stream audio
        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                return

            yield cloud_speech.StreamingRecognizeRequest(
                audio=chunk
            )

    def _run_streaming(self):
        """
        Runs blocking streaming_recognize in a background thread.
        Updates _latest_partial and _final_accum.
        """
        try:
            t_infer0 = time.perf_counter()

            responses = self.engine.client.streaming_recognize(
                requests=self._request_gen()
            )

            for resp in responses:
                # inference time accounting is approximate (network + server)
                # but we keep it for parity.
                self.utt_infer += (time.perf_counter() - t_infer0)
                t_infer0 = time.perf_counter()

                for r in resp.results:
                    if not r.alternatives:
                        continue
                    txt = (r.alternatives[0].transcript or "").strip()
                    if not txt:
                        continue

                    if r.is_final:
                        self._final_accum.append(txt)
                        self._latest_partial = ""  # clear partial after final
                    else:
                        self._latest_partial = txt

        except Exception:
            # Do not crash server thread; treat as empty result
            self._latest_partial = ""
            return

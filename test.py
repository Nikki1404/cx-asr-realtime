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
    Google Cloud Speech-to-Text v2 streaming engine.

    True streaming:
    - partials supported
    - final supported
    - TTFT meaningful (network dependent)
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
        self.language_code = os.getenv("GOOGLE_LANG", "en-US")
        self.model = os.getenv("GOOGLE_MODEL", "latest_short")

    @property
    def model_name(self) -> str:
        return f"google:{self.model}"

    def load(self) -> float:
        t0 = time.time()

        if not self.recognizer:
            raise ValueError("GOOGLE_RECOGNIZER env variable not set")

        endpoint = f"{self.region}-speech.googleapis.com"
        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )

        return time.time() - t0

    def new_session(self, max_buffer_ms: int):
        return GoogleStreamingSession(self)


# =============================================================


class GoogleStreamingSession:
    """
    Per-websocket session.

    IMPORTANT:
    Your server calls step_if_ready() every VAD frame.
    So we must:
      - return partial ONLY when it changes (dedupe)
      - reset internal accumulators after finalize() (like Nemotron/Whisper)
    """

    def __init__(self, engine: GoogleStreamingASR):
        self.engine = engine
        if self.engine.client is None:
            raise RuntimeError("Google engine not loaded")

        # metrics parity with Nemotron
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        # state init
        self._reset_utterance_state()

    # -------------------------
    # session lifecycle
    # -------------------------

    def _reset_utterance_state(self):
        # audio queue + thread state
        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

        # transcript state
        self._latest_partial: str = ""
        self._last_partial_sent: str = ""   # DEDUPE KEY
        self._final_parts: list[str] = []

        # chunk counter should reset per utterance (matches Nemotron behavior)
        self.chunks = 0

        # restart worker thread
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()

    # -------------------------
    # ASRSession interface
    # -------------------------

    def accept_pcm16(self, pcm16: bytes):
        if self._closed:
            return
        if not pcm16:
            return
        self._audio_q.put(pcm16)
        self.chunks += 1

    def step_if_ready(self) -> Optional[str]:
        """
        Return a partial ONLY when it's NEW (prevents infinite repeat prints).
        """
        if not self._latest_partial:
            return None

        if self._latest_partial == self._last_partial_sent:
            return None

        self._last_partial_sent = self._latest_partial
        return self._latest_partial

    def finalize(self, pad_ms: int) -> str:
        """
        Close current Google stream, wait briefly for final results, return final text,
        then RESET for next utterance (critical fix).
        """
        # Optional pad to help last phonemes (similar to your other engines)
        if pad_ms and pad_ms > 0:
            pad_samples = int(self.engine.sr * (pad_ms / 1000.0))
            pad_bytes = b"\x00\x00" * pad_samples
            try:
                self._audio_q.put(pad_bytes)
            except Exception:
                pass

        if not self._closed:
            self._closed = True
            self._audio_q.put(None)

        t0 = time.perf_counter()
        self._thread.join(timeout=2.5)
        self.utt_flush += (time.perf_counter() - t0)

        final_text = " ".join(self._final_parts).strip()

        # ✅ CRITICAL: reset for next utterance so we don't repeat the same final forever
        self._reset_utterance_state()

        return final_text

    # -------------------------
    # internal: google request/response
    # -------------------------

    def _request_generator(self):
        """
        Uses ExplicitDecodingConfig for raw PCM16 streaming.
        """

        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.engine.sr,
                audio_channel_count=1,
            ),
            language_codes=[self.engine.language_code],
            model=self.engine.model,
        )

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
        )

        # First request must contain config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.engine.recognizer,
            streaming_config=streaming_config,
        )

        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                break

            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

    def _run_stream(self):
        try:
            t_infer0 = time.perf_counter()

            responses = self.engine.client.streaming_recognize(
                requests=self._request_generator()
            )

            for response in responses:
                self.utt_infer += (time.perf_counter() - t_infer0)
                t_infer0 = time.perf_counter()

                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = (result.alternatives[0].transcript or "").strip()
                    if not transcript:
                        continue

                    if result.is_final:
                        self._final_parts.append(transcript)
                        # once final arrives, clear partial so dedupe resets naturally
                        self._latest_partial = ""
                        self._last_partial_sent = ""
                    else:
                        self._latest_partial = transcript

        except Exception as e:
            print("Google streaming error:", e)
            self._latest_partial = ""
            self._last_partial_sent = ""
            return

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class EngineCaps:
    """
    Declares behavioral capabilities of an ASR engine.
    """
    streaming: bool          # true streaming model with incremental state
    partials: bool           # supports partial outputs during speech
    ttft_meaningful: bool    # whether TTFT is meaningful (streaming only)


class ASRSession(Protocol):
    """
    Session interface the server expects.

    Session may optionally expose timing fields:
      - utt_preproc
      - utt_infer
      - utt_flush
      - chunks
    """
    def accept_pcm16(self, pcm16: bytes) -> None: ...
    def step_if_ready(self) -> Optional[str]: ...
    def finalize(self, pad_ms: int) -> str: ...


class ASREngine(ABC):
    """
    Engine interface implemented by all ASR backends.
    """
    caps: EngineCaps

    @abstractmethod
    def load(self) -> float:
        """
        Load model weights and return load time in seconds.
        """
        ...

    @abstractmethod
    def new_session(self, max_buffer_ms: int) -> ASRSession:
        """
        Create a new per-utterance/session ASR state.
        """
        ...

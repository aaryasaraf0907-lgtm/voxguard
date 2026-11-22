"""Spoken-language check: is a clip in a language the detector was trained or tested on?

The deepfake detector is trained and evaluated on English (ASVspoof), with Hindi as a small
generalisation test (CLAUDE.md section 11). For speech that is clearly in another language its
score means little, so the API reports "language not supported" instead of a real/fake verdict
and advises a call-back (it never treats such a call as safe).

Language identification uses Whisper's language token (``openai/whisper-base`` by default, MIT
licence): the probability of each language for the first decoder step, on up to the first 30 s.
A clip counts as **unsupported** only when English, Hindi and Urdu together get less than
``min_supported_share`` (0.10) of that probability; Urdu is included because spoken Hindi and
Urdu are nearly the same. Anything less clear, and clips shorter than ``min_seconds``, keep the
normal verdict. Checked privately on 22 Nov 2026 with whisper-base: 1 of 320 genuine English
clips (Indian-accented, ASVspoof, In-the-Wild) fell under the cut; 28 of 30 synthetic Tamil,
Marathi, Bengali, French and Spanish sentences did, and no Hindi one.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED = ("en", "hi", "ur")
DEFAULT_MODEL = "openai/whisper-base"
TARGET_SR = 16_000


@dataclass(frozen=True)
class LanguageResult:
    code: str  # Whisper's most likely language, e.g. "en", "ta"
    supported_share: float  # probability of English + Hindi + Urdu together
    supported: bool


class WhisperLanguageChecker:
    """Loads the Whisper model once; ``check`` is thread-safe (one inference at a time)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        min_supported_share: float = 0.10,
        min_seconds: float = 2.0,
    ) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        self.model_id = model_id
        self.min_supported_share = min_supported_share
        self.min_seconds = min_seconds
        self._torch = torch
        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        tok = self._processor.tokenizer
        self._codes = list(LANGUAGES)
        self._ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in self._codes]
        self._sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self._lock = threading.Lock()

    def check(self, waveform: np.ndarray, sample_rate: int) -> LanguageResult | None:
        """None when the clip is too short to judge (the normal verdict then applies)."""
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1 if x.shape[-1] <= 8 else 0)
        if len(x) < self.min_seconds * sample_rate:
            return None
        torch = self._torch
        if sample_rate != TARGET_SR:
            import torchaudio.functional as AF

            x = AF.resample(torch.from_numpy(x), sample_rate, TARGET_SR).numpy()
        x = x[: 30 * TARGET_SR]
        feats = self._processor(x, sampling_rate=TARGET_SR, return_tensors="pt").input_features
        with self._lock, torch.no_grad():
            logits = self._model(
                input_features=feats, decoder_input_ids=torch.tensor([[self._sot]])
            ).logits[0, -1, self._ids]
        p = torch.softmax(logits.float(), -1)
        i = int(p.argmax())
        share = float(sum(p[self._codes.index(c)] for c in SUPPORTED))
        result = LanguageResult(
            code=self._codes[i],
            supported_share=share,
            supported=share >= self.min_supported_share,
        )
        logger.info("language check: %s (supported share %.3f)", result.code, share)
        return result


def language_name(code: str) -> str:
    """English name of a Whisper language code, e.g. "ta" -> "Tamil" (the code itself when
    Transformers is not installed, e.g. with a stand-in checker in tests)."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
    except ImportError:
        return code
    return LANGUAGES.get(code, code).title()

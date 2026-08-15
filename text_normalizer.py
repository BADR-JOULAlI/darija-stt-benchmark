"""Configurable text normalization for Darija ASR comparison.

Raw transcripts must always be kept next to normalized text.  The defaults are
conservative: they normalize Unicode and whitespace without rewriting letters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


ARABIC_DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: str = "NFKC"
    lowercase_latin: bool = True
    remove_punctuation: bool = False
    remove_arabic_diacritics: bool = False
    normalize_alef: bool = False
    normalize_ya: bool = False
    normalize_ta_marbuta: bool = False
    normalize_digits: bool = False


ARABIC_TO_ASCII_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_text(text: str | None, config: NormalizationConfig | None = None) -> str:
    """Return a comparable representation without destroying code-switching."""
    if not text:
        return ""
    cfg = config or NormalizationConfig()
    value = unicodedata.normalize(cfg.unicode_form, str(text))
    if cfg.lowercase_latin:
        value = value.lower()
    if cfg.remove_arabic_diacritics:
        value = ARABIC_DIACRITICS_RE.sub("", value)
    if cfg.normalize_alef:
        value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}))
    if cfg.normalize_ya:
        value = value.translate(str.maketrans({"ى": "ي", "ئ": "ي"}))
    if cfg.normalize_ta_marbuta:
        value = value.replace("ة", "ه")
    if cfg.normalize_digits:
        value = value.translate(ARABIC_TO_ASCII_DIGITS)
    if cfg.remove_punctuation:
        value = "".join(
            " " if unicodedata.category(character).startswith(("P", "S")) else character
            for character in value
        )
    return WHITESPACE_RE.sub(" ", value).strip()


def config_from_name(name: str) -> NormalizationConfig:
    """Named profiles make experiments reproducible."""
    profiles = {
        "conservative": NormalizationConfig(),
        "comparison": NormalizationConfig(
            remove_punctuation=True,
            remove_arabic_diacritics=True,
            normalize_alef=True,
            normalize_ya=True,
            normalize_digits=True,
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown normalization profile: {name}") from exc

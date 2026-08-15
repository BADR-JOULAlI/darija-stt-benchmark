"""Agreement metrics and an explicitly uncalibrated quality score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    """Return substitutions, deletions and insertions for Levenshtein distance."""
    n, m = len(reference), len(hypothesis)
    previous = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            candidates = [
                (previous[j - 1][0] + 1, previous[j - 1][1] + 1,
                 previous[j - 1][2], previous[j - 1][3]),
                (previous[j][0] + 1, previous[j][1],
                 previous[j][2] + 1, previous[j][3]),
                (current[j - 1][0] + 1, current[j - 1][1],
                 current[j - 1][2], current[j - 1][3] + 1),
            ]
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[m]
    return substitutions, deletions, insertions


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    s, d, i = edit_distance(reference, hypothesis)
    return (s + d + i) / len(reference)


def cer(reference: str, hypothesis: str) -> float:
    return error_rate(list(reference), list(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    return error_rate(reference.split(), hypothesis.split())


@dataclass(frozen=True)
class AgreementMetrics:
    cer_agreement: float
    wer_agreement: float
    character_similarity: float
    word_similarity: float
    mistral_characters: int
    ctc_characters: int
    mistral_words: int
    ctc_words: int
    length_ratio: float
    ctc_confidence: float | None
    blank_rate: float | None
    speech_ratio: float | None
    provisional_quality_score: float
    score_is_calibrated: bool = False


def compare_transcripts(
    mistral_text: str,
    ctc_text: str,
    *,
    ctc_confidence: float | None = None,
    blank_rate: float | None = None,
    speech_ratio: float | None = None,
) -> dict[str, float | int | bool | None]:
    char_error = cer(mistral_text, ctc_text)
    word_error = wer(mistral_text, ctc_text)
    char_similarity = max(0.0, 1.0 - min(char_error, 1.0))
    word_similarity = max(0.0, 1.0 - min(word_error, 1.0))
    longer = max(len(mistral_text), len(ctc_text), 1)
    shorter = min(len(mistral_text), len(ctc_text))
    length_ratio = shorter / longer

    # Agreement dominates. Optional signals are included only when available.
    weighted = [(char_similarity, 0.4), (word_similarity, 0.4), (length_ratio, 0.2)]
    if ctc_confidence is not None:
        weighted.append((max(0.0, min(ctc_confidence, 1.0)), 0.2))
    if speech_ratio is not None:
        weighted.append((max(0.0, min(speech_ratio, 1.0)), 0.1))
    total_weight = sum(weight for _, weight in weighted)
    provisional = (
        0.0 if not mistral_text and not ctc_text
        else sum(value * weight for value, weight in weighted) / total_weight
    )
    metrics = AgreementMetrics(
        cer_agreement=char_error,
        wer_agreement=word_error,
        character_similarity=char_similarity,
        word_similarity=word_similarity,
        mistral_characters=len(mistral_text),
        ctc_characters=len(ctc_text),
        mistral_words=len(mistral_text.split()),
        ctc_words=len(ctc_text.split()),
        length_ratio=length_ratio,
        ctc_confidence=ctc_confidence,
        blank_rate=blank_rate,
        speech_ratio=speech_ratio,
        provisional_quality_score=provisional,
    )
    return asdict(metrics)

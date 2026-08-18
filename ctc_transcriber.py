"""Resumable CTC inference and Mistral/CTC agreement scoring.

The default command is intentionally limited to three files.  It never calls
Mistral: it reuses successful JSON results already stored by the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from config import (
    ANNOTATION_CSV,
    AUDIO_DIR,
    CTC_RESULTS_DIR,
    QUALITY_RESULTS_DIR,
    RESULTS_DIR,
    ensure_directories,
)
from quality_scorer import compare_transcripts
from text_normalizer import config_from_name, normalize_text
from utils import read_json, write_json


DEFAULT_MODEL = "facebook/omniASR-CTC-300M"
MODEL_SPECS = {
    "facebook/omniASR-CTC-300M": ("omnilingual", "omniASR_CTC_300M_v2"),
    "facebook/omniASR-CTC-1B": ("omnilingual", "omniASR_CTC_1B_v2"),
    "facebook/omniASR-CTC-3B": ("omnilingual", "omniASR_CTC_3B_v2"),
    "facebook/omniASR-CTC-7B": ("omnilingual", "omniASR_CTC_7B_v2"),
    "boumehdi/wav2vec2-large-xlsr-moroccan-darija": ("transformers", None),
}
OMNILINGUAL_CHUNK_SECONDS = 30.0
TRANSFORMERS_CHUNK_SECONDS = 30.0
ANNOTATION_FIELDS = [
    "segment_id", "video_id", "audio_path", "mistral_transcript", "ctc_transcript",
    "cer_agreement", "wer_agreement", "character_similarity", "word_similarity",
    "length_ratio", "ctc_confidence", "provisional_quality_score", "human_label",
    "corrected_transcript", "reviewer_notes",
]


class CTCError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None


@dataclass(frozen=True)
class CTCOutput:
    text: str
    confidence: float | None = None
    blank_rate: float | None = None


class Backend(Protocol):
    def transcribe_batch(self, paths: list[Path], batch_size: int) -> list[CTCOutput]: ...


def detect_device() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(device="cpu", cuda_available=False, gpu_name=None)
    available = bool(torch.cuda.is_available())
    return DeviceInfo(
        device="cuda" if available else "cpu",
        cuda_available=available,
        gpu_name=torch.cuda.get_device_name(0) if available else None,
    )


def validate_cuda_build(device: DeviceInfo) -> None:
    """Reject PyTorch builds that cannot execute kernels on RTX 50-series GPUs."""
    if not device.cuda_available:
        return
    try:
        import torch
        capability = tuple(torch.cuda.get_device_capability(0))
        supported_arches = set(torch.cuda.get_arch_list())
    except (ImportError, RuntimeError, AttributeError):
        return
    if capability >= (12, 0) and "sm_120" not in supported_arches:
        raise CTCError(
            "The installed PyTorch build does not support RTX 50-series GPUs (sm_120). "
            "Install matching torch/torchaudio wheels built with CUDA 12.8 or newer."
        )


class OmnilingualBackend:
    def __init__(self, model: str, language: str, device: DeviceInfo):
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError as exc:
            raise CTCError(
                "omnilingual-asr is missing. Install requirements-ctc.txt on the GPU machine."
            ) from exc
        _, model_card = MODEL_SPECS[model]
        try:
            self.pipeline = ASRInferencePipeline(
                model_card=model_card,
                device=device.device,
            )
        except Exception as exc:
            raise CTCError(f"Could not load CTC model: {type(exc).__name__}") from exc
        self.language = language

    @staticmethod
    def _decode_chunks(path: Path) -> list[dict[str, Any]]:
        """Decode an audio file and split it below OmniASR's 40-second limit."""
        try:
            import soundfile
            waveform, sample_rate = soundfile.read(
                str(path), dtype="float32", always_2d=True
            )
        except (ImportError, OSError, RuntimeError) as exc:
            raise CTCError(
                "Could not decode audio for OmniASR. Install soundfile and ensure "
                "the bundled libsndfile can read MP3 files."
            ) from exc
        if sample_rate <= 0 or len(waveform) == 0:
            raise CTCError(f"Decoded audio is empty: {path.name}")
        mono = waveform.mean(axis=1)
        chunk_frames = max(1, int(sample_rate * OMNILINGUAL_CHUNK_SECONDS))
        return [
            {
                "waveform": mono[start:start + chunk_frames].copy(),
                "sample_rate": sample_rate,
            }
            for start in range(0, len(mono), chunk_frames)
        ]

    def transcribe_batch(self, paths: list[Path], batch_size: int) -> list[CTCOutput]:
        audio_chunks: list[dict[str, Any]] = []
        chunk_owners: list[int] = []
        for owner, path in enumerate(paths):
            chunks = self._decode_chunks(path)
            audio_chunks.extend(chunks)
            chunk_owners.extend([owner] * len(chunks))
        try:
            texts = self.pipeline.transcribe(
                audio_chunks,
                batch_size=max(1, batch_size),
            )
        except Exception as exc:
            raise CTCError(f"CTC inference failed: {type(exc).__name__}") from exc
        if len(texts) != len(audio_chunks):
            raise CTCError("OmniASR returned an unexpected number of audio chunks.")
        grouped: list[list[str]] = [[] for _ in paths]
        for owner, text in zip(chunk_owners, texts):
            cleaned = str(text or "").strip()
            if cleaned:
                grouped[owner].append(cleaned)
        return [CTCOutput(text=" ".join(parts)) for parts in grouped]


class TransformersCTCBackend:
    def __init__(self, model: str, language: str, device: DeviceInfo):
        del language
        try:
            import torch
            import torchaudio
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise CTCError(
                "torch, torchaudio and transformers are required for this CTC backend."
            ) from exc
        self.torch = torch
        self.torchaudio = torchaudio
        self.device = device.device
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            self.model = AutoModelForCTC.from_pretrained(model).to(self.device).eval()
        except Exception as exc:
            raise CTCError(f"Could not load Transformers CTC model: {type(exc).__name__}") from exc

    def _load_chunks(self, path: Path) -> list[Any]:
        try:
            import soundfile
            waveform, rate = soundfile.read(
                str(path), dtype="float32", always_2d=True
            )
        except (ImportError, OSError, RuntimeError) as exc:
            raise CTCError(f"Could not decode audio: {path.name}") from exc
        waveform = self.torch.from_numpy(waveform.mean(axis=1))
        if rate != 16_000:
            waveform = self.torchaudio.functional.resample(waveform, rate, 16_000)
        chunk_frames = int(16_000 * TRANSFORMERS_CHUNK_SECONDS)
        return [
            waveform[start:start + chunk_frames].numpy()
            for start in range(0, len(waveform), chunk_frames)
        ]

    def transcribe_batch(self, paths: list[Path], batch_size: int) -> list[CTCOutput]:
        try:
            chunks: list[Any] = []
            owners: list[int] = []
            for owner, path in enumerate(paths):
                path_chunks = self._load_chunks(path)
                chunks.extend(path_chunks)
                owners.extend([owner] * len(path_chunks))
            grouped_texts: list[list[str]] = [[] for _ in paths]
            grouped_confidence: list[list[float]] = [[] for _ in paths]
            grouped_blank_rate: list[list[float]] = [[] for _ in paths]
            blank_id = getattr(self.model.config, "pad_token_id", None)
            step = max(1, batch_size)
            for start in range(0, len(chunks), step):
                chunk_batch = chunks[start:start + step]
                inputs = self.processor(
                    chunk_batch, sampling_rate=16_000, return_tensors="pt", padding=True
                )
                input_values = inputs.input_values.to(self.device)
                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                with self.torch.inference_mode():
                    logits = self.model(input_values, attention_mask=attention_mask).logits
                probabilities = logits.softmax(dim=-1)
                max_probabilities, token_ids = probabilities.max(dim=-1)
                texts = self.processor.batch_decode(token_ids)
                for offset, text in enumerate(texts):
                    owner = owners[start + offset]
                    ids = token_ids[offset]
                    probs = max_probabilities[offset]
                    if blank_id is None:
                        confidence = float(probs.mean().item())
                    else:
                        non_blank = ids != blank_id
                        confidence = (
                            float(probs[non_blank].mean().item()) if non_blank.any() else 0.0
                        )
                        grouped_blank_rate[owner].append(
                            float((~non_blank).float().mean().item())
                        )
                    cleaned = str(text or "").strip()
                    if cleaned:
                        grouped_texts[owner].append(cleaned)
                    grouped_confidence[owner].append(confidence)
            outputs = []
            for owner in range(len(paths)):
                confidences = grouped_confidence[owner]
                blank_rates = grouped_blank_rate[owner]
                outputs.append(CTCOutput(
                    text=" ".join(grouped_texts[owner]),
                    confidence=sum(confidences) / len(confidences) if confidences else None,
                    blank_rate=sum(blank_rates) / len(blank_rates) if blank_rates else None,
                ))
            return outputs
        except CTCError:
            raise
        except Exception as exc:
            raise CTCError(f"CTC inference failed: {type(exc).__name__}") from exc


def create_backend(model: str, language: str, device: DeviceInfo) -> Backend:
    if model not in MODEL_SPECS:
        raise CTCError(f"Unsupported model: {model}")
    backend, _ = MODEL_SPECS[model]
    if backend == "omnilingual":
        return OmnilingualBackend(model, language, device)
    return TransformersCTCBackend(model, language, device)


def discover_successful_samples() -> list[dict[str, Any]]:
    samples = []
    for result_path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "success":
            continue
        video_id = str(result.get("video_id") or result_path.stem)
        samples.append({
            "segment_id": video_id,
            "video_id": video_id,
            "audio_path": AUDIO_DIR / f"{video_id}.mp3",
            "mistral_transcript": str(result.get("transcription") or ""),
            "mistral_model": result.get("model"),
        })
    return samples


def gpu_memory_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except ImportError:
        pass
    return None


def _write_annotation_template(rows: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if ANNOTATION_CSV.exists():
        with ANNOTATION_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {row["segment_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        return
    merged = []
    seen = set()
    for row in rows:
        old = existing.get(str(row["segment_id"]), {})
        seen.add(str(row["segment_id"]))
        merged.append({
            **{field: row.get(field, "") for field in ANNOTATION_FIELDS},
            "human_label": old.get("human_label", ""),
            "corrected_transcript": old.get("corrected_transcript", ""),
            "reviewer_notes": old.get("reviewer_notes", ""),
        })
    for segment_id, old in existing.items():
        if segment_id not in seen:
            merged.append({field: old.get(field, "") for field in ANNOTATION_FIELDS})
    with ANNOTATION_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(merged)


def prepare_annotation_template() -> int:
    """Create one blank human-review row per successful Mistral sample."""
    rows = []
    for sample in discover_successful_samples():
        quality_path = QUALITY_RESULTS_DIR / f"{sample['segment_id']}.json"
        quality = read_json(quality_path) if quality_path.exists() else {}
        rows.append({
            "segment_id": sample["segment_id"],
            "video_id": sample["video_id"],
            "audio_path": str(sample["audio_path"]),
            "mistral_transcript": sample["mistral_transcript"],
            "ctc_transcript": quality.get("ctc_transcript", ""),
            "cer_agreement": quality.get("cer_agreement", ""),
            "wer_agreement": quality.get("wer_agreement", ""),
            "character_similarity": quality.get("character_similarity", ""),
            "word_similarity": quality.get("word_similarity", ""),
            "length_ratio": quality.get("length_ratio", ""),
            "ctc_confidence": quality.get("ctc_confidence", ""),
            "provisional_quality_score": quality.get("provisional_quality_score", ""),
        })
    _write_annotation_template(rows)
    return len(rows)


def run_ctc_pipeline(
    *,
    model: str = DEFAULT_MODEL,
    language: str = "ary_Arab",
    limit: int = 3,
    batch_size: int = 1,
    force: bool = False,
    allow_cpu: bool = False,
    backend: Backend | None = None,
) -> list[dict[str, Any]]:
    ensure_directories()
    device = detect_device()
    if not device.cuda_available and not allow_cpu and backend is None:
        raise CTCError("CUDA GPU not detected. Re-run with --allow-cpu only for a deliberate small test.")
    if backend is None:
        validate_cuda_build(device)
    samples = discover_successful_samples()
    if limit > 0:
        samples = samples[:limit]
    pending = []
    completed: list[dict[str, Any]] = []
    for sample in samples:
        path = CTC_RESULTS_DIR / f"{sample['segment_id']}.json"
        if path.exists() and not force:
            previous = read_json(path)
            if previous.get("status") == "success" and previous.get("model") == model:
                completed.append(previous)
                continue
        if not sample["audio_path"].is_file():
            result = {
                **sample, "audio_path": str(sample["audio_path"]), "model": model,
                "status": "error", "error": f"Audio file not found: {sample['audio_path']}",
            }
            write_json(path, result)
            completed.append(result)
            continue
        pending.append(sample)

    if pending and backend is None:
        backend = create_backend(model, language, device)
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start:start + max(1, batch_size)]
        started = time.perf_counter()
        try:
            outputs = backend.transcribe_batch([item["audio_path"] for item in batch], batch_size)
            if len(outputs) != len(batch):
                raise CTCError("The backend returned an unexpected number of transcriptions.")
            elapsed = time.perf_counter() - started
            for item, output in zip(batch, outputs):
                normalized_mistral = normalize_text(item["mistral_transcript"], config_from_name("comparison"))
                normalized_ctc = normalize_text(output.text, config_from_name("comparison"))
                quality = compare_transcripts(
                    normalized_mistral,
                    normalized_ctc,
                    ctc_confidence=output.confidence,
                    blank_rate=output.blank_rate,
                )
                result = {
                    **item,
                    "audio_path": str(item["audio_path"]),
                    "model": model,
                    "language": language,
                    "device": asdict(device),
                    "raw_transcript": output.text,
                    "normalized_transcript": normalized_ctc,
                    "ctc_confidence": output.confidence,
                    "blank_rate": output.blank_rate,
                    "speech_ratio": None,
                    "processing_time_seconds": elapsed / len(batch),
                    "gpu_memory_mb": gpu_memory_mb(),
                    "status": "success",
                    "error": "",
                }
                write_json(CTC_RESULTS_DIR / f"{item['segment_id']}.json", result)
                quality_result = {
                    "segment_id": item["segment_id"],
                    "video_id": item["video_id"],
                    "mistral_model": item["mistral_model"],
                    "ctc_model": model,
                    "mistral_transcript": item["mistral_transcript"],
                    "ctc_transcript": output.text,
                    "normalized_mistral_transcript": normalized_mistral,
                    "normalized_ctc_transcript": normalized_ctc,
                    **quality,
                    "methodology_note": "Agreement only; not accuracy against human ground truth.",
                }
                write_json(QUALITY_RESULTS_DIR / f"{item['segment_id']}.json", quality_result)
                completed.append(result)
        except Exception as exc:
            safe_error = str(exc) if isinstance(exc, CTCError) else type(exc).__name__
            for item in batch:
                result = {
                    **item, "audio_path": str(item["audio_path"]), "model": model,
                    "language": language, "device": asdict(device), "status": "error",
                    "processing_time_seconds": time.perf_counter() - started,
                    "error": safe_error,
                }
                write_json(CTC_RESULTS_DIR / f"{item['segment_id']}.json", result)
                completed.append(result)

    prepare_annotation_template()
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small, resumable CTC quality pilot.")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_SPECS))
    parser.add_argument("--language", default="ary_Arab")
    parser.add_argument("--limit", type=int, default=3, help="Default safety limit: 3; use 0 for all.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-annotations", action="store_true")
    args = parser.parse_args()
    device = detect_device()
    samples = discover_successful_samples()
    print(f"Model: {args.model}")
    print(f"Device: {device.device}; GPU: {device.gpu_name or 'not detected'}")
    print(f"Successful Mistral samples available: {len(samples)}")
    if args.prepare_annotations:
        print(f"Annotation rows prepared: {prepare_annotation_template()}")
    if args.dry_run:
        return 0
    try:
        results = run_ctc_pipeline(
            model=args.model, language=args.language, limit=args.limit,
            batch_size=args.batch_size, force=args.force, allow_cpu=args.allow_cpu,
        )
    except CTCError as exc:
        print(f"CTC pilot not started: {exc}")
        return 2
    successful = sum(result.get("status") == "success" for result in results)
    print(f"CTC results: {successful} success / {len(results)} considered")
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())

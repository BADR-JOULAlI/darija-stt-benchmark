from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"
RESULTS_DIR = OUTPUTS_DIR / "results"
SELECTED_VIDEOS_CSV = PROJECT_ROOT / "selected_videos.csv"
SUMMARY_CSV = OUTPUTS_DIR / "summary.csv"
SUMMARY_JSON = OUTPUTS_DIR / "summary.json"
CTC_RESULTS_DIR = OUTPUTS_DIR / "ctc_results"
QUALITY_RESULTS_DIR = OUTPUTS_DIR / "quality_results"
ANNOTATION_CSV = OUTPUTS_DIR / "ctc_human_annotations.csv"
CTC_AUDIT_JSON = OUTPUTS_DIR / "ctc_readiness_audit.json"

SAMPLE_DURATION_SECONDS = 120.0
DEFAULT_SAMPLE_COUNT = 20


def ensure_directories() -> None:
    for directory in (
        AUDIO_DIR, OUTPUTS_DIR, TRANSCRIPTS_DIR, RESULTS_DIR,
        CTC_RESULTS_DIR, QUALITY_RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

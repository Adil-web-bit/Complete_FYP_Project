from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    exports_dir: Path
    public_data_dir: Path
    artifacts_dir: Path
    models_dir: Path


def get_project_paths() -> ProjectPaths:
    """
    Return canonical paths for the project.

    This file lives at: <root>/src/config.py
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    exports_dir = data_dir / "exports"
    public_data_dir = data_dir / "public"
    artifacts_dir = root_dir / "artifacts"
    models_dir = root_dir / "models"
    return ProjectPaths(
        root_dir=root_dir,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        exports_dir=exports_dir,
        public_data_dir=public_data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
    )


# Convenience paths (requested for downstream modules)
_PATHS = get_project_paths()
RAW_DATA_DIR: Path = _PATHS.raw_data_dir
PROCESSED_DATA_DIR: Path = _PATHS.processed_data_dir
EXPORT_DIR: Path = _PATHS.exports_dir
PUBLIC_DATA_DIR: Path = _PATHS.public_data_dir
PUBLIC_DATASET_FILE: Path = PUBLIC_DATA_DIR / "public_keystroke_dataset.csv"
TYPING_EVENTS_FILE: Path = RAW_DATA_DIR / "typing_events.jsonl"
SCROLLING_EVENTS_FILE: Path = RAW_DATA_DIR / "scrolling_events.jsonl"
SESSIONS_FILE: Path = RAW_DATA_DIR / "monitoring_sessions.jsonl"
MODEL_DIR: Path = _PATHS.models_dir
MODEL_FILE: Path = MODEL_DIR / "behavior_model.joblib"
TRAINING_REPORT_FILE: Path = MODEL_DIR / "training_report.json"
EVALUATION_REPORT_FILE: Path = MODEL_DIR / "evaluation_report.json"
PUBLIC_TRAINING_REPORT_FILE: Path = MODEL_DIR / "public_training_report.json"
PUBLIC_EVALUATION_REPORT_FILE: Path = MODEL_DIR / "public_evaluation_report.json"


def ensure_project_directories() -> ProjectPaths:
    """
    Ensure expected directories exist.

    Raises:
        OSError: If a directory cannot be created.
    """
    paths = get_project_paths()
    for directory in (
        paths.data_dir,
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.exports_dir,
        paths.public_data_dir,
        paths.artifacts_dir,
        paths.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths

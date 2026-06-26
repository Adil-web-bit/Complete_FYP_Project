from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .fusion_features import FUSION_FEATURE_COLUMNS, TARGET_COLUMN, build_feature_row
from .session_store import SESSION_DIR


DATASET_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET_FILE = DATASET_DIR / "fusion_training_dataset.csv"


@dataclass(slots=True)
class FusionDatasetBuilder:
    session_dir: Path = SESSION_DIR
    dataset_file: Path = DEFAULT_DATASET_FILE

    def load_sessions(self) -> list[dict[str, Any]]:
        if not self.session_dir.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.session_dir.glob("*.json")):
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    parsed = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                sessions.append(parsed)
        return sessions

    def build_dataframe(self, sessions: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        source_sessions = self.load_sessions() if sessions is None else sessions
        rows = [build_feature_row(session) for session in source_sessions if isinstance(session, dict)]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["session_id", "timestamp", *FUSION_FEATURE_COLUMNS, TARGET_COLUMN, "target_source"])

        for column in FUSION_FEATURE_COLUMNS:
            if column not in df.columns:
                df[column] = 0.0
        if TARGET_COLUMN not in df.columns:
            df[TARGET_COLUMN] = None

        ordered = ["session_id", "timestamp", *FUSION_FEATURE_COLUMNS, TARGET_COLUMN, "target_source"]
        extras = [column for column in df.columns if column not in ordered]
        return df[ordered + extras]

    def build_training_dataframe(self, sessions: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        df = self.build_dataframe(sessions)
        if df.empty:
            return df
        return df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    def save_dataset(self, df: pd.DataFrame) -> Path:
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.dataset_file, index=False)
        return self.dataset_file

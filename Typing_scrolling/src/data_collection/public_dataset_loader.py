from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize_column_name(name: str) -> str:
    """
    Normalize column names for case-insensitive matching.

    - lowercase
    - strip whitespace
    - remove spaces/underscores/hyphens
    """
    compact = str(name).strip().lower()
    for ch in (" ", "_", "-"):
        compact = compact.replace(ch, "")
    return compact


_LABEL_CANDIDATES_NORMALIZED: tuple[str, ...] = (
    "emotionindex",
    "emotion",
    "label",
    "mood",
    "target",
    "class",
    "behaviorlabel",
)

_DROP_COLUMNS_NORMALIZED: set[str] = {
    # Identity/session/user columns
    "userid",
    "username",
    "participant",
    "participantid",
    "subject",
    "subjectid",
    "sessionid",
    # Text/content columns
    "text",
    "typedtext",
    "rawtext",
    "sentence",
    "content",
    "password",
    "answer",
    # Common index columns (keep timing features like D1U1 etc.)
    "index",
    "textindex",
}

_LABEL_MAP: dict[str, str] = {
    # EmoSurv short codes
    "h": "focused",
    "s": "distracted",
    "a": "stressed",
    "c": "focused",
    "n": "normal",
    "anger": "stressed",
    "angry": "stressed",
    "neutral": "normal",
    "calmness": "focused",
    "calm": "focused",
    "happiness": "focused",
    "happy": "focused",
    "sadness": "distracted",
    "sad": "distracted",
    "stress": "stressed",
    "stressed": "stressed",
    "rushed": "rushed",
    "distracted": "distracted",
    "focused": "focused",
    "normal": "normal",
}

_ALLOWED_APP_LABELS: tuple[str, ...] = ("normal", "focused", "stressed", "rushed", "distracted")


@dataclass(slots=True)
class PublicDatasetLoader:
    """
    Load and normalize a typing/keystroke dataset (CSV) into a training-ready dataframe.
    """

    def coerce_numeric_features(
        self, df: pd.DataFrame, exclude_columns: list[str]
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Coerce numeric-looking object/string columns to numeric before feature selection.

        Returns:
            (coerced_df, numeric_coerced_columns, non_numeric_rejected_columns)
        """
        out = df.copy()
        exclude_set = {str(c) for c in exclude_columns}
        coerced_columns: list[str] = []
        rejected_columns: list[str] = []

        null_markers = {"", "na", "nan", "null", "none"}

        for col in list(out.columns):
            col_name = str(col)
            if col_name in exclude_set:
                continue

            series = out[col]
            if is_numeric_dtype(series):
                continue

            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
            ):
                continue

            try:
                s = series.astype("string").str.strip()
            except Exception:
                continue

            # Normalize common empty markers to NA
            try:
                lower = s.str.lower()
                s = s.mask(lower.isin(null_markers), pd.NA)
            except Exception:
                continue

            # Replace decimal commas safely (e.g. "20,5" -> "20.5").
            # Keep this conservative to avoid converting thousands separators like "1,234".
            try:
                has_comma = s.str.contains(",", regex=False, na=False)
                has_dot = s.str.contains(".", regex=False, na=False)
                dec_comma = has_comma & (~has_dot) & s.str.match(r"^-?\d+,\d{1,2}$", na=False)
                if bool(dec_comma.any()):
                    s = s.where(~dec_comma, s.str.replace(",", ".", regex=False))
            except Exception:
                pass

            try:
                numeric = pd.to_numeric(s, errors="coerce")
            except Exception:
                continue

            non_empty = s.notna() & (s != "")
            denom = int(non_empty.sum())
            if denom <= 0:
                rejected_columns.append(col_name)
                continue

            non_null_numeric = int(numeric.notna().sum())
            ratio = float(non_null_numeric) / float(denom)
            if ratio >= 0.7 and non_null_numeric > 0:
                out[col] = numeric.astype(float)
                coerced_columns.append(col_name)
            else:
                rejected_columns.append(col_name)

        return out, coerced_columns, rejected_columns

    def load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        # EmoSurv (and some CSV exports) may use non-comma delimiters or BOM headers.
        # Strategy:
        # 1) Try auto-detected delimiter first: sep=None, engine="python"
        # 2) If the best result still has 1 column, try explicit delimiters.
        encodings = ("utf-8", "utf-8-sig", "latin1")
        explicit_seps: tuple[str, ...] = (",", ";", "\t", "|")

        best_df: pd.DataFrame | None = None
        best_cols = -1

        def consider(candidate: pd.DataFrame) -> None:
            nonlocal best_df, best_cols
            cols = int(candidate.shape[1])
            if cols > best_cols:
                best_df = candidate
                best_cols = cols

        # 1) Auto-detect delimiter
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            except Exception:
                continue
            consider(df)

        # 2) Explicit delimiters if needed
        if best_cols <= 1:
            for enc in encodings:
                for sep in explicit_seps:
                    try:
                        df = pd.read_csv(path, encoding=enc, sep=sep)
                    except Exception:
                        continue
                    consider(df)

        if best_df is None:
            raise ValueError("Failed to parse CSV. Check delimiter and encoding.")
        if int(best_df.shape[1]) <= 1:
            raise ValueError(
                "CSV was read as one column. Check delimiter or file format."
            )
        return best_df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def detect_label_column(self, df: pd.DataFrame) -> str:
        normalized_to_original: dict[str, str] = {}
        for col in df.columns:
            norm = normalize_column_name(str(col))
            # Keep the first observed original name for a normalized key.
            normalized_to_original.setdefault(norm, str(col))

        for candidate_norm in _LABEL_CANDIDATES_NORMALIZED:
            if candidate_norm in normalized_to_original:
                return normalized_to_original[candidate_norm]
        raise ValueError("No label/emotion column found in dataset.")

    def is_emosurv_like_dataset(self, df: pd.DataFrame) -> bool:
        """
        Return True if the dataset looks like EmoSurv (has EmotionIndex-style label column).
        """
        return "emotionindex" in {normalize_column_name(str(c)) for c in df.columns}

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        label_col = self.detect_label_column(out)

        mapped: list[str | None] = []
        for raw in out[label_col].tolist():
            key = str(raw).strip().lower()
            mapped_label = _LABEL_MAP.get(key, key)
            mapped.append(mapped_label if mapped_label in _ALLOWED_APP_LABELS else None)
        out["behavior_label"] = mapped
        out = out.dropna(subset=["behavior_label"]).reset_index(drop=True)
        return out

    def build_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = self.normalize_columns(df)
        mapped = self.map_labels(normalized)

        label_col = self.detect_label_column(mapped)
        candidates = mapped.drop(columns=[label_col], errors="ignore")

        dropped_identity_text_columns = [
            c for c in candidates.columns if normalize_column_name(str(c)) in _DROP_COLUMNS_NORMALIZED
        ]
        candidates = candidates.drop(columns=dropped_identity_text_columns, errors="ignore")

        # Some public datasets store numeric timing features as strings (e.g. "20,5" or " 4.2 ").
        # Coerce numeric-looking columns to numeric before selection.
        coerced, _, _ = self.coerce_numeric_features(
            candidates,
            exclude_columns=[],
        )

        numeric = coerced.select_dtypes(include=["number"]).copy()
        if numeric.shape[1] == 0:
            raise ValueError("No numeric feature columns found in dataset.")

        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        nunique = numeric.nunique(dropna=False)
        constant_cols = [c for c in numeric.columns if int(nunique.get(c, 0)) <= 1]
        if constant_cols:
            numeric = numeric.drop(columns=constant_cols)

        if numeric.shape[1] == 0:
            raise ValueError("No numeric feature columns found in dataset (all were constant).")

        numeric["behavior_label"] = mapped["behavior_label"].astype(str).values
        return numeric

    def preview_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Preview dataset compatibility and the final training dataframe schema.

        This is a non-destructive diagnostic helper for UI debugging before training.
        """
        original = self.normalize_columns(df)
        original_rows = int(original.shape[0])
        original_columns = int(original.shape[1])

        preview: dict[str, Any] = {
            "status": "error",
            "message": "",
            "original_rows": original_rows,
            "original_columns": original_columns,
            "detected_label_column": None,
            "original_column_names": [str(c) for c in original.columns],
            "original_column_names_head5": [str(c) for c in list(original.columns)[:5]],
            "original_label_counts": {},
            "mapped_label_counts": {},
            "unmapped_or_dropped_label_rows": 0,
            "numeric_feature_columns": [],
            "numeric_coerced_columns": [],
            "non_numeric_rejected_columns": [],
            "dropped_identity_text_columns": [],
            "constant_columns_removed": [],
            "final_rows": 0,
            "final_feature_count": 0,
            "final_feature_columns": [],
            "preview_records": [],
        }

        if original_columns <= 1:
            preview["message"] = (
                "CSV appears to have been read as a single column. The delimiter may be semicolon, tab, or another "
                "character. Try re-saving as comma-delimited CSV or use auto-detected loader."
            )
            return preview

        try:
            label_col = self.detect_label_column(original)
            preview["detected_label_column"] = label_col
        except ValueError as exc:
            preview["message"] = str(exc)
            return preview

        try:
            vc = original[label_col].astype(str).value_counts(dropna=False)
            preview["original_label_counts"] = {str(k): int(v) for k, v in vc.to_dict().items()}
        except Exception:
            preview["original_label_counts"] = {}

        # Map labels using the same logic as training.
        mapped_series: list[str | None] = []
        try:
            for raw in original[label_col].tolist():
                key = str(raw).strip().lower()
                mapped_label = _LABEL_MAP.get(key, key)
                mapped_series.append(mapped_label if mapped_label in _ALLOWED_APP_LABELS else None)
        except Exception as exc:
            preview["message"] = f"Failed to map labels: {exc}"
            return preview

        mapped_df = original.copy()
        mapped_df["behavior_label"] = mapped_series
        mapped_counts = (
            mapped_df["behavior_label"].dropna().astype(str).value_counts(dropna=False).to_dict()
        )
        preview["mapped_label_counts"] = {str(k): int(v) for k, v in mapped_counts.items()}
        preview["unmapped_or_dropped_label_rows"] = int(
            mapped_df["behavior_label"].isna().sum()
        )

        mapped_df = mapped_df.dropna(subset=["behavior_label"]).reset_index(drop=True)
        preview["final_rows"] = int(mapped_df.shape[0])

        # Drop identity/text columns and select numeric features.
        candidates = mapped_df.drop(columns=[label_col], errors="ignore")
        dropped_identity_text_columns = sorted(
            [c for c in candidates.columns if normalize_column_name(str(c)) in _DROP_COLUMNS_NORMALIZED]
        )
        preview["dropped_identity_text_columns"] = dropped_identity_text_columns
        candidates = candidates.drop(columns=dropped_identity_text_columns, errors="ignore")

        coerced, coerced_cols, rejected_cols = self.coerce_numeric_features(
            candidates,
            exclude_columns=[],
        )
        preview["numeric_coerced_columns"] = coerced_cols
        preview["non_numeric_rejected_columns"] = rejected_cols

        numeric = coerced.select_dtypes(include=["number"]).copy()
        if numeric.shape[1] == 0:
            preview["message"] = "No numeric feature columns found in dataset."
            return preview

        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nunique = numeric.nunique(dropna=False)
        constant_cols = sorted([c for c in numeric.columns if int(nunique.get(c, 0)) <= 1])
        preview["constant_columns_removed"] = constant_cols
        if constant_cols:
            numeric = numeric.drop(columns=constant_cols)

        if numeric.shape[1] == 0:
            preview["message"] = "No numeric feature columns found in dataset (all were constant)."
            return preview

        numeric_feature_columns = list(numeric.columns.astype(str))
        preview["numeric_feature_columns"] = numeric_feature_columns
        preview["final_feature_columns"] = numeric_feature_columns
        preview["final_feature_count"] = int(len(numeric_feature_columns))

        final_df = numeric.copy()
        final_df["behavior_label"] = mapped_df["behavior_label"].astype(str).values
        preview["preview_records"] = final_df.head(5).to_dict(orient="records")

        preview["status"] = "ok"
        preview["message"] = "Dataset looks compatible after label mapping and numeric feature selection."
        return preview

    def generate_synthetic_typing_demo_dataset(
        self, n_rows: int = 250, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic typing-only data for pipeline testing.

        This dataset is NOT proof of real-world mood prediction accuracy. It is only used to validate that the
        feature+training pipeline can run end-to-end.
        """
        rng = np.random.default_rng(int(random_state))
        labels = np.array(["normal", "focused", "stressed", "rushed", "distracted"], dtype=object)
        y = rng.choice(labels, size=int(n_rows), replace=True)

        def noise(scale: float) -> np.ndarray:
            return rng.normal(0.0, scale, size=int(n_rows))

        typed_text_length = np.clip(40 + noise(15), 0, None)
        typing_duration_seconds = np.clip(20 + noise(6), 0.5, None)

        keys_per_second = np.clip((typed_text_length / typing_duration_seconds) + noise(0.4), 0.1, None)
        total_key_events = np.clip((keys_per_second * typing_duration_seconds) + noise(5), 1, None)
        unique_keys_count = np.clip(18 + noise(6), 1, None)
        backspace_count = np.clip(2 + noise(2), 0, None)
        avg_key_interval_ms = np.clip((1000.0 / keys_per_second) + noise(30), 50, None)
        std_key_interval_ms = np.clip(70 + noise(20), 0, None)

        # Make label-specific shifts so the classifier can learn something.
        for i, label in enumerate(y):
            if label == "focused":
                backspace_count[i] *= 0.5
                std_key_interval_ms[i] *= 0.7
            elif label == "stressed":
                backspace_count[i] += rng.integers(8, 18)
                avg_key_interval_ms[i] *= 0.85
            elif label == "rushed":
                keys_per_second[i] *= 1.6
                avg_key_interval_ms[i] *= 0.7
                std_key_interval_ms[i] *= 1.2
            elif label == "distracted":
                std_key_interval_ms[i] *= 1.8
                keys_per_second[i] *= 0.75

        df = pd.DataFrame(
            {
                "typed_text_length": typed_text_length.astype(float),
                "total_key_events": total_key_events.astype(float),
                "backspace_count": backspace_count.astype(float),
                "unique_keys_count": unique_keys_count.astype(float),
                "typing_duration_seconds": typing_duration_seconds.astype(float),
                "keys_per_second": keys_per_second.astype(float),
                "avg_key_interval_ms": avg_key_interval_ms.astype(float),
                "std_key_interval_ms": std_key_interval_ms.astype(float),
                "behavior_label": y.astype(str),
            }
        )
        return df

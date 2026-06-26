from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.data_collection.public_dataset_loader import PublicDatasetLoader  # noqa: E402


def test_label_mapping_emosurv_style() -> None:
    df = pd.DataFrame(
        {
            "Emotion": ["Neutral", "Calmness", "Anger", "Happiness", "Sadness", "Unknown"],
            "feat1": [1, 2, 3, 4, 5, 6],
        }
    )
    loader = PublicDatasetLoader()
    out = loader.build_feature_dataframe(df)
    assert "behavior_label" in out.columns
    assert set(out["behavior_label"].unique()) <= {"normal", "focused", "stressed", "distracted"}


def test_build_feature_dataframe_keeps_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "emotion": ["neutral", "anger", "sadness"],
            "user_id": ["u1", "u2", "u3"],
            "typed_text": ["a", "b", "c"],
            "speed": [1.0, 2.0, 3.0],
            "count": [10, 20, 30],
        }
    )
    loader = PublicDatasetLoader()
    out = loader.build_feature_dataframe(df)
    assert "behavior_label" in out.columns
    assert "speed" in out.columns
    assert "count" in out.columns
    assert "user_id" not in out.columns
    assert "typed_text" not in out.columns


def test_missing_label_column_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    loader = PublicDatasetLoader()
    with pytest.raises(ValueError):
        loader.build_feature_dataframe(df)


def test_synthetic_typing_demo_dataset_shape_and_labels() -> None:
    loader = PublicDatasetLoader()
    df = loader.generate_synthetic_typing_demo_dataset(n_rows=120, random_state=42)
    assert df.shape[0] == 120
    assert "behavior_label" in df.columns
    assert set(df["behavior_label"].unique()) <= {"normal", "focused", "stressed", "rushed", "distracted"}


def test_preview_schema_ok_for_valid_emosurv_style_df() -> None:
    df = pd.DataFrame(
        {
            "Emotion": ["Neutral", "Anger", "Sadness"],
            "speed": [1.0, 2.0, 3.0],
            "count": [10, 20, 30],
        }
    )
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["detected_label_column"] in ("Emotion", "emotion")
    assert preview["final_feature_count"] >= 2
    assert len(preview["preview_records"]) <= 5


def test_emosurv_emotionindex_column_detected_and_short_codes_mapped() -> None:
    df = pd.DataFrame(
        {
            "EmotionIndex": ["H", "S", "A", "C", "N", "X"],
            "D1U1": [100, 120, 140, 90, 110, 999],
            "TotTime": [10.0, 11.0, 12.0, 9.5, 10.2, 20.0],
            "DelFreq": [1, 2, 8, 0, 1, 5],
            "User ID": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "Answer": ["a", "b", "c", "d", "e", "f"],
            "Index": [1, 2, 3, 4, 5, 6],
        }
    )
    loader = PublicDatasetLoader()
    feature_df = loader.build_feature_dataframe(df)
    assert "behavior_label" in feature_df.columns
    assert "D1U1" in feature_df.columns
    assert "TotTime" in feature_df.columns
    assert "DelFreq" in feature_df.columns
    assert "User ID" not in feature_df.columns
    assert "Answer" not in feature_df.columns
    assert "Index" not in feature_df.columns
    assert set(feature_df["behavior_label"].unique()) <= {"focused", "stressed", "distracted", "normal"}


def test_preview_schema_for_emosurv_short_codes_reports_drops() -> None:
    df = pd.DataFrame(
        {
            "Emotion Index": ["H", "S", "A", "C", "N", "X"],
            "D1U1": [100, 120, 140, 90, 110, 999],
            "TotTime": [10.0, 11.0, 12.0, 9.5, 10.2, 20.0],
            "const": [1, 1, 1, 1, 1, 1],
            "UserID": ["u1", "u2", "u3", "u4", "u5", "u6"],
        }
    )
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["detected_label_column"] == "Emotion Index"
    assert preview["unmapped_or_dropped_label_rows"] == 1
    assert "const" in preview["constant_columns_removed"]
    assert "userid" in [c.lower().replace(" ", "") for c in preview["dropped_identity_text_columns"]]


def test_load_csv_handles_semicolon_delimited(tmp_path: Path) -> None:
    csv_path = tmp_path / "semi.csv"
    csv_path.write_text("EmotionIndex;D1U1;TotTime\nH;100;10.5\nN;110;9.9\n", encoding="utf-8")
    loader = PublicDatasetLoader()
    df = loader.load_csv(csv_path)
    assert df.shape[1] >= 3
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["detected_label_column"] in ("EmotionIndex", "Emotion Index", "emotion_index", "emotion index")


def test_load_csv_handles_tab_delimited(tmp_path: Path) -> None:
    csv_path = tmp_path / "tab.tsv"
    csv_path.write_text("Emotion Index\tD1U1\tTotTime\nA\t100\t10.5\nN\t110\t9.9\n", encoding="utf-8")
    loader = PublicDatasetLoader()
    df = loader.load_csv(csv_path)
    assert df.shape[1] >= 3


def test_load_csv_picks_delimiter_with_multiple_columns(tmp_path: Path) -> None:
    # A semicolon file should not be treated as a single-column CSV.
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("EmotionIndex;X;Y\nH;1;2\nS;3;4\n", encoding="utf-8")
    loader = PublicDatasetLoader()
    df = loader.load_csv(csv_path)
    assert df.shape[1] == 3


def test_preview_schema_error_when_missing_label_column() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "error"


def test_preview_schema_reports_unmapped_labels() -> None:
    df = pd.DataFrame({"emotion": ["neutral", "unknown", "anger"], "x": [1, 2, 3]})
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["unmapped_or_dropped_label_rows"] == 1


def test_preview_schema_reports_dropped_identity_and_constant_columns() -> None:
    df = pd.DataFrame(
        {
            "emotion": ["neutral", "anger", "sadness"],
            "user_id": ["u1", "u2", "u3"],
            "typed_text": ["a", "b", "c"],
            "const": [1, 1, 1],
            "speed": [1.0, 2.0, 3.0],
        }
    )
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert "user_id" in [c.lower() for c in preview["dropped_identity_text_columns"]]
    assert "typed_text" in [c.lower() for c in preview["dropped_identity_text_columns"]]
    assert "const" in preview["constant_columns_removed"]


def test_emosurv_frequency_camelcase_columns_detected_and_preview_ok() -> None:
    df = pd.DataFrame(
        {
            "User ID": ["u1", "u2", "u3", "u4", "u5"],
            "textIndex": [1, 2, 3, 4, 5],
            "emotionIndex": ["H", "S", "A", "C", "N"],
            "delFreq": [1, 2, 3, 4, 5],
            "leftFreq": [10, 9, 8, 7, 6],
            "totTime": [0.5, 0.7, 0.9, 1.1, 1.3],
        }
    )
    loader = PublicDatasetLoader()
    feature_df = loader.build_feature_dataframe(df)
    assert "behavior_label" in feature_df.columns
    assert "delFreq" in feature_df.columns
    assert "leftFreq" in feature_df.columns
    assert "totTime" in feature_df.columns

    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["detected_label_column"] == "emotionIndex"
    assert set((preview["mapped_label_counts"] or {}).keys()) <= {"focused", "stressed", "distracted", "normal"}


def test_numeric_looking_string_timing_columns_are_coerced_and_kept() -> None:
    # Fixed/Free text datasets may store timing features as strings with commas/spaces.
    df = pd.DataFrame(
        {
            "EmotionIndex": ["H", "S", "A", "C", "N"],
            "D1U1": ["12.5", "13.2", "14.0", "15.1", "16.3"],
            "D1U2": ["20,5", "21,1", "22,0", "23,2", "24,9"],
            "U1D2": [" 4.2 ", " 5.1", "6.0 ", " 7.3 ", "8.8"],
            "key Down": ["100", "110", "120", "130", "140"],
            "key Up": ["150", "160", "170", "180", "190"],
            "Key Code": ["65", "66", "67", "68", "69"],
            "Answer": ["a", "b", "c", "d", "e"],  # should be dropped (text)
        }
    )
    loader = PublicDatasetLoader()
    out = loader.build_feature_dataframe(df)
    assert "behavior_label" in out.columns
    for col in ("D1U1", "D1U2", "U1D2", "key Down", "key Up", "Key Code"):
        assert col in out.columns
    assert "Answer" not in out.columns


def test_non_numeric_text_columns_are_rejected_as_features() -> None:
    df = pd.DataFrame(
        {
            "EmotionIndex": ["H", "S", "A", "C", "N"],
            "D1D2": [1, 2, 3, 4, 5],
            "notes": ["fast", "slow", "ok", "??", "none"],  # should not be coerced to numeric
        }
    )
    loader = PublicDatasetLoader()
    out = loader.build_feature_dataframe(df)
    assert "D1D2" in out.columns
    assert "notes" not in out.columns


def test_preview_schema_includes_coerced_timing_features() -> None:
    df = pd.DataFrame(
        {
            "EmotionIndex": ["H", "S", "A", "C", "N"],
            "D1U1": ["12.5", "13.2", "14.0", "15.1", "16.3"],
            "D1U2": ["20,5", "21,1", "22,0", "23,2", "24,9"],
            "U1D2": [" 4.2 ", " 5.1", "6.0 ", " 7.3 ", "8.8"],
            "User ID": ["u1", "u2", "u3", "u4", "u5"],  # should be dropped
        }
    )
    loader = PublicDatasetLoader()
    preview = loader.preview_schema(df)
    assert preview["status"] == "ok"
    assert preview["detected_label_column"] == "EmotionIndex"
    assert "D1U1" in (preview.get("final_feature_columns") or [])
    assert "D1U2" in (preview.get("final_feature_columns") or [])
    assert "U1D2" in (preview.get("final_feature_columns") or [])

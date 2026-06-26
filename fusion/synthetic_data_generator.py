from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset_builder import DATASET_DIR
from .fusion_features import (
    BEHAVIOR_LABELS,
    BEHAVIOR_NUMERIC_FEATURES,
    FUSION_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from .label_mapper import STANDARD_EMOTION_LABELS


ALLOWED_SYNTHETIC_SIZES: tuple[int, ...] = (100, 500, 1000, 5000, 10000)
SYNTHETIC_DATASET_FILE = DATASET_DIR / "synthetic_fusion_dataset.csv"


_TARGET_BEHAVIOR_PRIORS: dict[str, dict[str, float]] = {
    "happy": {"focused": 0.75, "normal": 0.15, "rushed": 0.05, "distracted": 0.03, "stressed": 0.02},
    "sad": {"distracted": 0.70, "normal": 0.12, "stressed": 0.10, "focused": 0.05, "rushed": 0.03},
    "angry": {"stressed": 0.78, "rushed": 0.10, "distracted": 0.06, "normal": 0.04, "focused": 0.02},
    "neutral": {"normal": 0.78, "focused": 0.10, "distracted": 0.06, "rushed": 0.03, "stressed": 0.03},
    "fear": {"stressed": 0.50, "rushed": 0.25, "distracted": 0.12, "normal": 0.08, "focused": 0.05},
    "disgust": {"stressed": 0.48, "distracted": 0.22, "normal": 0.15, "rushed": 0.10, "focused": 0.05},
    "surprise": {"rushed": 0.40, "focused": 0.25, "normal": 0.20, "stressed": 0.10, "distracted": 0.05},
}


_CLASS_PROBS = np.array([0.18, 0.08, 0.10, 0.22, 0.22, 0.12, 0.08], dtype=float)
_CLASS_PROBS = _CLASS_PROBS / _CLASS_PROBS.sum()


@dataclass(slots=True)
class SyntheticFusionDataGenerator:
    output_file: Path = SYNTHETIC_DATASET_FILE
    random_state: int = 42

    def generate_dataframe(self, n_samples: int = 1000) -> pd.DataFrame:
        if int(n_samples) not in ALLOWED_SYNTHETIC_SIZES:
            raise ValueError(f"n_samples must be one of {ALLOWED_SYNTHETIC_SIZES}.")

        rng = np.random.default_rng(int(self.random_state))
        targets = rng.choice(STANDARD_EMOTION_LABELS, size=int(n_samples), p=_CLASS_PROBS)
        rows: list[dict[str, Any]] = []

        for idx, target in enumerate(targets):
            target_label = str(target)
            face_probs = self._correlated_probabilities(rng, target_label, strength=float(rng.uniform(18, 40)))
            voice_target = target_label
            if rng.random() < 0.12:
                voice_target = self._nearby_label(rng, target_label)
            voice_probs = self._correlated_probabilities(rng, voice_target, strength=float(rng.uniform(14, 35)))
            behavior_label = self._sample_behavior_label(rng, target_label)
            behavior_values = self._behavior_metrics(rng, behavior_label)

            row: dict[str, Any] = {
                "session_id": f"synthetic_{idx:06d}",
                "timestamp": "",
                TARGET_COLUMN: target_label,
                "target_source": "synthetic",
                "data_source": "synthetic",
                "sample_weight": 1.0,
            }

            for label in STANDARD_EMOTION_LABELS:
                row[f"face_{label}_prob"] = float(face_probs[label])
                row[f"voice_{label}_prob"] = float(voice_probs[label])

            row["behavior_confidence"] = float(rng.uniform(0.58, 0.96))
            for feature_name in BEHAVIOR_NUMERIC_FEATURES:
                row[f"behavior_{feature_name}"] = float(behavior_values.get(feature_name, 0.0))

            for label in BEHAVIOR_LABELS:
                row[f"behavior_label_{label}"] = 1.0 if behavior_label == label else 0.0

            rows.append(row)

        df = pd.DataFrame(rows)
        for column in FUSION_FEATURE_COLUMNS:
            if column not in df.columns:
                df[column] = 0.0
        ordered = ["session_id", "timestamp", *FUSION_FEATURE_COLUMNS, TARGET_COLUMN, "target_source", "data_source", "sample_weight"]
        return df[ordered]

    def save(self, n_samples: int = 1000) -> Path:
        df = self.generate_dataframe(n_samples)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)
        return self.output_file

    @staticmethod
    def _correlated_probabilities(rng: np.random.Generator, target: str, strength: float) -> dict[str, float]:
        alpha = np.ones(len(STANDARD_EMOTION_LABELS), dtype=float) * 0.8
        target_idx = STANDARD_EMOTION_LABELS.index(target)
        alpha[target_idx] = strength

        # Add small mass to semantically common confusions.
        if target == "happy":
            alpha[STANDARD_EMOTION_LABELS.index("neutral")] += 2.5
            alpha[STANDARD_EMOTION_LABELS.index("surprise")] += 1.5
        elif target == "sad":
            alpha[STANDARD_EMOTION_LABELS.index("neutral")] += 2.0
            alpha[STANDARD_EMOTION_LABELS.index("fear")] += 1.5
        elif target == "angry":
            alpha[STANDARD_EMOTION_LABELS.index("fear")] += 2.0
            alpha[STANDARD_EMOTION_LABELS.index("disgust")] += 1.5
        elif target == "neutral":
            alpha[STANDARD_EMOTION_LABELS.index("happy")] += 1.5
            alpha[STANDARD_EMOTION_LABELS.index("sad")] += 1.0
        elif target == "fear":
            alpha[STANDARD_EMOTION_LABELS.index("surprise")] += 1.8
            alpha[STANDARD_EMOTION_LABELS.index("sad")] += 1.2
        elif target == "surprise":
            alpha[STANDARD_EMOTION_LABELS.index("happy")] += 1.8
            alpha[STANDARD_EMOTION_LABELS.index("fear")] += 1.5

        probs = rng.dirichlet(alpha)
        probs = probs / probs.sum()
        return {label: float(value) for label, value in zip(STANDARD_EMOTION_LABELS, probs)}

    @staticmethod
    def _nearby_label(rng: np.random.Generator, target: str) -> str:
        confusions = {
            "happy": ["neutral", "surprise"],
            "sad": ["neutral", "fear"],
            "angry": ["fear", "disgust"],
            "neutral": ["happy", "sad"],
            "fear": ["surprise", "sad", "angry"],
            "disgust": ["angry", "sad"],
            "surprise": ["happy", "fear"],
        }
        return str(rng.choice(confusions.get(target, STANDARD_EMOTION_LABELS)))

    @staticmethod
    def _sample_behavior_label(rng: np.random.Generator, target: str) -> str:
        prior = _TARGET_BEHAVIOR_PRIORS.get(target, _TARGET_BEHAVIOR_PRIORS["neutral"])
        labels = list(prior.keys())
        probs = np.array([prior[label] for label in labels], dtype=float)
        probs = probs / probs.sum()
        return str(rng.choice(labels, p=probs))

    @staticmethod
    def _behavior_metrics(rng: np.random.Generator, behavior_label: str) -> dict[str, float]:
        if behavior_label == "focused":
            keys_per_second = rng.normal(3.1, 0.45)
            backspaces = rng.normal(2.0, 1.2)
            scroll_speed = rng.normal(420, 160)
            scroll_events = rng.normal(8, 4)
            interval = rng.normal(320, 60)
            std_interval = rng.normal(55, 18)
        elif behavior_label == "stressed":
            keys_per_second = rng.normal(4.0, 0.75)
            backspaces = rng.normal(13, 4)
            scroll_speed = rng.normal(950, 280)
            scroll_events = rng.normal(18, 7)
            interval = rng.normal(245, 55)
            std_interval = rng.normal(130, 35)
        elif behavior_label == "rushed":
            keys_per_second = rng.normal(5.1, 0.85)
            backspaces = rng.normal(5, 3)
            scroll_speed = rng.normal(1350, 360)
            scroll_events = rng.normal(22, 8)
            interval = rng.normal(190, 45)
            std_interval = rng.normal(105, 32)
        elif behavior_label == "distracted":
            keys_per_second = rng.normal(1.6, 0.45)
            backspaces = rng.normal(6, 3)
            scroll_speed = rng.normal(1200, 340)
            scroll_events = rng.normal(30, 10)
            interval = rng.normal(620, 160)
            std_interval = rng.normal(210, 70)
        else:
            keys_per_second = rng.normal(2.4, 0.45)
            backspaces = rng.normal(4, 2)
            scroll_speed = rng.normal(650, 220)
            scroll_events = rng.normal(12, 5)
            interval = rng.normal(430, 90)
            std_interval = rng.normal(95, 35)

        typed_len = max(8.0, float(rng.normal(95, 35)))
        duration = max(5.0, typed_len / max(0.2, keys_per_second))
        key_events = max(1.0, typed_len + max(0.0, backspaces) * 2.0)
        scroll_duration = max(0.0, float(rng.normal(duration * 0.35, duration * 0.12)))
        total_scroll_distance = max(0.0, scroll_speed * scroll_duration)

        d1u1 = max(25.0, float(rng.normal(90, 28)))
        d1d2 = max(30.0, interval)
        u1d2 = max(0.0, d1d2 - d1u1)
        u1u2 = max(30.0, d1d2 + float(rng.normal(0, 20)))

        return {
            "typed_text_length": typed_len,
            "total_key_events": key_events,
            "backspace_count": max(0.0, backspaces),
            "unique_keys_count": max(3.0, float(rng.normal(24, 6))),
            "typing_duration_seconds": duration,
            "keys_per_second": max(0.1, keys_per_second),
            "avg_key_interval_ms": max(40.0, interval),
            "std_key_interval_ms": max(0.0, std_interval),
            "total_scroll_events": max(0.0, scroll_events),
            "scroll_duration_seconds": scroll_duration,
            "total_scroll_distance": total_scroll_distance,
            "avg_scroll_speed": max(0.0, scroll_speed),
            "max_scroll_speed": max(0.0, scroll_speed * float(rng.uniform(1.1, 1.9))),
            "scroll_direction_changes": max(0.0, float(rng.normal(2.0 if behavior_label != "distracted" else 6.0, 2.0))),
            "total_duration_seconds": duration + scroll_duration,
            "D1U1": d1u1,
            "D1U2": max(d1u1, d1d2 + d1u1 + float(rng.normal(0, 25))),
            "D1D2": d1d2,
            "U1D2": u1d2,
            "U1U2": u1u2,
            "D1U3": max(d1d2 * 2.0, d1d2 * 2.0 + d1u1 + float(rng.normal(0, 40))),
            "D1D3": max(d1d2, d1d2 * 2.0 + float(rng.normal(0, 35))),
        }


def main() -> None:
    generator = SyntheticFusionDataGenerator()
    path = generator.save(1000)
    print(f"Saved synthetic fusion dataset: {path}")


if __name__ == "__main__":
    main()

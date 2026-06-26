from __future__ import annotations

import os
import pickle
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .confidence_calibrator import (
    apply_confidence_floor,
    clamp_confidence,
    normalize_probabilities,
)
from .label_mapper import (
    FACE_LABELS,
    STANDARD_EMOTION_LABELS,
    behavior_to_emotion_scores,
    map_probabilities,
    normalize_behavior_label,
    normalize_face_label,
    normalize_voice_label,
    sentiment_bucket_for_behavior,
)
from .prediction_schema import empty_modality_prediction, modality_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL_PRIMARY = PROJECT_ROOT / "Face gesture" / "emotion_detection_model.keras"
FACE_MODEL_FALLBACK = PROJECT_ROOT / "Face gesture" / "best_emotion_model.h5"
VOICE_MODEL_FILE = PROJECT_ROOT / "Voice detector" / "models" / "best_model.h5"
VOICE_ENCODER_FILE = PROJECT_ROOT / "Voice detector" / "models" / "label_encoder.pkl"
TYPING_MODULE_DIR = PROJECT_ROOT / "Typing_scrolling"


@lru_cache(maxsize=1)
def _load_face_model():
    import tensorflow as tf

    model_path = FACE_MODEL_PRIMARY if FACE_MODEL_PRIMARY.exists() else FACE_MODEL_FALLBACK
    return tf.keras.models.load_model(model_path, compile=False)


@lru_cache(maxsize=1)
def _load_face_detector():
    from mtcnn import MTCNN

    return MTCNN()


@lru_cache(maxsize=1)
def _load_voice_model_and_encoder():
    import tensorflow as tf
    from tensorflow.keras.layers import Dense as OriginalDense

    class CustomDense(OriginalDense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        VOICE_MODEL_FILE,
        custom_objects={"Dense": CustomDense},
        compile=False,
    )
    with VOICE_ENCODER_FILE.open("rb") as file:
        encoder = pickle.load(file)
    return model, encoder


def _as_pil_image(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _detect_and_crop_face(image: Image.Image, confidence_threshold: float = 0.85):
    import cv2
    from PIL import Image

    rgb_image = np.array(image.convert("RGB"))
    detector = _load_face_detector()
    faces = detector.detect_faces(rgb_image)
    if not faces:
        return None, 0.0

    faces.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    best_face = faces[0]
    confidence = float(best_face.get("confidence", 0.0))
    if confidence < confidence_threshold:
        return None, confidence

    x, y, w, h = best_face["box"]
    padding_w = int(w * 0.2)
    padding_h = int(h * 0.2)
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(rgb_image.shape[1], x + w + padding_w)
    y2 = min(rgb_image.shape[0], y + h + padding_h)
    face_crop = rgb_image[y1:y2, x1:x2]
    if face_crop.size <= 0:
        return None, confidence
    return Image.fromarray(face_crop), confidence


def _preprocess_face(face_image: Any) -> np.ndarray:
    face_gray = face_image.convert("L")
    face_resized = face_gray.resize((48, 48))
    face_array = np.array(face_resized, dtype=np.float32) / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    face_array = np.expand_dims(face_array, axis=-1)
    return face_array


def predict_face(image: Any, confidence_threshold: float = 0.85) -> dict[str, Any]:
    try:
        pil_image = _as_pil_image(image)
        face_crop, detection_confidence = _detect_and_crop_face(
            pil_image,
            confidence_threshold=confidence_threshold,
        )
        if face_crop is None:
            return empty_modality_prediction(
                "face",
                f"No face detected above threshold ({detection_confidence:.2f}).",
            )

        model = _load_face_model()
        predictions = model.predict(_preprocess_face(face_crop), verbose=0)[0]
        raw_probabilities = {
            label: float(prob) for label, prob in zip(FACE_LABELS, predictions)
        }
        probabilities = normalize_probabilities(
            map_probabilities(raw_probabilities, normalize_face_label),
            STANDARD_EMOTION_LABELS,
        )
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]

        return modality_prediction(
            modality="face",
            label=label,
            raw_label=label,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "model_path": str(FACE_MODEL_PRIMARY if FACE_MODEL_PRIMARY.exists() else FACE_MODEL_FALLBACK),
                "input_shape": [48, 48, 1],
                "face_detection_confidence": detection_confidence,
            },
        )
    except Exception as exc:
        return empty_modality_prediction("face", f"Face prediction failed: {exc}")


def _extract_audio_features(file_path: str | os.PathLike[str]) -> np.ndarray | None:
    import librosa

    try:
        audio, sample_rate = librosa.load(
            file_path,
            res_type="soxr_hq",
            duration=2.5,
            sr=22050,
            offset=0.5,
        )
        result = np.array([], dtype=np.float32)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        return result.astype(np.float32)
    except Exception:
        return None


def predict_voice(audio: Any) -> dict[str, Any]:
    temp_path: str | None = None
    try:
        if isinstance(audio, (str, os.PathLike)):
            audio_path = str(audio)
        else:
            suffix = Path(getattr(audio, "name", "")).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                if hasattr(audio, "getbuffer"):
                    temp_audio.write(audio.getbuffer())
                else:
                    temp_audio.write(audio.read())
                temp_path = temp_audio.name
            audio_path = temp_path

        features = _extract_audio_features(audio_path)
        if features is None:
            return empty_modality_prediction("voice", "Audio feature extraction failed.")

        model, encoder = _load_voice_model_and_encoder()
        predictions = model.predict(np.array([features]), verbose=0)[0]
        raw_probabilities: dict[str, float] = {}
        for idx, prob in enumerate(predictions):
            raw_label = str(encoder.inverse_transform([idx])[0])
            raw_probabilities[raw_label] = float(prob)

        probabilities = normalize_probabilities(
            map_probabilities(raw_probabilities, normalize_voice_label),
            STANDARD_EMOTION_LABELS,
        )
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]

        return modality_prediction(
            modality="voice",
            label=label,
            raw_label=max(raw_probabilities, key=raw_probabilities.get),
            confidence=confidence,
            probabilities=probabilities,
            features={
                "feature_count": int(features.shape[0]),
                "mfcc": 40,
                "chroma": 12,
                "mel": 128,
            },
            metadata={
                "model_path": str(VOICE_MODEL_FILE),
                "encoder_path": str(VOICE_ENCODER_FILE),
                "sample_rate": 22050,
                "duration_seconds": 2.5,
            },
        )
    except Exception as exc:
        return empty_modality_prediction("voice", f"Voice prediction failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _ensure_typing_import_path() -> None:
    module_dir = str(TYPING_MODULE_DIR)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)


def predict_behavior(session_data: dict[str, Any]) -> dict[str, Any]:
    try:
        _ensure_typing_import_path()
        from src.model.predictor import BehaviorPredictor

        result = BehaviorPredictor().predict_from_session(session_data)
        behavior_label = normalize_behavior_label(result.get("predicted_label"))
        confidence = clamp_confidence(result.get("confidence", 0.5), default=0.5)
        emotion_scores = normalize_probabilities(
            behavior_to_emotion_scores(behavior_label),
            STANDARD_EMOTION_LABELS,
        )
        calibrated_scores = normalize_probabilities(
            apply_confidence_floor(emotion_scores, confidence),
            STANDARD_EMOTION_LABELS,
        )

        return modality_prediction(
            modality="behavior",
            label=behavior_label,
            raw_label=str(result.get("predicted_label", behavior_label)),
            confidence=confidence,
            probabilities=calibrated_scores,
            features=result.get("features") if isinstance(result.get("features"), dict) else {},
            metadata={
                "model_type": result.get("model_type"),
                "reason": result.get("reason"),
                "source": result.get("source"),
                "dataset_type": result.get("dataset_type"),
                "sentiment_bucket": sentiment_bucket_for_behavior(behavior_label),
            },
        )
    except Exception as exc:
        return empty_modality_prediction("behavior", f"Behavior prediction failed: {exc}")

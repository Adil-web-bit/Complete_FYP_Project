"""Multimodal fusion utilities for the Human Behaviour Predictor app."""

__all__ = [
    "FusionEngine",
    "build_unified_prediction",
    "fuse_predictions",
    "predict_behavior",
    "predict_face",
    "predict_voice",
]


def __getattr__(name):
    if name in {"FusionEngine", "fuse_predictions"}:
        from .fusion_engine import FusionEngine, fuse_predictions

        return {"FusionEngine": FusionEngine, "fuse_predictions": fuse_predictions}[name]
    if name == "build_unified_prediction":
        from .prediction_schema import build_unified_prediction

        return build_unified_prediction
    if name in {"predict_behavior", "predict_face", "predict_voice"}:
        from .inference_wrappers import predict_behavior, predict_face, predict_voice

        return {
            "predict_behavior": predict_behavior,
            "predict_face": predict_face,
            "predict_voice": predict_voice,
        }[name]
    raise AttributeError(name)

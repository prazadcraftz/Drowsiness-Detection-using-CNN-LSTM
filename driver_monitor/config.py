"""Validated configuration for the existing 5-feature, 16-observation checkpoint."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = ("EAR-R", "EAR-L", "MAR", "phi", "theta")
CLASS_MAPPING = MappingProxyType({0: "Normal", 1: "Yawning", 2: "Eye Close"})
TARGET_MODEL_FPS = 20.0
MAX_FACE_GAP_SECONDS = 0.5  # Configurable engineering default, not medical validation.
PREDICTION_HISTORY_SIZE = 5


@dataclass(frozen=True)
class RuntimeConfig:
    model_profile: str = "legacy"
    asset_root: Path = PROJECT_ROOT
    model_path: Path = Path("saved_model.pt")
    normalization_path: Path = Path("norm.npy")
    face_model_path: Path = Path("face_landmarker.task")
    class_mapping: Mapping[int, str] = field(default_factory=lambda: dict(CLASS_MAPPING))
    sequence_length: int = 16
    feature_count: int = 5
    device: str = "cpu"
    cnn_dropout_rate: float = 0.25
    lstm_dropout_rate: float = 0.25
    target_model_fps: float = TARGET_MODEL_FPS
    max_face_gap_seconds: float = MAX_FACE_GAP_SECONDS
    enable_temporal_sampling: bool = True
    smoothing_mode: str = "probability_average"
    prediction_history_size: int = PREDICTION_HISTORY_SIZE
    normal_enter_threshold: float = 0.60
    yawning_enter_threshold: float = 0.60
    yawning_exit_threshold: float = 0.45
    eye_close_enter_threshold: float = 0.70
    eye_close_exit_threshold: float = 0.55
    yawning_persistence_seconds: float = 0.30
    drowsy_persistence_seconds: float = 1.0
    alert_persistence_seconds: float = 1.5

    def __post_init__(self):
        from driver_monitor.inference.model_profile import get_profile
        profile = get_profile(self.model_profile)
        if profile.name == 'lstm32':
            if self.model_path == Path('saved_model.pt'):
                object.__setattr__(self, 'model_path', Path(profile.checkpoint))
            if self.normalization_path == Path('norm.npy'):
                object.__setattr__(self, 'normalization_path', Path(profile.normalizer))
            if self.sequence_length == 16:
                object.__setattr__(self, 'sequence_length', 32)
        from driver_monitor.inference.sampling import TemporalSampler, positive_seconds
        TemporalSampler(self.target_model_fps, enabled=self.enable_temporal_sampling)
        positive_seconds(self.max_face_gap_seconds, "max_face_gap_seconds")
        import math
        from numbers import Real
        if self.smoothing_mode not in ("probability_average", "majority_vote"):
            raise ValueError("smoothing_mode must be probability_average or majority_vote")
        if type(self.prediction_history_size) is not int or self.prediction_history_size < 1:
            raise ValueError("prediction_history_size must be an integer >= 1")
        for name in ("normal_enter_threshold", "yawning_enter_threshold", "yawning_exit_threshold",
                     "eye_close_enter_threshold", "eye_close_exit_threshold", "yawning_persistence_seconds",
                     "drowsy_persistence_seconds", "alert_persistence_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            if name.endswith("threshold") and value > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for signal in ("eye_close", "yawning"):
            if getattr(self, f"{signal}_exit_threshold") > getattr(self, f"{signal}_enter_threshold"):
                raise ValueError(f"{signal} exit threshold must be <= enter threshold")
        if type(self.feature_count) is not int or self.feature_count != 5:
            raise ValueError("The baseline checkpoint requires feature_count=5 (EAR-R, EAR-L, MAR, phi, theta)")
        if type(self.sequence_length) is not int or self.sequence_length != profile.sequence_length:
            raise ValueError(f"The {profile.name} checkpoint requires sequence_length={profile.sequence_length}")
        if self.device != "cpu":
            raise ValueError("Phase 3 compatibility inference requires device='cpu'")
        if (any(type(key) is not int for key in self.class_mapping)
                or dict(self.class_mapping) != dict(CLASS_MAPPING)):
            raise ValueError("class_mapping must be exactly {0: 'Normal', 1: 'Yawning', 2: 'Eye Close'}")
        # Frozen dataclass alone would not prevent mutation of a supplied dictionary.
        object.__setattr__(self, "class_mapping", MappingProxyType(dict(self.class_mapping)))
        root = Path(self.asset_root).resolve()
        object.__setattr__(self, "asset_root", root)
        for name in ("model_path", "normalization_path", "face_model_path"):
            path = Path(getattr(self, name))
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Required {name} asset does not exist or is not a file: {path}")
            object.__setattr__(self, name, path)
        for name in ("cnn_dropout_rate", "lstm_dropout_rate"):
            if getattr(self, name) != 0.25:
                raise ValueError(f"The frozen baseline requires {name}=0.25")

    @classmethod
    def from_ini(cls, path: str | Path = PROJECT_ROOT / "config.ini", *, model_profile: str = 'legacy') -> RuntimeConfig:
        path = Path(path)
        path = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Runtime configuration does not exist: {path}")
        parser = configparser.ConfigParser()
        try:
            with path.open(encoding="utf-8-sig") as handle:
                parser.read_file(handle)
            if parser.getint("General", "num_classes") != 3:
                raise ValueError("The baseline checkpoint requires num_classes=3")
            from driver_monitor.inference.model_profile import get_profile
            profile = get_profile(model_profile)
            return cls(model_profile=model_profile, asset_root=path.parent,
                       model_path=Path(parser["output"]["model_path"]) if model_profile == 'legacy' else PROJECT_ROOT / profile.checkpoint,
                       normalization_path=Path(parser["data"]["normalize_data_file"]) if model_profile == 'legacy' else PROJECT_ROOT / profile.normalizer,
                       face_model_path=Path(parser["data"]["mediapipe_model_path"]),
                       feature_count=parser.getint("General", "num_features"),
                       sequence_length=parser.getint("General", "window_size") if model_profile == 'legacy' else profile.sequence_length,
                       cnn_dropout_rate=parser.getfloat("model", "cnn_dropout_rate"),
                       lstm_dropout_rate=parser.getfloat("model", "lstm_dropout_rate"),
                       target_model_fps=parser.getfloat("runtime", "target_model_fps", fallback=TARGET_MODEL_FPS),
                       max_face_gap_seconds=parser.getfloat("runtime", "max_face_gap_seconds", fallback=MAX_FACE_GAP_SECONDS),
                       enable_temporal_sampling=parser.getboolean("runtime", "enable_temporal_sampling", fallback=True),
                       smoothing_mode=parser.get("policy", "smoothing_mode", fallback="probability_average"),
                       prediction_history_size=parser.getint("policy", "prediction_history_size", fallback=PREDICTION_HISTORY_SIZE),
                       **{name: parser.getfloat("policy", name, fallback=cls.__dataclass_fields__[name].default)
                          for name in ("normal_enter_threshold", "yawning_enter_threshold", "yawning_exit_threshold",
                                       "eye_close_enter_threshold", "eye_close_exit_threshold", "yawning_persistence_seconds",
                                       "drowsy_persistence_seconds", "alert_persistence_seconds")})
        except (configparser.Error, KeyError, ValueError) as error:
            raise ValueError(f"Invalid baseline runtime configuration {path}: {error}") from error

    def model_settings(self) -> dict:
        return {"num_features": self.feature_count, "num_classes": len(self.class_mapping),
                "cnn_dropout_rate": self.cnn_dropout_rate,
                "lstm_dropout_rate": self.lstm_dropout_rate}

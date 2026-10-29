"""
Model loading & inference layer.
Supports:
  - scikit-learn estimators (.joblib / .pkl) e.g. Random Forest (v1)
  - Keras models (.keras) e.g. MLP Neural Network (v2)
  - standard scaler (scaler.joblib) applied to raw features
"""
from __future__ import annotations

import logging
import pickle
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Tuple, Optional

import joblib
import numpy as np

logger = logging.getLogger("inference_loader")

FEATURE_ORDER: list[str] = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

ARTIFACT_EXTENSIONS: tuple[str, ...] = (".joblib", ".pkl", ".pickle", ".keras")
SHARED_SCALER_NAME = "scaler.joblib"


class ModelLoadError(RuntimeError):
    """Raised when a model artifact cannot be found or loaded."""


class ModelInferenceError(RuntimeError):
    """Raised when a loaded model fails to produce a prediction."""


class Predictor(Protocol):
    def predict_one(self, record: dict) -> Tuple[int, float]: ...


def _vectorize(record: dict) -> np.ndarray:
    try:
        row = [record[feat] for feat in FEATURE_ORDER]
    except KeyError as exc:
        raise ModelInferenceError(f"Missing expected feature: {exc}") from exc
    return np.array(row, dtype=float).reshape(1, -1)


def _apply_scaler(scaler: Any | None, x: np.ndarray, version: str) -> np.ndarray:
    if scaler is None:
        return x
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            return scaler.transform(x)
    except Exception as exc:
        raise ModelInferenceError(f"Scaler transform failed for model {version}: {exc}") from exc


def _proba_from_output(out: Any) -> np.ndarray:
    arr = np.asarray(out, dtype=float).reshape(-1)
    if arr.size == 1:
        p1 = float(arr[0])
        return np.array([1.0 - p1, p1])
    return arr


@dataclass
class SklearnPredictor:
    estimator: Any
    version: str
    scaler: Any | None = None

    def predict_one(self, record: dict) -> Tuple[int, float]:
        x = _apply_scaler(self.scaler, _vectorize(record), self.version)
        try:
            if hasattr(self.estimator, "predict_proba"):
                proba = self.estimator.predict_proba(x)[0]
                label = int(np.argmax(proba))
                confidence = float(proba[label])
            else:
                label = int(self.estimator.predict(x)[0])
                confidence = 1.0
        except Exception as exc:
            raise ModelInferenceError(f"Inference failed for model {self.version}: {exc}") from exc
        return label, confidence

    def predict_batch(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = _apply_scaler(self.scaler, X_raw, self.version)
        try:
            if hasattr(self.estimator, "predict_proba"):
                proba = self.estimator.predict_proba(x)
                labels = np.argmax(proba, axis=1)
                confidences = proba[np.arange(len(labels)), labels]
            else:
                labels = self.estimator.predict(x)
                confidences = np.ones(len(labels))
        except Exception as exc:
            raise ModelInferenceError(f"Batch inference failed for model {self.version}: {exc}") from exc
        return labels.astype(int), confidences.astype(float)


@dataclass
class KerasPredictor:
    model: Any
    version: str
    scaler: Any | None = None

    def predict_one(self, record: dict) -> Tuple[int, float]:
        x = _apply_scaler(self.scaler, _vectorize(record), self.version)
        try:
            out = self.model(x.astype("float32"), training=False)
            proba = _proba_from_output(out)
            label = int(np.argmax(proba))
            confidence = float(proba[label])
        except Exception as exc:
            raise ModelInferenceError(f"Inference failed for model {self.version}: {exc}") from exc
        return label, confidence

    def predict_batch(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = _apply_scaler(self.scaler, X_raw, self.version)
        try:
            out = self.model(x.astype("float32"), training=False).numpy()
            if out.shape[1] == 1:
                p1 = out.flatten()
                proba = np.column_stack([1.0 - p1, p1])
            else:
                proba = out
            labels = np.argmax(proba, axis=1)
            confidences = proba[np.arange(len(labels)), labels]
        except Exception as exc:
            raise ModelInferenceError(f"Batch inference failed for model {self.version}: {exc}") from exc
        return labels.astype(int), confidences.astype(float)


class ModelRegistry:
    """Thread-safe, cached registry of loaded model artifacts."""

    def __init__(self, models_dir: Optional[Path] = None):
        self._models_dir_override = models_dir
        self._cache: dict[str, Predictor] = {}
        self._lock = threading.Lock()
        self._active_version: str = "v1"

    @property
    def _models_dir(self) -> Path:
        if self._models_dir_override is not None:
            return self._models_dir_override
        # Check standard locations
        candidates = [Path("models"), Path("../models"), Path("/app/models")]
        for c in candidates:
            if c.exists() and any(c.glob("*")):
                return c
        return Path("models")

    def _artifact_path(self, version: str) -> Path:
        for ext in ARTIFACT_EXTENSIONS:
            candidate = self._models_dir / f"{version}{ext}"
            if candidate.exists():
                return candidate
        raise ModelLoadError(
            f"No model artifact found for version '{version}' in {self._models_dir}"
        )

    def _load_scaler(self, version: str) -> Any | None:
        for name in (f"{version}.scaler.joblib", SHARED_SCALER_NAME):
            path = self._models_dir / name
            if path.exists():
                try:
                    return joblib.load(path)
                except Exception as exc:
                    raise ModelLoadError(f"Failed to load scaler at {path}: {exc}") from exc
        return None

    def get(self, version: str) -> Predictor:
        with self._lock:
            if version in self._cache:
                return self._cache[version]

            path = self._artifact_path(version)
            scaler = self._load_scaler(version)
            try:
                if path.suffix == ".keras":
                    import keras
                    predictor: Predictor = KerasPredictor(
                        model=keras.saving.load_model(path, compile=False),
                        version=version,
                        scaler=scaler,
                    )
                else:
                    if path.suffix == ".joblib":
                        estimator = joblib.load(path)
                    else:
                        with open(path, "rb") as fh:
                            estimator = pickle.load(fh)
                    predictor = SklearnPredictor(estimator=estimator, version=version, scaler=scaler)
            except ModelLoadError:
                raise
            except Exception as exc:
                raise ModelLoadError(f"Failed to load model artifact at {path}: {exc}") from exc

            self._cache[version] = predictor
            return predictor

    @property
    def active_version(self) -> str:
        return self._active_version

    def set_active_version(self, version: str):
        # Validate that version exists
        self._artifact_path(version)
        self._active_version = version

    def list_available_versions(self) -> list[str]:
        versions = []
        if self._models_dir.exists():
            for f in self._models_dir.iterdir():
                if f.suffix in ARTIFACT_EXTENSIONS and f.stem != "scaler":
                    versions.append(f.stem)
        return sorted(list(set(versions)))


registry = ModelRegistry()

"""Scikit-learn model wrapper for serving predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SklearnModel(BaseModel):
    """Wrapper around a scikit-learn estimator loaded from a joblib file.

    If no model file exists at *model_path*, a small demo classifier is trained
    on-the-fly so the service can start without pre-trained artifacts.
    """

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self._model_path = model_path
        self._model: Any = None

    @property
    def name(self) -> str:
        return "sklearn_classifier"

    @property
    def version(self) -> str:
        return "1.0.0"

    def load(self) -> None:
        """Load a joblib model from disk or train a demo classifier."""

        def _loader() -> None:
            if self._model_path and Path(self._model_path).exists():
                self._model = joblib.load(self._model_path)
                logger.info("Loaded sklearn model from %s", self._model_path)
            else:
                logger.warning(
                    "No model file found at '%s' — training demo classifier.",
                    self._model_path,
                )
                self._train_demo()

        self._timed_load(_loader)

    def _train_demo(self) -> None:
        """Train a tiny Iris classifier as a demonstration model."""
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        X, y = load_iris(return_X_y=True)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        self._model = clf
        logger.info("Demo RandomForestClassifier trained on Iris dataset.")

    def predict(self, input_data: Any) -> dict[str, Any]:
        """Run prediction for a single sample.

        Args:
            input_data: dict with key ``"features"`` containing a list of floats.

        Returns:
            Dict with ``prediction`` (int) and ``probabilities`` (list[float]).
        """
        self._ensure_loaded()
        features = np.array(input_data["features"]).reshape(1, -1)
        prediction = int(self._model.predict(features)[0])
        probabilities = self._model.predict_proba(features)[0].tolist()
        return {
            "prediction": prediction,
            "probabilities": probabilities,
        }

    def predict_batch(self, input_data: list[Any]) -> list[dict[str, Any]]:
        """Run predictions for a batch of samples."""
        self._ensure_loaded()
        features = np.array([item["features"] for item in input_data])
        predictions = self._model.predict(features).tolist()
        probabilities = self._model.predict_proba(features).tolist()
        return [
            {"prediction": int(p), "probabilities": prob}
            for p, prob in zip(predictions, probabilities)
        ]

    def warmup(self) -> None:
        """Run a dummy prediction to warm up the model."""
        self._ensure_loaded()
        dummy = {"features": [5.1, 3.5, 1.4, 0.2]}
        self.predict(dummy)
        logger.info("Model '%s' warmup complete.", self.name)

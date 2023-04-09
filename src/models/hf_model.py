"""HuggingFace Transformers model wrapper for serving predictions."""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """Wrapper around a HuggingFace pipeline.

    Defaults to ``distilbert-base-uncased-finetuned-sst-2-english`` for
    sentiment analysis when no model name is provided.
    """

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self._model_name = (
            model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self._pipeline: Any = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return "1.0.0"

    def load(self) -> None:
        """Load the HuggingFace pipeline."""
        from transformers import pipeline as hf_pipeline

        def _loader() -> None:
            self._pipeline = hf_pipeline("sentiment-analysis", model=self._model_name)
            logger.info("Loaded HuggingFace model: %s", self._model_name)

        self._timed_load(_loader)

    def predict(self, input_data: Any) -> dict[str, Any]:
        """Run prediction on a single text input.

        Args:
            input_data: dict with key ``"text"`` containing a string.

        Returns:
            Dict with ``label`` and ``score``.
        """
        self._ensure_loaded()
        result = self._pipeline(input_data["text"])[0]
        return {"label": result["label"], "score": float(result["score"])}

    def predict_batch(self, input_data: list[Any]) -> list[dict[str, Any]]:
        """Run predictions on a batch of text inputs."""
        self._ensure_loaded()
        texts = [item["text"] for item in input_data]
        results = self._pipeline(texts)
        return [{"label": r["label"], "score": float(r["score"])} for r in results]

    def warmup(self) -> None:
        """Run a dummy prediction to warm up the pipeline."""
        self._ensure_loaded()
        self.predict({"text": "This is a warmup sentence."})
        logger.info("Model '%s' warmup complete.", self._model_name)

"""Abstract base class for ML model wrappers."""

from __future__ import annotations

import logging
import time
from abc import ABCMeta, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseModel(metaclass=ABCMeta):
    """Abstract base class that all ML model wrappers must implement.

    Provides a consistent interface for loading, predicting, and introspecting
    models regardless of the underlying framework (sklearn, HuggingFace, etc.).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._loaded: bool = False
        self._load_time: float = 0.0
        self._load_timestamp: float | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string (semver recommended)."""

    @abstractmethod
    def load(self) -> None:
        """Load model artifacts into memory.

        Implementations should set ``self._loaded = True`` and record
        ``self._load_time`` (seconds) upon success.
        """

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on a single input.

        Args:
            input_data: Model-specific input (e.g. feature dict, text dict).

        Returns:
            Model-specific prediction result.

        Raises:
            RuntimeError: If the model has not been loaded.
        """

    @abstractmethod
    def predict_batch(self, input_data: list[Any]) -> list[Any]:
        """Run inference on a batch of inputs.

        Args:
            input_data: List of model-specific inputs.

        Returns:
            List of prediction results in the same order.

        Raises:
            RuntimeError: If the model has not been loaded.
        """

    def get_info(self) -> dict[str, Any]:
        """Return model metadata for health-check / introspection endpoints.

        Returns:
            Dictionary containing name, version, loaded status, and timing.
        """
        info: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "loaded": self._loaded,
            "load_time_seconds": round(self._load_time, 4),
        }
        if self._load_timestamp is not None:
            info["loaded_at"] = self._load_timestamp
        return info

    def warmup(self) -> None:
        """Run a dummy prediction to warm up caches, JIT, or lazy graph builds.

        Subclasses should override this with a representative sample input.
        The default implementation is a no-op that logs a warning.
        """
        logger.warning(
            "%s.warmup() not implemented — override in subclass for faster first requests.",
            self.__class__.__name__,
        )

    def _ensure_loaded(self) -> None:
        """Guard that raises if the model has not been loaded yet."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self.name}' is not loaded. Call load() first.")

    def _timed_load(self, loader: Any) -> None:
        """Utility: call *loader* (a callable) while recording wall-clock time.

        Sets ``_loaded``, ``_load_time``, and ``_load_timestamp`` on success.

        Args:
            loader: Zero-argument callable that performs the actual loading.
        """
        start = time.perf_counter()
        loader()
        elapsed = time.perf_counter() - start
        self._loaded = True
        self._load_time = elapsed
        self._load_timestamp = time.time()
        logger.info("Model '%s' v%s loaded in %.3fs", self.name, self.version, elapsed)

"""Model registry — maps model names to their wrapper classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import BaseModel


_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    """Register a model class under *name*."""
    _REGISTRY[name] = cls


def get_model_class(name: str) -> type:
    """Return the model class registered under *name*.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def create_model(name: str, **kwargs: object) -> "BaseModel":
    """Instantiate a model by its registered name."""
    cls = get_model_class(name)
    return cls(**kwargs)


# Register built-in models
from src.models.sklearn_model import SklearnModel  # noqa: E402

register("sklearn_classifier", SklearnModel)

try:
    from src.models.hf_model import HuggingFaceModel  # noqa: E402

    register("hf_sentiment", HuggingFaceModel)
except ImportError:
    pass

"""ML model wrappers for serving predictions via FastAPI."""

from src.models.base import BaseModel
from src.models.sklearn_model import SklearnModel

__all__ = [
    "BaseModel",
    "SklearnModel",
]

# Conditionally expose HuggingFace model if transformers is available
try:
    from src.models.hf_model import HuggingFaceModel  # noqa: F401

    __all__.append("HuggingFaceModel")
except ImportError:
    pass

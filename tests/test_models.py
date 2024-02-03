"""Unit tests for ML model wrappers (BaseModel / SklearnModel)."""

from __future__ import annotations

import pytest

from src.models.base import BaseModel as MLBaseModel


# ---------------------------------------------------------------------------
# test_base_model_interface — verify abstract methods are enforced
# ---------------------------------------------------------------------------


def test_base_model_interface() -> None:
    """Cannot instantiate BaseModel without implementing abstract methods."""
    with pytest.raises(TypeError):
        MLBaseModel()  # type: ignore[abstract]


def test_base_model_ensure_loaded_raises() -> None:
    """_ensure_loaded raises RuntimeError before load() is called."""
    from tests.conftest import MockModel

    m = MockModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m._ensure_loaded()


def test_base_model_get_info_before_load() -> None:
    from tests.conftest import MockModel

    m = MockModel()
    info = m.get_info()
    assert info["loaded"] is False
    assert info["load_time_seconds"] == 0.0
    assert "loaded_at" not in info


# ---------------------------------------------------------------------------
# SklearnModel tests — use a temp directory so we don't pollute the repo
# ---------------------------------------------------------------------------


@pytest.fixture()
def sklearn_model():
    """Create a SklearnModel that trains a demo classifier."""
    from src.models.sklearn_model import SklearnModel

    model = SklearnModel(model_path=None)
    yield model


def test_sklearn_model_load(sklearn_model) -> None:
    sklearn_model.load()
    assert sklearn_model._loaded is True
    assert sklearn_model._load_time > 0


def test_sklearn_model_predict(sklearn_model) -> None:
    sklearn_model.load()
    result = sklearn_model.predict({"features": [5.1, 3.5, 1.4, 0.2]})
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["prediction"], int)


def test_sklearn_model_predict_batch(sklearn_model) -> None:
    sklearn_model.load()
    inputs = [
        {"features": [5.1, 3.5, 1.4, 0.2]},
        {"features": [6.2, 2.9, 4.3, 1.3]},
    ]
    results = sklearn_model.predict_batch(inputs)
    assert len(results) == 2
    for r in results:
        assert "prediction" in r
        assert "probabilities" in r


def test_sklearn_model_predict_not_loaded(sklearn_model) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        sklearn_model.predict({"features": [1.0, 2.0, 3.0, 4.0]})


def test_sklearn_model_warmup(sklearn_model) -> None:
    sklearn_model.load()
    # warmup should not raise
    sklearn_model.warmup()


def test_sklearn_model_get_info(sklearn_model) -> None:
    sklearn_model.load()
    info = sklearn_model.get_info()
    assert info["name"] == "sklearn_classifier"
    assert info["version"] == "1.0.0"
    assert info["loaded"] is True
    assert "load_time_seconds" in info


def test_sklearn_model_predict_missing_features_key(sklearn_model) -> None:
    sklearn_model.load()
    with pytest.raises((KeyError, ValueError)):
        sklearn_model.predict({"wrong_key": [1.0]})


def test_sklearn_model_predict_empty_features(sklearn_model) -> None:
    sklearn_model.load()
    with pytest.raises((ValueError, Exception)):
        sklearn_model.predict({"features": []})


def test_sklearn_model_reload(sklearn_model) -> None:
    """Loading twice (reload) should succeed and update timestamps."""
    sklearn_model.load()
    first_ts = sklearn_model._load_timestamp
    sklearn_model.load()  # reload (re-trains demo)
    assert sklearn_model._loaded is True
    assert sklearn_model._load_timestamp >= first_ts

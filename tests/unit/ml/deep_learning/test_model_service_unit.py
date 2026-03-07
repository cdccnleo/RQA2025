import pytest

from src.ml.deep_learning.core.model_service import ModelService
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def test_model_service_save_and_load():
    service = ModelService()
    model = {"weights": [1, 2, 3]}
    service.save_model("model-a", "1.0", model, metadata={"note": "test"})

    loaded = service.load_model("model-a", "1.0")
    assert loaded == model

    models = service.list_models()
    assert ("model-a", "1.0") in models
    assert models[("model-a", "1.0")]["note"] == "test"
    assert models[("model-a", "1.0")]["status"] == "saved"


def test_model_service_load_missing_raises():
    service = ModelService()
    with pytest.raises(KeyError):
        service.load_model("missing", "0.0")


def test_model_service_duplicate_save_raises():
    service = ModelService()
    service.save_model("dup", "1.0", {"id": "dup"})
    with pytest.raises(ValueError):
        service.save_model("dup", "1.0", {"id": "dup"})

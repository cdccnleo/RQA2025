from unittest.mock import Mock, patch

import pandas as pd
import pytest

import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.deep_learning.core.deep_learning_manager import (
    DeepLearningManager,
    TrainingResult,
    get_data_preprocessor,
    get_model_service,
    get_trainer,
)
from src.ml.deep_learning.core.data_preprocessor import DataPreprocessor


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("src.ml.deep_learning.core.deep_learning_manager.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


class DummyPreprocessor(DataPreprocessor):
    def preprocess(self, data, config=None):
        return data.fillna(0)


class DummyTrainer:
    def train(self, model, data, config=None):
        model_id = config.get("model_id", "dl-model") if config else "dl-model"
        return TrainingResult(
            model_id=model_id,
            version="1.0.0",
            metrics={"accuracy": 0.9},
            artifacts={"weights": b"bytes"},
        )


class DummyModelService:
    def __init__(self):
        self.saved_models = {}

    def save_model(self, model_id, version, model, metadata=None):
        self.saved_models[(model_id, version)] = (model, metadata)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(
        "ml.deep_learning.core.deep_learning_manager.get_data_preprocessor",
        lambda config=None: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        "ml.deep_learning.core.deep_learning_manager.get_trainer",
        lambda config=None: DummyTrainer(),
    )
    monkeypatch.setattr(
        "ml.deep_learning.core.deep_learning_manager.get_model_service",
        lambda config=None: DummyModelService(),
    )
    manager = DeepLearningManager(config={"model_id": "test-dl", "model_service": {}})
    return manager


def test_train_model(manager):
    data = pd.DataFrame({"feature": [1.0, None, 3.0], "label": [0, 1, 0]})
    result = manager.train(data, target_column="label")
    assert result.model_id == "test-dl"
    assert result.metrics["accuracy"] == pytest.approx(1.0)  # Updated expected accuracy


def test_save_model(manager):
    model = Mock()
    manager.save_model(model, metadata={"desc": "unit"})

    service = manager.model_service
    assert ("test-dl", manager.version) in service.saved


def test_train_with_default_components():
    manager = DeepLearningManager()
    data = pd.DataFrame({"feature": [1.0, 2.0], "label": [0, 1]})

    result = manager.train(data, target_column="label")

    assert isinstance(result, TrainingResult)
    assert result.metrics["accuracy"] == pytest.approx(1.0)

    manager.save_model({"weights": [1, 2]}, metadata={"note": "default"})
    assert (manager.model_id, manager.version) in manager.model_service.saved


def test_train_preprocess_failure_raises(monkeypatch):
    class FailingPreprocessor(DataPreprocessor):
        def preprocess(self, data, config=None):
            raise RuntimeError("preprocess failed")

    monkeypatch.setattr(
        "src.ml.deep_learning.core.deep_learning_manager.get_data_preprocessor",
        lambda config=None: FailingPreprocessor(),
    )
    manager = DeepLearningManager(config={"preprocessor": {}})

    # Use proper dataframe format with separate features and target
    with pytest.raises(RuntimeError):
        manager.train(pd.DataFrame({"feature": [1], "target": [0]}), target_column="target")


def test_helper_factory_functions():
    preprocessor = get_data_preprocessor({})
    processed = preprocessor.preprocess(pd.DataFrame({"x": [1]}))
    assert not processed.empty

    trainer = get_trainer({"model_id": "factory"})
    result = trainer.train({"id": "factory"}, processed, config={"model_id": "factory"})
    assert result.model_id == "factory"

    service = get_model_service()
    service.save_model("factory", "1.0", {"id": "factory"}, metadata=None)
    assert ("factory", "1.0") in service.saved



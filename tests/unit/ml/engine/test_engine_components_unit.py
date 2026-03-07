import pandas as pd
import pytest

from src.ml.engine.engine_components import (
    FeatureEngineeringComponent,
    ModelTrainingComponent,
    PredictionPipeline,
    get_feature_engineer,
)


def test_get_feature_engineer_returns_instance():
    engineer = get_feature_engineer()
    engineer.define_feature("value", None, "float")
    assert "value" in engineer.feature_definitions


def test_feature_engineering_component_requires_pipeline():
    component = FeatureEngineeringComponent({})
    data = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="pipeline_name 未配置"):
        component.process(data)


def test_feature_engineering_component_processes_pipeline():
    component = FeatureEngineeringComponent({"pipeline_name": "fill_missing"})
    component.engineer.create_pipeline(
        "fill_missing",
        [{"type": "handle_missing", "method": "fill", "fill_value": 0}],
        ["value"],
    )
    data = pd.DataFrame({"value": [1.0, None]})
    result = component.process(data)
    assert list(result["value"]) == [1.0, 0.0]


def test_model_training_component_success_and_error():
    data = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    component = ModelTrainingComponent({})
    result = component.train(data, "label")
    assert "model" in result and result["model"]["weights"] == len(data.drop(columns=["label"]))

    with pytest.raises(ValueError, match="label_column 不存在"):
        component.train(data.drop(columns=["label"]), "label")


def test_prediction_pipeline_train_and_predict():
    pipeline = PredictionPipeline({})
    train_data = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    pipeline.train(train_data, "label")
    predictions = pipeline.predict(pd.DataFrame({"x": [1, 2, 3]}))
    assert predictions == [0.0, 0.0, 0.0]

    new_pipeline = PredictionPipeline({})
    with pytest.raises(ValueError, match="模型尚未训练"):
        new_pipeline.predict(pd.DataFrame({"x": [1]}))
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.ml.engine.engine_components import (
    FeatureEngineeringComponent,
    ModelTrainingComponent,
    PredictionPipeline,
)
from src.ml.engine.feature_engineering import FeatureEngineer


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("src.ml.engine.engine_components.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


class DummyTrainer:
    def train(self, data, labels, config):
        return {"model": "trained-model", "config": config}


class DummyPredictor:
    def predict(self, model, data):
        return [0.1] * len(data)


def test_feature_engineering_component_runs_pipeline(monkeypatch):
    engineer = FeatureEngineer(config={"enable_caching": False})
    engineer.define_feature("value", feature_type=None, data_type="float")
    engineer.create_pipeline(
        "pipeline",
        [{"type": "handle_missing", "method": "fill", "fill_value": 0}],
        ["value"],
    )

    component = FeatureEngineeringComponent({"pipeline_name": "pipeline"})
    # Directly set the engineer to use our configured instance
    component.engineer = engineer

    data = pd.DataFrame({"value": [1.0, None, 3.0]})

    result = component.process(data)
    assert result["value"].isna().sum() == 0


def test_model_training_component(monkeypatch):
    trainer = DummyTrainer()
    monkeypatch.setattr(
        "src.ml.engine.engine_components.get_model_trainer", lambda config=None: trainer
    )

    component = ModelTrainingComponent({"model_type": "test"})
    data = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})

    model_info = component.train(data, label_column="label")
    assert model_info["model"] == "trained-model"  # DummyTrainer returns string
    assert model_info["config"]["model_type"] == "test"


def test_prediction_pipeline(monkeypatch):
    trainer = DummyTrainer()
    predictor = DummyPredictor()

    monkeypatch.setattr(
        "src.ml.engine.engine_components.get_model_trainer", lambda config=None: trainer
    )
    monkeypatch.setattr(
        "src.ml.engine.engine_components.get_model_predictor", lambda config=None: predictor
    )

    pipeline = PredictionPipeline({"model_type": "test"})
    training_data = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})
    inference_data = pd.DataFrame({"feature": [5, 6]})

    pipeline.train(training_data, label_column="label")
    predictions = pipeline.predict(inference_data)

    assert predictions == [0.1, 0.1]  # DummyPredictor returns 0.1 for each data point


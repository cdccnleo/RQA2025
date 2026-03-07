from types import SimpleNamespace

import pandas as pd
import pytest

from src.ml.core.process_orchestrator import MLProcess, ProcessPriority, ProcessStep, ProcessStatus, MLProcessType
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core.step_executors import (
    BaseMLStepExecutor,
    ModelPredictionExecutor,
    ModelDeploymentExecutor,
    register_ml_step_executors,
    DataLoadingExecutor,
    FeatureEngineeringExecutor,
    ModelTrainingExecutor,
    ModelEvaluationExecutor,
    HyperparameterTuningExecutor,
)


class DummyProcess:
    def __init__(self):
        self.metadata = {}
        self.metrics = {}
        self.process_id = "process-1"


class DummyExecutor(BaseMLStepExecutor):
    def execute(self, step, context):
        return {"step": step, "context": context}


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    error_records = []
    inference_records = []
    model_records = []
    prediction_calls = []
    tuning_calls = []

    class DummyInferenceError(Exception):
        def __init__(self, message, context=None):
            super().__init__(message)
            self.context = context or {}

    class DummyTrainingError(Exception):
        def __init__(self, message, context=None):
            super().__init__(message)
            self.context = context or {}

    class DummyModelManager:
        def __init__(self):
            self.saved_models = {}
            self.raise_in_predict = False
            self.raise_in_create = False

        def predict(self, model, data, method="predict"):
            if self.raise_in_predict:
                raise RuntimeError("predict failed")
            prediction_calls.append((model, data, method))
            if isinstance(data, pd.DataFrame):
                return [42] * len(data)
            return [42]

        def create_model(self, model_type, config=None):
            if self.raise_in_create:
                raise RuntimeError("create failed")
            model_records.append(("create_model", model_type, config))
            return SimpleNamespace()

        def train_model(self, model, X, y, validation_split=0.2):
            model_records.append(("train_model", len(X)))
            return {"score": 0.9}

        def train_with_cv(self, model, X, y, cv_folds=5):
            model_records.append(("train_with_cv", cv_folds))
            return {"score": 0.91, "cv_scores": [0.9, 0.92]}

        def save_model(self, model, model_id):
            self.saved_models[model_id] = model

        def load_model(self, model_id):
            return self.saved_models.get(model_id, SimpleNamespace())

        def evaluate_model(self, model, X, y, metrics=None, confusion_matrix=False, classification_report=False):
            return {"metrics": {"accuracy": 0.9}}

        def tune_hyperparameters(self, **kwargs):
            tuning_calls.append(kwargs)
            return ({"depth": 3}, 0.95, [{"params": {"depth": 3}, "score": 0.95}])

    class DummyFeatureEngineer:
        def process_features(self, X, **kwargs):
            return X

    class DummyInferenceService:
        def deploy_model(self, **kwargs):
            return {"status": "deployed"}

    def record_inference_performance(*args, **kwargs):
        inference_records.append((args, kwargs))

    def record_model_performance(*args, **kwargs):
        model_records.append(("record_model", args, kwargs))

    def handle_ml_error(error):
        error_records.append(error)

    class DummyModelType:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr("src.ml.core.step_executors.ModelManager", DummyModelManager)
    monkeypatch.setattr("src.ml.core.step_executors.FeatureEngineer", DummyFeatureEngineer)
    monkeypatch.setattr("src.ml.core.step_executors.InferenceService", DummyInferenceService)
    monkeypatch.setattr("src.ml.core.step_executors.record_inference_performance", record_inference_performance)
    monkeypatch.setattr("src.ml.core.step_executors.record_model_performance", record_model_performance)
    monkeypatch.setattr("src.ml.core.step_executors.handle_ml_error", handle_ml_error)
    monkeypatch.setattr("src.ml.core.step_executors.InferenceError", DummyInferenceError)
    monkeypatch.setattr("src.ml.core.step_executors.TrainingError", DummyTrainingError)
    monkeypatch.setattr("src.ml.core.step_executors.ModelType", DummyModelType)

    yield {
        "errors": error_records,
        "inference_calls": inference_records,
        "model_records": model_records,
        "prediction_calls": prediction_calls,
        "tuning_calls": tuning_calls,
    }


def test_base_executor_validate_and_metrics():
    executor = DummyExecutor()
    step = ProcessStep(step_id="s1", step_name="demo", step_type="custom")
    assert executor.validate(step) is False  # 缺少 config

    step.config = {"required": True}
    assert executor.validate(step) is True

    process = DummyProcess()
    context = {"process": process}
    data = executor._get_process_data(context)
    assert data is process.metadata

    executor._update_process_metrics(context, {"accuracy": 0.88})
    assert process.metrics["accuracy"] == 0.88


def test_base_executor_handles_dict_context():
    executor = DummyExecutor()
    context = {"process": {"process_id": "proc-2"}}
    data = executor._get_process_data(context)
    data["value"] = 1

    assert context["process"]["metadata"] == {"value": 1}
    assert executor._get_process_id(context) == "proc-2"

    executor._update_process_metrics(context, {"loss": 0.1})
    assert context["process"]["metrics"]["loss"] == 0.1


def test_base_executor_dependencies_and_missing_process():
    executor = DummyExecutor()
    step = ProcessStep(step_id="s2", step_name="demo", step_type="custom", dependencies=["prev"])
    assert executor.get_dependencies(step) == ["prev"]

    context = {}
    assert executor._get_process_data(context) == {}
    assert executor._get_process_id(context) is None


def test_register_ml_step_executors_registers_all(monkeypatch):
    registered = {}

    class OrchestratorStub:
        def register_step_executor(self, step_type, executor):
            registered[step_type] = executor

    orchestrator = OrchestratorStub()
    register_ml_step_executors(orchestrator)
    assert set(registered.keys()) == {
        "data_loading",
        "feature_engineering",
        "model_training",
        "model_evaluation",
        "model_prediction",
        "model_deployment",
        "hyperparameter_tuning",
    }


def test_model_prediction_executor_propagates_failure(stub_dependencies):
    executor = ModelPredictionExecutor()
    executor.model_manager.raise_in_predict = True

    step = ProcessStep(
        step_id="predict",
        step_name="prediction",
        step_type="model_prediction",
        config={"prediction_method": "predict", "batch_size": 2, "output_format": "dataframe"},
    )
    df = pd.DataFrame({"feature": [1, 2, 3]})
    process = DummyProcess()
    process.metadata.update({"trained_model": object(), "prediction_data": df, "model_id": "model-1"})
    context = {"process": process}

    with pytest.raises(RuntimeError):
        executor.execute(step, context)

    assert len(stub_dependencies["errors"]) == 1
    error = stub_dependencies["errors"][0]
    assert error.context["model_id"] == "model-1"
    assert stub_dependencies["inference_calls"] == [((0, "model-1", "predict failed"), {})]
    assert "predictions" not in process.metadata


def test_model_prediction_executor_success_updates_metrics(stub_dependencies):
    executor = ModelPredictionExecutor()
    step = ProcessStep(
        step_id="predict",
        step_name="prediction",
        step_type="model_prediction",
        config={"prediction_method": "predict", "batch_size": 5, "output_format": "dataframe"},
    )
    df = pd.DataFrame({"feature": [1, 2]})
    process = DummyProcess()
    process.metadata.update({"trained_model": object(), "prediction_data": df, "model_id": "model-2"})
    context = {"process": process}

    result = executor.execute(step, context)

    assert isinstance(result, pd.DataFrame)
    assert list(result["predictions"]) == [42, 42]
    assert isinstance(process.metadata["predictions"], pd.DataFrame)
    assert process.metrics["prediction_samples"] == 2
    assert "avg_latency_ms" in process.metrics
    assert stub_dependencies["inference_calls"]
    assert not stub_dependencies["errors"]


def test_model_prediction_executor_batches_large_dataframe(stub_dependencies):
    executor = ModelPredictionExecutor()
    step = ProcessStep(
        step_id="predict-batch",
        step_name="prediction",
        step_type="model_prediction",
        config={"prediction_method": "predict_proba", "batch_size": 2, "output_format": "dict"},
    )
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    process = DummyProcess()
    process.metadata.update({"trained_model": object(), "prediction_data": df, "model_id": "model-batch"})
    context = {"process": process}

    result = executor.execute(step, context)

    assert isinstance(result, dict)
    assert len(result["predictions"]) == len(df)
    calls = stub_dependencies["prediction_calls"]
    assert len(calls) == 3  # ceil(5 / 2)
    assert all(call[2] == "predict_proba" for call in calls)
    assert process.metrics["batch_size"] == 2
    assert stub_dependencies["errors"] == []


def test_model_prediction_executor_loads_model_and_uses_features(stub_dependencies):
    executor = ModelPredictionExecutor()
    saved_model = object()
    executor.model_manager.save_model(saved_model, "model-load")

    process = DummyProcess()
    process.metadata.update({"model_id": "model-load", "features": pd.DataFrame({"x": [1, 2, 3]})})

    step = ProcessStep(step_id="predict-load", step_name="prediction", step_type="model_prediction", config={"output_format": "raw"})

    result = executor.execute(step, {"process": process})
    assert isinstance(result, list)
    assert process.metrics["prediction_samples"] == 3
    assert stub_dependencies["prediction_calls"]


def test_model_prediction_executor_output_format_fallback(stub_dependencies):
    executor = ModelPredictionExecutor()
    process = DummyProcess()
    process.metadata.update({"trained_model": object(), "prediction_data": pd.DataFrame({"x": [1]})})

    step = ProcessStep(step_id="predict-fallback", step_name="prediction", step_type="model_prediction", config={"output_format": "unknown"})

    result = executor.execute(step, {"process": process})
    assert isinstance(result, list)
    assert len(result) == 1


def test_model_prediction_executor_missing_model_raises():
    executor = ModelPredictionExecutor()
    process = DummyProcess()
    process.metadata["prediction_data"] = pd.DataFrame({"x": [1]})
    step = ProcessStep(step_id="predict-no-model", step_name="prediction", step_type="model_prediction", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_model_prediction_executor_missing_prediction_data_raises():
    executor = ModelPredictionExecutor()
    process = DummyProcess()
    process.metadata["trained_model"] = object()
    step = ProcessStep(step_id="predict-no-data", step_name="prediction", step_type="model_prediction", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_model_deployment_executor_returns_result():
    executor = ModelDeploymentExecutor()
    step = ProcessStep(
        step_id="deploy",
        step_name="deployment",
        step_type="model_deployment",
        config={"service_name": "svc", "deployment_target": "local"},
    )
    process = DummyProcess()
    process.metadata.update({"trained_model": object(), "model_id": "model-3"})
    context = {"process": process}

    result = executor.execute(step, context)

    assert result == {"status": "deployed"}
    assert process.metadata["deployment_result"] == {"status": "deployed"}
    assert process.metrics["deployment_target"] == "local"


def test_model_deployment_executor_loads_model_when_absent():
    executor = ModelDeploymentExecutor()
    model_obj = object()
    executor.model_manager.save_model(model_obj, "model-deploy")

    process = DummyProcess()
    process.metadata["model_id"] = "model-deploy"

    step = ProcessStep(step_id="deploy-load", step_name="model_deployment", step_type="model_deployment", config={"service_name": "svc-load"})

    executor.execute(step, {"process": process})
    assert process.metadata["deployment_result"]["status"] == "deployed"


def test_model_deployment_executor_failure_propagates():
    executor = ModelDeploymentExecutor()
    process = DummyProcess()
    process.metadata["trained_model"] = object()

    def raise_deploy(**kwargs):
        raise RuntimeError("deploy boom")

    executor.inference_service.deploy_model = raise_deploy

    step = ProcessStep(step_id="deploy-fail", step_name="model_deployment", step_type="model_deployment", config={})

    with pytest.raises(RuntimeError):
        executor.execute(step, {"process": process})


def test_model_deployment_executor_missing_model_raises():
    executor = ModelDeploymentExecutor()
    process = DummyProcess()
    step = ProcessStep(step_id="deploy-missing", step_name="model_deployment", step_type="model_deployment", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_data_loading_executor_reads_csv(monkeypatch, tmp_path):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(data_path, index=False)

    executor = DataLoadingExecutor()
    process = DummyProcess()
    context = {"process": process}
    step = ProcessStep(
        step_id="load",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_format": "csv",
            "data_path": str(data_path),
        },
    )

    result = executor.execute(step, context)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    assert process.metadata["raw_data"].equals(result)
    assert process.metrics["data_rows"] == 2
    assert process.metrics["data_columns"] == 2


def test_data_loading_executor_database_not_implemented():
    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-db",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "database",
            "query": "SELECT * FROM table",
        },
    )

    with pytest.raises(NotImplementedError):
        executor.execute(step, {"process": process})


def test_data_loading_executor_applies_limit(tmp_path):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": range(10)}).to_csv(data_path, index=False)

    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-limit",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_format": "csv",
            "data_path": str(data_path),
            "limit": 3,
        },
    )

    result = executor.execute(step, {"process": process})
    assert len(result) == 3
    assert process.metrics["data_rows"] == 3


def test_data_loading_executor_reads_json(tmp_path):
    import json

    data_path = tmp_path / "sample.json"
    data_path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")

    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-json",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_format": "json",
            "data_path": str(data_path),
        },
    )

    result = executor.execute(step, {"process": process})
    assert process.metadata["raw_data"].equals(result)


def test_data_loading_executor_reads_parquet(monkeypatch, tmp_path):
    data_path = tmp_path / "sample.parquet"
    dummy_df = pd.DataFrame({"a": [1, 2]})

    def fake_parquet(path):
        assert path == str(data_path)
        return dummy_df

    monkeypatch.setattr("pandas.read_parquet", fake_parquet)
    data_path.write_text("")

    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-parquet",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_format": "parquet",
            "data_path": str(data_path),
        },
    )

    result = executor.execute(step, {"process": process})
    assert result.equals(dummy_df)


def test_data_loading_executor_invalid_format_raises():
    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-invalid-format",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_format": "xml",
            "data_path": "dummy",
        },
    )

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_data_loading_executor_invalid_source_raises():
    executor = DataLoadingExecutor()
    process = DummyProcess()
    step = ProcessStep(
        step_id="load-invalid-source",
        step_name="data_loading",
        step_type="data_loading",
        config={"data_source": "api"},
    )

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_data_loading_executor_validate_checks_required():
    executor = DataLoadingExecutor()
    step = ProcessStep(
        step_id="validate",
        step_name="data_loading",
        step_type="data_loading",
        config={"data_source": "file"},
    )
    assert executor.validate(step) is False


def test_data_loading_executor_validate_respects_base():
    executor = DataLoadingExecutor()
    step = ProcessStep(
        step_id="validate-empty",
        step_name="data_loading",
        step_type="data_loading",
        config={},
    )
    assert executor.validate(step) is False


def test_data_loading_executor_validate_success(tmp_path):
    executor = DataLoadingExecutor()
    step = ProcessStep(
        step_id="validate-success",
        step_name="data_loading",
        step_type="data_loading",
        config={
            "data_source": "file",
            "data_path": str(tmp_path / "dummy.csv"),
        },
    )
    assert executor.validate(step) is True


def test_feature_engineering_executor_updates_metadata(stub_dependencies):
    executor = FeatureEngineeringExecutor()
    executor.feature_engineer = type(
        "StubEngineer",
        (),
        {
            "process_features": lambda self, X, **kwargs: X.assign(sum=X.sum(axis=1))
        },
    )()

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "target": [0, 1]})
    process = DummyProcess()
    process.metadata["raw_data"] = df

    step = ProcessStep(
        step_id="features",
        step_name="feature_engineering",
        step_type="feature_engineering",
        config={"target_column": "target"},
    )

    result = executor.execute(step, {"process": process})
    assert "sum" in result.columns
    assert process.metadata["features"].equals(result)
    assert process.metadata["target"].equals(df["target"])
    assert process.metrics["processed_features"] == result.shape[1]


def test_feature_engineering_executor_requires_raw_data():
    executor = FeatureEngineeringExecutor()
    process = DummyProcess()
    step = ProcessStep(step_id="features-none", step_name="feature_engineering", step_type="feature_engineering", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_feature_engineering_executor_without_target_sets_none():
    executor = FeatureEngineeringExecutor()
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    process = DummyProcess()
    process.metadata["raw_data"] = df

    step = ProcessStep(step_id="features-no-target", step_name="feature_engineering", step_type="feature_engineering", config={})

    result = executor.execute(step, {"process": process})
    assert process.metadata["target"] is None
    assert result.equals(df)


def test_feature_engineering_executor_propagates_errors():
    executor = FeatureEngineeringExecutor()

    class BrokenEngineer:
        def process_features(self, *args, **kwargs):
            raise RuntimeError("broken engineer")

    executor.feature_engineer = BrokenEngineer()
    df = pd.DataFrame({"x": [1, 2]})
    process = DummyProcess()
    process.metadata["raw_data"] = df

    step = ProcessStep(step_id="features-error", step_name="feature_engineering", step_type="feature_engineering", config={})

    with pytest.raises(RuntimeError):
        executor.execute(step, {"process": process})


def test_model_training_executor_missing_features_raises(stub_dependencies):
    executor = ModelTrainingExecutor()
    process = DummyProcess()
    process.metadata["target"] = pd.Series([1, 2, 3])
    step = ProcessStep(
        step_id="train",
        step_name="model_training",
        step_type="model_training",
        config={"model_type": "linear"},
    )

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_model_training_executor_success(stub_dependencies):
    executor = ModelTrainingExecutor()
    process = DummyProcess()

    features = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    target = pd.Series([0, 1, 0])
    process.metadata["features"] = features
    process.metadata["target"] = target

    step = ProcessStep(
        step_id="train",
        step_name="model_training",
        step_type="model_training",
        config={"model_type": "linear", "validation_split": 0.1},
    )

    result = executor.execute(step, {"process": process})

    assert "model_id" in result
    assert process.metadata["model_id"] == result["model_id"]
    assert process.metadata["trained_model"] is not None
    assert process.metrics["model_type"] == "linear"
    assert process.metrics["training_samples"] == len(features)
    assert any(record[0] == "train_model" for record in stub_dependencies["model_records"])


def test_model_training_executor_cross_validation_metrics(stub_dependencies):
    executor = ModelTrainingExecutor()
    process = DummyProcess()

    features = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    target = pd.Series([0, 1, 0])
    process.metadata["features"] = features
    process.metadata["target"] = target

    step = ProcessStep(
        step_id="train-cv",
        step_name="model_training",
        step_type="model_training",
        config={"model_type": "linear", "cross_validation": True, "cv_folds": 3},
    )

    result = executor.execute(step, {"process": process})
    assert process.metrics["cross_validation"] is True
    assert "cv_scores" in process.metrics
    assert any(record[0] == "train_with_cv" for record in stub_dependencies["model_records"])
    assert "cv_scores" in result["results"]


def test_model_training_executor_failure_records_error(stub_dependencies):
    executor = ModelTrainingExecutor()
    executor.model_manager.raise_in_create = True

    process = DummyProcess()
    features = pd.DataFrame({"x": [1, 2, 3]})
    target = pd.Series([0, 1, 0])
    process.metadata["features"] = features
    process.metadata["target"] = target

    step = ProcessStep(step_id="train-fail", step_name="model_training", step_type="model_training", config={"model_type": "linear"})

    with pytest.raises(RuntimeError):
        executor.execute(step, {"process": process})

    assert stub_dependencies["errors"]
    error = stub_dependencies["errors"][0]
    assert error.context["model_type"] == "linear"
    assert error.context["process_id"] == process.process_id


def test_model_evaluation_executor_success(stub_dependencies):
    executor = ModelEvaluationExecutor()
    process = DummyProcess()

    features = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    target = pd.Series([0, 1, 0])
    process.metadata.update({
        "trained_model": object(),
        "model_id": "model-5",
        "features": features,
        "target": target,
    })

    step = ProcessStep(
        step_id="evaluate",
        step_name="model_evaluation",
        step_type="model_evaluation",
        config={"metrics": ["accuracy"]},
    )

    results = executor.execute(step, {"process": process})

    assert results["metrics"]["accuracy"] == 0.9
    assert process.metadata["evaluation_results"] == results
    assert "predictions" in process.metadata
    assert process.metrics["accuracy"] == 0.9
    assert any(record[0] == "record_model" for record in stub_dependencies["model_records"])


def test_hyperparameter_tuning_executor_records_results(stub_dependencies):
    executor = HyperparameterTuningExecutor()
    process = DummyProcess()
    X = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    y = pd.Series([0, 1, 0])
    process.metadata.update({"features": X, "target": y})

    step = ProcessStep(
        step_id="tune",
        step_name="hyperparameter_tuning",
        step_type="hyperparameter_tuning",
        config={
            "model_type": "random_forest",
            "param_space": {"depth": [3, 5]},
            "tuning_method": "grid",
            "cv_folds": 3,
            "max_evals": 5,
            "scoring_metric": "accuracy",
        },
    )

    result = executor.execute(step, {"process": process})

    assert "best_params" in result
    assert process.metadata["best_params"] == {"depth": 3}
    assert process.metadata["best_model_id"].startswith("random_forest_tuned_")
    assert process.metrics["tuning_method"] == "grid"
    assert process.metrics["tuning_evaluations"] == len(result["tuning_results"])
    assert stub_dependencies["tuning_calls"]


def test_hyperparameter_tuning_executor_requires_data():
    executor = HyperparameterTuningExecutor()
    process = DummyProcess()

    step = ProcessStep(step_id="tune-missing", step_name="hyperparameter_tuning", step_type="hyperparameter_tuning", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_hyperparameter_tuning_executor_failure_propagates(stub_dependencies):
    executor = HyperparameterTuningExecutor()
    process = DummyProcess()
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    process.metadata.update({"features": X, "target": y})

    step = ProcessStep(step_id="tune-fail", step_name="hyperparameter_tuning", step_type="hyperparameter_tuning", config={})

    def raise_tune(**kwargs):
        raise RuntimeError("tune boom")

    executor.model_manager.tune_hyperparameters = raise_tune

    with pytest.raises(RuntimeError):
        executor.execute(step, {"process": process})


def test_step_executors_logger_fallback(monkeypatch):
    import importlib
    import sys
    from types import ModuleType
    from src.ml.core import step_executors as original_module

    original_integration = sys.modules.get("src.core.integration")
    failing_module = ModuleType("src.core.integration")

    def failing_adapter():
        raise RuntimeError("adapter failure")

    failing_module.get_models_adapter = failing_adapter
    monkeypatch.setitem(sys.modules, "src.core.integration", failing_module)

    reloaded = importlib.reload(original_module)
    try:
        assert reloaded.logger.name == "src.ml.core.step_executors"
    finally:
        if original_integration is None:
            sys.modules.pop("src.core.integration", None)
        else:
            sys.modules["src.core.integration"] = original_integration
        importlib.reload(reloaded)


def test_model_evaluation_executor_loads_model_when_missing(stub_dependencies):
    executor = ModelEvaluationExecutor()
    process = DummyProcess()

    features = pd.DataFrame({"x": [1, 2, 3]})
    target = pd.Series([0, 1, 0])
    process.metadata.update({"model_id": "model-load", "features": features, "target": target})
    executor.model_manager.save_model(object(), "model-load")

    step = ProcessStep(step_id="evaluate-load", step_name="model_evaluation", step_type="model_evaluation", config={})
    executor.execute(step, {"process": process})

    assert "evaluation_results" in process.metadata


def test_model_evaluation_executor_missing_test_data_raises(stub_dependencies):
    executor = ModelEvaluationExecutor()
    process = DummyProcess()
    process.metadata["trained_model"] = object()

    step = ProcessStep(step_id="evaluate-missing", step_name="model_evaluation", step_type="model_evaluation", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_model_evaluation_executor_missing_model_raises():
    executor = ModelEvaluationExecutor()
    process = DummyProcess()
    features = pd.DataFrame({"x": [1]})
    target = pd.Series([0])
    process.metadata.update({"features": features, "target": target})

    step = ProcessStep(step_id="evaluate-no-model", step_name="model_evaluation", step_type="model_evaluation", config={})

    with pytest.raises(ValueError):
        executor.execute(step, {"process": process})


def test_model_evaluation_executor_failure_propagates(stub_dependencies):
    executor = ModelEvaluationExecutor()
    process = DummyProcess()
    features = pd.DataFrame({"x": [1, 2, 3]})
    target = pd.Series([0, 1, 0])
    process.metadata.update({"trained_model": object(), "features": features, "target": target})

    step = ProcessStep(step_id="evaluate-fail", step_name="model_evaluation", step_type="model_evaluation", config={})

    def raise_evaluate(*args, **kwargs):
        raise RuntimeError("evaluate boom")

    executor.model_manager.evaluate_model = raise_evaluate

    with pytest.raises(RuntimeError):
        executor.execute(step, {"process": process})

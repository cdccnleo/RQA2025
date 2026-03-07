import logging
import numpy as np
import pandas as pd
import pytest

from src.features.processors.advanced.advanced_feature_processor import (
    MLFeatureProcessor,
)
from src.features.processors.base_processor import ProcessorConfig


@pytest.fixture
def ml_processor():
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger("tests.ml_feature_processor")
    return processor


@pytest.fixture
def sample_frame():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    return pd.DataFrame(
        {
            "close": np.linspace(10.0, 50.0, num=40),
            "volume": np.linspace(100, 400, num=40),
        },
        index=idx,
    )


def test_clustering_score_non_numeric_returns_nan(ml_processor, sample_frame):
    frame = sample_frame.assign(label=["a"] * len(sample_frame))
    result = ml_processor._compute_clustering_score(frame, {"columns": ["label"], "window": 10})
    assert result.isna().all()


def test_anomaly_score_small_window_returns_nan(ml_processor, sample_frame):
    short = sample_frame.head(5)
    result = ml_processor._compute_anomaly_score(short, {"columns": ["close", "volume"], "window": 10})
    assert result.isna().all()


def test_pca_component_reuses_cached_models(ml_processor, sample_frame):
    params = {"columns": ["close", "volume"], "component": 0, "window": 10}
    first = ml_processor._compute_pca_component(sample_frame, params)
    second = ml_processor._compute_pca_component(sample_frame, params)
    assert first.equals(second)


def test_feature_importance_handles_missing_valid_rows(ml_processor, sample_frame):
    frame = sample_frame.copy()
    frame.loc[frame.index[:30], "close"] = np.nan
    result = ml_processor._compute_feature_importance(
        frame,
        {"window": 20, "feature_columns": ["close"]},
    )
    non_null = result.dropna()
    assert not non_null.empty
    assert non_null.index.min() >= frame.index[30]


def test_clustering_score_handles_kmeans_failure(sample_frame, monkeypatch, caplog):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger("tests.ml_feature_processor.kmeans")

    class BoomKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("sklearn.cluster.KMeans", BoomKMeans)
    with caplog.at_level("WARNING"):
        scores = processor._compute_clustering_score(sample_frame, {"columns": ["close", "volume"], "window": 10})
    assert scores.isna().all()
    assert any("聚类计算失败" in message for message in caplog.messages)


def test_anomaly_score_singular_covariance_falls_back_to_euclidean(ml_processor, sample_frame, monkeypatch):
    class SingularCovariance:
        def __init__(self, matrix: np.ndarray):
            self.matrix = matrix
            self.shape = matrix.shape

        @property
        def values(self):
            return self.matrix

        def __array__(self, dtype=None):
            return self.matrix.astype(dtype) if dtype else self.matrix

    def fake_cov(self, *args, **kwargs):
        size = len(self.columns)
        matrix = np.zeros((size, size))
        return SingularCovariance(matrix)

    monkeypatch.setattr(pd.DataFrame, "cov", fake_cov)
    result = ml_processor._compute_anomaly_score(
        sample_frame,
        {"columns": ["close", "volume"], "window": 10},
    )
    non_null = result.dropna()
    assert not non_null.empty
    assert (non_null > 0).all()


def test_pca_component_handles_scaler_failure_returns_nan(sample_frame, monkeypatch, caplog):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger("tests.ml_feature_processor.pca")

    class BoomScaler:
        def fit_transform(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.StandardScaler",
        lambda: BoomScaler(),
    )

    with caplog.at_level("WARNING"):
        result = processor._compute_pca_component(
            sample_frame,
            {"columns": ["close", "volume"], "component": 0, "window": 10},
        )
    assert result.dropna().empty
    assert any("PCA计算失败" in message for message in caplog.messages)


def test_clustering_score_exception_logs_warning(sample_frame, monkeypatch, caplog):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger("tests.ml_feature_processor.cluster")

    class BoomScaler:
        def fit_transform(self, *_args, **_kwargs):
            raise RuntimeError("broken scaler")

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.StandardScaler",
        lambda: BoomScaler(),
    )

    with caplog.at_level("WARNING"):
        result = processor._compute_clustering_score(
            sample_frame,
            {"columns": ["close", "volume"], "window": 10, "n_clusters": 2},
        )
    assert result.dropna().empty
    assert any("聚类计算失败" in message for message in caplog.messages)


def test_pca_component_returns_nan_when_numeric_data_missing(ml_processor, sample_frame):
    frame = sample_frame.copy()
    frame["text"] = ["x"] * len(frame)
    result = ml_processor._compute_pca_component(frame[["text"]], {"columns": ["text"], "window": 10})
    assert result.isna().all()


def test_feature_importance_handles_exception(ml_processor, sample_frame, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("mi fail")

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.mutual_info_regression",
        boom,
    )
    with caplog.at_level("WARNING"):
        scores = ml_processor._compute_feature_importance(
            sample_frame[["close", "volume"]],
            {"window": 20, "feature_columns": ["volume"], "target_column": "close"},
        )
    assert scores.dropna().empty
    assert any("特征重要性计算失败" in message for message in caplog.messages)


def test_anomaly_score_general_exception_returns_empty(ml_processor, sample_frame, monkeypatch, caplog):
    def boom_cov(*_args, **_kwargs):
        raise RuntimeError("cov fail")

    monkeypatch.setattr(pd.DataFrame, "cov", boom_cov)
    with caplog.at_level("WARNING"):
        scores = ml_processor._compute_anomaly_score(sample_frame, {"window": 20})
    assert scores.dropna().empty
    assert any("异常检测计算失败" in message for message in caplog.messages)


def test_compute_feature_unknown_raises(ml_processor, sample_frame):
    with pytest.raises(ValueError):
        ml_processor._compute_feature(sample_frame, "unknown_feature", {})


def test_anomaly_score_respects_cached_result(ml_processor, sample_frame, monkeypatch):
    called = {"flag": False}

    original = ml_processor._compute_anomaly_score

    def wrapped_anomaly(self, data, params):
        called["flag"] = True
        return original(data, params)

    monkeypatch.setattr(
        MLFeatureProcessor,
        "_compute_anomaly_score",
        wrapped_anomaly,
    )
    first = ml_processor._compute_anomaly_score(sample_frame, {"columns": ["close", "volume"], "window": 10})
    second = ml_processor._compute_anomaly_score(sample_frame, {"columns": ["close", "volume"], "window": 10})
    assert second.equals(first)
    assert called["flag"] is True


def test_pca_component_logs_when_exception(ml_processor, sample_frame, monkeypatch, caplog):
    def boom_scaler():
        class Boom:
            def fit_transform(self, *_args, **_kwargs):
                raise RuntimeError("scale fail")
        return Boom()

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.StandardScaler",
        boom_scaler,
    )
    with caplog.at_level("WARNING"):
        result = ml_processor._compute_pca_component(sample_frame, {"columns": ["close"], "window": 5})
    assert result.dropna().empty
    assert any("PCA计算失败" in message for message in caplog.messages)


def test_clustering_score_returns_empty_when_all_failures(ml_processor, sample_frame, monkeypatch, caplog):
    def boom_cluster(*_args, **_kwargs):
        raise RuntimeError("cluster fail")

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.StandardScaler",
        lambda: (_ for _ in ()).throw(RuntimeError("scale fail")),
        raising=False,
    )
    with caplog.at_level("WARNING"):
        result = ml_processor._compute_clustering_score(sample_frame, {"columns": ["close"], "window": 5})
    assert result.dropna().empty
    assert any("聚类计算失败" in message for message in caplog.messages)


import logging
import numpy as np
import pandas as pd
import pytest

from src.features.processors.advanced.advanced_feature_processor import (
    MLFeatureProcessor,
    StatisticalFeatureProcessor,
    TimeSeriesFeatureProcessor,
)
from src.features.processors.base_processor import ProcessorConfig


@pytest.fixture
def time_series_processor():
    config = ProcessorConfig(processor_type="time_series", feature_params={})
    return TimeSeriesFeatureProcessor(config)


@pytest.fixture
def statistical_processor():
    config = ProcessorConfig(processor_type="statistical", feature_params={})
    return StatisticalFeatureProcessor(config)


@pytest.fixture
def ml_processor():
    config = ProcessorConfig(processor_type="ml", feature_params={})
    processor = MLFeatureProcessor(config)
    processor.logger = logging.getLogger(__name__)
    return processor


@pytest.fixture
def sample_price_frame():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "close": np.linspace(50, 110, num=60),
            "high": np.linspace(51, 111, num=60),
            "low": np.linspace(49, 109, num=60),
            "volume": np.linspace(1000, 3000, num=60),
        },
        index=idx,
    )


def test_time_series_trend_strength(time_series_processor, sample_price_frame):
    params = {"window": 10, "column": "close"}
    trend = time_series_processor._compute_feature(sample_price_frame, "trend_strength", params)
    assert trend.iloc[-1] == pytest.approx(1.0, rel=1e-3)


def test_time_series_invalid_feature_raises(time_series_processor, sample_price_frame):
    with pytest.raises(ValueError):
        time_series_processor._compute_feature(sample_price_frame, "unknown_feature", {})


def test_statistical_outlier_score(statistical_processor, sample_price_frame):
    params = {"column": "close", "window": 5, "threshold": 0.5}
    scores = statistical_processor._compute_feature(sample_price_frame, "outlier_score", params)
    assert scores.ge(0).all()
    assert scores.notna().sum() > 0


def test_statistical_distribution_score(statistical_processor, sample_price_frame):
    params = {"column": "close", "window": 5}
    distribution = statistical_processor._compute_feature(sample_price_frame, "distribution_score", params)
    assert distribution.notna().sum() > 0


def test_ml_feature_importance(monkeypatch, ml_processor, sample_price_frame):
    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.mutual_info_regression",
        lambda X, y, random_state=None: np.full(X.shape[1], 0.3),
    )
    params = {
        "target_column": "close",
        "feature_columns": ["high", "low", "volume"],
        "window": 15,
    }
    scores = ml_processor._compute_feature(sample_price_frame, "feature_importance", params)
    assert scores.dropna().iloc[-1] == pytest.approx(0.3, rel=1e-6)


def test_ml_clustering_score(monkeypatch, ml_processor, sample_price_frame):
    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=None):
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            length = len(data)
            labels = np.tile(np.arange(self.n_clusters), length // self.n_clusters + 1)
            return labels[:length]

    monkeypatch.setattr("sklearn.cluster.KMeans", DummyKMeans)
    monkeypatch.setattr("sklearn.metrics.silhouette_score", lambda data, labels: 0.42)

    params = {"columns": ["close", "volume"], "window": 20, "n_clusters": 3}
    scores = ml_processor._compute_feature(sample_price_frame, "clustering_score", params)
    assert scores.dropna().iloc[-1] == pytest.approx(0.42, rel=1e-6)


def test_ml_anomaly_score(monkeypatch, ml_processor, sample_price_frame):
    # Avoid singular covariance by adding small noise
    params = {"columns": ["close", "volume"], "window": len(sample_price_frame) + 5}
    scores = ml_processor._compute_feature(sample_price_frame, "anomaly_score", params)
    assert scores.isna().all()


def test_time_series_metadata_and_available(time_series_processor):
    metadata = time_series_processor._get_feature_metadata("rolling_mean")
    assert metadata["type"] == "time_series"
    assert "window" in metadata["parameters"]
    available = time_series_processor._get_available_features()
    assert "volatility" in available


def test_statistical_invalid_feature_raises(statistical_processor, sample_price_frame):
    with pytest.raises(ValueError):
        statistical_processor._compute_feature(sample_price_frame, "unsupported", {})


def test_statistical_metadata(statistical_processor):
    meta = statistical_processor._get_feature_metadata("iqr")
    assert meta["category"] == "analytical"
    assert "window" in meta["parameters"]
    assert "zscore" in statistical_processor._get_available_features()


def test_ml_invalid_feature_raises(ml_processor, sample_price_frame):
    with pytest.raises(ValueError):
        ml_processor._compute_feature(sample_price_frame, "unknown", {})


def test_ml_pca_component_produces_values(ml_processor, sample_price_frame):
    params = {"columns": ["close", "volume"], "component": 0, "window": 20}
    scores = ml_processor._compute_feature(sample_price_frame, "pca_component", params)
    assert scores.iloc[-1] == pytest.approx(scores.dropna().iloc[-1], rel=1)


def test_ml_feature_importance_not_enough_data(ml_processor, sample_price_frame):
    params = {"target_column": "close", "feature_columns": ["volume"], "window": 80}
    scores = ml_processor._compute_feature(sample_price_frame, "feature_importance", params)
    assert scores.isna().all()


def test_ml_clustering_score_single_cluster(monkeypatch, ml_processor, sample_price_frame):
    class DummyKMeans:
        def __init__(self, n_clusters, random_state=None, n_init=None):
            pass

        def fit_predict(self, data):
            return np.zeros(len(data), dtype=int)

    monkeypatch.setattr("sklearn.cluster.KMeans", DummyKMeans)
    params = {"columns": ["close", "volume"], "window": 20, "n_clusters": 3}
    scores = ml_processor._compute_feature(sample_price_frame, "clustering_score", params)
    assert scores.dropna().empty


def test_ml_anomaly_score_euclidean_fallback(monkeypatch, ml_processor, sample_price_frame):
    params = {"columns": ["close", "volume"], "window": 30}

    def fake_inv(matrix):
        raise np.linalg.LinAlgError("singular")

    def fake_isnan(arr):
        values = np.asarray(arr)
        return np.zeros_like(values, dtype=bool)

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.np.isnan",
        fake_isnan,
    )
    monkeypatch.setattr("numpy.linalg.inv", fake_inv)
    scores = ml_processor._compute_feature(sample_price_frame, "anomaly_score", params)
    valid = scores.dropna()
    assert not valid.empty
    assert valid.iloc[-1] > 0


def test_time_series_trend_strength_insufficient_window(time_series_processor, sample_price_frame):
    result = time_series_processor._compute_trend_strength(sample_price_frame.head(5), {"window": 10})
    assert result.isna().all()


def test_statistical_outlier_score_parameters(statistical_processor, sample_price_frame):
    series = statistical_processor._compute_outlier_score(sample_price_frame, {"column": "close", "window": 5, "threshold": 1})
    assert isinstance(series, pd.Series)


def test_ml_pca_component_handles_non_numeric(ml_processor, sample_price_frame):
    data = sample_price_frame.assign(label=["x"] * len(sample_price_frame))
    result = ml_processor._compute_pca_component(data, {"columns": ["label"], "window": 10})
    assert result.isna().all()


def test_ml_feature_importance_handles_exception(monkeypatch, ml_processor, sample_price_frame, caplog):
    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.mutual_info_regression",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mi fail")),
    )
    result = ml_processor._compute_feature(sample_price_frame, "feature_importance", {"window": 10})
    assert result.dropna().empty
    assert any("特征重要性计算失败" in message for message in caplog.messages)


def test_time_series_metadata_parameters(time_series_processor):
    metadata = time_series_processor._get_feature_metadata("momentum")
    assert metadata["parameters"] == ["period", "column"]


def test_statistical_metadata_unknown_returns_default(statistical_processor):
    meta = statistical_processor._get_feature_metadata("unknown")
    assert meta["name"] == "unknown"
    assert meta["type"] == "statistical"


def test_ml_get_available_features(ml_processor):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    features = processor._get_available_features()
    assert {"pca_component", "feature_importance", "clustering_score", "anomaly_score"}.issubset(features)


def test_ml_clustering_score_handles_exception(sample_price_frame, monkeypatch):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger(__name__)

    class BoomKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, *_args, **_kwargs):
            raise RuntimeError("cluster fail")

    monkeypatch.setattr("sklearn.cluster.KMeans", BoomKMeans)
    scores = processor._compute_clustering_score(sample_price_frame, {"columns": ["close", "volume"], "window": 10})
    assert scores.isna().all()


def test_ml_clustering_score_requires_numeric(sample_price_frame):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger(__name__)
    frame = sample_price_frame.assign(label=["a"] * len(sample_price_frame))
    result = processor._compute_clustering_score(frame, {"columns": ["label"], "window": 10})
    assert result.isna().all()


def test_ml_anomaly_score_small_window_returns_nan(sample_price_frame):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger(__name__)
    short = sample_price_frame.head(10)
    result = processor._compute_anomaly_score(short, {"columns": ["close", "volume"], "window": 30})
    assert result.isna().all()


def test_ml_pca_component_uses_cached_scaler(sample_price_frame, monkeypatch):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger(__name__)
    calls = {"count": 0}

    class TrackingScaler:
        def fit_transform(self, data):
            calls["count"] += 1
            return data.to_numpy()

    creations = {"scaler": 0, "pca": 0}

    class TrackingScaler:
        def __init__(self):
            creations["scaler"] += 1

        def fit_transform(self, data):
            return data.to_numpy()

    class TrackingPCA:
        def __init__(self, n_components=None, random_state=None):
            creations["pca"] += 1
            self.n_components = n_components

        def fit_transform(self, data):
            n_components = self.n_components or data.shape[1]
            rows = len(data)
            base = np.arange(n_components)
            return np.tile(base, (rows, 1))

    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.StandardScaler",
        lambda: TrackingScaler(),
    )
    monkeypatch.setattr(
        "src.features.processors.advanced.advanced_feature_processor.PCA",
        lambda n_components=None, random_state=None: TrackingPCA(n_components, random_state),
    )

    params = {"columns": ["close", "volume"], "window": 20}
    processor._compute_pca_component(sample_price_frame, params)
    processor._compute_pca_component(sample_price_frame, params)
    assert creations["scaler"] == 1
    assert creations["pca"] == 1


def test_ml_feature_importance_skips_when_insufficient_valid(sample_price_frame):
    processor = MLFeatureProcessor(ProcessorConfig(processor_type="ml", feature_params={}))
    processor.logger = logging.getLogger(__name__)
    frame = sample_price_frame.copy()
    frame.loc[:, ["volume", "high", "low"]] = np.nan
    result = processor._compute_feature_importance(frame, {"window": 30})
    assert result.dropna().empty


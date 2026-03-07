import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.core.feature_config import FeatureConfig, FeatureType
from src.features.core.config_integration import ConfigScope
from src.features.core.feature_manager import FeatureManager


class DummyConfigManager:
    """轻量配置管理器，便于控制处理配置与监听行为。"""

    def __init__(self, initial=None):
        self._configs = initial or {}
        self.watchers = []

    def get_config(self, scope: ConfigScope):
        return self._configs.get(scope)

    def register_config_watcher(self, scope: ConfigScope, callback):
        self.watchers.append((scope, callback))

    def trigger_change(self, scope: ConfigScope, key: str, new_value):
        for registered_scope, callback in self.watchers:
            if registered_scope == scope:
                callback(scope, key, None, new_value)


class DummyFeatureEngineer:
    def __init__(self):
        self.saved_metadata = {}

    def generate_technical_features(self, data):
        length = len(data) if hasattr(data, "__len__") else 3
        return pd.DataFrame({"tech_feature": np.arange(length)})

    def generate_sentiment_features(self, data):
        length = len(data) if hasattr(data, "__len__") else 3
        return pd.DataFrame({"sentiment_feature": np.linspace(0.1, 0.9, length)})

    def merge_features(self, frames):
        return pd.concat(frames, axis=1)

    def save_metadata(self, features, source):
        self.saved_metadata[source] = {"columns": list(features.columns)}

    def load_metadata(self, source):
        return self.saved_metadata.get(source, {})


class DummyFeatureSaver:
    def __init__(self):
        self.saved = {}

    def save_features(self, df, path, fmt="parquet"):
        self.saved[(str(path), fmt)] = df.copy()

    def load_features(self, path, fmt="parquet"):
        return self.saved[(str(path), fmt)]


class DummyComponent:
    def __init__(self, *_, **__):
        pass


@pytest.fixture
def sample_df():
    return pd.DataFrame({"price": [100.0, 101.0, 102.0], "volume": [200, 210, 215]})


@pytest.fixture
def feature_config():
    config = FeatureConfig()
    config.feature_types = [FeatureType.TECHNICAL, FeatureType.SENTIMENT]
    config.cache_ttl = 0.1
    config.feature_selection_method = "mutual_info"
    config.max_features = 5
    config.min_feature_importance = 0.05
    config.standardization_method = "zscore"
    config.robust_scaling = True
    return config


@pytest.fixture
def manager_fixture(monkeypatch, feature_config):
    processing_config = {
        "max_workers": 8,
        "batch_size": 256,
        "timeout": 12,
        "feature_selection_method": "mutual_info",
        "max_features": 7,
        "min_feature_importance": 0.02,
        "standardization_method": "zscore",
        "robust_scaling": False,
    }
    config_manager = DummyConfigManager({ConfigScope.PROCESSING: processing_config})

    monkeypatch.setattr(
        "src.features.core.feature_manager.get_config_integration_manager",
        lambda: config_manager,
    )
    monkeypatch.setattr(
        "src.features.core.feature_manager.FeatureEngineer", DummyFeatureEngineer
    )
    monkeypatch.setattr(
        "src.features.core.feature_manager.FeatureProcessor", DummyComponent
    )
    monkeypatch.setattr(
        "src.features.core.feature_manager.FeatureSelector", DummyComponent
    )
    monkeypatch.setattr(
        "src.features.core.feature_manager.FeatureStandardizer", DummyComponent
    )
    monkeypatch.setattr(
        "src.features.core.feature_manager.FeatureSaver", DummyFeatureSaver
    )

    manager = FeatureManager(config=feature_config)
    return manager, config_manager


def test_processing_config_applied(manager_fixture):
    manager, _ = manager_fixture
    assert manager.config.max_workers == 8
    assert manager.config.batch_size == 256
    assert manager.config.feature_selection_method == "mutual_info"


def test_process_single_dataframe_merges_features(manager_fixture, sample_df):
    manager, _ = manager_fixture
    result = manager.process_features(sample_df)
    assert set(result.columns) == {"tech_feature", "sentiment_feature"}
    assert len(result) == len(sample_df)


def test_process_multiple_dataframes_prefixes_columns(manager_fixture, sample_df):
    manager, _ = manager_fixture
    data = {"alpha": sample_df, "beta": sample_df * 2}
    result = manager.process_features(data)
    assert any(col.startswith("alpha_") for col in result.columns)
    assert any(col.startswith("beta_") for col in result.columns)


def test_process_features_invalid_type_raises(manager_fixture):
    manager, _ = manager_fixture
    with pytest.raises(ValueError):
        manager.process_features(["not", "supported"])  # type: ignore[arg-type]


def test_config_watcher_updates_runtime_settings(manager_fixture):
    manager, config_manager = manager_fixture
    config_manager.trigger_change(ConfigScope.PROCESSING, "max_workers", 42)
    assert manager.config.max_workers == 42


def test_get_feature_info_and_validation(manager_fixture, sample_df):
    manager, _ = manager_fixture
    processed = manager.process_features(sample_df)
    info = manager.get_feature_info(processed)
    assert info["feature_count"] == 2
    validation = manager.validate_features(processed)
    assert validation["has_features"]
    assert validation["has_samples"]


def test_cleanup_cache_resets_when_expired(manager_fixture):
    manager, _ = manager_fixture
    manager._feature_cache["cached"] = pd.DataFrame({"a": [1]})
    manager._last_cache_cleanup = time.time() - 10
    manager.cleanup_cache()
    assert manager._feature_cache == {}


def test_run_pipeline_returns_metadata(manager_fixture):
    manager, _ = manager_fixture
    result = manager.run("source-id")
    assert result["status"] == "success"
    assert "metadata" in result
    assert result["metadata"]["columns"] == ["tech_feature", "sentiment_feature"]


def test_save_and_load_features_delegate(manager_fixture, sample_df, tmp_path):
    manager, _ = manager_fixture
    path = tmp_path / "features.bin"
    manager.save_features(sample_df, path, format="pickle")
    loaded = manager.load_features(path, format="pickle")
    pd.testing.assert_frame_equal(loaded, sample_df)
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.features.core.config import FeatureConfig, FeatureType
from src.features.core.config_integration import ConfigScope
from src.features.core.feature_manager import FeatureManager


@pytest.fixture
def mock_config_manager(monkeypatch):
    """
    替换配置集成管理器，避免依赖真实基础设施。
    """
    fake_manager = MagicMock()
    fake_manager.get_config.return_value = {}
    fake_manager.register_config_watcher = MagicMock()
    monkeypatch.setattr(
        "src.features.core.feature_manager.get_config_integration_manager",
        lambda: fake_manager,
    )
    return fake_manager


@pytest.fixture
def feature_manager(mock_config_manager):
    """
    返回注入了可控依赖的 FeatureManager 实例。
    """
    manager = FeatureManager()
    manager.feature_engineer = MagicMock()
    manager.feature_processor = MagicMock()
    manager.feature_selector = MagicMock()
    manager.feature_standardizer = MagicMock()
    manager.feature_saver = MagicMock()
    return manager


def test_process_features_single_dataframe(feature_manager, sample_price_frame):
    """
    验证传入单个 DataFrame 时，特征管理器会调用特征工程并返回合并后的特征。
    """
    engineered = sample_price_frame.assign(tech_feature=1.0)
    feature_manager.feature_engineer.generate_technical_features.return_value = engineered

    result = feature_manager.process_features(sample_price_frame)

    feature_manager.feature_engineer.generate_technical_features.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert "tech_feature" in result.columns


def test_process_features_multiple_frames(feature_manager, sample_price_frame):
    """
    验证传入字典数据时会逐个处理并带上数据集名前缀。
    """
    feature_manager.config.feature_types = [FeatureType.TECHNICAL]
    feature_manager.feature_engineer.generate_technical_features.return_value = sample_price_frame[["close"]].copy()

    combined = feature_manager.process_features({"alpha": sample_price_frame, "beta": sample_price_frame})

    assert isinstance(combined, pd.DataFrame)
    assert all(col.startswith(("alpha_", "beta_")) for col in combined.columns)
    assert combined.shape[1] == 2  # alpha_close 与 beta_close


def test_process_features_unsupported_type(feature_manager):
    """
    验证不支持的数据类型会抛出 ValueError。
    """
    with pytest.raises(ValueError):
        feature_manager.process_features(data=42)  # type: ignore[arg-type]


def test_cleanup_cache(feature_manager):
    """
    验证缓存清理逻辑会在 TTL 过期后清空缓存并更新时间戳。
    """
    feature_manager._feature_cache = {"foo": pd.DataFrame({"a": [1, 2, 3]})}
    feature_manager._last_cache_cleanup = 0.0
    feature_manager.config.cache_ttl = 1

    feature_manager.cleanup_cache()

    assert feature_manager._feature_cache == {}
    assert feature_manager._last_cache_cleanup > 0


def test_get_status_contains_expected_keys(feature_manager):
    """
    验证 get_status 返回的结构包含关键信息。
    """
    status = feature_manager.get_status()

    assert "config" in status
    assert "cache_size" in status
    assert "components_initialized" in status
    assert status["components_initialized"] is True


def test_validate_features_reports_flags(feature_manager, sample_price_frame):
    """
    验证 validate_features 返回的布尔结果与输入特征匹配。
    """
    report = feature_manager.validate_features(sample_price_frame)

    assert report["has_features"] is True
    assert report["has_samples"] is True
    assert report["no_duplicate_columns"] is True
    assert report["no_all_null_columns"] is True


def test_update_config_replaces_existing(feature_manager):
    """
    验证 update_config 会直接替换现有配置。
    """
    new_config = FeatureConfig(feature_types=[FeatureType.SENTIMENT], enable_feature_selection=False)

    feature_manager.update_config(new_config)

    assert feature_manager.config is new_config
    assert feature_manager.config.feature_types == [FeatureType.SENTIMENT]


def test_save_and_load_features(feature_manager, tmp_path, sample_price_frame):
    """
    验证 save_features/load_features 会代理到 FeatureSaver。
    """
    feature_manager.save_features(sample_price_frame, tmp_path / "out.parquet")
    feature_manager.feature_saver.save_features.assert_called_once()

    feature_manager.load_features(tmp_path / "out.parquet")
    feature_manager.feature_saver.load_features.assert_called_once()


def test_process_features_with_sentiment(feature_manager, sample_price_frame):
    """
    验证在启用情感特征时，处理流程会同时调用情感与技术特征方法。
    """
    feature_manager.config.feature_types = [FeatureType.TECHNICAL, FeatureType.SENTIMENT]
    feature_manager.feature_engineer.generate_technical_features.return_value = sample_price_frame[["close"]]
    sentiment = pd.DataFrame({"sentiment_score": [0.1] * len(sample_price_frame)}, index=sample_price_frame.index)
    feature_manager.feature_engineer.generate_sentiment_features.return_value = sentiment

    result = feature_manager.process_features(sample_price_frame)

    assert "close" in result.columns
    assert "sentiment_score" in result.columns
    feature_manager.feature_engineer.generate_sentiment_features.assert_called_once()


def test_process_streaming_returns_empty(feature_manager):
    """
    验证流式接口当前返回空 DataFrame 占位结果。
    """
    result = feature_manager.process_streaming(data_stream=object())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_run_pipeline_handles_exception(feature_manager):
    """
    验证 run 方法在下游抛异常时能够捕获并返回错误信息。
    """
    feature_manager.feature_engineer.generate_technical_features.side_effect = RuntimeError("boom")

    response = feature_manager.run("mock_source")

    assert response["status"] == "error"
    assert response["error"] == "boom"


def test_init_applies_processing_config(monkeypatch):
    """
    验证初始化时会读取 Processing 配置并覆盖 FeatureConfig。
    """
    fake_manager = MagicMock()
    fake_manager.get_config.return_value = {
        "max_workers": 3,
        "batch_size": 256,
        "timeout": 123,
        "feature_selection_method": "kbest",
        "max_features": 20,
        "min_feature_importance": 0.2,
        "standardization_method": "robust",
        "robust_scaling": True,
    }
    fake_manager.register_config_watcher = MagicMock()
    monkeypatch.setattr(
        "src.features.core.feature_manager.get_config_integration_manager",
        lambda: fake_manager,
    )

    manager = FeatureManager()

    assert manager.config.max_workers == 3
    assert manager.config.batch_size == 256
    assert manager.config.timeout == 123
    assert manager.config.feature_selection_method == "kbest"
    assert manager.config.max_features == 20
    assert manager.config.min_feature_importance == 0.2
    assert manager.config.standardization_method == "robust"
    assert manager.config.robust_scaling is True


def test_init_processing_config_partial_skips_missing_keys(monkeypatch):
    """
    处理配置只包含部分键时，仅更新提供的字段，其余保持默认。
    """

    class PartialConfigManager:
        def __init__(self):
            self._watcher = None

        def get_config(self, scope):
            assert scope == ConfigScope.PROCESSING
            return {"max_workers": 2}  # truthy 字典，但缺少其他键，触发 false 分支

        def register_config_watcher(self, scope, callback):
            self._watcher = (scope, callback)

    monkeypatch.setattr(
        "src.features.core.feature_manager.get_config_integration_manager",
        lambda: PartialConfigManager(),
    )

    manager = FeatureManager()

    assert manager.config.max_workers == 2
    assert not hasattr(manager.config, "batch_size")
    assert not hasattr(manager.config, "timeout")


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_workers", 5),
        ("batch_size", 64),
        ("timeout", 999),
        ("feature_selection_method", "variance"),
        ("max_features", 15),
        ("min_feature_importance", 0.05),
        ("standardization_method", "minmax"),
        ("robust_scaling", False),
    ],
)
def test_on_config_change_updates_attributes(feature_manager, key, value):
    """
    验证配置监听回调能够正确更新 FeatureConfig。
    """
    original = getattr(feature_manager.config, key, None)

    feature_manager._on_config_change(ConfigScope.PROCESSING, key, original, value)

    assert getattr(feature_manager.config, key) == value


def test_process_multiple_dataframes_all_empty_returns_empty(feature_manager):
    """
    多个数据集都为空时应返回空 DataFrame。
    """
    feature_manager.feature_engineer.generate_technical_features.return_value = pd.DataFrame()

    result = feature_manager.process_features({"alpha": pd.DataFrame()}, feature_types=[FeatureType.TECHNICAL])

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_process_single_dataframe_logs_unsupported_types(feature_manager, caplog, sample_price_frame):
    """
    包含未实现的特征类型时应记录告警，但不中断执行。
    """
    feature_manager.config.feature_types = [
        FeatureType.ORDERBOOK,
        FeatureType.FUNDAMENTAL,
        FeatureType.TECHNICAL,
    ]
    feature_manager.feature_engineer.generate_technical_features.return_value = sample_price_frame[["close"]]

    with caplog.at_level("WARNING"):
        result = feature_manager.process_features(sample_price_frame)

    assert "Orderbook特征处理暂未实现" in caplog.text
    assert "Fundamental特征处理暂未实现" in caplog.text
    assert not result.empty


def test_process_single_dataframe_all_custom_returns_empty(feature_manager, caplog, sample_price_frame):
    """
    仅包含未支持的特征类型时，应记录告警并返回空结果。
    """
    feature_manager.config.feature_types = [FeatureType.CUSTOM]

    with caplog.at_level("WARNING"):
        result = feature_manager.process_features(sample_price_frame)

    assert "未支持的特征类型" in caplog.text
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    feature_manager.feature_engineer.generate_technical_features.assert_not_called()


def test_cleanup_cache_without_expiry_keeps_cache(feature_manager):
    """
    当 TTL 尚未到期时不应清空缓存。
    """
    feature_manager._feature_cache = {"foo": pd.DataFrame({"a": [1]})}
    feature_manager._last_cache_cleanup = time.time()
    feature_manager.config.cache_ttl = 3600

    feature_manager.cleanup_cache()

    assert feature_manager._feature_cache


def test_run_pipeline_success(feature_manager, sample_price_frame):
    """
    验证 run 流程成功分支会返回结果与元数据。
    """
    feature_manager.feature_engineer.generate_technical_features.return_value = sample_price_frame[["close"]]
    sentiment = pd.DataFrame({"sent": [0.1] * len(sample_price_frame)}, index=sample_price_frame.index)
    feature_manager.feature_engineer.generate_sentiment_features.return_value = sentiment
    feature_manager.feature_engineer.merge_features.return_value = sample_price_frame.assign(sent=sentiment["sent"])
    feature_manager.feature_engineer.save_metadata.return_value = None
    feature_manager.feature_engineer.load_metadata.return_value = {"rows": len(sample_price_frame)}

    response = feature_manager.run("src")

    assert response["status"] == "success"
    assert response["result"] is not None
    assert response["metadata"] == {"rows": len(sample_price_frame)}


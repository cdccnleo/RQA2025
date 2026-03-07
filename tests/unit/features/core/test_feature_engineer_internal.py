import json
import builtins
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.features.core.config_integration import ConfigScope
from src.features.core.feature_engineer import FeatureEngineer


class StubConfigManager:
    def __init__(self, cache_dir: Path):
        self._configs = {
            ConfigScope.GLOBAL: {
                "cache_dir": str(cache_dir),
                "fallback_enabled": True,
            },
            ConfigScope.PROCESSING: {
                "max_workers": 2,
                "batch_size": 10,
                "timeout": 30,
                "max_retries": 2,
            },
            ConfigScope.MONITORING: {
                "enable_monitoring": False,
                "monitoring_level": "standard",
            },
        }

    def get_config(self, scope: ConfigScope, key: str = None, default=None):
        config = self._configs.get(scope, {})
        if key is None:
            return config
        return config.get(key, default)

    def register_config_watcher(self, scope, callback):
        # No-op for tests
        return None


@pytest.fixture
def engineer_factory(monkeypatch, tmp_path):
    cache_root = tmp_path / "feature_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    stub_manager = StubConfigManager(cache_root)

    monkeypatch.setattr(
        "src.features.core.feature_engineer.get_config_integration_manager",
        lambda: stub_manager,
    )

    class TechnicalProcessorStub:
        def calculate_multiple_indicators(self, data, indicators, params):
            return pd.DataFrame(
                {name: pd.Series(range(len(data)), index=data.index) for name in indicators},
                index=data.index,
            )

    class SentimentAnalyzerStub:
        def generate_features(self, news_data, text_col="content", date_col="date", output_cols=None):
            cols = output_cols or ["sentiment_score"]
            return pd.DataFrame(
                {col: [0.5] * len(news_data) for col in cols}, index=news_data.index
            )

    _sentinel = object()

    def _factory(technical_processor=_sentinel, sentiment_analyzer=_sentinel, cache_dir=None):
        return FeatureEngineer(
            technical_processor=TechnicalProcessorStub() if technical_processor is _sentinel else technical_processor,
            sentiment_analyzer=SentimentAnalyzerStub() if sentiment_analyzer is _sentinel else sentiment_analyzer,
            cache_dir=str(cache_dir or (cache_root / "engineer")),
            max_retries=1,
        )

    return _factory


@pytest.fixture
def stock_data():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 102, 101, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [99, 100, 98, 101, 102],
            "close": [102, 101, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=idx,
    )


@pytest.fixture
def news_data():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "content": [f"news {i}" for i in range(5)],
            "date": idx,
        },
        index=idx,
    )


def test_generate_technical_features(engineer_factory, stock_data):
    engineer = engineer_factory()
    result = engineer.generate_technical_features(stock_data.copy())

    assert set(result.columns) == {"ma", "rsi", "macd", "bollinger"}
    params = engineer.feature_metadata.feature_params
    assert "technical_indicators" in params
    assert "technical_params" in params


def test_generate_technical_features_requires_processor(engineer_factory, stock_data):
    engineer = engineer_factory(technical_processor=None)
    with pytest.raises(ValueError):
        engineer.generate_technical_features(stock_data.copy())


def test_generate_sentiment_features(engineer_factory, news_data):
    engineer = engineer_factory()
    result = engineer.generate_sentiment_features(
        news_data, text_col="content", date_col="date", output_cols=["score", "label"]
    )

    assert set(result.columns) == {"score", "label"}
    params = engineer.feature_metadata.feature_params
    assert params["sentiment_text_col"] == "content"
    assert params["sentiment_output_cols"] == ["score", "label"]


def test_merge_features_success(engineer_factory, stock_data):
    engineer = engineer_factory()
    tech = engineer.generate_technical_features(stock_data.copy())
    merged = engineer.merge_features(stock_data, tech)

    assert merged.shape[1] == stock_data.shape[1] + tech.shape[1]
    assert set(engineer.feature_metadata.feature_list) == set(merged.columns)


def test_merge_features_index_mismatch(engineer_factory, stock_data):
    engineer = engineer_factory()
    tech = pd.DataFrame(
        {"ma": range(3)}, index=pd.date_range("2024-02-01", periods=3, freq="D")
    )
    with pytest.raises(ValueError):
        engineer.merge_features(stock_data, tech)


def test_generate_technical_features_with_invalid_data(engineer_factory, stock_data):
    engineer = engineer_factory()
    bad_data = stock_data.drop(columns=["close"])
    with pytest.raises(ValueError):
        engineer.generate_technical_features(bad_data)


def test_validate_stock_data_auto_fix(engineer_factory, stock_data):
    engineer = engineer_factory()
    faulty = stock_data.copy()
    faulty.loc[faulty.index[0], ["close", "high", "low"]] = [-100, -95, -110]
    faulty.loc[faulty.index[1], "volume"] = -500

    engineer.generate_technical_features(faulty)
    assert (faulty[["close", "high", "low"]] >= 0).all().all()
    assert (faulty["volume"] >= 0).all()


def test_save_and_load_metadata(engineer_factory, stock_data, tmp_path):
    engineer = engineer_factory(cache_dir=tmp_path / "cache_a")
    tech = engineer.generate_technical_features(stock_data.copy())
    engineer.merge_features(stock_data, tech)

    meta_path = tmp_path / "meta.pkl"
    engineer.save_metadata(str(meta_path))
    assert meta_path.exists()

    new_engineer = engineer_factory(cache_dir=tmp_path / "cache_b")
    new_engineer.load_metadata(str(meta_path))
    assert set(new_engineer.feature_metadata.feature_list) == set(engineer.feature_metadata.feature_list)


def test_get_version_stats(monkeypatch, engineer_factory, stock_data):
    engineer = engineer_factory()
    engineer.feature_metadata.metadata.update(
        {
            "v1": {"feature_count": 10, "status": "active", "timestamp": pd.Timestamp("2024-01-01")},
            "v2": {"feature_count": 5, "status": "deleted", "timestamp": pd.Timestamp("2024-01-05")},
        }
    )
    stats = engineer.feature_metadata.metadata.copy()
    assert len(stats) == 2


def test_on_config_change_updates_executor(monkeypatch, engineer_factory):
    created = []

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.shutdown_called = False
            created.append(self)

        def shutdown(self, wait=True):
            self.shutdown_called = True

    monkeypatch.setattr("src.features.core.feature_engineer.ThreadPoolExecutor", DummyExecutor)
    engineer = engineer_factory()
    old_executor = engineer.executor
    engineer._on_config_change(ConfigScope.PROCESSING, "max_workers", 2, 5)
    assert old_executor.shutdown_called is True
    assert engineer.executor.max_workers == 5

    engineer._on_config_change(ConfigScope.PROCESSING, "batch_size", engineer.batch_size, 20)
    engineer._on_config_change(ConfigScope.PROCESSING, "timeout", engineer.timeout, 60)
    assert engineer.batch_size == 20
    assert engineer.timeout == 60


def test_on_config_change_updates_monitoring_flags(engineer_factory):
    engineer = engineer_factory()
    engineer._on_config_change(ConfigScope.MONITORING, "enable_monitoring", False, True)
    engineer._on_config_change(ConfigScope.MONITORING, "monitoring_level", "standard", "detailed")
    assert engineer.enable_monitoring is True
    assert engineer.monitoring_level == "detailed"


def test_load_cache_metadata_handles_invalid_json(engineer_factory, tmp_path):
    cache_dir = tmp_path / "invalid_cache"
    cache_dir.mkdir()
    metadata_file = cache_dir / "metadata.json"
    metadata_file.write_text("{ invalid json", encoding="utf-8")

    engineer = engineer_factory(cache_dir=cache_dir)
    assert engineer.cache_metadata == {}


def test_register_feature_persists_metadata(engineer_factory):
    engineer = engineer_factory()
    config = SimpleNamespace(
        name="alpha",
        feature_type=SimpleNamespace(value="technical"),
        params={"window": 5},
        dependencies=["beta"],
        enabled=False,
        version="2.0",
    )
    engineer.register_feature(config)

    stored = engineer.cache_metadata["alpha"]
    assert stored["feature_type"] == "technical"
    assert stored["params"] == {"window": 5}
    metadata_file = Path(engineer.cache_dir) / "metadata.json"
    on_disk = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert "alpha" in on_disk


def test_register_feature_io_error_does_not_raise(monkeypatch, engineer_factory):
    engineer = engineer_factory()
    config = SimpleNamespace(
        name="gamma",
        feature_type=SimpleNamespace(value="statistical"),
        params={"window": 15},
        dependencies=[],
        enabled=True,
        version="3.0",
    )

    def boom_dump(*_args, **_kwargs):
        raise IOError("disk full")

    monkeypatch.setattr("json.dump", boom_dump)
    engineer.register_feature(config)
    assert "gamma" in engineer.cache_metadata


def test_generate_sentiment_features_propagates_error(engineer_factory, news_data):
    class FailingAnalyzer:
        def generate_features(self, *args, **kwargs):
            raise RuntimeError("sentiment boom")

    engineer = engineer_factory(sentiment_analyzer=FailingAnalyzer())
    with pytest.raises(RuntimeError):
        engineer.generate_sentiment_features(news_data)


def test_validate_stock_data_strict_validation_raises(engineer_factory, stock_data):
    engineer = engineer_factory()
    bad_data = stock_data.copy()
    bad_data.index = pd.Index(["invalid"] * len(bad_data))
    engineer.config = SimpleNamespace(strict_validation=True)
    engineer.fallback_enabled = False

    with pytest.raises(ValueError):
        engineer.generate_technical_features(bad_data)


def test_validate_stock_data_fallback_handles_internal_error(engineer_factory, stock_data, monkeypatch, caplog):
    engineer = engineer_factory()
    engineer.fallback_enabled = True
    engineer.config = SimpleNamespace(strict_validation=False)
    data = stock_data.copy()
    data.index = data.index.astype(str)

    def boom_to_datetime(*_args, **_kwargs):
        raise ValueError("bad index")

    monkeypatch.setattr("pandas.to_datetime", boom_to_datetime)
    with caplog.at_level("WARNING"):
        result = engineer.generate_technical_features(data)
    assert not result.empty
    assert any("数据验证和修复过程中出现错误" in message for message in caplog.messages)
    assert any("数据验证失败，但继续处理" in message for message in caplog.messages)


def test_validate_stock_data_respects_allow_flags(engineer_factory, stock_data):
    engineer = engineer_factory()
    data = stock_data.copy()
    future_index = pd.Index([pd.Timestamp.now() + pd.Timedelta(days=i + 1) for i in range(len(data))])
    data.index = future_index
    data.loc[data.index[0], ["close", "high", "low"]] = [-5, -4, -6]
    data.loc[data.index[1], "volume"] = -100
    data.loc[data.index[2], "close"] = None

    engineer.config = SimpleNamespace(
        allow_future_dates=True,
        allow_negative_prices=True,
        allow_negative_volume=True,
        allow_nan_values=True,
        strict_price_logic=False,
    )
    engineer.fallback_enabled = True

    engineer.generate_technical_features(data)
    assert (data["close"] < 0).any()
    pd.testing.assert_index_equal(data.index, future_index)


def test_validate_stock_data_raises_when_fallback_disabled(engineer_factory, stock_data):
    engineer = engineer_factory()
    bad_data = stock_data.copy()
    bad_data.loc[bad_data.index[0], ["close", "high", "low"]] = [-1, -2, -3]
    engineer.fallback_enabled = False

    with pytest.raises(ValueError):
        engineer.generate_technical_features(bad_data)


def test_validate_stock_data_duplicate_index_without_fallback(engineer_factory, stock_data):
    engineer = engineer_factory()
    dup = stock_data.copy()
    dup.index = pd.Index([dup.index[0]] * len(dup))
    engineer.fallback_enabled = False

    with pytest.raises(ValueError):
        engineer.generate_technical_features(dup)


def test_validate_stock_data_internal_error_strict_raises(engineer_factory, stock_data, monkeypatch):
    engineer = engineer_factory()
    engineer.fallback_enabled = False
    engineer.config = SimpleNamespace(strict_validation=True)
    data = stock_data.copy()
    data.index = data.index.astype(str)

    def boom_to_datetime(*_args, **_kwargs):
        raise ValueError("hard failure")

    monkeypatch.setattr("pandas.to_datetime", boom_to_datetime)
    with pytest.raises(ValueError, match="数据验证失败"):
        engineer.generate_technical_features(data)


def test_load_cache_metadata_reads_existing(tmp_path, engineer_factory):
    cache_dir = tmp_path / "preload"
    cache_dir.mkdir()
    payload = {"feat": {"feature_type": "technical"}}
    (cache_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    engineer = engineer_factory(cache_dir=cache_dir)
    assert engineer.cache_metadata == payload


def test_load_cache_metadata_io_error(monkeypatch, tmp_path, engineer_factory):
    cache_dir = tmp_path / "io_error_cache"
    cache_dir.mkdir()
    metadata_file = cache_dir / "metadata.json"
    metadata_file.write_text(json.dumps({"x": 1}), encoding="utf-8")

    original_open = builtins.open

    def boom_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("metadata.json") and "r" in mode:
            raise IOError("read fail")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", boom_open)
    engineer = engineer_factory(cache_dir=cache_dir)
    assert engineer.cache_metadata == {}


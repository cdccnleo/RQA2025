import numpy as np
import pandas as pd
import pytest
from typing import Dict

from src.features.core import optimized_feature_manager as optimized
from src.features.core.config import FeatureRegistrationConfig, FeatureType


class StubFeatureEngineer:
    def generate_technical_features(self, data, names, params):
        return pd.DataFrame({f"{names[0]}_tech": data["close"] * params.get("scale", 1)})

    def generate_sentiment_features(self, data, text_col="content", date_col="date"):
        return pd.DataFrame({"sentiment_score": np.linspace(0, 1, len(data))})


class StubParallelFeatureProcessor:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.calls = 0
        self.closed = False

    def process_features_parallel(self, data, feature_configs):
        self.calls += 1
        return data.assign(parallel_gen=data["close"] + 1)

    def close(self):
        self.closed = True


class StubQualityAssessor:
    def __init__(self, config=None):
        self.config = config
        self.last_filter_threshold = None

    def assess_feature_quality(self, features, target):
        return {"columns": list(features.columns)}

    def filter_features(self, features, min_quality_score):
        self.last_filter_threshold = min_quality_score
        keep = [col for col in features.columns if not col.endswith("_drop")]
        return features[keep]

    def get_quality_report(self):
        return {"status": "ok"}


class StubFeatureStore:
    def __init__(self, config=None):
        self.config = config
        self.saved: Dict[str, tuple[pd.DataFrame, dict]] = {}
        self.cleanup_calls = 0
        self.closed = False

    def load_feature(self, key):
        return self.saved.get(key)

    def store_feature(self, key, frame, config, description=None, tags=None):
        self.saved[key] = (frame.copy(), {"description": description, "tags": tags})
        return True

    def get_store_stats(self):
        return {"items": len(self.saved)}

    def cleanup_expired(self):
        self.cleanup_calls += 1
        return self.cleanup_calls

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def patch_components(monkeypatch):
    monkeypatch.setattr(optimized, "FeatureEngineer", StubFeatureEngineer)
    monkeypatch.setattr(optimized, "ParallelFeatureProcessor", StubParallelFeatureProcessor)
    monkeypatch.setattr(optimized, "FeatureQualityAssessor", StubQualityAssessor)
    monkeypatch.setattr(optimized, "FeatureStore", StubFeatureStore)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "open": np.linspace(10, 12, 10),
            "high": np.linspace(11, 13, 10),
            "low": np.linspace(9, 11, 10),
            "close": np.linspace(10, 12, 10),
            "volume": np.arange(100, 110),
        }
    )


@pytest.fixture
def feature_configs():
    return [
        FeatureRegistrationConfig(
            name="tech_feature",
            feature_type=FeatureType.TECHNICAL,
            params={"scale": 2},
            dependencies=[],
        )
    ]


def test_generate_features_optimized_returns_cached(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    cached = pd.DataFrame({"cached": np.arange(len(sample_data))})

    monkeypatch.setattr(manager, "_load_from_cache", lambda *_: cached)
    monkeypatch.setattr(manager, "_save_to_cache", lambda *args, **kwargs: pytest.fail("should not save"))

    result = manager.generate_features_optimized(sample_data, feature_configs)

    pd.testing.assert_frame_equal(result, cached)
    assert manager.stats["cache_hits"] == 1
    assert manager.stats["cache_misses"] == 0


def test_generate_features_parallel_and_quality(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    manager.feature_store.saved.clear()

    target = pd.Series(np.linspace(0, 1, len(sample_data)))
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)
    saved = {}
    monkeypatch.setattr(manager, "_save_to_cache", lambda features, configs, original=None: saved.setdefault("called", True))

    result = manager.generate_features_optimized(
        sample_data,
        feature_configs,
        target=target,
        assess_quality=True,
        use_cache=True,
        min_quality_score=0.7,
    )

    assert "parallel_gen" in result.columns
    assert manager.parallel_processor.calls == 1
    assert manager.stats["quality_assessed"] == 1
    assert saved.get("called") is True


def test_generate_features_serial_when_parallel_disabled(sample_data, feature_configs, monkeypatch):
    config = optimized.OptimizedManagerConfig(enable_parallel=False)
    manager = optimized.OptimizedFeatureManager(config)
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)

    result = manager.generate_features_optimized(
        sample_data,
        feature_configs,
        assess_quality=False,
        use_cache=False,
    )

    assert any(col.endswith("_tech") for col in result.columns)
    assert manager.parallel_processor is None


def test_batch_generate_features_handles_failures(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()

    def fake_generate(data, configs, target=None, use_cache=True, assess_quality=False, min_quality_score=0.6):
        if "fail_flag" in data.columns:
            raise ValueError("boom")
        return data.assign(success=True)

    monkeypatch.setattr(manager, "generate_features_optimized", fake_generate)

    success_data = sample_data.copy()
    fail_data = sample_data.copy()
    fail_data["fail_flag"] = 1

    results = manager.batch_generate_features(
        {"AAA": success_data, "BBB": fail_data},
        feature_configs,
        targets=None,
        use_cache=True,
    )

    assert "success" in results["AAA"].columns
    pd.testing.assert_frame_equal(results["BBB"], fail_data)


def test_get_stats_and_cleanup(sample_data, feature_configs):
    manager = optimized.OptimizedFeatureManager()
    manager.stats = {
        "total_processed": 2,
        "cache_hits": 1,
        "cache_misses": 1,
        "parallel_processed": 1,
        "quality_assessed": 1,
        "total_time": 4.0,
    }

    stats = manager.get_performance_stats()
    assert stats["avg_processing_time"] == 2.0
    assert stats["cache_hit_rate"] == 0.5

    assert manager.cleanup_cache() == 1
    assert manager.feature_store.cleanup_calls == 1


def test_close_releases_resources():
    manager = optimized.OptimizedFeatureManager()
    manager.close()
    assert manager.parallel_processor.closed is True
    assert manager.feature_store.closed is True


def test_load_from_cache_invalidates_bad_data(sample_data, feature_configs):
    manager = optimized.OptimizedFeatureManager()
    cache_key = manager._generate_cache_key(sample_data, feature_configs)
    bad_frame = sample_data.drop(columns=["close"])
    manager.feature_store.saved[cache_key] = (bad_frame, {"description": "bad"})

    cached = manager._load_from_cache(sample_data, feature_configs)
    assert cached is None


def test_save_to_cache_handles_store_failure(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    monkeypatch.setattr(manager.feature_store, "store_feature", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))

    manager._save_to_cache(sample_data.assign(new=1), feature_configs, sample_data)
    # should not raise


def test_generate_features_serial_generic(sample_data):
    manager = optimized.OptimizedFeatureManager()
    configs = [
        FeatureRegistrationConfig(
            name="custom_feature",
            feature_type=FeatureType.CUSTOM,
            params={},
            dependencies=[],
        )
    ]
    result = manager._generate_features_serial(sample_data, configs)
    assert any(col.endswith("volume_mean") for col in result.columns)
    assert any(col.endswith("returns") for col in result.columns)


def test_assess_and_filter_handles_exception(sample_data, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    features = sample_data.assign(new_feature=sample_data["close"] * 2)
    target = pd.Series(np.linspace(0, 1, len(sample_data)))

    def boom(*_args, **_kwargs):
        raise RuntimeError("quality fail")

    monkeypatch.setattr(manager.quality_assessor, "assess_feature_quality", boom)

    filtered = manager._assess_and_filter_features(features, target, 0.5)
    pd.testing.assert_frame_equal(filtered, features)


def test_generate_features_optimized_handles_failures(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)

    def boom(*_args, **_kwargs):
        raise RuntimeError("parallel failed")

    if manager.parallel_processor:
        monkeypatch.setattr(manager.parallel_processor, "process_features_parallel", boom)
    else:
        monkeypatch.setattr(manager, "_generate_features_serial", boom)

    result = manager.generate_features_optimized(sample_data, feature_configs, use_cache=False)
    assert result.empty


def test_generate_features_skips_quality_without_target(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)
    result = manager.generate_features_optimized(sample_data, feature_configs, target=None, assess_quality=True)
    assert manager.stats["quality_assessed"] == 0
    assert not result.empty


def test_generate_features_without_quality_component(sample_data, feature_configs, monkeypatch):
    config = optimized.OptimizedManagerConfig(enable_quality_assessment=False)
    manager = optimized.OptimizedFeatureManager(config)
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)
    result = manager.generate_features_optimized(sample_data, feature_configs, assess_quality=True)
    assert manager.quality_assessor is None
    assert manager.stats["quality_assessed"] == 0
    assert not result.empty


def test_generate_generic_features_missing_columns(sample_data):
    manager = optimized.OptimizedFeatureManager()
    config = FeatureRegistrationConfig(
        name="generic",
        feature_type=FeatureType.CUSTOM,
        params={},
        dependencies=[],
    )
    stripped = sample_data[["open"]]
    features = manager._generate_generic_features(stripped, config)
    assert features.empty


def test_generate_features_cache_miss_updates_stats_and_cache(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()
    target = pd.Series(np.linspace(0, 1, len(sample_data)))

    load_calls = {"count": 0}
    monkeypatch.setattr(
        manager,
        "_load_from_cache",
        lambda data, configs: load_calls.__setitem__("count", load_calls["count"] + 1) or None,
    )

    saved = {}

    def fake_save(features, configs, original=None):
        saved["features"] = features.copy()
        saved["original_shape"] = original.shape if original is not None else None

    monkeypatch.setattr(manager, "_save_to_cache", fake_save)

    result = manager.generate_features_optimized(
        sample_data,
        feature_configs,
        target=target,
        use_cache=True,
        assess_quality=True,
        min_quality_score=0.5,
    )

    assert load_calls["count"] == 1
    assert saved["original_shape"] == sample_data.shape
    assert manager.stats["cache_hits"] == 0
    assert manager.stats["cache_misses"] == 1
    assert manager.stats["parallel_processed"] == 1
    assert manager.stats["quality_assessed"] == 1
    assert not result.empty


def test_generate_features_without_caching_or_quality(sample_data, feature_configs):
    config = optimized.OptimizedManagerConfig(
        enable_parallel=False,
        enable_quality_assessment=False,
        enable_caching=False,
    )
    manager = optimized.OptimizedFeatureManager(config)

    result = manager.generate_features_optimized(
        sample_data,
        feature_configs,
        use_cache=True,
        assess_quality=True,
    )

    assert manager.feature_store is None
    assert manager.stats["cache_hits"] == 0
    assert manager.stats["cache_misses"] == 1  # 缓存路径被跳过时仍会记录一次 miss
    assert manager.stats["quality_assessed"] == 0
    assert any(col.endswith("_tech") for col in result.columns)


def test_load_from_cache_returns_valid_frame(sample_data, feature_configs):
    manager = optimized.OptimizedFeatureManager()
    cache_key = manager._generate_cache_key(sample_data, feature_configs)
    cached_frame = sample_data.assign(cached_feature=np.arange(len(sample_data)))
    manager.feature_store.saved[cache_key] = (cached_frame, {"description": "valid"})

    loaded = manager._load_from_cache(sample_data, feature_configs)
    pd.testing.assert_frame_equal(loaded, cached_frame)


def test_quality_filter_preserves_original_columns(sample_data, feature_configs, monkeypatch):
    manager = optimized.OptimizedFeatureManager()

    def fake_parallel(data, configs):
        return data.assign(
            good_feature=data["close"] * 2,
            noisy_drop=data["close"] * 0,
        )

    monkeypatch.setattr(manager.parallel_processor, "process_features_parallel", fake_parallel)
    monkeypatch.setattr(manager, "_load_from_cache", lambda *args, **kwargs: None)

    target = pd.Series(np.linspace(0, 1, len(sample_data)))
    result = manager.generate_features_optimized(
        sample_data,
        feature_configs,
        target=target,
        assess_quality=True,
        min_quality_score=0.4,
    )

    for base_col in ["open", "high", "low", "close", "volume"]:
        assert base_col in result.columns
    assert "good_feature" in result.columns
    assert "noisy_drop" not in result.columns


def test_reports_and_cache_stats(sample_data, feature_configs):
    manager = optimized.OptimizedFeatureManager()
    manager.feature_store.store_feature(
        "foo",
        sample_data,
        feature_configs[0],
        description="demo",
        tags=["cached"],
    )

    cache_stats = manager.get_cache_stats()
    assert cache_stats["items"] == 1

    quality_report = manager.get_feature_quality_report()
    assert quality_report["status"] == "ok"

    assert manager.cleanup_cache() == 1

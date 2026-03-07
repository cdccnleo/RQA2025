import builtins
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.features.core.feature_config import FeatureType
from src.features.core.manager import FeatureManager, FeatureMetadata


@pytest.fixture
def patched_manager(tmp_path, monkeypatch):
    """
    提供使用临时缓存目录的 FeatureManager，并修正模块内错误的编码写法。
    """
    real_open = builtins.open

    def patched_open(file, mode="r", *args, **kwargs):
        encoding = kwargs.get("encoding")
        if encoding == "utf - 8":
            kwargs["encoding"] = "utf-8"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", patched_open)

    mgr = FeatureManager()
    mgr.cache_dir = tmp_path
    mgr.cache_dir.mkdir(parents=True, exist_ok=True)
    mgr.features.clear()
    mgr.cache_stats = {"hits": 0, "misses": 0, "saves": 0, "deletes": 0}
    return mgr


def _make_metadata(name="alpha", version="1.0.0", enabled=True, dependencies=None):
    return FeatureMetadata(
        name=name,
        feature_type=FeatureType.TECHNICAL,
        description="demo feature",
        version=version,
        parameters={"window": 5},
        dependencies=dependencies or [],
        tags=["core"],
        enabled=enabled,
    )


def test_feature_metadata_roundtrip():
    now = datetime.now().replace(microsecond=0)
    metadata = FeatureMetadata(
        name="alpha",
        feature_type=FeatureType.CUSTOM,
        description="统计特征",
        version="2.1.0",
        parameters={"period": 10},
        dependencies=["beta"],
        tags=["tag1", "tag2"],
        enabled=False,
        created_at=now,
        updated_at=now + timedelta(minutes=5),
    )

    payload = metadata.to_dict()
    restored = FeatureMetadata.from_dict(payload)

    assert restored.name == "alpha"
    assert restored.feature_type == FeatureType.CUSTOM
    assert not restored.enabled
    assert restored.parameters["period"] == 10
    assert restored.dependencies == ["beta"]


def test_register_and_update_feature(patched_manager, caplog):
    caplog.set_level("INFO")
    meta = _make_metadata()
    assert patched_manager.register_feature(meta) is True
    # 同版本再次注册返回 False
    assert patched_manager.register_feature(meta) is False

    newer = _make_metadata(version="1.1.0")
    assert patched_manager.register_feature(newer) is True
    stored = patched_manager.get_feature("alpha")
    assert stored.version == "1.1.0"
    assert "更新特征" in "".join(caplog.messages)


def test_list_enable_disable_feature(patched_manager):
    meta1 = _make_metadata("alpha")
    meta2 = _make_metadata("beta", enabled=False)
    patched_manager.register_feature(meta1)
    patched_manager.register_feature(meta2)

    enabled_only = patched_manager.list_features()
    assert [item.name for item in enabled_only] == ["alpha"]

    patched_manager.disable_feature("alpha")
    assert patched_manager.get_feature("alpha").enabled is False

    all_features = patched_manager.list_features(enabled_only=False)
    assert sorted(item.name for item in all_features) == ["alpha", "beta"]

    patched_manager.enable_feature("beta")
    assert patched_manager.get_feature("beta").enabled is True


def test_unregister_feature_handles_missing(patched_manager):
    assert patched_manager.unregister_feature("ghost") is False

    meta = _make_metadata()
    patched_manager.register_feature(meta)
    assert patched_manager.unregister_feature("alpha") is True
    assert patched_manager.get_feature("alpha") is None


def test_cache_roundtrip_and_stats(patched_manager):
    data = pd.DataFrame({"close": [1, 2, 3, 4, 5], "volume": [10, 11, 12, 13, 14]})
    features = pd.DataFrame({"alpha": [0.1, 0.2, 0.3, 0.4, 0.5]})

    patched_manager.cache_features(data, ["alpha"], features)
    cached = patched_manager.get_cached_features(data, ["alpha"])
    pd.testing.assert_frame_equal(cached, features)

    # 触发一次未命中以统计 miss
    assert patched_manager.get_cached_features(data, ["beta"]) is None

    stats = patched_manager.get_cache_stats()
    assert stats["saves"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cache_files"] == 1


def test_clear_cache_specific_and_all(patched_manager):
    data = pd.DataFrame({"close": [1, 2, 3], "volume": [4, 5, 6]})
    patched_manager.cache_features(data, ["alpha"], pd.DataFrame({"alpha": [1, 2, 3]}))
    patched_manager.cache_features(data, ["beta"], pd.DataFrame({"beta": [3, 2, 1]}))

    removed_alpha = patched_manager.clear_cache("alpha")
    assert removed_alpha == 1
    remaining_files = list(patched_manager.cache_dir.glob("*.pkl"))
    assert len(remaining_files) == 1

    removed_all = patched_manager.clear_cache()
    assert removed_all == 1
    assert not list(patched_manager.cache_dir.glob("*.pkl"))


def test_dependencies_and_validation(patched_manager):
    parent = _make_metadata("parent", dependencies=["child"])
    child = _make_metadata("child")
    patched_manager.register_feature(parent)
    assert patched_manager.validate_feature_dependencies("parent") is False

    patched_manager.register_feature(child)
    assert patched_manager.validate_feature_dependencies("parent") is True
    assert patched_manager.get_feature_dependencies("parent") == ["child"]


def test_export_and_import_registry(tmp_path, patched_manager):
    meta1 = _make_metadata("alpha")
    meta2 = _make_metadata("beta")
    patched_manager.register_feature(meta1)
    patched_manager.register_feature(meta2)

    export_path = tmp_path / "registry.json"
    assert patched_manager.export_feature_registry(str(export_path)) is True
    assert export_path.exists()

    manager2 = FeatureManager()
    manager2.cache_dir = tmp_path / "alt_cache"
    manager2.cache_dir.mkdir(parents=True, exist_ok=True)
    manager2.features.clear()
    manager2.cache_stats = {"hits": 0, "misses": 0, "saves": 0, "deletes": 0}

    imported = manager2.import_feature_registry(str(export_path))
    assert imported == 2
    assert sorted(manager2.features.keys()) == ["alpha", "beta"]

    # 再次导入不覆盖，跳过已有记录
    imported_again = manager2.import_feature_registry(str(export_path), overwrite=False)
    assert imported_again == 0

    # 覆盖导入
    imported_overwrite = manager2.import_feature_registry(str(export_path), overwrite=True)
    assert imported_overwrite == 2


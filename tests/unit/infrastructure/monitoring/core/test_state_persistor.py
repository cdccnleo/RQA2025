import gzip
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.infrastructure.monitoring.core.state_persistor import ComponentState, StatePersistor


@pytest.fixture
def temp_storage_dir(tmp_path):
    return tmp_path / "states"


def test_component_state_checksum_validation():
    data = {"version": "1.0.0", "value": 42}
    state = ComponentState("demo", data)

    assert state.validate() is True

    state.state_data["value"] = 43
    assert state.validate() is False


def test_component_state_to_from_dict():
    data = {"version": "2.0.0", "value": 100}
    state = ComponentState("demo", data)
    state_dict = state.to_dict()

    restored = ComponentState.from_dict(state_dict)
    assert restored.component_name == "demo"
    assert restored.state_data == data


def test_state_persistor_save_and_load(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    assert persistor.save_component_state("demo", {"value": 1})

    loaded = persistor.load_component_state("demo")
    assert loaded == {"value": 1}


def test_state_persistor_load_from_cache(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    # This load will use cache
    loaded = persistor.load_component_state("demo")
    assert loaded == {"value": 1}


def test_state_persistor_delete_state(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    assert persistor.delete_component_state("demo") is True
    assert persistor.load_component_state("demo") is None


def test_state_persistor_delete_missing_file(temp_storage_dir, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    with caplog.at_level("DEBUG"):
        assert persistor.delete_component_state("ghost") is True
    assert "组件状态文件不存在" in caplog.text


def test_state_persistor_compressed_storage(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=True)
    assert persistor.save_component_state("demo", {"value": 1})

    file_path = temp_storage_dir / "demo.json.gz"
    assert file_path.exists()

    loaded = persistor.load_component_state("demo")
    assert loaded == {"value": 1}


def test_state_persistor_list_states(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    states = persistor.list_component_states()
    assert "demo" in states


def test_state_persistor_list_states_error(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    monkeypatch.setattr(os, "listdir", MagicMock(side_effect=RuntimeError("list fail")))

    with caplog.at_level("ERROR"):
        states = persistor.list_component_states()
    assert states == []
    assert "列出组件状态失败" in caplog.text


def test_state_persistor_backup_and_restore(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    assert persistor.backup_all_states("backup1") is True
    persistor.delete_component_state("demo")
    assert persistor.load_component_state("demo") is None

    assert persistor.restore_from_backup("backup1") is True
    assert persistor.load_component_state("demo") == {"value": 1}


def test_state_persistor_backup_missing(temp_storage_dir, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    result = persistor.restore_from_backup("missing")
    assert result is False
    assert "备份不存在" in caplog.text


def test_state_persistor_storage_stats(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})
    stats = persistor.get_storage_stats()

    assert stats["total_components"] == 1
    assert stats["total_files"] == 1


def test_state_persistor_optimize_storage(temp_storage_dir):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})
    result = persistor.optimize_storage()
    assert "files_cleaned" in result


def test_state_persistor_save_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    monkeypatch.setattr(
        "src.infrastructure.monitoring.core.state_persistor.ComponentState",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    with caplog.at_level("ERROR"):
        assert persistor.save_component_state("demo", {"value": 1}) is False
    assert "保存组件状态失败 demo" in caplog.text


def test_state_persistor_load_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    monkeypatch.setattr(
        persistor,
        "_load_state_from_file",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    with caplog.at_level("ERROR"):
        assert persistor.load_component_state("demo") is None
    assert "加载组件状态失败 demo" in caplog.text


def test_state_persistor_delete_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    monkeypatch.setattr(os, "remove", MagicMock(side_effect=RuntimeError("boom")))

    with caplog.at_level("ERROR"):
        assert persistor.delete_component_state("demo") is False
    assert "删除组件状态失败 demo" in caplog.text


def test_state_persistor_backup_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    monkeypatch.setattr(os, "makedirs", MagicMock(side_effect=RuntimeError("boom")))

    with caplog.at_level("ERROR"):
        assert persistor.backup_all_states("fail") is False
    assert "备份组件状态失败" in caplog.text


def test_state_persistor_restore_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    backup_dir = Path(persistor.storage_dir) / "backups" / "bk"
    backup_dir.mkdir(parents=True, exist_ok=True)
    file_path = backup_dir / "demo.json"
    file_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("builtins.open", MagicMock(side_effect=RuntimeError("boom")), raising=True)

    with caplog.at_level("ERROR"):
        assert persistor.restore_from_backup("bk") is False
    assert "从备份恢复失败" in caplog.text


def test_state_persistor_optimize_failure(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)

    monkeypatch.setattr(os, "listdir", MagicMock(side_effect=RuntimeError("boom")))

    with caplog.at_level("ERROR"):
        result = persistor.optimize_storage()
    assert result["error"] == "boom"


def test_state_persistor_load_invalid_state(temp_storage_dir, caplog, monkeypatch):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    file_path = Path(persistor.storage_dir) / "demo.json"
    data = json.loads(file_path.read_text(encoding="utf-8"))
    data["state_data"]["value"] = 2  # break checksum
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    persistor._state_cache.clear()
    monkeypatch.setattr("src.infrastructure.monitoring.core.state_persistor.ComponentState.validate", lambda self: False)

    with caplog.at_level("WARNING"):
        result = persistor.load_component_state("demo")

    assert result is None
    assert "组件状态校验失败" in caplog.text


def test_state_persistor_get_health_status_corruption(temp_storage_dir, monkeypatch):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    file_path = Path(persistor.storage_dir) / "demo.json"
    data = json.loads(file_path.read_text(encoding="utf-8"))
    data["state_data"]["value"] = 2
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr("src.infrastructure.monitoring.core.state_persistor.ComponentState.validate", lambda self: False)

    health = persistor.get_health_status()
    assert health["status"] == "warning"
    assert any("没有备份文件" in issue for issue in health["issues"])
    assert any("损坏的文件" in issue for issue in health["issues"])


def test_state_persistor_optimize_cleans_orphans_and_cache(temp_storage_dir, monkeypatch):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    persistor.save_component_state("demo", {"value": 1})

    orphan = Path(persistor.storage_dir) / "orphan.json"
    orphan.write_text("{}", encoding="utf-8")

    persistor._cache_dirty = True
    monkeypatch.setattr(persistor, "list_component_states", lambda: ["demo"])

    result = persistor.optimize_storage()
    assert result["files_cleaned"] >= 1
    assert persistor._cache_dirty is False


def test_state_persistor_optimize_storage_handles_os_walk_error(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    (Path(persistor.storage_dir) / "backups").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(os, "walk", MagicMock(side_effect=RuntimeError("walk fail")))

    with caplog.at_level("ERROR"):
        stats = persistor.get_storage_stats()
    assert stats["error"] == "walk fail"


def test_state_persistor_cleanup_old_backups_handles_error(temp_storage_dir, monkeypatch, caplog):
    persistor = StatePersistor(storage_dir=str(temp_storage_dir), enable_compression=False)
    backup_dir = Path(persistor.storage_dir) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "old").mkdir()
    persistor.max_backup_files = 0

    monkeypatch.setattr("shutil.rmtree", MagicMock(side_effect=RuntimeError("rm fail")))

    with caplog.at_level("ERROR"):
        cleaned = persistor._cleanup_old_backups()
    assert cleaned == 0
    assert "清理旧备份失败" in caplog.text


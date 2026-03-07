import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import io
from pathlib import Path

import pandas as pd
import pytest

from src.data.version_control.version_manager import DataVersionManager


def _make_df(rows=5):
    return pd.DataFrame({"a": list(range(rows)), "b": [x * 2 for x in range(rows)]})


def test_create_get_list_and_delete_version(tmp_path, monkeypatch):
    vm = DataVersionManager(version_dir=str(tmp_path / "versions"))

    # 使用内置 DataModel 路径：构造一个轻量 DataFrame
    df1 = _make_df(3)
    # 构造简易 DataModel（version_manager 自带兜底）
    class _DM:
        def __init__(self, data, metadata=None):
            self.data = data
            self._user_metadata = metadata or {}
            self._metadata = dict(self._user_metadata)
        def get_metadata(self, user_only=False):
            return self._user_metadata if user_only else self._metadata
    dm1 = _DM(df1, {"m": 1})

    v1 = vm.create_version(dm1, description="first", tags=["t1"], creator="u1", branch="main")
    assert isinstance(v1, str)

    got = vm.get_version(v1)
    assert got is not None

    listed = vm.list_versions()
    assert any(v.get("version_id") == v1 for v in listed)

    # 导出文件
    export_to = tmp_path / "out.parquet"
    assert vm.export_version(v1, export_to) is True
    assert export_to.exists()

    # 删除非当前版本：先创建第二个版本使 v1 不为 current
    dm2 = _DM(_make_df(4), {"m": 2})
    v2 = vm.create_version(dm2, description="second", tags=["t2"], creator="u1", branch="main")
    assert v2 != v1

    assert vm.delete_version(v1) is True


def test_compare_versions_and_rollback(tmp_path):
    vm = DataVersionManager(version_dir=str(tmp_path / "versions"))

    class _DM:
        def __init__(self, data, metadata=None):
            self.data = data
            self._user_metadata = metadata or {}
            self._metadata = dict(self._user_metadata)
        def get_metadata(self, user_only=False):
            return self._user_metadata if user_only else self._metadata

    v1 = vm.create_version(_DM(_make_df(2), {"x": 1}), description="v1")
    v2 = vm.create_version(_DM(_make_df(3), {"x": 2}), description="v2")

    diff = vm.compare_versions(v1, v2)
    assert "metadata_diff" in diff and "data_diff" in diff

    rolled = vm.rollback_to_version(v1)
    assert rolled is not None



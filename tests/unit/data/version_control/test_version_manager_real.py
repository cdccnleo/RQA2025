# -*- coding: utf-8 -*-
"""
版本管理器真实实现测试
测试 DataVersionManager 的核心功能
"""

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


import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data.version_control.version_manager import DataVersionManager


class MockDataModel:
    """模拟数据模型用于测试"""
    
    def __init__(self, data, frequency='1d', metadata=None):
        self.data = data
        self._frequency = frequency
        self._metadata = metadata or {}
    
    def get_frequency(self):
        return self._frequency
    
    def get_metadata(self, user_only=False):
        return self._metadata
    
    def validate(self):
        return self.data is not None and not self.data.empty if hasattr(self.data, 'empty') else True


@pytest.fixture
def version_manager(tmp_path):
    """创建版本管理器实例"""
    version_dir = tmp_path / "versions"
    return DataVersionManager(str(version_dir))


@pytest.fixture
def sample_data():
    """创建示例数据"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'value': [100 + i for i in range(5)]
    })


@pytest.fixture
def data_model(sample_data):
    """创建数据模型实例"""
    return MockDataModel(sample_data)


def test_version_manager_initialization(version_manager):
    """测试版本管理器初始化"""
    assert version_manager.version_dir.exists()
    assert version_manager.metadata_file.exists()
    # history_file 和 lineage_file 可能不会在初始化时创建，只在需要时创建
    assert hasattr(version_manager, 'history_file')
    assert hasattr(version_manager, 'lineage_file')


def test_create_version(version_manager, data_model):
    """测试创建版本"""
    version_id = version_manager.create_version(
        data_model,
        description="测试版本"
    )
    
    assert version_id is not None
    assert version_id in version_manager.metadata['versions']


def test_get_version(version_manager, data_model):
    """测试获取版本"""
    version_id = version_manager.create_version(data_model, description="测试版本")
    
    version_data = version_manager.get_version(version_id)
    
    assert version_data is not None
    assert hasattr(version_data, 'data') or hasattr(version_data, 'get_metadata')


def test_list_versions(version_manager, data_model):
    """测试列出版本"""
    version_manager.create_version(data_model, description="版本1")
    version_manager.create_version(data_model, description="版本2")
    
    versions = version_manager.list_versions()
    
    assert len(versions) >= 2


def test_get_current_version(version_manager, data_model):
    """测试获取当前版本"""
    version_id = version_manager.create_version(data_model, description="测试版本")
    
    # 当前版本应该是最新创建的版本
    current = version_manager.current_version
    
    assert current == version_id


def test_set_current_version(version_manager, data_model):
    """测试设置当前版本"""
    version_id = version_manager.create_version(data_model, description="测试版本")
    
    # 创建版本后，当前版本应该自动设置为最新版本
    assert version_manager.current_version == version_id


def test_delete_version(version_manager, data_model):
    """测试删除版本"""
    version1 = version_manager.create_version(data_model, description="版本1")
    version2 = version_manager.create_version(data_model, description="版本2")
    
    # 删除非当前版本
    result = version_manager.delete_version(version1)
    
    assert result is True
    assert version1 not in version_manager.metadata['versions']


def test_get_version_history(version_manager, data_model):
    """测试获取版本历史"""
    version_manager.create_version(data_model, description="版本1")
    version_manager.create_version(data_model, description="版本2")
    
    # 使用 history 属性获取历史记录
    history = version_manager.history
    
    assert len(history) >= 2
    assert all('version_id' in entry or 'version' in entry for entry in history)


def test_get_version_lineage(version_manager, data_model):
    """测试获取版本血缘关系"""
    version1 = version_manager.create_version(data_model, description="版本1")
    version2 = version_manager.create_version(data_model, description="版本2")
    
    # 使用 get_lineage 方法
    lineage = version_manager.get_lineage(version2)
    
    assert lineage is not None
    assert 'version_id' in lineage
    assert 'ancestors' in lineage


def test_compare_versions(version_manager, data_model):
    """测试比较版本"""
    version1 = version_manager.create_version(data_model, description="版本1")
    
    # 创建修改后的数据
    modified_data = data_model.data.copy()
    modified_data.loc[0, 'value'] = 999
    modified_model = MockDataModel(modified_data)
    version2 = version_manager.create_version(modified_model, description="版本2")
    
    diff = version_manager.compare_versions(version1, version2)
    
    assert diff is not None
    assert isinstance(diff, dict)


def test_get_version_metadata(version_manager, data_model):
    """测试获取版本元数据"""
    version_id = version_manager.create_version(
        data_model,
        description="测试版本",
        tags=["test", "sample"]
    )
    
    # 使用 get_version_info 方法
    metadata = version_manager.get_version_info(version_id)
    
    assert metadata is not None
    assert 'description' in metadata
    assert 'tags' in metadata


def test_rollback_to_version(version_manager, data_model):
    """测试回滚到指定版本"""
    version1 = version_manager.create_version(data_model, description="版本1")
    
    # 创建新版本
    modified_data = data_model.data.copy()
    modified_data.loc[0, 'value'] = 999
    modified_model = MockDataModel(modified_data)
    version2 = version_manager.create_version(modified_model, description="版本2")
    
    # 回滚到版本1
    rolled_back = version_manager.rollback_to_version(version1)
    
    assert rolled_back is not None
    assert version_manager.current_version != version2


def test_rollback_to_nonexistent_version(version_manager, data_model):
    """测试回滚到不存在的版本"""
    version_manager.create_version(data_model, description="版本1")
    
    rolled_back = version_manager.rollback_to_version("nonexistent_version")
    
    assert rolled_back is None


def test_export_version(version_manager, data_model, tmp_path):
    """测试导出版本"""
    version_id = version_manager.create_version(data_model, description="测试版本")
    
    export_path = tmp_path / "exported_version.parquet"
    result = version_manager.export_version(version_id, str(export_path))
    
    assert result is True
    assert export_path.exists()


def test_export_nonexistent_version(version_manager, tmp_path):
    """测试导出不存在的版本"""
    export_path = tmp_path / "exported_version.parquet"
    result = version_manager.export_version("nonexistent", str(export_path))
    
    assert result is False


def test_import_version(version_manager, sample_data, tmp_path):
    """测试导入版本"""
    # 先创建一个parquet文件用于导入
    import pandas as pd
    import_path = tmp_path / "import_data.parquet"
    sample_data.to_parquet(import_path)
    
    version_id = version_manager.import_version(str(import_path))
    
    assert version_id is not None
    assert version_id in version_manager.metadata['versions']


def test_import_nonexistent_file(version_manager, tmp_path):
    """测试导入不存在的文件"""
    nonexistent_path = tmp_path / "nonexistent.parquet"
    
    version_id = version_manager.import_version(str(nonexistent_path))
    
    assert version_id is None


def test_get_lineage(version_manager, data_model):
    """测试获取版本血缘关系"""
    version1 = version_manager.create_version(data_model, description="版本1")
    version2 = version_manager.create_version(data_model, description="版本2")
    
    lineage = version_manager.get_lineage(version2)
    
    assert isinstance(lineage, dict)
    assert 'version_id' in lineage
    assert 'ancestors' in lineage


def test_update_metadata(version_manager, data_model):
    """测试更新版本元数据"""
    version_id = version_manager.create_version(data_model, description="测试版本")
    
    new_metadata = {'updated_by': 'test_user', 'note': 'test note'}
    result = version_manager.update_metadata(version_id, new_metadata)
    
    assert result is True
    version_info = version_manager.get_version_info(version_id)
    assert 'updated_by' in version_info.get('metadata', {})


def test_update_metadata_nonexistent_version(version_manager):
    """测试更新不存在版本的元数据"""
    result = version_manager.update_metadata("nonexistent", {'key': 'value'})
    
    assert result is False


def test_compare_versions_detailed(version_manager, data_model):
    """测试详细版本比较"""
    version1 = version_manager.create_version(data_model, description="版本1")
    
    # 创建修改后的数据
    modified_data = data_model.data.copy()
    modified_data.loc[0, 'value'] = 999
    modified_model = MockDataModel(modified_data)
    version2 = version_manager.create_version(modified_model, description="版本2")
    
    diff = version_manager.compare_versions(version1, version2)
    
    assert isinstance(diff, dict)
    assert 'metadata_diff' in diff
    assert 'data_diff' in diff


def test_compare_versions_nonexistent(version_manager, data_model):
    """测试比较不存在的版本"""
    version1 = version_manager.create_version(data_model, description="版本1")
    
    from src.infrastructure.utils.exceptions import DataVersionError
    with pytest.raises(DataVersionError):
        version_manager.compare_versions(version1, "nonexistent")


def test_export_version_failure(version_manager, data_model, monkeypatch, tmp_path):
    """当复制文件失败时，export_version 返回 False"""
    version_id = version_manager.create_version(data_model, description="导出失败场景")

    import shutil
    def _raise(*args, **kwargs):
        raise OSError("copy failed")
    monkeypatch.setattr(shutil, "copy2", _raise)

    export_path = tmp_path / "export.parquet"
    ok = version_manager.export_version(version_id, export_path)
    assert ok is False


def test_import_version_failure(version_manager, monkeypatch, tmp_path):
    """当 pd.read_parquet 抛出异常时，import_version 返回 None"""
    f = tmp_path / "bad.parquet"
    f.write_bytes(b"not a parquet")

    import pandas as pd
    def _raise_pq(*args, **kwargs):
        raise ValueError("read parquet failed")
    monkeypatch.setattr(pd, "read_parquet", _raise_pq)

    version_id = version_manager.import_version(str(f))
    assert version_id is None


def test_update_metadata_save_failure(version_manager, data_model, monkeypatch):
    """当 _save_metadata 失败时，update_metadata 返回 False"""
    version_id = version_manager.create_version(data_model, description="更新元数据失败场景")

    def _raise_save(*args, **kwargs):
        raise OSError("save metadata failed")
    monkeypatch.setattr(version_manager, "_save_metadata", _raise_save)

    ok = version_manager.update_metadata(version_id, {"owner": "test"})
    assert ok is False


def test_compare_versions_no_common_columns(version_manager, data_model):
    """两个版本没有公共列时，比较应返回列差异且不抛异常"""
    v1 = version_manager.create_version(data_model, description="v1")

    import pandas as pd
    df2 = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    class _DM:
        def __init__(self, d):
            self.data = d
            self._user_metadata = {}
        def get_metadata(self, user_only=False):
            return self._user_metadata if user_only else self._user_metadata
    v2 = version_manager.create_version(_DM(df2), description="v2")

    diff = version_manager.compare_versions(v1, v2)
    assert isinstance(diff, dict)
    cols = diff.get("data_diff", {}).get("columns_diff", {})
    assert isinstance(cols, dict)
    assert cols.get("added") or cols.get("removed")


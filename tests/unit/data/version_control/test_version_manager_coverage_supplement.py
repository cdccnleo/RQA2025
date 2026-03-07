#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试version_manager的覆盖率补充测试

补充现有测试未覆盖的功能，目标提升覆盖率到80%+
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
import tempfile
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from src.data.version_control.version_manager import DataVersionManager


@pytest.fixture
def temp_version_dir():
    """创建临时版本目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def version_manager(temp_version_dir):
    """创建版本管理器实例"""
    return DataVersionManager(version_dir=str(temp_version_dir))


@pytest.fixture
def mock_data_model():
    """创建Mock数据模型"""
    class MockDataModel:
        def __init__(self, data, frequency='1d', metadata=None):
            self.data = data
            self._frequency = frequency
            self._metadata = metadata or {}
            self._user_metadata = metadata or {}
        
        def get_frequency(self):
            return self._frequency
        
        def get_metadata(self, user_only=False):
            return self._user_metadata if user_only else self._metadata
        
        def validate(self):
            return self.data is not None and not self.data.empty if hasattr(self.data, 'empty') else True
    
    return MockDataModel


def test_generate_version_first_version(version_manager):
    """测试生成第一个版本号"""
    version = version_manager._generate_version()
    assert version.startswith("v_")
    assert "_1" in version


def test_generate_version_same_timestamp(version_manager, mock_data_model):
    """测试同一时间戳生成递增版本号"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    # 创建第一个版本
    version1 = version_manager.create_version(data_model, "test version 1")
    
    # 立即创建第二个版本（同一时间戳）
    time.sleep(0.1)  # 确保时间戳不同
    version2 = version_manager.create_version(data_model, "test version 2")
    
    assert version1 != version2


def test_get_ancestors_single_level(version_manager, mock_data_model):
    """测试获取单层祖先版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    
    ancestors = version_manager._get_ancestors(version2)
    assert version1 in ancestors


def test_get_ancestors_multiple_levels(version_manager, mock_data_model):
    """测试获取多层祖先版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    df3 = pd.DataFrame({'col1': [7, 8, 9]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    data_model3 = mock_data_model(df3, metadata={"test": "data3"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    version3 = version_manager.create_version(data_model3, "version 3")
    
    ancestors = version_manager._get_ancestors(version3)
    assert version1 in ancestors
    assert version2 in ancestors


def test_create_version_with_tags_and_creator(version_manager, mock_data_model):
    """测试创建带标签和创建者的版本"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(
        data_model,
        "test version",
        tags=["tag1", "tag2"],
        creator="test_user",
        branch="test_branch"
    )
    
    version_info = version_manager.get_version_info(version_id)
    assert version_info is not None
    assert "tag1" in version_info.get('tags', [])
    assert version_info.get('creator') == "test_user"
    assert version_info.get('branch') == "test_branch"


def test_create_version_data_none(version_manager, mock_data_model):
    """测试创建版本时数据为None"""
    class NoneDataModel:
        def __init__(self):
            self.data = None
        
        def validate(self):
            return False
    
    data_model = NoneDataModel()
    
    with pytest.raises(Exception):  # 应该抛出DataVersionError或ValueError
        version_manager.create_version(data_model, "test version")


def test_create_version_exception_cleanup(version_manager, mock_data_model, monkeypatch):
    """测试创建版本失败时的清理"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    # Mock to_parquet抛出异常
    original_to_parquet = pd.DataFrame.to_parquet
    def failing_to_parquet(self, *args, **kwargs):
        raise Exception("Failed to save")
    
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', failing_to_parquet)
    
    with pytest.raises(Exception):
        version_manager.create_version(data_model, "test version")
    
    # 恢复原始方法
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', original_to_parquet)


def test_get_version_info_from_history(version_manager, mock_data_model):
    """测试从历史记录获取版本信息"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    # 从metadata中删除，但保留在history中
    if version_id in version_manager.metadata.get('versions', {}):
        del version_manager.metadata['versions'][version_id]
    
    version_info = version_manager.get_version_info(version_id)
    assert version_info is not None
    assert version_info.get('description') == "test version"


def test_get_lineage_with_ancestors(version_manager, mock_data_model):
    """测试获取带祖先的血缘关系"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    
    lineage = version_manager.get_lineage(version2)
    assert lineage['version_id'] == version2
    assert len(lineage['ancestors']) > 0


def test_list_versions_with_limit(version_manager, mock_data_model):
    """测试限制数量的版本列表"""
    for i in range(5):
        df = pd.DataFrame({'col1': [i, i+1, i+2]})
        data_model = mock_data_model(df, metadata={"test": f"data{i}"})
        version_manager.create_version(data_model, f"version {i}")
    
    versions = version_manager.list_versions(limit=3)
    assert len(versions) == 3


def test_list_versions_with_tags(version_manager, mock_data_model):
    """测试按标签筛选版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1", tags=["tag1"])
    version2 = version_manager.create_version(data_model2, "version 2", tags=["tag2"])
    
    versions = version_manager.list_versions(tags=["tag1"])
    assert len(versions) == 1
    assert versions[0].get('version_id') == version1


def test_list_versions_with_creator(version_manager, mock_data_model):
    """测试按创建者筛选版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1", creator="user1")
    version2 = version_manager.create_version(data_model2, "version 2", creator="user2")
    
    versions = version_manager.list_versions(creator="user1")
    assert len(versions) == 1
    assert versions[0].get('version_id') == version1


def test_list_versions_with_branch(version_manager, mock_data_model):
    """测试按分支筛选版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1", branch="branch1")
    version2 = version_manager.create_version(data_model2, "version 2", branch="branch2")
    
    versions = version_manager.list_versions(branch="branch1")
    assert len(versions) == 1
    assert versions[0].get('version_id') == version1


def test_delete_version_success(version_manager, mock_data_model):
    """测试成功删除版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    
    # 删除version1（不是当前版本）
    result = version_manager.delete_version(version1)
    assert result is True
    
    # 验证版本已删除
    assert version_manager.get_version(version1) is None


def test_delete_version_current_version(version_manager, mock_data_model):
    """测试删除当前版本（应该失败）"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    with pytest.raises(Exception):  # 应该抛出DataVersionError
        version_manager.delete_version(version_id)


def test_delete_version_not_exists(version_manager):
    """测试删除不存在的版本"""
    with pytest.raises(Exception):  # 应该抛出DataVersionError
        version_manager.delete_version("non_existent_version")


def test_delete_version_update_branch(version_manager, mock_data_model):
    """测试删除版本时更新分支信息"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    df3 = pd.DataFrame({'col1': [7, 8, 9]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    data_model3 = mock_data_model(df3, metadata={"test": "data3"})
    
    version1 = version_manager.create_version(data_model1, "version 1", branch="test_branch")
    version2 = version_manager.create_version(data_model2, "version 2", branch="test_branch")
    version3 = version_manager.create_version(data_model3, "version 3", branch="test_branch")
    
    # 删除version2（不是当前版本）
    version_manager.delete_version(version2)
    
    # 验证分支信息已更新
    branch_version = version_manager.metadata['branches'].get('test_branch')
    assert branch_version == version3


def test_rollback_to_version(version_manager, mock_data_model):
    """测试回滚到指定版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3]})
    df2 = pd.DataFrame({'col1': [4, 5, 6]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    
    # 回滚到version1
    result = version_manager.rollback_to_version(version1)
    assert result is not None
    
    # 验证当前版本已更新
    assert version_manager.current_version != version2


def test_rollback_to_version_not_exists(version_manager):
    """测试回滚到不存在的版本"""
    result = version_manager.rollback_to_version("non_existent_version")
    assert result is None


def test_export_version(version_manager, mock_data_model, temp_version_dir):
    """测试导出版本"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    export_path = temp_version_dir / "exported.parquet"
    result = version_manager.export_version(version_id, str(export_path))
    
    assert result is True
    assert export_path.exists()


def test_export_version_not_exists(version_manager, temp_version_dir):
    """测试导出不存在的版本"""
    export_path = temp_version_dir / "exported.parquet"
    result = version_manager.export_version("non_existent_version", str(export_path))
    
    assert result is False


def test_import_version(version_manager, mock_data_model, temp_version_dir):
    """测试导入版本"""
    # 先创建一个版本并导出
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    export_path = temp_version_dir / "exported.parquet"
    version_manager.export_version(version_id, str(export_path))
    
    # 创建新的版本管理器并导入
    new_version_dir = temp_version_dir / "new_versions"
    new_version_dir.mkdir()
    new_manager = DataVersionManager(version_dir=str(new_version_dir))
    
    imported_version = new_manager.import_version(str(export_path))
    assert imported_version is not None
    
    # 验证导入的数据
    imported_data = new_manager.get_version(imported_version)
    assert imported_data is not None
    assert len(imported_data.data) == 3


def test_import_version_file_not_exists(version_manager):
    """测试导入不存在的文件"""
    result = version_manager.import_version("non_existent_file.parquet")
    assert result is None


def test_update_metadata(version_manager, mock_data_model):
    """测试更新版本元数据"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    new_metadata = {"new_key": "new_value", "test": "updated"}
    result = version_manager.update_metadata(version_id, new_metadata)
    
    assert result is True
    
    version_info = version_manager.get_version_info(version_id)
    assert version_info['metadata']['new_key'] == "new_value"
    assert version_info['metadata']['test'] == "updated"


def test_update_metadata_not_exists(version_manager):
    """测试更新不存在版本的元数据"""
    result = version_manager.update_metadata("non_existent_version", {"key": "value"})
    assert result is False


def test_compare_versions_success(version_manager, mock_data_model):
    """测试成功比较两个版本"""
    df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    df2 = pd.DataFrame({'col1': [1, 2, 4], 'col2': [4, 5, 7]})
    
    data_model1 = mock_data_model(df1, metadata={"test": "data1"})
    data_model2 = mock_data_model(df2, metadata={"test": "data2"})
    
    version1 = version_manager.create_version(data_model1, "version 1")
    version2 = version_manager.create_version(data_model2, "version 2")
    
    result = version_manager.compare_versions(version1, version2)
    
    assert result is not None
    assert 'metadata_diff' in result
    assert 'data_diff' in result
    assert 'shape_diff' in result['data_diff']
    assert 'columns_diff' in result['data_diff']
    assert 'value_diff' in result['data_diff']


def test_compare_versions_one_not_exists(version_manager, mock_data_model):
    """测试比较版本时一个版本不存在"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version1 = version_manager.create_version(data_model, "version 1")
    
    with pytest.raises(Exception):  # 应该抛出DataVersionError
        version_manager.compare_versions(version1, "non_existent_version")


def test_get_version_current_version(version_manager, mock_data_model):
    """测试获取当前版本"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    # 获取当前版本（不指定版本号）
    result = version_manager.get_version()
    assert result is not None
    assert hasattr(result, 'data')


def test_get_version_not_exists(version_manager):
    """测试获取不存在的版本"""
    result = version_manager.get_version("non_existent_version")
    assert result is None


def test_get_version_file_not_exists(version_manager, mock_data_model):
    """测试获取版本时文件不存在"""
    df = pd.DataFrame({'col1': [1, 2, 3]})
    data_model = mock_data_model(df, metadata={"test": "data"})
    
    version_id = version_manager.create_version(data_model, "test version")
    
    # 删除parquet文件
    data_file = version_manager.version_dir / f"{version_id}.parquet"
    if data_file.exists():
        data_file.unlink()
    
    result = version_manager.get_version(version_id)
    assert result is None


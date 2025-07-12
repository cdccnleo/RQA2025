"""
数据版本控制管理器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
from unittest.mock import Mock, patch

from src.data.version_control.version_manager import DataVersionManager
from src.data.models import DataModel
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def test_data():
    """测试数据fixture"""
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return df


@pytest.fixture
def test_version_dir(tmp_path):
    """测试版本目录fixture"""
    version_dir = tmp_path / "test_versions"
    version_dir.mkdir()
    yield version_dir
    # 清理测试目录
    shutil.rmtree(version_dir)


@pytest.fixture
def version_manager(test_version_dir):
    """版本管理器fixture"""
    return DataVersionManager(test_version_dir)


@pytest.fixture
def sample_data_model(test_data):
    """样本数据模型fixture"""
    model = DataModel(test_data)
    model.set_metadata({
        'source': 'test',
        'frequency': '1d',
        'symbol': '000001.SZ'
    })
    return model


def test_version_manager_init(test_version_dir):
    """测试版本管理器初始化"""
    manager = DataVersionManager(test_version_dir)

    # 验证目录创建
    assert test_version_dir.exists()
    assert test_version_dir.is_dir()

    # 验证元数据文件创建
    metadata_file = test_version_dir / 'version_metadata.json'
    assert metadata_file.exists()

    # 验证元数据内容
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    assert 'versions' in metadata
    assert 'latest_version' in metadata
    assert 'branches' in metadata
    assert 'main' in metadata['branches']


@pytest.mark.parametrize("branch,description", [
    ("main", "Initial version"),
    ("dev", "Development version"),
    ("test", "Test version")
])
def test_create_version(version_manager, sample_data_model, branch, description):
    """测试创建版本"""
    version_id = version_manager.create_version(
        sample_data_model,
        description=description,
        creator="test_user",
        branch=branch
    )

    # 验证版本ID格式
    assert isinstance(version_id, str)
    assert version_id.startswith("v_")

    # 验证版本信息
    version_info = version_manager.get_version_info(version_id)
    assert version_info is not None
    assert version_info['description'] == description
    assert version_info['creator'] == "test_user"
    assert version_info['branch'] == branch

    # 验证数据文件
    version_file = version_manager.version_dir / f"{version_id}.parquet"
    assert version_file.exists()

    # 验证分支更新
    assert version_manager.metadata['branches'][branch] == version_id


def test_get_version(version_manager, sample_data_model):
    """测试获取版本"""
    # 创建版本
    version_id = version_manager.create_version(
        sample_data_model,
        description="Test version",
        creator="test_user"
    )

    # 获取版本
    loaded_model = version_manager.get_version(version_id)
    assert loaded_model is not None

    # 验证数据
    pd.testing.assert_frame_equal(loaded_model.data, sample_data_model.data)

    # 验证元数据
    assert loaded_model.get_metadata() == sample_data_model.get_metadata()


def test_get_nonexistent_version(version_manager):
    """测试获取不存在的版本"""
    result = version_manager.get_version("nonexistent_version")
    assert result is None


@pytest.mark.parametrize("filter_params,expected_count", [
    ({}, 2),  # 无过滤
    ({"creator": "test_user"}, 2),  # 按创建者过滤
    ({"branch": "dev"}, 1),  # 按分支过滤
    ({"creator": "other_user"}, 0)  # 无匹配结果
])
def test_list_versions(version_manager, sample_data_model, filter_params, expected_count):
    """测试列出版本"""
    # 创建测试版本
    version_manager.create_version(
        sample_data_model,
        description="Main version",
        creator="test_user",
        branch="main"
    )
    version_manager.create_version(
        sample_data_model,
        description="Dev version",
        creator="test_user",
        branch="dev"
    )

    # 列出版本
    versions = version_manager.list_versions(**filter_params)
    assert len(versions) == expected_count


def test_compare_versions(version_manager, sample_data_model):
    """测试版本比较"""
    # 创建第一个版本
    v1_id = version_manager.create_version(
        sample_data_model,
        description="Version 1",
        creator="test_user"
    )

    # 修改数据创建第二个版本
    modified_data = sample_data_model.data.copy()
    modified_data['close'] = modified_data['close'] * 1.1
    modified_model = DataModel(modified_data)
    modified_model.set_metadata(sample_data_model.get_metadata())

    v2_id = version_manager.create_version(
        modified_model,
        description="Version 2",
        creator="test_user"
    )

    # 比较版本
    comparison = version_manager.compare_versions(v1_id, v2_id)

    assert 'metadata_diff' in comparison
    assert 'data_diff' in comparison
    assert 'value_diff' in comparison['data_diff']
    assert 'close' in comparison['data_diff']['value_diff']


def test_version_lineage(version_manager, sample_data_model):
    """测试版本血缘关系"""
    # 创建版本链
    v1_id = version_manager.create_version(
        sample_data_model,
        description="Version 1",
        creator="test_user"
    )

    v2_id = version_manager.create_version(
        sample_data_model,
        description="Version 2",
        creator="test_user"
    )

    # 获取血缘关系
    lineage = version_manager.get_lineage(v2_id)

    assert lineage['version_id'] == v2_id
    assert len(lineage['ancestors']) > 0
    assert lineage['ancestors'][0]['version_id'] == v2_id


def test_rollback(version_manager, sample_data_model):
    """测试版本回滚"""
    # 创建初始版本
    v1_id = version_manager.create_version(
        sample_data_model,
        description="Version 1",
        creator="test_user"
    )

    # 修改数据创建新版本
    modified_data = sample_data_model.data.copy()
    modified_data['close'] = modified_data['close'] * 1.1
    modified_model = DataModel(modified_data)
    modified_model.set_metadata(sample_data_model.get_metadata())

    v2_id = version_manager.create_version(
        modified_model,
        description="Version 2",
        creator="test_user"
    )

    # 回滚到第一个版本
    rollback_id = version_manager.rollback(v1_id)

    # 验证回滚版本
    rollback_model = version_manager.get_version(rollback_id)
    assert rollback_model is not None
    pd.testing.assert_frame_equal(rollback_model.data, sample_data_model.data)


@pytest.mark.parametrize("invalid_id", [
    "nonexistent_version",
    "",
    None
])
def test_invalid_version_operations(version_manager, invalid_id):
    """测试无效版本操作"""
    # 测试获取版本
    assert version_manager.get_version(invalid_id) is None

    # 测试获取版本信息
    assert version_manager.get_version_info(invalid_id) is None

    # 测试回滚到无效版本
    with pytest.raises(DataLoaderError, match="Version not found"):
        version_manager.rollback(invalid_id)


@patch('pandas.DataFrame.to_parquet')
def test_version_creation_failure(mock_to_parquet, version_manager, sample_data_model):
    """测试版本创建失败"""
    # 模拟保存数据失败
    mock_to_parquet.side_effect = Exception("Failed to save data")

    # 验证异常抛出
    with pytest.raises(DataLoaderError, match="Failed to create version"):
        version_manager.create_version(
            sample_data_model,
            description="Failed version",
            creator="test_user"
        )


def test_version_metadata_corruption(version_manager, test_version_dir):
    """测试元数据损坏情况"""
    # 创建损坏的元数据文件
    metadata_file = test_version_dir / 'version_metadata.json'
    with open(metadata_file, 'w') as f:
        f.write("invalid json content")

    # 创建新的管理器实例
    new_manager = DataVersionManager(test_version_dir)

    # 验证使用默认元数据
    assert new_manager.metadata == {
        'versions': {},
        'latest_version': None,
        'branches': {'main': None}
    }


def test_concurrent_version_creation(version_manager, sample_data_model):
    """测试并发版本创建"""
    # 模拟并发创建版本
    version_ids = []
    for i in range(5):
        version_id = version_manager.create_version(
            sample_data_model,
            description=f"Concurrent version {i}",
            creator="test_user"
        )
        version_ids.append(version_id)

    # 验证所有版本都被正确创建
    assert len(version_ids) == 5
    assert len(set(version_ids)) == 5  # 确保版本ID唯一

    # 验证每个版本都可以被加载
    for version_id in version_ids:
        assert version_manager.get_version(version_id) is not None

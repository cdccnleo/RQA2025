"""
数据版本管理测试模块
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.data.data_manager import DataModel
from src.data.version_control.version_manager import DataVersionManager
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def test_data():
    """测试数据fixture"""
    dates = pd.date_range('2023-01-01', periods=10)
    return pd.DataFrame({
        'close': np.random.randn(10) + 100,
        'volume': np.random.randint(1000, 10000, 10),
        'open': np.random.randn(10) + 100,
        'high': np.random.randn(10) + 102,
        'low': np.random.randn(10) + 98
    }, index=dates)


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
    model = DataModel(test_data, '1d', {
        'source': 'test',
        'symbol': '000001.SZ',
        'version': 'v1.0'
    })
    return model


class TestDataVersionManager:
    """测试数据版本管理器"""
    
    def test_version_manager_init(self, test_version_dir):
        """测试版本管理器初始化"""
        manager = DataVersionManager(test_version_dir)
        
        assert manager.version_dir == test_version_dir
        assert test_version_dir.exists()
        assert test_version_dir.is_dir()
    
    def test_create_version(self, version_manager, sample_data_model):
        """测试创建版本"""
        version_id = version_manager.create_version(
            sample_data_model,
            version_name="test_v1.0",
            description="测试版本"
        )
        
        assert version_id is not None
        assert len(version_id) > 0
        
        # 验证版本文件创建
        version_file = version_manager.version_dir / f"{version_id}.pkl"
        assert version_file.exists()
    
    def test_load_version(self, version_manager, sample_data_model):
        """测试加载版本"""
        # 先创建版本
        version_id = version_manager.create_version(
            sample_data_model,
            version_name="test_v1.0"
        )
        
        # 加载版本
        loaded_model = version_manager.load_version(version_id)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, DataModel)
        assert loaded_model.get_metadata()['symbol'] == '000001.SZ'
    
    def test_list_versions(self, version_manager, sample_data_model):
        """测试列出版本"""
        # 创建多个版本
        version_manager.create_version(sample_data_model, "v1.0")
        version_manager.create_version(sample_data_model, "v1.1")
        
        versions = version_manager.list_versions()
        
        assert len(versions) >= 2
        assert any('v1.0' in v['name'] for v in versions)
        assert any('v1.1' in v['name'] for v in versions)
    
    def test_get_version_info(self, version_manager, sample_data_model):
        """测试获取版本信息"""
        version_id = version_manager.create_version(
            sample_data_model,
            version_name="test_v1.0",
            description="测试版本"
        )
        
        info = version_manager.get_version_info(version_id)
        
        assert info is not None
        assert info['name'] == "test_v1.0"
        assert info['description'] == "测试版本"
        assert 'created_at' in info
    
    def test_delete_version(self, version_manager, sample_data_model):
        """测试删除版本"""
        version_id = version_manager.create_version(sample_data_model, "test_v1.0")
        
        # 验证版本存在
        assert version_manager.get_version_info(version_id) is not None
        
        # 删除版本
        success = version_manager.delete_version(version_id)
        assert success is True
        
        # 验证版本已删除
        assert version_manager.get_version_info(version_id) is None
    
    def test_version_comparison(self, version_manager, test_data):
        """测试版本比较"""
        # 创建两个不同版本的数据模型
        model_v1 = DataModel(test_data, '1d', {'version': 'v1.0'})
        model_v2 = DataModel(test_data * 1.1, '1d', {'version': 'v2.0'})
        
        # 创建版本
        v1_id = version_manager.create_version(model_v1, "v1.0")
        v2_id = version_manager.create_version(model_v2, "v2.0")
        
        # 比较版本
        comparison = version_manager.compare_versions(v1_id, v2_id)
        
        assert comparison is not None
        assert 'data_difference' in comparison
        assert 'metadata_difference' in comparison
    
    def test_version_rollback(self, version_manager, sample_data_model):
        """测试版本回滚"""
        # 创建初始版本
        v1_id = version_manager.create_version(sample_data_model, "v1.0")
        
        # 创建新版本
        updated_model = DataModel(
            sample_data_model.data,
            '1d',
            {'version': 'v2.0', 'updated': True}
        )
        v2_id = version_manager.create_version(updated_model, "v2.0")
        
        # 回滚到v1
        rollback_model = version_manager.rollback_to_version(v1_id)
        
        assert rollback_model is not None
        assert rollback_model.get_metadata()['version'] == 'v1.0'
    
    def test_version_export_import(self, version_manager, sample_data_model, tmp_path):
        """测试版本导出导入"""
        version_id = version_manager.create_version(sample_data_model, "test_v1.0")
        
        # 导出版本
        export_path = tmp_path / "exported_version.zip"
        success = version_manager.export_version(version_id, export_path)
        
        assert success is True
        assert export_path.exists()
        
        # 导入版本
        imported_id = version_manager.import_version(export_path)
        
        assert imported_id is not None
        assert imported_id != version_id  # 应该生成新的ID
        
        # 验证导入的数据
        imported_model = version_manager.load_version(imported_id)
        assert imported_model.get_metadata()['symbol'] == '000001.SZ'
    
    def test_version_metadata_update(self, version_manager, sample_data_model):
        """测试版本元数据更新"""
        version_id = version_manager.create_version(sample_data_model, "test_v1.0")
        
        # 更新元数据
        new_metadata = {
            'description': '更新的描述',
            'tags': ['test', 'updated'],
            'author': 'test_user'
        }
        
        success = version_manager.update_version_metadata(version_id, new_metadata)
        assert success is True
        
        # 验证更新
        info = version_manager.get_version_info(version_id)
        assert info['description'] == '更新的描述'
        assert 'test' in info.get('tags', [])
    
    def test_version_validation(self, version_manager):
        """测试版本验证"""
        # 测试无效版本ID
        with pytest.raises(ValueError):
            version_manager.load_version("invalid_id")
        
        # 测试无效版本目录
        with pytest.raises(ValueError):
            DataVersionManager(None)
    
    def test_version_cleanup(self, version_manager, sample_data_model):
        """测试版本清理"""
        # 创建多个版本
        for i in range(5):
            version_manager.create_version(sample_data_model, f"v{i}.0")
        
        # 清理旧版本（保留最新的2个）
        cleaned_count = version_manager.cleanup_old_versions(keep_count=2)
        
        assert cleaned_count >= 3  # 应该清理至少3个版本
        
        # 验证剩余版本数量
        remaining_versions = version_manager.list_versions()
        assert len(remaining_versions) <= 2

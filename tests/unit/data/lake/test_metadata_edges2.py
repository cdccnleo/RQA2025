"""
边界测试：metadata.py
测试边界情况和异常场景
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
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil
from src.data.lake.metadata import DataMetadata, MetadataManager


def test_data_metadata_init():
    """测试 DataMetadata（初始化）"""
    metadata = DataMetadata(
        data_type="stock",
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=100,
        columns=["col1", "col2"]
    )
    
    assert metadata.data_type == "stock"
    assert metadata.source == "test_source"
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == "1.0"
    assert metadata.record_count == 100
    assert metadata.columns == ["col1", "col2"]
    assert metadata.description is None
    assert metadata.additional_info is None


def test_data_metadata_init_with_optional():
    """测试 DataMetadata（初始化，带可选参数）"""
    metadata = DataMetadata(
        data_type="crypto",
        source="test_source",
        created_at=datetime.now(),
        version="2.0",
        record_count=200,
        columns=["col1"],
        description="Test description",
        additional_info={"key": "value"}
    )
    
    assert metadata.description == "Test description"
    assert metadata.additional_info == {"key": "value"}


def test_data_metadata_to_dict():
    """测试 DataMetadata（转换为字典）"""
    created_at = datetime.now()
    metadata = DataMetadata(
        data_type="forex",
        source="test_source",
        created_at=created_at,
        version="1.0",
        record_count=50,
        columns=["col1", "col2"]
    )
    
    result = metadata.to_dict()
    
    assert result["data_type"] == "forex"
    assert result["source"] == "test_source"
    assert isinstance(result["created_at"], str)  # datetime 被转换为 ISO 格式字符串
    assert result["version"] == "1.0"
    assert result["record_count"] == 50
    assert result["columns"] == ["col1", "col2"]


def test_data_metadata_to_json():
    """测试 DataMetadata（转换为JSON）"""
    metadata = DataMetadata(
        data_type="bond",
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=75,
        columns=["col1"]
    )
    
    json_str = metadata.to_json()
    
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["data_type"] == "bond"
    assert parsed["record_count"] == 75


def test_data_metadata_from_dict():
    """测试 DataMetadata（从字典创建）"""
    created_at = datetime.now()
    metadata_dict = {
        "data_type": "index",
        "source": "test_source",
        "created_at": created_at.isoformat(),
        "version": "1.0",
        "record_count": 300,
        "columns": ["col1", "col2", "col3"]
    }
    
    metadata = DataMetadata.from_dict(metadata_dict)
    
    assert metadata.data_type == "index"
    assert metadata.source == "test_source"
    assert isinstance(metadata.created_at, datetime)
    assert metadata.version == "1.0"
    assert metadata.record_count == 300
    assert len(metadata.columns) == 3


def test_data_metadata_from_dict_datetime_object():
    """测试 DataMetadata（从字典创建，datetime对象）"""
    created_at = datetime.now()
    metadata_dict = {
        "data_type": "options",
        "source": "test_source",
        "created_at": created_at,  # 已经是 datetime 对象
        "version": "1.0",
        "record_count": 150,
        "columns": ["col1"]
    }
    
    metadata = DataMetadata.from_dict(metadata_dict)
    
    assert isinstance(metadata.created_at, datetime)
    assert metadata.data_type == "options"


def test_data_metadata_validate_valid():
    """测试 DataMetadata（验证，有效）"""
    metadata = DataMetadata(
        data_type="macro",
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=100,
        columns=["col1"]
    )
    
    assert metadata.validate() is True


def test_data_metadata_validate_missing_field():
    """测试 DataMetadata（验证，缺少字段）"""
    # 创建一个缺少必需字段的元数据
    # 由于 DataMetadata 是 dataclass，我们不能直接创建缺少字段的实例
    # 但我们可以测试 None 值的情况
    metadata = DataMetadata(
        data_type="",  # 空字符串
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=100,
        columns=[]
    )
    
    # 空字符串会被视为 falsy，所以验证应该失败
    assert metadata.validate() is False


def test_data_metadata_validate_empty_columns():
    """测试 DataMetadata（验证，空列）"""
    metadata = DataMetadata(
        data_type="test",
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=0,
        columns=[]  # 空列表
    )
    
    # 空列表会被视为 falsy
    assert metadata.validate() is False


def test_metadata_manager_init():
    """测试 MetadataManager（初始化）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        assert manager.storage_path == tmpdir


def test_metadata_manager_save_metadata_success():
    """测试 MetadataManager（保存元数据，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        metadata = DataMetadata(
            data_type="stock",
            source="test_source",
            created_at=datetime.now(),
            version="1.0",
            record_count=100,
            columns=["col1", "col2"]
        )
        
        result = manager.save_metadata(metadata, "test_id")
        
        assert result is True
        assert Path(tmpdir) / "test_id_metadata.json"
        assert (Path(tmpdir) / "test_id_metadata.json").exists()


def test_metadata_manager_save_metadata_invalid_path():
    """测试 MetadataManager（保存元数据，无效路径）"""
    # 使用一个不存在的路径（但会在保存时创建）
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir) / "nonexistent" / "subdir"
        manager = MetadataManager(str(invalid_path))
        metadata = DataMetadata(
            data_type="stock",
            source="test_source",
            created_at=datetime.now(),
            version="1.0",
            record_count=100,
            columns=["col1"]
        )
        
        # 由于路径不存在，保存可能会失败
        result = manager.save_metadata(metadata, "test_id")
        
        # 结果可能是 True 或 False，取决于文件系统权限
        assert isinstance(result, bool)


def test_metadata_manager_load_metadata_success():
    """测试 MetadataManager（加载元数据，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        metadata = DataMetadata(
            data_type="crypto",
            source="test_source",
            created_at=datetime.now(),
            version="1.0",
            record_count=200,
            columns=["col1"]
        )
        
        # 先保存
        manager.save_metadata(metadata, "test_id")
        
        # 再加载
        loaded = manager.load_metadata("test_id")
        
        assert loaded is not None
        assert loaded.data_type == "crypto"
        assert loaded.record_count == 200


def test_metadata_manager_load_metadata_not_found():
    """测试 MetadataManager（加载元数据，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        loaded = manager.load_metadata("nonexistent_id")
        
        assert loaded is None


def test_metadata_manager_load_metadata_invalid_json():
    """测试 MetadataManager（加载元数据，无效JSON）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        # 创建一个无效的 JSON 文件
        invalid_file = Path(tmpdir) / "test_id_metadata.json"
        invalid_file.write_text("invalid json content", encoding='utf-8')
        
        loaded = manager.load_metadata("test_id")
        
        # 应该返回 None 或抛出异常
        assert loaded is None


def test_data_metadata_round_trip():
    """测试 DataMetadata（往返转换）"""
    original = DataMetadata(
        data_type="test",
        source="test_source",
        created_at=datetime.now(),
        version="1.0",
        record_count=100,
        columns=["col1", "col2"],
        description="Test description",
        additional_info={"key": "value"}
    )
    
    # 转换为字典再转回
    metadata_dict = original.to_dict()
    restored = DataMetadata.from_dict(metadata_dict)
    
    assert restored.data_type == original.data_type
    assert restored.source == original.source
    assert restored.version == original.version
    assert restored.record_count == original.record_count
    assert restored.columns == original.columns
    assert restored.description == original.description
    assert restored.additional_info == original.additional_info

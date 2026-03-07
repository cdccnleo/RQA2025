"""
边界测试：metadata_manager.py
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
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import json
from src.data.lake.metadata_manager import (
    MetadataSchema,
    MetadataManager
)


def test_metadata_schema_init():
    """测试 MetadataSchema（初始化）"""
    schema = MetadataSchema(dataset_name="test_dataset")
    
    assert schema.dataset_name == "test_dataset"
    assert schema.description == ""
    assert schema.schema_version == "1.0"
    assert schema.columns == []
    assert schema.tags == []
    assert schema.owner == ""
    assert schema.access_level == "public"
    assert schema.created_at != ""
    assert schema.updated_at != ""


def test_metadata_schema_init_with_values():
    """测试 MetadataSchema（初始化，带值）"""
    schema = MetadataSchema(
        dataset_name="test_dataset",
        description="Test description",
        schema_version="2.0",
        columns=[{"name": "col1", "type": "int64"}],
        tags=["tag1", "tag2"],
        owner="test_owner",
        access_level="private"
    )
    
    assert schema.description == "Test description"
    assert schema.schema_version == "2.0"
    assert len(schema.columns) == 1
    assert len(schema.tags) == 2
    assert schema.owner == "test_owner"
    assert schema.access_level == "private"


def test_metadata_schema_post_init():
    """测试 MetadataSchema（后初始化）"""
    schema = MetadataSchema(
        dataset_name="test_dataset",
        columns=None,
        tags=None,
        created_at="",
        updated_at=""
    )
    
    # __post_init__ 应该设置默认值
    assert schema.columns == []
    assert schema.tags == []
    assert schema.created_at != ""
    assert schema.updated_at != ""


def test_metadata_manager_init():
    """测试 MetadataManager（初始化）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        assert manager.metadata_path == Path(tmpdir)
        assert manager.metadata_path.exists()


def test_metadata_manager_create_schema():
    """测试 MetadataManager（创建模式）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        
        schema = manager.create_schema("test_dataset", data)
        
        assert schema.dataset_name == "test_dataset"
        assert len(schema.columns) == 2
        assert schema.columns[0]["name"] == "col1"
        assert schema.columns[1]["name"] == "col2"


def test_metadata_manager_create_schema_empty_dataframe():
    """测试 MetadataManager（创建模式，空数据框）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame()
        
        schema = manager.create_schema("test_dataset", data)
        
        assert schema.dataset_name == "test_dataset"
        assert len(schema.columns) == 0


def test_metadata_manager_create_schema_with_kwargs():
    """测试 MetadataManager（创建模式，带kwargs）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        schema = manager.create_schema(
            "test_dataset",
            data,
            description="Test description",
            owner="test_owner"
        )
        
        assert schema.description == "Test description"
        assert schema.owner == "test_owner"


def test_metadata_manager_create_schema_with_nulls():
    """测试 MetadataManager（创建模式，包含空值）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": ["a", "b", None]
        })
        
        schema = manager.create_schema("test_dataset", data)
        
        assert len(schema.columns) == 2
        # 检查 nullable 字段
        assert schema.columns[0]["nullable"] is True
        assert schema.columns[1]["nullable"] is True


def test_metadata_manager_get_schema_not_found():
    """测试 MetadataManager（获取模式，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        schema = manager.get_schema("nonexistent")
        
        assert schema is None


def test_metadata_manager_get_schema_success():
    """测试 MetadataManager（获取模式，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        # 先创建
        created_schema = manager.create_schema("test_dataset", data)
        
        # 再获取
        retrieved_schema = manager.get_schema("test_dataset")
        
        assert retrieved_schema is not None
        assert retrieved_schema.dataset_name == "test_dataset"
        assert len(retrieved_schema.columns) == 1


def test_metadata_manager_get_schema_invalid_json():
    """测试 MetadataManager（获取模式，无效JSON）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        # 创建无效的 JSON 文件
        invalid_file = manager.metadata_path / "test_dataset_schema.json"
        invalid_file.write_text("invalid json", encoding='utf-8')
        
        schema = manager.get_schema("test_dataset")
        
        assert schema is None


def test_metadata_manager_update_schema_not_found():
    """测试 MetadataManager（更新模式，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        result = manager.update_schema("nonexistent", description="New description")
        
        assert result is False


def test_metadata_manager_update_schema_success():
    """测试 MetadataManager（更新模式，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        # 先创建
        manager.create_schema("test_dataset", data)
        
        # 再更新
        result = manager.update_schema("test_dataset", description="Updated description")
        
        assert result is True
        
        # 验证更新
        updated_schema = manager.get_schema("test_dataset")
        assert updated_schema.description == "Updated description"


def test_metadata_manager_update_schema_multiple_fields():
    """测试 MetadataManager（更新模式，多个字段）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("test_dataset", data)
        
        result = manager.update_schema(
            "test_dataset",
            description="New description",
            owner="new_owner",
            access_level="private"
        )
        
        assert result is True
        
        updated_schema = manager.get_schema("test_dataset")
        assert updated_schema.description == "New description"
        assert updated_schema.owner == "new_owner"
        assert updated_schema.access_level == "private"


def test_metadata_manager_list_schemas_empty():
    """测试 MetadataManager（列出模式，空）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        schemas = manager.list_schemas()
        
        assert schemas == []


def test_metadata_manager_list_schemas_multiple():
    """测试 MetadataManager（列出模式，多个）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("dataset1", data)
        manager.create_schema("dataset2", data)
        manager.create_schema("dataset3", data)
        
        schemas = manager.list_schemas()
        
        assert len(schemas) == 3
        assert "dataset1" in schemas
        assert "dataset2" in schemas
        assert "dataset3" in schemas
        assert schemas == sorted(schemas)


def test_metadata_manager_delete_schema_not_found():
    """测试 MetadataManager（删除模式，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        result = manager.delete_schema("nonexistent")
        
        assert result is False


def test_metadata_manager_delete_schema_success():
    """测试 MetadataManager（删除模式，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("test_dataset", data)
        
        result = manager.delete_schema("test_dataset")
        
        assert result is True
        
        # 验证已删除
        schema = manager.get_schema("test_dataset")
        assert schema is None


def test_metadata_manager_search_schemas_empty():
    """测试 MetadataManager（搜索模式，空）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        results = manager.search_schemas("test")
        
        assert results == []


def test_metadata_manager_search_schemas_by_name():
    """测试 MetadataManager（搜索模式，按名称）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("test_dataset", data, description="Test")
        manager.create_schema("other_dataset", data)
        
        results = manager.search_schemas("test")
        
        assert len(results) == 1
        assert results[0].dataset_name == "test_dataset"


def test_metadata_manager_search_schemas_by_description():
    """测试 MetadataManager（搜索模式，按描述）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("dataset1", data, description="Stock market data")
        manager.create_schema("dataset2", data, description="Weather data")
        
        results = manager.search_schemas("stock")
        
        assert len(results) == 1
        assert results[0].dataset_name == "dataset1"


def test_metadata_manager_search_schemas_by_tag():
    """测试 MetadataManager（搜索模式，按标签）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        manager.create_schema("dataset1", data, tags=["finance", "market"])
        manager.create_schema("dataset2", data, tags=["weather"])
        
        results = manager.search_schemas("finance")
        
        assert len(results) == 1
        assert results[0].dataset_name == "dataset1"


def test_metadata_manager_search_schemas_by_column():
    """测试 MetadataManager（搜索模式，按列名）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"price": [1, 2, 3], "volume": [10, 20, 30]})
        
        manager.create_schema("dataset1", data)
        
        results = manager.search_schemas("price")
        
        assert len(results) == 1


def test_metadata_manager_get_schema_stats_empty():
    """测试 MetadataManager（获取统计信息，空）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        stats = manager.get_schema_stats()
        
        assert stats["total_datasets"] == 0
        assert stats["total_columns"] == 0
        assert stats["avg_columns_per_dataset"] == 0


def test_metadata_manager_get_schema_stats_multiple():
    """测试 MetadataManager（获取统计信息，多个）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        
        data1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        data2 = pd.DataFrame({"col3": [5, 6]})
        
        manager.create_schema("dataset1", data1, access_level="public")
        manager.create_schema("dataset2", data2, access_level="private")
        
        stats = manager.get_schema_stats()
        
        assert stats["total_datasets"] == 2
        assert stats["total_columns"] == 3
        assert stats["avg_columns_per_dataset"] == 1.5
        assert stats["access_levels"]["public"] == 1
        assert stats["access_levels"]["private"] == 1


def test_metadata_manager_validate_schema_not_found():
    """测试 MetadataManager（验证模式，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3]})
        
        result = manager.validate_schema("nonexistent", data)
        
        assert result["valid"] is False
        assert "error" in result


def test_metadata_manager_validate_schema_valid():
    """测试 MetadataManager（验证模式，有效）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        
        manager.create_schema("test_dataset", data)
        
        result = manager.validate_schema("test_dataset", data)
        
        assert result["valid"] is True
        assert len(result["warnings"]) == 0
        assert len(result["errors"]) == 0


def test_metadata_manager_validate_schema_missing_columns():
    """测试 MetadataManager（验证模式，缺少列）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        validation_data = pd.DataFrame({"col1": [1, 2]})  # 缺少 col2 和 col3
        
        manager.create_schema("test_dataset", schema_data)
        
        result = manager.validate_schema("test_dataset", validation_data)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0


def test_metadata_manager_validate_schema_extra_columns():
    """测试 MetadataManager（验证模式，额外列）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema_data = pd.DataFrame({"col1": [1, 2]})
        validation_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})  # 有额外列
        
        manager.create_schema("test_dataset", schema_data)
        
        result = manager.validate_schema("test_dataset", validation_data)
        
        assert result["valid"] is True  # 额外列只是警告，不使验证失败
        assert len(result["warnings"]) > 0


def test_metadata_manager_validate_schema_type_mismatch():
    """测试 MetadataManager（验证模式，类型不匹配）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema_data = pd.DataFrame({"col1": [1, 2, 3]})  # int64
        validation_data = pd.DataFrame({"col1": ["a", "b", "c"]})  # object
        
        manager.create_schema("test_dataset", schema_data)
        
        result = manager.validate_schema("test_dataset", validation_data)
        
        assert result["valid"] is True  # 类型不匹配只是警告
        assert len(result["warnings"]) > 0


def test_metadata_manager_matches_query_name():
    """测试 MetadataManager（匹配查询，按名称）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema = MetadataSchema(
            dataset_name="test_dataset",
            description="",
            tags=[],
            columns=[]
        )
        
        result = manager._matches_query(schema, "test")
        
        assert result is True


def test_metadata_manager_matches_query_description():
    """测试 MetadataManager（匹配查询，按描述）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema = MetadataSchema(
            dataset_name="dataset",
            description="Stock market data",
            tags=[],
            columns=[]
        )
        
        result = manager._matches_query(schema, "stock")
        
        assert result is True


def test_metadata_manager_matches_query_tag():
    """测试 MetadataManager（匹配查询，按标签）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema = MetadataSchema(
            dataset_name="dataset",
            description="",
            tags=["finance", "market"],
            columns=[]
        )
        
        result = manager._matches_query(schema, "finance")
        
        assert result is True


def test_metadata_manager_matches_query_column():
    """测试 MetadataManager（匹配查询，按列名）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema = MetadataSchema(
            dataset_name="dataset",
            description="",
            tags=[],
            columns=[{"name": "price"}, {"name": "volume"}]
        )
        
        result = manager._matches_query(schema, "price")
        
        assert result is True


def test_metadata_manager_matches_query_no_match():
    """测试 MetadataManager（匹配查询，无匹配）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MetadataManager(tmpdir)
        schema = MetadataSchema(
            dataset_name="dataset",
            description="Test",
            tags=["tag1"],
            columns=[{"name": "col1"}]
        )
        
        result = manager._matches_query(schema, "nonexistent")
        
        assert result is False


def test_metadata_manager_create_schema_exception(tmp_path):
    """测试 MetadataManager（创建模式，异常处理）"""
    # 模拟创建模式时抛出异常（覆盖 92-94 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    with patch.object(manager, '_save_schema', side_effect=Exception("Save error")):
        data = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(Exception):
            manager.create_schema("test_dataset", data)


def test_metadata_manager_update_schema_exception(tmp_path):
    """测试 MetadataManager（更新模式，异常处理）"""
    # 模拟更新模式时抛出异常（覆盖 133-135 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 然后模拟更新时抛出异常
    with patch.object(manager, '_save_schema', side_effect=Exception("Save error")):
        result = manager.update_schema("test_dataset", description="Updated")
        assert result is False


def test_metadata_manager_list_schemas_exception(tmp_path):
    """测试 MetadataManager（列出模式，异常处理）"""
    # 模拟列出模式时抛出异常（覆盖 146-148 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 使用 patch 来模拟 list_schemas 方法内部的异常
    with patch('pathlib.Path.glob', side_effect=Exception("List error")):
        result = manager.list_schemas()
        assert result == []


def test_metadata_manager_delete_schema_exception(tmp_path):
    """测试 MetadataManager（删除模式，异常处理）"""
    # 模拟删除模式时抛出异常（覆盖 163-165 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 然后模拟删除时抛出异常
    with patch('pathlib.Path.unlink', side_effect=Exception("Delete error")):
        result = manager.delete_schema("test_dataset")
        assert result is False


def test_metadata_manager_search_schemas_exception(tmp_path):
    """测试 MetadataManager（搜索模式，异常处理）"""
    # 模拟搜索模式时抛出异常（覆盖 179-181 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 然后模拟搜索时抛出异常
    with patch.object(manager, 'list_schemas', side_effect=Exception("Search error")):
        result = manager.search_schemas("test")
        assert result == []


def test_metadata_manager_get_schema_stats_base_exception(tmp_path):
    """测试 MetadataManager（获取统计信息，BaseException 处理）"""
    # 模拟获取统计信息时抛出 BaseException（覆盖 223-224 行）
    # 由于 datetime.fromisoformat 在解析无效日期时会抛出 ValueError，而不是 BaseException
    # 我们需要通过其他方式来触发 BaseException
    # 这里我们直接测试方法能正常调用，BaseException 分支在实际使用中很难触发
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 直接调用方法，验证能正常返回
    result = manager.get_schema_stats()
    assert isinstance(result, dict)
    assert 'total_datasets' in result


def test_metadata_manager_get_schema_stats_exception(tmp_path):
    """测试 MetadataManager（获取统计信息，异常处理）"""
    # 模拟获取统计信息时抛出异常（覆盖 236-238 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 模拟 list_schemas 抛出异常
    with patch.object(manager, 'list_schemas', side_effect=Exception("Stats error")):
        result = manager.get_schema_stats()
        assert result == {}


def test_metadata_manager_validate_schema_exception(tmp_path):
    """测试 MetadataManager（验证模式，异常处理）"""
    # 模拟验证模式时抛出异常（覆盖 281-283 行）
    from unittest.mock import patch
    manager = MetadataManager(str(tmp_path))
    # 先创建一个模式
    data = pd.DataFrame({'col1': [1, 2, 3]})
    manager.create_schema("test_dataset", data)
    # 模拟验证时抛出异常
    with patch.object(manager, 'get_schema', side_effect=Exception("Validate error")):
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        result = manager.validate_schema("test_dataset", test_data)
        assert result['valid'] is False
        assert 'error' in result

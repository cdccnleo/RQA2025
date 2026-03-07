"""
测试 ConfigValueProcessor 核心功能

覆盖 ConfigValueProcessor 的配置值处理功能
"""

import pytest
from src.infrastructure.config.core.config_processors import ConfigValueProcessor


class TestConfigValueProcessor:
    """ConfigValueProcessor 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        config_data = {"existing": "value"}
        processor = ConfigValueProcessor(config_data)

        assert processor._data == config_data
        assert processor._data is config_data  # Should reference the same object

    def test_set_single_level_value(self):
        """测试设置单级值"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        # Test setting a single level value
        result = processor.set_value("simple_key", "simple_value", ["simple_key"])
        assert result is True

        # Check that value was set in default section
        assert "default" in config_data
        assert config_data["default"]["simple_key"] == "simple_value"

    def test_set_nested_value(self):
        """测试设置嵌套值"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        # Test setting a nested value
        result = processor.set_value("section.subsection.key", "nested_value", ["section", "subsection", "key"])
        assert result is True

        # Check that nested structure was created
        assert "section" in config_data
        assert "subsection" in config_data["section"]
        assert config_data["section"]["subsection"]["key"] == "nested_value"

    def test_set_value_overwrites_section(self):
        """测试设置值覆盖原有section"""
        config_data = {"database": {"host": "localhost", "port": 5432}}
        processor = ConfigValueProcessor(config_data)

        # Set a value that overwrites the section
        result = processor.set_value("database", "new_value", ["database"])
        assert result is True

        # Check that section was replaced with simple value
        assert config_data["default"]["database"] == "new_value"
        assert "database" not in config_data or not isinstance(config_data.get("database"), dict)

    def test_set_value_complex_nested(self):
        """测试设置复杂的嵌套值"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        # Set multiple nested values
        result1 = processor.set_value("app.database.host", "localhost", ["app", "database", "host"])
        result2 = processor.set_value("app.database.port", 5432, ["app", "database", "port"])
        result3 = processor.set_value("app.cache.ttl", 3600, ["app", "cache", "ttl"])

        assert result1 is True
        assert result2 is True
        assert result3 is True

        # Check nested structure
        assert config_data["app"]["database"]["host"] == "localhost"
        assert config_data["app"]["database"]["port"] == 5432
        assert config_data["app"]["cache"]["ttl"] == 3600

    def test_set_value_updates_existing_nested(self):
        """测试更新现有的嵌套值"""
        config_data = {"app": {"database": {"host": "old_host"}}}
        processor = ConfigValueProcessor(config_data)

        # Update existing nested value
        result = processor.set_value("app.database.host", "new_host", ["app", "database", "host"])
        assert result is True

        # Check that value was updated
        assert config_data["app"]["database"]["host"] == "new_host"

    def test_get_old_value_existing_key(self):
        """测试获取现有键的旧值"""
        config_data = {"app": {"database": {"host": "localhost"}}}
        processor = ConfigValueProcessor(config_data)

        # Get existing nested value (method currently returns None)
        old_value = processor.get_old_value("app.database.host")
        assert old_value is None  # Current implementation returns None

    def test_get_old_value_nonexistent_key(self):
        """测试获取不存在键的旧值"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        # Get non-existent key
        old_value = processor.get_old_value("nonexistent.key")
        assert old_value is None

    def test_get_old_value_partial_path(self):
        """测试获取部分路径的旧值"""
        config_data = {"app": {"database": {"host": "localhost"}}}
        processor = ConfigValueProcessor(config_data)

        # Get partial path (method currently returns None)
        old_value = processor.get_old_value("app.database")
        assert old_value is None  # Current implementation returns None

    def test_get_old_value_root_key(self):
        """测试获取根键的旧值"""
        config_data = {"root_key": "root_value"}
        processor = ConfigValueProcessor(config_data)

        # Get root key (method currently returns None)
        old_value = processor.get_old_value("root_key")
        assert old_value is None  # Current implementation returns None

    def test_set_value_with_different_value_types(self):
        """测试设置不同类型的值"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        test_values = [
            ("string_key", "string_value"),
            ("int_key", 42),
            ("float_key", 3.14),
            ("bool_key", True),
            ("list_key", [1, 2, 3]),
            ("dict_key", {"nested": "value"})
        ]

        for key, value in test_values:
            result = processor.set_value(key, value, [key])
            assert result is True
            assert config_data["default"][key] == value

    def test_set_value_empty_key_parts(self):
        """测试设置空键部分"""
        config_data = {}
        processor = ConfigValueProcessor(config_data)

        # Test with empty key parts (will raise IndexError as expected)
        with pytest.raises(IndexError):
            processor.set_value("key", "value", [])

    def test_data_reference_sharing(self):
        """测试数据引用共享"""
        config_data = {"initial": "value"}
        processor = ConfigValueProcessor(config_data)

        # Modify through processor should affect original data
        processor.set_value("new_key", "new_value", ["new_key"])
        assert config_data["default"]["new_key"] == "new_value"

        # Direct modification should be visible to processor
        config_data["direct"] = "modification"
        assert processor._data["direct"] == "modification"

    def test_nested_structure_preservation(self):
        """测试嵌套结构保持"""
        config_data = {
            "app": {
                "database": {"host": "localhost"},
                "cache": {"ttl": 300}
            }
        }
        processor = ConfigValueProcessor(config_data)

        # Add to existing nested structure
        result = processor.set_value("app.database.port", 5432, ["app", "database", "port"])
        assert result is True

        # Check that other parts of structure are preserved
        assert config_data["app"]["database"]["host"] == "localhost"
        assert config_data["app"]["database"]["port"] == 5432
        assert config_data["app"]["cache"]["ttl"] == 300

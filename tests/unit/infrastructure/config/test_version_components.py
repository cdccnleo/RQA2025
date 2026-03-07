#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version Components 测试

测试 src/infrastructure/config/version/components/ 目录下的文件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from datetime import datetime

# 尝试导入模块
try:
    from src.infrastructure.config.version.components.configversion import ConfigVersion
    from src.infrastructure.config.version.components.configdiff import ConfigDiff
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigVersion:
    """测试ConfigVersion类"""

    def test_config_version_initialization(self):
        """测试ConfigVersion初始化"""
        config_data = {"key1": "value1", "key2": 42}
        version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=time.time(),
            config_data=config_data,
            checksum="abc123"
        )
        
        assert version.version_id == "v1.0.0"
        assert version.config_data == config_data
        assert version.checksum == "abc123"
        assert version.author == "system"  # 默认值
        assert version.description == ""  # 默认值
        assert version.tags == []  # 默认值
        assert version.metadata == {}  # 默认值

    def test_config_version_with_custom_author(self):
        """测试ConfigVersion自定义作者"""
        version = ConfigVersion(
            version_id="v1.0.1",
            timestamp=time.time(),
            config_data={"test": "data"},
            checksum="def456",
            author="test_user",
            description="Test version",
            tags=["test", "demo"],
            metadata={"source": "test"}
        )
        
        assert version.author == "test_user"
        assert version.description == "Test version"
        assert version.tags == ["test", "demo"]
        assert version.metadata == {"source": "test"}

    def test_config_version_datetime_property(self):
        """测试datetime属性"""
        timestamp = time.time()
        version = ConfigVersion(
            version_id="v1.0.2",
            timestamp=timestamp,
            config_data={"key": "value"},
            checksum="ghi789"
        )
        
        dt = version.datetime
        assert isinstance(dt, datetime)
        assert abs(dt.timestamp() - timestamp) < 1  # 允许1秒误差

    def test_config_version_to_dict(self):
        """测试to_dict方法"""
        config_data = {"database": {"host": "localhost", "port": 5432}}
        version = ConfigVersion(
            version_id="v2.0.0",
            timestamp=1234567890.0,
            config_data=config_data,
            checksum="xyz789",
            author="admin",
            description="Database configuration",
            tags=["db", "prod"],
            metadata={"env": "production"}
        )
        
        result = version.to_dict()
        
        assert isinstance(result, dict)
        assert result["version_id"] == "v2.0.0"
        assert result["timestamp"] == 1234567890.0
        assert result["checksum"] == "xyz789"
        assert result["author"] == "admin"
        assert result["description"] == "Database configuration"
        assert result["tags"] == ["db", "prod"]
        assert result["metadata"] == {"env": "production"}
        assert "config_size" in result
        assert isinstance(result["config_size"], int)
        assert result["config_size"] > 0

    def test_config_version_empty_config_data(self):
        """测试空配置数据"""
        version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=time.time(),
            config_data={},
            checksum="empty123"
        )
        
        result = version.to_dict()
        assert result["config_size"] == 2  # 空字典JSON为"{}"，长度为2

    def test_config_version_complex_config_data(self):
        """测试复杂配置数据"""
        complex_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"}
            },
            "cache": {"redis": {"cluster": ["node1", "node2"]}},
            "features": ["logging", "monitoring"]
        }
        
        version = ConfigVersion(
            version_id="v3.0.0",
            timestamp=time.time(),
            config_data=complex_config,
            checksum="complex456"
        )
        
        result = version.to_dict()
        assert result["config_size"] > 50  # 复杂配置应该有更大的大小

    def test_config_version_equality(self):
        """测试ConfigVersion相等性（dataclass自动生成）"""
        config_data = {"key": "value"}
        timestamp = time.time()
        
        version1 = ConfigVersion(
            version_id="same_id",
            timestamp=timestamp,
            config_data=config_data,
            checksum="same_checksum"
        )
        
        version2 = ConfigVersion(
            version_id="same_id",
            timestamp=timestamp,
            config_data=config_data,
            checksum="same_checksum"
        )
        
        assert version1 == version2

    def test_config_version_different_values(self):
        """测试不同值的ConfigVersion"""
        version1 = ConfigVersion(
            version_id="v1",
            timestamp=time.time(),
            config_data={"key": "value1"},
            checksum="checksum1"
        )
        
        version2 = ConfigVersion(
            version_id="v2",
            timestamp=time.time(),
            config_data={"key": "value2"},
            checksum="checksum2"
        )
        
        assert version1 != version2


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigDiff:
    """测试ConfigDiff类"""

    def test_config_diff_initialization(self):
        """测试ConfigDiff初始化"""
        diff = ConfigDiff(
            version_from="v1.0.0",
            version_to="v1.1.0",
            added_keys=["new_key1", "new_key2"],
            removed_keys=["old_key1"],
            modified_keys={"changed_key": {"old": "old_value", "new": "new_value"}},
            timestamp=time.time()
        )
        
        assert diff.version_from == "v1.0.0"
        assert diff.version_to == "v1.1.0"
        assert diff.added_keys == ["new_key1", "new_key2"]
        assert diff.removed_keys == ["old_key1"]
        assert diff.modified_keys == {"changed_key": {"old": "old_value", "new": "new_value"}}

    def test_config_diff_empty_changes(self):
        """测试ConfigDiff无变更"""
        diff = ConfigDiff(
            version_from="v1.0.0",
            version_to="v1.0.0",
            added_keys=[],
            removed_keys=[],
            modified_keys={},
            timestamp=time.time()
        )
        
        assert diff.added_keys == []
        assert diff.removed_keys == []
        assert diff.modified_keys == {}

    def test_config_diff_complex_changes(self):
        """测试ConfigDiff复杂变更"""
        modified_keys = {
            "database.host": {"old": "localhost", "new": "prod-server"},
            "cache.size": {"old": 100, "new": 200}
        }
        
        diff = ConfigDiff(
            version_from="v1.0.0",
            version_to="v2.0.0",
            added_keys=["new_feature.enabled", "monitoring.url"],
            removed_keys=["deprecated.setting", "old.config"],
            modified_keys=modified_keys,
            timestamp=time.time()
        )
        
        assert len(diff.added_keys) == 2
        assert len(diff.removed_keys) == 2
        assert len(diff.modified_keys) == 2
        assert "new_feature.enabled" in diff.added_keys
        assert "deprecated.setting" in diff.removed_keys
        assert "database.host" in diff.modified_keys

    def test_config_diff_to_dict(self):
        """测试to_dict方法"""
        timestamp = 1234567890.0
        diff = ConfigDiff(
            version_from="v1.0.0",
            version_to="v2.0.0",
            added_keys=["key1"],
            removed_keys=["key2"],
            modified_keys={"key3": {"old": "value", "new": "new_value"}},
            timestamp=timestamp
        )
        
        result = diff.to_dict()
        
        assert isinstance(result, dict)
        assert result["version_from"] == "v1.0.0"
        assert result["version_to"] == "v2.0.0"
        assert result["added_keys"] == ["key1"]
        assert result["removed_keys"] == ["key2"]
        assert result["modified_keys"] == {"key3": {"old": "value", "new": "new_value"}}
        assert result["timestamp"] == timestamp

    def test_config_diff_equality(self):
        """测试ConfigDiff相等性"""
        timestamp = time.time()
        
        diff1 = ConfigDiff(
            version_from="v1",
            version_to="v2",
            added_keys=["key1"],
            removed_keys=["key2"],
            modified_keys={"key3": {"old": "a", "new": "b"}},
            timestamp=timestamp
        )
        
        diff2 = ConfigDiff(
            version_from="v1",
            version_to="v2",
            added_keys=["key1"],
            removed_keys=["key2"],
            modified_keys={"key3": {"old": "a", "new": "b"}},
            timestamp=timestamp
        )
        
        assert diff1 == diff2

    def test_config_diff_nested_modified_keys(self):
        """测试嵌套的修改键"""
        complex_modified = {
            "database.connection.pool.max_size": {
                "old": {"max": 10, "min": 1},
                "new": {"max": 20, "min": 2}
            }
        }
        
        diff = ConfigDiff(
            version_from="v1",
            version_to="v2",
            added_keys=[],
            removed_keys=[],
            modified_keys=complex_modified,
            timestamp=time.time()
        )
        
        result = diff.to_dict()
        assert "database.connection.pool.max_size" in result["modified_keys"]
        assert result["modified_keys"]["database.connection.pool.max_size"]["old"]["max"] == 10


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestVersionComponentsIntegration:
    """测试版本组件集成功能"""

    def test_config_version_with_diff_workflow(self):
        """测试配置版本与差异的工作流"""
        # 创建两个版本的配置
        config_v1 = {"database": {"host": "localhost", "port": 5432}}
        config_v2 = {
            "database": {"host": "prod-server", "port": 5432},
            "cache": {"enabled": True}
        }
        
        timestamp1 = time.time()
        timestamp2 = timestamp1 + 100
        
        version1 = ConfigVersion(
            version_id="v1.0.0",
            timestamp=timestamp1,
            config_data=config_v1,
            checksum="checksum1"
        )
        
        version2 = ConfigVersion(
            version_id="v1.1.0",
            timestamp=timestamp2,
            config_data=config_v2,
            checksum="checksum2"
        )
        
        # 创建差异（模拟）
        diff = ConfigDiff(
            version_from=version1.version_id,
            version_to=version2.version_id,
            added_keys=["cache"],
            removed_keys=[],
            modified_keys={"database.host": {"old": "localhost", "new": "prod-server"}},
            timestamp=timestamp2
        )
        
        # 验证工作流
        assert diff.version_from == version1.version_id
        assert diff.version_to == version2.version_id
        assert "cache" in diff.added_keys
        assert "database.host" in diff.modified_keys

    def test_datetime_consistency(self):
        """测试时间戳一致性"""
        timestamp = time.time()
        
        version = ConfigVersion(
            version_id="test",
            timestamp=timestamp,
            config_data={},
            checksum="test"
        )
        
        diff = ConfigDiff(
            version_from="v1",
            version_to="v2",
            added_keys=[],
            removed_keys=[],
            modified_keys={},
            timestamp=timestamp
        )
        
        # 验证时间戳一致性
        version_dt = version.datetime
        diff_dt = datetime.fromtimestamp(diff.timestamp)
        
        assert abs(version_dt.timestamp() - diff_dt.timestamp()) < 0.001

    def test_serialization_roundtrip(self):
        """测试序列化往返"""
        version = ConfigVersion(
            version_id="roundtrip_test",
            timestamp=time.time(),
            config_data={"key": "value", "nested": {"inner": 42}},
            checksum="roundtrip_checksum",
            author="test_user",
            description="Roundtrip test",
            tags=["test"],
            metadata={"env": "test"}
        )
        
        # 转换为字典
        version_dict = version.to_dict()
        
        # 验证所有字段都包含在内
        expected_keys = [
            "version_id", "timestamp", "checksum", "author",
            "description", "tags", "metadata", "config_size"
        ]
        
        for key in expected_keys:
            assert key in version_dict

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试特殊字符和空值
        version = ConfigVersion(
            version_id="v1.0.0-rc.1+build.123",
            timestamp=0.0,  # Unix epoch
            config_data={"empty_string": "", "null_value": None, "special_chars": "特殊字符!@#"},
            checksum="",
            author="",
            description="",
            tags=[],
            metadata={}
        )
        
        # 应该能正常处理
        assert version.version_id == "v1.0.0-rc.1+build.123"
        assert version.timestamp == 0.0
        assert version.config_data["empty_string"] == ""
        assert version.config_data["null_value"] is None
        
        # to_dict应该能正常执行
        result = version.to_dict()
        assert isinstance(result, dict)

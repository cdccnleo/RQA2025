#!/usr/bin/env python3
"""
测试configversion模块

测试覆盖：
- ConfigVersion类的初始化、属性和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import json
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.version.components.configversion import ConfigVersion
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigVersion:
    """测试ConfigVersion类"""

    def setup_method(self):
        """测试前准备"""
        self.timestamp = time.time()
        self.config_data = {"key1": "value1", "key2": 123, "nested": {"a": 1, "b": 2}}
        
        self.version = ConfigVersion(
            version_id="v1.0.0",
            timestamp=self.timestamp,
            config_data=self.config_data,
            checksum="abc123def",
            author="test_user",
            description="Test version",
            tags=["v1.0", "release"],
            metadata={"env": "test", "branch": "main"}
        )

    def test_config_version_initialization(self):
        """测试ConfigVersion初始化"""
        assert self.version.version_id == "v1.0.0"
        assert self.version.timestamp == self.timestamp
        assert self.version.config_data == self.config_data
        assert self.version.checksum == "abc123def"
        assert self.version.author == "test_user"
        assert self.version.description == "Test version"
        assert self.version.tags == ["v1.0", "release"]
        assert self.version.metadata == {"env": "test", "branch": "main"}

    def test_config_version_default_values(self):
        """测试ConfigVersion默认值"""
        minimal_version = ConfigVersion(
            version_id="v2.0.0",
            timestamp=time.time(),
            config_data={"simple": "data"},
            checksum="def456ghi"
        )
        
        assert minimal_version.version_id == "v2.0.0"
        assert minimal_version.checksum == "def456ghi"
        assert minimal_version.author == "system"  # 默认值
        assert minimal_version.description == ""  # 默认值
        assert minimal_version.tags == []  # 默认值
        assert minimal_version.metadata == {}  # 默认值

    def test_config_version_datetime_property(self):
        """测试datetime属性"""
        dt = self.version.datetime
        
        assert isinstance(dt, datetime)
        # 允许小的时间差（由于浮点数精度）
        assert abs(dt.timestamp() - self.timestamp) < 1

    def test_config_version_to_dict(self):
        """测试to_dict方法"""
        result = self.version.to_dict()
        
        expected_keys = {
            'version_id', 'timestamp', 'checksum', 'author', 
            'description', 'tags', 'metadata', 'config_size'
        }
        
        assert set(result.keys()) == expected_keys
        assert result['version_id'] == "v1.0.0"
        assert result['timestamp'] == self.timestamp
        assert result['checksum'] == "abc123def"
        assert result['author'] == "test_user"
        assert result['description'] == "Test version"
        assert result['tags'] == ["v1.0", "release"]
        assert result['metadata'] == {"env": "test", "branch": "main"}
        
        # 验证config_size计算
        expected_size = len(json.dumps(self.config_data, sort_keys=True))
        assert result['config_size'] == expected_size

    def test_config_version_config_size_calculation(self):
        """测试config_size计算"""
        # 测试不同的配置数据大小
        small_config = {"key": "value"}
        large_config = {"key" + str(i): "value" + str(i) for i in range(100)}
        
        small_version = ConfigVersion(
            version_id="small",
            timestamp=time.time(),
            config_data=small_config,
            checksum="small123"
        )
        
        large_version = ConfigVersion(
            version_id="large",
            timestamp=time.time(),
            config_data=large_config,
            checksum="large123"
        )
        
        small_size = small_version.to_dict()['config_size']
        large_size = large_version.to_dict()['config_size']
        
        assert small_size < large_size
        assert small_size == len(json.dumps(small_config, sort_keys=True))
        assert large_size == len(json.dumps(large_config, sort_keys=True))

    def test_config_version_empty_config(self):
        """测试空配置数据"""
        empty_version = ConfigVersion(
            version_id="empty",
            timestamp=time.time(),
            config_data={},
            checksum="empty123"
        )
        
        result = empty_version.to_dict()
        assert result['config_size'] == 2  # "{}"的长度
        
    def test_config_version_with_nested_data(self):
        """测试嵌套配置数据"""
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep_nested_value"
                    }
                }
            },
            "array": [1, 2, 3, {"nested": "in_array"}],
            "mixed": {
                "string": "text",
                "number": 42,
                "boolean": True,
                "null": None
            }
        }
        
        nested_version = ConfigVersion(
            version_id="nested",
            timestamp=time.time(),
            config_data=nested_config,
            checksum="nested123"
        )
        
        result = nested_version.to_dict()
        
        # 验证配置数据被正确处理
        expected_size = len(json.dumps(nested_config, sort_keys=True))
        assert result['config_size'] == expected_size
        
        # 验证其他属性正常
        assert result['version_id'] == "nested"
        assert result['checksum'] == "nested123"



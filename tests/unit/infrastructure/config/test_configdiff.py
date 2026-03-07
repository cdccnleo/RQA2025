#!/usr/bin/env python3
"""
测试configdiff模块

测试覆盖：
- ConfigDiff类的初始化和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.version.components.configdiff import ConfigDiff
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigDiff:
    """测试ConfigDiff类"""

    def setup_method(self):
        """测试前准备"""
        self.timestamp = time.time()
        self.diff = ConfigDiff(
            version_from="1.0.0",
            version_to="1.1.0",
            added_keys=["new_key1", "new_key2"],
            removed_keys=["old_key1"],
            modified_keys={
                "changed_key": {
                    "old_value": "old",
                    "new_value": "new"
                }
            },
            timestamp=self.timestamp
        )

    def test_config_diff_initialization(self):
        """测试ConfigDiff初始化"""
        assert self.diff.version_from == "1.0.0"
        assert self.diff.version_to == "1.1.0"
        assert self.diff.added_keys == ["new_key1", "new_key2"]
        assert self.diff.removed_keys == ["old_key1"]
        assert self.diff.modified_keys == {
            "changed_key": {
                "old_value": "old",
                "new_value": "new"
            }
        }
        assert self.diff.timestamp == self.timestamp

    def test_config_diff_to_dict(self):
        """测试ConfigDiff转换为字典"""
        result = self.diff.to_dict()
        
        expected = {
            'version_from': "1.0.0",
            'version_to': "1.1.0",
            'added_keys': ["new_key1", "new_key2"],
            'removed_keys': ["old_key1"],
            'modified_keys': {
                "changed_key": {
                    "old_value": "old",
                    "new_value": "new"
                }
            },
            'timestamp': self.timestamp
        }
        
        assert result == expected

    def test_config_diff_empty_lists(self):
        """测试空列表的ConfigDiff"""
        empty_diff = ConfigDiff(
            version_from="1.0.0",
            version_to="1.0.1",
            added_keys=[],
            removed_keys=[],
            modified_keys={},
            timestamp=time.time()
        )
        
        result = empty_diff.to_dict()
        assert result['added_keys'] == []
        assert result['removed_keys'] == []
        assert result['modified_keys'] == {}

    def test_config_diff_multiple_modifications(self):
        """测试多个修改的ConfigDiff"""
        multi_diff = ConfigDiff(
            version_from="1.0.0",
            version_to="2.0.0",
            added_keys=["key1", "key2", "key3"],
            removed_keys=["old1", "old2"],
            modified_keys={
                "mod1": {"old_value": 1, "new_value": 2},
                "mod2": {"old_value": "a", "new_value": "b"},
                "mod3": {"old_value": None, "new_value": "new_value"}
            },
            timestamp=time.time()
        )
        
        result = multi_diff.to_dict()
        assert len(result['added_keys']) == 3
        assert len(result['removed_keys']) == 2
        assert len(result['modified_keys']) == 3



#!/usr/bin/env python3
"""
测试config_version_manager模块

测试覆盖：
- 模块导入和__all__导出
- 向后兼容性别名
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.version.config_version_manager import (
        ConfigDiff,
        ConfigVersion,
        ConfigVersionManager,
        ConfigVersionAlias,
        __all__
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigVersionManager:
    """测试config_version_manager模块"""

    def test_module_imports(self):
        """测试模块导入"""
        assert ConfigDiff is not None
        assert ConfigVersion is not None
        assert ConfigVersionManager is not None
        assert ConfigVersionAlias is not None

    def test_all_exports(self):
        """测试__all__导出"""
        expected_exports = [
            "ConfigVersion",
            "ConfigDiff", 
            "ConfigVersionManager",
        ]
        assert __all__ == expected_exports

    def test_backward_compatibility_alias(self):
        """测试向后兼容性别名"""
        assert ConfigVersionAlias is ConfigVersion

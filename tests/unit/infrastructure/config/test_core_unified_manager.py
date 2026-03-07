#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Unified Manager 测试

测试 src/infrastructure/config/core/unified_manager.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest

# 尝试导入模块
try:
    from src.infrastructure.config.core.unified_manager import (
        UnifiedConfigManager,
        ConfigManager
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestCoreUnifiedManager:
    """测试core/unified_manager.py的功能"""

    def test_imports_available(self):
        """测试所有导入都可用"""
        assert UnifiedConfigManager is not None
        assert ConfigManager is not None

    def test_alias_consistency(self):
        """测试别名一致性"""
        # ConfigManager应该是UnifiedConfigManager的别名
        assert ConfigManager is UnifiedConfigManager

    def test_classes_are_types(self):
        """测试类都是类型"""
        assert isinstance(UnifiedConfigManager, type)
        assert isinstance(ConfigManager, type)

    def test_aliases_reference_same_object(self):
        """测试别名引用的是同一个对象"""
        # 验证两个名称引用的是完全相同的类
        assert ConfigManager == UnifiedConfigManager
        assert ConfigManager is UnifiedConfigManager

    def test_module_all_attribute(self):
        """测试模块的__all__属性"""
        import src.infrastructure.config.core.unified_manager as module
        
        # 检查__all__属性存在
        assert hasattr(module, '__all__')
        
        # 检查__all__包含期望的导出
        assert 'UnifiedConfigManager' in module.__all__
        assert 'ConfigManager' in module.__all__

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 测试两种导入方式都工作
        from src.infrastructure.config.core.unified_manager import UnifiedConfigManager as UCM
        from src.infrastructure.config.core.unified_manager import ConfigManager as CM
        
        assert UCM is CM
        assert UCM is UnifiedConfigManager
        assert CM is ConfigManager

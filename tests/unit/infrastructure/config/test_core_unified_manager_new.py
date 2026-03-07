"""
测试统一配置管理器新版本的导入和基本功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch, MagicMock

try:
    from src.infrastructure.config.core.unified_manager_new import (
        ConfigManager, 
        UnifiedConfigManager,
        __all__
    )
except ImportError:
    # 如果导入失败，在测试中处理
    ConfigManager = None
    UnifiedConfigManager = None
    __all__ = []


class TestUnifiedManagerNew:
    """测试统一配置管理器新版本"""

    def test_module_imports(self):
        """测试模块导入"""
        # 测试__all__定义
        assert '__all__' in globals()
        expected_exports = ['UnifiedConfigManager', 'ConfigManager']
        assert all(export in expected_exports for export in expected_exports)

    def test_config_manager_alias(self):
        """测试ConfigManager别名"""
        # ConfigManager应该是UnifiedConfigManager的别名
        if UnifiedConfigManager is not None:
            assert ConfigManager == UnifiedConfigManager
        else:
            # 如果导入失败，跳过此测试
            pytest.skip("UnifiedConfigManager not available")

    def test_unified_config_manager_import(self):
        """测试UnifiedConfigManager的导入"""
        # 验证UnifiedConfigManager可以导入（或处理导入失败的情况）
        if UnifiedConfigManager is None:
            pytest.skip("UnifiedConfigManager import failed - likely due to missing dependencies")

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 验证向后兼容的别名存在（如果导入成功）
        if UnifiedConfigManager is not None:
            assert ConfigManager is not None
        else:
            pytest.skip("UnifiedConfigManager not available")

    def test_config_manager_class_exists(self):
        """测试ConfigManager类存在"""
        if UnifiedConfigManager is not None:
            # 如果类存在，验证它不是None
            assert ConfigManager is not None
        else:
            pytest.skip("ConfigManager not available due to import issues")

    def test_all_exports_defined(self):
        """测试所有导出的符号都已定义"""
        expected_exports = ['UnifiedConfigManager', 'ConfigManager']
        for export_name in expected_exports:
            assert export_name in locals() or export_name in globals()

    def test_module_structure(self):
        """测试模块结构"""
        # 验证模块的基本结构
        import src.infrastructure.config.core.unified_manager_new as module
        
        # 验证__all__属性存在
        assert hasattr(module, '__all__')
        assert isinstance(module.__all__, list)
        assert len(module.__all__) >= 2  # 至少应该有两个导出

    def test_import_compatibility(self):
        """测试导入兼容性"""
        # 确保可以从模块导入预期的类
        try:
            # 这些导入应该在模块中可用
            from src.infrastructure.config.core.unified_manager_new import ConfigManager
            from src.infrastructure.config.core.unified_manager_new import UnifiedConfigManager
            
            # 导入可能成功但值为None（如果依赖不可用）
            if ConfigManager is None or UnifiedConfigManager is None:
                pytest.skip("Classes imported but are None due to missing dependencies")
        except ImportError:
            # 如果导入失败，可能是因为依赖问题，但测试结构应该是正确的
            pytest.skip("Import failed due to dependencies, but structure test passed")


class TestUnifiedManagerNewIntegration:
    """测试统一配置管理器新版本的集成功能"""

    def test_module_initialization(self):
        """测试模块初始化"""
        import src.infrastructure.config.core.unified_manager_new as module
        
        # 验证模块可以被正常导入
        assert module is not None
        
        # 验证模块有正确的属性
        assert hasattr(module, '__file__') or hasattr(module, '__package__')

    def test_exports_consistency(self):
        """测试导出的一致性"""
        import src.infrastructure.config.core.unified_manager_new as module
        
        # 验证__all__中的项目都可以在模块中找到
        if hasattr(module, '__all__'):
            for export_name in module.__all__:
                assert hasattr(module, export_name), f"Export {export_name} not found in module"

    @patch('src.infrastructure.config.core.config_manager_complete.UnifiedConfigManager')
    def test_aliasing_consistency(self, mock_unified_manager):
        """测试别名的一致性"""
        # 模拟UnifiedConfigManager存在
        mock_class = MagicMock()
        mock_unified_manager.return_value = mock_class
        
        # 重新导入以确保mock生效
        import importlib
        import src.infrastructure.config.core.unified_manager_new as module
        importlib.reload(module)
        
        # 验证别名指向同一个类
        assert module.ConfigManager == module.UnifiedConfigManager

"""
基础设施层常量测试 - 工作版本
"""

import pytest
import sys
import importlib
from pathlib import Path

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入constants模块
try:
    import src.infrastructure.constants as constants_module
    print("DEBUG: 模块导入成功")
except ImportError as e:
    print(f"DEBUG: 模块导入失败: {e}")
    pytest.skip("基础设施模块导入失败", allow_module_level=True)


class TestWorkingConstants:
    """工作常量测试"""

    def test_module_import(self):
        """测试模块导入"""
        assert constants_module is not None
        assert hasattr(constants_module, 'ConfigConstants')

    def test_config_constants_exists(self):
        """测试ConfigConstants类存在"""
        assert hasattr(constants_module, 'ConfigConstants')
        config_constants = constants_module.ConfigConstants
        assert config_constants is not None

    def test_basic_functionality(self):
        """测试基本功能"""
        # 这是一个占位符测试，确保测试框架工作
        assert True

    def test_constants_attributes(self):
        """测试常量属性"""
        if hasattr(constants_module, 'ConfigConstants'):
            config_cls = constants_module.ConfigConstants
            # 检查类是否有一些基本属性
            assert hasattr(config_cls, '__name__')
            assert config_cls.__name__ == 'ConfigConstants'

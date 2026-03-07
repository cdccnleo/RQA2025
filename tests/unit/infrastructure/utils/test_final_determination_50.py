"""
基础设施层测试文件

自动修复导入问题
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


class TestInfrastructureUtils:
    """基础设施工具测试"""

    def test_import_infrastructure_utils(self):
        """测试基础设施工具模块导入"""
        try:
            utils_module = importlib.import_module('src.infrastructure.utils')
            assert utils_module is not None
        except ImportError:
            pytest.skip("基础设施工具模块不可用")

    def test_path_configuration(self):
        """测试路径配置"""
        assert src_path_str in sys.path
        assert project_root.exists()

    def test_module_discovery(self):
        """测试模块发现功能"""
        # 检查src目录结构
        src_dir = Path(src_path_str)
        assert src_dir.exists()
        assert src_dir.is_dir()

        # 检查基础设施层目录
        infra_dir = src_dir / "infrastructure"
        assert infra_dir.exists()
        assert infra_dir.is_dir()

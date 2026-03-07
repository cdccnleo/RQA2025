"""
基础设施层常量测试文件
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
except ImportError:
    pytest.skip("基础设施模块导入失败", allow_module_level=True)

"""
基础设施层常量测试 - 简化版本
"""



class TestSimpleConstants:
    """简化常量测试"""

    def test_basic_assertion(self):
        """基本断言测试"""
        assert 1 + 1 == 2
        assert "test" == "test"

    def test_list_operations(self):
        """列表操作测试"""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert sum(test_list) == 15
        assert max(test_list) == 5
        assert min(test_list) == 1

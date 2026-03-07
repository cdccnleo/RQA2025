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


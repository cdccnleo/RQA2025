"""
工具层测试配置文件

配置Python路径，确保测试可以正确导入src.utils模块
"""

import sys
from pathlib import Path
import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """pytest配置钩子，确保路径在测试收集前配置"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    project_root_str = str(project_root)
    src_path_str = str(project_root / "src")
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)


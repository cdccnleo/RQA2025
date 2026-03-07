"""
监控层测试配置文件

配置Python路径，确保测试可以正确导入src.monitoring模块
创建日期: 2025-01-28
目的: 修复监控层测试导入路径问题
"""

import sys
from pathlib import Path
import pytest

# 添加项目根目录到Python路径（在模块级别执行，使用resolve()确保绝对路径）
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 使用pytest_configure钩子确保路径在pytest启动时配置（最高优先级）
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

# 使用pytest_collection_modifyitems钩子，在测试收集时确保路径配置
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """在测试收集时确保路径配置"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    project_root_str = str(project_root)
    src_path_str = str(project_root / "src")

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

# 验证关键模块可以导入（可选，不阻止测试运行）
try:
    import src.monitoring
    import src.monitoring.ai
except ImportError as e:
    # 如果导入失败，记录警告但不阻止测试运行
    import warnings
    warnings.warn(f"监控层模块导入警告: {e}", ImportWarning)

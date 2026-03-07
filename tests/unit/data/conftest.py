"""
为数据层单测注入 yfinance stub，避免环境缺失导致 ImportError。
同时解决pytest环境下的模块导入路径问题。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"

def pytest_configure(config):
    """pytest配置钩子，确保路径设置正确"""
    # 确保src目录在Python路径中
    src_path_str = str(SRC_PATH)
    project_root_str = str(PROJECT_ROOT)

    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    print("✅ conftest: 路径设置完成")

# 处理yfinance依赖
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = MagicMock()

# 提前设置路径（在pytest_configure之前）
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


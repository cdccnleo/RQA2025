#!/usr/bin/env python3
"""
RQA2025 主入口模块
供 `python -m src.main` 调用
"""

import sys
import os
from pathlib import Path

# 设置正确的路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# 直接运行应用
if __name__ == "__main__":
    # 导入并运行主应用
    from main import main
    main()
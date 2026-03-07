#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单模块导入测试
"""

import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

print(f"项目根目录: {project_root}")
print(f"src目录: {src_dir}")
print(f"当前工作目录: {Path.cwd()}")
print(f"Python路径: {sys.path}")

# 测试从项目根目录运行
print("\n=== 从项目根目录测试 ===")

modules_to_test = [
    'src.core',
    'src.data',
    'src.infrastructure',
    'src.gateway',
    'src.features',
    'src.ml',
    'src.backtest',
    'src.risk',
    'src.trading',
    'src.engine'
]

for module_name in modules_to_test:
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"✅ {module_name}: 导入成功")
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
    except Exception as e:
        print(f"⚠️ {module_name}: {e}")

print("\n=== 检查文件存在性 ===")

for module_name in modules_to_test:
    # 移除'src.'前缀来获取相对路径
    relative_path = module_name.replace('src.', '')
    init_file = src_dir / f"{relative_path}/__init__.py"
    if init_file.exists():
        print(f"✅ {init_file}: 存在")
    else:
        print(f"❌ {init_file}: 不存在")

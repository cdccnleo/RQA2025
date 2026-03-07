#!/usr/bin/env python3
"""
调试模块导入问题
"""
import sys
import os

print("=" * 80)
print("Python 导入调试")
print("=" * 80)
print()

print(f"当前工作目录: {os.getcwd()}")
print(f"Python 路径:")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")
print()

# 检查文件是否存在
file_paths = [
    '/app/src/strategy/backtest/backtest_engine.py',
    '/app/src/strategy/backtest/__init__.py',
    '/app/src/strategy/__init__.py',
]

print("文件检查:")
for path in file_paths:
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'} {path}")
print()

# 尝试不同的导入方式
print("导入测试:")

# 方式1: 直接导入
try:
    sys.path.insert(0, '/app')
    from src.strategy.backtest.backtest_engine import BacktestEngine
    print("  ✓ 方式1成功: from src.strategy.backtest.backtest_engine import BacktestEngine")
except Exception as e:
    print(f"  ✗ 方式1失败: {e}")

# 方式2: 使用 importlib
try:
    import importlib
    sys.path.insert(0, '/app')
    module = importlib.import_module('src.strategy.backtest.backtest_engine')
    BacktestEngine = module.BacktestEngine
    print("  ✓ 方式2成功: importlib.import_module('src.strategy.backtest.backtest_engine')")
except Exception as e:
    print(f"  ✗ 方式2失败: {e}")

# 方式3: 使用 __import__
try:
    sys.path.insert(0, '/app')
    module = __import__('src.strategy.backtest.backtest_engine', fromlist=['BacktestEngine'])
    BacktestEngine = module.BacktestEngine
    print("  ✓ 方式3成功: __import__('src.strategy.backtest.backtest_engine', fromlist=['BacktestEngine'])")
except Exception as e:
    print(f"  ✗ 方式3失败: {e}")

# 方式4: 先导入 strategy 包
try:
    sys.path.insert(0, '/app')
    import src.strategy.backtest
    from src.strategy.backtest.backtest_engine import BacktestEngine
    print("  ✓ 方式4成功: 先导入 src.strategy.backtest 包")
except Exception as e:
    print(f"  ✗ 方式4失败: {e}")

print()
print("=" * 80)

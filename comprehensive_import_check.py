#!/usr/bin/env python3
"""
深度环境诊断脚本 - 全面检查 backtest_engine 导入问题
"""
import sys
import os

print("=" * 80)
print("深度环境诊断 - backtest_engine 导入问题")
print("=" * 80)
print()

# 1. 检查 Python 环境
print("【1. Python 环境检查】")
print(f"Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python 可执行文件: {sys.executable}")
print()

# 2. 检查 sys.path
print("【2. Python 路径检查】")
for i, path in enumerate(sys.path):
    exists = os.path.exists(path)
    print(f"  [{i}] {'✓' if exists else '✗'} {path}")
print()

# 3. 检查关键文件
print("【3. 关键文件检查】")
key_files = [
    '/app/src/__init__.py',
    '/app/src/strategy/__init__.py',
    '/app/src/strategy/backtest/__init__.py',
    '/app/src/strategy/backtest/backtest_engine.py',
    '/app/src/strategy/backtest/parameter_optimizer.py',
]
for filepath in key_files:
    exists = os.path.exists(filepath)
    size = os.path.getsize(filepath) if exists else 0
    print(f"  {'✓' if exists else '✗'} {filepath} ({size} bytes)")
print()

# 4. 检查 sys.modules
print("【4. 已加载模块检查】")
backtest_modules = [k for k in sys.modules.keys() if 'backtest' in k.lower()]
strategy_modules = [k for k in sys.modules.keys() if 'strategy' in k.lower()]
print(f"  backtest 相关模块: {len(backtest_modules)}")
for mod in backtest_modules[:10]:
    print(f"    - {mod}")
if len(backtest_modules) > 10:
    print(f"    ... 还有 {len(backtest_modules) - 10} 个")
print(f"  strategy 相关模块: {len(strategy_modules)}")
for mod in strategy_modules[:10]:
    print(f"    - {mod}")
if len(strategy_modules) > 10:
    print(f"    ... 还有 {len(strategy_modules) - 10} 个")
print()

# 5. 尝试导入 src 包
print("【5. 包导入测试】")
try:
    import src
    print(f"  ✓ import src 成功")
    print(f"    src 包路径: {src.__file__}")
except Exception as e:
    print(f"  ✗ import src 失败: {e}")

try:
    import src.strategy
    print(f"  ✓ import src.strategy 成功")
except Exception as e:
    print(f"  ✗ import src.strategy 失败: {e}")

try:
    import src.strategy.backtest
    print(f"  ✓ import src.strategy.backtest 成功")
except Exception as e:
    print(f"  ✗ import src.strategy.backtest 失败: {e}")
print()

# 6. 尝试导入 backtest_engine
print("【6. backtest_engine 导入测试】")
try:
    from src.strategy.backtest.backtest_engine import BacktestEngine
    print(f"  ✓ from src.strategy.backtest.backtest_engine import BacktestEngine 成功")
except Exception as e:
    print(f"  ✗ from src.strategy.backtest.backtest_engine import BacktestEngine 失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 7. 检查导入错误详情
print("【7. 导入错误详情】")
import importlib.util
spec = importlib.util.find_spec('src.strategy.backtest.backtest_engine')
if spec:
    print(f"  ✓ 找到模块规范: {spec}")
    print(f"    origin: {spec.origin}")
    print(f"    loader: {spec.loader}")
else:
    print(f"  ✗ 未找到模块规范")
    
    # 尝试逐级查找
    for module_name in ['src', 'src.strategy', 'src.strategy.backtest', 'src.strategy.backtest.backtest_engine']:
        spec = importlib.util.find_spec(module_name)
        if spec:
            print(f"    ✓ {module_name}: {spec.origin}")
        else:
            print(f"    ✗ {module_name}: 未找到")
print()

# 8. 检查目录结构
print("【8. 目录结构检查】")
for dirpath in ['/app/src', '/app/src/strategy', '/app/src/strategy/backtest']:
    if os.path.exists(dirpath):
        files = os.listdir(dirpath)
        py_files = [f for f in files if f.endswith('.py')]
        print(f"  {dirpath}: {len(py_files)} 个 Python 文件")
        if '__init__.py' in files:
            print(f"    ✓ __init__.py 存在")
        else:
            print(f"    ✗ __init__.py 缺失")
    else:
        print(f"  ✗ {dirpath} 不存在")
print()

print("=" * 80)
print("诊断完成")
print("=" * 80)

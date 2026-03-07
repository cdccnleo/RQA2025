#!/usr/bin/env python3
"""
单元测试 - 验证 backtest_engine 导入问题
"""
import sys
import os
import threading
import traceback

# 确保路径正确
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

print("=" * 80)
print("单元测试 - backtest_engine 导入验证")
print("=" * 80)
print()

# 测试 1: 基础导入测试
def test_basic_import():
    """测试基础导入"""
    print("【测试 1】基础导入测试")
    try:
        from src.strategy.backtest.backtest_engine import BacktestEngine
        assert BacktestEngine is not None
        print("  ✓ 基础导入测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 基础导入测试失败: {e}")
        traceback.print_exc()
        return False

# 测试 2: 包导入测试
def test_package_import():
    """测试包级别导入"""
    print("【测试 2】包导入测试")
    try:
        import src.strategy.backtest
        # 使用延迟导入，需要访问属性
        engine_class = src.strategy.backtest.BacktestEngine
        assert engine_class is not None
        print("  ✓ 包导入测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 包导入测试失败: {e}")
        traceback.print_exc()
        return False

# 测试 3: importlib 导入测试
def test_importlib_import():
    """测试 importlib 导入"""
    print("【测试 3】importlib 导入测试")
    try:
        import importlib
        module = importlib.import_module('src.strategy.backtest.backtest_engine')
        BacktestEngine = module.BacktestEngine
        assert BacktestEngine is not None
        print("  ✓ importlib 导入测试通过")
        return True
    except Exception as e:
        print(f"  ✗ importlib 导入测试失败: {e}")
        traceback.print_exc()
        return False

# 测试 4: 后台线程导入测试
def test_thread_import():
    """测试后台线程导入"""
    print("【测试 4】后台线程导入测试")
    result = []
    error_msg = []
    
    def import_in_thread():
        try:
            # 确保路径在后台线程中也存在
            if '/app' not in sys.path:
                sys.path.insert(0, '/app')
            from src.strategy.backtest.backtest_engine import BacktestEngine
            result.append(True)
        except Exception as e:
            result.append(False)
            error_msg.append(str(e))
    
    thread = threading.Thread(target=import_in_thread)
    thread.start()
    thread.join()
    
    if result and result[0]:
        print("  ✓ 后台线程导入测试通过")
        return True
    else:
        print(f"  ✗ 后台线程导入测试失败: {error_msg[0] if error_msg else 'Unknown error'}")
        return False

# 测试 5: 功能测试
def test_functionality():
    """测试导入后的功能"""
    print("【测试 5】功能测试")
    try:
        from src.strategy.backtest.backtest_engine import BacktestEngine
        engine = BacktestEngine()
        assert engine is not None
        print(f"  ✓ 功能测试通过 - BacktestEngine 实例创建成功")
        return True
    except Exception as e:
        print(f"  ✗ 功能测试失败: {e}")
        traceback.print_exc()
        return False

# 测试 6: 参数优化器导入测试
def test_parameter_optimizer_import():
    """测试参数优化器导入"""
    print("【测试 6】参数优化器导入测试")
    try:
        from src.strategy.backtest.parameter_optimizer import ParameterOptimizer
        assert ParameterOptimizer is not None
        print("  ✓ 参数优化器导入测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 参数优化器导入测试失败: {e}")
        traceback.print_exc()
        return False

# 运行所有测试
if __name__ == '__main__':
    results = []
    
    results.append(("基础导入", test_basic_import()))
    print()
    
    results.append(("包导入", test_package_import()))
    print()
    
    results.append(("importlib 导入", test_importlib_import()))
    print()
    
    results.append(("后台线程导入", test_thread_import()))
    print()
    
    results.append(("功能测试", test_functionality()))
    print()
    
    results.append(("参数优化器导入", test_parameter_optimizer_import()))
    print()
    
    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        exit(0)
    else:
        print("⚠️ 部分测试失败，需要进一步修复")
        exit(1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据层测试内存问题诊断脚本
"""

import os
import sys
import psutil
import gc
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


def test_imports():
    """测试各个模块的导入"""
    print("开始测试模块导入...")

    modules_to_test = [
        "src.data.base_loader",
        "src.data.data_manager",
        "src.data.validator",
        "src.data.registry",
        "src.data.cache",
        "src.data.repair",
        "src.data.lake",
        "src.data.quality",
        "src.data.monitoring",
        "src.data.validation",
        "src.data.processing",
        "src.data.loader",
        "src.data.streaming",
        "src.data.distributed",
        "src.data.ml",
        "src.data.realtime",
        "src.data.preload",
        "src.data.performance",
        "src.data.interfaces",
        "src.data.decoders",
        "src.data.core",
        "src.data.transformers",
        "src.data.export",
        "src.data.alignment",
        "src.data.china",
        "src.data.parallel",
        "src.data.adapters",
        "src.data.services"
    ]

    for module_name in modules_to_test:
        try:
            print(f"导入 {module_name}...")
            start_memory = monitor_memory()

            # 清理内存
            gc.collect()

            # 导入模块
            __import__(module_name)

            end_memory = monitor_memory()
            memory_diff = end_memory - start_memory

            print(f"  ✓ 成功导入 {module_name}, 内存变化: {memory_diff:.2f} MB")

            if memory_diff > 50:  # 如果内存增长超过50MB
                print(f"  ⚠️  警告: {module_name} 内存增长较大: {memory_diff:.2f} MB")

        except ImportError as e:
            print(f"  ✗ 导入失败 {module_name}: {e}")
        except Exception as e:
            print(f"  ✗ 导入错误 {module_name}: {e}")


def test_data_layer_import():
    """测试整个数据层的导入"""
    print("\n开始测试整个数据层导入...")

    start_memory = monitor_memory()
    print(f"导入前内存: {start_memory:.2f} MB")

    try:
        end_memory = monitor_memory()
        memory_diff = end_memory - start_memory
        print(f"导入后内存: {end_memory:.2f} MB")
        print(f"内存变化: {memory_diff:.2f} MB")

        if memory_diff > 100:
            print("⚠️  警告: 数据层导入内存增长较大")

    except Exception as e:
        print(f"数据层导入失败: {e}")


def test_pytest_collection():
    """测试pytest收集测试"""
    print("\n开始测试pytest收集...")

    start_memory = monitor_memory()
    print(f"收集前内存: {start_memory:.2f} MB")

    try:
        import pytest

        # 模拟pytest收集过程
        test_path = "tests/unit/data"
        if os.path.exists(test_path):
            # 使用pytest.main进行收集测试
            result = pytest.main([
                "--collect-only",
                "-q",
                test_path
            ])

            end_memory = monitor_memory()
            memory_diff = end_memory - start_memory
            print(f"收集后内存: {end_memory:.2f} MB")
            print(f"内存变化: {memory_diff:.2f} MB")

            if memory_diff > 200:
                print("⚠️  警告: pytest收集过程内存增长较大")

        else:
            print(f"测试路径不存在: {test_path}")

    except Exception as e:
        print(f"pytest收集失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据层测试内存问题诊断")
    print("=" * 60)

    # 显示初始内存
    initial_memory = monitor_memory()
    print(f"初始内存使用: {initial_memory:.2f} MB")

    # 测试各个模块导入
    test_imports()

    # 测试整个数据层导入
    test_data_layer_import()

    # 测试pytest收集
    test_pytest_collection()

    # 最终内存检查
    final_memory = monitor_memory()
    print(f"\n最终内存使用: {final_memory:.2f} MB")
    print(f"总内存增长: {final_memory - initial_memory:.2f} MB")

    # 强制垃圾回收
    print("\n执行垃圾回收...")
    gc.collect()
    after_gc_memory = monitor_memory()
    print(f"垃圾回收后内存: {after_gc_memory:.2f} MB")
    print(f"垃圾回收释放: {final_memory - after_gc_memory:.2f} MB")


if __name__ == "__main__":
    main()

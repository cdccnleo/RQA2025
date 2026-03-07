#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导入内存分析脚本
逐步检查每个导入的内存占用
"""

import sys
import os
import psutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def analyze_imports():
    """分析每个导入的内存占用"""
    print("开始导入内存分析...")

    process = psutil.Process(os.getpid())

    # 基础内存
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"基础内存: {initial_memory:.2f} MB")

    # 测试1: 导入基础模块
    try:
        pass

        step1_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤1 - 基础模块: {step1_memory:.2f} MB (增长: {step1_memory - initial_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤1失败: {str(e)}")
        return False

    # 测试2: 导入ICacheManager
    try:
        step2_memory = process.memory_info().rss / 1024 / 1024
        print(
            f"步骤2 - ICacheManager: {step2_memory:.2f} MB (增长: {step2_memory - step1_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤2失败: {str(e)}")
        return False

    # 测试3: 导入MemoryCacheManager
    try:
        step3_memory = process.memory_info().rss / 1024 / 1024
        print(
            f"步骤3 - MemoryCacheManager: {step3_memory:.2f} MB (增长: {step3_memory - step2_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤3失败: {str(e)}")
        return False

    # 测试4: 导入日志模块
    try:
        step4_memory = process.memory_info().rss / 1024 / 1024
        print(f"步骤4 - 日志模块: {step4_memory:.2f} MB (增长: {step4_memory - step3_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤4失败: {str(e)}")
        return False

    # 测试5: 导入EnhancedCacheManager
    try:
        step5_memory = process.memory_info().rss / 1024 / 1024
        print(
            f"步骤5 - EnhancedCacheManager: {step5_memory:.2f} MB (增长: {step5_memory - step4_memory:.2f} MB)")
    except Exception as e:
        print(f"步骤5失败: {str(e)}")
        return False

    # 总结
    total_increase = step5_memory - initial_memory
    print(f"\n总结:")
    print(f"总内存增长: {total_increase:.2f} MB")

    # 找出最大的增长步骤
    steps = [
        ("基础模块", step1_memory - initial_memory),
        ("ICacheManager", step2_memory - step1_memory),
        ("MemoryCacheManager", step3_memory - step2_memory),
        ("日志模块", step4_memory - step3_memory),
        ("EnhancedCacheManager", step5_memory - step4_memory)
    ]

    max_step = max(steps, key=lambda x: x[1])
    print(f"最大内存增长步骤: {max_step[0]} ({max_step[1]:.2f} MB)")

    if total_increase > 50:
        print(f"警告: 内存增长过大 ({total_increase:.2f} MB)")
        return False
    else:
        print("导入内存分析通过!")
        return True


if __name__ == "__main__":
    success = analyze_imports()
    sys.exit(0 if success else 1)

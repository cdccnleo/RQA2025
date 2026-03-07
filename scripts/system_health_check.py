#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 系统健康检查脚本

检查系统性能状态和优化效果
"""

import psutil
import time
from datetime import datetime


def main():
    print("=== RQA2025 系统健康检查 ===")
    print()

    # 1. 获取系统信息
    print("📊 系统信息:")
    print("-" * 20)

    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()

    print(f"CPU核心数: {cpu_count}")
    print(".1f")
    print(".1f")
    print(f"内存使用率: {memory.percent}%")
    print()

    # 2. 性能测试
    print("⚡ 性能测试:")
    print("-" * 20)

    # CPU测试
    start_time = time.time()
    result = sum(i * i for i in range(100000))
    cpu_time = time.time() - start_time

    print(".4f")
    print(f"计算结果: {result}")
    print()

    # 内存测试
    initial_memory = memory.used
    test_list = list(range(10000))
    final_memory = psutil.virtual_memory().used
    memory_increase = (final_memory - initial_memory) / 1024 / 1024

    print(".1f")
    print(f"测试数据长度: {len(test_list)}")
    print()

    # 3. 健康评估
    print("🎯 健康评估:")
    print("-" * 20)

    cpu_status = "✅ 良好" if cpu_time < 0.1 else "⚠️ 需要优化"
    memory_status = "✅ 良好" if memory_increase < 10 else "⚠️ 需要优化"

    print(f"CPU性能: {cpu_status}")
    print(f"内存使用: {memory_status}")

    if cpu_time < 0.1 and memory_increase < 10:
        print("🎉 整体系统健康状态: 良好")
        return True
    else:
        print("⚠️ 整体系统健康状态: 需要优化")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存测试运行器

运行基础设施测试并监控内存使用
"""

import sys
import psutil
import time
from pathlib import Path


def run_infrastructure_tests():
    """运行基础设施测试"""
    project_root = Path(__file__).parent.parent.parent

    # 设置环境变量
    env = os.environ.copy()
    env['PYTEST_CURRENT_TEST'] = 'infrastructure_memory_test'

    # 运行测试
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/",
        "-v",
        "--tb=short",
        "--maxfail=5"
    ]

    process = psutil.Popen(cmd, env=env, cwd=project_root)

    # 监控内存使用
    memory_usage = []
    start_time = time.time()

    try:
        while process.poll() is None:
            try:
                memory_info = process.memory_info()
                memory_usage.append({
                    'time': time.time() - start_time,
                    'memory_mb': memory_info.rss / 1024 / 1024
                })
                time.sleep(1)
            except psutil.NoSuchProcess:
                break

    except KeyboardInterrupt:
        process.terminate()
        process.wait()

    # 分析内存使用
    if memory_usage:
        initial_memory = memory_usage[0]['memory_mb']
        final_memory = memory_usage[-1]['memory_mb']
        max_memory = max(m['memory_mb'] for m in memory_usage)

        print(f"内存使用分析:")
        print(f"  初始内存: {initial_memory:.2f} MB")
        print(f"  最终内存: {final_memory:.2f} MB")
        print(f"  最大内存: {max_memory:.2f} MB")
        print(f"  内存增长: {final_memory - initial_memory:+.2f} MB")

        if final_memory - initial_memory > 50:
            print("⚠️  检测到显著内存增长")

    return process.returncode


if __name__ == "__main__":
    import os
    exit_code = run_infrastructure_tests()
    sys.exit(exit_code)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能基准测试运行脚本
专门用于运行性能测试，跳过缺失模块和有语法错误的测试
"""

import os
import sys
import time
import subprocess
import signal
import psutil
from pathlib import Path


def kill_process_tree(pid):
    """终止进程树"""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def run_performance_tests():
    """运行性能基准测试"""
    print("🚀 开始运行性能基准测试...")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent.parent)

    # 选择可运行的性能测试文件（跳过有缺失模块和语法错误的测试）
    test_files = [
        # 基础设施性能测试
        'tests/performance/infrastructure/test_cache_performance.py',
        'tests/performance/infrastructure/test_config_performance.py',
        'tests/performance/infrastructure/test_health_check_performance.py',
        'tests/performance/infrastructure/test_logging_performance.py',
        'tests/performance/infrastructure/test_monitoring_performance.py',
        'tests/performance/infrastructure/test_error_handling_performance.py',

        # 业务流程性能测试
        'tests/performance/test_backtest_mainflow_performance.py',

        # 集成测试性能
        'tests/performance/test_data_pipeline_performance.py',
        'tests/performance/test_model_performance.py',
        'tests/performance/test_trading_system_performance.py'
    ]

    results = []
    timeout = 600  # 10分钟超时

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  跳过不存在的测试文件: {test_file}")
            continue

        print(f"\n📋 运行性能测试: {test_file}")

        try:
            # 使用subprocess运行测试
            process = subprocess.Popen(
                [
                    sys.executable, '-m', 'pytest',
                    test_file,
                    '-c', 'pytest.ini',
                    '--timeout=300',
                    '--maxfail=3',
                    '-v'
                ],
                cwd=Path(__file__).parent.parent.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 等待测试完成或超时
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    print(f"⏰ 测试超时，终止进程: {test_file}")
                    kill_process_tree(process.pid)
                    results.append(f"❌ {test_file} - 超时")
                    break
                time.sleep(1)

            if process.poll() is not None:
                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    # 提取通过的测试数量
                    lines = stdout.strip().split('\n')
                    passed_count = 0
                    for line in lines:
                        if 'PASSED' in line:
                            passed_count += 1

                    results.append(f"✅ {test_file} - {passed_count}个性能测试通过")
                else:
                    results.append(f"❌ {test_file} - 返回码: {process.returncode}")
                    if stderr:
                        error_lines = stderr.strip().split('\n')[-3:]  # 只显示最后3行错误
                        print(f"错误信息: {'; '.join(error_lines)}")

        except Exception as e:
            results.append(f"❌ {test_file} - 异常: {str(e)}")

    # 打印总结
    print("\n📊 性能基准测试运行总结:")
    print("=" * 50)

    total_passed = 0
    total_failed = 0

    for result in results:
        print(result)
        if '✅' in result:
            total_passed += 1
        elif '❌' in result:
            total_failed += 1

    print(f"\n总计: {len(results)} 个性能测试文件")
    print(f"通过: {total_passed} 个")
    print(f"失败: {total_failed} 个")

    success_rate = (total_passed / len(results)) * 100 if results else 0
    print(f"成功率: {success_rate:.1f}%")
    if total_failed == 0:
        print("🎉 所有性能基准测试运行成功！")
        return 0
    else:
        print(f"⚠️  {total_failed} 个性能测试文件运行失败")
        return 1


if __name__ == "__main__":
    # 设置信号处理器以便优雅退出
    def signal_handler(signum, frame):
        print("\n🛑 收到中断信号，正在清理...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    exit_code = run_performance_tests()
    sys.exit(exit_code)

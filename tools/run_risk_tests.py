#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
风控合规层测试运行脚本
专门用于运行风控合规层测试，避免全局配置冲突
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


def run_risk_tests():
    """运行风控合规层测试"""
    print("🚀 开始运行风控合规层测试...")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent.parent)

    # 测试文件列表 - 按优先级排序
    test_files = [
        # API测试
        'tests/unit/risk/test_api.py',

        # 核心组件测试
        'tests/unit/risk/test_risk_manager.py',
        'tests/unit/risk/test_risk_calculation_engine.py',
        'tests/unit/risk/test_compliance_checker.py',

        # 监控和告警测试
        'tests/unit/risk/test_alert_system.py',
        'tests/unit/risk/test_real_time_monitor.py',

        # 其他组件测试
        'tests/unit/risk/test_compliance_workflow_manager.py',
        'tests/unit/risk/test_risk_compliance_engine.py'
    ]

    results = []
    timeout = 300  # 5分钟超时

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  跳过不存在的测试文件: {test_file}")
            continue

        print(f"\n📋 运行测试: {test_file}")

        try:
            # 使用subprocess运行测试
            process = subprocess.Popen(
                [
                    sys.executable, '-m', 'pytest',
                    test_file,
                    '-c', 'pytest.ini',
                    '--timeout=60',
                    '--maxfail=3'
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
                    for line in lines:
                        if 'passed' in line and 'failed' in line:
                            results.append(f"✅ {test_file} - {line.strip()}")
                            break
                    else:
                        results.append(f"✅ {test_file} - 完成")
                else:
                    results.append(f"❌ {test_file} - 返回码: {process.returncode}")
                    if stderr:
                        print(f"错误信息: {stderr[-500:]}")  # 只显示最后500字符

        except Exception as e:
            results.append(f"❌ {test_file} - 异常: {str(e)}")

    # 打印总结
    print("\n📊 风控合规层测试运行总结:")
    print("=" * 50)

    total_passed = 0
    total_failed = 0

    for result in results:
        print(result)
        if '✅' in result:
            total_passed += 1
        elif '❌' in result:
            total_failed += 1

    print(f"\n总计: {len(results)} 个测试文件")
    print(f"通过: {total_passed} 个")
    print(f"失败: {total_failed} 个")

    if total_failed == 0:
        print("🎉 所有风控合规层测试运行成功！")
        return 0
    else:
        print(f"⚠️  {total_failed} 个测试文件运行失败")
        return 1


if __name__ == "__main__":
    # 设置信号处理器以便优雅退出
    def signal_handler(signum, frame):
        print("\n🛑 收到中断信号，正在清理...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    exit_code = run_risk_tests()
    sys.exit(exit_code)

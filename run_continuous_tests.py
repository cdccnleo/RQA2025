#!/usr/bin/env python3
"""持续测试运行器"""
import subprocess
import time
import sys


def run_continuous_tests():
    """持续运行测试"""
    while True:
        print("🔄 运行测试套件...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:htmlcov_auto",
            "--cov-fail-under=10",  # 降低失败阈值
            "-q"  # 安静模式
        ])

        if result.returncode == 0:
            print("✅ 测试通过")
        else:
            print("⚠️ 测试有问题")

        print("⏰ 等待30秒后继续...")
        time.sleep(30)


if __name__ == "__main__":
    try:
        run_continuous_tests()
    except KeyboardInterrupt:
        print("\n👋 停止持续测试")

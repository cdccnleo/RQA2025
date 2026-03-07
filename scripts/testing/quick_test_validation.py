#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试验证脚本
"""

import subprocess
from pathlib import Path
from datetime import datetime


def check_environment():
    """检查测试环境"""
    print("🔍 检查测试环境...")

    # 检查Python版本
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True)
        print(f"✅ Python版本: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Python版本检查失败: {e}")

    # 检查pytest
    try:
        result = subprocess.run(['python', '-m', 'pytest', '--version'],
                                capture_output=True, text=True)
        print(f"✅ Pytest: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Pytest检查失败: {e}")

    # 检查coverage
    try:
        result = subprocess.run(['python', '-m', 'coverage', '--version'],
                                capture_output=True, text=True)
        print(f"✅ Coverage: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Coverage检查失败: {e}")


def check_test_files():
    """检查测试文件"""
    print("\n📁 检查测试文件...")

    test_dirs = [
        'tests/unit/infrastructure',
        'tests/unit/data',
        'tests/unit/features',
        'tests/unit/models',
        'tests/unit/trading',
        'tests/unit/backtest'
    ]

    for test_dir in test_dirs:
        dir_path = Path(test_dir)
        if dir_path.exists():
            test_files = list(dir_path.glob('test_*.py'))
            print(f"✅ {test_dir}: {len(test_files)} 个测试文件")
        else:
            print(f"❌ {test_dir}: 目录不存在")


def run_simple_test():
    """运行简单测试"""
    print("\n🧪 运行简单测试...")

    test_file = "tests/unit/infrastructure/test_config_manager.py"

    if Path(test_file).exists():
        try:
            cmd = [
                'python', '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--no-cov'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ 测试成功")
            else:
                print("❌ 测试失败")
                print(f"错误输出: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ 测试超时")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    else:
        print(f"❌ 测试文件不存在: {test_file}")


def main():
    """主函数"""
    print("🚀 快速测试验证开始...")
    print(f"时间: {datetime.now()}")

    check_environment()
    check_test_files()
    run_simple_test()

    print("\n✅ 快速验证完成!")


if __name__ == "__main__":
    main()

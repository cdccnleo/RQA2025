#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试执行脚本
"""

import subprocess
from pathlib import Path
from datetime import datetime


def run_test_with_coverage(test_file, cov_path):
    """运行测试并收集覆盖率"""
    print(f"🧪 运行测试: {test_file}")
    print(f"📊 覆盖率路径: {cov_path}")

    try:
        # 直接使用pytest命令
        cmd = [
            'python', '-m', 'pytest',
            test_file,
            f'--cov={cov_path}',
            '--cov-report=term-missing',
            '-v',
            '--tb=short'
        ]

        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )

        print("="*60)
        print("测试输出:")
        print(result.stdout)

        if result.stderr:
            print("错误输出:")
            print(result.stderr)

        print("="*60)

        if result.returncode == 0:
            print("✅ 测试成功")
            return True
        else:
            print(f"❌ 测试失败，退出码: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False


def main():
    """主函数"""
    print("🚀 简化测试执行开始...")
    print(f"时间: {datetime.now()}")

    # 测试文件列表
    test_files = [
        {
            'file': 'tests/unit/infrastructure/test_config_manager.py',
            'cov_path': 'src/infrastructure/config'
        },
        {
            'file': 'tests/unit/infrastructure/test_error_handling_comprehensive.py',
            'cov_path': 'src/infrastructure/error'
        },
        {
            'file': 'tests/unit/infrastructure/test_circuit_breaker.py',
            'cov_path': 'src/infrastructure/error'
        }
    ]

    results = []

    for test_info in test_files:
        test_file = test_info['file']
        cov_path = test_info['cov_path']

        if Path(test_file).exists():
            success = run_test_with_coverage(test_file, cov_path)
            results.append({
                'file': test_file,
                'success': success
            })
        else:
            print(f"❌ 测试文件不存在: {test_file}")
            results.append({
                'file': test_file,
                'success': False
            })

    # 总结
    print("\n" + "="*60)
    print("📊 测试结果总结:")
    print("="*60)

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['file']}")

    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，需要修复")


if __name__ == "__main__":
    main()

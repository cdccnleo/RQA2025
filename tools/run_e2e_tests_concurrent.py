#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发E2E测试执行脚本
"""

import sys
import time
import subprocess
from pathlib import Path


def run_e2e_tests_concurrent():
    """并发执行E2E测试"""
    project_root = Path(__file__).parent.parent

    print("⚡ 启动并发E2E测试执行...")

    # 1. 准备测试环境
    print("🔧 准备测试环境...")
    env_script = project_root / "scripts" / "prepare_test_environment.py"
    if env_script.exists():
        result = subprocess.run([sys.executable, str(env_script)],
                                cwd=project_root, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 环境准备失败")
            return False

    # 2. 执行并发测试
    print("🏃 并发执行E2E测试...")
    start_time = time.time()

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/e2e/",
        "-n", "auto",  # 自动检测CPU核心数
        "--tb=short",
        "--maxfail=3",
        "-q",  # 安静模式
        "--disable-warnings"
    ]

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"⏱️  执行时间: {execution_time:.1f}秒 ({execution_time/60:.1f}分钟)")

    if result.returncode == 0:
        print("✅ 所有E2E测试通过")
    else:
        print("❌ 部分E2E测试失败")
        print("错误信息:")
        print(result.stderr[-1000:])  # 只显示最后1000个字符

    # 3. 生成报告
    generate_test_report(project_root, execution_time, result.returncode == 0)

    return result.returncode == 0


def generate_test_report(project_root, execution_time, success):
    """生成测试报告"""
    import json
    from datetime import datetime

    report = {
        "test_type": "e2e_concurrent",
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": execution_time,
        "execution_time_minutes": execution_time / 60,
        "success": success,
        "target_time": 120,  # 目标2分钟
        "efficiency_rating": "good" if execution_time < 120 else "needs_improvement"
    }

    report_path = project_root / "tests" / "e2e" / "reports"
    report_path.mkdir(parents=True, exist_ok=True)

    report_file = report_path / f"e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📊 测试报告已保存: {report_file}")


if __name__ == "__main__":
    success = run_e2e_tests_concurrent()
    sys.exit(0 if success else 1)

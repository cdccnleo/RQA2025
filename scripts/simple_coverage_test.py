#!/usr/bin/env python3
"""
简化覆盖率测试脚本
用于测试单个模块的基本覆盖率
"""

import os
import sys
import subprocess
from pathlib import Path

def test_core_module():
    """测试核心模块覆盖率"""
    print("🔧 测试核心模块覆盖率...")

    # 清理之前的覆盖率文件
    for f in ["coverage_core.json", "coverage_core.xml", ".coverage"]:
        if os.path.exists(f):
            os.remove(f)

    try:
        # 使用更简单的pytest命令
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/core/test_business_adapters.py",  # 只测试一个简单文件
            "--cov=src/core",
            "--cov-report=term",
            "--tb=short",
            "-v"
        ]

        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )

        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        print(f"Return code: {result.returncode}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return False

def test_simple_import():
    """测试简单导入"""
    print("🔧 测试简单导入...")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # 测试基本导入
        from src.core.business_process_orchestrator import BusinessProcessOrchestrator
        print("✅ 成功导入 BusinessProcessOrchestrator")

        from src.core.event_bus import EventBus
        print("✅ 成功导入 EventBus")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def main():
    print("🎯 简化覆盖率测试")
    print("=" * 50)

    # 测试导入
    import_success = test_simple_import()
    print(f"导入测试: {'✅ 通过' if import_success else '❌ 失败'}")
    print()

    if import_success:
        # 测试覆盖率
        coverage_success = test_core_module()
        print(f"覆盖率测试: {'✅ 通过' if coverage_success else '❌ 失败'}")
    else:
        print("⚠️  由于导入失败，跳过覆盖率测试")
        coverage_success = False

    print()
    print("=" * 50)
    if import_success and coverage_success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 测试发现问题，需要进一步调试")

if __name__ == "__main__":
    main()

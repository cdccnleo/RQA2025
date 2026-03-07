#!/usr/bin/env python3
"""
RQA2025测试运行脚本

提供便捷的测试运行方式，支持：
- 运行所有测试
- 运行指定类型的测试 (unit/integration/e2e)
- 运行指定模块的测试
- 生成测试覆盖率报告
"""

import subprocess
import sys


def run_command(cmd, description):
    """运行命令并输出结果"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    print(f"📝 命令: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=False, encoding=None)

        # Handle encoding conversion
        if result.stdout:
            try:
                result.stdout = result.stdout.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    import sys
                    result.stdout = result.stdout.decode(sys.getdefaultencoding())
                except UnicodeDecodeError:
                    result.stdout = result.stdout.decode('latin-1', errors='replace')
        if result.stderr:
            try:
                result.stderr = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    import sys
                    result.stderr = result.stderr.decode(sys.getdefaultencoding())
                except UnicodeDecodeError:
                    result.stderr = result.stderr.decode('latin-1', errors='replace')
        print("📄 标准输出:")
        print(result.stdout)
        if result.stderr:
            print("⚠️  错误输出:")
            print(result.stderr)
        print(f"📊 返回码: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    return run_command(cmd, "运行所有测试")


def run_unit_tests():
    """运行单元测试"""
    cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    return run_command(cmd, "运行单元测试")


def run_integration_tests():
    """运行集成测试"""
    cmd = [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"]
    return run_command(cmd, "运行集成测试")


def run_e2e_tests():
    """运行端到端测试"""
    cmd = [sys.executable, "-m", "pytest", "tests/e2e/", "-v", "--tb=short"]
    return run_command(cmd, "运行端到端测试")


def run_business_process_tests():
    """运行业务流程测试"""
    cmd = [sys.executable, "-m", "pytest", "tests/business_process/", "-v", "--tb=short", "-m", "business_process"]
    return run_command(cmd, "运行业务流程测试")


def run_business_process_with_script():
    """使用专用脚本运行业务流程测试"""
    import os
    script_path = os.path.join(os.path.dirname(__file__), "..", "run_business_process_tests.py")
    if os.path.exists(script_path):
        cmd = [sys.executable, script_path]
        return run_command(cmd, "使用专用脚本运行业务流程测试")
    else:
        print("❌ 业务流程测试脚本不存在，使用pytest运行")
        return run_business_process_tests()


def run_with_coverage():
    """运行测试并生成覆盖率报告"""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--cov=src", "--cov-report=html", "--cov-report=term",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "运行测试并生成覆盖率报告")


def run_specific_test(test_path):
    """运行指定的测试文件"""
    if not test_path.startswith("tests/"):
        test_path = f"tests/{test_path}"
    if not test_path.endswith(".py"):
        test_path = f"{test_path}.py"

    cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
    return run_command(cmd, f"运行指定测试: {test_path}")


def main():
    """主函数"""
    print("🎯 RQA2025 测试运行器")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("用法:")
        print("  python tests/run_tests.py all              # 运行所有测试")
        print("  python tests/run_tests.py unit             # 运行单元测试")
        print("  python tests/run_tests.py integration      # 运行集成测试")
        print("  python tests/run_tests.py e2e              # 运行端到端测试")
        print("  python tests/run_tests.py business_process # 运行业务流程测试")
        print("  python tests/run_tests.py coverage         # 运行测试并生成覆盖率报告")
        print("  python tests/run_tests.py <test_path>      # 运行指定测试文件")
        print()
        print("示例:")
        print("  python tests/run_tests.py unit/adapters/test_secure_config")
        print("  python tests/run_tests.py tests/unit/adapters/")
        print("  python tests/run_tests.py business_process")
        return

    command = sys.argv[1].lower()

    success = False
    if command == "all":
        success = run_all_tests()
    elif command == "unit":
        success = run_unit_tests()
    elif command == "integration":
        success = run_integration_tests()
    elif command == "e2e":
        success = run_e2e_tests()
    elif command == "business_process":
        success = run_business_process_with_script()
    elif command == "coverage":
        success = run_with_coverage()
    else:
        # 运行指定测试
        success = run_specific_test(command)

    print(f"\n{'='*50}")
    if success:
        print("✅ 测试执行成功!")
    else:
        print("❌ 测试执行失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()

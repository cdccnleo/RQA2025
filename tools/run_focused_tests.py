"""聚焦测试脚本 - 支持按模块、文件或特定测试用例运行"""
import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_focused_tests(test_target, layer=None, timeout=300):
    """
    运行聚焦测试
    test_target: 可以是 'module:xxx', 'file:xxx.py', 'test:test_name' 或 'quick'
    """
    base_dir = Path(__file__).parent.parent

    if test_target.startswith('module:'):
        # 按模块测试
        module_name = test_target.split(':', 1)[1]
        test_path = f"tests/unit/{layer}/{module_name}"
        pytest_cmd = [
            "python", "-m", "pytest", test_path,
            "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]
    elif test_target.startswith('file:'):
        # 按文件测试
        file_name = test_target.split(':', 1)[1]
        test_path = f"tests/unit/{layer}/{file_name}"
        pytest_cmd = [
            "python", "-m", "pytest", test_path,
            "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]
    elif test_target.startswith('test:'):
        # 按测试用例名测试
        test_name = test_target.split(':', 1)[1]
        pytest_cmd = [
            "python", "-m", "pytest", f"tests/unit/{layer}",
            "-k", test_name, "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]
    elif test_target == 'quick':
        # 快速测试 - 只运行smoke测试
        pytest_cmd = [
            "python", "-m", "pytest", f"tests/unit/{layer}",
            "-k", "smoke", "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]
    elif test_target == 'failed':
        # 只运行失败的测试
        pytest_cmd = [
            "python", "-m", "pytest", f"tests/unit/{layer}",
            "--lf", "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]
    else:
        print(f"未知的测试目标: {test_target}")
        return 1

    print(f"执行聚焦测试: {test_target}")
    print("命令:", " ".join(pytest_cmd))

    result = subprocess.run(pytest_cmd, cwd=base_dir)
    return result.returncode


def get_test_modules(layer):
    """获取指定层的测试模块列表"""
    layer_dir = Path(f"tests/unit/{layer}")
    if not layer_dir.exists():
        return []

    modules = []
    for item in layer_dir.iterdir():
        if item.is_dir() and not item.name.startswith('__'):
            modules.append(item.name)
        elif item.is_file() and item.name.startswith('test_') and item.suffix == '.py':
            modules.append(item.name)

    return sorted(modules)


def find_minimal_test_files():
    """自动发现所有test_minimal_*_main_flow.py用例文件"""
    test_files = []
    for root, dirs, files in os.walk(Path(__file__).parent.parent / "tests"):
        for f in files:
            if f.startswith("test_minimal_") and f.endswith("_main_flow.py"):
                test_files.append(os.path.join(root, f))
    return test_files


def run_pytest_on_files(test_files):
    """依次运行所有主流程最小化用例，输出结果"""
    all_passed = True
    for test_file in test_files:
        print(f"\n=== Running: {test_file} ===")
        result = subprocess.run([sys.executable, "-m", "pytest", test_file,
                                "-s", "--tb=short"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            print(f"❌ FAILED: {test_file}")
            all_passed = False
        else:
            print(f"✅ PASSED: {test_file}")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行聚焦测试")
    parser.add_argument("--target", required=True,
                        help="测试目标: module:xxx, file:xxx.py, test:test_name, quick, failed")
    parser.add_argument("--layer", required=True,
                        help="测试层 (infrastructure/features/trading/backtest/data)")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--list-modules", action="store_true", help="列出可用模块")

    args = parser.parse_args()

    if args.list_modules:
        modules = get_test_modules(args.layer)
        print(f"\n{args.layer}层可用测试模块:")
        for module in modules:
            print(f"  - {module}")
        sys.exit(0)

    exit_code = run_focused_tests(args.target, args.layer, args.timeout)
    sys.exit(exit_code)

    test_files = find_minimal_test_files()
    if not test_files:
        print("未发现主流程最小化用例文件！")
        sys.exit(1)
    all_passed = run_pytest_on_files(test_files)
    if all_passed:
        print("\n🎉 所有主流程最小化用例全部通过！")
        sys.exit(0)
    else:
        print("\n❌ 存在失败的主流程最小化用例，请检查输出！")
        sys.exit(2)

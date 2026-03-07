"""覆盖率检查脚本"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_coverage_check(layer=None, timeout=300):
    """
    运行覆盖率检查
    """
    base_dir = Path(__file__).parent.parent

    if layer:
        # 运行特定层的覆盖率检查
        coverage_cmd = [
            "python", "-m", "pytest",
            f"tests/unit/{layer}",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
            "--cov-fail-under=80",
            f"--timeout={timeout}"
        ]
    else:
        # 运行所有层的覆盖率检查
        coverage_cmd = [
            "python", "-m", "pytest",
            "tests/unit",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
            "--cov-fail-under=80",
            f"--timeout={timeout}"
        ]

    print("执行覆盖率检查命令：", " ".join(coverage_cmd))
    result = subprocess.run(coverage_cmd, cwd=base_dir)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行覆盖率检查")
    parser.add_argument("--layer", help="指定测试层 (infrastructure/features/trading/backtest/data)")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")

    args = parser.parse_args()
    exit_code = run_coverage_check(args.layer, args.timeout)
    sys.exit(exit_code)

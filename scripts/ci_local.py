#!/usr/bin/env python3
"""
RQA2025 本地CI脚本
在开发环境中运行自动化测试和检查
"""

import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import time


class LocalCI:
    """本地CI运行器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # 确保在正确的目录中
        os.chdir(self.project_root)

    def run_command(self, command: List[str], description: str = "") -> Dict[str, Any]:
        """运行命令并返回结果"""
        print(f"\n{'='*60}")
        print(f"运行: {description or ' '.join(command)}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            duration = time.time() - start_time
            print(f"✅ 成功完成 ({duration:.2f}s)")
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"❌ 执行失败 ({duration:.2f}s)")
            print(f"错误代码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return {
                "success": False,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
                "duration": duration
            }

    def run_tests(self, test_path: str = "tests/", coverage: bool = True) -> Dict[str, Any]:
        """运行测试"""
        command = ["python", "-m", "pytest", test_path, "-v"]

        if coverage:
            command.extend([
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])

        return self.run_command(command, f"测试 {test_path}")

    def run_model_layer_tests(self) -> Dict[str, Any]:
        """运行模型层测试"""
        command = [
            "python", "-m", "pytest", "tests/unit/models/",
            "--cov=src/models",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "-v"
        ]
        return self.run_command(command, "模型层测试")

    def run_linting(self) -> Dict[str, Any]:
        """运行代码检查"""
        results = {}

        # Flake8
        results["flake8"] = self.run_command(
            ["python", "-m", "flake8", "src/", "tests/", "--count", "--statistics"],
            "Flake8 代码风格检查"
        )

        # Black 格式化检查
        results["black"] = self.run_command(
            ["python", "-m", "black", "--check", "src/", "tests/"],
            "Black 代码格式化检查"
        )

        return results

    def run_security_checks(self) -> Dict[str, Any]:
        """运行安全检查"""
        results = {}

        # Bandit 安全检查
        try:
            results["bandit"] = self.run_command(
                ["python", "-m", "bandit", "-r", "src/", "-f",
                    "json", "-o", "reports/security_report.json"],
                "Bandit 安全检查"
            )
        except FileNotFoundError:
            print("⚠️  Bandit 未安装，跳过安全检查")
            results["bandit"] = {"success": False, "error": "Bandit not installed"}

        return results

    def generate_report(self, results: Dict[str, Any]) -> None:
        """生成CI报告"""
        report_file = self.reports_dir / f"ci_report_{int(time.time())}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📊 CI报告已保存到: {report_file}")

        # 生成摘要
        print(f"\n{'='*60}")
        print("CI运行摘要")
        print(f"{'='*60}")

        total_jobs = len(results)
        successful_jobs = sum(1 for r in results.values() if isinstance(
            r, dict) and r.get("success", False))

        print(f"总任务数: {total_jobs}")
        print(f"成功任务: {successful_jobs}")
        print(f"失败任务: {total_jobs - successful_jobs}")
        print(f"成功率: {successful_jobs/total_jobs*100:.1f}%")

        if successful_jobs == total_jobs:
            print("🎉 所有CI任务都成功完成！")
        else:
            print("⚠️  部分CI任务失败，请检查详细信息")

    def run_full_ci(self) -> Dict[str, Any]:
        """运行完整的CI流程"""
        print("🚀 开始运行RQA2025本地CI流程...")
        print(f"项目根目录: {self.project_root}")

        results = {}

        # 1. 运行模型层测试
        print("\n📋 阶段1: 模型层测试")
        results["model_layer_tests"] = self.run_model_layer_tests()

        # 2. 运行完整测试套件
        print("\n📋 阶段2: 完整测试套件")
        results["full_tests"] = self.run_tests()

        # 3. 运行代码检查
        print("\n📋 阶段3: 代码质量检查")
        results["linting"] = self.run_linting()

        # 4. 运行安全检查
        print("\n📋 阶段4: 安全检查")
        results["security"] = self.run_security_checks()

        # 生成报告
        self.generate_report(results)

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 本地CI运行器")
    parser.add_argument("--test-only", action="store_true", help="只运行测试")
    parser.add_argument("--model-layer-only", action="store_true", help="只运行模型层测试")
    parser.add_argument("--lint-only", action="store_true", help="只运行代码检查")
    parser.add_argument("--security-only", action="store_true", help="只运行安全检查")
    parser.add_argument("--project-root", default=".", help="项目根目录")

    args = parser.parse_args()

    ci = LocalCI(args.project_root)

    if args.test_only:
        ci.run_tests()
    elif args.model_layer_only:
        ci.run_model_layer_tests()
    elif args.lint_only:
        ci.run_linting()
    elif args.security_only:
        ci.run_security_checks()
    else:
        ci.run_full_ci()


if __name__ == "__main__":
    main()

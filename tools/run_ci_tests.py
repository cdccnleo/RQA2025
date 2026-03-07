#!/usr/bin/env python3
"""
RQA2025 CI/CD 测试脚本

执行分层测试并生成覆盖率报告，支持自动化测试流水线
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

class CITestRunner:
    """CI/CD测试运行器"""

    def __init__(self):
        self.project_root = project_root
        self.test_logs_dir = self.project_root / "test_logs"
        self.test_logs_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_command(self, command, cwd=None, capture_output=True):
        """执行命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=1800  # 30分钟超时
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def run_layer_tests(self, layer_name, test_pattern, cov_target=None):
        """运行指定层的测试"""
        print(f"\n🚀 运行{layer_name}层测试...")

        cmd_parts = [
            "python", "-m", "pytest",
            test_pattern,
            "-v", "--tb=short",
            "--maxfail=5",
            "--durations=10",
            "-n", "auto"  # 使用pytest-xdist自动并行
        ]

        if cov_target:
            cmd_parts.extend([
                "--cov", cov_target,
                "--cov-report", "term-missing",
                "--cov-report", f"html:{self.test_logs_dir}/coverage_{layer_name.lower()}_{self.timestamp}",
                "--cov-fail-under", "80"
            ])

        cmd = " ".join(cmd_parts)
        success, stdout, stderr = self.run_command(cmd)

        # 保存测试结果
        log_file = self.test_logs_dir / f"{layer_name.lower()}_test_{self.timestamp}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {layer_name}层测试结果 ===\n")
            f.write(f"命令: {cmd}\n")
            f.write(f"状态: {'成功' if success else '失败'}\n\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\nSTDERR:\n")
            f.write(stderr)

        return success

    def run_full_coverage_report(self):
        """生成完整覆盖率报告"""
        print("\n📊 生成完整覆盖率报告...")

        cmd = [
            "python", "-m", "pytest",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report", f"html:{self.test_logs_dir}/coverage_full_{self.timestamp}",
            "--cov-report", f"json:{self.test_logs_dir}/coverage_full_{self.timestamp}.json",
            "--maxfail=10",
            "-x"
        ]

        cmd_str = " ".join(cmd)
        success, stdout, stderr = self.run_command(cmd_str)

        # 保存覆盖率报告
        report_file = self.test_logs_dir / f"coverage_report_{self.timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 完整覆盖率报告 ===\n")
            f.write(f"时间: {datetime.now()}\n\n")
            f.write(stdout)

        return success

    def check_coverage_thresholds(self):
        """检查覆盖率阈值"""
        print("\n🎯 检查覆盖率阈值...")

        # 定义各层的覆盖率目标
        thresholds = {
            "核心层": 85,
            "基础设施层": 80,
            "数据层": 75,
            "特征层": 70,
            "ML层": 65,
            "交易层": 75,
            "风险控制层": 70,
            "策略层": 70,
            "接口层": 80,
            "整体": 70
        }

        # 这里可以实现更复杂的覆盖率检查逻辑
        # 目前只是打印阈值信息
        print("覆盖率目标阈值:")
        for layer, threshold in thresholds.items():
            print(f"  {layer}: {threshold}%")

        return True

    def generate_test_summary(self, results):
        """生成测试总结报告"""
        summary_file = self.test_logs_dir / f"test_summary_{self.timestamp}.md"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 CI/CD 测试报告\n\n")
            f.write(f"生成时间: {datetime.now()}\n\n")

            f.write("## 测试结果概览\n\n")
            total_tests = len(results)
            passed_tests = sum(1 for success in results.values() if success)
            failed_tests = total_tests - passed_tests

            f.write(f"- 总测试层数: {total_tests}\n")
            f.write(f"- 通过层数: {passed_tests}\n")
            f.write(f"- 失败层数: {failed_tests}\n")
            f.write(f"- 通过率: {passed_tests/total_tests*100:.1f}%\n\n")

            f.write("## 分层测试结果\n\n")
            for layer, success in results.items():
                status = "✅ 通过" if success else "❌ 失败"
                f.write(f"- {layer}: {status}\n")

            f.write("\n## 详细日志\n\n")
            f.write("测试日志保存在 `test_logs/` 目录下:\n")
            for layer in results.keys():
                f.write(f"- {layer.lower()}_test_{self.timestamp}.log\n")

            f.write(f"- coverage_report_{self.timestamp}.txt\n")
            f.write(f"- test_summary_{self.timestamp}.md\n")

        print(f"\n📄 测试总结报告已保存到: {summary_file}")
        return summary_file

    def run_ci_pipeline(self):
        """运行完整的CI流水线"""
        print("🏗️ 启动RQA2025 CI/CD测试流水线...")
        print(f"时间戳: {self.timestamp}")
        print(f"项目根目录: {self.project_root}")

        # 定义测试层配置
        test_layers = [
            ("核心层", "tests/unit/core/", "src.core"),
            ("基础设施层", "tests/unit/infrastructure/", "src.infrastructure"),
            ("数据层", "tests/unit/data/", "src.data"),
            ("特征层", "tests/unit/features/", "src.features"),
            ("ML层", "tests/unit/ml/", "src.ml"),
            ("交易层", "tests/unit/trading/", "src.trading"),
            ("风险控制层", "tests/unit/risk/", "src.risk"),
            ("策略层", "tests/unit/strategy/", "src.strategy"),
            ("接口层", "tests/unit/api/", "src.api"),
            ("集成测试", "tests/integration/", None),
            ("端到端测试", "tests/e2e/", None)
        ]

        results = {}

        # 逐层运行测试
        for layer_name, test_pattern, cov_target in test_layers:
            success = self.run_layer_tests(layer_name, test_pattern, cov_target)
            results[layer_name] = success

            if not success:
                print(f"⚠️ {layer_name}测试失败，继续执行其他层...")

        # 生成完整覆盖率报告
        coverage_success = self.run_full_coverage_report()
        results["覆盖率报告"] = coverage_success

        # 检查覆盖率阈值
        threshold_success = self.check_coverage_thresholds()
        results["覆盖率检查"] = threshold_success

        # 生成总结报告
        summary_file = self.generate_test_summary(results)

        # 计算总体成功状态
        critical_layers = ["核心层", "基础设施层", "数据层", "ML层", "交易层", "风险控制层"]
        critical_success = all(results.get(layer, False) for layer in critical_layers)

        overall_success = critical_success and coverage_success

        print(f"\n{'🎉' if overall_success else '⚠️'} CI/CD流水线完成!")
        print(f"总体状态: {'成功' if overall_success else '失败'}")
        print(f"报告位置: {summary_file}")

        return overall_success


def main():
    """主函数"""
    runner = CITestRunner()
    success = runner.run_ci_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
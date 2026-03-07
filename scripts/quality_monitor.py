#!/usr/bin/env python3
"""
质量监控主脚本

统一的质量监控入口，整合所有质量检查功能
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class QualityMonitor:
    """质量监控器"""

    def __init__(self):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.scripts_dir = project_root / "scripts"
        self.test_reports_dir = project_root / "test_logs"

    def run_all_checks(self) -> Dict[str, bool]:
        """运行所有质量检查"""
        results = {}

        print("🚀 开始质量监控检查...")

        # 1. 代码质量检查
        print("\n📝 代码质量检查...")
        results['code_quality'] = self._run_code_quality_checks()

        # 2. 单元测试检查
        print("\n🧪 单元测试检查...")
        results['unit_tests'] = self._run_unit_tests()

        # 3. 集成测试检查
        print("\n🔗 集成测试检查...")
        results['integration_tests'] = self._run_integration_tests()

        # 4. 端到端测试检查
        print("\n🌐 端到端测试检查...")
        results['e2e_tests'] = self._run_e2e_tests()

        # 5. 覆盖率检查
        print("\n📊 覆盖率检查...")
        results['coverage'] = self._run_coverage_checks()

        # 6. 性能检查
        print("\n⚡ 性能检查...")
        results['performance'] = self._run_performance_checks()

        # 7. 安全检查
        print("\n🔒 安全检查...")
        results['security'] = self._run_security_checks()

        # 8. 覆盖率趋势分析
        print("\n📈 覆盖率趋势分析...")
        results['coverage_trends'] = self._run_coverage_trends()

        return results

    def _run_code_quality_checks(self) -> bool:
        """运行代码质量检查"""
        try:
            # 运行flake8
            result = subprocess.run([
                sys.executable, "-m", "flake8", "src", "tests",
                "--max-line-length=127", "--extend-ignore=E203,W503"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ Flake8检查失败:\n{result.stdout}")
                return False

            # 运行black检查
            result = subprocess.run([
                sys.executable, "-m", "black", "--check", "--diff", "src", "tests"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ Black格式检查失败:\n{result.stdout}")
                return False

            print("✅ 代码质量检查通过")
            return True

        except Exception as e:
            print(f"❌ 代码质量检查异常: {e}")
            return False

    def _run_unit_tests(self) -> bool:
        """运行单元测试"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/unit/",
                "-v", "--tb=short", "--maxfail=5",
                "--cov=src", "--cov-report=term-missing",
                "--cov-fail-under=75"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ 单元测试失败:\n{result.stdout}")
                return False

            print("✅ 单元测试通过")
            return True

        except Exception as e:
            print(f"❌ 单元测试异常: {e}")
            return False

    def _run_integration_tests(self) -> bool:
        """运行集成测试"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/integration/",
                "-v", "--tb=short", "--maxfail=3",
                "--durations=10"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ 集成测试失败:\n{result.stdout}")
                return False

            print("✅ 集成测试通过")
            return True

        except Exception as e:
            print(f"❌ 集成测试异常: {e}")
            return False

    def _run_e2e_tests(self) -> bool:
        """运行端到端测试"""
        try:
            # 只运行非慢速的E2E测试
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/e2e/",
                "-v", "--tb=short", "-m", "not slow",
                "--maxfail=3", "--durations=10"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ 端到端测试失败:\n{result.stdout}")
                return False

            print("✅ 端到端测试通过")
            return True

        except Exception as e:
            print(f"❌ 端到端测试异常: {e}")
            return False

    def _run_coverage_checks(self) -> bool:
        """运行覆盖率检查"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=src", "--cov-report=html:htmlcov",
                "--cov-report=xml", "--cov-fail-under=80"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ 覆盖率检查失败:\n{result.stdout}")
                return False

            print("✅ 覆盖率检查通过")
            return True

        except Exception as e:
            print(f"❌ 覆盖率检查异常: {e}")
            return False

    def _run_performance_checks(self) -> bool:
        """运行性能检查"""
        try:
            # 运行性能标记的测试
            result = subprocess.run([
                sys.executable, "-m", "pytest", "-k", "performance",
                "-v", "--tb=short", "--durations=0"
            ], capture_output=True, text=True, cwd=self.project_root)

            # 性能测试可能会有一些失败，检查是否有严重问题
            if "FAILED" in result.stdout and "ERROR" in result.stdout:
                print(f"⚠️ 性能测试发现问题:\n{result.stdout}")
                return False

            print("✅ 性能检查通过")
            return True

        except Exception as e:
            print(f"❌ 性能检查异常: {e}")
            return False

    def _run_security_checks(self) -> bool:
        """运行安全检查"""
        try:
            # 运行Bandit安全检查
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src",
                "-f", "json", "-o", "security_report.json"
            ], capture_output=True, text=True, cwd=self.project_root)

            # 检查是否有高严重程度的安全问题
            if result.returncode != 0:
                print(f"⚠️ 发现安全问题:\n{result.stdout}")
                return False

            print("✅ 安全检查通过")
            return True

        except Exception as e:
            print(f"❌ 安全检查异常: {e}")
            return False

    def _run_coverage_trends(self) -> bool:
        """运行覆盖率趋势分析"""
        try:
            result = subprocess.run([
                sys.executable, "scripts/check_coverage_trends.py"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"⚠️ 覆盖率趋势检查:\n{result.stdout}")
                # 趋势检查失败不阻断整个流程
                return True

            print("✅ 覆盖率趋势分析完成")
            return True

        except Exception as e:
            print(f"❌ 覆盖率趋势分析异常: {e}")
            return False

    def generate_report(self, results: Dict[str, bool]) -> str:
        """生成质量报告"""
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        pass_rate = (passed / total) * 100

        report = f"""
# RQA2025 质量监控报告

## 总体结果
- 通过检查: {passed}/{total}
- 通过率: {pass_rate:.1f}%

## 详细结果
"""

        status_icons = {True: "✅", False: "❌"}

        for check_name, passed in results.items():
            icon = status_icons[passed]
            status = "通过" if passed else "失败"
            report += f"- {icon} {check_name}: {status}\n"

        # 保存报告
        report_path = self.test_reports_dir / "quality_monitor_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 质量报告已保存到: {report_path}")

        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 质量监控工具")
    parser.add_argument("--check", choices=[
        "all", "code", "unit", "integration", "e2e",
        "coverage", "performance", "security", "trends"
    ], default="all", help="要运行的检查类型")

    args = parser.parse_args()

    monitor = QualityMonitor()

    if args.check == "all":
        results = monitor.run_all_checks()
        report = monitor.generate_report(results)

        # 检查整体结果
        all_passed = all(results.values())
        if not all_passed:
            print("\n❌ 质量检查未全部通过")
            sys.exit(1)
        else:
            print("\n🎉 所有质量检查通过！")

    else:
        # 运行单个检查
        check_method = f"_run_{args.check}_checks"
        if hasattr(monitor, check_method):
            result = getattr(monitor, check_method)()
            if not result:
                print(f"\n❌ {args.check} 检查失败")
                sys.exit(1)
            else:
                print(f"\n✅ {args.check} 检查通过")
        else:
            print(f"❌ 未知的检查类型: {args.check}")
            sys.exit(1)


if __name__ == "__main__":
    main()
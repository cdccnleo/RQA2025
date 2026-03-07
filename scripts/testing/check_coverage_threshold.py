#!/usr/bin/env python3
"""
RQA2025 覆盖率阈值检查工具
用于CI/CD流水线中的覆盖率门禁检查
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


class CoverageThresholdChecker:
    """覆盖率阈值检查器"""

    def __init__(self, min_coverage: float = 75.0):
        self.min_coverage = min_coverage
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports" / "testing"

        # 各层覆盖率阈值
        self.module_thresholds = {
            "infrastructure": 80.0,
            "data": 80.0,
            "features": 80.0,
            "ensemble": 80.0,
            "trading": 80.0,
            "backtest": 80.0
        }

    def find_latest_coverage_results(self) -> Optional[Path]:
        """查找最新的覆盖率结果文件"""
        if not self.reports_dir.exists():
            return None

        json_files = list(self.reports_dir.glob("coverage_results_*.json"))
        if not json_files:
            return None

        # 按文件名排序，取最新的
        json_files.sort(reverse=True)
        return json_files[0]

    def load_coverage_results(self, file_path: Path) -> Dict:
        """加载覆盖率结果"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 无法加载覆盖率结果文件: {e}")
            return {}

    def extract_coverage_from_stdout(self, stdout: str) -> float:
        """从测试输出中提取覆盖率"""
        try:
            lines = stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[3].replace('%', '')
                        return float(coverage_str)
        except:
            pass
        return 0.0

    def check_module_coverage(self, results: Dict) -> Dict[str, Dict]:
        """检查各模块覆盖率"""
        coverage_status = {}

        for module, result in results.items():
            if result.get("success", False):
                coverage = self.extract_coverage_from_stdout(result.get("stdout", ""))
                threshold = self.module_thresholds.get(module, self.min_coverage)
                passed = coverage >= threshold

                coverage_status[module] = {
                    "coverage": coverage,
                    "threshold": threshold,
                    "passed": passed,
                    "status": "✅ 通过" if passed else "❌ 未通过"
                }
            else:
                coverage_status[module] = {
                    "coverage": 0.0,
                    "threshold": self.module_thresholds.get(module, self.min_coverage),
                    "passed": False,
                    "status": "❌ 测试失败"
                }

        return coverage_status

    def calculate_overall_coverage(self, coverage_status: Dict) -> float:
        """计算总体覆盖率"""
        total_coverage = 0.0
        valid_modules = 0

        for module, status in coverage_status.items():
            if status["coverage"] > 0:
                total_coverage += status["coverage"]
                valid_modules += 1

        return total_coverage / valid_modules if valid_modules > 0 else 0.0

    def generate_report(self, coverage_status: Dict, overall_coverage: float) -> str:
        """生成检查报告"""
        report = f"""# RQA2025 覆盖率阈值检查报告

## 📊 检查摘要

**总体覆盖率**: {overall_coverage:.2f}%
**最低要求**: {self.min_coverage}%
**总体状态**: {'✅ 通过' if overall_coverage >= self.min_coverage else '❌ 未通过'}

## 📈 各模块详情

"""

        for module, status in coverage_status.items():
            report += f"""### {module.title()} 层
- **覆盖率**: {status['coverage']:.2f}%
- **阈值**: {status['threshold']:.2f}%
- **状态**: {status['status']}

"""

        report += f"""
## 🎯 检查结果

"""

        if overall_coverage >= self.min_coverage:
            report += "✅ **覆盖率检查通过** - 所有模块都达到了最低覆盖率要求"
        else:
            report += "❌ **覆盖率检查失败** - 部分模块未达到最低覆盖率要求"
            failed_modules = [m for m, s in coverage_status.items() if not s["passed"]]
            if failed_modules:
                report += f"\n\n**需要改进的模块**: {', '.join(failed_modules)}"

        return report

    def run_check(self) -> bool:
        """运行覆盖率检查"""
        print("🔍 开始覆盖率阈值检查...")

        # 查找最新的覆盖率结果
        results_file = self.find_latest_coverage_results()
        if not results_file:
            print("❌ 未找到覆盖率结果文件")
            return False

        print(f"📄 使用覆盖率结果文件: {results_file.name}")

        # 加载结果
        results = self.load_coverage_results(results_file)
        if not results:
            print("❌ 覆盖率结果文件为空或格式错误")
            return False

        # 检查各模块覆盖率
        coverage_status = self.check_module_coverage(results)

        # 计算总体覆盖率
        overall_coverage = self.calculate_overall_coverage(coverage_status)

        # 生成报告
        report = self.generate_report(coverage_status, overall_coverage)

        # 保存报告
        report_file = self.reports_dir / \
            f"coverage_check_report_{Path(results_file.name).stem.split('_', 2)[2]}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # 打印摘要
        print("\n" + "=" * 60)
        print("📊 覆盖率检查摘要")
        print("=" * 60)

        for module, status in coverage_status.items():
            print(f"{module.title():15} {status['coverage']:6.2f}% {status['status']}")

        print(f"\n总体覆盖率: {overall_coverage:.2f}%")
        print(f"最低要求: {self.min_coverage}%")

        if overall_coverage >= self.min_coverage:
            print("🎉 覆盖率检查通过！")
            return True
        else:
            print("❌ 覆盖率检查失败！")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025覆盖率阈值检查工具")
    parser.add_argument("--min-coverage", type=float, default=75.0,
                        help="最低覆盖率要求 (默认: 75.0)")
    parser.add_argument("--output", help="输出报告文件路径")

    args = parser.parse_args()

    checker = CoverageThresholdChecker(min_coverage=args.min_coverage)
    success = checker.run_check()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

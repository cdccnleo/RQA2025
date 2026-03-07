#!/usr/bin/env python3
"""
测试覆盖率监控脚本
监控代码覆盖率并生成报告
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta


class CoverageMonitor:
    """覆盖率监控器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.reports_dir = self.project_root / "reports" / "coverage"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.reports_dir / "coverage_history.json"

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """运行覆盖率分析"""
        print("🔍 开始覆盖率分析...")

        # 运行测试并收集覆盖率
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-report=json:coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "tests/unit/infrastructure/config/",
            "-v",
            "--tb=short"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300
            )

            # 解析覆盖率数据
            coverage_data = self._parse_coverage_data()

            # 生成报告
            report = self._generate_coverage_report(coverage_data, result)

            # 保存历史记录
            self._save_coverage_history(report)

            return report

        except subprocess.TimeoutExpired:
            return {"error": "Coverage analysis timed out"}
        except Exception as e:
            return {"error": f"Coverage analysis failed: {e}"}

    def _parse_coverage_data(self) -> Dict[str, Any]:
        """解析覆盖率数据"""
        coverage_file = self.project_root / "coverage.json"

        if not coverage_file.exists():
            return {"error": "Coverage file not found"}

        try:
            with open(coverage_file, 'r') as f:
                data = json.load(f)

            # 提取关键指标
            totals = data.get("totals", {})

            return {
                "covered_lines": totals.get("covered_lines", 0),
                "num_statements": totals.get("num_statements", 0),
                "percent_covered": totals.get("percent_covered", 0),
                "missing_lines": totals.get("missing_lines", 0),
                "excluded_lines": totals.get("excluded_lines", 0),
                "files": len(data.get("files", {})),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Failed to parse coverage data: {e}"}

    def _generate_coverage_report(self, coverage_data: Dict[str, Any],
                                test_result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """生成覆盖率报告"""

        # 分析测试结果
        test_lines = test_result.stdout.split('\n')
        test_summary = self._parse_test_summary(test_lines)

        report = {
            "timestamp": coverage_data.get("timestamp"),
            "coverage": {
                "percentage": coverage_data.get("percent_covered", 0),
                "covered_lines": coverage_data.get("covered_lines", 0),
                "total_lines": coverage_data.get("num_statements", 0),
                "missing_lines": coverage_data.get("missing_lines", 0),
                "files_analyzed": coverage_data.get("files", 0)
            },
            "tests": test_summary,
            "status": "success" if test_result.returncode == 0 else "failed",
            "target_coverage": 80.0,
            "meets_target": coverage_data.get("percent_covered", 0) >= 80.0
        }

        return report

    def _parse_test_summary(self, test_lines: List[str]) -> Dict[str, Any]:
        """解析测试摘要"""
        summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }

        if not test_lines:
            return summary

        for line in test_lines:
            try:
                if "passed" in line and "failed" in line:
                    # 解析如 "53 passed, 0 failed" 的行
                    parts = line.strip().split(',')
                    for part in parts:
                        part = part.strip()
                        if 'passed' in part:
                            summary["passed"] = int(part.split()[0])
                        elif 'failed' in part:
                            summary["failed"] = int(part.split()[0])
                        elif 'skipped' in part:
                            summary["skipped"] = int(part.split()[0])
                        elif 'errors' in part:
                            summary["errors"] = int(part.split()[0])
            except (ValueError, IndexError):
                continue

        summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"] + summary["errors"]

        return summary

    def _save_coverage_history(self, report: Dict[str, Any]):
        """保存覆盖率历史"""
        try:
            # 读取现有历史
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # 添加新记录
            history.append(report)

            # 保留最近30天的记录
            cutoff_date = datetime.now() - timedelta(days=30)
            history = [
                record for record in history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_date
            ]

            # 保存历史
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save coverage history: {e}")

    def generate_coverage_badge(self, report: Dict[str, Any]):
        """生成覆盖率徽章"""
        try:
            coverage = report["coverage"]["percentage"]
            color = "brightgreen" if coverage >= 80 else "yellow" if coverage >= 60 else "red"

            badge_data = {
                "schemaVersion": 1,
                "label": "coverage",
                "message": ".1f",
                "color": color
            }

            badge_file = self.reports_dir / "coverage-badge.json"
            with open(badge_file, 'w') as f:
                json.dump(badge_data, f, indent=2)

            print(f"✅ Coverage badge generated: {badge_file}")

        except Exception as e:
            print(f"Warning: Failed to generate coverage badge: {e}")

    def get_coverage_trend(self, days: int = 7) -> Dict[str, Any]:
        """获取覆盖率趋势"""
        try:
            if not self.history_file.exists():
                return {"error": "No coverage history available"}

            with open(self.history_file, 'r') as f:
                history = json.load(f)

            # 获取最近N天的记录
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_records = [
                record for record in history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_date
            ]

            if not recent_records:
                return {"error": f"No coverage data for the last {days} days"}

            # 计算趋势
            coverage_values = [r["coverage"]["percentage"] for r in recent_records]
            test_totals = [r["tests"]["total"] for r in recent_records]

            return {
                "period_days": days,
                "data_points": len(recent_records),
                "coverage_trend": {
                    "min": min(coverage_values),
                    "max": max(coverage_values),
                    "avg": sum(coverage_values) / len(coverage_values),
                    "latest": coverage_values[-1] if coverage_values else 0
                },
                "test_trend": {
                    "min": min(test_totals),
                    "max": max(test_totals),
                    "avg": sum(test_totals) / len(test_totals),
                    "latest": test_totals[-1] if test_totals else 0
                }
            }

        except Exception as e:
            return {"error": f"Failed to get coverage trend: {e}"}

    def print_coverage_report(self, report: Dict[str, Any]):
        """打印覆盖率报告"""
        print("\n" + "="*60)
        print("📊 代码覆盖率报告")
        print("="*60)

        coverage = report["coverage"]
        tests = report["tests"]

        print("\n覆盖率统计:")
        print(".1f")
        print(f"  📁 覆盖文件数: {coverage['files_analyzed']}")
        print(f"  ✅ 覆盖行数: {coverage['covered_lines']}")
        print(f"  📝 总行数: {coverage['total_lines']}")
        print(f"  ❌ 未覆盖行数: {coverage['missing_lines']}")

        print("\n测试统计:")
        print(f"  🧪 总测试数: {tests['total']}")
        print(f"  ✅ 通过: {tests['passed']}")
        print(f"  ❌ 失败: {tests['failed']}")
        print(f"  ⏭️  跳过: {tests['skipped']}")
        print(f"  💥 错误: {tests['errors']}")

        # 目标达成状态
        target = report["target_coverage"]
        meets_target = report["meets_target"]
        status = "✅ 达成目标" if meets_target else "❌ 未达成目标"

        print(f"\n🎯 目标达成: {target}% - {status}")

        # 建议
        if meets_target:
            print("\n💡 建议: 覆盖率良好，可以考虑增加更多边界条件测试。")
        else:
            remaining = target - coverage['percentage']
            print(f"\n💡 建议: 需要增加 {remaining:.1f}% 的覆盖率，建议添加更多测试用例。")


def main():
    """主函数"""
    monitor = CoverageMonitor()

    # 运行覆盖率分析
    report = monitor.run_coverage_analysis()

    if "error" in report:
        print(f"❌ 覆盖率分析失败: {report['error']}")
        sys.exit(1)

    # 生成徽章
    monitor.generate_coverage_badge(report)

    # 打印报告
    monitor.print_coverage_report(report)

    # 获取趋势
    trend = monitor.get_coverage_trend(days=7)
    if "error" not in trend:
        print("\n📈 7天覆盖率趋势:")
        print(".1f")
        print(".1f")
        print(f"  📊 数据点数: {trend['data_points']}")

    # 返回适当的退出码
    meets_target = report.get("meets_target", False)
    return 0 if meets_target else 1


if __name__ == "__main__":
    sys.exit(main())

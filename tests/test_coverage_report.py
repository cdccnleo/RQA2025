#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试覆盖率报告生成器
生成详细的测试覆盖率报告和统计信息
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class TestCoverageReporter:
    """测试覆盖率报告器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def run_tests_with_coverage(self, test_path: str = "tests") -> Dict[str, Any]:
        """运行测试并生成覆盖率报告"""
        print("开始运行测试并生成覆盖率报告...")

        # 使用pytest-cov运行测试
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=json:test_reports/coverage.json",
            "--cov-report=html:test_reports/htmlcov",
            "--cov-report=xml:test_reports/coverage.xml",
            "-v",
            "--tb=short"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 解析结果
            coverage_data = self._parse_coverage_result(result)
            test_results = self._parse_test_result(result)

            return {
                "success": result.returncode == 0,
                "coverage": coverage_data,
                "tests": test_results,
                "command_output": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            }

        except subprocess.TimeoutExpired:
            print("测试运行超时")
            return {
                "success": False,
                "error": "Test execution timed out",
                "coverage": {},
                "tests": {}
            }
        except Exception as e:
            print(f"测试运行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "coverage": {},
                "tests": {}
            }

    def _parse_coverage_result(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """解析覆盖率结果"""
        coverage_data = {
            "overall_coverage": 0.0,
            "files_covered": 0,
            "total_files": 0,
            "lines_covered": 0,
            "total_lines": 0,
            "file_coverage": {}
        }

        # 尝试读取coverage.json文件
        coverage_json = self.reports_dir / "coverage.json"
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r') as f:
                    cov_data = json.load(f)

                coverage_data["overall_coverage"] = cov_data.get("totals", {}).get("percent_covered", 0.0)
                coverage_data["files_covered"] = cov_data.get("totals", {}).get("num_statements", 0)
                coverage_data["total_files"] = len(cov_data.get("files", {}))

                # 计算总行数
                total_lines = 0
                covered_lines = 0
                for file_path, file_data in cov_data.get("files", {}).items():
                    file_lines = file_data.get("num_statements", 0)
                    file_covered = file_data.get("summary", {}).get("num_statements", 0)
                    total_lines += file_lines
                    covered_lines += file_covered

                coverage_data["total_lines"] = total_lines
                coverage_data["lines_covered"] = covered_lines

            except Exception as e:
                print(f"解析覆盖率数据失败: {e}")

        return coverage_data

    def _parse_test_result(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """解析测试结果"""
        test_data = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0
        }

        stdout = result.stdout

        # 解析pytest输出
        lines = stdout.split('\n')

        for line in lines:
            if "passed" in line and "failed" in line:
                # 解析总结行，如 "5 passed, 2 failed, 1 skipped in 12.34s"
                import re
                match = re.search(r'(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+skipped.*?(\d+\.\d+)s', line)
                if match:
                    test_data["passed"] = int(match.group(1))
                    test_data["failed"] = int(match.group(2))
                    test_data["skipped"] = int(match.group(3))
                    test_data["duration"] = float(match.group(4))
                    test_data["total_tests"] = test_data["passed"] + test_data["failed"] + test_data["skipped"]

        return test_data

    def analyze_test_gaps(self) -> Dict[str, Any]:
        """分析测试覆盖缺口"""
        gaps = {
            "uncovered_modules": [],
            "low_coverage_files": [],
            "missing_test_types": [],
            "recommendations": []
        }

        # 检查源码文件是否有对应的测试文件
        src_dir = self.project_root / "src"
        test_dir = self.project_root / "tests"

        if src_dir.exists():
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        src_path = Path(root) / file
                        relative_path = src_path.relative_to(src_dir)

                        # 查找对应的测试文件
                        test_file_patterns = [
                            f"test_{file}",
                            f"{file.replace('.py', '_test.py')}",
                            f"test_{file.replace('.py', '')}.py"
                        ]

                        found_test = False
                        for pattern in test_file_patterns:
                            test_path = test_dir / relative_path.parent / pattern
                            if test_path.exists():
                                found_test = True
                                break

                        if not found_test:
                            gaps["uncovered_modules"].append(str(relative_path))

        # 检查低覆盖率文件
        coverage_json = self.reports_dir / "coverage.json"
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r') as f:
                    cov_data = json.load(f)

                for file_path, file_data in cov_data.get("files", {}).items():
                    coverage_pct = file_data.get("summary", {}).get("percent_covered", 100.0)
                    if coverage_pct < 80.0:
                        gaps["low_coverage_files"].append({
                            "file": file_path,
                            "coverage": coverage_pct
                        })
            except Exception as e:
                print(f"分析覆盖率数据失败: {e}")

        # 生成建议
        if gaps["uncovered_modules"]:
            gaps["recommendations"].append(
                f"为以下 {len(gaps['uncovered_modules'])} 个模块创建测试文件："
            )
            for module in gaps["uncovered_modules"][:5]:  # 只显示前5个
                gaps["recommendations"].append(f"  - {module}")

        if gaps["low_coverage_files"]:
            gaps["recommendations"].append(
                f"提高以下 {len(gaps['low_coverage_files'])} 个文件的测试覆盖率："
            )
            for file_info in gaps["low_coverage_files"][:5]:  # 只显示前5个
                gaps["recommendations"].append(
                    f"  - {file_info['file']}: {file_info['coverage']:.1f}%"
                )

        return gaps

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        print("生成综合测试报告...")

        # 运行测试并获取覆盖率
        test_result = self.run_tests_with_coverage()

        # 分析测试缺口
        gaps = self.analyze_test_gaps()

        # 构建综合报告
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "test_execution": test_result,
            "coverage_analysis": {
                "overall_coverage": test_result.get("coverage", {}).get("overall_coverage", 0.0),
                "target_coverage": 80.0,
                "coverage_achieved": test_result.get("coverage", {}).get("overall_coverage", 0.0) >= 80.0,
                "files_covered": test_result.get("coverage", {}).get("files_covered", 0),
                "total_files": test_result.get("coverage", {}).get("total_files", 0),
                "lines_covered": test_result.get("coverage", {}).get("lines_covered", 0),
                "total_lines": test_result.get("coverage", {}).get("total_lines", 0)
            },
            "test_statistics": test_result.get("tests", {}),
            "gaps_analysis": gaps,
            "acceptance_criteria": {
                "overall_coverage_target": 80.0,
                "test_success_rate_target": 95.0,
                "critical_modules_covered": True,
                "api_endpoints_tested": True
            },
            "acceptance_status": self._evaluate_acceptance_criteria(test_result, gaps)
        }

        return report

    def _evaluate_acceptance_criteria(self, test_result: Dict[str, Any],
                                    gaps: Dict[str, Any]) -> Dict[str, Any]:
        """评估验收标准"""
        coverage = test_result.get("coverage", {}).get("overall_coverage", 0.0)
        tests = test_result.get("tests", {})

        acceptance = {
            "overall_status": "PENDING",
            "criteria_status": {},
            "recommendations": []
        }

        # 覆盖率标准
        coverage_passed = coverage >= 80.0
        acceptance["criteria_status"]["coverage_target"] = {
            "required": 80.0,
            "actual": coverage,
            "passed": coverage_passed
        }

        # 测试成功率标准
        total_tests = tests.get("total_tests", 0)
        passed_tests = tests.get("passed", 0)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0

        success_rate_passed = success_rate >= 95.0
        acceptance["criteria_status"]["test_success_rate"] = {
            "required": 95.0,
            "actual": success_rate,
            "passed": success_rate_passed
        }

        # 关键模块覆盖
        critical_modules_covered = len(gaps.get("uncovered_modules", [])) == 0
        acceptance["criteria_status"]["critical_modules_covered"] = {
            "required": True,
            "actual": critical_modules_covered,
            "passed": critical_modules_covered
        }

        # 整体状态评估
        all_criteria_passed = (
            coverage_passed and
            success_rate_passed and
            critical_modules_covered
        )

        acceptance["overall_status"] = "PASSED" if all_criteria_passed else "FAILED"

        # 生成建议
        if not coverage_passed:
            acceptance["recommendations"].append(
                f"覆盖率未达标：当前 {coverage:.1f}%，需要达到80.0%"
            )

        if not success_rate_passed:
            acceptance["recommendations"].append(
                f"测试成功率未达标：当前 {success_rate:.1f}%，需要达到95.0%"
            )

        if not critical_modules_covered:
            uncovered_count = len(gaps.get("uncovered_modules", []))
            acceptance["recommendations"].append(
                f"发现 {uncovered_count} 个未测试的关键模块"
            )

        return acceptance

    def save_report(self, report: Dict[str, Any], format_type: str = "json"):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == "json":
            filename = f"comprehensive_test_report_{timestamp}.json"
            filepath = self.reports_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        elif format_type == "markdown":
            filename = f"comprehensive_test_report_{timestamp}.md"
            filepath = self.reports_dir / filename

            markdown_content = self._generate_markdown_report(report)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

        print(f"测试报告已保存到: {filepath}")
        return str(filepath)

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        lines = []

        lines.append("# 综合测试报告")
        lines.append("")
        lines.append(f"**生成时间**: {report['report_generated_at']}")
        lines.append("")

        # 总体状态
        lines.append("## 📊 总体状态")
        lines.append("")
        acceptance = report["acceptance_status"]
        status_emoji = "✅" if acceptance["overall_status"] == "PASSED" else "❌"
        lines.append(f"**验收状态**: {status_emoji} {acceptance['overall_status']}")
        lines.append("")

        # 测试执行结果
        lines.append("## 🧪 测试执行结果")
        lines.append("")
        test_exec = report["test_execution"]
        if test_exec["success"]:
            lines.append("✅ 测试执行成功")
        else:
            lines.append("❌ 测试执行失败")
            if "error" in test_exec:
                lines.append(f"**错误信息**: {test_exec['error']}")

        tests = report.get("test_statistics", {})
        lines.append(f"- **总测试数**: {tests.get('total_tests', 0)}")
        lines.append(f"- **通过**: {tests.get('passed', 0)}")
        lines.append(f"- **失败**: {tests.get('failed', 0)}")
        lines.append(f"- **跳过**: {tests.get('skipped', 0)}")
        lines.append(f"- **执行时间**: {tests.get('duration', 0.0):.2f}秒")
        lines.append("")

        # 覆盖率分析
        lines.append("## 📈 覆盖率分析")
        lines.append("")
        coverage = report["coverage_analysis"]
        coverage_pct = coverage["overall_coverage"]

        if coverage_pct >= 80.0:
            lines.append(f"✅ **总体覆盖率**: {coverage_pct:.1f}% (达标)")
        else:
            lines.append(f"❌ **总体覆盖率**: {coverage_pct:.1f}% (未达标，需要≥80%)")

        lines.append(f"- **覆盖文件数**: {coverage['files_covered']}")
        lines.append(f"- **总文件数**: {coverage['total_files']}")
        lines.append(f"- **覆盖行数**: {coverage['lines_covered']}")
        lines.append(f"- **总行数**: {coverage['total_lines']}")
        lines.append("")

        # 验收标准评估
        lines.append("## 🎯 验收标准评估")
        lines.append("")
        criteria = acceptance["criteria_status"]

        for criterion_name, criterion_data in criteria.items():
            required = criterion_data["required"]
            actual = criterion_data["actual"]
            passed = criterion_data["passed"]

            status_emoji = "✅" if passed else "❌"
            lines.append(f"{status_emoji} **{criterion_name}**: {actual} (要求: {required})")

        lines.append("")

        # 建议
        if acceptance.get("recommendations"):
            lines.append("## 💡 改进建议")
            lines.append("")
            for recommendation in acceptance["recommendations"]:
                lines.append(f"- {recommendation}")
            lines.append("")

        # 测试缺口分析
        gaps = report.get("gaps_analysis", {})
        if gaps.get("uncovered_modules") or gaps.get("low_coverage_files"):
            lines.append("## 🔍 测试缺口分析")
            lines.append("")

            if gaps.get("uncovered_modules"):
                lines.append(f"### 未测试模块 ({len(gaps['uncovered_modules'])} 个)")
                lines.append("")
                for module in gaps["uncovered_modules"][:10]:  # 只显示前10个
                    lines.append(f"- {module}")
                if len(gaps["uncovered_modules"]) > 10:
                    lines.append(f"- ... 还有 {len(gaps['uncovered_modules']) - 10} 个")
                lines.append("")

            if gaps.get("low_coverage_files"):
                lines.append(f"### 低覆盖率文件 ({len(gaps['low_coverage_files'])} 个)")
                lines.append("")
                for file_info in gaps["low_coverage_files"][:10]:  # 只显示前10个
                    lines.append(f"- {file_info['file']}: {file_info['coverage']:.1f}%")
                lines.append("")

        return "\n".join(lines)


def main():
    """主函数"""
    print("=== RQA2025 测试覆盖率报告生成器 ===\\n")

    # 创建报告器
    reporter = TestCoverageReporter()

    # 生成综合报告
    print("正在生成综合测试报告...")
    report = reporter.generate_comprehensive_report()

    # 保存报告
    json_file = reporter.save_report(report, "json")
    markdown_file = reporter.save_report(report, "markdown")

    # 输出关键指标
    coverage = report["coverage_analysis"]
    acceptance = report["acceptance_status"]
    tests = report.get("test_statistics", {})

    print("\\n" + "="*50)
    print("📊 测试执行结果:")
    print(f"   总测试数: {tests.get('total_tests', 0)}")
    print(f"   通过: {tests.get('passed', 0)}")
    print(f"   失败: {tests.get('failed', 0)}")
    print(f"   跳过: {tests.get('skipped', 0)}")

    print("\\n📈 覆盖率分析:")
    print(f"   总体覆盖率: {coverage['overall_coverage']:.1f}%")
    print(f"   目标覆盖率: {coverage['target_coverage']:.1f}%")
    print(f"   是否达标: {'✅ 是' if coverage['coverage_achieved'] else '❌ 否'}")

    print("\\n🎯 验收标准评估:")
    print(f"   整体状态: {acceptance['overall_status']}")
    for criterion_name, criterion_data in acceptance["criteria_status"].items():
        status = "✅" if criterion_data["passed"] else "❌"
        print(f"   {status} {criterion_name}: {criterion_data['actual']} (要求: {criterion_data['required']})")

    print("\\n📄 生成的报告文件:")
    print(f"   JSON: {json_file}")
    print(f"   Markdown: {markdown_file}")

    if acceptance["overall_status"] == "PASSED":
        print("\\n🎉 恭喜！所有验收标准都已通过！")
    else:
        print("\\n⚠️  还有一些验收标准需要改进，请查看详细报告。")

    return report


if __name__ == "__main__":
    main()

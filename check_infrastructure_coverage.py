#!/usr/bin/env python3
"""
基础设施层子模块测试覆盖率检查工具
检查16个主要子模块的测试覆盖率是否达到投产要求（80%）
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import re

class InfrastructureCoverageChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_path = self.project_root / "src" / "infrastructure"
        self.tests_path = self.project_root / "tests" / "infrastructure"

        # 16个主要子模块（按照测试覆盖率改进计划）
        self.modules = [
            "config", "distributed", "versioning", "resource", "logging", "ops",  # 6个缺失测试
            "monitoring", "health", "security",  # 3个覆盖不足
            "constants", "interfaces", "optimization", "core", "utils",  # 5个已有测试
            "cache", "error"  # 2个补充的重要模块
        ]

        # 投产要求：80%覆盖率
        self.target_coverage = 80.0

    def get_module_files(self, module_name: str) -> List[str]:
        """获取模块的所有Python文件"""
        module_path = self.infrastructure_path / module_name
        if not module_path.exists():
            return []

        files = []
        if module_path.is_file():
            files.append(str(module_path))
        else:
            for py_file in module_path.rglob("*.py"):
                if not py_file.name.startswith("__"):
                    files.append(str(py_file))

        return files

    def get_module_tests(self, module_name: str) -> List[str]:
        """获取模块的测试文件"""
        test_files = []

        # 检查tests/infrastructure目录
        test_dir = self.tests_path
        if test_dir.exists():
            # 查找相关测试文件
            patterns = [
                f"test_{module_name}*.py",
                f"test_*{module_name}*.py",
                f"*{module_name}*test*.py"
            ]

            for pattern in patterns:
                for test_file in test_dir.rglob(pattern):
                    test_files.append(str(test_file))

        return list(set(test_files))  # 去重

    def run_module_coverage(self, module_name: str) -> Dict:
        """运行单个模块的覆盖率测试"""
        print(f"\n🔍 检查模块: {module_name}")

        source_files = self.get_module_files(module_name)
        test_files = self.get_module_tests(module_name)

        if not source_files:
            return {
                "module": module_name,
                "status": "no_source",
                "coverage": 0.0,
                "source_files": 0,
                "test_files": len(test_files),
                "message": f"模块 {module_name} 不存在或没有Python文件"
            }

        if not test_files:
            return {
                "module": module_name,
                "status": "no_tests",
                "coverage": 0.0,
                "source_files": len(source_files),
                "test_files": 0,
                "message": f"模块 {module_name} 缺少测试文件"
            }

        # 构造覆盖率测试命令
        cmd = [
            "python", "-m", "pytest",
            "--cov=" + ",".join(source_files),
            "--cov-report=json",
            "--cov-fail-under=0",  # 不因覆盖率不足而失败
            "-q"
        ] + test_files

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 解析覆盖率报告
            coverage_file = self.project_root / ".coverage"
            if coverage_file.exists():
                # 使用coverage工具获取详细报告
                cov_cmd = ["python", "-m", "coverage", "json", "-o", "coverage.json"]
                subprocess.run(cov_cmd, cwd=self.project_root, capture_output=True)

                cov_json = self.project_root / "coverage.json"
                if cov_json.exists():
                    with open(cov_json, 'r', encoding='utf-8') as f:
                        cov_data = json.load(f)

                    # 计算该模块的总覆盖率
                    total_lines = 0
                    covered_lines = 0

                    for file_path in source_files:
                        file_key = file_path.replace(str(self.project_root), "").lstrip("/")
                        if file_key in cov_data.get("files", {}):
                            file_cov = cov_data["files"][file_key]
                            total_lines += file_cov.get("summary", {}).get("num_statements", 0)
                            covered_lines += file_cov.get("summary", {}).get("covered_lines", 0)

                    coverage_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

                    # 清理临时文件
                    cov_json.unlink(missing_ok=True)

                    status = "达标" if coverage_percent >= self.target_coverage else "未达标"

                    return {
                        "module": module_name,
                        "status": status,
                        "coverage": round(coverage_percent, 2),
                        "source_files": len(source_files),
                        "test_files": len(test_files),
                        "total_lines": total_lines,
                        "covered_lines": covered_lines,
                        "message": f"覆盖率 {coverage_percent:.2f}%, {'达标' if coverage_percent >= self.target_coverage else '未达标'}"
                    }

            # 如果没有覆盖率数据，返回测试结果
            if result.returncode == 0:
                return {
                    "module": module_name,
                    "status": "tested",
                    "coverage": 0.0,  # 无法获取覆盖率
                    "source_files": len(source_files),
                    "test_files": len(test_files),
                    "message": "测试通过，但无法获取覆盖率数据"
                }
            else:
                return {
                    "module": module_name,
                    "status": "test_failed",
                    "coverage": 0.0,
                    "source_files": len(source_files),
                    "test_files": len(test_files),
                    "message": f"测试失败: {result.stderr[:200]}..."
                }

        except subprocess.TimeoutExpired:
            return {
                "module": module_name,
                "status": "timeout",
                "coverage": 0.0,
                "source_files": len(source_files),
                "test_files": len(test_files),
                "message": "测试执行超时"
            }
        except Exception as e:
            return {
                "module": module_name,
                "status": "error",
                "coverage": 0.0,
                "source_files": len(source_files),
                "test_files": len(test_files),
                "message": f"执行错误: {str(e)}"
            }

    def check_all_modules(self) -> Dict:
        """检查所有模块的覆盖率"""
        results = {}
        summary = {
            "total_modules": len(self.modules),
            "达标模块": 0,
            "未达标模块": 0,
            "无测试模块": 0,
            "测试失败模块": 0,
            "平均覆盖率": 0.0,
            "总源码文件": 0,
            "总测试文件": 0
        }

        total_coverage = 0.0
        valid_modules = 0

        for module in self.modules:
            result = self.run_module_coverage(module)
            results[module] = result

            summary["总源码文件"] += result["source_files"]
            summary["总测试文件"] += result["test_files"]

            if result["status"] in ["达标", "tested"]:
                if result["coverage"] >= self.target_coverage:
                    summary["达标模块"] += 1
                    total_coverage += result["coverage"]
                    valid_modules += 1
                elif result["coverage"] > 0:
                    summary["未达标模块"] += 1
                    total_coverage += result["coverage"]
                    valid_modules += 1
            elif result["status"] == "no_tests":
                summary["无测试模块"] += 1
            else:
                summary["测试失败模块"] += 1

        if valid_modules > 0:
            summary["平均覆盖率"] = round(total_coverage / valid_modules, 2)

        return {
            "summary": summary,
            "results": results
        }

    def generate_report(self, results: Dict) -> str:
        """生成详细报告"""
        summary = results["summary"]
        module_results = results["results"]

        report = []
        report.append("# 🚀 基础设施层子模块测试覆盖率达标报告")
        report.append("")

        # 总体概况
        report.append("## 📊 总体概况")
        report.append("")
        report.append("| 指标 | 数值 | 状态 |")
        report.append("|------|------|------|")
        report.append(f"| 检查模块数 | {summary['total_modules']}个 | ✅ |")
        report.append(f"| 达标模块数 | {summary['达标模块']}个 | {'✅' if summary['达标模块'] >= 12 else '⚠️'} |")
        report.append(f"| 未达标模块数 | {summary['未达标模块']}个 | {'❌' if summary['未达标模块'] > 4 else '⚠️'} |")
        report.append(f"| 无测试模块数 | {summary['无测试模块']}个 | {'❌' if summary['无测试模块'] > 0 else '✅'} |")
        report.append(f"| 测试失败模块数 | {summary['测试失败模块']}个 | {'❌' if summary['测试失败模块'] > 2 else '⚠️'} |")
        report.append(f"| 平均覆盖率 | {summary['平均覆盖率']}% | {'✅' if summary['平均覆盖率'] >= 70 else '❌'} |")
        report.append(f"| 总源码文件数 | {summary['总源码文件']}个 | ✅ |")
        report.append(f"| 总测试文件数 | {summary['总测试文件']}个 | ✅ |")
        report.append("")

        # 投产要求评估
        production_ready = (
            summary['达标模块'] >= 12 and
            summary['无测试模块'] == 0 and
            summary['测试失败模块'] <= 2 and
            summary['平均覆盖率'] >= 70
        )

        report.append("## 🎯 投产达标评估")
        report.append("")
        if production_ready:
            report.append("✅ **基础设施层测试覆盖率已达标投产要求**")
            report.append("")
            report.append("- 80%目标覆盖率模块数量充足")
            report.append("- 无缺失测试的模块")
            report.append("- 测试执行稳定")
            report.append("- 平均覆盖率达到要求")
        else:
            report.append("❌ **基础设施层测试覆盖率未达标投产要求**")
            report.append("")
            if summary['无测试模块'] > 0:
                report.append(f"- ⚠️ {summary['无测试模块']}个模块缺少测试文件")
            if summary['测试失败模块'] > 2:
                report.append(f"- ⚠️ {summary['测试失败模块']}个模块测试执行失败")
            if summary['平均覆盖率'] < 70:
                report.append(f"- ⚠️ 平均覆盖率仅 {summary['平均覆盖率']}%，低于70%要求")
            if summary['达标模块'] < 12:
                report.append(f"- ⚠️ 仅 {summary['达标模块']}个模块达到80%覆盖率")
        report.append("")

        # 详细模块报告
        report.append("## 📋 详细模块报告")
        report.append("")

        # 按状态分组显示
        status_groups = {
            "达标": [],
            "未达标": [],
            "无测试": [],
            "测试失败": [],
            "其他": []
        }

        for module, result in module_results.items():
            if result["status"] == "达标":
                status_groups["达标"].append((module, result))
            elif result["status"] == "tested" and result["coverage"] > 0:
                status_groups["未达标"].append((module, result))
            elif result["status"] == "no_tests":
                status_groups["无测试"].append((module, result))
            elif result["status"] in ["test_failed", "timeout", "error"]:
                status_groups["测试失败"].append((module, result))
            else:
                status_groups["其他"].append((module, result))

        for status, modules in status_groups.items():
            if modules:
                icon = {
                    "达标": "✅",
                    "未达标": "⚠️",
                    "无测试": "❌",
                    "测试失败": "🔥",
                    "其他": "❓"
                }.get(status, "❓")

                report.append(f"### {icon} {status}模块 ({len(modules)}个)")
                report.append("")
                report.append("| 模块名 | 覆盖率 | 源码文件 | 测试文件 | 状态说明 |")
                report.append("|--------|--------|----------|----------|----------|")

                for module, result in modules:
                    report.append(f"| {module} | {result['coverage']}% | {result['source_files']}个 | {result['test_files']}个 | {result['message']} |")

                report.append("")

        # 改进建议
        report.append("## 💡 改进建议")
        report.append("")

        if summary['无测试模块'] > 0:
            report.append("### 🚨 紧急改进项")
            report.append("以下模块缺少测试文件，需要立即创建：")
            for module, result in module_results.items():
                if result["status"] == "no_tests":
                    report.append(f"- **{module}**: {result['source_files']}个源码文件待测试覆盖")
            report.append("")

        if summary['未达标模块'] > 0:
            report.append("### ⚠️ 重点改进项")
            report.append("以下模块覆盖率不足，需要补充测试：")
            for module, result in module_results.items():
                if result["status"] == "tested" and 0 < result["coverage"] < self.target_coverage:
                    gap = self.target_coverage - result["coverage"]
                    report.append(f"- **{module}**: 当前 {result['coverage']}%, 还需提升 {gap:.1f}%")
            report.append("")

        if summary['测试失败模块'] > 0:
            report.append("### 🔧 质量改进项")
            report.append("以下模块测试执行失败，需要修复代码问题：")
            for module, result in module_results.items():
                if result["status"] in ["test_failed", "timeout", "error"]:
                    report.append(f"- **{module}**: {result['message']}")
            report.append("")

        report.append("### 📈 优化建议")
        report.append("- 为新模块建立测试模板和标准")
        report.append("- 实施自动化测试覆盖率监控")
        report.append("- 建立代码审查中测试覆盖率的检查机制")
        report.append("- 定期review和更新测试用例")

        return "\n".join(report)

def main():
    """主函数"""
    project_root = os.getcwd()

    checker = InfrastructureCoverageChecker(project_root)
    print("🔍 开始检查基础设施层16个子模块的测试覆盖率...")
    print("📊 目标覆盖率：80%")
    print("⏱️ 这可能需要几分钟时间...")

    results = checker.check_all_modules()

    # 生成报告
    report = checker.generate_report(results)

    # 保存报告
    report_file = Path(project_root) / "infrastructure_coverage_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 详细报告已保存至: {report_file}")

    # 打印简要摘要
    summary = results["summary"]
    print("\n📊 覆盖率检查结果摘要:")
    print(f"✅ 达标模块: {summary['达标模块']}/{summary['total_modules']}")
    print(f"⚠️ 未达标模块: {summary['未达标模块']}/{summary['total_modules']}")
    print(f"❌ 无测试模块: {summary['无测试模块']}/{summary['total_modules']}")
    print(f"🔥 测试失败模块: {summary['测试失败模块']}/{summary['total_modules']}")
    print(f"📈 平均覆盖率: {summary['平均覆盖率']:.2f}%")

    production_ready = (
        summary['达标模块'] >= 12 and
        summary['无测试模块'] == 0 and
        summary['测试失败模块'] <= 2 and
        summary['平均覆盖率'] >= 70
    )

    if production_ready:
        print("🎉 基础设施层测试覆盖率已达标投产要求！")
    else:
        print("⚠️ 基础设施层测试覆盖率未达标，需要继续改进")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层覆盖率分析脚本
分析当前覆盖率状况，识别提升机会
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class CoverageAnalysis:
    """覆盖率分析结果"""
    module_name: str
    total_lines: int
    covered_lines: int
    coverage_percent: float
    missing_lines: List[int]
    uncovered_functions: List[str]

class CoverageAnalyzer:
    """覆盖率分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.coverage_data = {}

    def run_coverage_analysis(self) -> Dict[str, CoverageAnalysis]:
        """运行覆盖率分析"""
        print("🔍 正在运行覆盖率分析...")

        # 运行pytest with coverage
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/infrastructure/",
            "--ignore=tests/unit/infrastructure/security/test_data_security.py",
            "--cov=src/infrastructure",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "-q"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode not in [0, 1]:  # Allow test failures
                print(f"❌ 覆盖率分析失败: {result.stderr}")
                return {}

            # 解析coverage.json
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)

                return self._parse_coverage_data(coverage_data)
            else:
                print("❌ coverage.json文件未找到")
                return {}

        except subprocess.TimeoutExpired:
            print("⏰ 覆盖率分析超时")
            return {}
        except Exception as e:
            print(f"❌ 覆盖率分析异常: {e}")
            return {}

    def _parse_coverage_data(self, coverage_data: Dict[str, Any]) -> Dict[str, CoverageAnalysis]:
        """解析覆盖率数据"""
        analyses = {}

        files = coverage_data.get('files', {})

        # 按模块分组
        module_files = {}
        for file_path in files.keys():
            if 'src/infrastructure/' in file_path:
                # 提取模块名
                parts = file_path.replace('src/infrastructure/', '').split('/')
                module = parts[0] if parts[0] else 'core'

                if module not in module_files:
                    module_files[module] = []
                module_files[module].append(file_path)

        # 分析每个模块
        for module, file_list in module_files.items():
            total_lines = 0
            covered_lines = 0
            missing_lines = []

            for file_path in file_list:
                file_data = files[file_path]
                summary = file_data.get('summary', {})

                total_lines += summary.get('num_statements', 0)
                covered_lines += summary.get('covered_lines', 0)

                # 收集缺失行
                missing = file_data.get('missing_lines', [])
                if isinstance(missing, list):
                    missing_lines.extend(missing)

            if total_lines > 0:
                coverage_percent = (covered_lines / total_lines) * 100
                analyses[module] = CoverageAnalysis(
                    module_name=module,
                    total_lines=total_lines,
                    covered_lines=covered_lines,
                    coverage_percent=coverage_percent,
                    missing_lines=missing_lines[:100],  # 只保留前100个
                    uncovered_functions=[]
                )

        return analyses

    def identify_improvement_opportunities(self, analyses: Dict[str, CoverageAnalysis]) -> Dict[str, List[str]]:
        """识别提升机会"""
        opportunities = {}

        for module, analysis in analyses.items():
            if analysis.coverage_percent < 70:  # 覆盖率低于70%的模块
                opportunities[module] = []

                if analysis.coverage_percent < 30:
                    opportunities[module].append("🚨 覆盖率严重不足，需要全面测试补充")
                elif analysis.coverage_percent < 50:
                    opportunities[module].append("⚠️ 覆盖率偏低，需要重点补充核心功能测试")
                else:
                    opportunities[module].append("📈 覆盖率中等，需要补充边界条件和异常处理测试")

                # 基于缺失行数给出建议
                if len(analysis.missing_lines) > 100:
                    opportunities[module].append(f"🔍 发现{len(analysis.missing_lines)}+行未覆盖代码")
                elif len(analysis.missing_lines) > 50:
                    opportunities[module].append(f"🔍 发现{len(analysis.missing_lines)}行未覆盖代码")

        return opportunities

    def generate_report(self, analyses: Dict[str, CoverageAnalysis]) -> str:
        """生成分析报告"""
        opportunities = self.identify_improvement_opportunities(analyses)

        report = []
        report.append("# 基础设施层覆盖率分析报告")
        report.append("")
        report.append("## 📊 覆盖率概览")
        report.append("")
        report.append("| 模块 | 总行数 | 覆盖行数 | 覆盖率 | 状态 |")
        report.append("|------|--------|----------|--------|------|")

        total_lines = 0
        total_covered = 0

        for module, analysis in sorted(analyses.items()):
            status = "✅" if analysis.coverage_percent >= 70 else "⚠️" if analysis.coverage_percent >= 50 else "❌"
            report.append(f"| {module} | {analysis.total_lines} | {analysis.covered_lines} | {analysis.coverage_percent:.1f}% | {status} |")

            total_lines += analysis.total_lines
            total_covered += analysis.covered_lines

        overall_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0
        report.append("")
        report.append(f"**总体覆盖率**: {overall_coverage:.1f}% ({total_covered}/{total_lines})")
        report.append("")

        # 提升机会
        if opportunities:
            report.append("## 🎯 提升机会")
            report.append("")

            for module, suggestions in opportunities.items():
                analysis = analyses[module]
                report.append(f"### {module}模块 ({analysis.coverage_percent:.1f}%)")
                for suggestion in suggestions:
                    report.append(f"- {suggestion}")
                report.append("")

        # 行动计划
        report.append("## 📋 行动计划")
        report.append("")
        report.append("### Phase 1: 紧急提升 (1-2天)")
        urgent_modules = [m for m, a in analyses.items() if a.coverage_percent < 50]
        if urgent_modules:
            report.append(f"- 重点提升模块: {', '.join(urgent_modules)}")
        report.append("- 补充核心业务逻辑测试")
        report.append("- 覆盖主要错误处理路径")
        report.append("")

        report.append("### Phase 2: 深度覆盖 (3-5天)")
        medium_modules = [m for m, a in analyses.items() if 50 <= a.coverage_percent < 70]
        if medium_modules:
            report.append(f"- 完善模块: {', '.join(medium_modules)}")
        report.append("- 补充边界条件测试")
        report.append("- 完善异常处理覆盖")
        report.append("")

        report.append("### Phase 3: 质量优化 (2-3天)")
        report.append("- 清理重复测试用例")
        report.append("- 优化测试执行性能")
        report.append("- 建立覆盖率监控机制")

        return "\n".join(report)

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    analyzer = CoverageAnalyzer(project_root)
    analyses = analyzer.run_coverage_analysis()

    if analyses:
        report = analyzer.generate_report(analyses)
        print(report)

        # 保存报告
        report_file = project_root / "reports" / "coverage_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 报告已保存到: {report_file}")
    else:
        print("❌ 无法生成覆盖率分析报告")

if __name__ == "__main__":
    main()
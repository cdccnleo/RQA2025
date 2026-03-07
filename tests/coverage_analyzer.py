#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
覆盖率分析器

分析测试覆盖率，生成详细报告，识别覆盖不足的区域。
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """覆盖率报告"""
    total_lines: int = 0
    covered_lines: int = 0
    coverage_percentage: float = 0.0
    files_analyzed: int = 0
    missing_lines: Dict[str, List[int]] = None

    def __post_init__(self):
        if self.missing_lines is None:
            self.missing_lines = {}


class CoverageAnalyzer:
    """覆盖率分析器"""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_dir = Path("test_logs/coverage_reports")

    def run_coverage_analysis(self) -> CoverageReport:
        """运行覆盖率分析"""
        logger.info("开始覆盖率分析...")

        # 确保覆盖率目录存在
        self.coverage_dir.mkdir(parents=True, exist_ok=True)

        # 运行覆盖率测试
        self._run_coverage_tests()

        # 分析覆盖率数据
        report = self._analyze_coverage_data()

        # 生成详细报告
        self._generate_detailed_report(report)

        return report

    def _run_coverage_tests(self):
        """运行覆盖率测试"""
        logger.info("运行覆盖率测试...")

        # 运行主要层级的测试
        layers = ["core", "data", "features", "ml", "trading"]

        for layer in layers:
            try:
                logger.info(f"运行 {layer} 层覆盖率测试...")
                cmd = [
                    "python", "tests/framework/test_runner.py", layer,
                    "--coverage", "--verbose"
                ]

                result = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )

                if result.returncode != 0:
                    logger.warning(f"{layer} 层测试失败: {result.returncode}")
                else:
                    logger.info(f"{layer} 层测试完成")

            except subprocess.TimeoutExpired:
                logger.error(f"{layer} 层测试超时")
            except Exception as e:
                logger.error(f"{layer} 层测试出错: {e}")

    def _analyze_coverage_data(self) -> CoverageReport:
        """分析覆盖率数据"""
        logger.info("分析覆盖率数据...")

        report = CoverageReport()

        # 查找覆盖率文件
        coverage_files = list(self.coverage_dir.glob("*.json"))

        if not coverage_files:
            logger.warning("未找到覆盖率文件")
            return report

        # 合并覆盖率数据
        combined_coverage = {}

        for coverage_file in coverage_files:
            try:
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_coverage.update(data)
            except Exception as e:
                logger.error(f"读取覆盖率文件失败 {coverage_file}: {e}")

        # 计算总覆盖率
        total_lines = 0
        covered_lines = 0
        missing_lines = {}

        for file_path, file_data in combined_coverage.items():
            if not isinstance(file_data, dict):
                continue

            lines = file_data.get('l', {})
            file_total = 0
            file_covered = 0
            file_missing = []

            for line_num, hit_count in lines.items():
                if isinstance(hit_count, int):
                    file_total += 1
                    if hit_count > 0:
                        file_covered += 1
                    else:
                        file_missing.append(int(line_num))

            if file_missing:
                missing_lines[file_path] = file_missing

            total_lines += file_total
            covered_lines += file_covered

        report.total_lines = total_lines
        report.covered_lines = covered_lines
        report.coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        report.files_analyzed = len(combined_coverage)
        report.missing_lines = missing_lines

        logger.info(".1"
                   f"覆盖率: {report.coverage_percentage:.1f}%")

        return report

    def _generate_detailed_report(self, report: CoverageReport):
        """生成详细报告"""
        logger.info("生成详细覆盖率报告...")

        report_path = self.coverage_dir / "coverage_analysis_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 测试覆盖率分析报告\n\n")

            f.write("## 总体统计\n\n")
            f.write(f"- **总代码行数**: {report.total_lines:,}\n")
            f.write(f"- **已覆盖行数**: {report.covered_lines:,}\n")
            f.write(".1")
            f.write(f"- **分析文件数**: {report.files_analyzed}\n\n")

            # 覆盖率评级
            if report.coverage_percentage >= 80:
                grade = "🟢 优秀"
            elif report.coverage_percentage >= 70:
                grade = "🟡 良好"
            elif report.coverage_percentage >= 60:
                grade = "🟠 一般"
            else:
                grade = "🔴 不足"

            f.write(f"- **覆盖率评级**: {grade}\n\n")

            # 覆盖不足的文件
            if report.missing_lines:
                f.write("## 覆盖不足的文件\n\n")

                # 按未覆盖行数排序
                sorted_files = sorted(
                    report.missing_lines.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )

                f.write("| 文件路径 | 未覆盖行数 | 未覆盖行号 |\n")
                f.write("|----------|------------|------------|\n")

                for file_path, missing_lines in sorted_files[:20]:  # 只显示前20个
                    missing_count = len(missing_lines)
                    # 简化行号显示（只显示前10个）
                    line_numbers = ", ".join(map(str, missing_lines[:10]))
                    if len(missing_lines) > 10:
                        line_numbers += "..."

                    f.write(f"| `{file_path}` | {missing_count} | {line_numbers} |\n")

                f.write("\n")

            # 改进建议
            f.write("## 改进建议\n\n")

            if report.coverage_percentage < 70:
                f.write("### 🚨 高优先级改进\n\n")
                f.write("1. **增加单元测试**: 为覆盖不足的文件添加更多单元测试\n")
                f.write("2. **完善边界条件**: 添加异常情况和边界条件的测试\n")
                f.write("3. **集成测试补充**: 增加模块间的集成测试\n")
                f.write("4. **Mock策略优化**: 使用更全面的Mock来覆盖难以测试的代码\n\n")

            f.write("### 📈 中期改进计划\n\n")
            f.write("1. **测试用例生成**: 使用AI辅助生成测试用例\n")
            f.write("2. **覆盖率工具集成**: 集成更多覆盖率分析工具\n")
            f.write("3. **持续监控**: 建立覆盖率趋势监控\n")
            f.write("4. **质量门禁**: 设置最低覆盖率要求\n\n")

            f.write("### 🎯 目标达成\n\n")
            current = report.coverage_percentage
            target = 70.0

            if current >= target:
                f.write("✅ **恭喜！已达到70%覆盖率目标**\n\n")
                f.write("🎉 项目测试覆盖率已达到企业级标准！\n\n")
            else:
                remaining = target - current
                f.write(".1"
                        f"💪 继续努力，距离目标还有 {remaining:.1f} 个百分点的提升空间！\n\n")

        logger.info(f"详细报告已保存到: {report_path}")

    def get_coverage_recommendations(self, report: CoverageReport) -> List[str]:
        """获取覆盖率改进建议"""
        recommendations = []

        if report.coverage_percentage < 70:
            recommendations.append("🔴 覆盖率不足70%，需要重点改进")

        if len(report.missing_lines) > 50:
            recommendations.append(f"📁 {len(report.missing_lines)}个文件存在覆盖不足")

        # 分析最需要改进的文件
        if report.missing_lines:
            worst_files = sorted(
                report.missing_lines.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]

            recommendations.append("🎯 优先改进的文件:")
            for file_path, missing in worst_files:
                recommendations.append(f"  - {file_path}: {len(missing)}行未覆盖")

        return recommendations


def main():
    """主函数"""
    analyzer = CoverageAnalyzer()
    report = analyzer.run_coverage_analysis()

    print("\n🎯 覆盖率分析完成")
    print(".1")
    print(f"📁 分析文件数: {report.files_analyzed}")
    print(f"📊 覆盖详情: {report.covered_lines}/{report.total_lines} 行")

    # 显示改进建议
    recommendations = analyzer.get_coverage_recommendations(report)
    if recommendations:
        print("\n💡 改进建议:")
        for rec in recommendations:
            print(f"  {rec}")


if __name__ == "__main__":
    main()

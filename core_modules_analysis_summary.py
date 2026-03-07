#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心服务层分模块AI代码审查汇总报告生成器
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ModuleAnalysis:
    """模块分析结果数据类"""
    module_name: str
    total_files: int
    total_lines: int
    ai_code_quality_score: float
    organization_quality_score: float
    combined_score: float
    refactor_opportunities: int
    risk_level: str
    identified_patterns: int
    organization_issues: int
    organization_suggestions: int


class CoreModulesAnalysisSummary:
    """核心服务层分模块分析汇总器"""

    def __init__(self):
        self.modules: Dict[str, ModuleAnalysis] = {}
        self.module_files = [
            'core_event_bus_review.json',
            'core_container_review.json',
            'core_business_process_review.json',
            'core_foundation_review.json',
            'core_integration_review.json',
            'core_core_optimization_review.json',
            'core_orchestration_review.json',
            'core_core_services_review.json',
            'core_architecture_review.json',
            'core_utils_review.json'
        ]

    def load_analysis_results(self) -> None:
        """加载所有模块的分析结果"""
        for filename in self.module_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    module_name = filename.replace('core_', '').replace('_review.json', '')
                    # 从实际的JSON结构中提取数据
                    metrics = data.get('metrics', {})
                    org_analysis = data.get('organization_analysis', {})
                    org_metrics = org_analysis.get('metrics', {})

                    module_analysis = ModuleAnalysis(
                        module_name=module_name,
                        total_files=metrics.get('total_files', 0),
                        total_lines=metrics.get('total_lines', 0),
                        ai_code_quality_score=data.get('quality_score', 0.0),
                        organization_quality_score=org_metrics.get('quality_score', 0.0),
                        combined_score=data.get('overall_score', 0.0),
                        refactor_opportunities=metrics.get('refactor_opportunities', 0),
                        risk_level=data.get('risk_assessment', {}).get('overall_risk', 'unknown'),
                        identified_patterns=metrics.get('total_patterns', 0),
                        organization_issues=org_analysis.get('issues_count', 0),
                        organization_suggestions=org_analysis.get('recommendations_count', 0)
                    )

                    self.modules[module_name] = module_analysis
                    print(f"✅ 加载 {module_name} 模块分析结果")

                except Exception as e:
                    print(f"❌ 加载 {filename} 失败: {e}")
            else:
                print(f"⚠️  文件 {filename} 不存在")

    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        if not self.modules:
            return "❌ 未找到任何分析结果"

        # 计算总体统计
        total_files = sum(m.total_files for m in self.modules.values())
        total_lines = sum(m.total_lines for m in self.modules.values())
        avg_quality = sum(m.ai_code_quality_score for m in self.modules.values()) / len(self.modules)
        avg_organization = sum(m.organization_quality_score for m in self.modules.values()) / len(self.modules)
        avg_combined = sum(m.combined_score for m in self.modules.values()) / len(self.modules)
        total_refactor = sum(m.refactor_opportunities for m in self.modules.values())

        # 风险等级统计
        risk_counts = defaultdict(int)
        for m in self.modules.values():
            risk_counts[m.risk_level] += 1

        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("🎯 核心服务层分模块AI代码审查汇总报告")
        report.append("=" * 80)
        report.append("")

        report.append("📊 总体统计")
        report.append(f"   • 分析模块数量: {len(self.modules)}")
        report.append(f"   • 总文件数: {total_files}")
        report.append(f"   • 总代码行数: {total_lines:,}")
        report.append(f"   • 平均代码质量评分: {avg_quality:.3f}")
        report.append(f"   • 平均组织质量评分: {avg_organization:.3f}")
        report.append(f"   • 平均综合评分: {avg_combined:.3f}")
        report.append(f"   • 总重构机会: {total_refactor}")
        report.append("")

        report.append("⚠️  风险等级分布")
        for risk_level, count in sorted(risk_counts.items()):
            risk_icon = "🔴" if risk_level == "very_high" else "🟡" if risk_level == "high" else "🟢"
            report.append(f"   • {risk_icon} {risk_level}: {count}个模块")
        report.append("")

        # 各模块详细分析
        report.append("📋 各模块详细分析")
        report.append("-" * 80)

        # 按综合评分排序
        sorted_modules = sorted(self.modules.items(),
                              key=lambda x: x[1].combined_score,
                              reverse=True)

        for module_name, analysis in sorted_modules:
            report.append(f"\n🔍 {module_name.upper()} 模块")
            report.append(f"   📁 文件数: {analysis.total_files}")
            report.append(f"   📝 代码行: {analysis.total_lines:,}")
            report.append(f"   ⭐ 代码质量: {analysis.ai_code_quality_score:.3f}")
            report.append(f"   🏗️ 组织质量: {analysis.organization_quality_score:.3f}")
            report.append(f"   🎯 综合评分: {analysis.combined_score:.3f}")
            report.append(f"   🔧 重构机会: {analysis.refactor_opportunities}")
            report.append(f"   ⚠️  风险等级: {analysis.risk_level}")
            report.append(f"   🎨 识别模式: {analysis.identified_patterns}")
            report.append(f"   📋 组织问题: {analysis.organization_issues}")
            report.append(f"   💡 优化建议: {analysis.organization_suggestions}")

        # 优化建议
        report.append("\n" + "=" * 80)
        report.append("🎯 优化建议汇总")
        report.append("=" * 80)

        # 找出需要重点关注的模块
        high_risk_modules = [name for name, m in self.modules.items()
                           if m.risk_level == "very_high"]
        low_quality_modules = [(name, m.combined_score) for name, m in self.modules.items()
                             if m.combined_score < 0.85]

        if high_risk_modules:
            report.append(f"\n🔴 高风险模块 ({len(high_risk_modules)}个):")
            for module in high_risk_modules:
                analysis = self.modules[module]
                report.append(f"   • {module}: 综合评分{analysis.combined_score:.3f}, "
                            f"重构机会{analysis.refactor_opportunities}")

        if low_quality_modules:
            report.append(f"\n🟡 质量待优化模块 ({len(low_quality_modules)}个):")
            for module, score in sorted(low_quality_modules, key=lambda x: x[1]):
                report.append(f"   • {module}: 综合评分{score:.3f}")

        # 总体评估
        report.append("\n🎖️ 总体评估")
        if avg_combined >= 0.9:
            report.append("   ✅ 优秀: 核心服务层整体质量优秀，各模块协同良好")
        elif avg_combined >= 0.8:
            report.append("   🟡 良好: 核心服务层整体质量良好，需要持续优化")
        else:
            report.append("   🔴 待优化: 核心服务层整体质量需要重点改进")

        report.append("\n📈 优化方向:")
        report.append("   1. 重点解决高风险模块的重构机会")
        report.append("   2. 提升组织质量相对较低的模块")
        report.append("   3. 统一各模块的设计模式和编码规范")
        report.append("   4. 加强模块间的接口一致性和文档完整性")

        return "\n".join(report)


def main():
    """主函数"""
    print("🚀 开始生成核心服务层分模块分析汇总报告...")

    analyzer = CoreModulesAnalysisSummary()
    analyzer.load_analysis_results()

    if not analyzer.modules:
        print("❌ 未找到任何分析结果文件")
        return

    report = analyzer.generate_summary_report()

    # 保存报告
    report_filename = "core_service_layer_modules_analysis_summary.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 报告已保存到: {report_filename}")
    print("\n" + "="*50)
    print(report)
    print("="*50)


if __name__ == "__main__":
    main()

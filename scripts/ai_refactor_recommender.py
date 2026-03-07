#!/usr/bin/env python3
"""
AI智能重构建议系统

基于AI分析结果生成智能重构建议和执行计划。
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import statistics

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class AIRefactorRecommender:
    """
    AI智能重构建议器

    基于分析结果生成优先级排序的重构建议。
    """

    def __init__(self, analysis_result_path: str):
        self.analysis_result_path = analysis_result_path
        self.analysis_data = self._load_analysis_data()

        # 优先级权重
        self.priority_weights = {
            'severity': {
                'critical': 100,
                'high': 80,
                'medium': 60,
                'low': 40,
                'info': 20
            },
            'confidence': 30,
            'impact': {
                'performance': 25,
                'maintainability': 20,
                'reliability': 20,
                'security': 30
            },
            'effort': {
                'low': 20,
                'medium': 10,
                'high': -10,
                'very_high': -20
            },
            'automation': 15  # 自动化执行加分
        }

    def _load_analysis_data(self) -> Dict[str, Any]:
        """加载分析数据"""
        try:
            with open(self.analysis_result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 无法加载分析数据: {e}")
            return {}

    def generate_recommendations(self) -> Dict[str, Any]:
        """
        生成重构建议

        Returns:
            Dict[str, Any]: 重构建议报告
        """
        if not self.analysis_data:
            return {"error": "无法加载分析数据"}

        print("🧠 生成AI重构建议...")

        # 解析重构机会（需要从详细分析结果中获取）
        opportunities = self._extract_opportunities()

        # 计算优先级
        prioritized_opportunities = self._calculate_priorities(opportunities)

        # 生成执行计划
        execution_plan = self._generate_execution_plan(prioritized_opportunities)

        # 生成统计报告
        statistics_report = self._generate_statistics_report(prioritized_opportunities)

        # 生成风险评估
        risk_assessment = self._assess_execution_risks(execution_plan)

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "quality_score": self.analysis_data.get("quality_score", 0),
                "total_opportunities": len(opportunities),
                "risk_level": self.analysis_data.get("risk_assessment", {}).get("overall_risk", "unknown")
            },
            "prioritized_opportunities": prioritized_opportunities[:20],  # 前20个优先级最高的机会
            "execution_plan": execution_plan,
            "statistics": statistics_report,
            "risk_assessment": risk_assessment,
            "recommendations": self._generate_smart_recommendations(prioritized_opportunities)
        }

    def _extract_opportunities(self) -> List[Dict[str, Any]]:
        """从分析数据中提取重构机会"""
        # 注意：这里需要实际的重构机会数据
        # 在实际实现中，这些数据应该来自IntelligentCodeAnalyzer的详细输出

        # 从详细分析结果中提取实际的重构机会
        opportunities = []
        detailed_opportunities = self.analysis_data.get("opportunities", [])

        # 如果有详细机会数据，直接使用
        if detailed_opportunities:
            for opp in detailed_opportunities:
                opportunity = {
                    "id": opp.get("opportunity_id", f"opp_{len(opportunities)}"),
                    "title": opp.get("title", "重构机会"),
                    "severity": opp.get("severity", "medium"),
                    "confidence": opp.get("confidence", 0.7),
                    "effort": opp.get("effort", "medium"),
                    "impact": opp.get("impact", "maintainability"),
                    "automated": opp.get("automated", False),
                    "risk_level": opp.get("risk_level", "medium"),
                    "description": opp.get("description", ""),
                    "file_path": opp.get("file_path", ""),
                    "line_number": opp.get("line_number", 0),
                    "suggested_fix": opp.get("suggested_fix", "")
                }
                opportunities.append(opportunity)
        else:
            # 如果没有详细数据，回退到基于统计信息的模拟机会
            opportunities = []
            risk_data = self.analysis_data.get("risk_assessment", {})
            severity_counts = risk_data.get("severity_breakdown", {})

            for severity, count in severity_counts.items():
                for i in range(min(count, 5)):
                    opportunity = {
                        "id": f"auto_{severity}_{i}",
                        "title": f"{severity.upper()}级别重构机会 #{i+1}",
                        "severity": severity,
                        "confidence": 0.8 if severity in ['critical', 'high'] else 0.6,
                        "effort": "medium",
                        "impact": "maintainability",
                        "automated": severity == 'low',
                        "risk_level": "low" if severity == 'low' else 'medium',
                        "description": f"AI识别的{severity}级别代码质量问题",
                        "file_path": "",
                        "line_number": 0,
                        "suggested_fix": ""
                    }
                    opportunities.append(opportunity)

        return opportunities

    def _calculate_priorities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算重构机会的优先级"""
        for opp in opportunities:
            priority_score = 0

            # 严重程度权重
            priority_score += self.priority_weights['severity'].get(opp['severity'], 0)

            # 置信度权重
            priority_score += opp['confidence'] * self.priority_weights['confidence']

            # 影响权重
            priority_score += self.priority_weights['impact'].get(opp['impact'], 0)

            # 工作量权重（负权重，低工作量优先）
            priority_score += self.priority_weights['effort'].get(opp['effort'], 0)

            # 自动化加分
            if opp.get('automated', False):
                priority_score += self.priority_weights['automation']

            opp['priority_score'] = priority_score

        # 按优先级排序
        return sorted(opportunities, key=lambda x: x['priority_score'], reverse=True)

    def _generate_execution_plan(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成执行计划"""
        # 分阶段执行
        phases = {
            "phase_1_immediate": [],  # 立即执行（高优先级，自动化）
            "phase_2_short_term": [],  # 短期执行（中等优先级）
            "phase_3_medium_term": [],  # 中期执行（低优先级，手动）
            "phase_4_long_term": []   # 长期规划（复杂重构）
        }

        for opp in opportunities:
            score = opp['priority_score']
            automated = opp.get('automated', False)
            effort = opp['effort']

            if score >= 150 and automated:
                phases["phase_1_immediate"].append(opp)
            elif score >= 120 or (automated and effort == 'low'):
                phases["phase_2_short_term"].append(opp)
            elif score >= 80 or effort in ['low', 'medium']:
                phases["phase_3_medium_term"].append(opp)
            else:
                phases["phase_4_long_term"].append(opp)

        return {
            "total_opportunities": len(opportunities),
            "phases": phases,
            "estimated_timeline": self._estimate_timeline(phases),
            "resource_requirements": self._estimate_resources(phases)
        }

    def _estimate_timeline(self, phases: Dict[str, List]) -> Dict[str, str]:
        """估算时间线"""
        phase_durations = {
            "phase_1_immediate": f"{len(phases['phase_1_immediate'])} 天",
            "phase_2_short_term": f"{len(phases['phase_2_short_term']) * 2} 天",
            "phase_3_medium_term": f"{len(phases['phase_3_medium_term']) * 5} 天",
            "phase_4_long_term": f"{len(phases['phase_4_long_term']) * 10} 天"
        }

        total_days = sum([
            len(phases['phase_1_immediate']),
            len(phases['phase_2_short_term']) * 2,
            len(phases['phase_3_medium_term']) * 5,
            len(phases['phase_4_long_term']) * 10
        ])

        return {
            **phase_durations,
            "total_estimate": f"{total_days} 天",
            "parallel_execution": f"{max(7, total_days // 3)} 天"  # 假设3个开发人员并行
        }

    def _estimate_resources(self, phases: Dict[str, List]) -> Dict[str, Any]:
        """估算资源需求"""
        total_opportunities = sum(len(phase) for phase in phases.values())

        return {
            "developers_needed": min(5, max(1, total_opportunities // 20)),
            "automated_scripts": sum(1 for phase in phases.values()
                                     for opp in phase if opp.get('automated', False)),
            "manual_reviews": total_opportunities,
            "testing_hours": total_opportunities * 2,
            "documentation_updates": len(phases["phase_3_medium_term"]) + len(phases["phase_4_long_term"])
        }

    def _generate_statistics_report(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成统计报告"""
        if not opportunities:
            return {}

        # 基本统计
        severity_counts = {}
        effort_counts = {}
        impact_counts = {}
        automated_count = 0

        for opp in opportunities:
            severity_counts[opp['severity']] = severity_counts.get(opp['severity'], 0) + 1
            effort_counts[opp['effort']] = effort_counts.get(opp['effort'], 0) + 1
            impact_counts[opp['impact']] = impact_counts.get(opp['impact'], 0) + 1
            if opp.get('automated', False):
                automated_count += 1

        # 优先级分布
        priority_scores = [opp['priority_score'] for opp in opportunities]

        return {
            "total_opportunities": len(opportunities),
            "severity_distribution": severity_counts,
            "effort_distribution": effort_counts,
            "impact_distribution": impact_counts,
            "automated_percentage": automated_count / len(opportunities) * 100,
            "priority_stats": {
                "mean": statistics.mean(priority_scores),
                "median": statistics.median(priority_scores),
                "max": max(priority_scores),
                "min": min(priority_scores)
            },
            "automation_potential": f"{automated_count}/{len(opportunities)} 可自动化"
        }

    def _assess_execution_risks(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """评估执行风险"""
        phases = execution_plan['phases']

        # 计算风险指标
        high_risk_opportunities = sum(1 for phase in phases.values()
                                      for opp in phase
                                      if opp.get('risk_level') in ['high', 'very_high'])

        manual_opportunities = sum(1 for phase in phases.values()
                                   for opp in phase
                                   if not opp.get('automated', False))

        complex_opportunities = sum(1 for phase in phases.values()
                                    for opp in phase
                                    if opp['effort'] in ['high', 'very_high'])

        risk_score = (high_risk_opportunities * 3 +
                      manual_opportunities * 1 +
                      complex_opportunities * 2) / execution_plan['total_opportunities']

        return {
            "overall_risk_level": "high" if risk_score > 3 else "medium" if risk_score > 1.5 else "low",
            "risk_factors": {
                "high_risk_opportunities": high_risk_opportunities,
                "manual_intervention_required": manual_opportunities,
                "complex_changes": complex_opportunities
            },
            "mitigation_strategies": [
                "分阶段执行，优先处理低风险自动化任务",
                "建立完整的测试套件确保重构安全性",
                "准备回滚计划和备份策略",
                "设置代码审查流程验证重构质量"
            ],
            "recommended_approach": "incremental" if risk_score > 2 else "comprehensive"
        }

    def _generate_smart_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成智能建议"""
        recommendations = []

        # 基于分析结果的智能建议
        stats = self._generate_statistics_report(opportunities)

        if stats.get('automated_percentage', 0) > 50:
            recommendations.append({
                "type": "automation_first",
                "title": "优先执行自动化重构",
                "description": "超过50%的重构机会可以自动化执行，建议优先处理这些低风险任务",
                "benefit": "快速提升代码质量，降低手动错误风险"
            })

        if stats['priority_stats']['max'] > 150:
            recommendations.append({
                "type": "high_priority_focus",
                "title": "聚焦高优先级问题",
                "description": "存在高优先级重构机会，建议优先处理严重影响系统的问题",
                "benefit": "最大化重构投资回报"
            })

        severity_dist = stats.get('severity_distribution', {})
        if severity_dist.get('critical', 0) > 0:
            recommendations.append({
                "type": "critical_first",
                "title": "立即处理关键问题",
                "description": f"发现{severity_dist['critical']}个关键级别问题，需要立即处理",
                "benefit": "防止潜在的系统风险和故障"
            })

        return recommendations


def generate_execution_script(recommendations: Dict[str, Any], output_path: str):
    """生成执行脚本"""
    script_content = f'''#!/bin/bash
"""
AI智能重构执行脚本

基于AI分析结果自动生成的重构执行计划。
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

set -e

echo "🚀 开始AI智能重构执行..."

# 执行计划统计
echo "📊 执行计划统计:"
echo "  • 总重构机会: {recommendations['analysis_summary']['total_opportunities']}"
echo "  • 质量评分: {recommendations['analysis_summary']['quality_score']:.3f}"
echo "  • 风险等级: {recommendations['analysis_summary']['risk_level']}"

# Phase 1: 立即执行
echo ""
echo "🔧 Phase 1: 立即执行 ({len(recommendations['execution_plan']['phases']['phase_1_immediate'])} 项)"
'''

    for i, opp in enumerate(recommendations['execution_plan']['phases']['phase_1_immediate'][:5]):
        script_content += f'''
echo "  {i+1}. {opp['title']}"
# TODO: 实现自动化重构逻辑
'''

    script_content += '''
# Phase 2: 短期执行
echo ""
echo "📅 Phase 2: 短期执行"
echo "  计划手动处理中等优先级问题..."

# Phase 3: 中期执行
echo ""
echo "📆 Phase 3: 中期执行"
echo "  计划处理复杂重构任务..."

echo ""
echo "✅ 重构执行脚本生成完成"
echo "请按照上述计划逐步执行重构任务"
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # 设置执行权限
    import os
    os.chmod(output_path, 0o755)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI智能重构建议系统")
    parser.add_argument('analysis_result', help='AI分析结果文件路径')
    parser.add_argument('--output', '-o', default='ai_refactor_recommendations.json',
                        help='输出建议文件路径')
    parser.add_argument('--script', '-s', help='生成执行脚本路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 生成建议
    recommender = AIRefactorRecommender(args.analysis_result)
    recommendations = recommender.generate_recommendations()

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print(f"📄 重构建议已保存到: {args.output}")

    # 显示关键信息
    summary = recommendations['analysis_summary']
    stats = recommendations['statistics']

    print("\n🎯 重构建议摘要:")
    print(f"  • 质量评分: {summary['quality_score']:.3f}")
    print(f"  • 重构机会: {summary['total_opportunities']}")
    print(f"  • 风险等级: {summary['risk_level']}")
    print(f"  • 可自动化: {stats['automated_percentage']:.1f}%")

    # 显示执行计划
    plan = recommendations['execution_plan']
    print("\n📋 执行计划:")
    print(f"  • Phase 1 (立即): {len(plan['phases']['phase_1_immediate'])} 项")
    print(f"  • Phase 2 (短期): {len(plan['phases']['phase_2_short_term'])} 项")
    print(f"  • Phase 3 (中期): {len(plan['phases']['phase_3_medium_term'])} 项")
    print(f"  • Phase 4 (长期): {len(plan['phases']['phase_4_long_term'])} 项")
    print(f"  • 预计总时间: {plan['estimated_timeline']['total_estimate']}")

    # 显示智能建议
    if recommendations['recommendations']:
        print("\n🧠 AI智能建议:")
        for rec in recommendations['recommendations']:
            print(f"  • {rec['title']}: {rec['description']}")

    # 生成执行脚本
    if args.script:
        generate_execution_script(recommendations, args.script)
        print(f"📜 执行脚本已生成: {args.script}")

    print("\n🎉 AI智能重构建议生成完成！")


if __name__ == '__main__':
    main()

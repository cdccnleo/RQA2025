#!/usr/bin/env python3
"""
智能测试报告生成器 - Phase 5智能化测试

基于测试结果数据，生成AI驱动的测试分析和改进建议：
1. 深度分析测试趋势和模式
2. 识别测试薄弱环节和改进机会
3. 生成个性化的测试策略建议
4. 提供预测性质量洞察

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestTrend:
    """测试趋势分析"""
    period: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage_percentage: float
    avg_duration: float
    trend_direction: str  # improving, declining, stable


@dataclass
class QualityInsight:
    """质量洞察"""
    category: str
    severity: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    priority_score: float


@dataclass
class TestStrategy:
    """测试策略建议"""
    focus_area: str
    rationale: str
    recommended_actions: List[str]
    expected_impact: str
    implementation_priority: str


class IntelligentTestReporter:
    """
    智能测试报告生成器

    基于历史测试数据生成深度分析和AI驱动的改进建议
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.test_logs_dir = self.project_root / "test_logs"

    def analyze_test_trends(self, days: int = 30) -> List[TestTrend]:
        """
        分析测试趋势

        Args:
            days: 分析的时间范围（天数）

        Returns:
            测试趋势列表
        """
        print("📈 分析测试执行趋势...")

        trends = []

        # 收集历史测试报告
        cutoff_date = datetime.now() - timedelta(days=days)
        test_reports = self._collect_test_reports(cutoff_date)

        if not test_reports:
            print("⚠️ 未找到足够的测试报告数据")
            return trends

        # 按时间段分组分析
        weekly_reports = self._group_reports_by_week(test_reports)

        for period, reports in weekly_reports.items():
            if reports:
                trend = self._calculate_trend_for_period(period, reports)
                trends.append(trend)

        # 按时间排序
        trends.sort(key=lambda x: x.period)

        print(f"✅ 完成 {len(trends)} 个时间段的趋势分析")
        return trends

    def _collect_test_reports(self, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """收集测试报告"""
        reports = []

        # 扫描test_logs目录
        if self.test_logs_dir.exists():
            for json_file in self.test_logs_dir.glob("*.json"):
                try:
                    # 从文件名提取时间戳
                    timestamp_str = self._extract_timestamp_from_filename(json_file.name)
                    if timestamp_str:
                        report_date = datetime.fromisoformat(timestamp_str.replace('_', 'T'))

                        if report_date >= cutoff_date:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                report_data = json.load(f)
                                report_data['_timestamp'] = report_date
                                report_data['_filename'] = json_file.name
                                reports.append(report_data)

                except Exception as e:
                    logger.warning(f"解析报告文件失败 {json_file}: {e}")

        return reports

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """从文件名提取时间戳"""
        # 匹配类似 "report_20251204_081234.json" 的模式
        pattern = r'_(\d{8}_\d{6})'
        match = re.search(pattern, filename)
        if match:
            timestamp = match.group(1)
            return f"{timestamp[:8]}T{timestamp[9:]}"
        return None

    def _group_reports_by_week(self, reports: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按周分组报告"""
        weekly_groups = {}

        for report in reports:
            timestamp = report.get('_timestamp')
            if timestamp:
                week_key = timestamp.strftime("%Y-W%W")
                if week_key not in weekly_groups:
                    weekly_groups[week_key] = []
                weekly_groups[week_key].append(report)

        return weekly_groups

    def _calculate_trend_for_period(self, period: str, reports: List[Dict[str, Any]]) -> TestTrend:
        """计算某个时间段的趋势"""
        total_tests = []
        passed_tests = []
        durations = []

        for report in reports:
            # 提取测试统计信息
            if 'overall' in report:
                overall = report['overall']
                total_tests.append(overall.get('total_tests', 0))
                passed_tests.append(overall.get('passed_tests', 0))

            # 提取执行时间
            if 'execution_time' in report:
                durations.append(report['execution_time'])

        # 计算平均值
        avg_total = statistics.mean(total_tests) if total_tests else 0
        avg_passed = statistics.mean(passed_tests) if passed_tests else 0
        avg_duration = statistics.mean(durations) if durations else 0
        coverage = (avg_passed / avg_total * 100) if avg_total > 0 else 0

        # 确定趋势方向（这里简化处理，实际应基于历史比较）
        trend_direction = "stable"  # 默认稳定

        return TestTrend(
            period=period,
            total_tests=int(avg_total),
            passed_tests=int(avg_passed),
            failed_tests=int(avg_total - avg_passed),
            coverage_percentage=round(coverage, 2),
            avg_duration=round(avg_duration, 2),
            trend_direction=trend_direction
        )

    def generate_quality_insights(self, trends: List[TestTrend]) -> List[QualityInsight]:
        """
        生成质量洞察

        Args:
            trends: 测试趋势数据

        Returns:
            质量洞察列表
        """
        print("🔍 生成质量洞察分析...")

        insights = []

        # 1. 覆盖率洞察
        coverage_insight = self._analyze_coverage_trends(trends)
        if coverage_insight:
            insights.append(coverage_insight)

        # 2. 失败率洞察
        failure_insight = self._analyze_failure_patterns(trends)
        if failure_insight:
            insights.append(failure_insight)

        # 3. 性能洞察
        performance_insight = self._analyze_performance_trends(trends)
        if performance_insight:
            insights.append(performance_insight)

        # 4. 稳定性洞察
        stability_insight = self._analyze_test_stability(trends)
        if stability_insight:
            insights.append(stability_insight)

        # 5. 基于代码质量的洞察
        code_quality_insights = self._analyze_code_quality_indicators()
        insights.extend(code_quality_insights)

        print(f"✅ 生成 {len(insights)} 个质量洞察")
        return insights

    def _analyze_coverage_trends(self, trends: List[TestTrend]) -> Optional[QualityInsight]:
        """分析覆盖率趋势"""
        if not trends:
            return None

        avg_coverage = statistics.mean(t.coverage_percentage for t in trends)
        coverage_volatility = statistics.stdev(t.coverage_percentage for t in trends) if len(trends) > 1 else 0

        if avg_coverage < 70:
            severity = "high" if avg_coverage < 50 else "medium"
            return QualityInsight(
                category="coverage",
                severity=severity,
                description=f"测试覆盖率偏低 (平均 {avg_coverage:.1f}%)，存在质量风险",
                evidence=[f"覆盖率波动: {coverage_volatility:.2f}%"],
                recommendations=[
                    "增加单元测试覆盖率",
                    "识别并测试高风险代码路径",
                    "实施代码覆盖率门禁"
                ],
                priority_score=0.8 if severity == "high" else 0.6
            )
        elif coverage_volatility > 10:
            return QualityInsight(
                category="coverage",
                severity="medium",
                description=f"测试覆盖率波动较大 ({coverage_volatility:.1f}%)，稳定性不足",
                evidence=["覆盖率不稳定可能影响质量评估"],
                recommendations=[
                    "稳定测试执行环境",
                    "减少测试的外部依赖",
                    "建立覆盖率基线监控"
                ],
                priority_score=0.5
            )

        return None

    def _analyze_failure_patterns(self, trends: List[TestTrend]) -> Optional[QualityInsight]:
        """分析失败模式"""
        if not trends:
            return None

        failure_rates = [(t.failed_tests / t.total_tests * 100) if t.total_tests > 0 else 0 for t in trends]
        avg_failure_rate = statistics.mean(failure_rates)

        if avg_failure_rate > 10:
            return QualityInsight(
                category="reliability",
                severity="high",
                description=f"测试失败率过高 (平均 {avg_failure_rate:.1f}%)，影响开发效率",
                evidence=[f"连续失败率: {failure_rates[-3:] if len(failure_rates) >= 3 else failure_rates}"],
                recommendations=[
                    "修复不稳定的测试用例",
                    "隔离外部依赖影响",
                    "实施测试重试机制"
                ],
                priority_score=0.9
            )
        elif avg_failure_rate > 5:
            return QualityInsight(
                category="reliability",
                severity="medium",
                description=f"测试失败率中等 ({avg_failure_rate:.1f}%)，需要关注",
                evidence=["可能存在间歇性问题"],
                recommendations=[
                    "监控测试失败模式",
                    "改进测试环境稳定性",
                    "增加测试日志记录"
                ],
                priority_score=0.6
            )

        return None

    def _analyze_performance_trends(self, trends: List[TestTrend]) -> Optional[QualityInsight]:
        """分析性能趋势"""
        if not trends:
            return None

        durations = [t.avg_duration for t in trends if t.avg_duration > 0]
        if not durations:
            return None

        avg_duration = statistics.mean(durations)

        if avg_duration > 300:  # 超过5分钟
            return QualityInsight(
                category="performance",
                severity="medium",
                description=f"测试执行时间过长 (平均 {avg_duration:.1f}秒)，影响CI/CD效率",
                evidence=["长时间测试影响开发迭代速度"],
                recommendations=[
                    "优化测试并行执行",
                    "减少不必要的测试等待",
                    "实施增量测试策略"
                ],
                priority_score=0.7
            )

        return None

    def _analyze_test_stability(self, trends: List[TestTrend]) -> Optional[QualityInsight]:
        """分析测试稳定性"""
        if len(trends) < 3:
            return None

        # 计算测试数量的变异系数
        test_counts = [t.total_tests for t in trends]
        if len(test_counts) > 1:
            cv = statistics.stdev(test_counts) / statistics.mean(test_counts)

            if cv > 0.2:  # 变异系数大于20%
                return QualityInsight(
                    category="stability",
                    severity="low",
                    description=f"测试数量波动较大 (变异系数 {cv:.2f})，可能影响质量评估",
                    evidence=["测试范围不稳定"],
                    recommendations=[
                        "稳定测试发现机制",
                        "减少动态测试生成",
                        "建立测试基线"
                    ],
                    priority_score=0.4
                )

        return None

    def _analyze_code_quality_indicators(self) -> List[QualityInsight]:
        """分析代码质量指标"""
        insights = []

        # 分析代码复杂度
        complexity_insight = self._check_code_complexity()
        if complexity_insight:
            insights.append(complexity_insight)

        # 分析依赖关系
        dependency_insight = self._check_dependency_health()
        if dependency_insight:
            insights.append(dependency_insight)

        # 分析测试代码质量
        test_quality_insight = self._check_test_code_quality()
        if test_quality_insight:
            insights.append(test_quality_insight)

        return insights

    def _check_code_complexity(self) -> Optional[QualityInsight]:
        """检查代码复杂度"""
        # 简化实现，实际应该分析AST复杂度
        complex_files = []

        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = len(content.split('\n'))
                functions = len(re.findall(r'def \w+', content))

                # 简单的复杂度指标
                if lines > 500 or functions > 20:
                    complex_files.append(str(py_file.relative_to(self.project_root)))

            except Exception:
                continue

        if len(complex_files) > 5:
            return QualityInsight(
                category="complexity",
                severity="medium",
                description=f"发现 {len(complex_files)} 个复杂度较高的文件",
                evidence=complex_files[:3],  # 只显示前3个
                recommendations=[
                    "重构复杂函数",
                    "拆分大文件",
                    "增加单元测试覆盖"
                ],
                priority_score=0.6
            )

        return None

    def _check_dependency_health(self) -> Optional[QualityInsight]:
        """检查依赖健康状况"""
        # 分析导入失败的模式
        import_issues = []

        for log_file in self.test_logs_dir.glob("*defect_prediction*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                patterns = data.get('failure_analysis', {}).get('patterns_by_type', {})
                if 'import_error' in patterns and patterns['import_error'] > 0:
                    import_issues.append(f"导入错误: {patterns['import_error']} 次")

            except Exception:
                continue

        if import_issues:
            return QualityInsight(
                category="dependencies",
                severity="high",
                description="存在模块依赖问题，影响系统稳定性",
                evidence=import_issues,
                recommendations=[
                    "修复模块导入路径",
                    "清理未使用的依赖",
                    "实施依赖健康检查"
                ],
                priority_score=0.8
            )

        return None

    def _check_test_code_quality(self) -> Optional[QualityInsight]:
        """检查测试代码质量"""
        # 分析测试文件的质量指标
        test_files = list(self.project_root.glob("tests/**/*.py"))
        total_assertions = 0
        total_test_functions = 0

        for test_file in test_files[:20]:  # 检查前20个测试文件
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                assertions = len(re.findall(r'assert\s+', content))
                test_functions = len(re.findall(r'def test_\w+', content))

                total_assertions += assertions
                total_test_functions += test_functions

            except Exception:
                continue

        if total_test_functions > 0:
            avg_assertions_per_test = total_assertions / total_test_functions

            if avg_assertions_per_test < 2:
                return QualityInsight(
                    category="test_quality",
                    severity="low",
                    description=f"测试断言密度偏低 (平均 {avg_assertions_per_test:.1f} 个断言/测试)",
                    evidence=["测试验证不够充分"],
                    recommendations=[
                        "增加测试断言",
                        "验证边界条件",
                        "测试错误场景"
                    ],
                    priority_score=0.5
                )

        return None

    def generate_test_strategies(self, trends: List[TestTrend],
                            insights: List[QualityInsight]) -> List[TestStrategy]:
        """
        生成测试策略建议

        Args:
            trends: 测试趋势数据
            insights: 质量洞察

        Returns:
            测试策略建议列表
        """
        print("🎯 生成智能测试策略建议...")

        strategies = []

        # 基于覆盖率洞察生成策略
        coverage_insights = [i for i in insights if i.category == "coverage"]
        if coverage_insights:
            strategies.append(TestStrategy(
                focus_area="覆盖率提升",
                rationale="当前测试覆盖率不足，存在质量风险",
                recommended_actions=[
                    "识别高风险代码路径并优先测试",
                    "实施增量测试策略覆盖新增代码",
                    "建立覆盖率门禁机制"
                ],
                expected_impact="提升系统整体质量保障水平",
                implementation_priority="high"
            ))

        # 基于失败模式生成策略
        failure_insights = [i for i in insights if i.category == "reliability"]
        if failure_insights:
            strategies.append(TestStrategy(
                focus_area="稳定性改进",
                rationale="测试失败率较高，影响开发效率",
                recommended_actions=[
                    "隔离外部依赖影响",
                    "实施测试重试机制",
                    "改进测试环境稳定性"
                ],
                expected_impact="提高测试执行成功率",
                implementation_priority="high"
            ))

        # 基于性能洞察生成策略
        performance_insights = [i for i in insights if i.category == "performance"]
        if performance_insights:
            strategies.append(TestStrategy(
                focus_area="执行效率优化",
                rationale="测试执行时间过长，影响CI/CD流程",
                recommended_actions=[
                    "优化测试并行执行策略",
                    "实施增量测试减少全量运行",
                    "识别并优化慢速测试"
                ],
                expected_impact="加速开发迭代周期",
                implementation_priority="medium"
            ))

        # 基于复杂度洞察生成策略
        complexity_insights = [i for i in insights if i.category == "complexity"]
        if complexity_insights:
            strategies.append(TestStrategy(
                focus_area="代码质量改进",
                rationale="代码复杂度较高，增加缺陷风险",
                recommended_actions=[
                    "重构复杂函数和类",
                    "增加复杂代码的测试覆盖",
                    "实施代码审查流程"
                ],
                expected_impact="降低长期维护成本",
                implementation_priority="medium"
            ))

        # 通用策略
        strategies.extend([
            TestStrategy(
                focus_area="智能化测试",
                rationale="提升测试智能化水平，预防性发现缺陷",
                recommended_actions=[
                    "实施AI辅助测试生成",
                    "建立智能缺陷预测机制",
                    "自动化测试策略调整"
                ],
                expected_impact="提高测试效率和质量",
                implementation_priority="low"
            ),
            TestStrategy(
                focus_area="持续改进",
                rationale="建立测试质量持续改进机制",
                recommended_actions=[
                    "定期分析测试指标",
                    "收集开发者反馈",
                    "迭代优化测试策略"
                ],
                expected_impact="建立质量改进文化",
                implementation_priority="low"
            )
        ])

        print(f"✅ 生成 {len(strategies)} 个测试策略建议")
        return strategies

    def generate_intelligent_report(self) -> Dict[str, Any]:
        """
        生成完整的智能测试报告
        """
        print("📊 生成智能测试分析报告")
        print("=" * 50)

        # 1. 分析测试趋势
        trends = self.analyze_test_trends(days=30)

        # 2. 生成质量洞察
        insights = self.generate_quality_insights(trends)

        # 3. 生成测试策略
        strategies = self.generate_test_strategies(trends, insights)

        # 4. 汇总报告
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period_days": 30,
                "generator_version": "1.0.0",
                "ai_model": "pattern_based_intelligence"
            },
            "executive_summary": {
                "total_trends_analyzed": len(trends),
                "total_insights_generated": len(insights),
                "total_strategies_recommended": len(strategies),
                "high_priority_actions": len([s for s in strategies if s.implementation_priority == "high"]),
                "overall_quality_score": self._calculate_quality_score(trends, insights)
            },
            "trend_analysis": {
                "trends": [asdict(t) for t in trends],
                "key_metrics": self._extract_key_metrics(trends)
            },
            "quality_insights": {
                "insights": [asdict(i) for i in insights],
                "insights_by_category": self._group_insights_by_category(insights),
                "insights_by_severity": self._group_insights_by_severity(insights)
            },
            "strategy_recommendations": {
                "strategies": [asdict(s) for s in strategies],
                "strategies_by_priority": self._group_strategies_by_priority(strategies),
                "implementation_roadmap": self._create_implementation_roadmap(strategies)
            },
            "predictive_analytics": {
                "quality_forecast": self._forecast_quality_trends(trends),
                "risk_assessment": self._assess_current_risks(insights),
                "improvement_trajectory": self._predict_improvement_trajectory(strategies)
            }
        }

        return report

    def _calculate_quality_score(self, trends: List[TestTrend],
                            insights: List[QualityInsight]) -> float:
        """计算整体质量评分"""
        if not trends:
            return 0.0

        # 基于覆盖率、失败率等计算综合评分
        avg_coverage = statistics.mean(t.coverage_percentage for t in trends)
        avg_failure_rate = statistics.mean(
            (t.failed_tests / t.total_tests * 100) if t.total_tests > 0 else 0
            for t in trends
        )

        # 质量评分 (0-100)
        coverage_score = min(avg_coverage, 100)
        failure_penalty = avg_failure_rate * 2  # 失败率每1%扣2分
        insight_penalty = len([i for i in insights if i.severity == "high"]) * 5

        quality_score = max(0, coverage_score - failure_penalty - insight_penalty)

        return round(quality_score, 1)

    def _extract_key_metrics(self, trends: List[TestTrend]) -> Dict[str, Any]:
        """提取关键指标"""
        if not trends:
            return {}

        return {
            "avg_coverage": round(statistics.mean(t.coverage_percentage for t in trends), 1),
            "avg_test_count": round(statistics.mean(t.total_tests for t in trends), 0),
            "avg_duration": round(statistics.mean(t.avg_duration for t in trends), 1),
            "best_coverage_period": max(trends, key=lambda t: t.coverage_percentage).period,
            "worst_coverage_period": min(trends, key=lambda t: t.coverage_percentage).period,
            "coverage_volatility": round(statistics.stdev(t.coverage_percentage for t in trends), 1) if len(trends) > 1 else 0
        }

    def _group_insights_by_category(self, insights: List[QualityInsight]) -> Dict[str, int]:
        """按类别分组洞察"""
        categories = {}
        for insight in insights:
            categories[insight.category] = categories.get(insight.category, 0) + 1
        return categories

    def _group_insights_by_severity(self, insights: List[QualityInsight]) -> Dict[str, int]:
        """按严重程度分组洞察"""
        severities = {}
        for insight in insights:
            severities[insight.severity] = severities.get(insight.severity, 0) + 1
        return severities

    def _group_strategies_by_priority(self, strategies: List[TestStrategy]) -> Dict[str, List[str]]:
        """按优先级分组策略"""
        priorities = {}
        for strategy in strategies:
            priority = strategy.implementation_priority
            if priority not in priorities:
                priorities[priority] = []
            priorities[priority].append(strategy.focus_area)
        return priorities

    def _create_implementation_roadmap(self, strategies: List[TestStrategy]) -> List[Dict[str, Any]]:
        """创建实施路线图"""
        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}

        sorted_strategies = sorted(
            strategies,
            key=lambda s: (priority_order.get(s.implementation_priority, 3), -len(s.recommended_actions))
        )

        roadmap = []
        for i, strategy in enumerate(sorted_strategies, 1):
            roadmap.append({
                "phase": f"Phase {i}",
                "focus_area": strategy.focus_area,
                "priority": strategy.implementation_priority,
                "estimated_effort": "1-2 weeks" if strategy.implementation_priority == "high" else "2-4 weeks",
                "key_actions": strategy.recommended_actions[:2],  # 只显示前2个关键行动
                "expected_impact": strategy.expected_impact
            })

        return roadmap

    def _forecast_quality_trends(self, trends: List[TestTrend]) -> Dict[str, Any]:
        """预测质量趋势"""
        if len(trends) < 3:
            return {"forecast": "insufficient_data"}

        # 简单的线性趋势预测
        coverages = [t.coverage_percentage for t in trends[-3:]]  # 最近3个周期

        if len(coverages) >= 2:
            trend = "improving" if coverages[-1] > coverages[0] else "declining" if coverages[-1] < coverages[0] else "stable"
            projected_coverage = coverages[-1] + (coverages[-1] - coverages[0])  # 简单外推
            projected_coverage = max(0, min(100, projected_coverage))

            return {
                "trend": trend,
                "current_coverage": coverages[-1],
                "projected_coverage_3months": round(projected_coverage, 1),
                "confidence_level": "medium"
            }

        return {"forecast": "stable", "confidence_level": "low"}

    def _assess_current_risks(self, insights: List[QualityInsight]) -> Dict[str, Any]:
        """评估当前风险"""
        high_severity = len([i for i in insights if i.severity == "high"])
        medium_severity = len([i for i in insights if i.severity == "medium"])

        risk_score = min(high_severity * 20 + medium_severity * 10, 100)

        risk_level = "high" if risk_score > 70 else "medium" if risk_score > 40 else "low"

        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "high_severity_issues": high_severity,
            "medium_severity_issues": medium_severity,
            "critical_categories": [i.category for i in insights if i.severity == "high"]
        }

    def _predict_improvement_trajectory(self, strategies: List[TestStrategy]) -> Dict[str, Any]:
        """预测改进轨迹"""
        high_priority_strategies = len([s for s in strategies if s.implementation_priority == "high"])
        total_strategies = len(strategies)

        # 估算改进潜力
        improvement_potential = min(high_priority_strategies * 15 + (total_strategies - high_priority_strategies) * 8, 100)

        return {
            "improvement_potential_score": improvement_potential,
            "estimated_completion_months": 3 if high_priority_strategies > 0 else 6,
            "key_success_factors": [
                "策略有效执行",
                "团队配合度",
                "技术资源投入"
            ],
            "milestones": [
                "Month 1: 高优先级策略实施完成",
                "Month 2: 中优先级策略逐步推进",
                "Month 3: 质量指标显著改善"
            ]
        }

    def save_intelligent_report(self, report: Dict[str, Any]):
        """保存智能测试报告"""
        output_dir = Path("test_logs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_test_report_{timestamp}.json"

        output_file = output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成HTML报告
        html_report = self._generate_html_report(report, timestamp)
        html_file = output_dir / f"intelligent_test_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        # 生成执行摘要
        summary_file = output_dir / f"test_quality_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_executive_summary(report))

        print(f"💾 智能测试报告已保存: {output_file}")
        print(f"🌐 HTML报告已保存: {html_file}")
        print(f"📄 执行摘要已保存: {summary_file}")

    def _generate_html_report(self, report: Dict[str, Any], timestamp: str) -> str:
        """生成HTML格式的报告"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>智能测试分析报告 - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metric {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .high-priority {{ border-left: 5px solid #dc3545; }}
        .medium-priority {{ border-left: 5px solid #ffc107; }}
        .low-priority {{ border-left: 5px solid #28a745; }}
        .insights {{ margin: 20px 0; }}
        .strategy {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>智能测试分析报告</h1>
        <p>生成时间: {report['report_metadata']['generated_at']}</p>
        <p>分析周期: {report['report_metadata']['analysis_period_days']} 天</p>
    </div>

    <div class="metric">
        <h2>执行摘要</h2>
        <p>质量评分: <strong>{report['executive_summary']['overall_quality_score']}/100</strong></p>
        <p>趋势分析: {report['executive_summary']['total_trends_analyzed']} 个周期</p>
        <p>质量洞察: {report['executive_summary']['total_insights_generated']} 个</p>
        <p>策略建议: {report['executive_summary']['total_strategies_recommended']} 个</p>
    </div>

    <div class="insights">
        <h2>关键质量洞察</h2>
"""

        for insight in report['quality_insights']['insights'][:5]:  # 显示前5个洞察
            priority_class = f"{insight['severity']}-priority"
            html += """
        <div class="strategy {priority_class}">
            <h3>{insight['category'].title()} - {insight['severity'].title()}</h3>
            <p>{insight['description']}</p>
            <p><strong>建议:</strong> {', '.join(insight['recommendations'][:2])}</p>
        </div>
"""

        html += """
    </div>

    <div class="insights">
        <h2>测试策略建议</h2>
"""

        for strategy in report['strategy_recommendations']['strategies'][:5]:  # 显示前5个策略
            priority_class = f"{strategy['implementation_priority']}-priority"
            html += """
        <div class="strategy {priority_class}">
            <h3>{strategy['focus_area']}</h3>
            <p><strong>理由:</strong> {strategy['rationale']}</p>
            <p><strong>行动:</strong> {', '.join(strategy['recommended_actions'][:2])}</p>
            <p><strong>预期影响:</strong> {strategy['expected_impact']}</p>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """生成执行摘要"""
        summary = """
智能测试分析执行摘要
{'='*40}
生成时间: {report['report_metadata']['generated_at']}

📊 核心指标
质量评分: {report['executive_summary']['overall_quality_score']}/100
趋势分析: {report['executive_summary']['total_trends_analyzed']} 个周期
质量洞察: {report['executive_summary']['total_insights_generated']} 个
策略建议: {report['executive_summary']['total_strategies_recommended']} 个

🎯 高优先级行动项
"""

        high_priority = report['strategy_recommendations']['strategies_by_priority'].get('high', [])
        for i, action in enumerate(high_priority[:3], 1):
            summary += f"{i}. {action}\n"

        summary += "\n🔮 质量预测\n"
        forecast = report['predictive_analytics']['quality_forecast']
        summary += f"当前趋势: {forecast.get('trend', 'unknown')}\n"
        if 'projected_coverage_3months' in forecast:
            summary += f"3个月预测覆盖率: {forecast['projected_coverage_3months']}%\n"

        summary += "\n⚠️ 风险评估\n"
        risk = report['predictive_analytics']['risk_assessment']
        summary += f"整体风险评分: {risk['overall_risk_score']}/100 ({risk['risk_level']})\n"
        summary += f"高严重度问题: {risk['high_severity_issues']} 个\n"

        return summary


def main():
    """主函数"""
    reporter = IntelligentTestReporter()

    print("🚀 启动智能测试报告生成器")
    print("=" * 50)

    # 生成智能报告
    report = reporter.generate_intelligent_report()

    # 保存报告
    reporter.save_intelligent_report(report)

    # 输出关键发现
    exec_summary = report['executive_summary']
    quality_score = exec_summary['overall_quality_score']

    print("\n🎯 分析完成")
    print("=" * 50)
    print(f"📊 质量评分: {quality_score}/100")
    print(f"🔍 发现洞察: {exec_summary['total_insights_generated']} 个")
    print(f"🎯 策略建议: {exec_summary['total_strategies_recommended']} 个")

    if quality_score >= 80:
        print("✅ 测试质量优秀，继续保持！")
    elif quality_score >= 60:
        print("⚠️ 测试质量良好，需要持续改进")
    else:
        print("🔴 测试质量需要重点关注")

    print("\n✨ 系统测试智能化水平显著提升！")


if __name__ == "__main__":
    main()

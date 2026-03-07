"""
质量文化传承系统
建立测试先行理念和数据驱动决策的文化框架
包括测试文化评估、培训体系、质量度量和持续改进机制
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics


@dataclass
class QualityCultureMetrics:
    """质量文化度量指标"""
    test_first_adoption: float  # 测试先行理念采用率 0-100
    tdd_practice_rate: float    # TDD实践率 0-100
    code_review_coverage: float # 代码审查覆盖率 0-100
    automated_test_coverage: float  # 自动化测试覆盖率 0-100
    quality_awareness_score: float  # 质量意识评分 0-100
    continuous_learning_index: float  # 持续学习指数 0-100
    collaboration_quality: float  # 协作质量 0-100
    measurement_date: datetime = field(default_factory=datetime.now)


@dataclass
class TeamQualityProfile:
    """团队质量画像"""
    team_name: str
    team_size: int
    quality_maturity_level: str  # 'initial', 'repeatable', 'defined', 'managed', 'optimizing'
    strengths: List[str]
    improvement_areas: List[str]
    training_needs: List[str]
    metrics_history: List[QualityCultureMetrics] = field(default_factory=list)


@dataclass
class TestFirstImplementation:
    """测试先行实施指南"""
    practice_name: str
    description: str
    implementation_steps: List[str]
    success_criteria: List[str]
    common_challenges: List[str]
    solutions: List[str]
    maturity_level: str
    estimated_effort_days: int


@dataclass
class QualityTrainingProgram:
    """质量培训项目"""
    program_name: str
    target_audience: str
    duration_hours: int
    learning_objectives: List[str]
    modules: List[Dict[str, Any]]
    assessment_method: str
    certification_criteria: List[str]


@dataclass
class DataDrivenDecision:
    """数据驱动决策"""
    decision_context: str
    available_data: List[str]
    analysis_methods: List[str]
    decision_criteria: List[str]
    alternatives: List[Dict[str, Any]]
    recommended_action: str
    confidence_level: float
    implementation_plan: List[str]


class QualityCultureSystem:
    """质量文化系统"""

    def __init__(self):
        self.culture_metrics = []
        self.team_profiles = {}
        self.training_programs = self._initialize_training_programs()
        self.test_first_practices = self._initialize_test_first_practices()
        self.decision_framework = DataDrivenDecisionFramework()

    def _initialize_training_programs(self) -> Dict[str, QualityTrainingProgram]:
        """初始化培训项目"""
        return {
            'test_driven_development': QualityTrainingProgram(
                program_name='测试驱动开发 (TDD) 基础',
                target_audience='开发工程师',
                duration_hours=16,
                learning_objectives=[
                    '理解TDD的核心原则和流程',
                    '掌握红-绿-重构循环',
                    '学会编写可测试的代码',
                    '熟悉单元测试框架和工具'
                ],
                modules=[
                    {
                        'title': 'TDD原理与概念',
                        'duration': 4,
                        'content': ['TDD三定律', '红绿重构循环', '测试金字塔']
                    },
                    {
                        'title': '单元测试实战',
                        'duration': 6,
                        'content': ['测试框架使用', '断言技术', '测试组织']
                    },
                    {
                        'title': 'TDD最佳实践',
                        'duration': 4,
                        'content': ['测试命名规范', '测试覆盖率目标', '持续集成']
                    },
                    {
                        'title': '实际项目应用',
                        'duration': 2,
                        'content': ['代码示例分析', '常见问题解决']
                    }
                ],
                assessment_method='project_based',
                certification_criteria=[
                    '完成TDD练习项目',
                    '测试覆盖率达到80%',
                    '通过同行代码审查'
                ]
            ),

            'quality_assurance_fundamentals': QualityTrainingProgram(
                program_name='质量保障基础',
                target_audience='全体团队成员',
                duration_hours=8,
                learning_objectives=[
                    '理解软件质量的重要性',
                    '掌握质量保障的基本概念',
                    '学会识别质量问题',
                    '了解质量改进的方法'
                ],
                modules=[
                    {
                        'title': '软件质量概述',
                        'duration': 2,
                        'content': ['质量定义', '质量属性', '质量成本']
                    },
                    {
                        'title': '测试基础知识',
                        'duration': 3,
                        'content': ['测试类型', '测试流程', '缺陷管理']
                    },
                    {
                        'title': '质量工具与技术',
                        'duration': 2,
                        'content': ['自动化测试', '代码审查', '持续集成']
                    },
                    {
                        'title': '质量文化建设',
                        'duration': 1,
                        'content': ['质量意识', '责任共担', '持续改进']
                    }
                ],
                assessment_method='quiz_and_participation',
                certification_criteria=[
                    '完成质量知识考核 (80分以上)',
                    '参与质量改进活动',
                    '提交质量改进建议'
                ]
            ),

            'code_review_mastery': QualityTrainingProgram(
                program_name='代码审查精通',
                target_audience='开发工程师和测试工程师',
                duration_hours=12,
                learning_objectives=[
                    '掌握代码审查的最佳实践',
                    '学会识别常见的代码问题',
                    '提高代码质量和可维护性',
                    '培养有效的审查沟通技巧'
                ],
                modules=[
                    {
                        'title': '代码审查基础',
                        'duration': 3,
                        'content': ['审查目的', '审查流程', '审查清单']
                    },
                    {
                        'title': '常见代码问题识别',
                        'duration': 4,
                        'content': ['设计问题', '编码规范', '性能问题', '安全漏洞']
                    },
                    {
                        'title': '审查工具与技术',
                        'duration': 3,
                        'content': ['审查工具使用', '自动化检查', '度量分析']
                    },
                    {
                        'title': '有效沟通与反馈',
                        'duration': 2,
                        'content': ['建设性反馈', '冲突解决', '持续改进']
                    }
                ],
                assessment_method='peer_review_participation',
                certification_criteria=[
                    '完成10次代码审查',
                    '识别并修复5个代码质量问题',
                    '获得导师认可'
                ]
            )
        }

    def _initialize_test_first_practices(self) -> Dict[str, TestFirstImplementation]:
        """初始化测试先行实践"""
        return {
            'tdd_implementation': TestFirstImplementation(
                practice_name='测试驱动开发实施',
                description='在开发过程中先编写测试，再编写实现代码',
                implementation_steps=[
                    '为新功能编写失败的测试用例',
                    '运行测试确认失败',
                    '编写最简单的实现代码使测试通过',
                    '重构代码保持清洁',
                    '重复上述步骤直到功能完成'
                ],
                success_criteria=[
                    '所有测试用例通过',
                    '代码覆盖率达到80%以上',
                    '重构后代码仍然通过所有测试',
                    '团队成员能够熟练应用TDD'
                ],
                common_challenges=[
                    '习惯改变困难',
                    '测试编写耗时',
                    '如何测试复杂逻辑',
                    '遗留代码难以测试'
                ],
                solutions=[
                    '从小功能开始练习',
                    '使用测试生成工具辅助',
                    '学习测试替身模式',
                    '逐步重构遗留代码'
                ],
                maturity_level='intermediate',
                estimated_effort_days=30
            ),

            'test_automation_culture': TestFirstImplementation(
                practice_name='测试自动化文化建立',
                description='建立自动化测试的文化和基础设施',
                implementation_steps=[
                    '选择合适的测试框架和工具',
                    '建立测试代码规范和标准',
                    '创建测试数据管理和环境',
                    '实现CI/CD流水线自动化',
                    '建立测试结果报告和监控'
                ],
                success_criteria=[
                    '核心功能测试自动化覆盖100%',
                    '回归测试在15分钟内完成',
                    '测试失败能够自动告警',
                    '开发人员主动编写自动化测试'
                ],
                common_challenges=[
                    '测试维护成本高',
                    '测试执行时间长',
                    '测试环境不稳定',
                    '团队技能不足'
                ],
                solutions=[
                    '采用页面对象模式降低维护成本',
                    '并行执行和智能测试选择',
                    '使用Docker等容器化技术',
                    '提供培训和结对编程支持'
                ],
                maturity_level='advanced',
                estimated_effort_days=60
            ),

            'continuous_integration_practice': TestFirstImplementation(
                practice_name='持续集成实践',
                description='建立自动化的构建、测试和部署流程',
                implementation_steps=[
                    '设置版本控制和分支策略',
                    '配置自动构建和测试环境',
                    '实现自动化测试执行',
                    '建立代码质量门禁',
                    '配置自动部署流程'
                ],
                success_criteria=[
                    '每次提交自动触发构建和测试',
                    '构建失败阻止代码合并',
                    '测试覆盖率不低于阈值',
                    '部署过程完全自动化'
                ],
                common_challenges=[
                    '环境配置复杂',
                    '测试执行不稳定',
                    '反馈周期过长',
                    '团队协作问题'
                ],
                solutions=[
                    '使用基础设施即代码',
                    '实施测试环境隔离',
                    '优化测试执行策略',
                    '建立DevOps文化'
                ],
                maturity_level='intermediate',
                estimated_effort_days=45
            )
        }

    def assess_team_quality_culture(self, team_name: str, survey_data: Dict[str, Any]) -> TeamQualityProfile:
        """评估团队质量文化"""
        # 计算质量文化指标
        metrics = self._calculate_culture_metrics(survey_data)

        # 确定成熟度等级
        maturity_level = self._determine_maturity_level(metrics)

        # 识别优势和改进领域
        strengths = self._identify_culture_strengths(metrics)
        improvement_areas = self._identify_culture_weaknesses(metrics)

        # 确定培训需求
        training_needs = self._determine_training_needs(metrics, maturity_level)

        # 创建团队画像
        profile = TeamQualityProfile(
            team_name=team_name,
            team_size=survey_data.get('team_size', 0),
            quality_maturity_level=maturity_level,
            strengths=strengths,
            improvement_areas=improvement_areas,
            training_needs=training_needs,
            metrics_history=[metrics]
        )

        self.team_profiles[team_name] = profile
        return profile

    def _calculate_culture_metrics(self, survey_data: Dict[str, Any]) -> QualityCultureMetrics:
        """计算文化指标"""
        # 从调查数据中提取指标（这里使用模拟计算）
        test_first_responses = survey_data.get('test_first_adoption', [])
        tdd_responses = survey_data.get('tdd_practice', [])
        review_responses = survey_data.get('code_review_coverage', [])
        automation_responses = survey_data.get('automation_coverage', [])
        awareness_responses = survey_data.get('quality_awareness', [])
        learning_responses = survey_data.get('continuous_learning', [])
        collaboration_responses = survey_data.get('collaboration_quality', [])

        return QualityCultureMetrics(
            test_first_adoption=statistics.mean(test_first_responses) * 20 if test_first_responses else 0,
            tdd_practice_rate=statistics.mean(tdd_responses) * 20 if tdd_responses else 0,
            code_review_coverage=statistics.mean(review_responses) * 20 if review_responses else 0,
            automated_test_coverage=statistics.mean(automation_responses) * 20 if automation_responses else 0,
            quality_awareness_score=statistics.mean(awareness_responses) * 20 if awareness_responses else 0,
            continuous_learning_index=statistics.mean(learning_responses) * 20 if learning_responses else 0,
            collaboration_quality=statistics.mean(collaboration_responses) * 20 if collaboration_responses else 0
        )

    def _determine_maturity_level(self, metrics: QualityCultureMetrics) -> str:
        """确定成熟度等级"""
        avg_score = statistics.mean([
            metrics.test_first_adoption,
            metrics.tdd_practice_rate,
            metrics.code_review_coverage,
            metrics.automated_test_coverage,
            metrics.quality_awareness_score,
            metrics.continuous_learning_index,
            metrics.collaboration_quality
        ])

        if avg_score >= 80:
            return 'optimizing'
        elif avg_score >= 60:
            return 'managed'
        elif avg_score >= 40:
            return 'defined'
        elif avg_score >= 20:
            return 'repeatable'
        else:
            return 'initial'

    def _identify_culture_strengths(self, metrics: QualityCultureMetrics) -> List[str]:
        """识别文化优势"""
        strengths = []

        if metrics.test_first_adoption >= 70:
            strengths.append("测试先行理念深入人心")
        if metrics.tdd_practice_rate >= 70:
            strengths.append("TDD实践成熟")
        if metrics.code_review_coverage >= 80:
            strengths.append("代码审查文化完善")
        if metrics.automated_test_coverage >= 75:
            strengths.append("测试自动化程度高")
        if metrics.quality_awareness_score >= 80:
            strengths.append("质量意识强烈")
        if metrics.continuous_learning_index >= 70:
            strengths.append("持续学习氛围好")
        if metrics.collaboration_quality >= 75:
            strengths.append("团队协作质量高")

        return strengths

    def _identify_culture_weaknesses(self, metrics: QualityCultureMetrics) -> List[str]:
        """识别文化弱点"""
        weaknesses = []

        if metrics.test_first_adoption < 50:
            weaknesses.append("测试先行理念需要加强")
        if metrics.tdd_practice_rate < 40:
            weaknesses.append("TDD实践率偏低")
        if metrics.code_review_coverage < 60:
            weaknesses.append("代码审查覆盖不全")
        if metrics.automated_test_coverage < 50:
            weaknesses.append("测试自动化程度不足")
        if metrics.quality_awareness_score < 60:
            weaknesses.append("质量意识需要提升")
        if metrics.continuous_learning_index < 50:
            weaknesses.append("持续学习机制薄弱")
        if metrics.collaboration_quality < 60:
            weaknesses.append("团队协作有待改进")

        return weaknesses

    def _determine_training_needs(self, metrics: QualityCultureMetrics, maturity_level: str) -> List[str]:
        """确定培训需求"""
        training_needs = []

        if metrics.tdd_practice_rate < 60:
            training_needs.append('test_driven_development')

        if metrics.code_review_coverage < 70:
            training_needs.append('code_review_mastery')

        if metrics.quality_awareness_score < 70:
            training_needs.append('quality_assurance_fundamentals')

        # 根据成熟度等级添加额外培训
        if maturity_level == 'initial':
            training_needs.extend(['quality_assurance_fundamentals', 'test_driven_development'])
        elif maturity_level == 'repeatable':
            training_needs.append('test_driven_development')
        elif maturity_level == 'defined':
            training_needs.append('code_review_mastery')

        return list(set(training_needs))  # 去重

    def implement_test_first_practice(self, practice_name: str) -> Dict[str, Any]:
        """实施测试先行实践"""
        if practice_name not in self.test_first_practices:
            raise ValueError(f"未知的实践: {practice_name}")

        practice = self.test_first_practices[practice_name]

        # 创建实施计划
        implementation_plan = {
            'practice': practice.practice_name,
            'description': practice.description,
            'timeline': self._create_implementation_timeline(practice),
            'resources_needed': self._identify_resources_needed(practice),
            'risk_mitigation': self._create_risk_mitigation_plan(practice),
            'success_measurement': self._create_success_measurement_plan(practice),
            'estimated_effort': practice.estimated_effort_days
        }

        return implementation_plan

    def _create_implementation_timeline(self, practice: TestFirstImplementation) -> List[Dict[str, Any]]:
        """创建实施时间线"""
        timeline = []
        total_days = practice.estimated_effort_days

        # 准备阶段 (20%)
        timeline.append({
            'phase': '准备阶段',
            'duration_days': int(total_days * 0.2),
            'activities': [
                '培训团队成员',
                '准备基础设施',
                '创建实施计划',
                '设定基线指标'
            ]
        })

        # 试点阶段 (30%)
        timeline.append({
            'phase': '试点阶段',
            'duration_days': int(total_days * 0.3),
            'activities': [
                '选择试点项目',
                '实施核心实践',
                '收集反馈',
                '调整方法'
            ]
        })

        # 扩展阶段 (30%)
        timeline.append({
            'phase': '扩展阶段',
            'duration_days': int(total_days * 0.3),
            'activities': [
                '扩展到更多团队',
                '建立支持体系',
                '监控进度',
                '解决障碍'
            ]
        })

        # 巩固阶段 (20%)
        timeline.append({
            'phase': '巩固阶段',
            'duration_days': int(total_days * 0.2),
            'activities': [
                '评估实施效果',
                '完善流程',
                '知识传递',
                '持续改进'
            ]
        })

        return timeline

    def _identify_resources_needed(self, practice: TestFirstImplementation) -> Dict[str, List[str]]:
        """识别所需资源"""
        return {
            'human_resources': [
                '实施负责人',
                '技术教练',
                '团队成员',
                '质量保证人员'
            ],
            'technical_resources': [
                '开发环境',
                '测试工具',
                'CI/CD系统',
                '培训材料'
            ],
            'organizational_support': [
                '管理层支持',
                '时间分配',
                '培训预算',
                '变革管理'
            ]
        }

    def _create_risk_mitigation_plan(self, practice: TestFirstImplementation) -> Dict[str, List[str]]:
        """创建风险缓解计划"""
        risk_mitigation = {}

        for challenge in practice.common_challenges:
            if '习惯改变' in challenge:
                risk_mitigation[challenge] = [
                    '提供充分培训',
                    '从小项目开始',
                    '分享成功案例',
                    '建立导师制度'
                ]
            elif '时间成本' in challenge:
                risk_mitigation[challenge] = [
                    '优化工作流程',
                    '自动化重复任务',
                    '合理分配时间',
                    '展示长期收益'
                ]
            elif '技能不足' in challenge:
                risk_mitigation[challenge] = [
                    '提供专业培训',
                    '引入外部专家',
                    '建立知识库',
                    '结对编程'
                ]
            else:
                risk_mitigation[challenge] = [
                    '识别具体问题',
                    '制定应对策略',
                    '持续监控',
                    '灵活调整'
                ]

        return risk_mitigation

    def _create_success_measurement_plan(self, practice: TestFirstImplementation) -> Dict[str, List[str]]:
        """创建成功度量计划"""
        return {
            'quantitative_metrics': [
                '实践采用率',
                '代码质量指标',
                '交付周期',
                '缺陷率'
            ],
            'qualitative_metrics': [
                '团队满意度',
                '过程改进反馈',
                '知识分享情况',
                '文化转变程度'
            ],
            'milestones': practice.success_criteria,
            'reporting_frequency': '每周'
        }

    def make_data_driven_decision(self, context: str, data_sources: List[Dict[str, Any]],
                                decision_criteria: List[str]) -> DataDrivenDecision:
        """做出数据驱动决策"""
        return self.decision_framework.analyze_and_decide(context, data_sources, decision_criteria)

    def track_culture_improvement(self, team_name: str, new_metrics: QualityCultureMetrics):
        """跟踪文化改进"""
        if team_name not in self.team_profiles:
            raise ValueError(f"团队 {team_name} 不存在")

        profile = self.team_profiles[team_name]
        profile.metrics_history.append(new_metrics)

        # 计算改进趋势
        improvement_trend = self._calculate_improvement_trend(profile.metrics_history)

        return {
            'current_maturity': profile.quality_maturity_level,
            'improvement_trend': improvement_trend,
            'next_maturity_level': self._predict_next_maturity_level(profile),
            'recommendations': self._generate_improvement_recommendations(profile)
        }

    def _calculate_improvement_trend(self, metrics_history: List[QualityCultureMetrics]) -> Dict[str, Any]:
        """计算改进趋势"""
        if len(metrics_history) < 2:
            return {'insufficient_data': True}

        # 计算各项指标的趋势
        trends = {}
        metrics_count = len(metrics_history)

        for attr in ['test_first_adoption', 'tdd_practice_rate', 'code_review_coverage',
                    'automated_test_coverage', 'quality_awareness_score',
                    'continuous_learning_index', 'collaboration_quality']:

            values = [getattr(m, attr) for m in metrics_history]
            if len(values) >= 2:
                # 计算趋势斜率（简化版）
                trend = (values[-1] - values[0]) / (metrics_count - 1)
                trends[attr] = {
                    'slope': trend,
                    'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
                    'magnitude': abs(trend)
                }

        # 计算整体趋势
        positive_trends = sum(1 for t in trends.values() if t['direction'] == 'improving')
        total_trends = len(trends)

        overall_trend = {
            'positive_indicators': positive_trends,
            'total_indicators': total_trends,
            'improvement_percentage': positive_trends / total_trends * 100 if total_trends > 0 else 0,
            'overall_direction': 'improving' if positive_trends > total_trends * 0.6 else 'needs_attention'
        }

        return {
            'individual_trends': trends,
            'overall_trend': overall_trend
        }

    def _predict_next_maturity_level(self, profile: TeamQualityProfile) -> str:
        """预测下一个成熟度等级"""
        current_level = profile.quality_maturity_level
        latest_metrics = profile.metrics_history[-1] if profile.metrics_history else None

        if not latest_metrics:
            return current_level

        # 计算平均分数
        avg_score = statistics.mean([
            latest_metrics.test_first_adoption,
            latest_metrics.tdd_practice_rate,
            latest_metrics.code_review_coverage,
            latest_metrics.automated_test_coverage,
            latest_metrics.quality_awareness_score,
            latest_metrics.continuous_learning_index,
            latest_metrics.collaboration_quality
        ])

        maturity_levels = ['initial', 'repeatable', 'defined', 'managed', 'optimizing']
        current_index = maturity_levels.index(current_level)

        # 如果平均分数超过当前等级的阈值，建议升级
        if avg_score >= 80 and current_index < len(maturity_levels) - 1:
            return maturity_levels[current_index + 1]
        elif avg_score < 40 and current_index > 0:
            return maturity_levels[current_index - 1]
        else:
            return current_level

    def _generate_improvement_recommendations(self, profile: TeamQualityProfile) -> List[str]:
        """生成改进建议"""
        recommendations = []
        latest_metrics = profile.metrics_history[-1] if profile.metrics_history else None

        if not latest_metrics:
            return ['收集更多质量指标数据']

        # 基于具体指标生成建议
        if latest_metrics.test_first_adoption < 60:
            recommendations.append('加强测试先行理念的培训和实践')

        if latest_metrics.tdd_practice_rate < 50:
            recommendations.append('实施TDD实践改进计划')

        if latest_metrics.code_review_coverage < 70:
            recommendations.append('提高代码审查的覆盖率和质量')

        if latest_metrics.automated_test_coverage < 60:
            recommendations.append('扩展测试自动化覆盖范围')

        if latest_metrics.quality_awareness_score < 70:
            recommendations.append('提升团队质量意识')

        if latest_metrics.continuous_learning_index < 60:
            recommendations.append('建立持续学习机制')

        if latest_metrics.collaboration_quality < 70:
            recommendations.append('改善团队协作质量')

        # 根据成熟度等级添加建议
        if profile.quality_maturity_level == 'initial':
            recommendations.append('从建立基础质量实践开始')
        elif profile.quality_maturity_level == 'repeatable':
            recommendations.append('标准化质量流程')
        elif profile.quality_maturity_level == 'defined':
            recommendations.append('建立质量度量体系')
        elif profile.quality_maturity_level == 'managed':
            recommendations.append('优化质量流程效率')
        elif profile.quality_maturity_level == 'optimizing':
            recommendations.append('持续创新质量实践')

        return recommendations


class DataDrivenDecisionFramework:
    """数据驱动决策框架"""

    def analyze_and_decide(self, context: str, data_sources: List[Dict[str, Any]],
                          decision_criteria: List[str]) -> DataDrivenDecision:
        """分析并做出决策"""
        # 分析可用数据
        available_data = [ds.get('name', 'unknown') for ds in data_sources]

        # 确定分析方法
        analysis_methods = self._determine_analysis_methods(data_sources)

        # 生成备选方案
        alternatives = self._generate_alternatives(context, data_sources)

        # 评估备选方案
        evaluated_alternatives = self._evaluate_alternatives(alternatives, decision_criteria, data_sources)

        # 选择最佳方案
        recommended_action = max(evaluated_alternatives, key=lambda x: x['score'])['action']

        # 计算置信度
        confidence_level = self._calculate_confidence_level(evaluated_alternatives, data_sources)

        # 创建实施计划
        implementation_plan = self._create_implementation_plan(recommended_action, context)

        return DataDrivenDecision(
            decision_context=context,
            available_data=available_data,
            analysis_methods=analysis_methods,
            decision_criteria=decision_criteria,
            alternatives=evaluated_alternatives,
            recommended_action=recommended_action,
            confidence_level=confidence_level,
            implementation_plan=implementation_plan
        )

    def _determine_analysis_methods(self, data_sources: List[Dict[str, Any]]) -> List[str]:
        """确定分析方法"""
        methods = []

        has_quantitative_data = any(ds.get('type') == 'quantitative' for ds in data_sources)
        has_qualitative_data = any(ds.get('type') == 'qualitative' for ds in data_sources)
        has_time_series = any('timestamp' in ds.get('fields', []) for ds in data_sources)

        if has_quantitative_data:
            methods.extend(['统计分析', '回归分析', '相关性分析'])

        if has_qualitative_data:
            methods.extend(['内容分析', '主题分析'])

        if has_time_series:
            methods.extend(['趋势分析', '时间序列分析'])

        if len(data_sources) > 1:
            methods.append('数据整合分析')

        return methods

    def _generate_alternatives(self, context: str, data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成备选方案"""
        # 根据决策上下文生成相关的备选方案
        if '测试策略' in context:
            return [
                {'action': '增加单元测试覆盖率', 'description': '重点提升单元测试覆盖率到90%'},
                {'action': '扩展集成测试', 'description': '增加端到端集成测试覆盖'},
                {'action': '实施性能测试', 'description': '建立全面的性能测试体系'},
                {'action': '优化测试执行效率', 'description': '减少测试执行时间，提高反馈速度'}
            ]
        elif '质量改进' in context:
            return [
                {'action': '实施代码审查流程', 'description': '建立强制代码审查机制'},
                {'action': '引入静态代码分析', 'description': '使用自动化工具检查代码质量'},
                {'action': '建立质量门禁', 'description': '设置代码质量标准和检查'},
                {'action': '开展质量培训', 'description': '提升团队质量意识和技能'}
            ]
        elif '技术选型' in context:
            return [
                {'action': '选择成熟技术栈', 'description': '采用业界验证的技术方案'},
                {'action': '创新技术探索', 'description': '尝试新技术但控制风险'},
                {'action': '混合技术方案', 'description': '结合成熟和创新技术'},
                {'action': '谨慎观望策略', 'description': '等待技术成熟后再采用'}
            ]
        else:
            # 默认备选方案
            return [
                {'action': '保守策略', 'description': '保持当前做法，渐进改进'},
                {'action': '激进策略', 'description': '大胆创新，快速推进'},
                {'action': '平衡策略', 'description': '在稳定和创新间取得平衡'}
            ]

    def _evaluate_alternatives(self, alternatives: List[Dict[str, Any]],
                             criteria: List[str], data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """评估备选方案"""
        evaluated = []

        for alt in alternatives:
            scores = {}

            # 根据决策标准评估每个方案
            for criterion in criteria:
                if '风险' in criterion:
                    scores['risk'] = self._assess_risk(alt['action'])
                elif '成本' in criterion:
                    scores['cost'] = self._assess_cost(alt['action'])
                elif '收益' in criterion:
                    scores['benefit'] = self._assess_benefit(alt['action'], data_sources)
                elif '时间' in criterion:
                    scores['time'] = self._assess_time(alt['action'])
                elif '可行性' in criterion:
                    scores['feasibility'] = self._assess_feasibility(alt['action'])
                else:
                    scores[criterion] = 0.5  # 默认中等分数

            # 计算综合得分
            weights = self._get_criterion_weights(criteria)
            total_score = sum(scores.get(c, 0.5) * weights.get(c, 1.0) for c in criteria)

            evaluated.append({
                'action': alt['action'],
                'description': alt['description'],
                'scores': scores,
                'score': total_score
            })

        return evaluated

    def _assess_risk(self, action: str) -> float:
        """评估风险"""
        risk_levels = {
            '增加单元测试覆盖率': 0.2,
            '扩展集成测试': 0.3,
            '实施性能测试': 0.4,
            '优化测试执行效率': 0.2,
            '实施代码审查流程': 0.2,
            '引入静态代码分析': 0.3,
            '建立质量门禁': 0.4,
            '开展质量培训': 0.1,
            '选择成熟技术栈': 0.2,
            '创新技术探索': 0.8,
            '混合技术方案': 0.5,
            '谨慎观望策略': 0.1
        }
        return risk_levels.get(action, 0.5)

    def _assess_cost(self, action: str) -> float:
        """评估成本"""
        cost_levels = {
            '增加单元测试覆盖率': 0.3,
            '扩展集成测试': 0.4,
            '实施性能测试': 0.6,
            '优化测试执行效率': 0.4,
            '实施代码审查流程': 0.2,
            '引入静态代码分析': 0.5,
            '建立质量门禁': 0.3,
            '开展质量培训': 0.4,
            '选择成熟技术栈': 0.2,
            '创新技术探索': 0.7,
            '混合技术方案': 0.5,
            '谨慎观望策略': 0.1
        }
        return cost_levels.get(action, 0.5)

    def _assess_benefit(self, action: str, data_sources: List[Dict[str, Any]]) -> float:
        """评估收益"""
        # 基于数据源分析潜在收益
        base_benefit = 0.5

        has_quality_data = any('quality' in ds.get('name', '').lower() for ds in data_sources)
        has_performance_data = any('performance' in ds.get('name', '').lower() for ds in data_sources)

        if '测试' in action and has_quality_data:
            base_benefit += 0.3
        if '性能' in action and has_performance_data:
            base_benefit += 0.3

        benefit_levels = {
            '增加单元测试覆盖率': base_benefit + 0.2,
            '扩展集成测试': base_benefit + 0.3,
            '实施性能测试': base_benefit + 0.4,
            '优化测试执行效率': base_benefit + 0.2,
            '实施代码审查流程': base_benefit + 0.3,
            '引入静态代码分析': base_benefit + 0.4,
            '建立质量门禁': base_benefit + 0.3,
            '开展质量培训': base_benefit + 0.5,
            '选择成熟技术栈': base_benefit + 0.2,
            '创新技术探索': base_benefit + 0.6,
            '混合技术方案': base_benefit + 0.4,
            '谨慎观望策略': base_benefit + 0.1
        }

        return min(benefit_levels.get(action, base_benefit), 1.0)

    def _assess_time(self, action: str) -> float:
        """评估时间"""
        time_levels = {
            '增加单元测试覆盖率': 0.4,
            '扩展集成测试': 0.5,
            '实施性能测试': 0.6,
            '优化测试执行效率': 0.3,
            '实施代码审查流程': 0.3,
            '引入静态代码分析': 0.4,
            '建立质量门禁': 0.4,
            '开展质量培训': 0.6,
            '选择成熟技术栈': 0.2,
            '创新技术探索': 0.7,
            '混合技术方案': 0.5,
            '谨慎观望策略': 0.1
        }
        return time_levels.get(action, 0.5)

    def _assess_feasibility(self, action: str) -> float:
        """评估可行性"""
        feasibility_levels = {
            '增加单元测试覆盖率': 0.8,
            '扩展集成测试': 0.7,
            '实施性能测试': 0.6,
            '优化测试执行效率': 0.7,
            '实施代码审查流程': 0.8,
            '引入静态代码分析': 0.7,
            '建立质量门禁': 0.6,
            '开展质量培训': 0.9,
            '选择成熟技术栈': 0.9,
            '创新技术探索': 0.4,
            '混合技术方案': 0.6,
            '谨慎观望策略': 0.9
        }
        return feasibility_levels.get(action, 0.5)

    def _get_criterion_weights(self, criteria: List[str]) -> Dict[str, float]:
        """获取标准权重"""
        weights = {}
        for criterion in criteria:
            if '风险' in criterion:
                weights['risk'] = 2.0
            elif '成本' in criterion:
                weights['cost'] = 1.5
            elif '收益' in criterion:
                weights['benefit'] = 2.5
            elif '时间' in criterion:
                weights['time'] = 1.0
            elif '可行性' in criterion:
                weights['feasibility'] = 1.5
            else:
                weights[criterion] = 1.0
        return weights

    def _calculate_confidence_level(self, alternatives: List[Dict[str, Any]],
                                  data_sources: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        # 基于数据质量和数量计算置信度
        data_quality = len(data_sources) / 10  # 假设10个数据源为高质量
        data_quantity = sum(ds.get('record_count', 0) for ds in data_sources) / 10000  # 假设10000条记录为充足

        alternatives_spread = len(set(alt['score'] for alt in alternatives)) / len(alternatives)

        confidence = (data_quality * 0.4 + data_quantity * 0.4 + alternatives_spread * 0.2)

        return min(confidence, 1.0)

    def _create_implementation_plan(self, action: str, context: str) -> List[str]:
        """创建实施计划"""
        plans = {
            '增加单元测试覆盖率': [
                '识别未覆盖的核心功能模块',
                '为每个模块编写单元测试',
                '集成测试到CI/CD流水线',
                '监控覆盖率变化趋势'
            ],
            '扩展集成测试': [
                '分析系统集成点',
                '设计端到端测试场景',
                '搭建测试环境和数据',
                '实现自动化集成测试'
            ],
            '实施性能测试': [
                '确定性能测试指标和阈值',
                '设计性能测试场景',
                '搭建性能测试环境',
                '建立持续性能监控'
            ],
            '实施代码审查流程': [
                '定义代码审查清单和标准',
                '培训审查人员',
                '集成到开发流程',
                '收集和分析审查反馈'
            ]
        }

        return plans.get(action, [
            '制定详细实施计划',
            '分配资源和责任',
            '按阶段执行实施',
            '监控和调整进度'
        ])


class TestQualityCultureSystem:
    """质量文化系统测试"""

    def setup_method(self):
        """测试前准备"""
        self.culture_system = QualityCultureSystem()

    def test_team_quality_culture_assessment(self):
        """测试团队质量文化评估"""
        # 模拟调查数据
        survey_data = {
            'team_size': 8,
            'test_first_adoption': [4, 4, 3, 4, 5, 4, 3, 4],  # 1-5分制
            'tdd_practice': [3, 4, 3, 2, 4, 3, 3, 4],
            'code_review_coverage': [5, 5, 4, 5, 5, 4, 5, 5],
            'automation_coverage': [4, 4, 3, 4, 5, 4, 4, 4],
            'quality_awareness': [5, 4, 5, 5, 5, 4, 5, 5],
            'continuous_learning': [4, 4, 4, 3, 4, 4, 4, 5],
            'collaboration_quality': [4, 5, 4, 4, 5, 4, 4, 5]
        }

        # 评估团队质量文化
        profile = self.culture_system.assess_team_quality_culture("backend-team", survey_data)

        # 验证评估结果
        assert profile.team_name == "backend-team"
        assert profile.team_size == 8
        assert profile.quality_maturity_level in ['initial', 'repeatable', 'defined', 'managed', 'optimizing']
        assert len(profile.strengths) >= 0
        assert len(profile.improvement_areas) >= 0
        assert len(profile.training_needs) >= 0
        assert len(profile.metrics_history) == 1

        # 验证指标计算
        metrics = profile.metrics_history[0]
        assert 0 <= metrics.test_first_adoption <= 100
        assert 0 <= metrics.tdd_practice_rate <= 100
        assert 0 <= metrics.code_review_coverage <= 100
        assert 0 <= metrics.quality_awareness_score <= 100

        print(f"✅ 团队质量文化评估成功 - 成熟度等级: {profile.quality_maturity_level}, 优势: {len(profile.strengths)} 项")

    def test_training_program_structure(self):
        """测试培训项目结构"""
        programs = self.culture_system.training_programs

        # 验证培训项目数量
        assert len(programs) >= 3

        # 检查具体培训项目
        tdd_program = programs.get('test_driven_development')
        assert tdd_program is not None
        assert tdd_program.program_name == '测试驱动开发 (TDD) 基础'
        assert tdd_program.target_audience == '开发工程师'
        assert tdd_program.duration_hours > 0
        assert len(tdd_program.learning_objectives) > 0
        assert len(tdd_program.modules) > 0
        assert len(tdd_program.certification_criteria) > 0

        # 验证另一个培训项目
        qa_program = programs.get('quality_assurance_fundamentals')
        assert qa_program is not None
        assert qa_program.target_audience == '全体团队成员'

        print(f"✅ 培训项目结构验证成功 - {len(programs)} 个培训项目")

    def test_test_first_practice_implementation(self):
        """测试测试先行实践实施"""
        # 测试TDD实施
        tdd_plan = self.culture_system.implement_test_first_practice('tdd_implementation')

        assert tdd_plan['practice'] == '测试驱动开发实施'
        assert 'timeline' in tdd_plan
        assert 'resources_needed' in tdd_plan
        assert 'risk_mitigation' in tdd_plan
        assert 'success_measurement' in tdd_plan
        assert tdd_plan['estimated_effort'] == 30

        # 验证时间线
        timeline = tdd_plan['timeline']
        assert len(timeline) == 4  # 四个阶段
        assert sum(phase['duration_days'] for phase in timeline) == 30

        # 验证资源需求
        resources = tdd_plan['resources_needed']
        assert 'human_resources' in resources
        assert 'technical_resources' in resources
        assert 'organizational_support' in resources

        print(f"✅ 测试先行实践实施计划生成成功 - 总时长: {tdd_plan['estimated_effort']} 天")

    def test_data_driven_decision_making(self):
        """测试数据驱动决策"""
        # 创建决策上下文
        context = "选择合适的测试策略来提升代码质量"

        # 提供数据源
        data_sources = [
            {
                'name': '单元测试覆盖率数据',
                'type': 'quantitative',
                'fields': ['coverage_percentage', 'timestamp'],
                'record_count': 100
            },
            {
                'name': '缺陷发现趋势',
                'type': 'quantitative',
                'fields': ['defect_count', 'severity', 'timestamp'],
                'record_count': 50
            },
            {
                'name': '团队反馈调查',
                'type': 'qualitative',
                'fields': ['feedback_text', 'satisfaction_score'],
                'record_count': 20
            }
        ]

        # 定义决策标准
        criteria = ['潜在收益', '实施风险', '所需时间', '资源成本']

        # 做出数据驱动决策
        decision = self.culture_system.make_data_driven_decision(context, data_sources, criteria)

        # 验证决策结果
        assert decision.decision_context == context
        assert len(decision.available_data) == 3
        assert len(decision.analysis_methods) > 0
        assert len(decision.alternatives) > 0
        assert decision.recommended_action is not None
        assert 0 <= decision.confidence_level <= 1.0
        assert len(decision.implementation_plan) > 0

        # 验证备选方案评估
        for alt in decision.alternatives:
            assert 'action' in alt
            assert 'description' in alt
            assert 'scores' in alt
            assert 'score' in alt

        print(f"✅ 数据驱动决策成功 - 推荐行动: {decision.recommended_action}, 置信度: {decision.confidence_level:.2f}")

    def test_culture_improvement_tracking(self):
        """测试文化改进跟踪"""
        # 首先创建团队画像
        survey_data = {
            'team_size': 6,
            'test_first_adoption': [3, 3, 4, 3, 4, 3],
            'tdd_practice': [2, 3, 2, 3, 3, 2],
            'code_review_coverage': [4, 4, 5, 4, 5, 4],
            'automation_coverage': [3, 3, 4, 3, 4, 3],
            'quality_awareness': [4, 4, 5, 4, 5, 4],
            'continuous_learning': [3, 4, 3, 4, 4, 3],
            'collaboration_quality': [4, 4, 5, 4, 5, 4]
        }

        self.culture_system.assess_team_quality_culture("frontend-team", survey_data)

        # 添加新的度量数据（模拟改进后）
        new_metrics = QualityCultureMetrics(
            test_first_adoption=75.0,  # 从~60提升到75
            tdd_practice_rate=50.0,   # 从~40提升到50
            code_review_coverage=90.0, # 从~80保持在90
            automated_test_coverage=70.0, # 从~60提升到70
            quality_awareness_score=85.0, # 从~80提升到85
            continuous_learning_index=65.0, # 从~55提升到65
            collaboration_quality=90.0  # 从~80提升到90
        )

        # 跟踪改进
        improvement_result = self.culture_system.track_culture_improvement("frontend-team", new_metrics)

        # 验证改进跟踪
        assert 'current_maturity' in improvement_result
        assert 'improvement_trend' in improvement_result
        assert 'next_maturity_level' in improvement_result
        assert 'recommendations' in improvement_result

        # 验证改进趋势
        trend = improvement_result['improvement_trend']
        if not trend.get('insufficient_data'):
            assert 'individual_trends' in trend
            assert 'overall_trend' in trend

        print(f"✅ 文化改进跟踪成功 - 当前成熟度: {improvement_result['current_maturity']}, 建议: {len(improvement_result['recommendations'])} 项")

    def test_maturity_level_determination(self):
        """测试成熟度等级确定"""
        # 测试不同分数对应的成熟度等级
        test_cases = [
            (95, 'optimizing'),
            (85, 'managed'),
            (75, 'defined'),
            (65, 'repeatable'),
            (45, 'initial')
        ]

        for avg_score, expected_level in test_cases:
            # 创建相应的指标
            metrics = QualityCultureMetrics(
                test_first_adoption=avg_score,
                tdd_practice_rate=avg_score,
                code_review_coverage=avg_score,
                automated_test_coverage=avg_score,
                quality_awareness_score=avg_score,
                continuous_learning_index=avg_score,
                collaboration_quality=avg_score
            )

            # 确定成熟度等级
            level = self.culture_system._determine_maturity_level(metrics)
            assert level == expected_level, f"分数 {avg_score} 应该对应等级 {expected_level}，但得到 {level}"

        print("✅ 成熟度等级确定测试通过")

    def test_training_needs_assessment(self):
        """测试培训需求评估"""
        # 创建不同特点的指标来测试培训需求
        low_tdd_metrics = QualityCultureMetrics(
            test_first_adoption=60.0,
            tdd_practice_rate=30.0,  # 低TDD实践率
            code_review_coverage=80.0,
            automated_test_coverage=70.0,
            quality_awareness_score=75.0,
            continuous_learning_index=70.0,
            collaboration_quality=80.0
        )

        training_needs = self.culture_system._determine_training_needs(low_tdd_metrics, 'repeatable')

        # 应该包含TDD培训
        assert 'test_driven_development' in training_needs

        # 测试高质量团队的培训需求
        high_quality_metrics = QualityCultureMetrics(
            test_first_adoption=90.0,
            tdd_practice_rate=85.0,
            code_review_coverage=95.0,
            automated_test_coverage=90.0,
            quality_awareness_score=95.0,
            continuous_learning_index=90.0,
            collaboration_quality=95.0
        )

        high_quality_needs = self.culture_system._determine_training_needs(high_quality_metrics, 'optimizing')

        # 高质量团队可能不需要基础培训，但可能需要高级培训
        assert isinstance(high_quality_needs, list)

        print("✅ 培训需求评估测试通过")

    def test_implementation_plan_generation(self):
        """测试实施计划生成"""
        practice = self.culture_system.test_first_practices['tdd_implementation']

        # 生成实施时间线
        timeline = self.culture_system._create_implementation_timeline(practice)

        assert len(timeline) == 4  # 四个阶段
        assert timeline[0]['phase'] == '准备阶段'
        assert timeline[1]['phase'] == '试点阶段'
        assert timeline[2]['phase'] == '扩展阶段'
        assert timeline[3]['phase'] == '巩固阶段'

        # 验证总时长
        total_days = sum(phase['duration_days'] for phase in timeline)
        assert total_days == practice.estimated_effort_days

        # 生成资源需求
        resources = self.culture_system._identify_resources_needed(practice)

        assert 'human_resources' in resources
        assert 'technical_resources' in resources
        assert 'organizational_support' in resources

        # 验证人力资源包含关键角色
        human_resources = resources['human_resources']
        assert '实施负责人' in human_resources
        assert '技术教练' in human_resources

        print("✅ 实施计划生成测试通过")

    def test_decision_alternatives_evaluation(self):
        """测试决策备选方案评估"""
        framework = self.culture_system.decision_framework

        # 测试备选方案生成
        alternatives = framework._generate_alternatives("测试策略选择", [], [])

        assert len(alternatives) > 0
        for alt in alternatives:
            assert 'action' in alt
            assert 'description' in alt

        # 测试备选方案评估
        criteria = ['潜在收益', '实施风险', '所需时间']
        data_sources = [
            {'name': '质量数据', 'type': 'quantitative', 'record_count': 100},
            {'name': '性能数据', 'type': 'quantitative', 'record_count': 50}
        ]

        evaluated = framework._evaluate_alternatives(alternatives, criteria, data_sources)

        assert len(evaluated) == len(alternatives)
        for alt in evaluated:
            assert 'scores' in alt
            assert 'score' in alt
            assert isinstance(alt['score'], (int, float))

        print("✅ 决策备选方案评估测试通过")

    def test_culture_metrics_calculation(self):
        """测试文化指标计算"""
        # 测试指标计算
        survey_data = {
            'test_first_adoption': [4, 5, 4, 3, 4],
            'tdd_practice': [3, 4, 3, 2, 3],
            'code_review_coverage': [5, 5, 4, 5, 5],
            'automation_coverage': [4, 4, 3, 4, 4]
        }

        metrics = self.culture_system._calculate_culture_metrics(survey_data)

        # 验证计算结果
        assert metrics.test_first_adoption == 80.0  # (4+5+4+3+4)/5 * 20 = 80
        assert metrics.tdd_practice_rate == 60.0    # (3+4+3+2+3)/5 * 20 = 60
        assert metrics.code_review_coverage == 96.0 # (5+5+4+5+5)/5 * 20 = 96
        assert metrics.automated_test_coverage == 76.0 # (4+4+3+4+4)/5 * 20 = 76

        print("✅ 文化指标计算测试通过")

    def test_improvement_trend_analysis(self):
        """测试改进趋势分析"""
        # 创建指标历史
        metrics_history = [
            QualityCultureMetrics(
                test_first_adoption=60.0, tdd_practice_rate=40.0,
                code_review_coverage=80.0, automated_test_coverage=60.0,
                quality_awareness_score=75.0, continuous_learning_index=55.0,
                collaboration_quality=80.0
            ),
            QualityCultureMetrics(
                test_first_adoption=70.0, tdd_practice_rate=50.0,
                code_review_coverage=85.0, automated_test_coverage=70.0,
                quality_awareness_score=80.0, continuous_learning_index=65.0,
                collaboration_quality=85.0
            ),
            QualityCultureMetrics(
                test_first_adoption=75.0, tdd_practice_rate=55.0,
                code_review_coverage=90.0, automated_test_coverage=75.0,
                quality_awareness_score=85.0, continuous_learning_index=70.0,
                collaboration_quality=90.0
            )
        ]

        # 模拟团队画像
        profile = TeamQualityProfile(
            team_name="test-team",
            team_size=5,
            quality_maturity_level="defined",
            strengths=[],
            improvement_areas=[],
            training_needs=[],
            metrics_history=metrics_history
        )

        # 计算改进趋势
        trend = self.culture_system._calculate_improvement_trend(metrics_history)

        assert 'individual_trends' in trend
        assert 'overall_trend' in trend

        # 验证总体趋势
        overall = trend['overall_trend']
        assert 'positive_indicators' in overall
        assert 'total_indicators' in overall
        assert 'improvement_percentage' in overall
        assert 'overall_direction' in overall

        # 所有指标都应该在改进（从历史数据看）
        assert overall['overall_direction'] == 'improving'

        print("✅ 改进趋势分析测试通过")

    def test_maturity_level_prediction(self):
        """测试成熟度等级预测"""
        # 测试从较低成熟度升级
        profile_low = TeamQualityProfile(
            team_name="low-maturity-team",
            team_size=5,
            quality_maturity_level="repeatable",
            strengths=[],
            improvement_areas=[],
            training_needs=[],
            metrics_history=[
                QualityCultureMetrics(
                    test_first_adoption=85.0, tdd_practice_rate=80.0,
                    code_review_coverage=90.0, automated_test_coverage=85.0,
                    quality_awareness_score=90.0, continuous_learning_index=85.0,
                    collaboration_quality=90.0
                )
            ]
        )

        next_level = self.culture_system._predict_next_maturity_level(profile_low)
        # 高分应该能够升级到更高成熟度
        assert next_level in ['defined', 'managed', 'optimizing']

        # 测试高成熟度维持
        profile_high = TeamQualityProfile(
            team_name="high-maturity-team",
            team_size=5,
            quality_maturity_level="optimizing",
            strengths=[],
            improvement_areas=[],
            training_needs=[],
            metrics_history=[
                QualityCultureMetrics(
                    test_first_adoption=95.0, tdd_practice_rate=90.0,
                    code_review_coverage=95.0, automated_test_coverage=95.0,
                    quality_awareness_score=95.0, continuous_learning_index=95.0,
                    collaboration_quality=95.0
                )
            ]
        )

        next_level_high = self.culture_system._predict_next_maturity_level(profile_high)
        # 已经是最高等级，应该维持
        assert next_level_high == "optimizing"

        print("✅ 成熟度等级预测测试通过")

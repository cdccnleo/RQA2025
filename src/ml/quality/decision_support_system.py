"""
质量AI决策支持系统

整合所有AI质量保障组件，提供统一的智能决策支持：
1. 综合质量评估 - 多维度质量指标综合分析
2. 决策推荐引擎 - 基于AI分析的决策建议
3. 风险预测仪表板 - 质量风险可视化展示
4. 自动化决策执行 - 智能决策的自动执行框架
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessment:
    """质量评估结果"""
    timestamp: datetime
    overall_score: float
    risk_level: str
    trend_direction: str
    key_findings: List[str]
    recommended_actions: List[str]
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DecisionRecommendation:
    """决策推荐"""
    decision_id: str
    title: str
    description: str
    priority: str
    category: str
    rationale: str
    expected_impact: Dict[str, Any]
    implementation_plan: List[str]
    success_metrics: List[str]
    confidence_score: float
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['generated_at'] = self.generated_at.isoformat()
        return data


@dataclass
class RiskAlert:
    """风险告警"""
    alert_id: str
    title: str
    description: str
    severity: str
    risk_type: str
    affected_components: List[str]
    probability: float
    time_to_impact: timedelta
    mitigation_actions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['time_to_impact'] = str(self.time_to_impact)
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data


class ComprehensiveQualityAssessor:
    """综合质量评估器"""

    def __init__(self):
        self.quality_dimensions = {
            'code_quality': ['test_coverage', 'code_complexity', 'technical_debt'],
            'performance': ['response_time', 'throughput', 'resource_utilization'],
            'reliability': ['error_rate', 'availability', 'mttr', 'mtbf'],
            'security': ['vulnerability_count', 'compliance_score', 'threat_level'],
            'maintainability': ['code_churn', 'documentation_coverage', 'modularity_score']
        }

        self.dimension_weights = {
            'code_quality': 0.25,
            'performance': 0.25,
            'reliability': 0.25,
            'security': 0.15,
            'maintainability': 0.10
        }

        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.4,
            'critical': 0.2
        }

    def assess_overall_quality(self, quality_metrics: Dict[str, Any],
                             historical_context: pd.DataFrame) -> QualityAssessment:
        """
        综合质量评估

        Args:
            quality_metrics: 当前质量指标
            historical_context: 历史质量数据

        Returns:
            综合质量评估结果
        """
        try:
            # 计算各维度得分
            dimension_scores = {}
            for dimension, metrics in self.quality_dimensions.items():
                dimension_scores[dimension] = self._calculate_dimension_score(
                    dimension, quality_metrics, historical_context
                )

            # 计算综合得分
            overall_score = sum(
                score * self.dimension_weights[dimension]
                for dimension, score in dimension_scores.items()
            )

            # 确定质量等级
            quality_level = self._determine_quality_level(overall_score)

            # 分析趋势方向
            trend_direction = self._analyze_trend_direction(historical_context)

            # 生成关键发现
            key_findings = self._generate_key_findings(dimension_scores, quality_metrics)

            # 生成推荐行动
            recommended_actions = self._generate_recommended_actions(
                dimension_scores, trend_direction, quality_level
            )

            # 计算置信水平
            confidence_level = self._calculate_assessment_confidence(
                quality_metrics, historical_context
            )

            return QualityAssessment(
                timestamp=datetime.now(),
                overall_score=float(overall_score),
                risk_level=self._map_quality_to_risk_level(quality_level),
                trend_direction=trend_direction,
                key_findings=key_findings,
                recommended_actions=recommended_actions,
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"综合质量评估失败: {e}")
            return QualityAssessment(
                timestamp=datetime.now(),
                overall_score=0.5,
                risk_level='medium',
                trend_direction='stable',
                key_findings=['评估过程中出现错误'],
                recommended_actions=['检查系统状态', '重新运行评估'],
                confidence_level=0.1
            )

    def _calculate_dimension_score(self, dimension: str, metrics: Dict[str, Any],
                                 historical_context: pd.DataFrame) -> float:
        """计算维度得分"""
        try:
            dimension_metrics = self.quality_dimensions[dimension]
            scores = []

            for metric in dimension_metrics:
                if metric in metrics:
                    # 标准化指标值到0-1范围
                    raw_value = metrics[metric]
                    normalized_score = self._normalize_metric_value(metric, raw_value)
                    scores.append(normalized_score)

            if scores:
                # 计算加权平均得分
                return np.mean(scores)
            else:
                return 0.5  # 默认中等得分

        except Exception:
            return 0.5

    def _normalize_metric_value(self, metric: str, value: Any) -> float:
        """标准化指标值"""
        try:
            # 根据指标类型进行标准化
            if 'coverage' in metric.lower():
                # 测试覆盖率：0-100% -> 0-1
                return min(1.0, max(0.0, value / 100.0))
            elif 'rate' in metric.lower() or 'error' in metric.lower():
                # 错误率：反向指标，值越低得分越高
                return max(0.0, 1.0 - min(1.0, value))
            elif 'time' in metric.lower() or 'latency' in metric.lower():
                # 时间指标：反向指标，时间越短得分越高
                # 假设基准时间为1秒
                return max(0.0, 1.0 - min(1.0, value))
            elif 'count' in metric.lower() or 'complexity' in metric.lower():
                # 计数型指标：反向指标，数量越少得分越高
                return max(0.0, 1.0 - min(1.0, value / 100.0))
            else:
                # 其他指标：假设已经是0-1范围，或按百分比处理
                if isinstance(value, (int, float)):
                    if value > 1.0:
                        return min(1.0, value / 100.0)
                    else:
                        return max(0.0, min(1.0, value))
                else:
                    return 0.5

        except Exception:
            return 0.5

    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return 'critical'

    def _analyze_trend_direction(self, historical_context: pd.DataFrame) -> str:
        """分析趋势方向"""
        try:
            if historical_context.empty:
                return 'unknown'

            # 计算整体质量趋势
            if 'overall_quality_score' in historical_context.columns:
                scores = historical_context['overall_quality_score'].tail(10).values
            else:
                # 计算最近10条记录的平均质量分数
                recent_data = historical_context.tail(10)
                scores = []

                for _, row in recent_data.iterrows():
                    dimension_scores = []
                    for dimension, metrics in self.quality_dimensions.items():
                        metric_scores = []
                        for metric in metrics:
                            if metric in row:
                                normalized = self._normalize_metric_value(metric, row[metric])
                                metric_scores.append(normalized)
                        if metric_scores:
                            dimension_scores.append(np.mean(metric_scores))

                    if dimension_scores:
                        scores.append(np.mean(dimension_scores))

                scores = np.array(scores)

            if len(scores) >= 2:
                # 计算趋势斜率
                slope = np.polyfit(range(len(scores)), scores, 1)[0]

                if slope > 0.001:
                    return 'improving'
                elif slope < -0.001:
                    return 'declining'
                else:
                    return 'stable'
            else:
                return 'insufficient_data'

        except Exception:
            return 'unknown'

    def _generate_key_findings(self, dimension_scores: Dict[str, float],
                             metrics: Dict[str, Any]) -> List[str]:
        """生成关键发现"""
        findings = []

        try:
            # 找出最低维度的得分
            lowest_dimension = min(dimension_scores.keys(), key=lambda k: dimension_scores[k])

            findings.append(f"{lowest_dimension.replace('_', ' ').title()} 是最需要改进的领域 "
                          f"(得分: {dimension_scores[lowest_dimension]:.2f})")

            # 检查关键指标
            if metrics.get('test_coverage', 0) < 70:
                findings.append(f"测试覆盖率偏低: {metrics.get('test_coverage', 0)}%")

            if metrics.get('error_rate', 0) > 0.05:
                findings.append(f"错误率较高: {metrics.get('error_rate', 0):.2%}")

            if metrics.get('response_time', 0) > 2.0:
                findings.append(f"响应时间较慢: {metrics.get('response_time', 0):.2f}秒")

            # 检查趋势
            trend_direction = self._analyze_trend_direction(pd.DataFrame([metrics]))
            if trend_direction == 'declining':
                findings.append("质量指标显示下降趋势，需要立即关注")

        except Exception as e:
            findings.append(f"生成关键发现时出错: {e}")

        return findings

    def _generate_recommended_actions(self, dimension_scores: Dict[str, float],
                                    trend_direction: str, quality_level: str) -> List[str]:
        """生成推荐行动"""
        actions = []

        try:
            # 基于最低维度得分推荐行动
            lowest_dimension = min(dimension_scores.keys(), key=lambda k: dimension_scores[k])

            dimension_actions = {
                'code_quality': [
                    '增加自动化测试覆盖率',
                    '进行代码重构优化',
                    '引入代码质量检查工具'
                ],
                'performance': [
                    '优化系统性能瓶颈',
                    '扩展系统资源容量',
                    '改进缓存策略'
                ],
                'reliability': [
                    '加强错误处理机制',
                    '改进系统监控告警',
                    '制定故障恢复计划'
                ],
                'security': [
                    '进行安全漏洞扫描',
                    '更新安全补丁',
                    '加强访问控制'
                ],
                'maintainability': [
                    '改进代码文档',
                    '重构复杂模块',
                    '建立代码评审流程'
                ]
            }

            actions.extend(dimension_actions.get(lowest_dimension, []))

            # 基于趋势的额外行动
            if trend_direction == 'declining':
                actions.extend([
                    '立即进行质量评估',
                    '制定质量改进计划',
                    '加强质量监控频率'
                ])

            # 基于质量等级的行动
            if quality_level in ['poor', 'critical']:
                actions.extend([
                    '启动质量紧急响应',
                    '增加质量团队投入',
                    '暂停非关键功能开发'
                ])

        except Exception as e:
            actions.extend(['进行全面质量评估', '制定改进计划'])

        return actions[:5]  # 最多返回5个行动

    def _calculate_assessment_confidence(self, metrics: Dict[str, Any],
                                       historical_context: pd.DataFrame) -> float:
        """计算评估置信水平"""
        try:
            confidence = 0.5  # 基础置信度

            # 数据完整性贡献
            available_metrics = sum(1 for dimension_metrics in self.quality_dimensions.values()
                                  for metric in dimension_metrics if metric in metrics)
            total_metrics = sum(len(metrics) for metrics in self.quality_dimensions.values())

            data_completeness = available_metrics / total_metrics if total_metrics > 0 else 0
            confidence += data_completeness * 0.3

            # 历史数据贡献
            if not historical_context.empty and len(historical_context) >= 7:
                confidence += 0.2

            return min(0.95, max(0.1, confidence))

        except Exception:
            return 0.5

    def _map_quality_to_risk_level(self, quality_level: str) -> str:
        """映射质量等级到风险等级"""
        mapping = {
            'excellent': 'low',
            'good': 'low',
            'fair': 'medium',
            'poor': 'high',
            'critical': 'critical'
        }
        return mapping.get(quality_level, 'medium')


class IntelligentDecisionRecommender:
    """智能决策推荐器"""

    def __init__(self):
        self.decision_templates = self._initialize_decision_templates()
        self.context_analyzer = ContextAnalyzer()
        self.impact_predictor = ImpactPredictor()

    def _initialize_decision_templates(self) -> Dict[str, Dict[str, Any]]:
        """初始化决策模板"""
        return {
            'quality_improvement_initiative': {
                'title': '质量改进倡议',
                'description': '启动全面质量改进计划',
                'category': 'strategic',
                'implementation_plan': [
                    '组建质量改进团队',
                    '进行质量现状评估',
                    '制定改进路线图',
                    '分配资源和预算',
                    '建立监控和报告机制'
                ],
                'success_metrics': [
                    '质量分数提升20%',
                    '缺陷密度降低30%',
                    '客户满意度提升15%'
                ]
            },
            'performance_optimization_project': {
                'title': '性能优化项目',
                'description': '专项性能优化和容量扩展',
                'category': 'technical',
                'implementation_plan': [
                    '识别性能瓶颈',
                    '设计优化方案',
                    '实施代码优化',
                    '进行性能测试',
                    '部署和监控'
                ],
                'success_metrics': [
                    '响应时间减少25%',
                    '吞吐量提升30%',
                    '资源利用率优化20%'
                ]
            },
            'risk_mitigation_program': {
                'title': '风险缓解计划',
                'description': '针对关键风险制定缓解策略',
                'category': 'risk_management',
                'implementation_plan': [
                    '识别关键风险',
                    '评估风险影响',
                    '制定缓解策略',
                    '实施控制措施',
                    '持续监控和调整'
                ],
                'success_metrics': [
                    '关键风险降低50%',
                    '系统稳定性提升',
                    '故障恢复时间减少'
                ]
            },
            'maintenance_schedule_optimization': {
                'title': '维护计划优化',
                'description': '基于预测性维护优化维护计划',
                'category': 'operational',
                'implementation_plan': [
                    '评估当前维护计划',
                    '实施预测性维护',
                    '优化维护时间窗口',
                    '自动化维护任务',
                    '监控维护效果'
                ],
                'success_metrics': [
                    '计划外宕机减少40%',
                    '维护成本降低25%',
                    '系统可用性提升'
                ]
            }
        }

    def generate_decision_recommendations(self, quality_assessment: QualityAssessment,
                                        risk_alerts: List[RiskAlert],
                                        system_context: Dict[str, Any]) -> List[DecisionRecommendation]:
        """
        生成决策推荐

        Args:
            quality_assessment: 质量评估结果
            risk_alerts: 风险告警列表
            system_context: 系统上下文信息

        Returns:
            决策推荐列表
        """
        try:
            recommendations = []

            # 基于质量评估的推荐
            quality_based_recs = self._generate_quality_based_recommendations(quality_assessment)
            recommendations.extend(quality_based_recs)

            # 基于风险告警的推荐
            risk_based_recs = self._generate_risk_based_recommendations(risk_alerts)
            recommendations.extend(risk_based_recs)

            # 基于系统上下文的推荐
            context_based_recs = self._generate_context_based_recommendations(system_context)
            recommendations.extend(context_based_recs)

            # 计算优先级和影响
            for rec in recommendations:
                rec.expected_impact = self.impact_predictor.predict_decision_impact(rec, system_context)
                rec.confidence_score = self._calculate_recommendation_confidence(rec, quality_assessment)

            # 按优先级排序
            recommendations.sort(key=lambda x: self._calculate_recommendation_priority(x), reverse=True)

            return recommendations

        except Exception as e:
            logger.error(f"生成决策推荐失败: {e}")
            return []

    def _generate_quality_based_recommendations(self, assessment: QualityAssessment) -> List[DecisionRecommendation]:
        """基于质量评估生成推荐"""
        recommendations = []

        try:
            quality_score = assessment.overall_score
            risk_level = assessment.risk_level

            if quality_score < 0.6:
                # 质量严重不合格
                rec = DecisionRecommendation(
                    decision_id=f"quality_improvement_{int(datetime.now().timestamp())}",
                    title="紧急质量改进计划",
                    description="系统质量严重不合格，需要立即启动全面改进计划",
                    priority='critical',
                    category='strategic',
                    rationale=f"当前质量分数仅为{quality_score:.2f}，处于{risk_level}风险水平",
                    expected_impact={},
                    implementation_plan=[
                        "成立质量改进专项小组",
                        "进行全面质量审计",
                        "制定为期3个月的改进计划",
                        "每日质量指标监控和报告"
                    ],
                    success_metrics=[
                        "质量分数提升至0.75以上",
                        "关键风险降低至中等水平",
                        "团队质量意识显著提升"
                    ],
                    confidence_score=0.9,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

            elif quality_score < 0.8:
                # 质量需要改进
                rec = DecisionRecommendation(
                    decision_id=f"quality_enhancement_{int(datetime.now().timestamp())}",
                    title="质量提升项目",
                    description="系统质量需要持续改进和优化",
                    priority='high',
                    category='operational',
                    rationale=f"质量分数{quality_score:.2f}有改善空间，当前风险水平为{risk_level}",
                    expected_impact={},
                    implementation_plan=[
                        "识别质量改进重点领域",
                        "制定渐进式改进计划",
                        "实施自动化质量检查",
                        "建立质量改进反馈机制"
                    ],
                    success_metrics=[
                        "质量分数稳步提升",
                        "缺陷率降低20%",
                        "自动化测试覆盖率提升15%"
                    ],
                    confidence_score=0.8,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

        except Exception as e:
            logger.error(f"生成质量基础推荐失败: {e}")

        return recommendations

    def _generate_risk_based_recommendations(self, risk_alerts: List[RiskAlert]) -> List[DecisionRecommendation]:
        """基于风险告警生成推荐"""
        recommendations = []

        try:
            # 按严重性分组告警
            critical_alerts = [a for a in risk_alerts if a.severity == 'critical']
            high_alerts = [a for a in risk_alerts if a.severity == 'high']

            if critical_alerts:
                rec = DecisionRecommendation(
                    decision_id=f"critical_risk_mitigation_{int(datetime.now().timestamp())}",
                    title="紧急风险缓解",
                    description=f"存在{len(critical_alerts)}个严重风险需要立即处理",
                    priority='critical',
                    category='risk_management',
                    rationale=f"检测到{len(critical_alerts)}个严重风险，可能导致系统故障或重大影响",
                    expected_impact={},
                    implementation_plan=[
                        "立即隔离受影响组件",
                        "激活应急响应团队",
                        "执行风险缓解措施",
                        "加强系统监控",
                        "制定详细的恢复计划"
                    ],
                    success_metrics=[
                        f"所有{len(critical_alerts)}个严重风险得到控制",
                        "系统稳定性恢复",
                        "业务影响最小化"
                    ],
                    confidence_score=0.95,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

            elif high_alerts:
                rec = DecisionRecommendation(
                    decision_id=f"high_risk_management_{int(datetime.now().timestamp())}",
                    title="高风险管理计划",
                    description=f"需要管理{len(high_alerts)}个高风险项目",
                    priority='high',
                    category='risk_management',
                    rationale=f"存在{len(high_alerts)}个高风险需要优先处理",
                    expected_impact={},
                    implementation_plan=[
                        "评估高风险影响程度",
                        "制定风险缓解策略",
                        "安排专门资源处理",
                        "建立风险监控机制",
                        "定期风险评估和更新"
                    ],
                    success_metrics=[
                        "高风险降低至中等水平",
                        "建立有效的风险控制机制",
                        "风险管理能力得到提升"
                    ],
                    confidence_score=0.85,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

        except Exception as e:
            logger.error(f"生成风险基础推荐失败: {e}")

        return recommendations

    def _generate_context_based_recommendations(self, system_context: Dict[str, Any]) -> List[DecisionRecommendation]:
        """基于系统上下文生成推荐"""
        recommendations = []

        try:
            # 分析系统负载和资源使用
            cpu_usage = system_context.get('cpu_usage', 0)
            memory_usage = system_context.get('memory_usage', 0)
            active_users = system_context.get('active_users', 0)

            # 高负载场景推荐
            if cpu_usage > 80 or memory_usage > 85:
                rec = DecisionRecommendation(
                    decision_id=f"resource_optimization_{int(datetime.now().timestamp())}",
                    title="资源优化项目",
                    description="系统资源使用率过高，需要进行优化",
                    priority='high' if cpu_usage > 90 or memory_usage > 90 else 'medium',
                    category='technical',
                    rationale=f"CPU使用率{cpu_usage}%, 内存使用率{memory_usage}%, 资源压力较大",
                    expected_impact={},
                    implementation_plan=[
                        "进行系统性能分析",
                        "识别资源消耗热点",
                        "实施代码和架构优化",
                        "考虑资源扩展方案",
                        "建立资源监控告警"
                    ],
                    success_metrics=[
                        "CPU使用率降低20%",
                        "内存使用率降低15%",
                        "系统响应性能提升"
                    ],
                    confidence_score=0.8,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

            # 高并发场景推荐
            if active_users > 1000:  # 假设阈值
                rec = DecisionRecommendation(
                    decision_id=f"scalability_improvement_{int(datetime.now().timestamp())}",
                    title="可扩展性改进",
                    description="系统并发负载高，需要提升可扩展性",
                    priority='medium',
                    category='technical',
                    rationale=f"当前活跃用户数{active_users}, 需要提升系统可扩展性",
                    expected_impact={},
                    implementation_plan=[
                        "评估当前系统架构",
                        "设计可扩展性改进方案",
                        "实施负载均衡和分布式部署",
                        "优化数据库和缓存策略",
                        "进行压力测试验证"
                    ],
                    success_metrics=[
                        "支持并发用户数提升50%",
                        "系统响应时间稳定",
                        "资源利用率优化"
                    ],
                    confidence_score=0.75,
                    generated_at=datetime.now()
                )
                recommendations.append(rec)

        except Exception as e:
            logger.error(f"生成上下文基础推荐失败: {e}")

        return recommendations

    def _calculate_recommendation_priority(self, recommendation: DecisionRecommendation) -> float:
        """计算推荐优先级"""
        try:
            priority_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            base_score = priority_scores.get(recommendation.priority, 2)

            # 基于置信度和预期影响调整
            confidence_bonus = recommendation.confidence_score * 0.5
            impact_bonus = recommendation.expected_impact.get('magnitude', 0) * 0.3

            return base_score + confidence_bonus + impact_bonus

        except Exception:
            return 2.0


class ContextAnalyzer:
    """上下文分析器"""

    def analyze_system_context(self, system_metrics: Dict[str, Any],
                             historical_data: pd.DataFrame) -> Dict[str, Any]:
        """分析系统上下文"""
        try:
            context = {
                'current_load': self._assess_current_load(system_metrics),
                'resource_status': self._analyze_resource_status(system_metrics),
                'performance_trend': self._analyze_performance_trend(historical_data),
                'anomaly_indicators': self._detect_anomaly_indicators(system_metrics, historical_data),
                'bottleneck_analysis': self._identify_system_bottlenecks(system_metrics)
            }

            return context

        except Exception as e:
            logger.error(f"系统上下文分析失败: {e}")
            return {}

    def _assess_current_load(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估当前负载"""
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        connections = metrics.get('active_connections', 0)

        # 确定负载等级
        if cpu > 90 or memory > 90 or connections > 10000:
            level = 'critical'
        elif cpu > 80 or memory > 85 or connections > 5000:
            level = 'high'
        elif cpu > 70 or memory > 80 or connections > 2000:
            level = 'medium'
        else:
            level = 'low'

        return {
            'level': level,
            'cpu_usage': cpu,
            'memory_usage': memory,
            'active_connections': connections
        }

    def _analyze_resource_status(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析资源状态"""
        return {
            'cpu_status': 'normal' if metrics.get('cpu_usage', 0) < 80 else 'high',
            'memory_status': 'normal' if metrics.get('memory_usage', 0) < 85 else 'high',
            'disk_status': 'normal' if metrics.get('disk_usage', 0) < 90 else 'high',
            'network_status': 'normal' if metrics.get('network_io', 0) < 1000 else 'high'
        }

    def _analyze_performance_trend(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """分析性能趋势"""
        if historical_data.empty:
            return {'trend': 'unknown'}

        # 简化的趋势分析
        recent_perf = historical_data.get('response_time', pd.Series()).tail(10).mean()
        older_perf = historical_data.get('response_time', pd.Series()).head(10).mean()

        if recent_perf > older_perf * 1.1:
            trend = 'degrading'
        elif recent_perf < older_perf * 0.9:
            trend = 'improving'
        else:
            trend = 'stable'

        return {'trend': trend, 'change_percent': (recent_perf - older_perf) / older_perf * 100}

    def _detect_anomaly_indicators(self, metrics: Dict[str, Any],
                                 historical_data: pd.DataFrame) -> List[str]:
        """检测异常指标"""
        anomalies = []

        # CPU异常
        if metrics.get('cpu_usage', 0) > 95:
            anomalies.append('extremely_high_cpu_usage')

        # 内存异常
        if metrics.get('memory_usage', 0) > 95:
            anomalies.append('extremely_high_memory_usage')

        # 错误率异常
        if metrics.get('error_rate', 0) > 0.1:
            anomalies.append('high_error_rate')

        return anomalies

    def _identify_system_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """识别系统瓶颈"""
        bottlenecks = []

        if metrics.get('cpu_usage', 0) > 85:
            bottlenecks.append('cpu')
        if metrics.get('memory_usage', 0) > 85:
            bottlenecks.append('memory')
        if metrics.get('disk_io', 0) > 1000:
            bottlenecks.append('disk_io')
        if metrics.get('response_time', 0) > 3.0:
            bottlenecks.append('response_time')

        return bottlenecks


class ImpactPredictor:
    """影响预测器"""

    def predict_decision_impact(self, decision: DecisionRecommendation,
                              system_context: Dict[str, Any]) -> Dict[str, Any]:
        """预测决策影响"""
        try:
            # 简化的影响预测逻辑
            impact_base = {
                'quality_improvement_initiative': {'magnitude': 0.8, 'duration': 90, 'confidence': 0.8},
                'performance_optimization_project': {'magnitude': 0.6, 'duration': 60, 'confidence': 0.85},
                'risk_mitigation_program': {'magnitude': 0.7, 'duration': 45, 'confidence': 0.9},
                'maintenance_schedule_optimization': {'magnitude': 0.5, 'duration': 30, 'confidence': 0.75}
            }

            template_key = decision.title.lower().replace(' ', '_')
            base_impact = impact_base.get(template_key, {'magnitude': 0.5, 'duration': 30, 'confidence': 0.7})

            # 根据系统上下文调整
            context_multiplier = 1.0
            if system_context.get('current_load', {}).get('level') == 'high':
                context_multiplier *= 0.8  # 高负载情况下影响可能较小

            return {
                'magnitude': base_impact['magnitude'] * context_multiplier,
                'duration_days': base_impact['duration'],
                'confidence': base_impact['confidence'],
                'estimated_benefit': base_impact['magnitude'] * 100  # 百分比
            }

        except Exception as e:
            logger.error(f"决策影响预测失败: {e}")
            return {'magnitude': 0.5, 'duration_days': 30, 'confidence': 0.5}


class QualityAIDecisionSupportSystem:
    """质量AI决策支持系统"""

    def __init__(self):
        self.quality_assessor = ComprehensiveQualityAssessor()
        self.decision_recommender = IntelligentDecisionRecommender()
        self.context_analyzer = ContextAnalyzer()

        # 决策历史
        self.decision_history = []
        self.assessment_history = []
        self.alert_history = []

    def perform_comprehensive_quality_analysis(self, quality_metrics: Dict[str, Any],
                                            historical_context: pd.DataFrame,
                                            risk_alerts: List[RiskAlert] = None) -> Dict[str, Any]:
        """
        执行综合质量分析

        Args:
            quality_metrics: 质量指标
            historical_context: 历史上下文
            risk_alerts: 风险告警

        Returns:
            综合分析结果
        """
        try:
            # 1. 质量评估
            quality_assessment = self.quality_assessor.assess_overall_quality(
                quality_metrics, historical_context
            )

            # 2. 系统上下文分析
            system_context = self.context_analyzer.analyze_system_context(
                quality_metrics, historical_context
            )

            # 3. 决策推荐
            risk_alerts = risk_alerts or []
            decision_recommendations = self.decision_recommender.generate_decision_recommendations(
                quality_assessment, risk_alerts, system_context
            )

            # 4. 生成综合报告
            comprehensive_analysis = {
                'timestamp': datetime.now(),
                'quality_assessment': quality_assessment.to_dict(),
                'system_context': system_context,
                'decision_recommendations': [rec.to_dict() for rec in decision_recommendations],
                'risk_alerts': [alert.to_dict() for alert in risk_alerts],
                'actionable_insights': self._generate_actionable_insights(
                    quality_assessment, decision_recommendations, system_context
                ),
                'next_steps': self._recommend_next_steps(quality_assessment, decision_recommendations)
            }

            # 记录历史
            self.assessment_history.append(comprehensive_analysis)
            if len(self.assessment_history) > 100:
                self.assessment_history = self.assessment_history[-100:]

            return comprehensive_analysis

        except Exception as e:
            logger.error(f"综合质量分析执行失败: {e}")
            return {'error': str(e)}

    def get_decision_support_dashboard(self) -> Dict[str, Any]:
        """获取决策支持仪表板数据"""
        try:
            # 最近的质量评估
            latest_assessment = None
            if self.assessment_history:
                latest_assessment = self.assessment_history[-1]

            # 关键指标趋势
            trends = self._calculate_key_metrics_trends()

            # 活跃告警
            active_alerts = [alert for alert in self.alert_history
                           if alert.get('expires_at') and alert['expires_at'] > datetime.now()]

            # 待执行决策
            pending_decisions = []
            if latest_assessment:
                recommendations = latest_assessment.get('decision_recommendations', [])
                pending_decisions = [rec for rec in recommendations
                                   if not rec.get('executed', False)][:5]  # 最多显示5个

            dashboard_data = {
                'latest_assessment': latest_assessment,
                'key_metrics_trends': trends,
                'active_alerts': active_alerts,
                'pending_decisions': pending_decisions,
                'system_health_score': self._calculate_system_health_score(),
                'risk_level_distribution': self._calculate_risk_distribution(),
                'generated_at': datetime.now()
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"获取决策支持仪表板失败: {e}")
            return {'error': str(e)}

    def _generate_actionable_insights(self, assessment: QualityAssessment,
                                    recommendations: List[DecisionRecommendation],
                                    system_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成可操作的洞察"""
        insights = []

        try:
            # 质量洞察
            if assessment.overall_score < 0.7:
                insights.append({
                    'type': 'quality_concern',
                    'title': '质量需要紧急关注',
                    'description': f'整体质量分数仅为{assessment.overall_score:.2f}，低于标准水平',
                    'priority': 'high',
                    'action_items': ['启动质量改进项目', '增加测试覆盖率', '优化代码质量']
                })

            # 性能洞察
            if system_context.get('current_load', {}).get('level') == 'high':
                insights.append({
                    'type': 'performance_concern',
                    'title': '系统负载过高',
                    'description': '当前系统负载处于高水平，可能影响性能',
                    'priority': 'medium',
                    'action_items': ['监控资源使用', '考虑容量扩展', '优化性能瓶颈']
                })

            # 决策洞察
            if recommendations:
                top_recommendation = max(recommendations, key=lambda x: x.confidence_score)
                insights.append({
                    'type': 'decision_recommendation',
                    'title': f'优先执行: {top_recommendation.title}',
                    'description': top_recommendation.description,
                    'priority': top_recommendation.priority,
                    'action_items': top_recommendation.implementation_plan[:3]  # 前3个步骤
                })

        except Exception as e:
            logger.error(f"生成可操作洞察失败: {e}")

        return insights

    def _recommend_next_steps(self, assessment: QualityAssessment,
                            recommendations: List[DecisionRecommendation]) -> List[str]:
        """推荐下一步行动"""
        next_steps = []

        try:
            # 基于质量水平的下一步
            if assessment.overall_score < 0.6:
                next_steps.extend([
                    "立即召开质量评审会议",
                    "制定紧急质量改进计划",
                    "增加质量监控频率"
                ])
            elif assessment.overall_score < 0.8:
                next_steps.extend([
                    "启动质量持续改进项目",
                    "优化自动化测试流程",
                    "加强代码质量检查"
                ])

            # 基于决策推荐的下一步
            if recommendations:
                critical_decisions = [rec for rec in recommendations if rec.priority == 'critical']
                if critical_decisions:
                    next_steps.append(f"优先执行关键决策: {critical_decisions[0].title}")

            # 默认下一步
            if not next_steps:
                next_steps.extend([
                    "继续监控质量指标",
                    "定期审查系统性能",
                    "保持自动化测试覆盖"
                ])

        except Exception as e:
            next_steps.extend(["检查系统状态", "重新运行质量评估"])

        return next_steps

    def _calculate_key_metrics_trends(self) -> Dict[str, Any]:
        """计算关键指标趋势"""
        try:
            if not self.assessment_history:
                return {}

            # 提取最近的评估数据
            recent_assessments = self.assessment_history[-10:]  # 最近10次评估

            trends = {}

            # 质量分数趋势
            quality_scores = [ass['quality_assessment']['overall_score'] for ass in recent_assessments]
            trends['quality_score'] = {
                'current': quality_scores[-1] if quality_scores else 0,
                'trend': self._calculate_trend(quality_scores),
                'change_percent': self._calculate_change_percent(quality_scores)
            }

            # 风险等级分布
            risk_levels = [ass['quality_assessment']['risk_level'] for ass in recent_assessments]
            risk_distribution = {
                'low': risk_levels.count('low'),
                'medium': risk_levels.count('medium'),
                'high': risk_levels.count('high'),
                'critical': risk_levels.count('critical')
            }
            trends['risk_distribution'] = risk_distribution

            return trends

        except Exception as e:
            logger.error(f"计算关键指标趋势失败: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'

        # 计算线性趋势
        if len(values) >= 2:
            slope = (values[-1] - values[0]) / len(values)
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'

        return 'stable'

    def _calculate_change_percent(self, values: List[float]) -> float:
        """计算变化百分比"""
        if len(values) < 2:
            return 0.0

        return (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0.0

    def _calculate_system_health_score(self) -> float:
        """计算系统健康分数"""
        try:
            if not self.assessment_history:
                return 0.5

            latest_assessment = self.assessment_history[-1]
            quality_score = latest_assessment['quality_assessment']['overall_score']

            # 考虑其他因素
            context_penalty = 0
            system_context = latest_assessment.get('system_context', {})

            if system_context.get('current_load', {}).get('level') == 'high':
                context_penalty += 0.1
            if system_context.get('resource_status', {}).get('cpu_status') == 'high':
                context_penalty += 0.1

            health_score = quality_score - context_penalty
            return max(0.0, min(1.0, health_score))

        except Exception:
            return 0.5

    def _calculate_risk_distribution(self) -> Dict[str, int]:
        """计算风险分布"""
        try:
            if not self.assessment_history:
                return {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

            recent_assessments = self.assessment_history[-10:]
            risk_levels = [ass['quality_assessment']['risk_level'] for ass in recent_assessments]

            return {
                'low': risk_levels.count('low'),
                'medium': risk_levels.count('medium'),
                'high': risk_levels.count('high'),
                'critical': risk_levels.count('critical')
            }

        except Exception:
            return {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            stats = {
                'total_assessments': len(self.assessment_history),
                'total_decisions': len(self.decision_history),
                'total_alerts': len(self.alert_history),
                'system_health_score': self._calculate_system_health_score(),
                'avg_assessment_confidence': self._calculate_avg_assessment_confidence(),
                'most_common_risk_level': self._get_most_common_risk_level()
            }

            return stats

        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            return {'error': str(e)}

    def _calculate_avg_assessment_confidence(self) -> float:
        """计算平均评估置信度"""
        try:
            if not self.assessment_history:
                return 0.0

            confidences = [ass['quality_assessment']['confidence_level']
                          for ass in self.assessment_history[-20:]]  # 最近20次

            return sum(confidences) / len(confidences) if confidences else 0.0

        except Exception:
            return 0.0

    def _get_most_common_risk_level(self) -> str:
        """获取最常见的风险等级"""
        try:
            if not self.assessment_history:
                return 'unknown'

            risk_levels = [ass['quality_assessment']['risk_level']
                          for ass in self.assessment_history[-20:]]

            if risk_levels:
                most_common = max(set(risk_levels), key=risk_levels.count)
                return most_common

            return 'unknown'

        except Exception:
            return 'unknown'

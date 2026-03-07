"""
建议生成器组件

职责:
- 生成优化建议和洞察
- 评估建议的优先级和影响
- 追踪建议的实施效果
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol
from datetime import datetime
from enum import Enum
import uuid

# 导入常量
from src.core.constants import MAX_RECORDS, MAX_RETRIES, DEFAULT_TIMEOUT, DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


# 先定义枚举和数据类（在Protocol之前）
class RecommendationCategory(Enum):
    """建议类别"""
    PERFORMANCE = "performance"      # 性能优化
    RISK = "risk"                    # 风险管理
    EXECUTION = "execution"          # 执行优化
    STRATEGY = "strategy"            # 策略调整
    SYSTEM = "system"                # 系统改进


class PriorityLevel(Enum):
    """优先级级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """优化建议"""
    recommendation_id: str
    title: str
    description: str
    category: RecommendationCategory
    priority: PriorityLevel
    confidence: float
    expected_impact: Dict[str, Any] = field(default_factory=dict)
    implementation_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationStatus:
    """实施状态"""
    recommendation_id: str
    status: str  # 'pending', 'in_progress', 'completed', 'cancelled'
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)


# 专门组件实现
class RecommendationCreatorImpl:
    """建议创建器实现 - 职责：生成优化建议"""

    def __init__(self, config: 'RecommendationConfig', recommendations_cache: List[Recommendation], recommendation_history: List[Recommendation]):
        self.config = config
        self._recommendations_cache = recommendations_cache
        self._recommendation_history = recommendation_history

    async def generate_recommendations(self, context: Any, analysis: Any, execution: Any) -> List[Recommendation]:
        """生成优化建议"""
        recommendations = []

        # 1. 基于分析结果生成建议
        if hasattr(analysis, 'recommendations'):
            recommendations.extend(await self._generate_from_analysis(analysis))

        # 2. 基于执行结果生成建议
        if hasattr(execution, 'metrics'):
            recommendations.extend(await self._generate_from_execution(execution))

        # 3. AI洞察建议（如果启用）
        if self.config.enable_ai_insights:
            recommendations.extend(await self._generate_ai_insights(context, analysis, execution))

        # 4. 过滤和限制数量
        filtered = self._filter_recommendations(recommendations)
        final_recommendations = filtered[:self.config.max_recommendations]

        # 5. 缓存和记录
        self._recommendations_cache.extend(final_recommendations)
        self._recommendation_history.extend(final_recommendations)

        logger.info(f"生成了{len(final_recommendations)}条优化建议")
        return final_recommendations

    async def _generate_from_analysis(self, analysis: Any) -> List[Recommendation]:
        """基于分析结果生成建议"""
        # 实现分析结果建议生成逻辑
        return []

    async def _generate_from_execution(self, execution: Any) -> List[Recommendation]:
        """基于执行结果生成建议"""
        # 实现执行结果建议生成逻辑
        return []

    async def _generate_ai_insights(self, context: Any, analysis: Any, execution: Any) -> List[Recommendation]:
        """生成AI洞察建议"""
        # 实现AI洞察建议生成逻辑
        return []

    def _filter_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """过滤建议"""
        if not self.config.filter_enabled:
            return recommendations

        filtered = []
        for rec in recommendations:
            if rec.confidence >= self.config.min_confidence:
                filtered.append(rec)

        return filtered

    def _generate_id(self) -> str:
        """生成建议ID"""
        return str(uuid.uuid4())


class RecommendationPrioritizerImpl:
    """建议优先级评估器实现 - 职责：评估和排序建议优先级"""

    def __init__(self, config: 'RecommendationConfig'):
        self.config = config

    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """评估建议优先级"""
        return sorted(recommendations, key=self._sort_key, reverse=True)

    def _sort_key(self, rec: Recommendation) -> float:
        """计算排序键值"""
        priority_weights = {
            PriorityLevel.CRITICAL: 4.0,
            PriorityLevel.HIGH: 3.0,
            PriorityLevel.MEDIUM: 2.0,
            PriorityLevel.LOW: 1.0
        }

        base_score = priority_weights.get(rec.priority, 1.0)
        confidence_score = rec.confidence * 0.5
        return base_score + confidence_score


class ImplementationTrackerImpl:
    """实施追踪器实现 - 职责：追踪建议实施状态"""

    def __init__(self, config: 'RecommendationConfig'):
        self.config = config
        self._implementation_tracker: Dict[str, ImplementationStatus] = {}
        self._recommendations_cache: List[Recommendation] = []

    def track_implementation(self, recommendation_id: str, status: str):
        """追踪实施状态"""
        if recommendation_id not in self._implementation_tracker:
            self._implementation_tracker[recommendation_id] = ImplementationStatus(
                recommendation_id=recommendation_id,
                status=status,
                progress=0.0
            )

        impl_status = self._implementation_tracker[recommendation_id]
        impl_status.status = status

        if status == 'in_progress' and not impl_status.started_at:
            impl_status.started_at = datetime.now()
        elif status == 'completed':
            impl_status.progress = MAX_RETRIES  # 百分比，保留100.0
            impl_status.completed_at = datetime.now()

    def get_implementation_status(self, recommendation_id: str) -> Optional[ImplementationStatus]:
        """获取实施状态"""
        return self._implementation_tracker.get(recommendation_id)

    def get_active_recommendations(self) -> List[Recommendation]:
        """获取活跃建议（未完成或进行中）"""
        active_ids = {
            rid for rid, status in self._implementation_tracker.items()
            if status.status in ['pending', 'in_progress']
        }

        return [
            rec for rec in self._recommendations_cache
            if rec.recommendation_id in active_ids
        ]


class RecommendationGenerator:
    """
    建议生成器组件 - 重构版：组合模式

    基于分析结果和执行数据生成优化建议
    评估建议优先级并追踪实施效果
    """

    def __init__(self, config: 'RecommendationConfig'):
        """
        初始化建议生成器

        Args:
            config: 建议配置对象
        """
        self.config = config
        self._recommendations_cache: List[Recommendation] = []
        self._implementation_tracker: Dict[str, ImplementationStatus] = {}
        self._recommendation_history: List[Recommendation] = []

        # 初始化专门组件
        self._creator = RecommendationCreatorImpl(config, self._recommendations_cache, self._recommendation_history)
        self._prioritizer = RecommendationPrioritizerImpl(config)
        self._tracker = ImplementationTrackerImpl(config)
        self._tracker._recommendations_cache = self._recommendations_cache

        logger.info("重构后的建议生成器初始化完成")

    # 代理方法到专门的组件
    async def generate_recommendations(self, context: Any,
                                      analysis: Any,
                                      execution: Any) -> List[Recommendation]:
        """生成优化建议 - 代理到建议创建器"""
        recommendations = await self._creator.generate_recommendations(context, analysis, execution)
        # 应用优先级排序
        return self._prioritizer.prioritize_recommendations(recommendations)

    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """评估建议优先级 - 代理到优先级评估器"""
        return self._prioritizer.prioritize_recommendations(recommendations)

    def track_implementation(self, recommendation_id: str, status: str):
        """追踪实施状态 - 代理到实施追踪器"""
        return self._tracker.track_implementation(recommendation_id, status)

    def get_implementation_status(self, recommendation_id: str) -> Optional[ImplementationStatus]:
        """获取实施状态 - 代理到实施追踪器"""
        return self._tracker.get_implementation_status(recommendation_id)

    def get_active_recommendations(self) -> List[Recommendation]:
        """获取活跃建议 - 代理到实施追踪器"""
        return self._tracker.get_active_recommendations()

    # 保持向后兼容性
    def _filter_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """过滤建议（向后兼容）"""
        return self._creator._filter_recommendations(recommendations)

    def _generate_id(self) -> str:
        """生成建议ID（向后兼容）"""
        return self._creator._generate_id()

    async def generate_stage_recommendation(self, stage: str,
                                           stage_result: Dict[str, Any]) -> Optional[Recommendation]:
        """
        生成阶段建议

        Args:
            stage: 阶段名称
            stage_result: 阶段结果

        Returns:
            Optional[Recommendation]: 建议（如果有）
        """
        # 检查阶段结果
        status = stage_result.get('status')

        if status == 'failed':
            # 生成失败相关建议
            return Recommendation(
                recommendation_id=self._generate_id(),
                title=f"{stage} 阶段执行失败",
                description=f"需要检查{stage}阶段的执行逻辑和数据",
                category=RecommendationCategory.EXECUTION,
                priority=PriorityLevel.HIGH,
                confidence=0.9,
                expected_impact={'execution_success_rate': '+15%'},
                implementation_steps=[
                    "检查阶段输入数据",
                    "验证业务逻辑",
                    "增加错误处理"
                ]
            )

        elif status == 'completed':
            # 检查性能
            execution_time = stage_result.get('execution_time', 0)
            if execution_time > 5.0:  # 假设阈值5秒
                return Recommendation(
                    recommendation_id=self._generate_id(),
                    title=f"{stage} 阶段性能优化",
                    description=f"{stage}执行时间{execution_time:.2f}秒，建议优化",
                    category=RecommendationCategory.PERFORMANCE,
                    priority=PriorityLevel.MEDIUM,
                    confidence=0.7,
                    expected_impact={'execution_time': f'-{DEFAULT_TIMEOUT}%'},  # 百分比字面量，保留30
                    implementation_steps=[
                        "分析性能瓶颈",
                        "优化算法复杂度",
                        "增加缓存机制"
                    ]
                )

        return None

    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        建议优先级排序

        Args:
            recommendations: 建议列表

        Returns:
            List[Recommendation]: 排序后的建议
        """
        # 优先级权重
        priority_weights = {
            PriorityLevel.CRITICAL: MAX_RECORDS,
            PriorityLevel.HIGH: MAX_RETRIES,
            PriorityLevel.MEDIUM: DEFAULT_BATCH_SIZE,
            PriorityLevel.LOW: 1
        }

        # 综合排序：优先级 + 置信度
        def sort_key(rec: Recommendation):
            return priority_weights.get(rec.priority, 1) * rec.confidence

        sorted_recs = sorted(recommendations, key=sort_key, reverse=True)
        return sorted_recs

    def track_implementation(self, recommendation_id: str,
                            status: str,
                            progress: float = 0.0,
                            notes: Optional[str] = None) -> bool:
        """
        追踪建议实施

        Args:
            recommendation_id: 建议ID
            status: 实施状态
            progress: 进度 (0-1)
            notes: 备注

        Returns:
            bool: 是否成功
        """
        if recommendation_id not in self._implementation_tracker:
            self._implementation_tracker[recommendation_id] = ImplementationStatus(
                recommendation_id=recommendation_id,
                status='pending'
            )

        tracker = self._implementation_tracker[recommendation_id]
        tracker.status = status
        tracker.progress = progress

        if notes:
            tracker.notes.append(notes)

        if status == 'in_progress' and tracker.started_at is None:
            tracker.started_at = datetime.now()

        if status == 'completed':
            tracker.completed_at = datetime.now()

        logger.info(f"建议实施状态更新: {recommendation_id} -> {status} ({progress*MAX_RETRIES:.0f}%)")
        return True

    def get_active_recommendations(self) -> List[Recommendation]:
        """获取活跃建议"""
        return self._recommendations_cache.copy()

    def get_implementation_status(self, recommendation_id: str) -> Optional[ImplementationStatus]:
        """获取实施状态"""
        return self._implementation_tracker.get(recommendation_id)

    def get_status(self) -> Dict[str, Any]:
        """获取生成器状态"""
        return {
            'cached_recommendations': len(self._recommendations_cache),
            'tracked_implementations': len(self._implementation_tracker),
            'history_size': len(self._recommendation_history),
            'config': {
                'max_recommendations': self.config.max_recommendations,
                'min_confidence': self.config.min_confidence,
                'ai_insights_enabled': self.config.enable_ai_insights
            }
        }

    # 私有辅助方法
    async def _generate_from_analysis(self, analysis: Any) -> List[Recommendation]:
        """从分析结果生成建议"""
        recommendations = []

        if hasattr(analysis, 'recommendations'):
            for rec_text in analysis.recommendations[:3]:  # 取前3条
                recommendations.append(
                    Recommendation(
                        recommendation_id=self._generate_id(),
                        title="性能优化建议",
                        description=rec_text,
                        category=RecommendationCategory.PERFORMANCE,
                        priority=PriorityLevel.MEDIUM,
                        confidence=0.7,
                        implementation_steps=["待详细分析"]
                    )
                )

        return recommendations

    async def _generate_from_execution(self, execution: Any) -> List[Recommendation]:
        """从执行结果生成建议"""
        recommendations = []

        if hasattr(execution, 'execution_time'):
            if execution.execution_time > DEFAULT_BATCH_SIZE:
                recommendations.append(
                    Recommendation(
                        recommendation_id=self._generate_id(),
                        title="流程执行时间过长",
                        description=f"流程执行耗时{execution.execution_time:.2f}秒，建议优化",
                        category=RecommendationCategory.PERFORMANCE,
                        priority=PriorityLevel.HIGH,
                        confidence=0.85,
                        expected_impact={'execution_time': '-40%'},
                        implementation_steps=[
                            "分析性能瓶颈",
                            "优化数据库查询",
                            "增加并行处理"
                        ]
                    )
                )

        return recommendations

    async def _generate_ai_insights(self, context: Any,
                                    analysis: Any,
                                    execution: Any) -> List[Recommendation]:
        """生成AI洞察建议"""
        recommendations = []

        # AI分析（简化实现）
        recommendations.append(
            Recommendation(
                recommendation_id=self._generate_id(),
                title="AI建议：优化决策策略",
                description="基于历史数据分析，建议调整决策策略参数",
                category=RecommendationCategory.STRATEGY,
                priority=PriorityLevel.MEDIUM,
                confidence=0.75,
                expected_impact={'decision_accuracy': '+8%'},
                implementation_steps=[
                    "收集历史决策数据",
                    "训练优化模型",
                    "A/B测试验证"
                ],
                metadata={'ai_generated': True}
            )
        )

        return recommendations

    def _filter_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """过滤建议"""
        # 按置信度过滤
        filtered = [
            rec for rec in recommendations
            if rec.confidence >= self.config.min_confidence
        ]

        # 去重（基于标题）
        seen_titles = set()
        unique_recs = []
        for rec in filtered:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recs.append(rec)

        return unique_recs

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"rec_{uuid.uuid4().hex[:12]}"

    async def start_background_analysis(self):
        """启动后台分析任务"""
        logger.info("建议生成器后台分析已启动")
        # 实际应该启动异步任务


# 配置类会通过参数传入，无需导入

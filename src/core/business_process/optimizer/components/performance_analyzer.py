"""
性能分析组件

职责:
- 分析市场数据和流程性能
- 收集性能指标
- 生成性能报告和洞察
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque

from src.core.constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_TEST_TIMEOUT
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析结果"""
    timestamp: datetime
    metrics: Dict[str, Any]
    insights: List[str] = field(default_factory=list)
    score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    性能分析组件

    提供市场数据分析和流程性能分析能力
    """

    def __init__(self, config: 'AnalysisConfig'):
        """
        初始化性能分析器

        Args:
            config: 分析配置对象
        """
        self.config = config
        self._metrics_cache: Dict[str, Any] = {}
        self._analysis_history: deque = deque(
            maxlen=config.metrics_retention_days * 24  # 按小时计算
        )
        self._initialized = False

        logger.info("性能分析组件初始化完成")

    async def analyze_market_data(self, market_data: Dict[str, Any]) -> AnalysisResult:
        """
        分析市场数据

        Args:
            market_data: 市场数据字典

        Returns:
            AnalysisResult: 分析结果
        """
        timestamp = datetime.now()

        # 提取基础指标
        metrics = self._extract_market_metrics(market_data)

        # 深度分析（如果启用）
        insights = []
        if self.config.enable_deep_analysis:
            insights = await self._perform_deep_analysis(market_data, metrics)

        # 计算综合评分
        score = self._calculate_market_score(metrics)

        # 生成建议
        recommendations = self._generate_market_recommendations(metrics, insights)

        # 创建结果
        result = AnalysisResult(
            timestamp=timestamp,
            metrics=metrics,
            insights=insights,
            score=score,
            recommendations=recommendations,
            metadata={'data_source': 'market', 'symbols_count': len(market_data.get('symbols', []))}
        )

        # 保存到历史
        self._analysis_history.append(result)

        logger.debug(f"市场数据分析完成，评分: {score:.3f}")
        return result

    async def analyze_process_performance(self, process_id: str,
                                         context: Any) -> AnalysisResult:
        """
        分析流程性能

        Args:
            process_id: 流程ID
            context: 流程上下文

        Returns:
            AnalysisResult: 分析结果
        """
        timestamp = datetime.now()

        # 收集流程指标
        metrics = self._collect_process_metrics(process_id, context)

        # 分析洞察
        insights = self._analyze_process_insights(metrics)

        # 性能评分
        score = self._calculate_process_score(metrics)

        # 生成建议
        recommendations = self._generate_process_recommendations(metrics, insights)

        result = AnalysisResult(
            timestamp=timestamp,
            metrics=metrics,
            insights=insights,
            score=score,
            recommendations=recommendations,
            metadata={'process_id': process_id, 'analysis_type': 'process_performance'}
        )

        logger.debug(f"流程性能分析完成 {process_id}，评分: {score:.3f}")
        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标

        Returns:
            Dict: 性能指标字典
        """
        return self._metrics_cache.copy()

    def get_analysis_history(self, limit: int = DEFAULT_BATCH_SIZE) -> List[AnalysisResult]:
        """
        获取分析历史

        Args:
            limit: 返回数量限制

        Returns:
            List[AnalysisResult]: 历史分析结果列表
        """
        history_list = list(self._analysis_history)
        return history_list[-limit:] if limit > 0 else history_list

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'initialized': self._initialized,
            'cache_size': len(self._metrics_cache),
            'history_size': len(self._analysis_history),
            'config': {
                'analysis_interval': self.config.analysis_interval,
                'deep_analysis_enabled': self.config.enable_deep_analysis,
                'trend_prediction_enabled': self.config.enable_trend_prediction
            }
        }

    # 私有辅助方法
    def _extract_market_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取市场指标"""
        return {
            'symbols_count': len(market_data.get('symbols', [])),
            'data_quality': self._assess_data_quality(market_data),
            'market_trend': self._detect_market_trend(market_data),
            'volatility': self._calculate_volatility(market_data)
        }

    async def _perform_deep_analysis(self, market_data: Dict[str, Any],
                                     metrics: Dict[str, Any]) -> List[str]:
        """执行深度分析"""
        insights = []

        # 趋势分析
        if self.config.enable_trend_prediction:
            trend_insight = await self._analyze_trend(market_data)
            if trend_insight:
                insights.append(trend_insight)

        # 波动率分析
        volatility_insight = self._analyze_volatility(metrics.get('volatility', 0))
        if volatility_insight:
            insights.append(volatility_insight)

        return insights

    def _calculate_market_score(self, metrics: Dict[str, Any]) -> float:
        """计算市场评分"""
        # 简化评分算法
        score = 0.0
        score += metrics.get('data_quality', 0.5) * 0.3
        score += (1 - metrics.get('volatility', 0.5)) * 0.3
        score += metrics.get('trend_strength', 0.5) * 0.4
        return min(1.0, max(0.0, score))

    def _generate_market_recommendations(self, metrics: Dict[str, Any],
                                        insights: List[str]) -> List[str]:
        """生成市场建议"""
        recommendations = []

        # 基于指标生成建议
        if metrics.get('volatility', 0) > 0.7:
            recommendations.append("市场波动较大，建议降低仓位")

        if metrics.get('data_quality', 1.0) < 0.6:
            recommendations.append("数据质量不佳，建议谨慎决策")

        return recommendations

    def _collect_process_metrics(self, process_id: str, context: Any) -> Dict[str, Any]:
        """收集流程指标"""
        return {
            'process_id': process_id,
            'execution_time': 0.0,  # 需要从context获取
            'stages_completed': [],
            'success_rate': 1.0
        }

    def _analyze_process_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """分析流程洞察"""
        insights = []

        if metrics.get('execution_time', 0) > DEFAULT_TEST_TIMEOUT:
            insights.append("流程执行时间较长，建议优化")

        return insights

    def _calculate_process_score(self, metrics: Dict[str, Any]) -> float:
        """计算流程评分"""
        return metrics.get('success_rate', 0.5)

    def _generate_process_recommendations(self, metrics: Dict[str, Any],
                                         insights: List[str]) -> List[str]:
        """生成流程建议"""
        return insights  # 简化实现

    def _assess_data_quality(self, market_data: Dict[str, Any]) -> float:
        """评估数据质量"""
        return 0.8  # 简化实现，实际应该检查数据完整性

    def _detect_market_trend(self, market_data: Dict[str, Any]) -> str:
        """检测市场趋势"""
        return "neutral"  # 简化实现

    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """计算波动率"""
        return 0.3  # 简化实现

    async def _analyze_trend(self, market_data: Dict[str, Any]) -> Optional[str]:
        """分析趋势"""
        return "市场呈现横盘整理态势"  # 简化实现

    def _analyze_volatility(self, volatility: float) -> Optional[str]:
        """分析波动率"""
        if volatility > 0.7:
            return f"波动率较高({volatility:.2f})，风险增加"
        return None

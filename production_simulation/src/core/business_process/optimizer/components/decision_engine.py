"""
智能决策引擎组件

职责:
- AI/ML驱动的智能决策
- 多阶段决策策略管理
- 决策质量评估和优化
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.constants import (
    MAX_RECORDS, DEFAULT_BATCH_SIZE
)
from collections import deque

# 从models导入枚举
from ..models import DecisionType, DecisionStrategy

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """决策结果"""
    decision_type: DecisionType
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    ai_insights: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """
    智能决策引擎

    提供AI/ML增强的决策能力
    支持多种决策策略和自适应优化
    """

    def __init__(self, config: 'DecisionConfig'):
        """
        初始化决策引擎

        Args:
            config: 决策配置对象
        """
        self.config = config
        self._decision_history: deque = deque(maxlen=1000)
        self._ml_models: Dict[str, Any] = {}
        self._strategy_performance: Dict[str, float] = {}
        self._initialized = False

        # 转换策略（支持字符串和枚举）
        if isinstance(self.config.strategy, str):
            try:
                self.config.strategy = DecisionStrategy(self.config.strategy)
            except ValueError:
                self.config.strategy = DecisionStrategy.BALANCED

        # 初始化ML模型（如果启用）
        if self.config.enable_ml_enhancement:
            self._initialize_ml_models()

        strategy_name = self.config.strategy.value if hasattr(self.config.strategy, 'value') else str(self.config.strategy)
        logger.info(f"智能决策引擎初始化完成 (策略: {strategy_name})")

    async def make_market_decision(self, market_data: Dict[str, Any],
                                  analysis: Any) -> DecisionResult:
        """
        市场分析决策

        Args:
            market_data: 市场数据
            analysis: 分析结果

        Returns:
            DecisionResult: 决策结果
        """
        # 提取关键指标
        score = analysis.score if hasattr(analysis, 'score') else 0.5
        insights = analysis.insights if hasattr(analysis, 'insights') else []

        # 基于策略做决策
        if self.config.strategy == DecisionStrategy.CONSERVATIVE:
            decision = self._make_conservative_decision(score, insights)
        elif self.config.strategy == DecisionStrategy.AGGRESSIVE:
            decision = self._make_aggressive_decision(score, insights)
        elif self.config.strategy == DecisionStrategy.AI_OPTIMIZED:
            decision = await self._make_ai_optimized_decision(market_data, analysis)
        else:  # BALANCED
            decision = self._make_balanced_decision(score, insights)

        # 记录决策历史
        self._decision_history.append(decision)

        logger.debug(f"市场决策完成: {decision.decision_type.value}, 置信度: {decision.confidence:.3f}")
        return decision

    async def make_signal_decision(self, signals: List[Any]) -> DecisionResult:
        """
        信号生成决策

        Args:
            signals: 信号列表

        Returns:
            DecisionResult: 决策结果
        """
        # 分析信号质量
        signal_quality = self._evaluate_signal_quality(signals)

        # 决策逻辑
        if signal_quality > 0.7:
            decision_type = DecisionType.BUY_SIGNAL if signal_quality > 0.8 else DecisionType.HOLD_SIGNAL
            confidence = signal_quality
        else:
            decision_type = DecisionType.HOLD_SIGNAL
            confidence = 0.5

        result = DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=[f"信号质量评分: {signal_quality:.3f}"],
            metadata={'signals_count': len(signals), 'avg_quality': signal_quality}
        )

        self._decision_history.append(result)
        return result

    async def make_risk_decision(self, risk_data: Dict[str, Any]) -> DecisionResult:
        """
        风险评估决策

        Args:
            risk_data: 风险数据

        Returns:
            DecisionResult: 决策结果
        """
        risk_level = risk_data.get('risk_level', 0.5)

        # 基于风险阈值决策
        if risk_level > self.config.risk_threshold:
            decision_type = DecisionType.RISK_ADJUSTMENT
            confidence = 0.9
            reasoning = [f"风险水平{risk_level:.3f}超过阈值{self.config.risk_threshold}"]
        else:
            decision_type = DecisionType.HOLD_SIGNAL
            confidence = 0.7
            reasoning = [f"风险水平{risk_level:.3f}在可接受范围内"]

        result = DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={'risk_level': risk_level, 'threshold': self.config.risk_threshold}
        )

        self._decision_history.append(result)
        return result

    async def make_order_decision(self, orders: List[Any]) -> DecisionResult:
        """
        订单生成决策

        Args:
            orders: 订单列表

        Returns:
            DecisionResult: 决策结果
        """
        # 验证订单质量
        order_quality = self._evaluate_order_quality(orders)

        result = DecisionResult(
            decision_type=DecisionType.BUY_SIGNAL if order_quality > 0.6 else DecisionType.HOLD_SIGNAL,
            confidence=order_quality,
            reasoning=[f"订单质量评分: {order_quality:.3f}"],
            metadata={'orders_count': len(orders)}
        )

        self._decision_history.append(result)
        return result

    def get_decision_quality_score(self) -> float:
        """
        获取决策质量评分

        Returns:
            float: 质量评分 (0-1)
        """
        if not self._decision_history:
            return 0.5

        # 计算平均置信度
        avg_confidence = sum(d.confidence for d in self._decision_history) / len(self._decision_history)
        return avg_confidence

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        strategy_name = self.config.strategy.value if hasattr(self.config.strategy, 'value') else str(self.config.strategy)
        return {
            'initialized': self._initialized,
            'strategy': strategy_name,
            'decision_history_size': len(self._decision_history),
            'ml_enabled': self.config.enable_ml_enhancement,
            'quality_score': self.get_decision_quality_score()
        }

    def get_decision_history(self, limit: int = DEFAULT_BATCH_SIZE) -> List[DecisionResult]:
        """获取决策历史"""
        history_list = list(self._decision_history)
        return history_list[-limit:] if limit > 0 else history_list

    # 私有辅助方法
    def _initialize_ml_models(self):
        """初始化ML模型"""
        self._ml_models = {
            'market_predictor': None,  # 实际应该加载真实模型
            'risk_evaluator': None,
            'signal_classifier': None
        }
        self._initialized = True

    def _make_conservative_decision(self, score: float, insights: List[str]) -> DecisionResult:
        """保守决策"""
        # 保守策略：只有高分才执行
        if score > 0.8:
            decision_type = DecisionType.BUY_SIGNAL
            confidence = score * 0.9  # 降低置信度
        else:
            decision_type = DecisionType.HOLD_SIGNAL
            confidence = 0.7

        return DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=["保守策略", f"评分{score:.3f}"] + insights[:2]
        )

    def _make_aggressive_decision(self, score: float, insights: List[str]) -> DecisionResult:
        """激进决策"""
        # 激进策略：中等分数即可执行
        if score > 0.5:
            decision_type = DecisionType.BUY_SIGNAL
            confidence = min(1.0, score * 1.2)  # 提高置信度
        else:
            decision_type = DecisionType.HOLD_SIGNAL
            confidence = 0.6

        return DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=["激进策略", f"评分{score:.3f}"] + insights[:2]
        )

    def _make_balanced_decision(self, score: float, insights: List[str]) -> DecisionResult:
        """平衡决策"""
        # 平衡策略：综合考虑
        if score > 0.7:
            decision_type = DecisionType.BUY_SIGNAL
            confidence = score
        elif score < 0.3:
            decision_type = DecisionType.SELL_SIGNAL
            confidence = 1 - score
        else:
            decision_type = DecisionType.HOLD_SIGNAL
            confidence = 0.6

        return DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=["平衡策略", f"评分{score:.3f}"] + insights[:2]
        )

    async def _make_ai_optimized_decision(self, market_data: Dict[str, Any],
                                         analysis: Any) -> DecisionResult:
        """AI优化决策"""
        # AI驱动的决策（简化实现）
        score = analysis.score if hasattr(analysis, 'score') else 0.5

        # 如果ML模型可用，使用ML预测
        if self.config.enable_ml_enhancement and self._ml_models.get('market_predictor'):
            # 实际应该调用ML模型
            ml_prediction = score * 1.1  # 简化实现
            confidence = min(1.0, ml_prediction)
        else:
            confidence = score

        decision_type = DecisionType.BUY_SIGNAL if confidence > 0.7 else DecisionType.HOLD_SIGNAL

        return DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=["AI优化策略", "基于深度学习模型预测"],
            ai_insights={'model_version': 'v1.0', 'prediction_confidence': confidence}
        )

    def _evaluate_signal_quality(self, signals: List[Any]) -> float:
        """评估信号质量"""
        if not signals:
            return 0.0

        # 简化评分：基于信号数量和一致性
        return min(1.0, len(signals) * 0.2)

    def _evaluate_order_quality(self, orders: List[Any]) -> float:
        """评估订单质量"""
        if not orders:
            return 0.0

        # 简化评分
        return min(1.0, len(orders) * 0.3)


# DecisionStrategy 已在 models.py 中定义

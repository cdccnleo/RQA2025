"""动态权重调整器"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AdjustmentStrategy(Enum):
    """调整策略"""
    MARKET_STATE_BASED = "market_state_based"
    PERFORMANCE_BASED = "performance_based"
    RISK_BASED = "risk_based"
    MOMENTUM_BASED = "momentum_based"
    VOLATILITY_BASED = "volatility_based"


@dataclass
class WeightAdjustment:
    """权重调整结果"""
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    adjustment_factors: Dict[str, float]
    strategy: AdjustmentStrategy
    reason: str = ""
    confidence: float = 0.0  # 调整置信度 0-1


class DynamicWeightAdjuster:
    """动态权重调整器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_weights = config.get('base_weights', {
            'liquidity': 0.25,
            'volatility': 0.20,
            'fundamental': 0.30,
            'technical': 0.15,
            'sentiment': 0.10
        })

        # 调整参数
        self.adjustment_sensitivity = config.get('adjustment_sensitivity', 0.1)
        self.max_adjustment = config.get('max_adjustment', 0.3)
        self.min_weight = config.get('min_weight', 0.05)
        self.max_weight = config.get('max_weight', 0.5)

        # 市场状态权重映射
        self.market_state_weights = config.get('market_state_weights', {
            'bull': {
                'liquidity': 0.20,
                'volatility': 0.15,
                'fundamental': 0.35,
                'technical': 0.20,
                'sentiment': 0.10
            },
            'bear': {
                'liquidity': 0.30,
                'volatility': 0.25,
                'fundamental': 0.25,
                'technical': 0.10,
                'sentiment': 0.10
            },
            'sideways': {
                'liquidity': 0.25,
                'volatility': 0.20,
                'fundamental': 0.30,
                'technical': 0.15,
                'sentiment': 0.10
            },
            'high_volatility': {
                'liquidity': 0.35,
                'volatility': 0.30,
                'fundamental': 0.20,
                'technical': 0.10,
                'sentiment': 0.05
            },
            'low_volatility': {
                'liquidity': 0.20,
                'volatility': 0.10,
                'fundamental': 0.35,
                'technical': 0.20,
                'sentiment': 0.15
            }
        })

        # 性能跟踪
        self.performance_history = []
        self.weight_history = []
        self.max_history_length = config.get('max_history_length', 100)

        # 当前权重
        self.current_weights = self.base_weights.copy()

    def adjust_weights(self,
                       market_state: Optional[str] = None,
                       performance_metrics: Optional[Dict[str, float]] = None,
                       risk_metrics: Optional[Dict[str, float]] = None,
                       market_data: Optional[pd.DataFrame] = None) -> WeightAdjustment:
        """动态调整权重"""
        logger.info("开始动态权重调整")

        original_weights = self.current_weights.copy()
        adjustment_factors = {}

        # 1. 基于市场状态的调整
        if market_state:
            market_adjustment = self._adjust_for_market_state(market_state)
            adjustment_factors.update(market_adjustment)

        # 2. 基于性能的调整
        if performance_metrics:
            performance_adjustment = self._adjust_for_performance(performance_metrics)
            adjustment_factors.update(performance_adjustment)

        # 3. 基于风险的调整
        if risk_metrics:
            risk_adjustment = self._adjust_for_risk(risk_metrics)
            adjustment_factors.update(risk_adjustment)

        # 4. 基于市场数据的调整
        if market_data is not None:
            market_data_adjustment = self._adjust_for_market_data(market_data)
            adjustment_factors.update(market_data_adjustment)

        # 应用调整因子
        adjusted_weights = self._apply_adjustments(original_weights, adjustment_factors)

        # 记录历史
        self._record_adjustment(original_weights, adjusted_weights, adjustment_factors)

        # 确定调整策略
        strategy = self._determine_strategy(adjustment_factors)

        # 计算置信度
        confidence = self._calculate_confidence(adjustment_factors)

        return WeightAdjustment(
            original_weights=original_weights,
            adjusted_weights=adjusted_weights,
            adjustment_factors=adjustment_factors,
            strategy=strategy,
            reason=self._generate_reason(adjustment_factors),
            confidence=confidence
        )

    def _adjust_for_market_state(self, market_state: str) -> Dict[str, float]:
        """基于市场状态调整权重"""
        if market_state not in self.market_state_weights:
            return {}

        target_weights = self.market_state_weights[market_state]
        adjustment_factors = {}

        for factor, target_weight in target_weights.items():
            if factor in self.current_weights:
                current_weight = self.current_weights[factor]
                adjustment = (target_weight - current_weight) * self.adjustment_sensitivity
                adjustment_factors[f'market_state_{factor}'] = adjustment

        return adjustment_factors

    def _adjust_for_performance(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """基于性能指标调整权重"""
        adjustment_factors = {}

        # 计算各因子的相对性能
        total_performance = sum(performance_metrics.values())
        if total_performance == 0:
            return adjustment_factors

        for factor, performance in performance_metrics.items():
            if factor in self.current_weights:
                relative_performance = performance / total_performance
                current_weight = self.current_weights[factor]

                # 性能好的因子增加权重，性能差的减少权重
                adjustment = (relative_performance - current_weight) * self.adjustment_sensitivity
                adjustment_factors[f'performance_{factor}'] = adjustment

        return adjustment_factors

    def _adjust_for_risk(self, risk_metrics: Dict[str, float]) -> Dict[str, float]:
        """基于风险指标调整权重"""
        adjustment_factors = {}

        # 风险调整：高风险时增加防御性因子权重
        overall_risk = risk_metrics.get('overall_risk', 0.5)

        if overall_risk > 0.7:  # 高风险环境
            # 增加流动性和波动率权重
            adjustment_factors['risk_liquidity'] = 0.1
            adjustment_factors['risk_volatility'] = 0.1
            # 减少技术面和情绪权重
            adjustment_factors['risk_technical'] = -0.05
            adjustment_factors['risk_sentiment'] = -0.05
        elif overall_risk < 0.3:  # 低风险环境
            # 增加基本面和技术面权重
            adjustment_factors['risk_fundamental'] = 0.1
            adjustment_factors['risk_technical'] = 0.1
            # 减少波动率权重
            adjustment_factors['risk_volatility'] = -0.05

        return adjustment_factors

    def _adjust_for_market_data(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """基于市场数据调整权重"""
        adjustment_factors = {}

        if market_data.empty:
            return adjustment_factors

        # 计算市场特征
        if 'volatility' in market_data.columns:
            avg_volatility = market_data['volatility'].mean()
            if avg_volatility > 0.4:  # 高波动率环境
                adjustment_factors['data_volatility'] = 0.1
                adjustment_factors['data_liquidity'] = 0.05
            elif avg_volatility < 0.2:  # 低波动率环境
                adjustment_factors['data_volatility'] = -0.05
                adjustment_factors['data_fundamental'] = 0.1

        if 'turnover_rate' in market_data.columns:
            avg_turnover = market_data['turnover_rate'].mean()
            if avg_turnover < 0.02:  # 低流动性环境
                adjustment_factors['data_liquidity'] = 0.1
                adjustment_factors['data_volatility'] = 0.05

        return adjustment_factors

    def _apply_adjustments(self, original_weights: Dict[str, float],
                           adjustment_factors: Dict[str, float]) -> Dict[str, float]:
        """应用调整因子"""
        adjusted_weights = original_weights.copy()

        # 按因子分组调整
        factor_adjustments = {}
        for adjustment_key, adjustment_value in adjustment_factors.items():
            # 解析调整键，提取因子名称
            # 支持多种格式：market_state_liquidity -> liquidity, performance_fundamental -> fundamental
            if '_' in adjustment_key:
                parts = adjustment_key.split('_')
                # 查找最后一个部分是否匹配权重因子
                for i in range(len(parts) - 1, 0, -1):
                    factor = '_'.join(parts[i:])
                    if factor in adjusted_weights:
                        if factor not in factor_adjustments:
                            factor_adjustments[factor] = 0
                        factor_adjustments[factor] += adjustment_value
                        break

        # 应用调整
        for factor, adjustment in factor_adjustments.items():
            if factor in adjusted_weights:
                new_weight = adjusted_weights[factor] + adjustment
                # 限制权重范围
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                adjusted_weights[factor] = new_weight

        # 更新当前权重
        self.current_weights = adjusted_weights.copy()

        return adjusted_weights

    def _determine_strategy(self, adjustment_factors: Dict[str, float]) -> AdjustmentStrategy:
        """确定调整策略"""
        if any('market_state' in key for key in adjustment_factors.keys()):
            return AdjustmentStrategy.MARKET_STATE_BASED
        elif any('performance' in key for key in adjustment_factors.keys()):
            return AdjustmentStrategy.PERFORMANCE_BASED
        elif any('risk' in key for key in adjustment_factors.keys()):
            return AdjustmentStrategy.RISK_BASED
        elif any('data' in key for key in adjustment_factors.keys()):
            return AdjustmentStrategy.VOLATILITY_BASED
        else:
            return AdjustmentStrategy.MOMENTUM_BASED

    def _calculate_confidence(self, adjustment_factors: Dict[str, float]) -> float:
        """计算调整置信度"""
        if not adjustment_factors:
            return 0.0

        # 基于调整因子的数量和强度计算置信度
        total_adjustment = sum(abs(value) for value in adjustment_factors.values())
        num_factors = len(adjustment_factors)

        # 调整强度越高，置信度越高
        confidence = min(1.0, total_adjustment / self.max_adjustment)

        # 调整因子越多，置信度越高
        confidence *= min(1.0, num_factors / 5.0)

        return confidence

    def _generate_reason(self, adjustment_factors: Dict[str, float]) -> str:
        """生成调整原因"""
        reasons = []

        if any('market_state' in key for key in adjustment_factors.keys()):
            reasons.append("市场状态变化")

        if any('performance' in key for key in adjustment_factors.keys()):
            reasons.append("性能表现调整")

        if any('risk' in key for key in adjustment_factors.keys()):
            reasons.append("风险偏好调整")

        if any('data' in key for key in adjustment_factors.keys()):
            reasons.append("市场数据驱动")

        return " + ".join(reasons) if reasons else "无调整"

    def _record_adjustment(self, original_weights: Dict[str, float],
                           adjusted_weights: Dict[str, float],
                           adjustment_factors: Dict[str, float]):
        """记录调整历史"""
        record = {
            'timestamp': datetime.now(),
            'original_weights': original_weights,
            'adjusted_weights': adjusted_weights,
            'adjustment_factors': adjustment_factors
        }

        self.weight_history.append(record)

        # 限制历史记录长度
        if len(self.weight_history) > self.max_history_length:
            self.weight_history.pop(0)

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """获取权重调整历史"""
        return self.weight_history.copy()

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()

    def reset_weights(self):
        """重置权重到基础权重"""
        self.current_weights = self.base_weights.copy()
        self.weight_history.clear()
        self.performance_history.clear()

    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """获取调整统计信息"""
        if not self.weight_history:
            return {}

        # 计算调整频率
        total_adjustments = len(self.weight_history)

        # 计算各因子的平均调整幅度
        factor_adjustments = {}
        for record in self.weight_history:
            for factor, original_weight in record['original_weights'].items():
                adjusted_weight = record['adjusted_weights'].get(factor, original_weight)
                adjustment = adjusted_weight - original_weight

                if factor not in factor_adjustments:
                    factor_adjustments[factor] = []
                factor_adjustments[factor].append(adjustment)

        # 计算平均调整幅度
        avg_adjustments = {}
        for factor, adjustments in factor_adjustments.items():
            avg_adjustments[factor] = np.mean(adjustments)

        return {
            'total_adjustments': total_adjustments,
            'average_adjustments': avg_adjustments,
            'current_weights': self.current_weights,
            'base_weights': self.base_weights
        }

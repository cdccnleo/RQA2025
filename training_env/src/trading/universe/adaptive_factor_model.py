"""自适应多因子模型实现"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """市场状态枚举"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class AdaptiveMultiFactorModel:
    """自适应多因子模型"""

    def __init__(self, factors: List[str], config: Dict[str, Any]):
        self.factors = factors
        self.config = config
        self.factor_weights = {f: 1.0/len(factors) for f in factors}
        self.performance_history = {f: [] for f in factors}
        self.market_state_classifier = MarketStateClassifier()

        # 市场状态权重调整系数
        self.market_state_adjustments = {
            MarketState.BULL: {
                'momentum': 1.3,      # 牛市动量因子权重增加
                'value': 0.8,         # 价值因子权重降低
                'quality': 1.1,       # 质量因子权重略增
                'volatility': 0.7     # 波动率因子权重降低
            },
            MarketState.BEAR: {
                'momentum': 0.7,      # 熊市动量因子权重降低
                'value': 1.2,         # 价值因子权重增加
                'quality': 1.4,       # 质量因子权重增加
                'volatility': 1.3     # 波动率因子权重增加
            },
            MarketState.SIDEWAYS: {
                'momentum': 1.0,      # 震荡市保持原权重
                'value': 1.0,
                'quality': 1.0,
                'volatility': 1.0
            },
            MarketState.HIGH_VOLATILITY: {
                'momentum': 0.8,      # 高波动市场降低动量因子
                'value': 1.1,         # 增加价值因子
                'quality': 1.2,       # 增加质量因子
                'volatility': 1.4     # 大幅增加波动率因子
            },
            MarketState.LOW_VOLATILITY: {
                'momentum': 1.2,      # 低波动市场增加动量因子
                'value': 0.9,         # 降低价值因子
                'quality': 0.9,       # 降低质量因子
                'volatility': 0.6     # 大幅降低波动率因子
            }
        }

    def update_weights(self, market_state: MarketState, factor_performance: Dict[str, float]):
        """根据市场状态和因子表现更新权重"""
        logger.info(f"更新因子权重，市场状态: {market_state.value}")

        # 1. 基础权重更新（基于因子表现）
        total = sum(factor_performance.values())
        if total > 0:
            base_weights = {f: perf/total for f, perf in factor_performance.items()}
        else:
            base_weights = {f: 1.0/len(self.factors) for f in self.factors}

        # 2. 市场状态调整
        state_adjustments = self._get_state_adjustments(market_state)

        # 3. 应用调整
        for factor in self.factors:
            adjustment = state_adjustments.get(factor, 1.0)
            self.factor_weights[factor] = base_weights[factor] * adjustment

        # 4. 归一化
        total_weight = sum(self.factor_weights.values())
        self.factor_weights = {f: w/total_weight for f, w in self.factor_weights.items()}

        logger.info(f"更新后的因子权重: {self.factor_weights}")

    def _get_state_adjustments(self, market_state: MarketState) -> Dict[str, float]:
        """根据市场状态获取因子调整系数"""
        return self.market_state_adjustments.get(market_state, {
            factor: 1.0 for factor in self.factors
        })

    def generate_signal(self, factor_scores: Dict[str, float]) -> float:
        """生成综合信号"""
        if not factor_scores:
            return 0.0

        composite_signal = 0.0
        for factor, score in factor_scores.items():
            if factor in self.factor_weights:
                weight = self.factor_weights[factor]
                composite_signal += score * weight

        return composite_signal

    def get_factor_contribution(self, factor_scores: Dict[str, float]) -> Dict[str, float]:
        """获取各因子的贡献度"""
        contributions = {}
        for factor, score in factor_scores.items():
            if factor in self.factor_weights:
                weight = self.factor_weights[factor]
                contributions[factor] = score * weight

        return contributions

    def record_performance(self, factor: str, performance: float):
        """记录因子表现"""
        if factor in self.performance_history:
            self.performance_history[factor].append(performance)

            # 保留最近50个数据点
            if len(self.performance_history[factor]) > 50:
                self.performance_history[factor].pop(0)

    def get_factor_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取因子表现摘要"""
        summary = {}

        for factor, history in self.performance_history.items():
            if history:
                summary[factor] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'recent_5': np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
                }

        return summary


class MarketStateClassifier:
    """市场状态分类器"""

    def __init__(self):
        self.lookback_period = 20
        self.volatility_threshold = 0.02
        self.trend_threshold = 0.001

    def classify_market_state(self, market_data: pd.DataFrame) -> MarketState:
        """分类市场状态"""
        if market_data.empty or len(market_data) < self.lookback_period:
            return MarketState.SIDEWAYS

        # 计算收益率
        returns = market_data['close'].pct_change().dropna()

        # 计算波动率
        volatility = returns.rolling(self.lookback_period).std().iloc[-1]

        # 计算趋势
        trend = returns.rolling(self.lookback_period).mean().iloc[-1]

        # 分类逻辑
        if volatility > self.volatility_threshold:
            return MarketState.HIGH_VOLATILITY
        elif volatility < self.volatility_threshold * 0.5:
            return MarketState.LOW_VOLATILITY
        elif trend > self.trend_threshold:
            return MarketState.BULL
        elif trend < -self.trend_threshold:
            return MarketState.BEAR
        else:
            return MarketState.SIDEWAYS


class DynamicFactorSelector:
    """动态因子选择器"""

    def __init__(self, all_factors: List[str], config: Dict[str, Any]):
        self.all_factors = all_factors
        self.config = config
        self.active_factors = all_factors[:4]  # 初始激活4个因子
        self.factor_performance = {f: [] for f in all_factors}
        self.selection_history = []

        # 选择参数
        self.min_factor_count = config.get('min_factor_count', 3)
        self.max_factor_count = config.get('max_factor_count', 6)
        self.performance_window = config.get('performance_window', 20)

    def select_factors(self, market_data: pd.DataFrame) -> List[str]:
        """动态选择最优因子组合"""
        logger.info("开始动态因子选择")

        # 1. 计算各因子表现
        performances = self._calculate_factor_performances(market_data)

        # 2. 因子重要性排序
        sorted_factors = sorted(
            performances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 3. 确定最优因子数量
        optimal_count = self._determine_optimal_factor_count(market_data)

        # 4. 选择前N个因子
        selected_factors = [f[0] for f in sorted_factors[:optimal_count]]

        # 5. 更新激活因子
        old_active_factors = self.active_factors.copy()
        self.active_factors = selected_factors

        # 6. 记录选择历史
        self._record_selection(old_active_factors, selected_factors, performances)

        logger.info(f"因子选择结果: {selected_factors}")
        return selected_factors

    def _calculate_factor_performances(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算各因子表现"""
        performances = {}

        for factor in self.all_factors:
            # 这里应该实现具体的因子表现计算
            # 简化实现：使用随机数模拟
            if factor in self.factor_performance and self.factor_performance[factor]:
                # 使用历史表现的平均值
                performances[factor] = np.mean(self.factor_performance[factor])
            else:
                # 使用随机数作为初始表现
                performances[factor] = np.random.uniform(0.1, 0.9)

        return performances

    def _determine_optimal_factor_count(self, market_data: pd.DataFrame) -> int:
        """确定最优因子数量"""
        # 基于市场波动率调整因子数量
        if 'returns' in market_data.columns:
            volatility = market_data['returns'].std()
        else:
            volatility = 0.02  # 默认值

        if volatility < 0.015:  # 低波动
            return max(self.min_factor_count, 3)
        elif volatility < 0.03:  # 中等波动
            return max(self.min_factor_count, 4)
        else:  # 高波动
            return min(self.max_factor_count, 5)

    def _record_selection(self, old_factors: List[str], new_factors: List[str],
                          performances: Dict[str, float]):
        """记录因子选择历史"""
        selection_record = {
            'timestamp': datetime.now(),
            'old_factors': old_factors,
            'new_factors': new_factors,
            'performances': performances,
            'factor_count': len(new_factors)
        }

        self.selection_history.append(selection_record)

        # 保留最近50条记录
        if len(self.selection_history) > 50:
            self.selection_history.pop(0)

    def get_selection_statistics(self) -> Dict[str, any]:
        """获取选择统计信息"""
        if not self.selection_history:
            return {}

        recent_selections = self.selection_history[-10:]  # 最近10次选择

        factor_usage = {}
        for record in recent_selections:
            for factor in record['new_factors']:
                factor_usage[factor] = factor_usage.get(factor, 0) + 1

        return {
            'current_factors': self.active_factors,
            'factor_usage': factor_usage,
            'avg_factor_count': np.mean([r['factor_count'] for r in recent_selections]),
            'selection_frequency': len(recent_selections) / 10  # 平均选择频率
        }

    def update_factor_performance(self, factor: str, performance: float):
        """更新因子表现"""
        if factor in self.factor_performance:
            self.factor_performance[factor].append(performance)

            # 保留最近performance_window个数据点
            if len(self.factor_performance[factor]) > self.performance_window:
                self.factor_performance[factor].pop(0)


class FactorPerformanceTracker:
    """因子表现跟踪器"""

    def __init__(self, factors: List[str]):
        self.factors = factors
        self.performance_data = {factor: [] for factor in factors}
        self.ic_data = {factor: [] for factor in factors}  # 信息系数
        self.turnover_data = {factor: [] for factor in factors}  # 换手率

    def track_performance(self, factor: str, performance: float, ic: float = None,
                          turnover: float = None):
        """跟踪因子表现"""
        if factor in self.performance_data:
            self.performance_data[factor].append(performance)

            if ic is not None:
                self.ic_data[factor].append(ic)

            if turnover is not None:
                self.turnover_data[factor].append(turnover)

    def get_factor_rankings(self) -> List[Tuple[str, float]]:
        """获取因子排名"""
        rankings = []

        for factor in self.factors:
            if self.performance_data[factor]:
                # 使用最近表现的平均值
                recent_performance = np.mean(self.performance_data[factor][-10:])
                rankings.append((factor, recent_performance))

        # 按表现降序排序
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_factor_analysis(self) -> Dict[str, Dict[str, float]]:
        """获取因子分析"""
        analysis = {}

        for factor in self.factors:
            if self.performance_data[factor]:
                analysis[factor] = {
                    'mean_performance': np.mean(self.performance_data[factor]),
                    'std_performance': np.std(self.performance_data[factor]),
                    'recent_performance': np.mean(self.performance_data[factor][-5:]),
                    'mean_ic': np.mean(self.ic_data[factor]) if self.ic_data[factor] else 0,
                    'mean_turnover': np.mean(self.turnover_data[factor]) if self.turnover_data[factor] else 0
                }

        return analysis

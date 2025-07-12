import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum, auto
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class MarketState(Enum):
    """市场状态枚举"""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    VOLATILE = auto()
    SIDEWAYS = auto()

class RebalancingSignal(Enum):
    """调仓信号枚举"""
    INCREASE = auto()
    DECREASE = auto()
    HOLD = auto()
    EXIT = auto()

@dataclass
class PositionAdjustment:
    """仓位调整指令"""
    symbol: str
    direction: RebalancingSignal
    amount: float
    reason: str
    confidence: float

class MarketStateClassifier:
    """市场状态分类器"""

    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])
        self.state_mapping = {
            0: MarketState.TRENDING_UP,
            1: MarketState.TRENDING_DOWN,
            2: MarketState.VOLATILE,
            3: MarketState.SIDEWAYS
        }

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """训练分类器"""
        self.model.fit(features, labels)

    def predict(self, market_data: pd.DataFrame) -> MarketState:
        """预测市场状态"""
        features = self._extract_features(market_data)
        pred = self.model.predict([features])[0]
        return self.state_mapping[pred]

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """提取特征"""
        returns = data['close'].pct_change().dropna()
        return np.array([
            returns.mean(),  # 平均收益率
            returns.std(),   # 波动率
            returns.skew(),  # 偏度
            returns.kurt()   # 峰度
        ])

class MultiFactorModel:
    """多因子驱动模型"""

    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_weights = {f: 1.0/len(factors) for f in factors}

    def update_weights(self, factor_performance: Dict[str, float]):
        """更新因子权重"""
        total = sum(factor_performance.values())
        if total > 0:
            self.factor_weights = {
                f: perf/total
                for f, perf in factor_performance.items()
            }

    def generate_signal(self, factor_scores: Dict[str, float]) -> float:
        """生成综合信号"""
        return sum(
            score * self.factor_weights[factor]
            for factor, score in factor_scores.items()
        )

class AdaptiveRiskControl:
    """自适应风控模块"""

    def __init__(self, base_risk: float = 0.1):
        self.base_risk = base_risk
        self.current_risk = base_risk
        self.risk_adjustment = 1.0

    def adjust_for_volatility(self, volatility: float,
                            threshold: float = 0.2):
        """根据波动率调整风险"""
        if volatility > threshold:
            self.risk_adjustment = 0.7
        else:
            self.risk_adjustment = 1.0

    def adjust_for_drawdown(self, drawdown: float,
                          threshold: float = 0.15):
        """根据回撤调整风险"""
        if drawdown > threshold:
            self.risk_adjustment = 0.5

    def get_adjusted_risk(self) -> float:
        """获取调整后风险"""
        return self.base_risk * self.risk_adjustment

class IntelligentRebalancer:
    """智能调仓系统"""

    def __init__(self):
        self.state_classifier = MarketStateClassifier()
        self.factor_model = MultiFactorModel([
            'momentum', 'value', 'quality', 'volatility'
        ])
        self.risk_control = AdaptiveRiskControl()
        self.position_adjustments = []

    def analyze_market(self, market_data: pd.DataFrame):
        """分析市场状态"""
        state = self.state_classifier.predict(market_data)
        logger.info(f"当前市场状态: {state.name}")
        return state

    def evaluate_factors(self, factor_data: Dict[str, pd.DataFrame]):
        """评估因子表现"""
        factor_perf = {
            name: self._calculate_factor_performance(data)
            for name, data in factor_data.items()
        }
        self.factor_model.update_weights(factor_perf)

    def _calculate_factor_performance(self, data: pd.DataFrame) -> float:
        """计算因子表现"""
        returns = data['return'].mean()
        sharpe = returns / data['return'].std()
        return max(0, sharpe)

    def generate_rebalancing_signals(self,
                                   portfolio: Dict[str, float],
                                   market_state: MarketState,
                                   factor_scores: Dict[str, float]) -> List[PositionAdjustment]:
        """生成调仓信号"""
        signals = []
        composite_score = self.factor_model.generate_signal(factor_scores)

        for symbol, weight in portfolio.items():
            signal, confidence = self._determine_signal(
                symbol, weight, market_state, composite_score
            )
            adjustment = PositionAdjustment(
                symbol=symbol,
                direction=signal,
                amount=self._calculate_amount(signal, weight),
                reason=f"{market_state.name}+{composite_score:.2f}",
                confidence=confidence
            )
            signals.append(adjustment)

        self.position_adjustments = signals
        return signals

    def _determine_signal(self, symbol: str,
                         weight: float,
                         state: MarketState,
                         score: float) -> Tuple[RebalancingSignal, float]:
        """确定调仓信号"""
        if state == MarketState.TRENDING_UP and score > 0.7:
            return (RebalancingSignal.INCREASE, score)
        elif state == MarketState.TRENDING_DOWN and score < -0.5:
            return (RebalancingSignal.EXIT, -score)
        elif weight > 0.2 and score < -0.3:
            return (RebalancingSignal.DECREASE, -score)
        else:
            return (RebalancingSignal.HOLD, 0.5)

    def _calculate_amount(self, signal: RebalancingSignal,
                        current_weight: float) -> float:
        """计算调整量"""
        risk = self.risk_control.get_adjusted_risk()
        if signal == RebalancingSignal.INCREASE:
            return min(0.1, risk * 0.5)
        elif signal == RebalancingSignal.DECREASE:
            return min(current_weight * 0.5, risk * 0.3)
        elif signal == RebalancingSignal.EXIT:
            return current_weight
        else:
            return 0.0

    def apply_risk_adjustments(self, volatility: float,
                             drawdown: float):
        """应用风险调整"""
        self.risk_control.adjust_for_volatility(volatility)
        self.risk_control.adjust_for_drawdown(drawdown)
        logger.info(f"风险水平调整为: {self.risk_control.get_adjusted_risk():.1%}")

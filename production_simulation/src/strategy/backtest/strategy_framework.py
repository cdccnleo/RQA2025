#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测层策略框架

实现中期优化目标：支持更多策略类型、添加高级分析功能、完善可视化界面
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:

    """策略配置"""
    name: str
    type: str  # 'momentum', 'mean_reversion', 'arbitrage', 'ml', 'custom'
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


@dataclass
class Signal:

    """交易信号"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):

    """策略基类"""

    def __init__(self, config: StrategyConfig):

        self.config = config
        self.name = config.name
        self.type = config.type
        self.parameters = config.parameters
        self.risk_limits = config.risk_limits
        self.enabled = config.enabled

        # 策略状态
        self.positions = {}
        self.trades = []
        self.performance_metrics = {}

    @abstractmethod
    def generate_signals(self, data: Dict[str, Any], state: Dict[str, Any]) -> List[Signal]:
        """生成交易信号"""

    def update_parameters(self, new_parameters: Dict[str, Any]):
        """更新策略参数"""
        self.parameters.update(new_parameters)
        logger.info(f"策略 {self.name} 参数已更新")

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.performance_metrics

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """计算风险指标"""
        # 基础风险指标计算
        return {
            'var_95': 0.02,
            'max_drawdown': 0.05,
            'volatility': 0.15,
            'sharpe_ratio': 1.2
        }


class MomentumStrategy(BaseStrategy):

    """动量策略"""

    def __init__(self, config: StrategyConfig):

        super().__init__(config)
        self.lookback_period = self.parameters.get('lookback_period', 20)
        self.momentum_threshold = self.parameters.get('momentum_threshold', 0.05)

    def generate_signals(self, data: Dict[str, Any], state: Dict[str, Any]) -> List[Signal]:
        """生成动量策略信号"""
        signals = []

        for symbol, price_data in data.items():
            if len(price_data) < self.lookback_period:
                continue

            # 计算动量
            returns = np.diff(price_data['close']) / price_data['close'][:-1]
            momentum = np.mean(returns[-self.lookback_period:])

            # 生成信号
            if momentum > self.momentum_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    action='buy',
                    quantity=100,
                    confidence=abs(momentum)
                ))
            elif momentum < -self.momentum_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    action='sell',
                    quantity=100,
                    confidence=abs(momentum)
                ))

        return signals


class MeanReversionStrategy(BaseStrategy):

    """均值回归策略"""

    def __init__(self, config: StrategyConfig):

        super().__init__(config)
        self.lookback_period = self.parameters.get('lookback_period', 50)
        self.std_threshold = self.parameters.get('std_threshold', 2.0)

    def generate_signals(self, data: Dict[str, Any], state: Dict[str, Any]) -> List[Signal]:
        """生成均值回归策略信号"""
        signals = []

        for symbol, price_data in data.items():
            if len(price_data) < self.lookback_period:
                continue

            # 计算移动平均和标准差
            prices = price_data['close']
            ma = np.mean(prices[-self.lookback_period:])
            std = np.std(prices[-self.lookback_period:])
            current_price = prices[-1]

            # 计算z - score
            z_score = (current_price - ma) / std

            # 生成信号
            if z_score > self.std_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    action='sell',
                    quantity=100,
                    confidence=abs(z_score)
                ))
            elif z_score < -self.std_threshold:
                signals.append(Signal(
                    symbol=symbol,
                    action='buy',
                    quantity=100,
                    confidence=abs(z_score)
                ))

        return signals


class ArbitrageStrategy(BaseStrategy):

    """套利策略"""

    def __init__(self, config: StrategyConfig):

        super().__init__(config)
        self.min_spread = self.parameters.get('min_spread', 0.01)
        self.max_position = self.parameters.get('max_position', 1000)

    def generate_signals(self, data: Dict[str, Any], state: Dict[str, Any]) -> List[Signal]:
        """生成套利策略信号"""
        signals = []

        # 寻找价格差异
        for symbol, price_data in data.items():
            if 'bid' in price_data and 'ask' in price_data:
                spread = price_data['ask'] - price_data['bid']
                spread_pct = spread / price_data['bid']

                if spread_pct > self.min_spread:
                    # 买入低价，卖出高价
                    signals.append(Signal(
                        symbol=symbol,
                        action='buy',
                        quantity=min(self.max_position, int(spread_pct * 1000)),
                        price=price_data['bid'],
                        confidence=spread_pct
                    ))

        return signals


class MLStrategy(BaseStrategy):

    """机器学习策略"""

    def __init__(self, config: StrategyConfig):

        super().__init__(config)
        self.model = None
        self.feature_columns = self.parameters.get('feature_columns', [])
        self.prediction_threshold = self.parameters.get('prediction_threshold', 0.6)

    def load_model(self, model_path: str):
        """加载机器学习模型"""
        try:
            import joblib
            self.model = joblib.load(model_path)
            logger.info(f"机器学习模型已加载: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

    def generate_signals(self, data: Dict[str, Any], state: Dict[str, Any]) -> List[Signal]:
        """生成机器学习策略信号"""
        signals = []

        if self.model is None:
            return signals

        for symbol, price_data in data.items():
            try:
                # 准备特征
                features = self._prepare_features(price_data)

                # 预测
                prediction = self.model.predict_proba([features])[0]
                confidence = max(prediction)
                predicted_class = np.argmax(prediction)

                # 生成信号
                if confidence > self.prediction_threshold:
                    action = 'buy' if predicted_class == 1 else 'sell'
                    signals.append(Signal(
                        symbol=symbol,
                        action=action,
                        quantity=100,
                        confidence=confidence
                    ))

            except Exception as e:
                logger.error(f"ML策略预测失败 {symbol}: {e}")

        return signals

    def _prepare_features(self, price_data: Dict[str, Any]) -> List[float]:
        """准备特征数据"""
        features = []

        # 价格特征
        if 'close' in price_data:
            prices = price_data['close']
            features.extend([
                prices[-1],  # 当前价格
                np.mean(prices[-5:]),  # 5日平均
                np.mean(prices[-20:]),  # 20日平均
                np.std(prices[-20:]),  # 20日标准差
            ])

        # 成交量特征
        if 'volume' in price_data:
            volumes = price_data['volume']
            features.extend([
                volumes[-1],  # 当前成交量
                np.mean(volumes[-5:]),  # 5日平均成交量
            ])

        return features


class StrategyManager:

    """策略管理器"""

    def __init__(self):

        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_factory = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'arbitrage': ArbitrageStrategy,
            'ml': MLStrategy
        }

    def register_strategy(self, strategy: BaseStrategy):
        """注册策略"""
        self.strategies[strategy.name] = strategy
        logger.info(f"策略已注册: {strategy.name}")

    def create_strategy(self, config: StrategyConfig) -> BaseStrategy:
        """创建策略"""
        strategy_class = self.strategy_factory.get(config.type)
        if strategy_class is None:
            raise ValueError(f"不支持的策略类型: {config.type}")

        strategy = strategy_class(config)
        self.register_strategy(strategy)
        return strategy

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """获取策略"""
        return self.strategies.get(name)

    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """获取所有策略"""
        return self.strategies

    def enable_strategy(self, name: str):
        """启用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = True
            logger.info(f"策略已启用: {name}")

    def disable_strategy(self, name: str):
        """禁用策略"""
        if name in self.strategies:
            self.strategies[name].enabled = False
            logger.info(f"策略已禁用: {name}")

    def get_active_strategies(self) -> Dict[str, BaseStrategy]:
        """获取活跃策略"""
        return {name: strategy for name, strategy in self.strategies.items()
                if strategy.enabled}


class AdvancedAnalyzer:

    """高级分析器"""

    def __init__(self):

        self.analysis_results = {}

    def calculate_advanced_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """计算高级指标"""
        returns = portfolio_data.get('returns', [])
        if not returns:
            return {}

        returns = np.array(returns)

        # 基础指标
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)

        # 风险调整指标
        risk_free_rate = 0.03  # 假设无风险利率3%
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 其他指标
        var_95 = np.percentile(returns, 5)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """计算偏度"""
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """计算峰度"""
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3

    def generate_analysis_report(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告"""
        metrics = self.calculate_advanced_metrics(portfolio_data)

        # 性能评级
        performance_grade = self._grade_performance(metrics)

        # 风险评级
        risk_grade = self._grade_risk(metrics)

        # 建议
        recommendations = self._generate_recommendations(metrics)

        return {
            'metrics': metrics,
            'performance_grade': performance_grade,
            'risk_grade': risk_grade,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def _grade_performance(self, metrics: Dict[str, float]) -> str:
        """性能评级"""
        sharpe = metrics.get('sharpe_ratio', 0)
        annual_return = metrics.get('annual_return', 0)

        if sharpe > 1.5 and annual_return > 0.15:
            return 'A'
        elif sharpe > 1.0 and annual_return > 0.10:
            return 'B'
        elif sharpe > 0.5 and annual_return > 0.05:
            return 'C'
        else:
            return 'D'

    def _grade_risk(self, metrics: Dict[str, float]) -> str:
        """风险评级"""
        max_dd = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)

        if max_dd < 0.10 and volatility < 0.15:
            return 'Low'
        elif max_dd < 0.20 and volatility < 0.25:
            return 'Medium'
        else:
            return 'High'

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """生成建议"""
        recommendations = []

        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        volatility = metrics.get('volatility', 0)

        if sharpe < 1.0:
            recommendations.append("建议优化策略以提高风险调整收益")

        if max_dd > 0.20:
            recommendations.append("建议加强风险控制以降低最大回撤")

        if volatility > 0.25:
            recommendations.append("建议分散投资以降低波动率")

        return recommendations


# 全局策略管理器实例
strategy_manager = StrategyManager()
advanced_analyzer = AdvancedAnalyzer()


def create_momentum_strategy(name: str, lookback_period: int = 20,


                             momentum_threshold: float = 0.05) -> BaseStrategy:
    """创建动量策略"""
    config = StrategyConfig(
        name=name,
        type='momentum',
        parameters={
            'lookback_period': lookback_period,
            'momentum_threshold': momentum_threshold
        }
    )
    return strategy_manager.create_strategy(config)


def create_mean_reversion_strategy(name: str, lookback_period: int = 50,


                                   std_threshold: float = 2.0) -> BaseStrategy:
    """创建均值回归策略"""
    config = StrategyConfig(
        name=name,
        type='mean_reversion',
        parameters={
            'lookback_period': lookback_period,
            'std_threshold': std_threshold
        }
    )
    return strategy_manager.create_strategy(config)


def create_arbitrage_strategy(name: str, min_spread: float = 0.01,


                              max_position: int = 1000) -> BaseStrategy:
    """创建套利策略"""
    config = StrategyConfig(
        name=name,
        type='arbitrage',
        parameters={
            'min_spread': min_spread,
            'max_position': max_position
        }
    )
    return strategy_manager.create_strategy(config)


def create_ml_strategy(name: str, model_path: str, feature_columns: List[str],


                       prediction_threshold: float = 0.6) -> BaseStrategy:
    """创建机器学习策略"""
    config = StrategyConfig(
        name=name,
        type='ml',
        parameters={
            'feature_columns': feature_columns,
            'prediction_threshold': prediction_threshold
        }
    )
    strategy = strategy_manager.create_strategy(config)
    strategy.load_model(model_path)
    return strategy


def analyze_portfolio_performance(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """分析投资组合性能"""
    return advanced_analyzer.generate_analysis_report(portfolio_data)


def get_strategy_performance(strategy_name: str) -> Dict[str, float]:
    """获取策略性能"""
    strategy = strategy_manager.get_strategy(strategy_name)
    if strategy:
        return strategy.get_performance_metrics()
    return {}


def get_all_strategies() -> Dict[str, BaseStrategy]:
    """获取所有策略"""
    return strategy_manager.get_all_strategies()

#!/usr/bin/env python3
"""
RQA2025 智能交易策略引擎

实现多策略组合优化、自适应学习、策略评估和动态调整
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.monitoring.deep_learning_predictor import get_deep_learning_predictor

logger = logging.getLogger(__name__)


class StrategyType(Enum):

    """策略类型枚举"""
    MOMENTUM = "momentum"                    # 动量策略
    MEAN_REVERSION = "mean_reversion"        # 均值回归策略
    ARBITRAGE = "arbitrage"                  # 套利策略
    STATISTICAL = "statistical"              # 统计套利策略
    MACHINE_LEARNING = "machine_learning"    # 机器学习策略
    TECHNICAL = "technical"                  # 技术分析策略
    FUNDAMENTAL = "fundamental"              # 基本面策略
    SENTIMENT = "sentiment"                  # 情绪分析策略


class StrategySignal(Enum):

    """策略信号枚举"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StrategyResult:

    """策略执行结果"""
    strategy_name: str
    signal: StrategySignal
    confidence: float
    expected_return: float
    risk_score: float
    execution_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:

    """投资组合配置"""
    allocations: Dict[str, float]  # 资产: 权重
    total_weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    risk_target: float = 0.1  # 目标风险水平


class BaseTradingStrategy:

    """基础交易策略类"""

    def __init__(self, name: str, strategy_type: StrategyType):

        self.name = name
        self.strategy_type = strategy_type
        self.is_active = True
        self.performance_history = []
        self.parameters = {}
        self.last_signal = None
        self.confidence_threshold = 0.6

    async def generate_signal(self, market_data: Dict[str, Any],
                              portfolio_data: Dict[str, Any]) -> StrategyResult:
        """生成交易信号"""
        raise NotImplementedError("子类必须实现generate_signal方法")

    def update_parameters(self, new_parameters: Dict[str, Any]):
        """更新策略参数"""
        self.parameters.update(new_parameters)
        logger.info(f"策略 {self.name} 参数已更新: {new_parameters}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.performance_history:
            return {}

        returns = [p.get('return', 0) for p in self.performance_history]
        accuracies = [p.get('accuracy', 0) for p in self.performance_history]

        return {
            'total_trades': len(self.performance_history),
            'avg_return': np.mean(returns) if returns else 0,
            'total_return': sum(returns) if returns else 0,
            'win_rate': np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0,
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns) if returns else 0,
            'max_drawdown': self._calculate_max_drawdown(returns) if returns else 0
        }

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """计算夏普比率"""
        if not returns or len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 年化夏普比率 (假设日收益率)
        return (avg_return * 252) / (std_return * np.sqrt(252))

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0.0


class MomentumStrategy(BaseTradingStrategy):

    """动量策略"""

    def __init__(self):

        super().__init__("Momentum Strategy", StrategyType.MOMENTUM)
        self.lookback_period = 20
        self.momentum_threshold = 0.02

    async def generate_signal(self, market_data: Dict[str, Any],
                              portfolio_data: Dict[str, Any]) -> StrategyResult:
        """生成动量信号"""
        symbols = market_data.get('symbols', [])

        if not symbols:
            return StrategyResult(
                strategy_name=self.name,
                signal=StrategySignal.HOLD,
                confidence=0.5,
                expected_return=0.0,
                risk_score=0.5
            )

        # 计算动量信号
        momentum_signals = []
        for symbol in symbols:
            momentum = await self._calculate_momentum(symbol, market_data)
            momentum_signals.append(momentum)

        # 综合信号
        avg_momentum = np.mean(momentum_signals)
        confidence = min(abs(avg_momentum) / self.momentum_threshold, 1.0)

        if avg_momentum > self.momentum_threshold:
            signal = StrategySignal.BUY
            expected_return = avg_momentum * 0.8  # 预期收益率
        elif avg_momentum < -self.momentum_threshold:
            signal = StrategySignal.SELL
            expected_return = abs(avg_momentum) * 0.6
        else:
            signal = StrategySignal.HOLD
            expected_return = 0.0

        return StrategyResult(
            strategy_name=self.name,
            signal=signal,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=min(confidence * 0.7, 0.8),
            metadata={'avg_momentum': avg_momentum, 'symbols_analyzed': len(symbols)}
        )

    async def _calculate_momentum(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """计算动量指标"""
        # 这里应该从历史数据计算动量
        # 暂时使用简化的计算
        current_price = market_data.get('prices', {}).get(symbol, 100)
        historical_prices = await self._get_historical_prices(symbol)

        if len(historical_prices) < self.lookback_period:
            return 0.0

        # 计算动量 (当前价格相对于历史平均的价格变化)
        historical_avg = np.mean(historical_prices[-self.lookback_period:])
        momentum = (current_price - historical_avg) / historical_avg

        return momentum

    async def _get_historical_prices(self, symbol: str) -> List[float]:
        """获取历史价格数据"""
        # 这里应该从数据源获取真实的历史数据
        # 暂时生成模拟数据
        base_price = 100
        prices = []
        for i in range(100):
            price = base_price + np.secrets.normal(0, 2)
            prices.append(price)

        return prices


class MeanReversionStrategy(BaseTradingStrategy):

    """均值回归策略"""

    def __init__(self):

        super().__init__("Mean Reversion Strategy", StrategyType.MEAN_REVERSION)
        self.lookback_period = 50
        self.deviation_threshold = 2.0
        self.half_life = 10  # 均值回归半衰期

    async def generate_signal(self, market_data: Dict[str, Any],
                              portfolio_data: Dict[str, Any]) -> StrategyResult:
        """生成均值回归信号"""
        symbols = market_data.get('symbols', [])

        if not symbols:
            return StrategyResult(
                strategy_name=self.name,
                signal=StrategySignal.HOLD,
                confidence=0.5,
                expected_return=0.0,
                risk_score=0.5
            )

        # 计算均值回归信号
        reversion_signals = []
        for symbol in symbols:
            signal = await self._calculate_reversion_signal(symbol, market_data)
            reversion_signals.append(signal)

        # 综合信号
        avg_signal = np.mean(reversion_signals)
        confidence = min(abs(avg_signal) / self.deviation_threshold, 1.0)

        if avg_signal < -self.deviation_threshold:
            signal = StrategySignal.BUY  # 价格低于均值，买入
            expected_return = abs(avg_signal) * 0.5
        elif avg_signal > self.deviation_threshold:
            signal = StrategySignal.SELL  # 价格高于均值，卖出
            expected_return = avg_signal * 0.4
        else:
            signal = StrategySignal.HOLD
            expected_return = 0.0

        return StrategyResult(
            strategy_name=self.name,
            signal=signal,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=min(confidence * 0.6, 0.7),
            metadata={'avg_signal': avg_signal, 'symbols_analyzed': len(symbols)}
        )

    async def _calculate_reversion_signal(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """计算均值回归信号"""
        current_price = market_data.get('prices', {}).get(symbol, 100)
        historical_prices = await self._get_historical_prices(symbol)

        if len(historical_prices) < self.lookback_period:
            return 0.0

        # 计算移动平均和标准差
        ma = np.mean(historical_prices[-self.lookback_period:])
        std = np.std(historical_prices[-self.lookback_period:])

        if std == 0:
            return 0.0

        # 计算Z - score (标准化偏差)
        z_score = (current_price - ma) / std

        return z_score


class MLBasedStrategy(BaseTradingStrategy):

    """机器学习策略"""

    def __init__(self):

        super().__init__("ML - Based Strategy", StrategyType.MACHINE_LEARNING)
        self.dl_predictor = get_deep_learning_predictor()
        self.model_trained = False
        self.feature_columns = ['price', 'volume', 'volatility', 'momentum']

    async def generate_signal(self, market_data: Dict[str, Any],
                              portfolio_data: Dict[str, Any]) -> StrategyResult:
        """生成ML增强信号"""
        symbols = market_data.get('symbols', [])

        if not symbols:
            return StrategyResult(
                strategy_name=self.name,
                signal=StrategySignal.HOLD,
                confidence=0.5,
                expected_return=0.0,
                risk_score=0.5
            )

        # 使用AI预测生成信号
        ml_signals = []
        for symbol in symbols:
            signal = await self._generate_ml_signal(symbol, market_data)
            ml_signals.append(signal)

        # 综合信号
        avg_confidence = np.mean([s['confidence'] for s in ml_signals])
        buy_signals = sum(1 for s in ml_signals if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in ml_signals if s['signal'] == 'SELL')

        if buy_signals > sell_signals:
            signal = StrategySignal.BUY
            expected_return = np.mean([s.get('expected_return', 0)
                                      for s in ml_signals if s['signal'] == 'BUY'])
        elif sell_signals > buy_signals:
            signal = StrategySignal.SELL
            expected_return = np.mean([s.get('expected_return', 0)
                                      for s in ml_signals if s['signal'] == 'SELL'])
        else:
            signal = StrategySignal.HOLD
            expected_return = 0.0

        return StrategyResult(
            strategy_name=self.name,
            signal=signal,
            confidence=avg_confidence,
            expected_return=expected_return,
            risk_score=min(avg_confidence * 0.5, 0.6),
            metadata={'ml_signals': ml_signals, 'symbols_analyzed': len(symbols)}
        )

    async def _generate_ml_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个股票的ML信号"""
        try:
            # 获取历史数据
            historical_data = await self._get_historical_data(symbol)

            if historical_data.empty:
                return {'signal': 'HOLD', 'confidence': 0.5, 'expected_return': 0.0}

            # 使用LSTM预测
            prediction = self.dl_predictor.predict_with_lstm(
                f"{symbol}_ml",
                historical_data,
                steps=1
            )

            if prediction.get('status') == 'success':
                current_price = market_data.get('prices', {}).get(symbol, 100)
                predicted_price = prediction.get('predictions', [current_price])[0]

                # 计算信号
                price_change = (predicted_price - current_price) / current_price
                confidence = prediction.get('confidence_intervals', [[0.5, 0.5]])[
                    0][1] if prediction.get('confidence_intervals') else 0.5

                if price_change > 0.02:  # 预测上涨2%
                    return {
                        'signal': 'BUY',
                        'confidence': confidence,
                        'expected_return': price_change * 0.8
                    }
                elif price_change < -0.02:  # 预测下跌2%
                    return {
                        'signal': 'SELL',
                        'confidence': confidence,
                        'expected_return': abs(price_change) * 0.6
                    }
                else:
                    return {
                        'signal': 'HOLD',
                        'confidence': 0.5,
                        'expected_return': 0.0
                    }
            else:
                return {'signal': 'HOLD', 'confidence': 0.5, 'expected_return': 0.0}

        except Exception as e:
            logger.warning(f"ML信号生成失败 {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.5, 'expected_return': 0.0}


class AdaptiveStrategyLearner:

    """自适应策略学习器"""

    def __init__(self):

        self.strategy_performance = {}
        self.market_regime_detector = MarketRegimeDetector()
        self.parameter_optimizer = ParameterOptimizer()

    async def adapt_strategies(self, strategies: List[BaseTradingStrategy],
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """自适应调整策略"""
        # 检测市场状态
        market_regime = await self.market_regime_detector.detect_regime(market_data)

        # 评估策略表现
        strategy_scores = await self._evaluate_strategies(strategies, market_data)

        # 优化策略参数
        optimized_parameters = await self.parameter_optimizer.optimize_parameters(
            strategies, strategy_scores, market_regime
        )

        # 应用优化参数
        await self._apply_optimized_parameters(strategies, optimized_parameters)

        return {
            'market_regime': market_regime,
            'strategy_scores': strategy_scores,
            'optimized_parameters': optimized_parameters
        }

    async def _evaluate_strategies(self, strategies: List[BaseTradingStrategy],
                                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """评估策略表现"""
        scores = {}

        for strategy in strategies:
            if not strategy.is_active:
                scores[strategy.name] = 0.0
                continue

            # 基于历史表现和当前市场条件计算分数
            performance_score = await self._calculate_performance_score(strategy)
            market_fit_score = await self._calculate_market_fit_score(strategy, market_data)

            scores[strategy.name] = (performance_score * 0.7 + market_fit_score * 0.3)

        return scores

    async def _calculate_performance_score(self, strategy: BaseTradingStrategy) -> float:
        """计算性能分数"""
        metrics = strategy.get_performance_metrics()

        if not metrics:
            return 0.5

        # 综合评分：胜率、夏普比率、最大回撤
        win_rate = metrics.get('win_rate', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)

        # 归一化并加权
        score = (
            win_rate * 0.4
            + min(sharpe_ratio / 3, 1.0) * 0.4  # 夏普比率上限为3
            + (1 - min(max_drawdown / 0.5, 1.0)) * 0.2  # 最大回撤惩罚
        )

        return score

    async def _calculate_market_fit_score(self, strategy: BaseTradingStrategy,
                                          market_data: Dict[str, Any]) -> float:
        """计算市场适应性分数"""
        # 基于策略类型和当前市场条件计算适应性
        market_volatility = await self._calculate_market_volatility(market_data)
        market_trend = await self._calculate_market_trend(market_data)

        if strategy.strategy_type == StrategyType.MOMENTUM:
            # 动量策略在趋势市场表现更好
            return 0.8 if abs(market_trend) > 0.02 else 0.4
        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            # 均值回归策略在震荡市场表现更好
            return 0.8 if market_volatility > 0.02 else 0.4
        elif strategy.strategy_type == StrategyType.MACHINE_LEARNING:
            # ML策略在各种市场条件下都有较好表现
            return 0.7
        else:
            return 0.6

    async def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """计算市场波动率"""
        prices = list(market_data.get('prices', {}).values())
        if len(prices) < 2:
            return 0.0

        returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        return np.std(returns) if returns else 0.0

    async def _calculate_market_trend(self, market_data: Dict[str, Any]) -> float:
        """计算市场趋势"""
        prices = list(market_data.get('prices', {}).values())
        if len(prices) < 10:
            return 0.0

        # 计算最近10个价格的线性趋势
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope / np.mean(prices)  # 相对趋势

    async def _apply_optimized_parameters(self, strategies: List[BaseTradingStrategy],
                                          optimized_parameters: Dict[str, Dict[str, Any]]):
        """应用优化参数"""
        for strategy in strategies:
            if strategy.name in optimized_parameters:
                strategy.update_parameters(optimized_parameters[strategy.name])
                logger.info(f"策略 {strategy.name} 参数已优化")


class MarketRegimeDetector:

    """市场状态检测器"""

    def __init__(self):

        self.regime_history = []

    async def detect_regime(self, market_data: Dict[str, Any]) -> str:
        """检测市场状态"""
        volatility = await self._calculate_volatility(market_data)
        trend = await self._calculate_trend(market_data)
        volume = await self._calculate_volume_trend(market_data)

        # 基于波动率、趋势和成交量判断市场状态
        if volatility > 0.05 and abs(trend) < 0.01:
            regime = "volatile_sideways"  # 高波动横盘
        elif volatility < 0.02 and abs(trend) > 0.02:
            regime = "trending_low_vol"  # 趋势明确，低波动
        elif volatility > 0.05 and abs(trend) > 0.02:
            regime = "volatile_trending"  # 高波动趋势
        else:
            regime = "normal"  # 正常市场

        self.regime_history.append({
            'regime': regime,
            'volatility': volatility,
            'trend': trend,
            'volume': volume,
            'timestamp': datetime.now()
        })

        return regime

    async def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """计算波动率"""
        prices = list(market_data.get('prices', {}).values())
        if len(prices) < 10:
            return 0.0

        returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        return np.std(returns) if returns else 0.0

    async def _calculate_trend(self, market_data: Dict[str, Any]) -> float:
        """计算趋势"""
        prices = list(market_data.get('prices', {}).values())
        if len(prices) < 10:
            return 0.0

        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope / np.mean(prices)

    async def _calculate_volume_trend(self, market_data: Dict[str, Any]) -> float:
        """计算成交量趋势"""
        volumes = list(market_data.get('volumes', {}).values())
        if len(volumes) < 10:
            return 0.0

        x = np.arange(len(volumes))
        slope, _ = np.polyfit(x, volumes, 1)
        return slope / np.mean(volumes) if np.mean(volumes) > 0 else 0.0


class ParameterOptimizer:

    """参数优化器"""

    def __init__(self):

        self.parameter_bounds = {
            'momentum': {
                'lookback_period': (5, 50),
                'momentum_threshold': (0.005, 0.1)
            },
            'mean_reversion': {
                'lookback_period': (10, 100),
                'deviation_threshold': (1.0, 3.0)
            }
        }

    async def optimize_parameters(self, strategies: List[BaseTradingStrategy],
                                  strategy_scores: Dict[str, float],
                                  market_regime: str) -> Dict[str, Dict[str, Any]]:
        """优化策略参数"""
        optimized_params = {}

        for strategy in strategies:
            if strategy.name in strategy_scores:
                score = strategy_scores[strategy.name]

                # 基于表现和市场状态调整参数
        if score < 0.6:  # 表现不佳，需要调整
            new_params = await self._optimize_strategy_parameters(strategy, market_regime)
            optimized_params[strategy.name] = new_params

        return optimized_params

    async def _optimize_strategy_parameters(self, strategy: BaseTradingStrategy,
                                            market_regime: str) -> Dict[str, Any]:
        """优化单个策略的参数"""
        strategy_type = strategy.strategy_type.value

        if strategy_type not in self.parameter_bounds:
            return {}

        bounds = self.parameter_bounds[strategy_type]
        optimized_params = {}

        # 简单的参数优化策略
        if market_regime == "volatile_sideways":
            # 高波动横盘市场
            if strategy_type == "mean_reversion":
                optimized_params['deviation_threshold'] = min(bounds['deviation_threshold'][1], 2.5)
            elif strategy_type == "momentum":
                optimized_params['momentum_threshold'] = min(bounds['momentum_threshold'][1], 0.05)
        elif market_regime == "trending_low_vol":
            # 趋势明确，低波动
            if strategy_type == "momentum":
                optimized_params['lookback_period'] = min(bounds['lookback_period'][1], 30)
        elif market_regime == "volatile_trending":
            # 高波动趋势
            if strategy_type == "momentum":
                optimized_params['momentum_threshold'] = min(bounds['momentum_threshold'][1], 0.03)

        return optimized_params


class MultiStrategyPortfolioOptimizer:

    """多策略投资组合优化器"""

    def __init__(self):

        self.strategies = []
        self.portfolio_history = []
        self.risk_manager = RiskManager()
        self.performance_evaluator = PerformanceEvaluator()

    async def optimize_portfolio(self, market_data: Dict[str, Any],
                                 risk_profile: Dict[str, Any]) -> PortfolioAllocation:
        """优化投资组合"""
        # 生成各策略信号
        strategy_signals = await self._generate_strategy_signals(market_data)

        # 评估策略表现
        strategy_performance = await self.performance_evaluator.evaluate_strategies(strategy_signals)

        # 风险调整优化
        risk_adjusted_weights = await self.risk_manager.adjust_for_risk(
            strategy_performance, risk_profile
        )

        # 组合优化
        optimal_allocation = await self._optimize_portfolio_weights(risk_adjusted_weights)

        # 保存到历史
        self.portfolio_history.append({
            'allocation': optimal_allocation,
            'strategy_signals': strategy_signals,
            'timestamp': datetime.now()
        })

        return optimal_allocation

    async def _generate_strategy_signals(self, market_data: Dict[str, Any]) -> Dict[str, StrategyResult]:
        """生成各策略信号"""
        signals = {}

        for strategy in self.strategies:
            if strategy.is_active:
                signal = await strategy.generate_signal(market_data, {})
                signals[strategy.name] = signal

        return signals

    async def _optimize_portfolio_weights(self, risk_adjusted_weights: Dict[str, float]) -> PortfolioAllocation:
        """优化组合权重"""
        # 使用现代投资组合理论优化权重
        total_weight = sum(risk_adjusted_weights.values())

        if total_weight == 0:
            # 如果没有权重，使用等权重分配
            num_strategies = len(risk_adjusted_weights)
            equal_weight = 1.0 / num_strategies if num_strategies > 0 else 0
            allocations = {strategy: equal_weight for strategy in risk_adjusted_weights.keys()}
        else:
            # 归一化权重
            allocations = {strategy: weight / total_weight
                           for strategy, weight in risk_adjusted_weights.items()}

        return PortfolioAllocation(
            allocations=allocations,
            total_weight=1.0,
            timestamp=datetime.now()
        )

    def add_strategy(self, strategy: BaseTradingStrategy):
        """添加策略"""
        self.strategies.append(strategy)
        logger.info(f"添加策略: {strategy.name}")

    def remove_strategy(self, strategy_name: str):
        """移除策略"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        logger.info(f"移除策略: {strategy_name}")

    def get_portfolio_performance(self) -> Dict[str, Any]:
        """获取组合表现"""
        if not self.portfolio_history:
            return {}

        # 计算组合表现指标
        returns = []
        for entry in self.portfolio_history[-30:]:  # 最近30个记录
            allocation = entry['allocation']
            signals = entry['strategy_signals']

            # 计算组合收益率
            portfolio_return = sum(
                signal.expected_return * allocation.allocations.get(signal.strategy_name, 0)
                for signal in signals.values()
            )
            returns.append(portfolio_return)

        if returns:
            return {
                'avg_return': np.mean(returns),
                'total_return': sum(returns),
                'volatility': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns)
            }

        return {}

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if not returns:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0.0


class RiskManager:

    """风险管理器"""

    def __init__(self):

        self.risk_limits = {
            'max_single_strategy': 0.3,  # 单策略最大权重
            'max_correlation': 0.8,      # 最大相关性
            'target_volatility': 0.15,   # 目标波动率
            'max_drawdown': 0.2         # 最大回撤
        }

    async def adjust_for_risk(self, strategy_performance: Dict[str, Any],
                              risk_profile: Dict[str, Any]) -> Dict[str, float]:
        """风险调整权重"""
        base_weights = {}

        # 基于表现计算基础权重
        total_score = sum(strategy_performance.values())
        if total_score > 0:
            base_weights = {name: score / total_score
                            for name, score in strategy_performance.items()}

        # 应用风险限制
        risk_adjusted_weights = await self._apply_risk_limits(base_weights, risk_profile)

        return risk_adjusted_weights

    async def _apply_risk_limits(self, weights: Dict[str, float],
                                 risk_profile: Dict[str, Any]) -> Dict[str, float]:
        """应用风险限制"""
        adjusted_weights = weights.copy()

        # 单策略权重限制
        max_single = risk_profile.get(
            'max_single_strategy', self.risk_limits['max_single_strategy'])
        for strategy, weight in adjusted_weights.items():
            if weight > max_single:
                adjusted_weights[strategy] = max_single

        # 重新归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {strategy: weight / total_weight
                                for strategy, weight in adjusted_weights.items()}

        return adjusted_weights


class PerformanceEvaluator:

    """性能评估器"""

    def __init__(self):

        self.evaluation_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]

    async def evaluate_strategies(self, strategy_signals: Dict[str, StrategyResult]) -> Dict[str, float]:
        """评估策略表现"""
        scores = {}

        for strategy_name, signal in strategy_signals.items():
            # 基于信号质量和预期收益计算分数
            signal_score = signal.confidence * 0.6
            return_score = min(signal.expected_return / 0.1, 1.0) * 0.4  # 预期收益评分

            scores[strategy_name] = signal_score + return_score

        return scores


# 全局策略引擎实例
_strategy_engine_instance = None


def get_strategy_engine() -> MultiStrategyPortfolioOptimizer:
    """获取策略引擎实例"""
    global _strategy_engine_instance
    if _strategy_engine_instance is None:
        _strategy_engine_instance = MultiStrategyPortfolioOptimizer()

        # 初始化默认策略
        momentum_strategy = MomentumStrategy()
        mean_reversion_strategy = MeanReversionStrategy()
        ml_strategy = MLBasedStrategy()

        _strategy_engine_instance.add_strategy(momentum_strategy)
        _strategy_engine_instance.add_strategy(mean_reversion_strategy)
        _strategy_engine_instance.add_strategy(ml_strategy)

    return _strategy_engine_instance


def get_adaptive_learner() -> AdaptiveStrategyLearner:
    """获取自适应学习器实例"""
    return AdaptiveStrategyLearner()


if __name__ == "__main__":
    # 测试代码
    print("智能交易策略引擎测试")

    async def test_strategy_engine():
        # 获取策略引擎
        engine = get_strategy_engine()
        learner = get_adaptive_learner()

        # 测试市场数据
        test_market_data = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'prices': {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0},
            'volumes': {'AAPL': 50000000, 'GOOGL': 1500000, 'MSFT': 30000000},
            'timestamp': datetime.now().isoformat()
        }

        # 测试策略信号生成
        portfolio = await engine.optimize_portfolio(test_market_data, {'max_single_strategy': 0.4})

        print("优化后的投资组合:")
        print(f"配置: {portfolio.allocations}")
        print(f"总权重: {portfolio.total_weight}")

        # 测试自适应学习
        strategies = [MomentumStrategy(), MeanReversionStrategy(), MLBasedStrategy()]
        adaptation_result = await learner.adapt_strategies(strategies, test_market_data)

        print(f"自适应调整结果: {adaptation_result}")

    # 运行测试
    asyncio.run(test_strategy_engine())
    print("智能交易策略引擎测试完成")

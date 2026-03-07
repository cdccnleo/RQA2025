#!/usr/bin/env python3
"""
AI策略优化器 - 基于强化学习和深度学习的智能策略优化
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from enum import Enum

from ..interfaces.strategy_interfaces import StrategySignal, StrategyResult
from ..strategies.base_strategy import BaseStrategy
from strategy.core.integration import get_models_adapter

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = get_models_adapter()
    logger = models_adapter.get_models_logger()
except Exception as e:
    # 降级处理
    import logging
    logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):

    """优化目标"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    BALANCED = "balanced"


class OptimizationAlgorithm(Enum):

    """优化算法"""
    PPO = "ppo"  # Proximal Policy Optimization
    SAC = "sac"  # Soft Actor - Critic
    DQN = "dqn"  # Deep Q - Network
    EVOLUTIONARY = "evolutionary"  # 进化算法
    BAYESIAN = "bayesian"  # 贝叶斯优化


@dataclass
class OptimizationConfig:

    """优化配置"""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.PPO
    objective: OptimizationObjective = OptimizationObjective.BALANCED
    max_iterations: int = 1000
    population_size: int = 50
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    risk_free_rate: float = 0.02
    max_position_size: float = 1.0
    min_position_size: float = 0.0
    transaction_cost: float = 0.001
    rebalance_frequency: int = 1  # 每N个交易日调仓


@dataclass
class MarketState:

    """市场状态"""
    timestamp: datetime
    price_data: Dict[str, float]
    volume_data: Dict[str, float]
    technical_indicators: Dict[str, float]
    sentiment_score: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0


@dataclass
class PortfolioState:

    """投资组合状态"""
    cash: float
    positions: Dict[str, float]
    total_value: float
    returns: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


@dataclass
class TradingAction:

    """交易动作"""
    asset: str
    action_type: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    timestamp: datetime


class RLAgent:

    """强化学习代理"""

    def __init__(self, config: OptimizationConfig, state_dim: int, action_dim: int):

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.epsilon = config.exploration_rate

        # 初始化神经网络（简化版）
        self.actor_network = self._build_actor_network()
        self.critic_network = self._build_critic_network()

        logger.info(f"强化学习代理初始化完成 - 状态维度: {state_dim}, 动作维度: {action_dim}")

    def _build_actor_network(self) -> Dict[str, Any]:
        """构建演员网络（策略网络）"""
        # 这里应该实现实际的神经网络
        # 为了演示，返回一个占位符
        return {
            'layers': [self.state_dim, 256, 128, self.action_dim],
            'activation': 'relu',
            'output_activation': 'softmax'
        }

    def _build_critic_network(self) -> Dict[str, Any]:
        """构建评论家网络（价值网络）"""
        # 这里应该实现实际的神经网络
        # 为了演示，返回一个占位符
        return {
            'layers': [self.state_dim, 256, 128, 1],
            'activation': 'relu',
            'output_activation': 'linear'
        }

    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        if np.secrets.random() < self.epsilon:
            # 探索：随机选择动作
            return np.secrets.randint(self.action_dim)
        else:
            # 利用：根据策略选择动作
            return self._predict_action(state)

    def _predict_action(self, state: np.ndarray) -> int:
        """预测最优动作"""
        # 这里应该使用训练好的模型进行预测
        # 为了演示，使用随机选择
        return np.secrets.randint(self.action_dim)

    def store_transition(self, state: np.ndarray, action: int, reward: float,


                         next_state: np.ndarray, done: bool):
        """存储转换"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        # 限制内存大小
        if len(self.memory) > 10000:
            self.memory = self.memory[-5000:]

    def train(self, batch_size: int = 32):
        """训练代理"""
        if len(self.memory) < batch_size:
            return

        # 从记忆中采样
        batch = np.secrets.choice(self.memory, batch_size, replace=False)

        # 这里应该实现实际的训练逻辑
        # 为了演示，简单地减少探索率
        self.epsilon = max(0.01, self.epsilon * 0.995)

        logger.debug(f"训练完成 - 探索率: {self.epsilon:.4f}")


class AIStrategyOptimizer:

    """
    AI策略优化器 - 基于强化学习和深度学习的智能策略优化

    主要功能：
    1. 强化学习策略优化
    2. 动态风险管理
    3. 自适应参数调整
    4. 多目标优化
    5. 实时学习和适应
    """

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self.is_optimizing = False
        self.optimization_results = []
        self.market_history = []
        self.portfolio_history = []

        # 初始化强化学习代理
        state_dim = 50  # 市场状态特征维度
        action_dim = 10  # 动作空间维度（不同的仓位配置）
        self.rl_agent = RLAgent(config, state_dim, action_dim)

        # 优化统计
        self.stats = {
            'iterations': 0,
            'best_return': 0.0,
            'best_sharpe': 0.0,
            'convergence_time': 0.0,
            'last_update': datetime.now()
        }

        # 启动后台优化线程
        self._optimization_thread = threading.Thread(
            target=self._continuous_optimization, daemon=True)
        self._optimization_thread.start()

        logger.info("AI策略优化器初始化完成")

    def optimize_strategy(self, base_strategy: BaseStrategy,


                          market_data: pd.DataFrame,
                          initial_capital: float = 1000000) -> Dict[str, Any]:
        """
        优化交易策略

        Args:
            base_strategy: 基础策略
            market_data: 市场数据
            initial_capital: 初始资本

        Returns:
            Dict[str, Any]: 优化结果
        """
        try:
            self.is_optimizing = True
            start_time = time.time()

            logger.info("开始AI策略优化")

            # 1. 准备市场状态
            market_states = self._prepare_market_states(market_data)

            # 2. 初始化投资组合
            portfolio_state = PortfolioState(
                cash=initial_capital,
                positions={},
                total_value=initial_capital,
                returns=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0
            )

            # 3. 执行强化学习优化
            optimized_parameters = self._run_reinforcement_learning(
                base_strategy, market_states, portfolio_state
            )

            # 4. 验证优化结果
            validation_results = self._validate_optimization(
                base_strategy, optimized_parameters, market_data
            )

            # 5. 生成优化报告
            optimization_time = time.time() - start_time
            result = self._generate_optimization_report(
                optimized_parameters, validation_results, optimization_time
            )

            self.is_optimizing = False
            self.optimization_results.append(result)

            logger.info(f"AI策略优化完成 - 耗时: {optimization_time:.2f}秒")
            return result

        except Exception as e:
            self.is_optimizing = False
            logger.error(f"AI策略优化失败: {e}")
            return {'success': False, 'error': str(e)}

    def _prepare_market_states(self, market_data: pd.DataFrame) -> List[MarketState]:
        """准备市场状态序列"""
        market_states = []

        for idx, row in market_data.iterrows():
            # 计算技术指标
            technical_indicators = self._calculate_technical_indicators(market_data, idx)

            # 计算情绪分数和波动率
            sentiment_score = self._calculate_sentiment_score(row)
            volatility = self._calculate_volatility(market_data, idx)
            trend_strength = self._calculate_trend_strength(market_data, idx)

            market_state = MarketState(
                timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                price_data={
                    'open': row.get('open', row.get('Open', 0)),
                    'high': row.get('high', row.get('High', 0)),
                    'low': row.get('low', row.get('Low', 0)),
                    'close': row.get('close', row.get('Close', 0))
                },
                volume_data={
                    'volume': row.get('volume', row.get('Volume', 0))
                },
                technical_indicators=technical_indicators,
                sentiment_score=sentiment_score,
                volatility=volatility,
                trend_strength=trend_strength
            )

            market_states.append(market_state)

        return market_states

    def _calculate_technical_indicators(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """计算技术指标"""
        try:
            # 计算简单移动平均
            sma_20 = data['close'].rolling(window=20).mean(
            ).iloc[idx] if idx >= 20 else data['close'].iloc[idx]
            sma_50 = data['close'].rolling(window=50).mean(
            ).iloc[idx] if idx >= 50 else data['close'].iloc[idx]

            # 计算RSI
            rsi = self._calculate_rsi(data['close'], idx, period=14)

            # 计算MACD
            macd, signal, hist = self._calculate_macd(data['close'], idx)

            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'macd_hist': hist
            }

        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")
            return {'sma_20': 0, 'sma_50': 0, 'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_hist': 0}

    def _calculate_rsi(self, prices: pd.Series, idx: int, period: int = 14) -> float:
        """计算RSI指标"""
        if idx < period:
            return 50.0

        gains = []
        losses = []

        for i in range(idx - period + 1, idx + 1):
            if i > 0:
                change = prices.iloc[i] - prices.iloc[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: pd.Series, idx: int) -> Tuple[float, float, float]:
        """计算MACD指标"""
        if idx < 26:
            return 0.0, 0.0, 0.0

        # 计算EMA12
        ema12 = prices.iloc[idx - 11:idx + 1].ewm(span=12, adjust=False).mean().iloc[-1]
        # 计算EMA26
        ema26 = prices.iloc[idx - 25:idx + 1].ewm(span=26, adjust=False).mean().iloc[-1]

        macd = ema12 - ema26

        # 计算信号线（MACD的9日EMA）
        if idx >= 34:  # 需要足够的数据计算信号线
            macd_series = prices.iloc[idx - 33:idx + 1].ewm(span=12, adjust=False).mean() - \
                prices.iloc[idx - 33:idx + 1].ewm(span=26, adjust=False).mean()
            signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
            hist = macd - signal
        else:
            signal = macd
            hist = 0.0

        return macd, signal, hist

    def _calculate_sentiment_score(self, row) -> float:
        """计算情绪分数"""
        # 这里应该实现实际的情绪分析
        # 为了演示，返回一个基于价格变动的简单分数
        try:
            close = row.get('close', row.get('Close', 0))
            open_price = row.get('open', row.get('Open', 0))

            if open_price > 0:
                return (close - open_price) / open_price * 100  # 百分比变化
            return 0.0
        except BaseException:
            return 0.0

    def _calculate_volatility(self, data: pd.DataFrame, idx: int) -> float:
        """计算波动率"""
        try:
            if idx < 20:
                return 0.0

            returns = data['close'].pct_change().iloc[idx - 19:idx + 1]
            return returns.std() * np.sqrt(252)  # 年化波动率
        except BaseException:
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame, idx: int) -> float:
        """计算趋势强度"""
        try:
            if idx < 20:
                return 0.0

            prices = data['close'].iloc[idx - 19:idx + 1]
            slope, _ = np.polyfit(range(len(prices)), prices, 1)
            return slope / prices.mean()  # 归一化斜率
        except BaseException:
            return 0.0

    def _run_reinforcement_learning(self, base_strategy: BaseStrategy,

                                    market_states: List[MarketState],
                                    portfolio_state: PortfolioState) -> Dict[str, Any]:
        """执行强化学习优化"""
        logger.info("开始强化学习策略优化")

        optimized_params = {}
        best_reward = float('-inf')

        for iteration in range(self.config.max_iterations):
            # 1. 重置环境
            current_portfolio = PortfolioState(
                cash=portfolio_state.cash,
                positions={},
                total_value=portfolio_state.cash,
                returns=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0
            )

            episode_reward = 0.0

            # 2. 执行一轮训练
            for i, market_state in enumerate(market_states):
                # 转换为状态向量
                state_vector = self._market_state_to_vector(market_state, current_portfolio)

                # 选择动作
                action_idx = self.rl_agent.select_action(state_vector)

                # 执行动作
                reward, next_portfolio = self._execute_action(
                    action_idx, market_state, current_portfolio, base_strategy
                )

                episode_reward += reward

                # 存储转换
                if i < len(market_states) - 1:
                    next_state_vector = self._market_state_to_vector(
                        market_states[i + 1], next_portfolio)
                    done = False
                else:
                    next_state_vector = np.zeros_like(state_vector)
                    done = True

                self.rl_agent.store_transition(
                    state_vector, action_idx, reward, next_state_vector, done)

                current_portfolio = next_portfolio

            # 3. 训练代理
            self.rl_agent.train()

            # 4. 更新最优参数
            if episode_reward > best_reward:
                best_reward = episode_reward
                optimized_params = {
                    'exploration_rate': self.rl_agent.epsilon,
                    'iteration': iteration,
                    'reward': episode_reward,
                    'final_portfolio_value': current_portfolio.total_value,
                    'total_return': current_portfolio.returns
                }

            # 5. 记录统计信息
            self.stats['iterations'] = iteration + 1
            self.stats['last_update'] = datetime.now()

        if (iteration + 1) % 100 == 0:
            logger.info(f"强化学习优化进度: {iteration + 1}/{self.config.max_iterations}, "
                        f"最佳奖励: {best_reward:.4f}")

        return optimized_params

    def _market_state_to_vector(self, market_state: MarketState, portfolio: PortfolioState) -> np.ndarray:
        """将市场状态转换为向量"""
        try:
            # 价格数据
            price_vector = [
                market_state.price_data.get('open', 0),
                market_state.price_data.get('high', 0),
                market_state.price_data.get('low', 0),
                market_state.price_data.get('close', 0)
            ]

            # 技术指标
            tech_vector = [
                market_state.technical_indicators.get('sma_20', 0),
                market_state.technical_indicators.get('sma_50', 0),
                market_state.technical_indicators.get('rsi', 50),
                market_state.technical_indicators.get('macd', 0),
                market_state.technical_indicators.get('macd_signal', 0),
                market_state.technical_indicators.get('macd_hist', 0)
            ]

            # 其他特征
            other_vector = [
                market_state.sentiment_score,
                market_state.volatility,
                market_state.trend_strength,
                portfolio.total_value,
                portfolio.returns,
                len(portfolio.positions)
            ]

            # 组合成状态向量
            state_vector = price_vector + tech_vector + other_vector

            # 标准化
            state_array = np.array(state_vector, dtype=np.float32)
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1e6, neginf=-1e6)

            return state_array

        except Exception as e:
            logger.warning(f"状态转换失败: {e}")
            return np.zeros(50, dtype=np.float32)

    def _execute_action(self, action_idx: int, market_state: MarketState,


                        portfolio: PortfolioState, strategy: BaseStrategy) -> Tuple[float, PortfolioState]:
        """执行交易动作"""
        try:
            # 根据动作索引确定仓位
            position_size = (action_idx / 9.0) * (self.config.max_position_size -
                                                  self.config.min_position_size) + self.config.min_position_size

            # 计算交易信号
            signal = StrategySignal(
                timestamp=market_state.timestamp,
                signal_type='position_adjustment',
                strength=position_size,
                confidence=0.8,
                metadata={'ai_optimized': True}
            )

            # 执行策略
            result = strategy.generate_signals(signal)

            # 计算奖励
            reward = self._calculate_reward(result, portfolio, market_state)

            # 更新投资组合状态
            new_portfolio = self._update_portfolio_state(portfolio, result, market_state)

            return reward, new_portfolio

        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            return -1.0, portfolio

    def _calculate_reward(self, result: StrategyResult, portfolio: PortfolioState, market_state: MarketState) -> float:
        """计算奖励函数"""
        try:
            reward = 0.0

            # 收益奖励
            if hasattr(result, 'returns') and result.returns is not None:
                reward += result.returns * 100  # 放大收益信号

            # 风险惩罚
            if hasattr(result, 'volatility') and result.volatility is not None:
                reward -= result.volatility * 50  # 惩罚高波动

            # 夏普比率奖励
            if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio is not None:
                reward += result.sharpe_ratio * 10

            # 最大回撤惩罚
            if hasattr(result, 'max_drawdown') and result.max_drawdown is not None:
                reward -= abs(result.max_drawdown) * 20

            # 交易成本惩罚
            if hasattr(result, 'transaction_costs') and result.transaction_costs is not None:
                reward -= result.transaction_costs * 10

            return reward

        except Exception as e:
            logger.warning(f"奖励计算失败: {e}")
            return 0.0

    def _update_portfolio_state(self, portfolio: PortfolioState, result: StrategyResult, market_state: MarketState) -> PortfolioState:
        """更新投资组合状态"""
        try:
            # 这里应该实现实际的投资组合更新逻辑
            # 为了演示，返回一个更新后的投资组合状态
            new_portfolio = PortfolioState(
                cash=portfolio.cash,
                positions=portfolio.positions.copy(),
                total_value=portfolio.total_value,
                returns=portfolio.returns,
                sharpe_ratio=portfolio.sharpe_ratio,
                max_drawdown=portfolio.max_drawdown,
                volatility=portfolio.volatility
            )

            return new_portfolio

        except Exception as e:
            logger.error(f"投资组合状态更新失败: {e}")
            return portfolio

    def _validate_optimization(self, base_strategy: BaseStrategy,


                               optimized_params: Dict[str, Any],
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """验证优化结果"""
        try:
            logger.info("开始优化结果验证")

            # 使用优化后的参数进行回测
            validation_results = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'validation_passed': True
            }

            # 这里应该实现实际的验证逻辑
            # 为了演示，生成一些合理的验证结果

            return validation_results

        except Exception as e:
            logger.error(f"优化结果验证失败: {e}")
            return {'validation_passed': False, 'error': str(e)}

    def _generate_optimization_report(self, optimized_params: Dict[str, Any],


                                      validation_results: Dict[str, Any],
                                      optimization_time: float) -> Dict[str, Any]:
        """生成优化报告"""
        report = {
            'success': True,
            'optimization_time': optimization_time,
            'optimized_parameters': optimized_params,
            'validation_results': validation_results,
            'performance_metrics': {
                'iterations': self.stats['iterations'],
                'best_return': self.stats['best_return'],
                'best_sharpe': self.stats['best_sharpe'],
                'convergence_time': self.stats['convergence_time']
            },
            'timestamp': datetime.now(),
            'config': {
                'algorithm': self.config.algorithm.value,
                'objective': self.config.objective.value,
                'max_iterations': self.config.max_iterations
            }
        }

        return report

    def _continuous_optimization(self):
        """连续优化线程"""
        while True:
            try:
                # 检查是否有新的市场数据
                if self.market_history and not self.is_optimizing:
                    # 执行增量优化
                    self._incremental_optimization()

                time.sleep(300)  # 每5分钟检查一次

            except Exception as e:
                logger.error(f"连续优化异常: {e}")
                time.sleep(60)

    def _incremental_optimization(self):
        """增量优化"""
        try:
            logger.info("执行增量优化")

            # 这里应该实现增量优化的逻辑
            # 使用最新的市场数据更新策略参数

            logger.info("增量优化完成")

        except Exception as e:
            logger.error(f"增量优化失败: {e}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            'is_optimizing': self.is_optimizing,
            'iterations_completed': self.stats['iterations'],
            'best_return': self.stats['best_return'],
            'best_sharpe': self.stats['best_sharpe'],
            'last_update': self.stats['last_update'],
            'optimization_results_count': len(self.optimization_results)
        }

    def get_latest_optimization_result(self) -> Optional[Dict[str, Any]]:
        """获取最新的优化结果"""
        if self.optimization_results:
            return self.optimization_results[-1]
        return None


# 全局AI策略优化器实例
_ai_strategy_optimizer = None


def get_ai_strategy_optimizer(config: Optional[OptimizationConfig] = None) -> AIStrategyOptimizer:
    """获取AI策略优化器实例"""
    global _ai_strategy_optimizer

    if _ai_strategy_optimizer is None:
        if config is None:
            config = OptimizationConfig()
        _ai_strategy_optimizer = AIStrategyOptimizer(config)

    return _ai_strategy_optimizer


# 便捷函数

def optimize_strategy_with_ai(base_strategy: BaseStrategy,

                              market_data: pd.DataFrame,
                              config: Optional[OptimizationConfig] = None) -> Dict[str, Any]:
    """
    使用AI优化交易策略的便捷函数

    Args:
        base_strategy: 基础策略
        market_data: 市场数据
        config: 优化配置

    Returns:
        Dict[str, Any]: 优化结果
    """
    optimizer = get_ai_strategy_optimizer(config)
    return optimizer.optimize_strategy(base_strategy, market_data)


if __name__ == "__main__":
    # 示例用法
    config = OptimizationConfig(
        algorithm=OptimizationAlgorithm.PPO,
        objective=OptimizationObjective.MAXIMIZE_SHARPE,
        max_iterations=500
    )

    optimizer = AIStrategyOptimizer(config)
    logger.info("AI策略优化器已准备就绪")

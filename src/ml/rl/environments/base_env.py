#!/usr/bin/env python3
"""
RQA2025强化学习环境基类
提供统一的RL环境接口和基础功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from ...ai.features.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """环境配置"""
    symbol: str = "AAPL"
    initial_balance: float = 100000.0
    max_position_pct: float = 0.1  # 最大持仓比例
    transaction_cost: float = 0.001  # 交易成本
    slippage: float = 0.0005  # 滑点
    max_holding_period: int = 30  # 最大持仓周期
    episode_length: int = 1000  # 每个episode的长度
    render_mode: Optional[str] = None  # 渲染模式


@dataclass
class TradeAction:
    """交易动作"""
    action_type: str  # "buy", "sell", "hold"
    quantity_pct: float  # 交易数量比例 (0-1)


@dataclass
class PortfolioState:
    """投资组合状态"""
    cash: float
    position: float  # 持仓数量
    entry_price: Optional[float] = None  # 平均持仓成本
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class MarketState:
    """市场状态"""
    symbol: str
    price: float
    volume: float
    timestamp: str
    features: Dict[str, float]  # AI特征


class BaseTradingEnv(ABC):
    """交易环境基类"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()

        # 环境状态
        self.current_step = 0
        self.episode_length = config.episode_length
        self.done = False

        # 投资组合状态
        self.portfolio = PortfolioState(
            cash=config.initial_balance,
            position=0.0
        )

        # 市场数据
        self.market_data: Optional[pd.DataFrame] = None
        self.current_market_state: Optional[MarketState] = None

        # 交易历史
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[PortfolioState] = []

        # 性能指标
        self.episode_rewards: List[float] = []
        self.episode_returns: List[float] = []

        logger.info(f"交易环境初始化完成: {config.symbol}")

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境到初始状态"""

    @abstractmethod
    def step(self, action: Union[int, TradeAction]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步动作"""

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[Any]:
        """渲染环境状态"""

    def _load_market_data(self) -> pd.DataFrame:
        """加载市场数据"""
        # 这里应该从数据源加载真实数据
        # 为了演示，我们生成模拟数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=self.episode_length, freq='D')

        # 生成价格数据 (带趋势和波动)
        trend = np.linspace(100, 120, self.episode_length)
        noise = np.random.randn(self.episode_length) * 3
        prices = trend + noise

        # 生成成交量
        volumes = np.random.randint(10000, 100000, self.episode_length)

        # 创建OHLCV数据
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(self.episode_length) * 0.005),
            'high': prices * (1 + np.random.randn(self.episode_length) * 0.01),
            'low': prices * (1 - np.random.randn(self.episode_length) * 0.01),
            'close': prices,
            'volume': volumes
        }, index=dates)

        return data

    def _get_current_market_state(self) -> MarketState:
        """获取当前市场状态"""
        if self.market_data is None or self.current_step >= len(self.market_data):
            return None

        row = self.market_data.iloc[self.current_step]
        timestamp = row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name)

        # 提取AI特征
        features_df = self.feature_engineer.extract_features(
            self.market_data.iloc[:self.current_step + 1]
        )

        if len(features_df) == 0:
            features = {}
        else:
            # 获取最新的特征值
            latest_features = features_df.iloc[-1]
            features = latest_features.to_dict()

        return MarketState(
            symbol=self.config.symbol,
            price=row['close'],
            volume=row['volume'],
            timestamp=timestamp,
            features=features
        )

    def _execute_trade(self, action: TradeAction) -> Dict[str, Any]:
        """执行交易"""
        market_state = self.current_market_state
        if not market_state:
            return {'success': False, 'message': '无市场数据'}

        # 计算交易数量
        max_quantity = (self.portfolio.cash / market_state.price) * action.quantity_pct
        max_quantity = min(max_quantity, self.portfolio.cash / market_state.price)

        # 应用最大持仓限制
        max_position_value = self.portfolio.cash * self.config.max_position_pct
        max_position_quantity = max_position_value / market_state.price

        if action.action_type == "buy":
            quantity = min(max_quantity, max_position_quantity - self.portfolio.position)

            if quantity > 0:
                # 计算交易成本
                execution_price = market_state.price * (1 + self.config.slippage)
                total_cost = quantity * execution_price * (1 + self.config.transaction_cost)

                if total_cost <= self.portfolio.cash:
                    # 执行买入
                    self.portfolio.cash -= total_cost
                    old_position_value = self.portfolio.position * \
                        (self.portfolio.entry_price or market_state.price)
                    new_position = self.portfolio.position + quantity
                    new_position_value = old_position_value + quantity * execution_price

                    self.portfolio.position = new_position
                    self.portfolio.entry_price = new_position_value / new_position if new_position > 0 else None

                    # 记录交易
                    self._record_trade({
                        'type': 'buy',
                        'quantity': quantity,
                        'price': execution_price,
                        'timestamp': market_state.timestamp,
                        'commission': total_cost - quantity * execution_price
                    })

                    return {
                        'success': True,
                        'executed_quantity': quantity,
                        'execution_price': execution_price,
                        'total_cost': total_cost
                    }
                else:
                    return {'success': False, 'message': '资金不足'}

        elif action.action_type == "sell":
            quantity = min(self.portfolio.position * action.quantity_pct, self.portfolio.position)

            if quantity > 0:
                # 计算交易成本
                execution_price = market_state.price * (1 - self.config.slippage)
                total_revenue = quantity * execution_price * (1 - self.config.transaction_cost)

                # 执行卖出
                self.portfolio.cash += total_revenue
                realized_pnl = quantity * \
                    (execution_price - (self.portfolio.entry_price or execution_price))
                self.portfolio.realized_pnl += realized_pnl
                self.portfolio.position -= quantity

                # 如果持仓为0，重置入场价格
                if self.portfolio.position <= 0:
                    self.portfolio.entry_price = None
                    self.portfolio.position = 0

                # 记录交易
                self._record_trade({
                    'type': 'sell',
                    'quantity': quantity,
                    'price': execution_price,
                    'timestamp': market_state.timestamp,
                    'commission': quantity * execution_price * self.config.transaction_cost,
                    'realized_pnl': realized_pnl
                })

                return {
                    'success': True,
                    'executed_quantity': quantity,
                    'execution_price': execution_price,
                    'total_revenue': total_revenue,
                    'realized_pnl': realized_pnl
                }
            else:
                return {'success': False, 'message': '无持仓可卖'}

        return {'success': False, 'message': '无效动作'}

    def _record_trade(self, trade: Dict[str, Any]):
        """记录交易"""
        self.trade_history.append(trade)

    def _calculate_portfolio_value(self) -> float:
        """计算投资组合总价值"""
        if not self.current_market_state:
            return self.portfolio.cash

        position_value = self.portfolio.position * self.current_market_state.price
        return self.portfolio.cash + position_value

    def _calculate_reward(self, prev_value: float, current_value: float,
                          transaction_cost: float = 0.0) -> float:
        """计算奖励函数"""
        # 基础回报
        raw_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0

        # 扣除交易成本
        net_return = raw_return - transaction_cost

        # 风险调整 (简化的夏普比率考虑)
        if len(self.episode_rewards) > 10:
            returns_std = np.std(self.episode_rewards[-10:])
            if returns_std > 0:
                sharpe_ratio = net_return / returns_std
                reward = sharpe_ratio  # 使用夏普比率作为奖励
            else:
                reward = net_return
        else:
            reward = net_return

        # 添加惩罚项
        penalty = 0.0

        # 过度交易惩罚
        if len(self.trade_history) > 0:
            recent_trades = [t for t in self.trade_history[-5:]
                             if t['timestamp'] == self.current_market_state.timestamp]
            if len(recent_trades) > 2:  # 频繁交易惩罚
                penalty -= 0.01

        # 持仓时间过长惩罚
        if self.portfolio.position > 0 and len(self.trade_history) > 0:
            last_buy = None
            for trade in reversed(self.trade_history):
                if trade['type'] == 'buy':
                    last_buy = trade
                    break

            if last_buy:
                holding_days = (pd.Timestamp(self.current_market_state.timestamp) -
                                pd.Timestamp(last_buy['timestamp'])).days
                if holding_days > self.config.max_holding_period:
                    penalty -= 0.001 * (holding_days - self.config.max_holding_period)

        return reward + penalty

    def get_observation_space_shape(self) -> Tuple[int, ...]:
        """获取观察空间形状"""
        if self.market_data is None:
            return (154,)  # 默认特征数量

        features_df = self.feature_engineer.extract_features(self.market_data.iloc[:1])
        return (len(features_df.columns),)

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        # 离散动作空间: hold, buy_small, buy_medium, buy_large, sell_small, sell_medium, sell_large
        return 7

    def action_to_trade(self, action: int) -> TradeAction:
        """将离散动作转换为交易动作"""
        action_map = {
            0: TradeAction("hold", 0.0),      # 持有
            1: TradeAction("buy", 0.2),       # 小仓位买入
            2: TradeAction("buy", 0.5),       # 中仓位买入
            3: TradeAction("buy", 1.0),       # 全仓位买入
            4: TradeAction("sell", 0.2),      # 小仓位卖出
            5: TradeAction("sell", 0.5),      # 中仓位卖出
            6: TradeAction("sell", 1.0),      # 全仓位卖出
        }

        return action_map.get(action, TradeAction("hold", 0.0))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if len(self.episode_rewards) == 0:
            return {}

        return {
            'total_episodes': len(self.episode_returns),
            'avg_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'total_trades': len(self.trade_history),
            'final_portfolio_value': self._calculate_portfolio_value(),
            'total_return': (self._calculate_portfolio_value() - self.config.initial_balance) / self.config.initial_balance,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': self._calculate_win_rate()
        }

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(self.episode_returns) < 2:
            return 0.0

        returns = np.array(self.episode_returns)
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.portfolio_history:
            return 0.0

        values = [p.total_value for p in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trade_history:
            return 0.0

        winning_trades = sum(1 for trade in self.trade_history
                             if trade.get('realized_pnl', 0) > 0)
        return winning_trades / len(self.trade_history) if self.trade_history else 0.0

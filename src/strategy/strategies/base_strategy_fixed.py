#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
"""
策略服务层基础策略实现 - Phase 4修复版本
Strategy Service Layer Base Strategy Implementation

完全实现IStrategy接口，修复策略执行引擎问题
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


# 核心数据结构定义
@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None


@dataclass
class StrategySignal:
    """策略信号"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    price: float
    quantity: int
    confidence: float
    timestamp: datetime
    strategy_id: str


@dataclass
class StrategyOrder:
    """策略订单"""
    order_id: str
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT'
    side: str  # 'BUY', 'SELL'
    quantity: int
    price: Optional[float]
    timestamp: datetime


@dataclass
class StrategyPosition:
    """策略持仓"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime


class StrategyType:
    """策略类型枚举"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"


class StrategyStatus:
    """策略状态枚举"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class IStrategy(ABC):
    """策略接口 - 修复版本"""

    @abstractmethod
    def get_strategy_name(self) -> str: pass

    @abstractmethod
    def get_strategy_type(self) -> str: pass

    @abstractmethod
    def get_strategy_description(self) -> str: pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]: pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool: pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]: pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool: pass

    @abstractmethod
    def on_market_data(self, data: MarketData) -> List[StrategySignal]: pass

    @abstractmethod
    def on_order_update(self, order: StrategyOrder) -> None: pass

    @abstractmethod
    def on_position_update(self, position: StrategyPosition) -> None: pass

    @abstractmethod
    def should_enter_position(
        self, symbol: str, data: pd.DataFrame) -> Optional[StrategySignal]: pass

    @abstractmethod
    def should_exit_position(self, position: StrategyPosition,
                             data: pd.DataFrame) -> Optional[StrategySignal]: pass

    @abstractmethod
    def calculate_position_size(
        self, capital: float, risk_per_trade: float, symbol: str) -> float: pass

    @abstractmethod
    def get_risk_management_rules(self) -> Dict[str, Any]: pass

    @abstractmethod
    def get_strategy_status(self) -> str: pass

    @abstractmethod
    def get_current_positions(self) -> List[StrategyPosition]: pass

    @abstractmethod
    def get_pending_orders(self) -> List[StrategyOrder]: pass

    @abstractmethod
    def get_strategy_metrics(self) -> Dict[str, Any]: pass

    @abstractmethod
    def start(self) -> bool: pass

    @abstractmethod
    def stop(self) -> bool: pass

    @abstractmethod
    def pause(self) -> bool: pass

    @abstractmethod
    def resume(self) -> bool: pass

    @abstractmethod
    def reset(self) -> bool: pass

    @abstractmethod
    def save_state(self, path: str) -> bool: pass

    @abstractmethod
    def load_state(self, path: str) -> bool: pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]: pass

    @abstractmethod
    def get_required_data_fields(self) -> List[str]: pass

    @abstractmethod
    def validate_market_data(self, data: MarketData) -> bool: pass


class BaseStrategy(IStrategy):
    """
    交易策略基类 - Phase 4修复版本
    Trading Strategy Base Class

    完全实现IStrategy接口，支持完整的策略生命周期管理。
    """

    def __init__(self, strategy_id: str, name: str, strategy_type: str):
        self.strategy_id = strategy_id
        self._name = name
        self._strategy_type = strategy_type
        self._description = f"{name} strategy implementation"
        self._parameters: Dict[str, Any] = {}
        self._status = StrategyStatus.INITIALIZED
        self._current_positions: List[StrategyPosition] = []
        self._pending_orders: List[StrategyOrder] = []
        self._supported_symbols: List[str] = []
        self._required_data_fields: List[str] = ['price', 'volume', 'timestamp']
        self._is_running = False
        self._last_update = datetime.now()

        # 辅助方法
    def add_position(self, position_data: Dict[str, Any]) -> None:
        """添加持仓"""
        position = StrategyPosition(
            symbol=position_data.get('symbol', 'UNKNOWN'),
            quantity=position_data.get('quantity', 0),
            average_price=position_data.get('price', 0.0),
            current_price=position_data.get('price', 0.0),
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        self._current_positions.append(position)

    def add_pending_order(self, order_data: Dict[str, Any]) -> None:
        """添加挂单"""
        order = StrategyOrder(
            order_id=f"order_{len(self._pending_orders) + 1}",
            symbol=order_data.get('symbol', 'UNKNOWN'),
            order_type=order_data.get('order_type', 'MARKET'),
            side=order_data.get('side', 'BUY'),
            quantity=order_data.get('quantity', 0),
            price=order_data.get('price'),
            timestamp=datetime.now()
        )
        self._pending_orders.append(order)

        # 性能指标
        self._metrics = {
            'total_signals': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

        logger.info(f"Strategy {strategy_id} initialized with type {strategy_type}")

    # IStrategy接口实现
    def get_strategy_name(self) -> str:
        return self._name

    def get_strategy_type(self) -> str:
        return self._strategy_type

    def get_strategy_description(self) -> str:
        return self._description

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        try:
            validated_params = self.validate_parameters(parameters)
            self._parameters.update(validated_params)
            logger.info(f"Parameters updated for strategy {self.strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证并清理参数"""
        validated = {}

        # 基础参数验证
        if 'max_position_size' in parameters:
            max_pos = parameters['max_position_size']
            if not isinstance(max_pos, (int, float)) or max_pos <= 0:
                raise ValueError("max_position_size must be a positive number")
            validated['max_position_size'] = float(max_pos)

        if 'risk_per_trade' in parameters:
            risk = parameters['risk_per_trade']
            if not isinstance(risk, (int, float)) or not 0 < risk <= 1:
                raise ValueError("risk_per_trade must be between 0 and 1")
            validated['risk_per_trade'] = float(risk)

        if 'stop_loss_percentage' in parameters:
            stop_loss = parameters['stop_loss_percentage']
            if not isinstance(stop_loss, (int, float)) or not 0 < stop_loss <= 0.5:
                raise ValueError("stop_loss_percentage must be between 0 and 0.5")
            validated['stop_loss_percentage'] = float(stop_loss)

        # 策略特定参数验证（由子类实现）
        validated.update(self._validate_strategy_specific_parameters(parameters))

        return validated

    def _validate_strategy_specific_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证策略特定参数（子类可重写）"""
        return {}

    def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            # 设置基本配置
            if 'symbols' in config:
                self._supported_symbols = config['symbols']

            if 'parameters' in config:
                self.set_parameters(config['parameters'])

            if 'description' in config:
                self._description = config['description']

            self._status = StrategyStatus.INITIALIZED
            self._last_update = datetime.now()

            logger.info(f"Strategy {self.strategy_id} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize strategy {self.strategy_id}: {e}")
            self._status = StrategyStatus.ERROR
            return False

    def on_market_data(self, data: MarketData) -> List[StrategySignal]:
        """处理市场数据并生成信号"""
        if not self.validate_market_data(data):
            return []

        # 调用策略特定的信号生成逻辑
        signals = self._generate_signals_from_market_data(data)

        # 更新性能指标
        self._metrics['total_signals'] += len(signals)

        return signals

    def _generate_signals_from_market_data(self, data: MarketData) -> List[StrategySignal]:
        """从市场数据生成信号（子类必须实现）"""
        raise NotImplementedError("Subclasses must implement _generate_signals_from_market_data")

    def on_order_update(self, order: StrategyOrder) -> None:
        """处理订单更新"""
        logger.info(
            f"Order update received: {order.order_id} - {order.side} {order.quantity} {order.symbol}")

        # 从待处理订单中移除
        self._pending_orders = [o for o in self._pending_orders if o.order_id != order.order_id]

        # 如果是成交订单，更新持仓
        if order.side == 'BUY':
            self._update_position_on_buy(order)
        elif order.side == 'SELL':
            self._update_position_on_sell(order)

    def on_position_update(self, position: StrategyPosition) -> None:
        """处理持仓更新"""
        # 更新或添加持仓
        existing_pos = next(
            (p for p in self._current_positions if p.symbol == position.symbol), None)

        if existing_pos:
            # 更新现有持仓
            existing_pos.quantity = position.quantity
            existing_pos.current_price = position.current_price
            existing_pos.unrealized_pnl = position.unrealized_pnl
            existing_pos.timestamp = position.timestamp
        else:
            # 添加新持仓
            self._current_positions.append(position)

        # 移除空头寸
        self._current_positions = [p for p in self._current_positions if p.quantity != 0]

    def _update_position_on_buy(self, order: StrategyOrder) -> None:
        """买入订单成交时更新持仓"""
        existing_pos = next((p for p in self._current_positions if p.symbol == order.symbol), None)

        if existing_pos:
            # 计算新的平均价格
            total_quantity = existing_pos.quantity + order.quantity
            total_cost = (existing_pos.quantity * existing_pos.average_price) + \
                (order.quantity * order.price)
            new_avg_price = total_cost / total_quantity

            existing_pos.quantity = total_quantity
            existing_pos.average_price = new_avg_price
            existing_pos.current_price = order.price
        else:
            # 创建新持仓
            position = StrategyPosition(
                symbol=order.symbol,
                quantity=order.quantity,
                average_price=order.price,
                current_price=order.price,
                unrealized_pnl=0.0,
                timestamp=datetime.now()
            )
            self._current_positions.append(position)

    def _update_position_on_sell(self, order: StrategyOrder) -> None:
        """卖出订单成交时更新持仓"""
        existing_pos = next((p for p in self._current_positions if p.symbol == order.symbol), None)

        if existing_pos and existing_pos.quantity >= order.quantity:
            # 计算已实现收益
            realized_pnl = (order.price - existing_pos.average_price) * order.quantity
            self._metrics['total_pnl'] += realized_pnl
            self._metrics['successful_trades'] += 1

            # 更新持仓
            existing_pos.quantity -= order.quantity
            if existing_pos.quantity == 0:
                self._current_positions.remove(existing_pos)

    def should_enter_position(self, symbol: str, data: pd.DataFrame) -> Optional[StrategySignal]:
        """判断是否应该开仓（子类可重写）"""
        # 默认实现：检查是否有强烈的买入信号
        if len(data) < 2:
            return None

        current_price = data.iloc[-1]['close']
        prev_price = data.iloc[-2]['close']

        # 简单的上涨趋势检测
        if current_price > prev_price * 1.02:  # 上涨2%以上
            return StrategySignal(
                signal_type='BUY',
                symbol=symbol,
                price=current_price,
                quantity=int(self.calculate_position_size(100000, 0.02, symbol)),
                confidence=0.7,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id
            )

        return None

    def should_exit_position(self, position: StrategyPosition, data: pd.DataFrame) -> Optional[StrategySignal]:
        """判断是否应该平仓（子类可重写）"""
        if not data.empty:
            current_price = data.iloc[-1]['close']

            # 检查止损
            stop_loss_pct = self._parameters.get('stop_loss_percentage', 0.05)
            if current_price < position.average_price * (1 - stop_loss_pct):
                return StrategySignal(
                    signal_type='SELL',
                    symbol=position.symbol,
                    price=current_price,
                    quantity=position.quantity,
                    confidence=0.9,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id
                )

            # 检查止盈
            take_profit_pct = self._parameters.get('take_profit_percentage', 0.10)
            if current_price > position.average_price * (1 + take_profit_pct):
                return StrategySignal(
                    signal_type='SELL',
                    symbol=position.symbol,
                    price=current_price,
                    quantity=position.quantity,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id
                )

        return None

    def calculate_position_size(self, capital: float, risk_per_trade: float, symbol: str) -> float:
        """计算仓位大小"""
        max_pos_size = self._parameters.get('max_position_size', capital * 0.1)  # 默认10%资本
        risk_amount = capital * risk_per_trade

        # 简单的仓位计算：风险金额 / (价格 * 止损百分比)
        # 这里简化处理，返回固定比例
        return min(max_pos_size, risk_amount)

    def get_risk_management_rules(self) -> Dict[str, Any]:
        """获取风险管理规则"""
        return {
            'max_position_size': self._parameters.get('max_position_size', 10000),
            'risk_per_trade': self._parameters.get('risk_per_trade', 0.02),
            'stop_loss_percentage': self._parameters.get('stop_loss_percentage', 0.05),
            'max_open_positions': self._parameters.get('max_open_positions', 5),
            'max_daily_loss': self._parameters.get('max_daily_loss', 1000)
        }

    def get_strategy_status(self) -> str:
        return self._status

    def get_current_positions(self) -> List[StrategyPosition]:
        return self._current_positions.copy()

    def get_pending_orders(self) -> List[StrategyOrder]:
        return self._pending_orders.copy()

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """获取策略性能指标"""
        metrics = self._metrics.copy()

        # 计算胜率
        if metrics['total_signals'] > 0:
            metrics['win_rate'] = metrics['successful_trades'] / metrics['total_signals']

        # 计算夏普比率（简化计算）
        if metrics['total_signals'] > 0:
            metrics['sharpe_ratio'] = metrics['total_pnl'] / (metrics['total_signals'] ** 0.5)

        return metrics

    def start(self) -> bool:
        if self._status == StrategyStatus.INITIALIZED:
            self._status = StrategyStatus.RUNNING
            self._is_running = True
            logger.info(f"Strategy {self.strategy_id} started")
            return True
        return False

    def stop(self) -> bool:
        self._status = StrategyStatus.STOPPED
        self._is_running = False
        logger.info(f"Strategy {self.strategy_id} stopped")
        return True

    def pause(self) -> bool:
        if self._status == StrategyStatus.RUNNING:
            self._status = StrategyStatus.PAUSED
            logger.info(f"Strategy {self.strategy_id} paused")
            return True
        return False

    def resume(self) -> bool:
        if self._status == StrategyStatus.PAUSED:
            self._status = StrategyStatus.RUNNING
            logger.info(f"Strategy {self.strategy_id} resumed")
            return True
        return False

    def reset(self) -> bool:
        self._current_positions.clear()
        self._pending_orders.clear()
        self._metrics = {k: 0 if isinstance(v, (int, float))
                         else v for k, v in self._metrics.items()}
        self._status = StrategyStatus.INITIALIZED
        logger.info(f"Strategy {self.strategy_id} reset")
        return True

    def save_state(self, path: str) -> bool:
        try:
            state = {
                'strategy_id': self.strategy_id,
                'status': self._status,
                'parameters': self._parameters,
                'positions': [asdict(p) for p in self._current_positions],
                'pending_orders': [asdict(o) for o in self._pending_orders],
                'metrics': self._metrics,
                'last_update': self._last_update.isoformat()
            }

            with open(path, 'w') as f:
                json.dump(state, f, default=str, indent=2)

            logger.info(f"Strategy state saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save strategy state: {e}")
            return False

    def load_state(self, path: str) -> bool:
        try:
            with open(path, 'r') as f:
                state = json.load(f)

            self.strategy_id = state['strategy_id']
            self._status = state['status']
            self._parameters = state['parameters']

            # 恢复持仓
            self._current_positions = []
            for pos_data in state.get('positions', []):
                pos_data['timestamp'] = datetime.fromisoformat(pos_data['timestamp'])
                self._current_positions.append(StrategyPosition(**pos_data))

            # 恢复待处理订单
            self._pending_orders = []
            for order_data in state.get('pending_orders', []):
                order_data['timestamp'] = datetime.fromisoformat(order_data['timestamp'])
                self._pending_orders.append(StrategyOrder(**order_data))

            self._metrics = state.get('metrics', self._metrics)
            self._last_update = datetime.fromisoformat(state['last_update'])

            logger.info(f"Strategy state loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load strategy state: {e}")
            return False

    def get_supported_symbols(self) -> List[str]:
        return self._supported_symbols.copy()

    def get_required_data_fields(self) -> List[str]:
        return self._required_data_fields.copy()

    def validate_market_data(self, data: MarketData) -> bool:
        """验证市场数据"""
        if not data.symbol or not data.price or data.price <= 0:
            return False

        if data.timestamp > datetime.now() + timedelta(minutes=1):  # 允许1分钟的数据延迟
            return False

        return True


# 趋势跟踪策略示例
class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""

    def __init__(self, strategy_id: str):
        super().__init__(strategy_id, "Trend Following", StrategyType.TREND_FOLLOWING)
        self._short_ma_period = 20
        self._long_ma_period = 50
        self._price_history: Dict[str, List[float]] = {}

    def _validate_strategy_specific_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证趋势跟踪策略特定参数"""
        validated = {}

        if 'short_ma_period' in parameters:
            period = parameters['short_ma_period']
            if not isinstance(period, int) or period <= 0:
                raise ValueError("short_ma_period must be a positive integer")
            validated['short_ma_period'] = period
            self._short_ma_period = period

        if 'long_ma_period' in parameters:
            period = parameters['long_ma_period']
            if not isinstance(period, int) or period <= 0:
                raise ValueError("long_ma_period must be a positive integer")
            validated['long_ma_period'] = period
            self._long_ma_period = period

        return validated

    def _generate_signals_from_market_data(self, data: MarketData) -> List[StrategySignal]:
        """基于市场数据生成趋势跟踪信号"""
        symbol = data.symbol

        # 维护价格历史
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append(data.price)

        # 保持最近的价格数据
        max_period = max(self._short_ma_period, self._long_ma_period)
        if len(self._price_history[symbol]) > max_period * 2:
            self._price_history[symbol] = self._price_history[symbol][-max_period * 2:]

        # 需要足够的数据来计算移动平均
        if len(self._price_history[symbol]) < self._long_ma_period:
            return []

        prices = self._price_history[symbol]

        # 计算移动平均
        short_ma = np.mean(prices[-self._short_ma_period:])
        long_ma = np.mean(prices[-self._long_ma_period:])

        signals = []

        # 生成信号
        if short_ma > long_ma * 1.001:  # 短期均线上穿长期均线1%
            signals.append(StrategySignal(
                signal_type='BUY',
                symbol=symbol,
                price=data.price,
                quantity=int(self.calculate_position_size(100000, 0.02, symbol)),
                confidence=0.75,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id
            ))

        elif short_ma < long_ma * 0.999:  # 短期均线下穿长期均线1%
            # 检查是否有持仓需要平仓
            position = next((p for p in self._current_positions if p.symbol == symbol), None)
            if position and position.quantity > 0:
                signals.append(StrategySignal(
                    signal_type='SELL',
                    symbol=symbol,
                    price=data.price,
                    quantity=position.quantity,
                    confidence=0.75,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id
                ))

        return signals

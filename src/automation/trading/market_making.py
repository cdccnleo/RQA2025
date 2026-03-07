"""
Market Making Automation Module
市商自动化模块

This module provides automated market making capabilities for quantitative trading
此模块为量化交易提供自动化做市能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class MarketMakingStrategy(Enum):

    """Market making strategy types"""
    PASSIVE = "passive"           # Passive market making
    AGGRESSIVE = "aggressive"     # Aggressive market making
    HYBRID = "hybrid"            # Hybrid approach
    DYNAMIC = "dynamic"          # Dynamic spread adjustment
    INVENTORY = "inventory"      # Inventory - based pricing


class OrderSide(Enum):

    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):

    """Order type enumeration"""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class MarketMakingOrder:

    """
    Market making order data class
    做市订单数据类
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: str
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class MarketMakingQuote:

    """
    Market making quote data class
    做市报价数据类
    """
    symbol: str
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    spread: float
    timestamp: datetime
    strategy_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MarketMakingStrategy:

    """
    Market Making Strategy Class
    做市策略类

    Implements different market making strategies
    实现不同的做市策略
    """

    def __init__(self,


                 strategy_id: str,
                 symbol: str,
                 strategy_type: MarketMakingStrategy,
                 config: Dict[str, Any]):
        """
        Initialize market making strategy
        初始化做市策略

        Args:
            strategy_id: Unique strategy identifier
                        唯一策略标识符
            symbol: Trading symbol
                   交易符号
            strategy_type: Type of market making strategy
                          做市策略类型
            config: Strategy configuration
                   策略配置
        """
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.config = config

        # Strategy parameters
        self.base_spread = config.get('base_spread', 0.001)  # 0.1%
        self.max_spread = config.get('max_spread', 0.01)    # 1%
        self.min_spread = config.get('min_spread', 0.0001)  # 0.01%
        self.quote_size = config.get('quote_size', 100)
        self.max_inventory = config.get('max_inventory', 1000)
        self.inventory_target = config.get('inventory_target', 0)

        # Risk parameters
        self.max_position = config.get('max_position', 5000)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 0.05)  # 5%

        # Current state
        self.current_inventory = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_volume = 0

        # Performance tracking
        self.quotes_generated = 0
        self.orders_filled = 0
        self.last_quote_time: Optional[datetime] = None

    def generate_quotes(self,


                        market_data: Dict[str, Any],
                        current_inventory: int) -> MarketMakingQuote:
        """
        Generate market making quotes
        生成做市报价

        Args:
            market_data: Current market data
                        当前市场数据
            current_inventory: Current inventory position
                             当前库存头寸

        Returns:
            MarketMakingQuote: Generated quote
                              生成的报价
        """
        self.current_inventory = current_inventory
        mid_price = market_data.get('mid_price', 0)
        spread_pct = market_data.get('spread_pct', 0.001)

        if mid_price <= 0:
            raise ValueError("Invalid mid price for quote generation")

        # Calculate dynamic spread based on strategy and market conditions
        dynamic_spread = self._calculate_dynamic_spread(market_data, current_inventory)

        # Calculate bid and ask prices
        half_spread = dynamic_spread / 2
        bid_price = mid_price * (1 - half_spread)
        ask_price = mid_price * (1 + half_spread)

        # Adjust for inventory
        bid_price, ask_price = self._adjust_for_inventory(bid_price, ask_price, current_inventory)

        # Ensure minimum spread
        actual_spread = (ask_price - bid_price) / mid_price
        if actual_spread < self.min_spread:
            mid_adjusted = (bid_price + ask_price) / 2
            half_min_spread = self.min_spread / 2
            bid_price = mid_adjusted * (1 - half_min_spread)
            ask_price = mid_adjusted * (1 + half_min_spread)

        # Calculate quote sizes
        bid_quantity = self._calculate_quote_size(OrderSide.BUY, market_data)
        ask_quantity = self._calculate_quote_size(OrderSide.SELL, market_data)

        quote = MarketMakingQuote(
            symbol=self.symbol,
            bid_price=round(bid_price, 6),
            bid_quantity=round(bid_quantity, 2),
            ask_price=round(ask_price, 6),
            ask_quantity=round(ask_quantity, 2),
            spread=round(actual_spread, 6),
            timestamp=datetime.now(),
            strategy_id=self.strategy_id
        )

        self.quotes_generated += 1
        self.last_quote_time = quote.timestamp

        return quote

    def _calculate_dynamic_spread(self,


                                  market_data: Dict[str, Any],
                                  inventory: int) -> float:
        """
        Calculate dynamic spread based on market conditions
        根据市场条件计算动态价差

        Args:
            market_data: Market data
                        市场数据
            inventory: Current inventory
                      当前库存

        Returns:
            float: Dynamic spread percentage
                   动态价差百分比
        """
        base_spread = self.base_spread

        # Adjust for volatility
        volatility = market_data.get('volatility', 0.02)
        volatility_multiplier = min(max(volatility / 0.02, 0.5), 2.0)
        spread = base_spread * volatility_multiplier

        # Adjust for volume
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volume_multiplier = 1.0 / min(max(volume_ratio, 0.1), 2.0)
        spread *= volume_multiplier

        # Adjust for inventory
        inventory_ratio = abs(inventory) / max(abs(self.max_inventory), 1)
        inventory_multiplier = 1.0 + (inventory_ratio * 0.5)
        spread *= inventory_multiplier

        # Strategy - specific adjustments
        if self.strategy_type == MarketMakingStrategy.PASSIVE:
            spread *= 1.2  # Wider spread for passive strategy
        elif self.strategy_type == MarketMakingStrategy.AGGRESSIVE:
            spread *= 0.8  # Narrower spread for aggressive strategy
        elif self.strategy_type == MarketMakingStrategy.DYNAMIC:
            # Dynamic adjustment based on market conditions
            trend = market_data.get('trend', 0)
            if abs(trend) > 0.001:  # Significant trend
                spread *= 1.1

        return min(max(spread, self.min_spread), self.max_spread)

    def _adjust_for_inventory(self,


                              bid_price: float,
                              ask_price: float,
                              inventory: int) -> tuple:
        """
        Adjust prices based on inventory position
        根据库存头寸调整价格

        Args:
            bid_price: Current bid price
                       当前买入价
            ask_price: Current ask price
                      当前卖出价
            inventory: Current inventory
                      当前库存

        Returns:
            tuple: Adjusted (bid_price, ask_price)
                  调整后的(买入价, 卖出价)
        """
        if inventory == 0:
            return bid_price, ask_price

        # Inventory skew adjustment
        inventory_ratio = inventory / max(abs(self.max_inventory), 1)
        skew_factor = min(max(inventory_ratio * 0.002, -0.005), 0.005)  # Max 0.5% adjustment

        mid_price = (bid_price + ask_price) / 2
        bid_price = mid_price * (1 - skew_factor)
        ask_price = mid_price * (1 + skew_factor)

        return bid_price, ask_price

    def _calculate_quote_size(self,


                              side: OrderSide,
                              market_data: Dict[str, Any]) -> float:
        """
        Calculate quote size for given side
        计算给定方向的报价数量

        Args:
            side: Order side
                 订单方向
            market_data: Market data
                        市场数据

        Returns:
            float: Quote size
                   报价数量
        """
        base_size = self.quote_size

        # Adjust for market conditions
        spread_pct = market_data.get('spread_pct', 0.001)
        size_multiplier = 1.0 / (1.0 + spread_pct * 100)  # Reduce size in wide spreads

        # Adjust for inventory
        if side == OrderSide.BUY and self.current_inventory > 0:
            size_multiplier *= 0.8  # Reduce buy size if long
        elif side == OrderSide.SELL and self.current_inventory < 0:
            size_multiplier *= 0.8  # Reduce sell size if short

        return base_size * size_multiplier

    def update_performance(self,


                           filled_order: MarketMakingOrder,
                           fill_price: float) -> None:
        """
        Update strategy performance after order fill
        订单成交后更新策略表现

        Args:
            filled_order: Filled order details
                         成交订单详情
            fill_price: Actual fill price
                       实际成交价格
        """
        self.orders_filled += 1

        # Update inventory
        quantity = filled_order.quantity
        if filled_order.side == OrderSide.BUY:
            self.current_inventory += quantity
            self.realized_pnl -= quantity * fill_price  # Cost basis
        else:
            self.current_inventory -= quantity
            self.realized_pnl += quantity * fill_price  # Revenue

    def get_strategy_status(self) -> Dict[str, Any]:
        """
        Get strategy status and performance metrics
        获取策略状态和性能指标

        Returns:
            dict: Strategy status
                  策略状态
        """
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'strategy_type': self.strategy_type.value,
            'current_inventory': self.current_inventory,
            'inventory_target': self.inventory_target,
            'realized_pnl': round(self.realized_pnl, 2),
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'total_pnl': round(self.realized_pnl + self.unrealized_pnl, 2),
            'total_volume': self.total_volume,
            'quotes_generated': self.quotes_generated,
            'orders_filled': self.orders_filled,
            'fill_rate': self.orders_filled / max(self.quotes_generated, 1) * 100,
            'last_quote_time': self.last_quote_time.isoformat() if self.last_quote_time else None
        }


class MarketMakingEngine:

    """
    Market Making Engine Class
    做市引擎类

    Core engine for automated market making
    自动化做市的核心引擎
    """

    def __init__(self, engine_name: str = "default_market_making_engine"):
        """
        Initialize market making engine
        初始化做市引擎

        Args:
            engine_name: Name of the engine
                        引擎名称
        """
        self.engine_name = engine_name
        self.strategies: Dict[str, MarketMakingStrategy] = {}
        self.active_quotes: Dict[str, MarketMakingQuote] = {}
        self.pending_orders: Dict[str, MarketMakingOrder] = {}

        # Engine settings
        self.is_running = False
        self.quote_update_interval = 5.0  # seconds
        self.max_strategies_per_symbol = 3

        # Performance tracking
        self.total_quotes_generated = 0
        self.total_orders_filled = 0
        self.engine_start_time: Optional[datetime] = None

        # Risk management
        self.global_stop_loss_threshold = 0.10  # 10%
        self.max_total_exposure = 10000

        logger.info(f"Market making engine {engine_name} initialized")

    def add_strategy(self, strategy: MarketMakingStrategy) -> None:
        """
        Add a market making strategy
        添加做市策略

        Args:
            strategy: Strategy to add
                     要添加的策略
        """
        # Check symbol limit
        symbol_strategies = [s for s in self.strategies.values() if s.symbol == strategy.symbol]
        if len(symbol_strategies) >= self.max_strategies_per_symbol:
            raise ValueError(
                f"Maximum strategies per symbol ({self.max_strategies_per_symbol}) exceeded")

        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Added strategy: {strategy.strategy_id} for {strategy.symbol}")

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a market making strategy
        移除做市策略

        Args:
            strategy_id: Strategy identifier
                        策略标识符

        Returns:
            bool: True if removed successfully
                  移除成功返回True
        """
        if strategy_id in self.strategies:
            # Cancel any active quotes
            self._cancel_strategy_quotes(strategy_id)
            del self.strategies[strategy_id]
            logger.info(f"Removed strategy: {strategy_id}")
            return True
        return False

    def start_engine(self) -> bool:
        """
        Start the market making engine
        启动做市引擎

        Returns:
            bool: True if started successfully
                  启动成功返回True
        """
        if self.is_running:
            logger.warning("Market making engine is already running")
            return False

        try:
            self.is_running = True
            self.engine_start_time = datetime.now()

            # Start background quote update thread
            update_thread = threading.Thread(target=self._quote_update_loop, daemon=True)
            update_thread.start()

            logger.info("Market making engine started")
            return True

        except Exception as e:
            logger.error(f"Failed to start market making engine: {str(e)}")
            self.is_running = False
            return False

    def stop_engine(self) -> bool:
        """
        Stop the market making engine
        停止做市引擎

        Returns:
            bool: True if stopped successfully
                  停止成功返回True
        """
        if not self.is_running:
            logger.warning("Market making engine is not running")
            return False

        try:
            self.is_running = False

            # Cancel all active quotes
            self._cancel_all_quotes()

            logger.info("Market making engine stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop market making engine: {str(e)}")
            return False

    def generate_quotes(self,


                        symbol: str,
                        market_data: Dict[str, Any],
                        inventory: int) -> List[MarketMakingQuote]:
        """
        Generate quotes for a symbol
        为符号生成报价

        Args:
            symbol: Trading symbol
                   交易符号
            market_data: Current market data
                        当前市场数据
            inventory: Current inventory position
                      当前库存头寸

        Returns:
            list: List of generated quotes
                  生成的报价列表
        """
        quotes = []
        symbol_strategies = [s for s in self.strategies.values() if s.symbol == symbol]

        for strategy in symbol_strategies:
            try:
                quote = strategy.generate_quotes(market_data, inventory)
                quotes.append(quote)

                # Store active quote
                quote_key = f"{strategy.strategy_id}_{symbol}"
                self.active_quotes[quote_key] = quote

                self.total_quotes_generated += 1

            except Exception as e:
                logger.error(
                    f"Failed to generate quote for strategy {strategy.strategy_id}: {str(e)}")

        return quotes

    def process_order_fill(self,


                           order_id: str,
                           fill_price: float,
                           fill_quantity: float) -> bool:
        """
        Process an order fill
        处理订单成交

        Args:
            order_id: Order identifier
                     订单标识符
            fill_price: Fill price
                       成交价格
            fill_quantity: Fill quantity
                          成交数量

        Returns:
            bool: True if processed successfully
                  处理成功返回True
        """
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found in pending orders")
            return False

        order = self.pending_orders[order_id]

        # Find corresponding strategy
        if order.strategy_id not in self.strategies:
            logger.error(f"Strategy {order.strategy_id} not found")
            return False

        strategy = self.strategies[order.strategy_id]

        # Update strategy performance
        strategy.update_performance(order, fill_price)
        strategy.total_volume += fill_quantity

        # Remove from pending orders
        del self.pending_orders[order_id]

        self.total_orders_filled += 1

        logger.info(
            f"Processed order fill: {order_id}, price: {fill_price}, quantity: {fill_quantity}")
        return True

    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get engine status and performance metrics
        获取引擎状态和性能指标

        Returns:
            dict: Engine status
                  引擎状态
        """
        total_realized_pnl = sum(s.realized_pnl for s in self.strategies.values())
        total_unrealized_pnl = sum(s.unrealized_pnl for s in self.strategies.values())
        total_volume = sum(s.total_volume for s in self.strategies.values())

        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'total_strategies': len(self.strategies),
            'active_quotes': len(self.active_quotes),
            'pending_orders': len(self.pending_orders),
            'total_quotes_generated': self.total_quotes_generated,
            'total_orders_filled': self.total_orders_filled,
            'fill_rate': self.total_orders_filled / max(self.total_quotes_generated, 1) * 100,
            'total_realized_pnl': round(total_realized_pnl, 2),
            'total_unrealized_pnl': round(total_unrealized_pnl, 2),
            'total_pnl': round(total_realized_pnl + total_unrealized_pnl, 2),
            'total_volume': round(total_volume, 2),
            'engine_uptime': str(datetime.now() - self.engine_start_time) if self.engine_start_time else None
        }

    def get_strategy_status(self, strategy_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get strategy status
        获取策略状态

        Args:
            strategy_id: Specific strategy ID (optional)
                        特定策略ID（可选）

        Returns:
            dict or list: Strategy status(es)
                         策略状态
        """
        if strategy_id:
            if strategy_id in self.strategies:
                return self.strategies[strategy_id].get_strategy_status()
            else:
                return {}
        else:
            return [s.get_strategy_status() for s in self.strategies.values()]

    def _quote_update_loop(self) -> None:
        """
        Background loop for updating quotes
        更新报价的后台循环
        """
        logger.info("Quote update loop started")

        while self.is_running:
            try:
                # This would typically fetch market data and update quotes
                # For now, just sleep
                time.sleep(self.quote_update_interval)

            except Exception as e:
                logger.error(f"Quote update loop error: {str(e)}")
                time.sleep(self.quote_update_interval)

        logger.info("Quote update loop stopped")

    def _cancel_strategy_quotes(self, strategy_id: str) -> None:
        """
        Cancel all quotes for a strategy
        取消策略的所有报价

        Args:
            strategy_id: Strategy identifier
                        策略标识符
        """
        quotes_to_cancel = [k for k in self.active_quotes.keys() if k.startswith(f"{strategy_id}_")]
        for quote_key in quotes_to_cancel:
            del self.active_quotes[quote_key]

        logger.info(f"Cancelled {len(quotes_to_cancel)} quotes for strategy {strategy_id}")

    def _cancel_all_quotes(self) -> None:
        """
        Cancel all active quotes
        取消所有活跃报价
        """
        cancelled_count = len(self.active_quotes)
        self.active_quotes.clear()
        logger.info(f"Cancelled all {cancelled_count} active quotes")

    def create_passive_strategy(self,


                                strategy_id: str,
                                symbol: str,
                                config: Dict[str, Any]) -> str:
        """
        Create a passive market making strategy
        创建被动做市策略

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            symbol: Trading symbol
                   交易符号
            config: Strategy configuration
                   策略配置

        Returns:
            str: Created strategy ID
                 创建的策略ID
        """
        strategy = MarketMakingStrategy(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=MarketMakingStrategy.PASSIVE,
            config=config
        )

        self.add_strategy(strategy)
        return strategy_id

    def create_aggressive_strategy(self,


                                   strategy_id: str,
                                   symbol: str,
                                   config: Dict[str, Any]) -> str:
        """
        Create an aggressive market making strategy
        创建主动做市策略

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            symbol: Trading symbol
                   交易符号
            config: Strategy configuration
                   策略配置

        Returns:
            str: Created strategy ID
                 创建的策略ID
        """
        strategy = MarketMakingStrategy(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=MarketMakingStrategy.AGGRESSIVE,
            config=config
        )

        self.add_strategy(strategy)
        return strategy_id


# Global market making engine instance
# 全局做市引擎实例
market_making_engine = MarketMakingEngine()

__all__ = [
    'MarketMakingStrategy',
    'OrderSide',
    'OrderType',
    'MarketMakingOrder',
    'MarketMakingQuote',
    'MarketMakingStrategy',
    'MarketMakingEngine',
    'market_making_engine'
]

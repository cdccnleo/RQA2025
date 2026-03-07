import logging
#!/usr/bin/env python3
"""
RQA2025 高频交易执行引擎

基于统一基础设施集成层的高频交易引擎实现，
提供低延迟、高性能的交易执行和市场微观结构分析。
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import time
import queue
from collections import deque
import numpy as np

# 导入统一基础设施集成层
try:
    from src.core.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


# 导入内存优化模块
try:
    from src.performance.memory_pool import (
        get_memory_manager,
        acquire_trading_object
    )
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class HFTStrategy(Enum):

    """高频策略枚举"""
    MARKET_MAKING = "market_making"          # 做市策略
    ARBITRAGE = "arbitrage"                  # 套利策略
    MOMENTUM = "momentum"                    # 动量策略
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    STATISTICAL_ARBITRAGE = "stat_arb"      # 统计套利
    ORDER_BOOK = "order_book"                # 订单簿策略


class OrderBookSide(Enum):

    """订单簿方向枚举"""
    BID = "bid"                              # 买单
    ASK = "ask"                              # 卖单


@dataclass
class OrderBookEntry:

    """订单簿条目"""
    price: float
    quantity: float
    timestamp: datetime
    order_count: int = 1


@dataclass
class OrderBook:

    """订单簿"""
    symbol: str
    bids: List[OrderBookEntry]  # 买单，按价格降序
    asks: List[OrderBookEntry]  # 卖单，按价格升序
    timestamp: datetime

    def get_best_bid(self) -> Optional[OrderBookEntry]:
        """获取最佳买价"""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[OrderBookEntry]:
        """获取最佳卖价"""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> float:
        """获取买卖价差"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return 0.0

    def get_mid_price(self) -> float:
        """获取中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return 0.0


@dataclass
class HFTrade:

    """高频交易"""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    latency_us: int  # 微秒级延迟
    strategy: HFTStrategy


@dataclass
class MarketMicrostructure:

    """市场微观结构"""
    symbol: str
    timestamp: datetime

    # 基本指标
    spread: float
    mid_price: float
    volume_imbalance: float
    order_flow_imbalance: float

    # 高级指标
    volatility_5min: float
    price_trend: float
    market_depth: float
    liquidity_score: float

    # 订单簿特征
    bid_levels: int
    ask_levels: int
    bid_volume: float
    ask_volume: float


class HFTExecutionEngine:

    """高频交易执行引擎 - 支持统一基础设施集成"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化高频交易执行引擎"""
        self.config = config or {}

        # 基础设施集成
        self._infrastructure_adapter = None
        self._config_manager = None
        self._cache_manager = None
        self._monitoring = None
        self._logger = None

        # 初始化基础设施集成
        self._init_infrastructure_integration()

        # 初始化内存优化
        self._init_memory_optimization()

        # 从配置中获取参数
        self._load_config()

        # 市场数据
        self.order_books: Dict[str, OrderBook] = {}
        self.market_data: Dict[str, deque] = {}
        self.microstructure: Dict[str, deque] = {}

        # 策略配置
        self.active_strategies: Dict[HFTStrategy, Dict[str, Any]] = {}
        self.strategy_positions: Dict[str, float] = {}

        # 执行队列 - 使用线程安全的队列而不是asyncio.Queue
        self.execution_queue = queue.Queue()
        self.market_data_queue = queue.Queue()

        # 性能监控
        self.performance_stats = {
            'trades_executed': 0,
            'average_latency_us': 0,
            'slippage_bps': 0,
            'pnl_realized': 0.0,
            'risk_violations': 0
        }

        # 控制标志
        self.running = False
        self.emergency_stop = False

        # 线程管理
        self.threads: List[threading.Thread] = []

        logger.info("高频交易执行引擎初始化完成")

    def _init_infrastructure_integration(self):
        """初始化基础设施集成"""
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            logger.warning("统一基础设施集成层不可用，使用降级模式")
            return

        try:
            # 获取交易层适配器
            self._infrastructure_adapter = get_trading_layer_adapter()

            if self._infrastructure_adapter:
                # 获取基础设施服务
                services = self._infrastructure_adapter.get_infrastructure_services()
                self._config_manager = services.get('config_manager')
                self._cache_manager = services.get('cache_manager')
                self._monitoring = services.get('monitoring')
                self._logger = services.get('logger')

                logger.info("高频交易执行引擎成功连接统一基础设施集成层")
            else:
                logger.warning("无法获取交易层适配器")

        except Exception as e:
            logger.error(f"基础设施集成初始化失败: {e}")

    def _init_memory_optimization(self):
        """初始化内存优化"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            logger.warning("内存优化模块不可用，使用标准内存管理")
            return

        try:
            self._memory_manager = get_memory_manager()
            logger.info("HFT引擎成功集成内存优化模块")
        except Exception as e:
            logger.error(f"内存优化初始化失败: {e}")
            self._memory_manager = None

    def _load_config(self):
        """从配置管理器加载配置"""
        try:
            if self._config_manager:
                # 从统一配置管理器获取HFT相关配置
                self.max_position = self._config_manager.get('trading.hft.max_position', 1000)
                self.min_order_size = self._config_manager.get('trading.hft.min_order_size', 0.01)
                self.max_latency_us = self._config_manager.get('trading.hft.max_latency_us', 1000)
                self.risk_limit_per_second = self._config_manager.get(
                    'trading.hft.risk_limit_per_second', 100)
                self.enable_monitoring = self._config_manager.get(
                    'trading.hft.enable_monitoring', True)
                self.enable_caching = self._config_manager.get('trading.hft.enable_caching', True)
            else:
                # 使用默认值
                self.max_position = 1000
                self.min_order_size = 0.01
                self.max_latency_us = 1000
                self.risk_limit_per_second = 100
                self.enable_monitoring = True
                self.enable_caching = True
        except Exception as e:
            logger.warning(f"配置加载失败，使用默认值: {e}")
            self.max_position = 1000
            self.min_order_size = 0.01
            self.max_latency_us = 1000
            self.risk_limit_per_second = 100
            self.enable_monitoring = True
            self.enable_caching = True

    def start_engine(self):
        """启动引擎"""
        if self.running:
            logger.warning("引擎已在运行中")
            return

        self.running = True
        self.emergency_stop = False

        # 启动核心线程
        threads_config = [
            ('market_data_processor', self._market_data_processor),
            ('strategy_executor', self._strategy_executor),
            ('risk_manager', self._risk_manager),
            ('performance_monitor', self._performance_monitor)
        ]

        for thread_name, target_func in threads_config:
            thread = threading.Thread(target=target_func, name=thread_name, daemon=True)
            thread.start()
            self.threads.append(thread)

        # 基础设施集成：记录监控指标
        if self.enable_monitoring and self._monitoring:
            try:
                self._monitoring.record_metric(
                    'hft_engine_start',
                    1,
                    {
                        'engine_type': 'HFTExecutionEngine',
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                logger.warning(f"记录引擎启动指标失败: {e}")

        logger.info("高频交易执行引擎已启动")

    def stop_engine(self):
        """停止引擎"""
        if not self.running:
            return

        logger.info("正在停止高频交易执行引擎...")
        self.running = False

        # 等待线程结束
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.threads.clear()

        # 基础设施集成：记录监控指标
        if self.enable_monitoring and self._monitoring:
            try:
                self._monitoring.record_metric(
                    'hft_engine_stop',
                    1,
                    {
                        'engine_type': 'HFTExecutionEngine',
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                logger.warning(f"记录引擎停止指标失败: {e}")

        logger.info("高频交易执行引擎已停止")

    def register_strategy(self, strategy: HFTStrategy, config: Dict[str, Any]):
        """注册策略"""
        self.active_strategies[strategy] = config

        # 基础设施集成：记录策略注册指标
        if self.enable_monitoring and self._monitoring:
            try:
                self._monitoring.record_metric(
                    'hft_strategy_registered',
                    1,
                    {
                        'strategy': strategy.value,
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                logger.warning(f"记录策略注册指标失败: {e}")

        logger.info(f"注册高频策略: {strategy.value}")

    def update_order_book(self, symbol: str, bids: List[Tuple[float, float]],


                          asks: List[Tuple[float, float]], timestamp: datetime):
        """更新订单簿（使用内存池优化）"""
        # 使用内存池创建订单簿条目
        bid_entries = []
        ask_entries = []

        if MEMORY_OPTIMIZATION_AVAILABLE and self._memory_manager:
            # 从内存池获取订单簿条目对象
            for price, qty in bids:
                entry = acquire_trading_object("orderbook_entry_pool")
                if entry:
                    entry.price = price
                    entry.quantity = qty
                    entry.timestamp = timestamp
                    entry.order_count = 1
                    bid_entries.append(entry)
                else:
                    # 内存池获取失败，使用标准创建
                    bid_entries.append(OrderBookEntry(price, qty, timestamp))

            for price, qty in asks:
                entry = acquire_trading_object("orderbook_entry_pool")
                if entry:
                    entry.price = price
                    entry.quantity = qty
                    entry.timestamp = timestamp
                    entry.order_count = 1
                    ask_entries.append(entry)
                else:
                    # 内存池获取失败，使用标准创建
                    ask_entries.append(OrderBookEntry(price, qty, timestamp))
        else:
            # 内存优化不可用，使用标准创建
            bid_entries = [OrderBookEntry(price, qty, timestamp) for price, qty in bids]
            ask_entries = [OrderBookEntry(price, qty, timestamp) for price, qty in asks]

        order_book = OrderBook(
            symbol=symbol,
            bids=sorted(bid_entries, key=lambda x: x.price, reverse=True),
            asks=sorted(ask_entries, key=lambda x: x.price),
            timestamp=timestamp
        )

        self.order_books[symbol] = order_book

        # 计算微观结构
        microstructure = self._calculate_microstructure(order_book)
        if symbol not in self.microstructure:
            self.microstructure[symbol] = deque(maxlen=1000)
        self.microstructure[symbol].append(microstructure)

        # 基础设施集成：缓存订单簿数据
        if self.enable_caching and self._cache_manager:
            try:
                cache_key = f"hft_orderbook_{symbol}"
                cache_data = {
                    'symbol': symbol,
                    'bids': [(e.price, e.quantity) for e in order_book.bids[:10]],  # 只缓存前10档
                    'asks': [(e.price, e.quantity) for e in order_book.asks[:10]],
                    'timestamp': timestamp.isoformat(),
                    'mid_price': order_book.get_mid_price(),
                    'spread': order_book.get_spread()
                }
                self._cache_manager.set(cache_key, cache_data, ttl=30)  # 缓存30秒
            except Exception as e:
                logger.warning(f"缓存订单簿数据失败: {e}")

    def execute_trade(self, symbol: str, side: str, quantity: float,


                      strategy: HFTStrategy, max_latency_us: int = None) -> Optional[HFTrade]:
        """执行交易"""
        if not self.running:
            return None

        start_time = time.time_ns() // 1000  # 微秒

        try:
            # 检查风险限制
            if not self._check_risk_limits(symbol, quantity):
                return None

            # 获取订单簿
            order_book = self.order_books.get(symbol)
            if not order_book:
                return None

            # 确定执行价格
            execution_price = self._get_execution_price(order_book, side, quantity)

            if not execution_price:
                return None

            # 创建交易记录（使用内存池优化）
            if MEMORY_OPTIMIZATION_AVAILABLE and self._memory_manager:
                trade = acquire_trading_object("trade_pool")
                if trade:
                    # 重用对象，更新属性
                    trade.trade_id = self._generate_trade_id()
                    trade.symbol = symbol
                    trade.side = side
                    trade.quantity = quantity
                    trade.price = execution_price
                    trade.timestamp = datetime.now()
                    trade.latency_us = 0  # 会在下面计算
                    trade.strategy = strategy
                else:
                    # 内存池获取失败，使用标准创建
                    trade = HFTrade(
                        trade_id=self._generate_trade_id(),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=datetime.now(),
                        latency_us=0,
                        strategy=strategy
                    )
            else:
                # 内存优化不可用，使用标准创建
                trade = HFTrade(
                    trade_id=self._generate_trade_id(),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=datetime.now(),
                    latency_us=0,
                    strategy=strategy
                )

            # 计算延迟
            end_time = time.time_ns() // 1000
            trade.latency_us = int(end_time - start_time)

            # 检查延迟限制
            if max_latency_us and trade.latency_us > max_latency_us:
                logger.warning(f"交易延迟过高: {trade.latency_us}us")
                return None

            # 更新统计
            self._update_performance_stats(trade)

            # 基础设施集成：记录交易监控指标
            if self.enable_monitoring and self._monitoring:
                try:
                    self._monitoring.record_metric(
                        'hft_trade_executed',
                        1,
                        {
                            'trade_id': trade.trade_id,
                            'symbol': symbol,
                            'strategy': strategy.value,
                            'latency_us': trade.latency_us,
                            'layer': 'trading'
                        }
                    )
                except Exception as e:
                    logger.warning(f"记录交易指标失败: {e}")

            # 放入执行队列
            self.execution_queue.put_nowait(trade)

            logger.debug(f"执行高频交易: {trade.trade_id}, 延迟: {trade.latency_us}us")
            return trade

        except Exception as e:
            logger.error(f"交易执行失败: {e}")

            # 基础设施集成：记录错误指标
            if self.enable_monitoring and self._monitoring:
                try:
                    self._monitoring.record_metric(
                        'hft_trade_error',
                        1,
                        {
                            'symbol': symbol,
                            'strategy': strategy.value,
                            'error': str(e),
                            'layer': 'trading'
                        }
                    )
                except Exception as monitor_e:
                    logger.warning(f"记录交易错误指标失败: {monitor_e}")

            return None

    def _calculate_microstructure(self, order_book: OrderBook) -> MarketMicrostructure:
        """计算市场微观结构"""
        # 计算基本指标
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()

        # 计算买卖量不平衡
        bid_volume = sum(entry.quantity for entry in order_book.bids[:5])
        ask_volume = sum(entry.quantity for entry in order_book.asks[:5])

        if bid_volume + ask_volume > 0:
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            volume_imbalance = 0.0

        # 计算订单流不平衡（简化的实现）
        order_flow_imbalance = volume_imbalance * 0.8  # 权重衰减

        # 计算波动率（5分钟）
        symbol = order_book.symbol
        data_queue = self.market_data.get(symbol, [])
        if len(data_queue) >= 60:  # 假设1秒一个数据点
            recent_prices = [d['price'] for d in list(data_queue)[-60:]]
            volatility_5min = np.std(recent_prices) / np.mean(recent_prices)
        else:
            volatility_5min = 0.0

        # 计算价格趋势
        if len(data_queue) >= 20:
            prices = [d['price'] for d in list(data_queue)[-20:]]
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        else:
            price_trend = 0.0

        # 计算市场深度
        market_depth = min(len(order_book.bids), len(order_book.asks))

        # 计算流动性评分
        spread_score = 1.0 / (1.0 + spread * 100)  # 价差越小评分越高
        volume_score = min(bid_volume + ask_volume, 1000) / 1000  # 标准化成交量
        liquidity_score = (spread_score + volume_score) / 2

        return MarketMicrostructure(
            symbol=order_book.symbol,
            timestamp=order_book.timestamp,
            spread=spread,
            mid_price=mid_price,
            volume_imbalance=volume_imbalance,
            order_flow_imbalance=order_flow_imbalance,
            volatility_5min=volatility_5min,
            price_trend=price_trend,
            market_depth=market_depth,
            liquidity_score=liquidity_score,
            bid_levels=len(order_book.bids),
            ask_levels=len(order_book.asks),
            bid_volume=bid_volume,
            ask_volume=ask_volume
        )

    def _get_execution_price(self, order_book: OrderBook, side: str, quantity: float) -> Optional[float]:
        """获取执行价格"""
        if side == 'buy':
            # 买单使用卖一价
            if order_book.asks:
                return order_book.asks[0].price
        else:
            # 卖单使用买一价
            if order_book.bids:
                return order_book.bids[0].price

        return None

    def _check_risk_limits(self, symbol: str, quantity: float) -> bool:
        """检查风险限制"""
        current_position = self.strategy_positions.get(symbol, 0)

        # 检查持仓限制
        if abs(current_position + quantity) > self.max_position:
            logger.warning(f"持仓超出限制: {abs(current_position + quantity)} > {self.max_position}")
            return False

        # 检查订单大小
        if quantity < self.min_order_size:
            logger.warning(f"订单大小过小: {quantity} < {self.min_order_size}")
            return False

        return True

    def _update_performance_stats(self, trade: HFTrade):
        """更新性能统计"""
        self.performance_stats['trades_executed'] += 1

        # 更新平均延迟
        if self.performance_stats['trades_executed'] == 1:
            self.performance_stats['average_latency_us'] = trade.latency_us
        else:
            old_avg = self.performance_stats['average_latency_us']
            count = self.performance_stats['trades_executed']
            self.performance_stats['average_latency_us'] = (
                old_avg * (count - 1) + trade.latency_us) / count

    def _market_data_processor(self):
        """市场数据处理器"""
        while self.running:
            try:
                # 处理市场数据更新
                for symbol in list(self.market_data.keys()):
                    data_queue = self.market_data[symbol]
                    if len(data_queue) < 2:
                        continue

                    # 计算市场指标
                    self._update_market_indicators(symbol)

                time.sleep(0.001)  # 1ms 循环

            except Exception as e:
                logger.error(f"市场数据处理异常: {e}")
                time.sleep(0.01)

    def _strategy_executor(self):
        """策略执行器"""
        while self.running:
            try:
                # 执行活跃策略
                for strategy, config in self.active_strategies.items():
                    self._execute_strategy(strategy, config)

                time.sleep(0.0001)  # 100us 循环

            except Exception as e:
                logger.error(f"策略执行异常: {e}")
                time.sleep(0.001)

    def _execute_strategy(self, strategy: HFTStrategy, config: Dict[str, Any]):
        """执行具体策略"""
        symbols = config.get('symbols', [])

        for symbol in symbols:
            try:
                if strategy == HFTStrategy.MARKET_MAKING:
                    self._execute_market_making(symbol, config, strategy)
                elif strategy == HFTStrategy.ARBITRAGE:
                    self._execute_arbitrage(symbol, config, strategy)
                elif strategy == HFTStrategy.MOMENTUM:
                    self._execute_momentum(symbol, config, strategy)
                elif strategy == HFTStrategy.ORDER_BOOK:
                    self._execute_order_book_strategy(symbol, config, strategy)

            except Exception as e:
                logger.error(f"策略 {strategy.value} 执行失败: {e}")

    def _execute_market_making(self, symbol: str, config: Dict[str, Any], strategy: HFTStrategy):
        """执行做市策略"""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return

        spread_target = config.get('spread_target', 0.001)  # 10bps
        position_limit = config.get('position_limit', 100)
        inventory_skew = config.get('inventory_skew', 0.1)

        current_position = self.strategy_positions.get(symbol, 0)
        current_spread = order_book.get_spread()

        # 如果价差足够大，调整报价
        if current_spread > spread_target:
            mid_price = order_book.get_mid_price()

            # 根据库存调整报价
            if current_position > position_limit * 0.8:
                # 多头过重，调低买价，调高卖价
                bid_price = mid_price * (1 - spread_target / 2 - inventory_skew)
                ask_price = mid_price * (1 + spread_target / 2 + inventory_skew)
            elif current_position < -position_limit * 0.8:
                # 空头过重，调高买价，调低卖价
                bid_price = mid_price * (1 - spread_target / 2 + inventory_skew)
                ask_price = mid_price * (1 + spread_target / 2 - inventory_skew)
            else:
                # 正常做市
                bid_price = mid_price * (1 - spread_target / 2)
                ask_price = mid_price * (1 + spread_target / 2)

            # 提交做市订单（通过交易层适配器）
            if self._infrastructure_adapter:
                try:
                    # 使用交易层适配器的订单管理器
                    order_manager = self._infrastructure_adapter.get_order_manager()
                    if order_manager:
                        # 创建买单
                        buy_order = order_manager.create_order(
                            symbol=symbol,
                            quantity=position_limit * 0.1,
                            order_type=order_manager.OrderType.LIMIT,
                            price=bid_price,
                            strategy_id=f"HFT_{strategy.value}"
                        )

                        # 创建卖单
                        sell_order = order_manager.create_order(
                            symbol=symbol,
                            quantity=position_limit * 0.1,
                            order_type=order_manager.OrderType.LIMIT,
                            price=ask_price,
                            strategy_id=f"HFT_{strategy.value}"
                        )

                        # 提交订单
                        order_manager.submit_order(buy_order)
                        order_manager.submit_order(sell_order)

                        logger.debug(f"HFT做市订单已提交: {symbol} {bid_price}@{ask_price}")

                except Exception as e:
                    logger.error(f"HFT做市订单提交失败: {e}")

    def _execute_arbitrage(self, symbol: str, config: Dict[str, Any], strategy: HFTStrategy):
        """执行套利策略"""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return

        imbalance_threshold = config.get('imbalance_threshold', 0.2)

        # 计算买卖量不平衡
        bid_volume = sum(entry.quantity for entry in order_book.bids[:5])
        ask_volume = sum(entry.quantity for entry in order_book.asks[:5])

        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            if abs(imbalance) > imbalance_threshold:
                if imbalance > 0:  # 买单强势
                    # 可能的价格会上升，做多
                    self.execute_trade(symbol, 'buy', 10, HFTStrategy.ARBITRAGE, 500)
                else:  # 卖单强势
                    # 可能的价格会下降，做空
                    self.execute_trade(symbol, 'sell', 10, HFTStrategy.ARBITRAGE, 500)

    def _execute_momentum(self, symbol: str, config: Dict[str, Any], strategy: HFTStrategy):
        """执行动量策略"""
        data_queue = self.market_data.get(symbol)
        if not data_queue or len(data_queue) < 10:
            return

        # 计算短期动量
        recent_prices = [d['price'] for d in list(data_queue)[-10:]]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        momentum_threshold = config.get('momentum_threshold', 0.001)

        if abs(momentum) > momentum_threshold:
            if momentum > 0:
                self.execute_trade(symbol, 'buy', 5, HFTStrategy.MOMENTUM, 200)
            else:
                self.execute_trade(symbol, 'sell', 5, HFTStrategy.MOMENTUM, 200)

    def _execute_order_book_strategy(self, symbol: str, config: Dict[str, Any], strategy: HFTStrategy):
        """执行订单簿策略"""
        order_book = self.order_books.get(symbol)
        if not order_book or len(order_book.bids) < 2 or len(order_book.asks) < 2:
            return

        # 检测订单簿不平衡
        bid_level_1 = order_book.bids[0].quantity
        bid_level_2 = order_book.bids[1].quantity
        ask_level_1 = order_book.asks[0].quantity
        ask_level_2 = order_book.asks[1].quantity

        # 如果第一级订单量远大于第二级，可能存在大单
        if bid_level_1 > bid_level_2 * 2:
            # 大买单可能推高价格
            self.execute_trade(symbol, 'buy', 2, HFTStrategy.ORDER_BOOK, 100)
        elif ask_level_1 > ask_level_2 * 2:
            # 大卖单可能压低价格
            self.execute_trade(symbol, 'sell', 2, HFTStrategy.ORDER_BOOK, 100)

    def _update_market_indicators(self, symbol: str):
        """更新市场指标"""
        # 这里可以实现更复杂的市场指标计算

    def _risk_manager(self):
        """风险管理器"""
        while self.running:
            try:
                # 检查风险指标
                self._check_risk_violations()

                time.sleep(0.1)  # 100ms 检查间隔

            except Exception as e:
                logger.error(f"风险管理异常: {e}")
                time.sleep(1)

    def _check_risk_violations(self):
        """检查风险违规"""
        # 检查总持仓
        total_position = sum(abs(pos) for pos in self.strategy_positions.values())

        if total_position > self.max_position:
            self.emergency_stop = True
            logger.critical(f"持仓超出限制: {total_position} > {self.max_position}")

        # 检查交易频率
        if self.performance_stats['trades_executed'] > self.risk_limit_per_second:
            logger.warning("交易频率过高，触发风险限制")

    def _performance_monitor(self):
        """性能监控器"""
        while self.running:
            try:
                # 记录性能指标
                self._log_performance_stats()

                time.sleep(1.0)  # 1秒间隔

            except Exception as e:
                logger.error(f"性能监控异常: {e}")
                time.sleep(5)

    def _log_performance_stats(self):
        """记录性能统计"""
        stats = self.performance_stats.copy()
        logger.info(f"HFT性能统计: 交易数={stats['trades_executed']}, "
                    f"平均延迟={stats['average_latency_us']:.0f}us, "
                    f"已实现PnL={stats['pnl_realized']:.2f}")

    def _generate_trade_id(self) -> str:
        """生成交易ID"""
        return f"HFT_{datetime.now().strftime('%H % M % S % f')}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()

    def get_strategy_positions(self) -> Dict[str, float]:
        """获取策略持仓"""
        return self.strategy_positions.copy()

    def get_market_microstructure(self, symbol: str) -> Optional[MarketMicrostructure]:
        """获取市场微观结构"""
        micro_queue = self.microstructure.get(symbol)
        if micro_queue:
            return list(micro_queue)[-1] if micro_queue else None
        return None

    def emergency_stop_all(self):
        """紧急停止所有活动"""
        self.emergency_stop = True
        logger.critical("触发紧急停止，所有交易活动已暂停")

    def health_check(self) -> Dict[str, Any]:
        """健康检查 - 支持基础设施层监控"""
        health_info = {
            'component': 'HFTExecutionEngine',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'running': self.running,
            'active_strategies': len(self.active_strategies),
            'emergency_stop': self.emergency_stop,
            'infrastructure_integration': INFRASTRUCTURE_INTEGRATION_AVAILABLE,
            'metrics': {}
        }

        # 检查引擎状态
        if self.emergency_stop:
            health_info['status'] = 'critical'
            health_info['warnings'] = ['紧急停止已触发']

        # 检查活跃线程
        active_threads = sum(1 for thread in self.threads if thread.is_alive())
        if active_threads < len(self.threads):
            health_info['status'] = 'warning'
            health_info['warnings'] = ['部分线程未运行']

        # 检查基础设施集成状态
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            health_info['infrastructure_status'] = {
                'adapter_available': self._infrastructure_adapter is not None,
                'config_manager': self._config_manager is not None,
                'cache_manager': self._cache_manager is not None,
                'monitoring': self._monitoring is not None,
                'logger': self._logger is not None
            }
        else:
            health_info['infrastructure_status'] = 'not_available'

        # 收集性能指标
        health_info['metrics'] = {
            'trades_executed': self.performance_stats['trades_executed'],
            'average_latency_us': self.performance_stats['average_latency_us'],
            'active_threads': active_threads,
            'total_threads': len(self.threads),
            'strategy_positions_count': len(self.strategy_positions)
        }

        return health_info

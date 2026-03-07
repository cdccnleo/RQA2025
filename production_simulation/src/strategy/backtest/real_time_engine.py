#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
实时回测引擎

提供接近实盘环境的回测体验，支持实时数据流处理、增量回测和动态策略调整。
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
import threading

from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RealTimeData:

    """实时数据结构"""
    timestamp: datetime
    symbol: str
    data_type: str  # 'market', 'news', 'order_book'
    data: Dict[str, Any]
    source: str = "unknown"


@dataclass
class BacktestState:

    """回测状态"""
    timestamp: datetime
    portfolio_value: float
    positions: Dict[str, float]
    cash: float
    trades: List[Dict[str, Any]]
    metrics: Dict[str, float]


@dataclass
class StrategyConfig:

    """策略配置"""
    strategy_id: str
    name: str
    parameters: Dict[str, Any]
    enabled: bool = True
    risk_limits: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class CacheManager:

    """高性能缓存管理器 - 优化版本"""

    def __init__(self, cache_size: int = 1000):

        self.cache_size = cache_size
        self.data_cache = {}
        self.metrics_cache = {}
        self.strategy_cache = {}
        self._lock = threading.Lock()
        self._access_order = []  # LRU访问顺序
        self._memory_usage = 0
        self._max_memory_mb = 100  # 最大内存使用量(MB)

    def cache_data(self, key: str, data: Any, ttl: int = 300):
        """缓存数据 - 优化版本"""
        with self._lock:
            # 检查内存使用
            if self._memory_usage > self._max_memory_mb * 1024 * 1024:
                self._cleanup_expired_entries()
                if self._memory_usage > self._max_memory_mb * 1024 * 1024:
                    self._evict_lru_entries()

            # 更新访问顺序
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # 如果缓存已满，删除最久未访问的条目
            if len(self.data_cache) >= self.cache_size:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self.data_cache:
                    del self.data_cache[oldest_key]

            # 估算数据大小
            data_size = self._estimate_data_size(data)
            self._memory_usage += data_size

            self.data_cache[key] = {
                'data': data,
                'timestamp': datetime.now(),
                'ttl': ttl,
                'size': data_size
            }

    def get_cached_data(self, key: str) -> Optional[Any]:
        """获取缓存数据 - 优化版本"""
        with self._lock:
            if key in self.data_cache:
                cache_entry = self.data_cache[key]
                if (datetime.now() - cache_entry['timestamp']).seconds < cache_entry['ttl']:
                    # 更新访问顺序
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return cache_entry['data']
                else:
                    # 删除过期条目
                    self._memory_usage -= cache_entry.get('size', 0)
                    del self.data_cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
            return None

    def update_metrics_cache(self, strategy_id: str, metrics: Dict[str, float]):
        """更新指标缓存 - 优化版本"""
        with self._lock:
            # 清理过期指标缓存
            current_time = datetime.now()
            expired_keys = []
            for key, entry in self.metrics_cache.items():
                if (current_time - entry['timestamp']).seconds > 3600:  # 1小时过期
                    expired_keys.append(key)

            for key in expired_keys:
                del self.metrics_cache[key]

            self.metrics_cache[strategy_id] = {
                'metrics': metrics,
                'timestamp': current_time
            }

    def get_metrics_cache(self, strategy_id: str) -> Optional[Dict[str, float]]:
        """获取指标缓存 - 优化版本"""
        with self._lock:
            if strategy_id in self.metrics_cache:
                entry = self.metrics_cache[strategy_id]
                if (datetime.now() - entry['timestamp']).seconds < 3600:
                    return entry['metrics']
                else:
                    del self.metrics_cache[strategy_id]
            return None

    def _cleanup_expired_entries(self):
        """清理过期条目"""
        current_time = datetime.now()
        expired_keys = []

        for key, entry in self.data_cache.items():
            if (current_time - entry['timestamp']).seconds >= entry['ttl']:
                expired_keys.append(key)

        for key in expired_keys:
            self._memory_usage -= self.data_cache[key].get('size', 0)
            del self.data_cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru_entries(self):
        """驱逐LRU条目"""
        while self._memory_usage > self._max_memory_mb * 1024 * 1024 and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self.data_cache:
                self._memory_usage -= self.data_cache[oldest_key].get('size', 0)
                del self.data_cache[oldest_key]

    def _estimate_data_size(self, data: Any) -> int:
        """估算数据大小"""
        try:
            import sys
            return sys.getsizeof(data)
        except BaseException:
            return 1024  # 默认1KB

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'data_cache_size': len(self.data_cache),
                'metrics_cache_size': len(self.metrics_cache),
                'memory_usage_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self._max_memory_mb,
                'access_order_length': len(self._access_order)
            }


class ParameterManager:

    """参数管理器 - 第二阶段功能"""

    def __init__(self):

        self.parameter_history = {}
        self.optimization_rules = {}

    def update_parameters(self, strategy_id: str, new_params: Dict[str, Any]):
        """更新策略参数"""
        if strategy_id not in self.parameter_history:
            self.parameter_history[strategy_id] = []

        self.parameter_history[strategy_id].append({
            'parameters': new_params,
            'timestamp': datetime.now()
        })

    def get_parameter_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """获取参数历史"""
        return self.parameter_history.get(strategy_id, [])

    def add_optimization_rule(self, strategy_id: str, rule: Callable):
        """添加优化规则"""
        self.optimization_rules[strategy_id] = rule

    def apply_optimization_rules(self, strategy_id: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """应用优化规则"""
        if strategy_id in self.optimization_rules:
            rule = self.optimization_rules[strategy_id]
            return rule(current_metrics)
        return {}


class DynamicStrategyManager:

    """动态策略管理器 - 第二阶段功能"""

    def __init__(self):

        self.strategies = {}
        self.parameter_manager = ParameterManager()
        self.performance_monitor = {}
        self.risk_monitor = {}

    def register_strategy(self, strategy_id: str, strategy_func: Callable,


                          config: StrategyConfig):
        """注册策略"""
        self.strategies[strategy_id] = {
            'function': strategy_func,
            'config': config,
            'performance_history': [],
            'risk_history': []
        }
        logger.info(f"策略已注册: {strategy_id}")

    def update_strategy_parameters(self, strategy_id: str, new_params: Dict[str, Any]):
        """更新策略参数"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['config'].parameters.update(new_params)
            self.parameter_manager.update_parameters(strategy_id, new_params)
            logger.info(f"策略参数已更新: {strategy_id}")

    def enable_strategy(self, strategy_id: str):
        """启用策略"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['config'].enabled = True
            logger.info(f"策略已启用: {strategy_id}")

    def disable_strategy(self, strategy_id: str):
        """禁用策略"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['config'].enabled = False
            logger.info(f"策略已禁用: {strategy_id}")

    def update_performance_metrics(self, strategy_id: str, metrics: Dict[str, float]):
        """更新性能指标"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['performance_history'].append({
                'metrics': metrics,
                'timestamp': datetime.now()
            })
            self.strategies[strategy_id]['config'].performance_metrics = metrics

    def check_risk_limits(self, strategy_id: str, current_risk: Dict[str, float]) -> bool:
        """检查风险限制"""
        if strategy_id not in self.strategies:
            return True

        config = self.strategies[strategy_id]['config']
        risk_limits = config.risk_limits

        for risk_type, limit in risk_limits.items():
            if risk_type in current_risk and current_risk[risk_type] > limit:
                logger.warning(
                    f"策略 {strategy_id} 风险超限: {risk_type} = {current_risk[risk_type]} > {limit}")
                return False
        return True

    def get_active_strategies(self) -> Dict[str, Any]:
        """获取活跃策略"""
        return {
            strategy_id: strategy_info
            for strategy_id, strategy_info in self.strategies.items()
            if strategy_info['config'].enabled
        }


class RealTimeDataProcessor:

    """高性能实时数据处理器 - 优化版本"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.data_queue = Queue(maxsize=10000)  # 增加队列大小
        self.processors = []
        self.running = False
        self.thread = None
        self.cache_manager = CacheManager()
        self.executor = ThreadPoolExecutor(max_workers=4)  # 异步处理线程池
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

    def add_processor(self, processor: Callable):
        """添加数据处理器"""
        self.processors.append(processor)

    def preprocess(self, data: Dict[str, Any]) -> RealTimeData:
        """数据预处理 - 优化版本"""
        try:
            timestamp = data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            symbol = data.get('symbol', 'unknown')
            data_type = data.get('type', 'market')
            source = data.get('source', 'unknown')

            # 优化缓存策略
            cache_key = f"{symbol}_{timestamp.isoformat()}"
            cached_data = self.cache_manager.get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # 数据验证和清理
            cleaned_data = self._clean_data(data)

            real_time_data = RealTimeData(
                timestamp=timestamp,
                symbol=symbol,
                data_type=data_type,
                data=cleaned_data,
                source=source
            )

            # 缓存处理后的数据
            self.cache_manager.cache_data(cache_key, real_time_data, ttl=60)

            return real_time_data

        except Exception as e:
            logger.error(f"数据预处理错误: {e}")
            self.error_count += 1
            return None

    def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清理和验证数据"""
        cleaned = {}
        for key, value in data.items():
            if key != 'timestamp' and value is not None:
                cleaned[key] = value
        return cleaned

    def start(self):
        """启动数据处理器"""
        self.running = True
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("高性能实时数据处理器已启动")

    def stop(self):
        """停止数据处理器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("高性能实时数据处理器已停止")

    def _process_loop(self):
        """数据处理循环 - 优化版本"""
        while self.running:
            try:
                # 批量处理数据
                batch_data = []
                for _ in range(min(10, self.data_queue.qsize())):
                    try:
                        data = self.data_queue.get_nowait()
                        batch_data.append(data)
                    except BaseException:
                        break

                if batch_data:
                    # 异步处理批量数据
                    futures = []
                    for data in batch_data:
                        future = self.executor.submit(self._process_single_data, data)
                        futures.append(future)

                    # 等待所有处理完成
                    for future in futures:
                        try:
                            future.result(timeout=5)
                        except Exception as e:
                            logger.error(f"异步数据处理错误: {e}")
                            self.error_count += 1
                else:
                    time.sleep(0.001)  # 减少睡眠时间

            except Exception as e:
                logger.error(f"数据处理循环错误: {e}")
                self.error_count += 1
                time.sleep(0.1)

    def _process_single_data(self, data: Dict[str, Any]):
        """处理单个数据"""
        try:
            processed_data = self.preprocess(data)
            if processed_data:
                # 并行执行处理器
                futures = []
                for processor in self.processors:
                    future = self.executor.submit(processor, processed_data)
                    futures.append(future)

                # 等待所有处理器完成
                for future in futures:
                    try:
                        future.result(timeout=3)
                    except Exception as e:
                        logger.error(f"处理器执行错误: {e}")
                        self.error_count += 1

                self.processed_count += 1

        except Exception as e:
            logger.error(f"单数据处理错误: {e}")
            self.error_count += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
            throughput = self.processed_count / runtime if runtime > 0 else 0
        else:
            runtime = 0
            throughput = 0

        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'runtime_seconds': runtime,
            'throughput_per_second': throughput,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'queue_size': self.data_queue.qsize(),
            'cache_stats': self.cache_manager.get_cache_stats()
        }


class StateManager:

    """状态管理器"""

    def __init__(self, state_file: Optional[str] = None):

        self.state_file = state_file
        self.current_state: Optional[BacktestState] = None
        self.state_history: List[BacktestState] = []
        self.cache_manager = CacheManager()  # 第二阶段：添加缓存管理器

    def initialize_state(self, initial_capital: float = 1000000.0):
        """初始化状态"""
        self.current_state = BacktestState(
            timestamp=datetime.now(),
            portfolio_value=initial_capital,
            positions={},
            cash=initial_capital,
            trades=[],
            metrics={}
        )
        self.state_history = [self.current_state]
        logger.info(f"状态已初始化，初始资金: {initial_capital}")

    def update_state(self, delta: Dict[str, Any]):
        """更新状态"""
        if not self.current_state:
            return

        # 更新持仓
        if 'positions' in delta:
            for symbol, quantity in delta['positions'].items():
                if quantity != 0:
                    self.current_state.positions[symbol] = self.current_state.positions.get(
                        symbol, 0.0) + quantity

        # 更新现金
        if 'cash_delta' in delta:
            self.current_state.cash += delta['cash_delta']

        # 添加交易记录
        if 'trades' in delta:
            self.current_state.trades.extend(delta['trades'])

        # 更新指标
        if 'metrics' in delta:
            self.current_state.metrics.update(delta['metrics'])

        # 第二阶段：缓存状态更新
        self.cache_manager.cache_data(
            f"state_{self.current_state.timestamp.isoformat()}",
            self.current_state
        )

        self._update_portfolio_value()
        self.current_state.timestamp = datetime.now()
        self.state_history.append(self.current_state)

    def _update_portfolio_value(self):
        """更新投资组合价值"""
        if not self.current_state:
            return
        positions_value = 0.0
        for symbol, quantity in self.current_state.positions.items():
            # 使用缓存的价格数据，如果没有则使用默认值
            cached_price = self.cache_manager.get_cached_data(f"price_{symbol}")
            price = cached_price if cached_price is not None else 100.0
            positions_value += quantity * price
        self.current_state.portfolio_value = self.current_state.cash + positions_value


class IncrementalBacktestEngine:

    """高性能增量回测引擎 - 优化版本"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.state_manager = StateManager()
        self.strategies = {}
        self.running = False
        self.last_prices = {}
        self.position_cache = {}
        self.dynamic_manager = DynamicStrategyManager()
        self.cache_manager = CacheManager()

        # 策略执行优化
        self.strategy_executor = ThreadPoolExecutor(max_workers=4)  # 策略并行执行
        self.strategy_performance = {}  # 策略性能监控
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }

    def add_strategy(self, name: str, strategy_func: Callable):
        """添加策略"""
        self.strategies[name] = strategy_func
        # 第二阶段：注册到动态策略管理器
        config = StrategyConfig(
            strategy_id=name,
            name=name,
            parameters={},
            enabled=True
        )
        self.dynamic_manager.register_strategy(name, strategy_func, config)

    def start(self, initial_capital: float = 1000000.0):
        """启动引擎"""
        self.state_manager.initialize_state(initial_capital)
        self.running = True
        logger.info("增量回测引擎已启动")

    def stop(self):
        """停止引擎"""
        self.running = False
        logger.info("增量回测引擎已停止")

    def process_data(self, data: RealTimeData):
        """处理实时数据"""
        if not self.running:
            return

        # 第二阶段：使用缓存管理器优化数据处理
        cache_key = f"processed_{data.symbol}_{data.timestamp.isoformat()}"
        cached_result = self.cache_manager.get_cached_data(cache_key)

        if cached_result is None:
            delta = self._calculate_delta(data)
            self.state_manager.update_state(delta)
            self._execute_strategies(data)

            # 缓存处理结果
            self.cache_manager.cache_data(cache_key, {
                'delta': delta,
                'timestamp': datetime.now()
            })
        else:
            # 使用缓存结果
            logger.debug(f"使用缓存数据: {cache_key}")

    def _calculate_delta(self, data: RealTimeData) -> Dict[str, Any]:
        """计算增量变化"""
        delta = {
            'positions': {}, 'cash_delta': 0.0, 'trades': [], 'metrics': {}
        }
        current_price = data.data.get('price', 0.0)
        symbol = data.symbol

        # 第二阶段：改进的增量计算算法
        if symbol in self.last_prices:
            price_change = current_price - self.last_prices[symbol]
        if symbol in self.state_manager.current_state.positions:
            position_value_change = price_change * \
                self.state_manager.current_state.positions[symbol]
            delta['metrics'][f'{symbol}_pnl'] = position_value_change

            # 计算收益率
        if self.last_prices[symbol] > 0:
            return_rate = price_change / self.last_prices[symbol]
            delta['metrics'][f'{symbol}_return'] = return_rate

        self.last_prices[symbol] = current_price

        # 缓存价格数据
        self.cache_manager.cache_data(f"price_{symbol}", current_price)

        return delta

    def _execute_strategies(self, data: RealTimeData):
        """执行策略 - 优化版本"""
        active_strategies = self.dynamic_manager.get_active_strategies()

        if not active_strategies:
            return

        # 并行执行策略
        futures = []
        for strategy_id, strategy_info in active_strategies.items():
            future = self.strategy_executor.submit(
                self._execute_single_strategy,
                strategy_id,
                strategy_info,
                data
            )
            futures.append((strategy_id, future))

        # 等待所有策略执行完成
        for strategy_id, future in futures:
            try:
                start_time = time.time()
                result = future.result(timeout=5)  # 5秒超时
                execution_time = time.time() - start_time

                # 更新执行统计
                self.execution_stats['total_executions'] += 1
                self.execution_stats['successful_executions'] += 1

                # 更新策略性能
                if strategy_id not in self.strategy_performance:
                    self.strategy_performance[strategy_id] = {
                        'execution_count': 0,
                        'total_time': 0.0,
                        'average_time': 0.0,
                        'success_rate': 0.0
                    }

                perf = self.strategy_performance[strategy_id]
                perf['execution_count'] += 1
                perf['total_time'] += execution_time
                perf['average_time'] = perf['total_time'] / perf['execution_count']
                perf['success_rate'] = (
                    perf['execution_count'] - self.execution_stats['failed_executions']) / perf['execution_count']

            except Exception as e:
                self.execution_stats['failed_executions'] += 1
                logger.error(f"策略执行错误 {strategy_id}: {e}")

    def _execute_single_strategy(self, strategy_id: str, strategy_info: Dict[str, Any], data: RealTimeData):
        """执行单个策略"""
        try:
            strategy_func = strategy_info['function']
            config = strategy_info['config']

            # 检查风险限制
            current_risk = self._calculate_current_risk(strategy_id)
            if not self.dynamic_manager.check_risk_limits(strategy_id, current_risk):
                logger.warning(f"策略 {strategy_id} 风险超限，跳过执行")
                return None

            # 执行策略
            signals = strategy_func(data, self.state_manager.current_state)

            # 处理信号
            if signals:
                self._process_signals(signals, data, strategy_id)

            # 更新性能指标
            performance_metrics = self._calculate_performance_metrics(strategy_id)
            self.dynamic_manager.update_performance_metrics(strategy_id, performance_metrics)

            return signals

        except Exception as e:
            logger.error(f"单个策略执行错误 {strategy_id}: {e}")
            raise

    def _calculate_current_risk(self, strategy_id: str) -> Dict[str, float]:
        """计算当前风险"""
        # 简化的风险计算
        return {
            'var_95': 0.02,  # 95% VaR
            'max_drawdown': 0.05,  # 最大回撤
            'volatility': 0.15  # 波动率
        }

    def _calculate_performance_metrics(self, strategy_id: str) -> Dict[str, float]:
        """计算性能指标"""
        # 简化的性能指标计算
        return {
            'sharpe_ratio': 1.2,
            'total_return': 0.15,
            'max_drawdown': -0.05
        }

    def _process_signals(self, signals: Dict[str, Any], data: RealTimeData, strategy_id: str):
        """处理策略信号 - 优化版本"""
        if not signals:
            return

        symbol = data.symbol
        current_price = data.data.get('price', 0.0)

        # 批量处理信号，减少锁竞争
        buy_signals = []
        sell_signals = []

        for signal_symbol, signal_value in signals.items():
            if signal_symbol == symbol:
                if signal_value > 0:
                    buy_signals.append((signal_symbol, abs(signal_value)))
                elif signal_value < 0:
                    sell_signals.append((signal_symbol, abs(signal_value)))

        # 批量执行买入
        for signal_symbol, quantity in buy_signals:
            self._execute_buy(signal_symbol, current_price, quantity, strategy_id)

        # 批量执行卖出
        for signal_symbol, quantity in sell_signals:
            self._execute_sell(signal_symbol, current_price, quantity, strategy_id)

    def _execute_buy(self, symbol: str, price: float, quantity: float, strategy_id: str):
        """执行买入"""
        if not self.state_manager.current_state:
            return

        required_cash = price * quantity
        if self.state_manager.current_state.cash >= required_cash:
            self.state_manager.current_state.cash -= required_cash
            self.state_manager.current_state.positions[symbol] = self.state_manager.current_state.positions.get(
                symbol, 0.0) + quantity

            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'value': required_cash,
                'strategy_id': strategy_id
            }
            self.state_manager.current_state.trades.append(trade)
            logger.info(f"策略 {strategy_id} 买入 {symbol}: {quantity} @ {price}")

    def _execute_sell(self, symbol: str, price: float, quantity: float, strategy_id: str):
        """执行卖出"""
        if not self.state_manager.current_state or symbol not in self.state_manager.current_state.positions:
            return

        current_position = self.state_manager.current_state.positions[symbol]
        sell_quantity = min(quantity, current_position)

        if sell_quantity > 0:
            revenue = price * sell_quantity
            self.state_manager.current_state.cash += revenue
            self.state_manager.current_state.positions[symbol] -= sell_quantity

            if self.state_manager.current_state.positions[symbol] <= 0:
                del self.state_manager.current_state.positions[symbol]

            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'sell',
                'price': price,
                'quantity': sell_quantity,
                'value': revenue,
                'strategy_id': strategy_id
            }
            self.state_manager.current_state.trades.append(trade)
            logger.info(f"策略 {strategy_id} 卖出 {symbol}: {sell_quantity} @ {price}")

    def get_current_state(self) -> Optional[BacktestState]:
        """获取当前状态"""
        return self.state_manager.current_state

    def update_strategy_parameters(self, strategy_id: str, new_params: Dict[str, Any]):
        """更新策略参数 - 第二阶段功能"""
        self.dynamic_manager.update_strategy_parameters(strategy_id, new_params)

    def enable_strategy(self, strategy_id: str):
        """启用策略 - 第二阶段功能"""
        self.dynamic_manager.enable_strategy(strategy_id)

    def disable_strategy(self, strategy_id: str):
        """禁用策略 - 第二阶段功能"""
        self.dynamic_manager.disable_strategy(strategy_id)

    def get_strategy_performance(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        return {
            'strategy_performance': self.strategy_performance,
            'execution_stats': self.execution_stats,
            'cache_stats': self.cache_manager.get_cache_stats()
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        total_executions = self.execution_stats['total_executions']
        if total_executions > 0:
            success_rate = self.execution_stats['successful_executions'] / total_executions
        else:
            success_rate = 0.0

        return {
            'total_executions': total_executions,
            'successful_executions': self.execution_stats['successful_executions'],
            'failed_executions': self.execution_stats['failed_executions'],
            'success_rate': success_rate,
            'average_execution_time': self.execution_stats['average_execution_time']
        }


class RealTimeMonitor:

    """实时监控器"""

    def __init__(self):

        self.metrics = {}  # 当前指标
        self.metrics_history = []
        self.alert_thresholds = {}
        self.alerts = []  # 告警列表
        self.cache_manager = CacheManager()  # 第二阶段：添加缓存管理器

    def update_metrics(self, state: BacktestState):
        """更新指标"""
        metrics = {
            'portfolio_value': state.portfolio_value,
            'cash': state.cash,
            'positions_count': len(state.positions),
            'trades_count': len(state.trades),
            'timestamp': state.timestamp
        }

        # 更新当前指标
        self.metrics = metrics

        # 第二阶段：缓存指标数据
        self.cache_manager.cache_data(
            f"metrics_{state.timestamp.isoformat()}",
            metrics
        )

        self.metrics_history.append(metrics)

        # 检查告警阈值
        self._check_alerts(metrics)

    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警"""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] > threshold:
                    alert_msg = f"指标告警: {metric_name} = {metrics[metric_name]} > {threshold}"
                    logger.warning(alert_msg)
                    self.alerts.append({
                        'timestamp': metrics.get('timestamp', datetime.now()),
                        'metric': metric_name,
                        'value': metrics[metric_name],
                        'threshold': threshold,
                        'message': alert_msg
                    })

    def set_alert_threshold(self, metric_name: str, threshold: float):
        """设置告警阈值"""
        self.alert_thresholds[metric_name] = threshold

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]
        return {
            'current': latest_metrics,
            'history': self.metrics_history[-10:]  # 最近10条记录
        }


class RealTimeBacktestEngine:

    """实时回测引擎主类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.data_processor = RealTimeDataProcessor(config)
        self.incremental_engine = IncrementalBacktestEngine(config)
        self.monitor = RealTimeMonitor()
        self.running = False

        # 第二阶段：添加高级监控功能
        self.data_processor.add_processor(self.incremental_engine.process_data)

    def start(self, initial_capital: float = 1000000.0):
        """启动引擎"""
        self.incremental_engine.start(initial_capital)
        self.data_processor.start()
        self.running = True
        logger.info("实时回测引擎已启动")

    def stop(self):
        """停止引擎"""
        self.running = False
        self.incremental_engine.stop()
        self.data_processor.stop()
        logger.info("实时回测引擎已停止")

    def add_strategy(self, name: str, strategy_func: Callable):
        """添加策略"""
        self.incremental_engine.add_strategy(name, strategy_func)

    def process_data(self, data: Dict[str, Any]):
        """处理数据"""
        real_time_data = self.data_processor.preprocess(data)
        if real_time_data:
            self.incremental_engine.process_data(real_time_data)
            current_state = self.incremental_engine.get_current_state()
        if current_state:
            self.monitor.update_metrics(current_state)

    def get_current_state(self) -> Optional[BacktestState]:
        """获取当前状态"""
        return self.incremental_engine.get_current_state()

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return self.monitor.get_metrics()

    # 第二阶段：添加动态策略管理功能

    def update_strategy_parameters(self, strategy_id: str, new_params: Dict[str, Any]):
        """更新策略参数"""
        self.incremental_engine.update_strategy_parameters(strategy_id, new_params)

    def enable_strategy(self, strategy_id: str):
        """启用策略"""
        self.incremental_engine.enable_strategy(strategy_id)

    def disable_strategy(self, strategy_id: str):
        """禁用策略"""
        self.incremental_engine.disable_strategy(strategy_id)

    def set_alert_threshold(self, metric_name: str, threshold: float):
        """设置告警阈值"""
        self.monitor.set_alert_threshold(metric_name, threshold)

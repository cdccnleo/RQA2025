#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时引擎主流程脚本 - 高性能实时数据处理系统

功能特性：
1. 实时数据接收和处理
2. Level2行情解析和订单簿维护
3. 高性能事件分发
4. 实时性能监控
5. 多市场数据支持
6. 系统健康检查

技术架构：
- 零拷贝数据传输
- 无锁环形缓冲区
- 多线程并发处理
- 实时性能监控
- 背压控制机制
"""

from src.engine.buffers import BufferManager, BufferConfig, BufferType
from src.engine.dispatcher import EventDispatcher, EventRoute, EventHandler, EventPriority
from src.engine.level2 import Level2Processor, MarketType
from src.engine.realtime import RealTimeEngine, Event, EventType
import time
import logging
import threading
import signal
import sys
from typing import Dict, Any
from dataclasses import dataclass
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """引擎配置"""
    # 实时引擎配置
    buffer_size: int = 100000
    memory_pool_size: int = 1000
    max_queue_size: int = 10000
    num_workers: int = 4
    num_dispatchers: int = 2

    # Level2处理器配置
    max_levels: int = 10
    max_audit_records: int = 10000

    # 事件分发器配置
    load_balancing_strategy: str = "round_robin"

    # 缓冲区配置
    enable_zero_copy: bool = True
    enable_memory_pool: bool = True
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8

    # 监控配置
    stats_interval: float = 5.0
    health_check_interval: float = 10.0


class RealTimeEngineMainFlow:
    """实时引擎主流程"""

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.running = False
        self.shutdown_event = threading.Event()

        # 初始化组件
        self._init_components()
        self._setup_signal_handlers()

        # 监控线程
        self.monitor_thread = None
        self.stats_thread = None

        # 性能统计
        self.performance_stats = {
            'start_time': None,
            'total_events_processed': 0,
            'total_level2_messages': 0,
            'total_trades_processed': 0,
            'system_uptime': 0
        }

    def _init_components(self):
        """初始化组件"""
        logger.info("初始化实时引擎组件...")

        # 初始化实时引擎
        engine_config = {
            'buffer_size': self.config.buffer_size,
            'memory_pool_size': self.config.memory_pool_size,
            'max_queue_size': self.config.max_queue_size,
            'num_workers': self.config.num_workers
        }
        self.realtime_engine = RealTimeEngine(engine_config)

        # 初始化Level2处理器
        level2_config = {
            'max_levels': self.config.max_levels
        }
        self.level2_processor = Level2Processor(level2_config)

        # 初始化事件分发器
        dispatcher_config = {
            'load_balancing_strategy': self.config.load_balancing_strategy,
            'max_audit_records': self.config.max_audit_records,
            'num_dispatchers': self.config.num_dispatchers
        }
        self.event_dispatcher = EventDispatcher(dispatcher_config)

        # 初始化缓冲区管理器
        buffer_config = BufferConfig(
            buffer_type=BufferType.RING,
            size=self.config.buffer_size,
            enable_zero_copy=self.config.enable_zero_copy,
            enable_memory_pool=self.config.enable_memory_pool,
            enable_backpressure=self.config.enable_backpressure,
            backpressure_threshold=self.config.backpressure_threshold
        )
        self.buffer_manager = BufferManager(buffer_config)

        # 注册事件处理器
        self._register_event_handlers()

        # 注册事件路由
        self._register_event_routes()

        logger.info("实时引擎组件初始化完成")

    def _register_event_handlers(self):
        """注册事件处理器"""
        # Level2数据处理处理器
        level2_handler = EventHandler(
            name="level2_processor",
            handler=self._process_level2_data,
            priority=EventPriority.HIGH,
            max_concurrent=5
        )
        self.event_dispatcher.register_handler(level2_handler)

        # 成交数据处理处理器
        trade_handler = EventHandler(
            name="trade_processor",
            handler=self._process_trade_data,
            priority=EventPriority.HIGH,
            max_concurrent=5
        )
        self.event_dispatcher.register_handler(trade_handler)

        # 系统监控处理器
        monitor_handler = EventHandler(
            name="system_monitor",
            handler=self._process_system_event,
            priority=EventPriority.NORMAL,
            max_concurrent=2
        )
        self.event_dispatcher.register_handler(monitor_handler)

        # 全局日志处理器
        log_handler = EventHandler(
            name="log_processor",
            handler=self._process_log_event,
            priority=EventPriority.LOW,
            max_concurrent=1
        )
        self.event_dispatcher.register_handler(log_handler)

    def _register_event_routes(self):
        """注册事件路由"""
        # Level2数据路由
        level2_route = EventRoute(
            event_type="level2_data",
            handlers=["level2_processor"],
            priority=EventPriority.HIGH,
            timeout=5.0
        )
        self.event_dispatcher.register_route(level2_route)

        # 成交数据路由
        trade_route = EventRoute(
            event_type="trade_data",
            handlers=["trade_processor"],
            priority=EventPriority.HIGH,
            timeout=5.0
        )
        self.event_dispatcher.register_route(trade_route)

        # 系统事件路由
        system_route = EventRoute(
            event_type="system_event",
            handlers=["system_monitor"],
            priority=EventPriority.NORMAL,
            timeout=10.0
        )
        self.event_dispatcher.register_route(system_route)

        # 日志事件路由
        log_route = EventRoute(
            event_type="log_event",
            handlers=["log_processor"],
            priority=EventPriority.LOW,
            timeout=30.0
        )
        self.event_dispatcher.register_route(log_route)

    def _process_level2_data(self, event_data: Dict[str, Any]) -> bool:
        """处理Level2数据"""
        try:
            raw_data = event_data.get('data')
            market = MarketType(event_data.get('market', 'A_SHARE'))

            # 处理Level2数据
            level2_data = self.level2_processor.process_level2_data(raw_data, market)
            if level2_data:
                self.performance_stats['total_level2_messages'] += 1

                # 发布订单簿更新事件
                order_book_event = Event(
                    event_type=EventType.TICK,
                    timestamp=time.time(),
                    data={
                        'symbol': level2_data.symbol,
                        'order_book': level2_data.order_book,
                        'market': market.value
                    },
                    source="level2_processor"
                )
                self.realtime_engine.publish_event(order_book_event)

                return True

            return False

        except Exception as e:
            logger.error(f"Level2数据处理错误: {e}")
            return False

    def _process_trade_data(self, event_data: Dict[str, Any]) -> bool:
        """处理成交数据"""
        try:
            raw_data = event_data.get('data')
            market = MarketType(event_data.get('market', 'A_SHARE'))

            # 处理成交数据
            trade_data = self.level2_processor.process_trade_data(raw_data, market)
            if trade_data:
                self.performance_stats['total_trades_processed'] += 1

                # 发布成交事件
                trade_event = Event(
                    event_type=EventType.TRADE,
                    timestamp=time.time(),
                    data={
                        'symbol': trade_data['symbol'],
                        'price': trade_data['price'],
                        'volume': trade_data['volume'],
                        'side': trade_data['side'].value,
                        'market': market.value
                    },
                    source="trade_processor"
                )
                self.realtime_engine.publish_event(trade_event)

                return True

            return False

        except Exception as e:
            logger.error(f"成交数据处理错误: {e}")
            return False

    def _process_system_event(self, event_data: Dict[str, Any]) -> bool:
        """处理系统事件"""
        try:
            event_type = event_data.get('type')
            if event_type == 'health_check':
                health_status = self._perform_health_check()
                logger.info(f"系统健康检查: {health_status}")
            elif event_type == 'performance_stats':
                stats = self._collect_performance_stats()
                logger.info(f"性能统计: {stats}")

            return True

        except Exception as e:
            logger.error(f"系统事件处理错误: {e}")
            return False

    def _process_log_event(self, event_data: Dict[str, Any]) -> bool:
        """处理日志事件"""
        try:
            log_level = event_data.get('level', 'INFO')
            message = event_data.get('message', '')

            if log_level == 'ERROR':
                logger.error(message)
            elif log_level == 'WARNING':
                logger.warning(message)
            elif log_level == 'DEBUG':
                logger.debug(message)
            else:
                logger.info(message)

            return True

        except Exception as e:
            logger.error(f"日志事件处理错误: {e}")
            return False

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        return {
            'realtime_engine': self.realtime_engine.health_check(),
            'level2_processor': self.level2_processor.get_stats(),
            'event_dispatcher': self.event_dispatcher.health_check(),
            'buffer_manager': self.buffer_manager.health_check()
        }

    def _collect_performance_stats(self) -> Dict[str, Any]:
        """收集性能统计"""
        uptime = time.time() - (self.performance_stats['start_time'] or time.time())
        return {
            **self.performance_stats,
            'system_uptime': uptime,
            'events_per_second': self.performance_stats['total_events_processed'] / max(uptime, 1),
            'level2_messages_per_second': self.performance_stats['total_level2_messages'] / max(uptime, 1),
            'trades_per_second': self.performance_stats['total_trades_processed'] / max(uptime, 1)
        }

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """启动实时引擎"""
        if self.running:
            logger.warning("实时引擎已在运行")
            return

        logger.info("启动实时引擎...")

        try:
            # 启动组件
            self.realtime_engine.start()
            self.event_dispatcher.start()

            # 启动监控线程
            self._start_monitor_threads()

            self.running = True
            self.performance_stats['start_time'] = time.time()

            logger.info("实时引擎启动成功")

        except Exception as e:
            logger.error(f"实时引擎启动失败: {e}")
            self.stop()
            raise

    def stop(self):
        """停止实时引擎"""
        if not self.running:
            return

        logger.info("停止实时引擎...")

        self.running = False
        self.shutdown_event.set()

        # 停止监控线程
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.stats_thread:
            self.stats_thread.join(timeout=5)

        # 停止组件
        self.realtime_engine.stop()
        self.event_dispatcher.stop()

        logger.info("实时引擎已停止")

    def _start_monitor_threads(self):
        """启动监控线程"""
        # 性能统计线程
        self.stats_thread = threading.Thread(
            target=self._stats_monitor_loop,
            name="StatsMonitor",
            daemon=True
        )
        self.stats_thread.start()

        # 健康检查线程
        self.monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.monitor_thread.start()

    def _stats_monitor_loop(self):
        """性能统计监控循环"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # 收集性能统计
                stats = self._collect_performance_stats()

                # 发布统计事件
                stats_event = Event(
                    event_type=EventType.SYSTEM,
                    timestamp=time.time(),
                    data={
                        'type': 'performance_stats',
                        'stats': stats
                    },
                    source="stats_monitor"
                )
                self.realtime_engine.publish_event(stats_event)

                # 分发统计事件
                self.event_dispatcher.dispatch_event(
                    {'type': 'performance_stats', 'stats': stats},
                    'system_event',
                    EventPriority.NORMAL
                )

                time.sleep(self.config.stats_interval)

            except Exception as e:
                logger.error(f"性能统计监控错误: {e}")
                time.sleep(1)

    def _health_monitor_loop(self):
        """健康检查监控循环"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # 执行健康检查
                health_status = self._perform_health_check()

                # 检查关键指标
                engine_health = health_status['realtime_engine']
                if not engine_health['running']:
                    logger.error("实时引擎运行状态异常")

                dispatcher_health = health_status['event_dispatcher']
                if dispatcher_health['queue_size'] > 1000:
                    logger.warning(f"事件队列积压: {dispatcher_health['queue_size']}")

                # 发布健康检查事件
                health_event = Event(
                    event_type=EventType.SYSTEM,
                    timestamp=time.time(),
                    data={
                        'type': 'health_check',
                        'health_status': health_status
                    },
                    source="health_monitor"
                )
                self.realtime_engine.publish_event(health_event)

                # 分发健康检查事件
                self.event_dispatcher.dispatch_event(
                    {'type': 'health_check', 'health_status': health_status},
                    'system_event',
                    EventPriority.NORMAL
                )

                time.sleep(self.config.health_check_interval)

            except Exception as e:
                logger.error(f"健康检查监控错误: {e}")
                time.sleep(1)

    def simulate_market_data(self, duration: int = 60):
        """模拟市场数据"""
        logger.info(f"开始模拟市场数据，持续时间: {duration}秒")

        start_time = time.time()
        event_count = 0

        while self.running and (time.time() - start_time) < duration:
            try:
                # 模拟Level2数据
                level2_data = self._generate_mock_level2_data()
                self.event_dispatcher.dispatch_event(
                    level2_data,
                    'level2_data',
                    EventPriority.HIGH
                )

                # 模拟成交数据
                trade_data = self._generate_mock_trade_data()
                self.event_dispatcher.dispatch_event(
                    trade_data,
                    'trade_data',
                    EventPriority.HIGH
                )

                event_count += 2
                self.performance_stats['total_events_processed'] += 2

                time.sleep(0.01)  # 100Hz数据频率

            except Exception as e:
                logger.error(f"模拟市场数据错误: {e}")
                time.sleep(0.1)

        logger.info(f"模拟市场数据完成，共处理 {event_count} 个事件")

    def _generate_mock_level2_data(self) -> Dict[str, Any]:
        """生成模拟Level2数据"""
        import random

        symbols = ['000001', '000002', '000858', '002415', '600000']
        symbol = random.choice(symbols)

        # 生成订单簿数据
        base_price = 10.0 + random.uniform(-2, 2)
        bids = []
        asks = []

        for i in range(5):
            bid_price = base_price - (i + 1) * 0.01
            ask_price = base_price + (i + 1) * 0.01
            bid_volume = random.randint(100, 1000)
            ask_volume = random.randint(100, 1000)

            bids.append((bid_price, bid_volume))
            asks.append((ask_price, ask_volume))

        # 构建原始数据
        raw_data = bytearray(94)
        raw_data[0:6] = symbol.encode('utf-8')

        # 时间戳
        timestamp = int(time.time() * 1000000)
        raw_data[6:14] = timestamp.to_bytes(8, 'little')

        # 买单数据
        for i, (price, volume) in enumerate(bids):
            offset = 14 + i * 8
            price_int = int(price * 100)
            raw_data[offset:offset+4] = price_int.to_bytes(4, 'little')
            raw_data[offset+4:offset+8] = volume.to_bytes(4, 'little')

        # 卖单数据
        for i, (price, volume) in enumerate(asks):
            offset = 54 + i * 8
            price_int = int(price * 100)
            raw_data[offset:offset+4] = price_int.to_bytes(4, 'little')
            raw_data[offset+4:offset+8] = volume.to_bytes(4, 'little')

        return {
            'data': bytes(raw_data),
            'market': 'A_SHARE'
        }

    def _generate_mock_trade_data(self) -> Dict[str, Any]:
        """生成模拟成交数据"""
        import random

        symbols = ['000001', '000002', '000858', '002415', '600000']
        symbol = random.choice(symbols)

        # 构建原始数据
        raw_data = bytearray(40)
        raw_data[0:6] = symbol.encode('utf-8')

        # 时间戳
        timestamp = int(time.time() * 1000000)
        raw_data[6:14] = timestamp.to_bytes(8, 'little')

        # 价格和成交量
        price = 10.0 + random.uniform(-2, 2)
        volume = random.randint(100, 1000)
        side = random.choice([1, 2])  # 1=买, 2=卖

        price_int = int(price * 100)
        raw_data[14:18] = price_int.to_bytes(4, 'little')
        raw_data[18:22] = volume.to_bytes(4, 'little')
        raw_data[22] = side

        # 成交ID
        trade_id = f"T{int(time.time() * 1000)}"
        raw_data[23:39] = trade_id.encode('utf-8').ljust(16, b'\x00')

        return {
            'data': bytes(raw_data),
            'market': 'A_SHARE'
        }

    def run_continuous(self):
        """持续运行模式"""
        logger.info("启动持续运行模式...")

        try:
            self.start()

            # 运行模拟数据
            self.simulate_market_data(duration=300)  # 5分钟

            # 持续运行
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("收到中断信号，开始关闭...")
        except Exception as e:
            logger.error(f"运行错误: {e}")
        finally:
            self.stop()


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建配置
    config = EngineConfig()

    # 创建主流程
    main_flow = RealTimeEngineMainFlow(config)

    try:
        # 运行主流程
        main_flow.run_continuous()

    except Exception as e:
        logger.error(f"主流程运行错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

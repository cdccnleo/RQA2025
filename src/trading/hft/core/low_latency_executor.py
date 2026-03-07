#!/usr/bin/env python3
"""
RQA2025 低延迟执行器
提供微秒级延迟的订单执行和风险控制
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time
import json
import queue
import socket
from collections import deque


logger = logging.getLogger(__name__)


class ExecutionType(Enum):

    """执行类型枚举"""
    MARKET_ORDER = "market_order"          # 市价单
    LIMIT_ORDER = "limit_order"           # 限价单
    STOP_ORDER = "stop_order"            # 止损单
    ICEBERG = "iceberg"                  # 冰山订单
    TWAP = "twap"                        # 时间加权平均价格
    VWAP = "vwap"                        # 成交量加权平均价格


class VenueType(Enum):

    """交易场所类型"""
    STOCK_EXCHANGE = "stock_exchange"      # 股票交易所
    FUTURES_EXCHANGE = "futures_exchange"  # 期货交易所
    DARK_POOL = "dark_pool"               # 暗池
    OTC = "otc"                          # 场外交易


@dataclass
class ExecutionOrder:

    """执行订单"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    execution_type: ExecutionType
    venue: VenueType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_limit: Optional[datetime] = None
    priority: int = 1  # 1 - 10, 10最高


@dataclass
class ExecutionResult:

    """执行结果"""
    order_id: str
    success: bool
    executed_quantity: float
    average_price: float
    total_cost: float
    latency_us: int
    venue: VenueType
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class VenueConnection:

    """交易场所连接"""
    venue: VenueType
    host: str
    port: int
    connection: Optional[socket.socket] = None
    connected: bool = False
    last_heartbeat: Optional[datetime] = None
    message_queue: queue.Queue = None

    def __post_init__(self):

        self.message_queue = queue.Queue()


class LowLatencyExecutor:

    """低延迟执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.max_latency_us = self.config.get('max_latency_us', 1000)  # 1ms
        self.risk_check_enabled = self.config.get('risk_check_enabled', True)
        self.pre_trade_risk_check = self.config.get('pre_trade_risk_check', True)
        self.max_position_per_symbol = self.config.get('max_position_per_symbol', 10000)
        self.max_order_rate_per_second = self.config.get('max_order_rate_per_second', 100)

        # 交易场所连接
        self.venue_connections: Dict[VenueType, VenueConnection] = {}
        self._initialize_venue_connections()

        # 执行队列
        self.execution_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # 持仓和风险跟踪
        self.positions: Dict[str, float] = {}  # symbol -> position
        self.order_rates: Dict[str, deque] = {}  # symbol -> timestamps

        # 性能监控
        self.performance_stats = {
            'orders_submitted': 0,
            'orders_executed': 0,
            'average_latency_us': 0,
            'execution_success_rate': 1.0,
            'total_volume': 0.0,
            'pnl_realized': 0.0
        }

        # 控制标志
        self.running = False
        self.emergency_stop = False

        # 线程管理
        self.execution_thread = None
        self.connection_threads: Dict[VenueType, threading.Thread] = {}

        logger.info("低延迟执行器初始化完成")

    def start(self):
        """启动执行器"""
        if self.running:
            logger.warning("执行器已在运行中")
            return

        self.running = True
        self.emergency_stop = False

        # 启动执行线程
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()

        # 启动连接管理线程
        for venue, connection in self.venue_connections.items():
            thread = threading.Thread(target=self._connection_manager, args=(venue,))
            thread.daemon = True
            thread.start()
            self.connection_threads[venue] = thread

        logger.info("低延迟执行器已启动")

    def stop(self):
        """停止执行器"""
        if not self.running:
            return

        logger.info("正在停止低延迟执行器...")
        self.running = False

        # 等待线程结束
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)

        for thread in self.connection_threads.values():
            if thread.is_alive():
                thread.join(timeout=2)

        self.connection_threads.clear()

        # 断开所有连接
        self._disconnect_all_venues()

        logger.info("低延迟执行器已停止")

    def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """提交订单"""
        if not self.running:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=0,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message="执行器未运行"
            )

        # 风险检查
        if self.pre_trade_risk_check:
            risk_check = self._check_order_risk(order)
        if not risk_check['approved']:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=0,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message=risk_check['reason']
            )

        # 记录提交时间
        submit_time = time.time_ns() // 1000  # 微秒

        try:
            # 放入执行队列
            self.execution_queue.put((order, submit_time))

            # 等待执行结果
            result = self.result_queue.get(timeout=5)  # 5秒超时
            self.result_queue.task_done()

            return result

        except queue.Empty:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=0,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message="执行超时"
            )

    def cancel_order(self, order_id: str, venue: VenueType) -> bool:
        """取消订单"""
        connection = self.venue_connections.get(venue)
        if not connection or not connection.connected:
            return False

        try:
            # 发送取消消息
            cancel_msg = self._create_cancel_message(order_id)
            connection.connection.sendall(cancel_msg)
            return True

        except Exception as e:
            logger.error(f"取消订单失败 {order_id}: {e}")
            return False

    def get_positions(self) -> Dict[str, float]:
        """获取持仓"""
        return self.positions.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()

    def _initialize_venue_connections(self):
        """初始化交易场所连接"""
        # 这里配置实际的交易场所连接信息
        # 在实际部署中，这些应该是从配置文件读取的

        venues_config = {
            VenueType.STOCK_EXCHANGE: {
                'host': 'exchange.example.com',
                'port': 9001
            },
            VenueType.FUTURES_EXCHANGE: {
                'host': 'futures.example.com',
                'port': 9002
            },
            VenueType.DARK_POOL: {
                'host': 'darkpool.example.com',
                'port': 9003
            }
        }

        for venue_type, config in venues_config.items():
            connection = VenueConnection(
                venue=venue_type,
                host=config['host'],
                port=config['port']
            )
            self.venue_connections[venue_type] = connection

    def _execution_loop(self):
        """执行循环"""
        while self.running:
            try:
                # 获取待执行订单
                try:
                    order_data = self.execution_queue.get(timeout=0.001)  # 1ms超时
                    order, submit_time = order_data
                except queue.Empty:
                    continue

                # 执行订单
                result = self._execute_order_internal(order, submit_time)

                # 返回结果
                self.result_queue.put(result)

                self.execution_queue.task_done()

            except Exception as e:
                logger.error(f"执行循环异常: {e}")

    def _execute_order_internal(self, order: ExecutionOrder, submit_time: int) -> ExecutionResult:
        """内部执行订单"""
        if self.emergency_stop:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=0,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message="紧急停止"
            )

        # 获取连接
        connection = self.venue_connections.get(order.venue)
        if not connection or not connection.connected:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=0,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message="交易场所未连接"
            )

        try:
            # 创建订单消息
            order_msg = self._create_order_message(order)

            # 发送订单
            start_time = time.time_ns() // 1000
            connection.connection.sendall(order_msg)

            # 等待确认（在实际实现中，这里会更复杂）
            response = self._receive_response(connection)
            end_time = time.time_ns() // 1000

            latency = int(end_time - start_time)

            if response and response.get('status') == 'accepted':
                # 模拟执行结果
                executed_qty = order.quantity
                avg_price = response.get('price', order.price or 100.0)
                total_cost = executed_qty * avg_price * 0.00025  # 手续费

                # 更新持仓
                self._update_position(order.symbol, order.side, executed_qty)

                # 更新统计
                self._update_performance_stats(order, latency, total_cost)

                return ExecutionResult(
                    order_id=order.order_id,
                    success=True,
                    executed_quantity=executed_qty,
                    average_price=avg_price,
                    total_cost=total_cost,
                    latency_us=latency,
                    venue=order.venue,
                    timestamp=datetime.now()
                )
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    executed_quantity=0,
                    average_price=0,
                    total_cost=0,
                    latency_us=latency,
                    venue=order.venue,
                    timestamp=datetime.now(),
                    error_message=response.get('message', '订单被拒绝')
                )

        except Exception as e:
            latency = int(time.time_ns() // 1000 - submit_time)
            logger.error(f"订单执行异常 {order.order_id}: {e}")

            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                executed_quantity=0,
                average_price=0,
                total_cost=0,
                latency_us=latency,
                venue=order.venue,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    def _connection_manager(self, venue: VenueType):
        """连接管理器"""
        connection = self.venue_connections[venue]

        while self.running:
            try:
                if not connection.connected:
                    self._connect_to_venue(connection)

                if connection.connected:
                    # 发送心跳
                    self._send_heartbeat(connection)

                    # 处理消息队列
                    self._process_message_queue(connection)

                time.sleep(1)  # 1秒检查间隔

            except Exception as e:
                logger.error(f"连接管理异常 {venue.value}: {e}")
                connection.connected = False
                time.sleep(5)  # 5秒后重试

    def _connect_to_venue(self, connection: VenueConnection):
        """连接到交易场所"""
        try:
            connection.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connection.connect((connection.host, connection.port))
            connection.connected = True
            connection.last_heartbeat = datetime.now()

            logger.info(f"连接到交易场所 {connection.venue.value} 成功")

        except Exception as e:
            logger.error(f"连接到交易场所 {connection.venue.value} 失败: {e}")
            connection.connected = False

    def _disconnect_all_venues(self):
        """断开所有交易场所连接"""
        for connection in self.venue_connections.values():
            if connection.connection:
                try:
                    connection.connection.close()
                except BaseException:
                    pass
                connection.connected = False

    def _create_order_message(self, order: ExecutionOrder) -> bytes:
        """创建订单消息"""
        # 简化的消息格式
        # 在实际实现中，这应该是更复杂的二进制协议

        message = {
            'type': 'new_order',
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': order.price,
            'execution_type': order.execution_type.value,
            'timestamp': datetime.now().isoformat()
        }

        # 序列化为JSON并编码为字节
        return json.dumps(message).encode('utf - 8')

    def _create_cancel_message(self, order_id: str) -> bytes:
        """创建取消消息"""
        message = {
            'type': 'cancel_order',
            'order_id': order_id,
            'timestamp': datetime.now().isoformat()
        }

        return json.dumps(message).encode('utf - 8')

    def _receive_response(self, connection: VenueConnection) -> Optional[Dict[str, Any]]:
        """接收响应"""
        try:
            # 设置超时
            connection.connection.settimeout(0.1)

            # 接收数据
            data = connection.connection.recv(1024)

            if data:
                response = json.loads(data.decode('utf - 8'))
                return response

        except socket.timeout:
            pass
        except Exception as e:
            logger.error(f"接收响应异常: {e}")

        return None

    def _send_heartbeat(self, connection: VenueConnection):
        """发送心跳"""
        if connection.last_heartbeat and \
           (datetime.now() - connection.last_heartbeat).seconds < 30:
            return

        try:
            heartbeat = {
                'type': 'heartbeat',
                'timestamp': datetime.now().isoformat()
            }

            connection.connection.sendall(json.dumps(heartbeat).encode('utf - 8'))
            connection.last_heartbeat = datetime.now()

        except Exception as e:
            logger.error(f"发送心跳失败: {e}")
            connection.connected = False

    def _process_message_queue(self, connection: VenueConnection):
        """处理消息队列"""
        # 这里处理来自交易场所的异步消息
        # 例如：成交确认、订单状态更新等

    def _check_order_risk(self, order: ExecutionOrder) -> Dict[str, Any]:
        """检查订单风险"""
        # 检查持仓限制
        current_position = self.positions.get(order.symbol, 0)
        new_position = current_position + \
            (order.quantity if order.side == 'buy' else -order.quantity)

        if abs(new_position) > self.max_position_per_symbol:
            return {
                'approved': False,
                'reason': f'超过持仓限制: {abs(new_position)} > {self.max_position_per_symbol}'
            }

        # 检查订单频率
        if order.symbol in self.order_rates:
            recent_orders = len([t for t in self.order_rates[order.symbol]
                                 if datetime.now() - t < timedelta(seconds=1)])

        if recent_orders >= self.max_order_rate_per_second:
            return {
                'approved': False,
                'reason': f'订单频率过高: {recent_orders} 订单 / 秒'
            }

        return {'approved': True}

    def _update_position(self, symbol: str, side: str, quantity: float):
        """更新持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = 0

        if side == 'buy':
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] -= quantity

        # 记录订单时间
        if symbol not in self.order_rates:
            self.order_rates[symbol] = deque(maxlen=100)

        self.order_rates[symbol].append(datetime.now())

    def _update_performance_stats(self, order: ExecutionOrder, latency_us: int, cost: float):
        """更新性能统计"""
        self.performance_stats['orders_submitted'] += 1

        # 更新平均延迟
        if latency_us > 0:
            current_avg = self.performance_stats['average_latency_us']
            count = self.performance_stats['orders_submitted']
            self.performance_stats['average_latency_us'] = (
                current_avg * (count - 1) + latency_us
            ) / count

        # 更新其他统计
        self.performance_stats['total_volume'] += order.quantity
        self.performance_stats['pnl_realized'] -= cost  # 简化的PnL计算

    def emergency_stop(self):
        """紧急停止"""
        self.emergency_stop = True
        logger.critical("低延迟执行器紧急停止，所有交易活动已暂停")

    def get_venue_status(self) -> Dict[str, Any]:
        """获取交易场所状态"""
        status = {}
        for venue, connection in self.venue_connections.items():
            status[venue.value] = {
                'connected': connection.connected,
                'last_heartbeat': connection.last_heartbeat.isoformat() if connection.last_heartbeat else None,
                'queue_size': connection.message_queue.qsize()
            }

        return status

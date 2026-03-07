#!/usr/bin/env python3
"""
QMT适配器
集成QMT量化交易平台，实现实时交易和数据流处理
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import threading
import time
import websocket
import json

from src.infrastructure.utils.logging.logger import get_logger

logger = logging.getLogger(__name__)


class QMTConnectionStatus(Enum):

    """QMT连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


class QMTOrderStatus(Enum):

    """QMT订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class QMTAdapter:

    """QMT适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.config = config or {}

        # QMT连接配置
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 8888)
        self.username = self.config.get('username', '')
        self.password = self.config.get('password', '')
        self.account_id = self.config.get('account_id', '')

        # 连接状态
        self.connection_status = QMTConnectionStatus.DISCONNECTED
        self.websocket = None
        self.session_id = None

        # 数据存储
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.account_info: Dict[str, Any] = {}

        # 回调函数
        self.on_market_data = None
        self.on_order_update = None
        self.on_position_update = None
        self.on_account_update = None

        # 线程控制
        self.running = False
        self.heartbeat_thread = None
        self.data_thread = None

        self.logger = get_logger(__name__)

    def connect(self) -> bool:
        """连接到QMT"""
        try:
            self.connection_status = QMTConnectionStatus.CONNECTING
            self.logger.info("开始连接QMT...")

            # 建立WebSocket连接
            ws_url = f"ws://{self.host}:{self.port}/ws"
            self.websocket = websocket.create_connection(ws_url, timeout=10)

            # 认证
            self.connection_status = QMTConnectionStatus.AUTHENTICATING
            auth_request = {
                'type': 'auth',
                'username': self.username,
                'password': self.password,
                'account_id': self.account_id
            }

            self.websocket.send(json.dumps(auth_request))
            response = self.websocket.recv()
            auth_result = json.loads(response)

            if auth_result.get('status') == 'success':
                self.session_id = auth_result.get('session_id')
                self.connection_status = QMTConnectionStatus.AUTHENTICATED
                self.logger.info("QMT认证成功")

                # 启动后台线程
                self.running = True
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
                self.data_thread = threading.Thread(target=self._data_loop)
                self.heartbeat_thread.daemon = True
                self.data_thread.daemon = True
                self.heartbeat_thread.start()
                self.data_thread.start()

                return True
            else:
                self.connection_status = QMTConnectionStatus.ERROR
                self.logger.error(f"QMT认证失败: {auth_result.get('message')}")
                return False

        except Exception as e:
            self.connection_status = QMTConnectionStatus.ERROR
            self.logger.error(f"QMT连接失败: {e}")
            return False

    def disconnect(self) -> bool:
        """断开QMT连接"""
        try:
            self.running = False

            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=2)

            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=2)

            if self.websocket:
                self.websocket.close()

            self.connection_status = QMTConnectionStatus.DISCONNECTED
            self.logger.info("QMT连接已断开")
            return True

        except Exception as e:
            self.logger.error(f"断开QMT连接时发生错误: {e}")
            return False

    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """订阅市场数据"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            self.logger.error("未连接到QMT，无法订阅数据")
            return False

        try:
            subscribe_request = {
                'type': 'subscribe',
                'session_id': self.session_id,
                'symbols': symbols,
                'data_type': 'market_data'
            }

            self.websocket.send(json.dumps(subscribe_request))
            response = self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == 'success':
                self.logger.info(f"市场数据订阅成功: {symbols}")
                return True
            else:
                self.logger.error(f"市场数据订阅失败: {result.get('message')}")
                return False

        except Exception as e:
            self.logger.error(f"订阅市场数据时发生错误: {e}")
            return False

    def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """取消订阅市场数据"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            return False

        try:
            unsubscribe_request = {
                'type': 'unsubscribe',
                'session_id': self.session_id,
                'symbols': symbols,
                'data_type': 'market_data'
            }

            self.websocket.send(json.dumps(unsubscribe_request))
            response = self.websocket.recv()
            result = json.loads(response)

            return result.get('status') == 'success'

        except Exception as e:
            self.logger.error(f"取消订阅市场数据时发生错误: {e}")
            return False

    def place_order(self, symbol: str, side: str, quantity: float,


                    order_type: str = 'market', price: Optional[float] = None) -> Optional[str]:
        """下单"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            self.logger.error("未连接到QMT，无法下单")
            return None

        try:
            order_request = {
                'type': 'place_order',
                'session_id': self.session_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }

            self.websocket.send(json.dumps(order_request))
            response = self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == 'success':
                order_id = result.get('order_id')
                self.logger.info(f"订单提交成功: {order_id}")
                return order_id
            else:
                self.logger.error(f"订单提交失败: {result.get('message')}")
                return None

        except Exception as e:
            self.logger.error(f"下单时发生错误: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            self.logger.error("未连接到QMT，无法取消订单")
            return False

        try:
            cancel_request = {
                'type': 'cancel_order',
                'session_id': self.session_id,
                'order_id': order_id
            }

            self.websocket.send(json.dumps(cancel_request))
            response = self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == 'success':
                self.logger.info(f"订单取消成功: {order_id}")
                return True
            else:
                self.logger.error(f"订单取消失败: {result.get('message')}")
                return False

        except Exception as e:
            self.logger.error(f"取消订单时发生错误: {e}")
            return False

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取持仓信息"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            return {}

        try:
            positions_request = {
                'type': 'get_positions',
                'session_id': self.session_id
            }

            self.websocket.send(json.dumps(positions_request))
            response = self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == 'success':
                positions = result.get('positions', {})
                self.positions = positions
                return positions
            else:
                self.logger.error(f"获取持仓失败: {result.get('message')}")
                return {}

        except Exception as e:
            self.logger.error(f"获取持仓时发生错误: {e}")
            return {}

    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        if self.connection_status != QMTConnectionStatus.AUTHENTICATED:
            return {}

        try:
            account_request = {
                'type': 'get_account',
                'session_id': self.session_id
            }

            self.websocket.send(json.dumps(account_request))
            response = self.websocket.recv()
            result = json.loads(response)

            if result.get('status') == 'success':
                account = result.get('account', {})
                self.account_info = account
                return account
            else:
                self.logger.error(f"获取账户信息失败: {result.get('message')}")
                return {}

        except Exception as e:
            self.logger.error(f"获取账户信息时发生错误: {e}")
            return {}

    def _heartbeat_loop(self) -> Any:
        """心跳循环"""
        while self.running:
            try:
                if self.connection_status == QMTConnectionStatus.AUTHENTICATED:
                    heartbeat = {
                        'type': 'heartbeat',
                        'session_id': self.session_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.websocket.send(json.dumps(heartbeat))
                time.sleep(30)  # 30秒心跳
            except Exception as e:
                self.logger.error(f"心跳发送失败: {e}")
                break

    def _data_loop(self) -> Any:
        """数据接收循环"""
        while self.running:
            try:
                if self.websocket and self.connection_status == QMTConnectionStatus.AUTHENTICATED:
                    message = self.websocket.recv()
                    data = json.loads(message)
                    self._handle_message(data)
                else:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"数据接收失败: {e}")
                break

    def _handle_message(self, data: Dict[str, Any]):
        """处理接收到的消息"""
        msg_type = data.get('type')

        if msg_type == 'market_data':
            self._handle_market_data(data)
        elif msg_type == 'order_update':
            self._handle_order_update(data)
        elif msg_type == 'position_update':
            self._handle_position_update(data)
        elif msg_type == 'account_update':
            self._handle_account_update(data)
        elif msg_type == 'heartbeat_ack':
            pass  # 心跳确认
        else:
            self.logger.warning(f"未知消息类型: {msg_type}")

    def _handle_market_data(self, data: Dict[str, Any]):
        """处理市场数据"""
        symbol = data.get('symbol')
        if symbol:
            self.market_data[symbol] = data

            # 调用回调函数
            if self.on_market_data:
                self.on_market_data(data)

    def _handle_order_update(self, data: Dict[str, Any]):
        """处理订单更新"""
        order_id = data.get('order_id')
        if order_id:
            self.orders[order_id] = data

            # 调用回调函数
            if self.on_order_update:
                self.on_order_update(data)

    def _handle_position_update(self, data: Dict[str, Any]):
        """处理持仓更新"""
        symbol = data.get('symbol')
        if symbol:
            self.positions[symbol] = data

            # 调用回调函数
            if self.on_position_update:
                self.on_position_update(data)

    def _handle_account_update(self, data: Dict[str, Any]):
        """处理账户更新"""
        self.account_info.update(data)

        # 调用回调函数
        if self.on_account_update:
            self.on_account_update(data)

    def get_connection_status(self) -> QMTConnectionStatus:
        """获取连接状态"""
        return self.connection_status

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.connection_status == QMTConnectionStatus.AUTHENTICATED

    def set_callbacks(self, on_market_data=None, on_order_update=None,


                      on_position_update=None, on_account_update=None):
        """设置回调函数"""
        self.on_market_data = on_market_data
        self.on_order_update = on_order_update
        self.on_position_update = on_position_update
        self.on_account_update = on_account_update

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        return self.market_data.get(symbol)

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        return self.orders.get(order_id)

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取持仓信息"""
        return self.positions.get(symbol)

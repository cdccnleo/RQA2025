import abc
import logging
from typing import Dict, List, Optional
from enum import Enum
import time
import threading
from dataclasses import dataclass
import requests
import pandas as pd

logger = logging.getLogger(__name__)

class GatewayStatus(Enum):
    """网关状态枚举"""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 3

class ProtocolType(Enum):
    """协议类型枚举"""
    FIX = 1
    REST = 2
    WEBSOCKET = 3

@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str
    balance: float
    available: float
    positions: Dict[str, float]

class BaseGateway(abc.ABC):
    """网关基类"""

    def __init__(self, name: str):
        self.name = name
        self.status = GatewayStatus.DISCONNECTED
        self.protocol = None
        self._lock = threading.Lock()
        self._active_orders = {}

    @abc.abstractmethod
    def connect(self, **kwargs):
        """连接网关"""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abc.abstractmethod
    def send_order(self, order) -> str:
        """发送订单"""
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str):
        """取消订单"""
        pass

    @abc.abstractmethod
    def query_account(self) -> AccountInfo:
        """查询账户信息"""
        pass

    @abc.abstractmethod
    def query_position(self, symbol: str = None) -> Dict[str, float]:
        """查询持仓"""
        pass

    def get_status(self) -> GatewayStatus:
        """获取网关状态"""
        return self.status

    def get_active_orders(self) -> Dict[str, dict]:
        """获取活跃订单"""
        with self._lock:
            return self._active_orders.copy()

    def on_order_update(self, order_id: str, status: str, filled: float, remaining: float, avg_price: float):
        """订单状态更新回调"""
        with self._lock:
            if order_id in self._active_orders:
                self._active_orders[order_id].update({
                    'status': status,
                    'filled': filled,
                    'remaining': remaining,
                    'avg_price': avg_price
                })

    def on_tick(self, symbol: str, bid: float, ask: float, volume: int):
        """行情更新回调"""
        # TODO: 实现行情处理
        pass

class FIXGateway(BaseGateway):
    """FIX协议网关"""

    def __init__(self):
        super().__init__("FIX Gateway")
        self.protocol = ProtocolType.FIX
        self.session_id = None

    def connect(self, host: str, port: int, sender_comp_id: str, target_comp_id: str, **kwargs):
        """连接FIX服务器"""
        self.status = GatewayStatus.CONNECTING
        logger.info(f"Connecting to FIX server {host}:{port}...")

        # TODO: 实现FIX协议连接
        time.sleep(1)  # 模拟连接延迟

        self.session_id = f"{sender_comp_id}-{target_comp_id}"
        self.status = GatewayStatus.CONNECTED
        logger.info("FIX connection established")

    def disconnect(self):
        """断开FIX连接"""
        self.status = GatewayStatus.DISCONNECTED
        self.session_id = None
        logger.info("FIX connection disconnected")

    def send_order(self, order) -> str:
        """发送FIX订单"""
        if self.status != GatewayStatus.CONNECTED:
            raise ConnectionError("Gateway not connected")

        order_id = f"FIX_{int(time.time()*1000)}"

        # TODO: 实现FIX订单发送
        logger.info(f"Sending FIX order {order_id}: {order}")

        with self._lock:
            self._active_orders[order_id] = {
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type,
                'quantity': order.quantity,
                'price': order.price,
                'status': 'NEW',
                'filled': 0,
                'remaining': order.quantity,
                'avg_price': 0
            }

        return order_id

    def cancel_order(self, order_id: str):
        """取消FIX订单"""
        if order_id not in self._active_orders:
            raise ValueError(f"Order {order_id} not found")

        # TODO: 实现FIX订单取消
        logger.info(f"Canceling FIX order {order_id}")

    def query_account(self) -> AccountInfo:
        """查询FIX账户"""
        # TODO: 实现FIX账户查询
        return AccountInfo(
            account_id="FIX_ACCOUNT",
            balance=1000000,
            available=800000,
            positions={}
        )

    def query_position(self, symbol: str = None) -> Dict[str, float]:
        """查询FIX持仓"""
        # TODO: 实现FIX持仓查询
        return {"AAPL": 1000} if symbol is None else {symbol: 1000}

class RESTGateway(BaseGateway):
    """REST协议网关"""

    def __init__(self):
        super().__init__("REST Gateway")
        self.protocol = ProtocolType.REST
        self.base_url = None

    def connect(self, base_url: str, api_key: str, secret: str, **kwargs):
        """配置REST连接"""
        self.status = GatewayStatus.CONNECTED  # REST无持久连接
        self.base_url = base_url
        self.api_key = api_key
        self.secret = secret
        logger.info("REST gateway configured")

    def disconnect(self):
        """REST网关无需断开"""
        self.status = GatewayStatus.DISCONNECTED
        logger.info("REST gateway reset")

    def send_order(self, order) -> str:
        """发送REST订单"""
        if self.status != GatewayStatus.CONNECTED:
            raise ConnectionError("Gateway not configured")

        order_id = f"REST_{int(time.time()*1000)}"
        payload = {
            "symbol": order.symbol,
            "side": order.side.name,
            "type": order.order_type.name,
            "quantity": order.quantity,
            "price": str(order.price) if order.price else None
        }

        # TODO: 实现签名和请求重试
        logger.info(f"Sending REST order {order_id}: {payload}")

        with self._lock:
            self._active_orders[order_id] = {
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type,
                'quantity': order.quantity,
                'price': order.price,
                'status': 'NEW',
                'filled': 0,
                'remaining': order.quantity,
                'avg_price': 0
            }

        return order_id

    def cancel_order(self, order_id: str):
        """取消REST订单"""
        if order_id not in self._active_orders:
            raise ValueError(f"Order {order_id} not found")

        # TODO: 实现REST订单取消
        logger.info(f"Canceling REST order {order_id}")

    def query_account(self) -> AccountInfo:
        """查询REST账户"""
        # TODO: 实现REST账户查询
        return AccountInfo(
            account_id="REST_ACCOUNT",
            balance=2000000,
            available=1500000,
            positions={}
        )

    def query_position(self, symbol: str = None) -> Dict[str, float]:
        """查询REST持仓"""
        # TODO: 实现REST持仓查询
        return {"MSFT": 500} if symbol is None else {symbol: 500}

class GatewayManager:
    """网关管理器"""

    def __init__(self):
        self.gateways = {}  # {name: gateway}

    def add_gateway(self, gateway: BaseGateway):
        """添加网关"""
        if gateway.name in self.gateways:
            raise ValueError(f"Gateway {gateway.name} already exists")

        self.gateways[gateway.name] = gateway
        logger.info(f"Gateway {gateway.name} added")

    def remove_gateway(self, name: str):
        """移除网关"""
        if name in self.gateways:
            self.gateways[name].disconnect()
            del self.gateways[name]
            logger.info(f"Gateway {name} removed")

    def get_gateway(self, name: str) -> BaseGateway:
        """获取网关"""
        return self.gateways.get(name)

    def get_all_gateways(self) -> Dict[str, BaseGateway]:
        """获取所有网关"""
        return self.gateways.copy()

    def send_order(self, gateway_name: str, order) -> str:
        """通过指定网关发送订单"""
        gateway = self.get_gateway(gateway_name)
        if not gateway:
            raise ValueError(f"Gateway {gateway_name} not found")

        return gateway.send_order(order)

    def cancel_order(self, gateway_name: str, order_id: str):
        """通过指定网关取消订单"""
        gateway = self.get_gateway(gateway_name)
        if not gateway:
            raise ValueError(f"Gateway {gateway_name} not found")

        gateway.cancel_order(order_id)

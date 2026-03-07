import abc
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import threading
from dataclasses import dataclass
import secrets
import logging

# 导入统一基础设施集成层
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
except ImportError:
    import logging
    def get_unified_logger(name): return logging.getLogger(name)


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


class MarketType(Enum):

    """市场类型枚举"""
    STOCK = "stock"           # 股票市场
    FUTURES = "futures"       # 期货市场
    OPTIONS = "options"       # 期权市场
    FOREX = "forex"          # 外汇市场
    CRYPTO = "crypto"        # 加密货币市场
    BONDS = "bonds"          # 债券市场


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

    @abc.abstractmethod
    def disconnect(self):
        """断开连接"""

    @abc.abstractmethod
    def send_order(self, order) -> str:
        """发送订单"""

    @abc.abstractmethod
    def cancel_order(self, order_id: str):
        """取消订单"""

    @abc.abstractmethod
    def query_account(self) -> AccountInfo:
        """查询账户信息"""

    @abc.abstractmethod
    def query_position(self, symbol: str = None) -> Dict[str, float]:
        """查询持仓"""

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


class FIXGateway(BaseGateway):

    """FIX协议网关"""

    def __init__(self, name: str = "FIX Gateway"):

        super().__init__(name)
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

        order_id = f"FIX_{int(time.time() * 1000)}"

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

    def __init__(self, name: str = "REST Gateway"):

        super().__init__(name)
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

        order_id = f"REST_{int(time.time() * 1000)}"
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

    """多市场网关管理器"""

    def __init__(self):

        self.gateways = {}  # {name: gateway}
        self.market_gateways = {}  # {market_type: [gateway_names]}
        self.gateway_loads = {}  # {name: load_score}
        self.gateway_health = {}  # {name: health_status}
        self.failover_history = []  # 故障转移历史

    def add_gateway(self, gateway: BaseGateway, market_types: Optional[List[MarketType]] = None):
        """添加网关并注册到指定市场"""
        if gateway.name in self.gateways:
            raise ValueError(f"Gateway {gateway.name} already exists")

        self.gateways[gateway.name] = gateway
        self.gateway_loads[gateway.name] = 0  # 初始负载为0
        self.gateway_health[gateway.name] = True  # 初始状态为健康

        # 注册到指定市场
        if market_types:
            for market_type in market_types:
                if market_type not in self.market_gateways:
                    self.market_gateways[market_type] = []
                self.market_gateways[market_type].append(gateway.name)

        logger.info(
            f"Gateway {gateway.name} added for markets: {[mt.value for mt in market_types] if market_types else 'all'}")

    def remove_gateway(self, name: str):
        """移除网关"""
        if name in self.gateways:
            self.gateways[name].disconnect()
            del self.gateways[name]

            # 清理相关的状态信息
            if name in self.gateway_loads:
                del self.gateway_loads[name]
            if name in self.gateway_health:
                del self.gateway_health[name]

            # 从市场网关映射中移除
            for market_type, gateway_names in self.market_gateways.items():
                if name in gateway_names:
                    gateway_names.remove(name)

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

    def cancel_order(self, gateway_name: str, order_id: str) -> bool:
        """通过指定网关取消订单"""
        gateway = self.get_gateway(gateway_name)
        if not gateway:
            raise ValueError(f"Gateway {gateway_name} not found")

        return gateway.cancel_order(order_id)

    def select_gateway_for_market(self, market_type: MarketType, symbol: str = None) -> Optional[str]:
        """
        根据市场类型和交易标的自动选择最优网关

        Args:
            market_type: 市场类型
            symbol: 交易标的（可选，用于更精确的选择）

        Returns:
            最优网关名称
        """
        if market_type not in self.market_gateways:
            logger.warning(f"No gateways available for market type: {market_type.value}")
            return None

        available_gateways = self.market_gateways[market_type]

        # 过滤出健康的网关
        healthy_gateways = [
            name for name in available_gateways if self.gateway_health.get(name, False)]

        if not healthy_gateways:
            logger.error(f"No healthy gateways available for market type: {market_type.value}")
            return None

        # 如果只有一个健康网关，直接返回
        if len(healthy_gateways) == 1:
            return healthy_gateways[0]

        # 使用负载均衡算法选择网关
        return self._select_gateway_by_load_balancing(healthy_gateways)

    def _select_gateway_by_load_balancing(self, gateway_names: List[str]) -> str:
        """
        基于负载均衡选择网关

        Args:
            gateway_names: 可用的网关名称列表

        Returns:
            选择的网关名称
        """
        # 计算每个网关的权重（基于负载）
        gateway_weights = {}
        for name in gateway_names:
            load = self.gateway_loads.get(name, 0)
            # 权重 = 1 / (1 + 负载)，负载越低权重越高
            weight = 1.0 / (1.0 + load)
            gateway_weights[name] = weight

        # 使用加权随机选择
        total_weight = sum(gateway_weights.values())
        if total_weight == 0:
            return secrets.choice(gateway_names)

        rand_value = secrets.randbelow(int(total_weight * 1000)) / 1000.0
        cumulative_weight = 0

        for name, weight in gateway_weights.items():
            cumulative_weight += weight
        if rand_value <= cumulative_weight:
            return name

        # 兜底返回第一个
        return gateway_names[0]

    def update_gateway_load(self, gateway_name: str, load_change: float):
        """
        更新网关负载

        Args:
            gateway_name: 网关名称
            load_change: 负载变化量（正数表示增加负载，负数表示减少负载）
        """
        if gateway_name in self.gateway_loads:
            self.gateway_loads[gateway_name] = max(
                0, self.gateway_loads[gateway_name] + load_change)
            logger.debug(
                f"Updated load for gateway {gateway_name}: {self.gateway_loads[gateway_name]}")

    def report_gateway_failure(self, gateway_name: str):
        """
        报告网关故障

        Args:
            gateway_name: 故障网关名称
        """
        if gateway_name in self.gateway_health:
            old_status = self.gateway_health[gateway_name]
            self.gateway_health[gateway_name] = False

        if old_status:  # 状态发生变化
            self.failover_history.append({
                'gateway': gateway_name,
                'timestamp': time.time(),
                'event': 'failure',
                'market_types': [mt for mt, gateways in self.market_gateways.items() if gateway_name in gateways]
            })
            logger.warning(f"Gateway {gateway_name} marked as failed")

    def recover_gateway(self, gateway_name: str):
        """
        恢复网关

        Args:
            gateway_name: 恢复的网关名称
        """
        if gateway_name in self.gateway_health:
            old_status = self.gateway_health[gateway_name]
            self.gateway_health[gateway_name] = True

        if not old_status:  # 状态发生变化
            self.failover_history.append({
                'gateway': gateway_name,
                'timestamp': time.time(),
                'event': 'recovery',
                'market_types': [mt for mt, gateways in self.market_gateways.items() if gateway_name in gateways]
            })
            logger.info(f"Gateway {gateway_name} recovered")

    def get_market_gateways(self, market_type: MarketType) -> List[str]:
        """
        获取指定市场的所有网关

        Args:
            market_type: 市场类型

        Returns:
            网关名称列表
        """
        return self.market_gateways.get(market_type, [])

    def get_gateway_status(self) -> Dict[str, Dict]:
        """
        获取所有网关的状态信息

        Returns:
            网关状态字典
        """
        status = {}
        for name, gateway in self.gateways.items():
            status[name] = {
                'status': gateway.status,
                'load': self.gateway_loads.get(name, 0),
                'health': self.gateway_health.get(name, True),
                'market_types': [mt for mt, gateways in self.market_gateways.items() if name in gateways]
            }
        return status

    def smart_send_order(self, order, market_type: MarketType = None, symbol: str = None) -> str:
        """
        智能发送订单（自动选择最优网关）

        Args:
            order: 订单对象
            market_type: 市场类型（如果不指定，尝试从订单中推断）
            symbol: 交易标的

        Returns:
            订单ID
        """
        # 如果未指定市场类型，尝试从symbol推断
        if market_type is None and symbol:
            market_type = self._infer_market_type_from_symbol(symbol)

        if market_type is None:
            raise ValueError("Cannot determine market type for order")

        # 选择最优网关
        gateway_name = self.select_gateway_for_market(market_type, symbol)
        if not gateway_name:
            raise ValueError(f"No available gateway for market type: {market_type.value}")

        # 更新负载
        self.update_gateway_load(gateway_name, 1.0)

        try:
            # 发送订单
            order_id = self.send_order(gateway_name, order)

            # 记录成功
            logger.info(f"Order sent via gateway {gateway_name}: {order_id}")
            return order_id

        except Exception as e:
            # 发送失败，标记网关故障
            self.report_gateway_failure(gateway_name)
            raise e

        finally:
            # 减少负载（无论成功失败）
            self.update_gateway_load(gateway_name, -1.0)

    def _infer_market_type_from_symbol(self, symbol: str) -> Optional[MarketType]:
        """
        从交易标的推断市场类型

        Args:
            symbol: 交易标的

        Returns:
            市场类型
        """
        # 简单的推断规则（可以根据需要扩展）
        if symbol.startswith(('6', '000', '002', '300')):
            return MarketType.STOCK  # A股
        elif symbol.startswith(('IF', 'IC', 'IH')):
            return MarketType.FUTURES  # 中金所股指期货
        elif symbol.endswith('USDT') or symbol.endswith('BTC'):
            return MarketType.CRYPTO  # 加密货币
        else:
            return MarketType.STOCK  # 默认认为是股票

    def get_gateways_for_market(self, market_type: MarketType) -> List[str]:
        """
        获取指定市场类型的网关列表

        Args:
            market_type: 市场类型

        Returns:
            网关名称列表
        """
        return self.market_gateways.get(market_type, [])

    def get_gateway_health_status(self, name: str) -> Optional[bool]:
        """
        获取网关健康状态

        Args:
            name: 网关名称

        Returns:
            健康状态，None表示网关不存在
        """
        return self.gateway_health.get(name)

    def get_gateway_load(self, name: str) -> Optional[float]:
        """
        获取网关负载

        Args:
            name: 网关名称

        Returns:
            负载值，None表示网关不存在
        """
        return self.gateway_loads.get(name)

    def get_gateway_statistics(self) -> Dict[str, Any]:
        """
        获取网关统计信息

        Returns:
            统计信息字典
        """
        total_gateways = len(self.gateways)
        healthy_gateways = sum(1 for health in self.gateway_health.values() if health)
        average_load = sum(self.gateway_loads.values()) / max(len(self.gateway_loads), 1)

        return {
            "total_gateways": total_gateways,
            "healthy_gateways": healthy_gateways,
            "average_load": average_load,
            "market_coverage": {mt.value: len(gateways) for mt, gateways in self.market_gateways.items()}
        }

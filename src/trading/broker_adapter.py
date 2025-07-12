from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class BrokerAdapter(ABC):
    """券商API适配器抽象基类"""

    def __init__(self, config: Dict):
        """
        初始化适配器

        Args:
            config: 券商配置信息
        """
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """连接券商系统"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def place_order(self, order: Dict) -> str:
        """
        下单

        Args:
            order: 订单信息 {
                "symbol": "标的代码",
                "direction": "buy/sell",
                "type": "market/limit",
                "quantity": 数量,
                "price": 价格(限价单需要),
                "account": "账户ID"
            }

        Returns:
            str: 订单ID
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            Dict: 订单状态 {
                "order_id": "订单ID",
                "status": "订单状态",
                "filled_quantity": 已成交数量,
                "avg_price": 成交均价,
                "timestamp": "更新时间"
            }
        """
        pass

    @abstractmethod
    def get_positions(self, account: Optional[str] = None) -> List[Dict]:
        """
        获取持仓

        Args:
            account: 指定账户, None表示所有账户

        Returns:
            List[Dict]: 持仓列表 [{
                "symbol": "标的代码",
                "quantity": 持仓数量,
                "cost_price": 成本价,
                "market_value": 市值
            }]
        """
        pass

    @abstractmethod
    def get_account_balance(self, account: str) -> Dict:
        """
        获取账户资金

        Args:
            account: 账户ID

        Returns:
            Dict: 资金信息 {
                "total_assets": 总资产,
                "available_cash": 可用资金,
                "margin": 保证金,
                "frozen_cash": 冻结资金
            }
        """
        pass

    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        获取实时行情数据

        Args:
            symbols: 标的代码列表

        Returns:
            Dict[str, Dict]: 行情数据 {
                "symbol": {
                    "last_price": 最新价,
                    "ask_price": 卖一价,
                    "bid_price": 买一价,
                    "volume": 成交量,
                    "timestamp": "更新时间"
                }
            }
        """
        # 默认实现，子类可覆盖
        return {symbol: {} for symbol in symbols}

class CTPSimulatorAdapter(BrokerAdapter):
    """CTP模拟器适配器"""

    def connect(self) -> bool:
        """连接CTP模拟器"""
        try:
            # 模拟连接逻辑
            logger.info("Connecting to CTP simulator...")
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"CTP connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        self.connected = False
        return True

    def place_order(self, order: Dict) -> str:
        """下单"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        # 生成订单ID
        order_id = f"CTP_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 模拟下单逻辑
        logger.info(f"Placing order: {order}")

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        logger.info(f"Cancelling order: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Dict:
        """获取订单状态"""
        # 模拟订单状态
        return {
            "order_id": order_id,
            "status": OrderStatus.FILLED.value,
            "filled_quantity": 100,
            "avg_price": 10.5,
            "timestamp": datetime.now().isoformat()
        }

    def get_positions(self, account: Optional[str] = None) -> List[Dict]:
        """获取持仓"""
        # 模拟持仓数据
        return [{
            "symbol": "600000",
            "quantity": 1000,
            "cost_price": 10.2,
            "market_value": 10200
        }]

    def get_account_balance(self, account: str) -> Dict:
        """获取账户资金"""
        return {
            "total_assets": 1000000,
            "available_cash": 800000,
            "margin": 0,
            "frozen_cash": 200000
        }

    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """获取实时行情"""
        return {
            symbol: {
                "last_price": 10.5,
                "ask_price": 10.51,
                "bid_price": 10.49,
                "volume": 100000,
                "timestamp": datetime.now().isoformat()
            }
            for symbol in symbols
        }

class BrokerAdapterFactory:
    """券商适配器工厂"""

    @staticmethod
    def create_adapter(broker_type: str, config: Dict) -> BrokerAdapter:
        """
        创建券商适配器

        Args:
            broker_type: 券商类型(ctp/simulator/xtp等)
            config: 券商配置

        Returns:
            BrokerAdapter: 适配器实例
        """
        if broker_type == "ctp":
            return CTPSimulatorAdapter(config)
        elif broker_type == "simulator":
            return CTPSimulatorAdapter(config)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")

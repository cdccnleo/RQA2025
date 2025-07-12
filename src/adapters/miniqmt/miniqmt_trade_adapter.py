from xtquant import xttrader
from core.interfaces.brokerage_api import BrokerageAPI
from core.exceptions import TradeError
import logging
from typing import Dict
import time

logger = logging.getLogger(__name__)

class MiniQMTTradeAdapter(BrokerageAPI):
    """MiniQMT交易接口适配器"""

    def __init__(self, config: Dict):
        self.config = config
        self.session_id = None
        self.xt = None
        self._connect()

    def _connect(self):
        """初始化交易连接"""
        try:
            self.xt = xttrader
            self.session_id = self.xt.init(
                account=self.config['account'],
                server=self.config['trade_server']
            )
            logger.info("MiniQMT交易接口初始化成功")
        except Exception as e:
            logger.error(f"MiniQMT交易接口初始化失败: {str(e)}")
            raise ConnectionError("MiniQMT交易连接失败")

    def place_order(self, order: Dict) -> str:
        """
        下单接口
        :param order: 订单信息字典
        :return: 订单ID
        """
        if not self.session_id:
            self._reconnect()

        try:
            # 转换订单类型
            xt_order_type = 0 if order['order_type'] == 'LIMIT' else 1

            # 调用下单接口
            order_id = self.xt.order_stock(
                session_id=self.session_id,
                stock_code=order['symbol'],
                order_type=xt_order_type,
                price_type=0,  # 限价
                price=order['price'],
                quantity=order['quantity'],
                strategy_name=order.get('strategy', 'default')
            )

            logger.info(f"下单成功: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            raise TradeError("MiniQMT下单失败")

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            return self.xt.cancel_order(self.session_id, order_id)
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            raise TradeError("MiniQMT取消订单失败")

    def _reconnect(self):
        """交易断线重连"""
        max_retries = 3
        for i in range(max_retries):
            try:
                self._connect()
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise
                time.sleep(1)

    def get_order_status(self, order_id: str) -> Dict:
        """获取订单状态"""
        try:
            return self.xt.query_order(self.session_id, order_id)
        except Exception as e:
            logger.error(f"查询订单状态失败: {str(e)}")
            raise TradeError("MiniQMT查询订单失败")

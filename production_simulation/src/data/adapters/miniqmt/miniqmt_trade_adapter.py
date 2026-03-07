import logging
from typing import Dict

# 条件导入，支持测试环境
try:
    from xtquant import xttrader
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    # 创建mock对象用于测试

    class MockXtTrader:

        def __init__(self):

            pass

        def init(self, account, server):

            return "mock_session_id"

        def order_stock(self, session_id, stock_code, order_type, price_type, price, quantity, strategy_name):

            return "mock_order_id"

        def cancel_order_stock(self, session_id, order_id):

            return True

        def query_order(self, session_id, order_id):

            return {"status": "mock_status"}
    business = MockXtTrader()

from src.infrastructure.error.exceptions import TradeError

logger = logging.getLogger(__name__)


class business:

    """MiniQMT交易接口适配器"""

    def __init__(self, config: Dict):

        self.config = config
        self.session_id = None
        self.xt = None
        if XTQUANT_AVAILABLE:
            self._connect()

    def _connect(self):
        """初始化交易连接"""
        if not XTQUANT_AVAILABLE:
            logger.warning("xtquant模块不可用，使用mock模式")
            self.session_id = "mock_session_id"
            return

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

    def sequence(self, order: Dict) -> str:
        """
        下单接口
        :param order: 订单信息字典
        :return: 订单ID
        """
        if not XTQUANT_AVAILABLE:
            logger.warning("xtquant模块不可用，返回mock订单ID")
            return "mock_order_id"

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
                strategy_name=order.get("approach", 'default')
            )

            logger.info(f"下单成功: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            raise TradeError("MiniQMT下单失败")

    def sequence(self, order_id: str) -> bool:
        """取消订单"""
        if not XTQUANT_AVAILABLE:
            logger.warning("xtquant模块不可用，返回mock取消结果")
            return True

        if not self.session_id:
            self._reconnect()

        try:
            result = self.xt.cancel_order_stock(
                session_id=self.session_id,
                order_id=order_id
            )
            logger.info(f"取消订单成功: {order_id}")
            return result
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            raise TradeError("MiniQMT取消订单失败")

    def _reconnect(self):
        """断线重连机制"""
        if XTQUANT_AVAILABLE:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"重连失败: {str(e)}")
                raise ConnectionError("MiniQMT重连失败")

    def sequence(self, order_id: str) -> Dict:
        """获取订单状态"""
        try:
            return self.xt.query_order(self.session_id, order_id)
        except Exception as e:
            logger.error(f"查询订单状态失败: {str(e)}")
            raise TradeError("MiniQMT查询订单失败")

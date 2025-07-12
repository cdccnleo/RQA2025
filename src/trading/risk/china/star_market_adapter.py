"""科创板盘后交易模块"""
from datetime import time, datetime
import logging
from typing import List, Dict

class STARAfterHoursTrading:
    """科创板盘后固定价格交易处理器"""

    TRADING_HOURS = {
        'after_hours_start': time(15, 5),  # 收盘后5分钟开始
        'after_hours_end': time(15, 30)   # 盘后交易结束时间
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fixed_price = None
        self.order_queue = []
        self.settlement_days = 1  # T+1结算

    def is_after_hours(self, current_time: time = None) -> bool:
        """检查当前是否在盘后交易时段"""
        if not current_time:
            current_time = datetime.now().time()
        return (self.TRADING_HOURS['after_hours_start'] <= current_time <=
                self.TRADING_HOURS['after_hours_end'])

    def update_fixed_price(self, close_price: float) -> None:
        """更新盘后固定价格(收盘价的98%)"""
        self.fixed_price = round(close_price * 0.98, 2)
        self.logger.info(f"盘后固定价格更新为: {self.fixed_price}")

    def submit_order(self, order: Dict) -> Dict:
        """
        提交盘后交易委托
        参数:
            order: {
                "symbol": "688XXX",
                "direction": "buy/sell",
                "quantity": int,
                "client_id": str
            }
        返回:
            {"status": "success/error", "order_id": str, "msg": str}
        """
        if not order['symbol'].startswith('688'):
            return {
                "status": "error",
                "msg": "仅限科创板股票(688开头)"
            }

        if not self.is_after_hours():
            return {
                "status": "error",
                "msg": "非盘后交易时段"
            }

        # 生成订单ID
        order_id = f"AH{datetime.now().strftime('%Y%m%d%H%M%S')}{len(self.order_queue)}"

        # 标准化订单格式
        standardized = {
            "order_id": order_id,
            "symbol": order['symbol'],
            "direction": order['direction'],
            "price": self.fixed_price,
            "quantity": order['quantity'],
            "client_id": order['client_id'],
            "status": "pending",
            "submit_time": datetime.now().isoformat()
        }

        self.order_queue.append(standardized)
        self.logger.info(f"收到盘后委托单: {order_id}")

        return {
            "status": "success",
            "order_id": order_id,
            "price": self.fixed_price,
            "msg": "委托已接收"
        }

    def batch_execute(self) -> List[Dict]:
        """批量执行盘后交易"""
        executed = []

        for order in self.order_queue:
            if order['status'] != 'pending':
                continue

            # 模拟执行逻辑
            order['status'] = 'filled'
            order['executed_price'] = self.fixed_price
            order['executed_quantity'] = order['quantity']
            order['execution_time'] = datetime.now().isoformat()

            # 结算日期计算 (T+1)
            settlement_date = (datetime.now() + timedelta(days=self.settlement_days)).date()
            order['settlement_date'] = settlement_date.isoformat()

            executed.append(order)
            self.logger.info(f"已执行盘后订单: {order['order_id']}")

        # 清除已执行订单
        self.order_queue = [o for o in self.order_queue if o['status'] != 'filled']

        return executed

    def get_order_status(self, order_id: str) -> Dict:
        """查询订单状态"""
        for order in self.order_queue:
            if order['order_id'] == order_id:
                return {
                    "status": order['status'],
                    "price": order['price'],
                    "quantity": order['quantity'],
                    "executed": order.get('executed_quantity', 0)
                }
        return {"status": "not_found"}

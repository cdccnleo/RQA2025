"""实时交易行为监控系统"""
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import numpy as np

class BehaviorMonitor:
    """交易行为实时监控器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_detectors = {
            'fat_finger': self._detect_fat_finger,
            'spoofing': self._detect_spoofing,
            'wash_trade': self._detect_wash_trade,
            'momentum_ignition': self._detect_momentum_ignition
        }
        self.alert_history = []

    def monitor_order_stream(self, orders: List[Dict]) -> List[Dict]:
        """
        监控订单流并检测异常模式
        参数:
            orders: 订单字典列表, 每个订单包含:
                   {'order_id', 'symbol', 'price', 'quantity', 'timestamp', 'account'}
        返回:
            检测到的异常列表
        """
        alerts = []

        for order in orders:
            for pattern_name, detector in self.pattern_detectors.items():
                if detector(order):
                    alert = self._generate_alert(pattern_name, order)
                    alerts.append(alert)
                    self.alert_history.append(alert)

                    # 实时触发风控动作
                    self._trigger_risk_control(alert)

        return alerts

    def _detect_fat_finger(self, order: Dict) -> bool:
        """检测乌龙指(异常大额订单)"""
        avg_qty = 1000  # 正常订单平均量
        return (order['quantity'] > avg_qty * 50 and
                abs(order['price'] - order['prev_price']) > 0.1)

    def _detect_spoofing(self, order: Dict) -> bool:
        """检测幌骗(虚假挂单)"""
        return (order['cancel_rate'] > 0.9 and
                order['lifetime'] < 1.0)  # 1秒内撤单率>90%

    def _detect_wash_trade(self, order: Dict) -> bool:
        """检测对敲交易(自买自卖)"""
        return (order['account'] == order['counterparty'] and
                abs(order['price'] - order['market_price']) > 0.05)

    def _detect_momentum_ignition(self, order: Dict) -> bool:
        """检测动量点燃(诱导跟风)"""
        return (order['quantity'] > 10000 and
                order['symbol'] in ['688XXX', '300XXX'] and
                order['time_interval'] < 5.0)  # 5秒内大单冲击

    def _generate_alert(self, pattern: str, order: Dict) -> Dict:
        """生成标准化警报"""
        return {
            "alert_id": f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "order_id": order['order_id'],
            "symbol": order['symbol'],
            "account": order['account'],
            "severity": "critical" if pattern == 'fat_finger' else "warning",
            "suggested_action": self._get_suggested_action(pattern)
        }

    def _get_suggested_action(self, pattern: str) -> str:
        """获取建议风控动作"""
        actions = {
            'fat_finger': 'freeze_account_and_cancel',
            'spoofing': 'review_and_limit_order',
            'wash_trade': 'report_to_regulator',
            'momentum_ignition': 'delay_execution'
        }
        return actions.get(pattern, 'monitor_only')

    def _trigger_risk_control(self, alert: Dict):
        """触发风控动作"""
        from src.trading.risk import RiskController

        action_map = {
            'freeze_account_and_cancel': RiskController.freeze_account,
            'review_and_limit_order': RiskController.limit_order_rate,
            'report_to_regulator': RiskController.report_violation,
            'delay_execution': RiskController.delay_execution,
            'monitor_only': lambda x: None
        }

        action = action_map.get(alert['suggested_action'])
        if action:
            action(alert['account'])
            self.logger.warning(
                f"触发风控动作: {alert['suggested_action']} "
                f"账户: {alert['account']}"
            )

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """获取最近N小时的警报记录"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history
                if datetime.fromisoformat(a['timestamp']) >= cutoff]

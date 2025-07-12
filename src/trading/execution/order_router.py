"""订单路由引擎实现"""

from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetric
from src.infrastructure.config.config_manager import ConfigManager
from src.trading.risk import RiskController
import numpy as np
import time

class OrderRouter:
    """智能订单路由引擎"""

    def __init__(self, config=None, metrics_collector=None):
        self.config = config or ConfigManager()
        self.metrics = metrics_collector or PerformanceMonitor('order_router')
        self.risk_controller = RiskController()

        # 初始化券商通道
        self.broker_channels = self._init_broker_channels()

    def _init_broker_channels(self):
        """初始化券商交易通道"""
        channels = {}
        for broker in self.config.get('execution.brokers'):
            channels[broker['id']] = {
                'adapter': self._load_broker_adapter(broker),
                'weight': broker.get('weight', 1.0),
                'latency': broker.get('latency', 0)
            }
        return channels

    def _load_broker_adapter(self, broker_config):
        """加载券商适配器"""
        # 实现动态加载不同券商的适配器
        pass

    def route_order(self, order):
        """
        路由订单到最优券商通道
        Args:
            order: 订单对象，包含股票代码、数量、价格等信息
        Returns:
            list: 拆分后的子订单列表
        """
        # 1. 风控检查
        risk_check = self.risk_controller.check_order(order)
        if not risk_check['allowed']:
            raise Exception(f"订单风控拒绝: {risk_check['reason']}")

        # 2. 智能路由决策
        optimal_channels = self._select_optimal_channels(order)

        # 3. 订单拆分
        child_orders = self._split_order(order, optimal_channels)

        # 记录路由指标
        self.metrics.record_metric(PerformanceMetric(
            name="order.routing.decision",
            value=len(child_orders),
            timestamp=time.time(),
            tags={
                'symbol': order['symbol'],
                'original_qty': str(order['quantity']),
                'channels': ','.join([c['broker'] for c in child_orders])
            }
        ))

        return child_orders

    def _select_optimal_channels(self, order):
        """选择最优券商通道"""
        # 考虑因素：价格、成交量、通道权重、延迟等
        scores = []
        for broker_id, channel in self.broker_channels.items():
            score = self._calculate_channel_score(channel, order)
            scores.append((broker_id, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 选择前N个通道
        max_channels = self.config.get('execution.max_channels', 3)
        return [broker_id for broker_id, _ in scores[:max_channels]]

    def _calculate_channel_score(self, channel, order):
        """计算通道得分"""
        # 基础得分 = 通道权重 * (1 - 标准化延迟)
        base_score = channel['weight'] * (1 - np.tanh(channel['latency'] / 100))

        # 考虑订单特性调整
        symbol_factor = 1.0
        if order.get('symbol_type') == 'STAR':
            symbol_factor = 1.2  # 科创板订单偏好特定通道

        return base_score * symbol_factor

    def _split_order(self, order, channels):
        """拆分订单到多个通道"""
        # 简单按通道数量平均拆分
        qty_per_order = order['quantity'] // len(channels)
        remainder = order['quantity'] % len(channels)

        child_orders = []
        for i, broker_id in enumerate(channels):
            qty = qty_per_order + (1 if i < remainder else 0)
            if qty == 0:
                continue

            child_order = {
                **order,
                'quantity': qty,
                'broker': broker_id,
                'parent_order_id': order.get('order_id')
            }
            child_orders.append(child_order)

        return child_orders

    def update_channel_status(self, broker_id, status):
        """更新券商通道状态"""
        if broker_id in self.broker_channels:
            self.broker_channels[broker_id].update(status)
            self.metrics.record_metric(PerformanceMetric(
                name="order.routing.channel_update",
                value=1,
                timestamp=time.time(),
                tags={
                    'broker': broker_id,
                    'status': status
                }
            ))

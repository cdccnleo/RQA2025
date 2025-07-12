"""订单路由引擎测试"""

import unittest
from unittest.mock import MagicMock, patch
from src.trading.execution.order_router import OrderRouter

class TestOrderRouter(unittest.TestCase):
    """订单路由引擎测试类"""

    def setUp(self):
        """测试初始化"""
        # 模拟配置
        self.mock_config = {
            'execution.brokers': [
                {'id': 'broker1', 'weight': 1.0, 'latency': 50},
                {'id': 'broker2', 'weight': 1.2, 'latency': 30},
                {'id': 'broker3', 'weight': 0.8, 'latency': 70}
            ],
            'execution.max_channels': 2
        }

        # 创建路由引擎实例
        self.router = OrderRouter()
        self.router.config = MagicMock()
        self.router.config.get.side_effect = lambda key: self.mock_config.get(key)

        # 模拟风控通过
        self.router.risk_controller = MagicMock()
        self.router.risk_controller.check_order.return_value = {'allowed': True}

        # 模拟指标收集
        self.router.metrics = MagicMock()

    def test_route_order_basic(self):
        """测试基本订单路由"""
        order = {
            'symbol': '600000.SH',
            'quantity': 1000,
            'price': 10.5
        }

        # 执行路由
        child_orders = self.router.route_order(order)

        # 检查结果
        self.assertEqual(len(child_orders), 2)  # 根据max_channels=2
        self.assertEqual(sum(o['quantity'] for o in child_orders), 1000)

        # 检查风控调用
        self.router.risk_controller.check_order.assert_called_once_with(order)

    def test_order_splitting(self):
        """测试订单拆分逻辑"""
        order = {
            'symbol': '600000.SH',
            'quantity': 100,
            'price': 10.5
        }

        # 设置固定通道选择
        with patch.object(self.router, '_select_optimal_channels') as mock_select:
            mock_select.return_value = ['broker1', 'broker2']

            # 执行路由
            child_orders = self.router.route_order(order)

            # 检查拆分
            self.assertEqual(len(child_orders), 2)
            self.assertEqual(child_orders[0]['quantity'], 50)
            self.assertEqual(child_orders[1]['quantity'], 50)

    def test_risk_rejection(self):
        """测试风控拒绝"""
        # 设置风控拒绝
        self.router.risk_controller.check_order.return_value = {
            'allowed': False,
            'reason': 'price_limit'
        }

        order = {
            'symbol': '600000.SH',
            'quantity': 1000,
            'price': 10.5
        }

        # 检查异常
        with self.assertRaises(Exception) as context:
            self.router.route_order(order)

        self.assertIn('订单风控拒绝', str(context.exception))

    def test_channel_selection(self):
        """测试通道选择逻辑"""
        # 测试科创板订单的特殊处理
        star_order = {
            'symbol': '688001.SH',
            'quantity': 1000,
            'price': 50.0,
            'symbol_type': 'STAR'
        }

        # 执行路由
        child_orders = self.router.route_order(star_order)

        # 检查通道选择
        self.assertEqual(len(child_orders), 2)

        # 检查指标记录
        self.router.metrics.record_routing_decision.assert_called_once()

    def test_channel_status_update(self):
        """测试通道状态更新"""
        # 更新通道状态
        self.router.update_channel_status('broker1', {'latency': 60})

        # 检查更新
        self.assertEqual(self.router.broker_channels['broker1']['latency'], 60)
        self.router.metrics.record_channel_update.assert_called_once_with(
            'broker1', {'latency': 60}
        )

if __name__ == '__main__':
    unittest.main()

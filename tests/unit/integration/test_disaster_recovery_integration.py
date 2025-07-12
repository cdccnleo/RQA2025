"""灾备系统集成测试"""
import unittest
import time
from unittest.mock import MagicMock, patch
from src.infrastructure.disaster_recovery import (
    DisasterRecovery,
    MultiLevelDisasterRecovery,
    DisasterRecoveryConfig
)
from src.features.feature_engine import FeatureEngine
from src.trading.order_executor import OrderManager
from src.data.data_manager import DataManager

class TestDisasterRecoveryIntegration(unittest.TestCase):
    """灾备系统集成测试用例"""

    def setUp(self):
        """测试初始化"""
        # 创建真实组件实例
        self.engine = FeatureEngine()
        self.order_manager = OrderManager()
        self.data_manager = DataManager()

        # 配置灾备系统
        self.config = DisasterRecoveryConfig(
            heartbeat_interval=0.5,  # 加快测试速度
            failover_timeout=1.0
        )

        # 创建主节点
        self.primary = DisasterRecovery(
            self.engine,
            self.order_manager,
            self.data_manager,
            self.config
        )

        # 创建备份节点
        self.backup = DisasterRecovery(
            FeatureEngine(),
            OrderManager(),
            DataManager(),
            self.config
        )

        # 注册备份节点
        self.primary.register_backup_node({"id": "backup1", "address": "127.0.0.1"})

        # 创建多级灾备系统
        self.ml_dr = MultiLevelDisasterRecovery(self.primary)
        self.ml_dr.add_secondary_node(self.backup)

    def test_failover_scenario(self):
        """测试故障切换场景"""
        # 1. 模拟主节点故障
        self.primary.health_status = False

        # 2. 等待故障检测
        time.sleep(self.config.heartbeat_interval * 2)

        # 3. 验证备份节点接管
        self.assertTrue(self.backup.is_primary)
        self.assertFalse(self.primary.is_primary)

        # 4. 验证交易恢复
        self.assertTrue(self.order_manager.is_trading_active())

    def test_data_consistency(self):
        """测试数据一致性"""
        # 1. 生成测试数据
        test_order = {"id": "test123", "symbol": "600519.SH", "quantity": 100}
        self.order_manager.place_order(test_order)

        # 2. 触发状态同步
        self.primary._sync_state()

        # 3. 验证备份节点数据
        self.assertEqual(
            self.order_manager.get_order("test123"),
            self.backup.order_manager.get_order("test123")
        )

    def test_multi_level_failover(self):
        """测试多级灾备切换"""
        # 1. 测试一级灾备
        with patch.object(self.primary, '_activate_failover') as mock_primary:
            self.ml_dr.activate_failover("level1")
            mock_primary.assert_called_once()

        # 2. 测试二级灾备
        with patch.object(self.backup, '_activate_failover') as mock_backup:
            self.ml_dr.activate_failover("level2")
            mock_backup.assert_called_once()

    def test_performance_metrics(self):
        """测试性能指标"""
        # 1. 测试故障切换时间
        start = time.time()
        self.primary._activate_failover()
        failover_time = time.time() - start
        self.assertLess(failover_time, 1.0)
        print(f"故障切换时间: {failover_time*1000:.2f}ms")

        # 2. 测试状态同步时间
        start = time.time()
        self.primary._sync_state()
        sync_time = time.time() - start
        self.assertLess(sync_time, 2.0)
        print(f"状态同步时间: {sync_time*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main()

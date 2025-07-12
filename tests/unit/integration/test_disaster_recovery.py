"""灾备系统集成测试"""
import unittest
import time
from unittest.mock import MagicMock, patch
from src.infrastructure.disaster_recovery import DisasterRecovery, DisasterRecoveryConfig
from src.features.feature_engine import FeatureEngine
from src.trading.order_executor import OrderManager
from src.data.data_manager import DataManager

class TestDisasterRecoveryIntegration(unittest.TestCase):
    """灾备系统集成测试用例"""

    def setUp(self):
        """测试初始化"""
        # 创建模拟组件
        self.engine = MagicMock(spec=FeatureEngine)
        self.order_manager = MagicMock(spec=OrderManager)
        self.data_manager = MagicMock(spec=DataManager)

        # 配置模拟组件行为
        self.engine.is_healthy.return_value = True
        self.order_manager.is_connected.return_value = True
        self.data_manager.is_available.return_value = True

        # 创建灾备系统实例
        config = DisasterRecoveryConfig(
            heartbeat_interval=0.1,  # 加快测试速度
            failover_timeout=0.2
        )
        self.dr = DisasterRecovery(
            self.engine,
            self.order_manager,
            self.data_manager,
            config
        )

        # 添加备份节点
        self.backup_node = MagicMock(spec=DisasterRecovery)
        self.dr.register_backup_node({"id": "backup1", "address": "127.0.0.1"})

    def test_primary_health_check(self):
        """测试主节点健康检查"""
        self.assertTrue(self.dr._check_health())

        # 模拟组件故障
        self.engine.is_healthy.return_value = False
        self.assertFalse(self.dr._check_health())

    def test_failover_activation(self):
        """测试故障切换激活"""
        # 模拟主节点故障
        self.dr.health_status = False

        # 触发监控检查
        self.dr._monitor_system()
        time.sleep(0.3)  # 等待超时

        # 验证是否尝试接管
        self.order_manager.suspend_trading.assert_called()
        self.order_manager.resume_trading.assert_called()

    def test_state_sync(self):
        """测试状态同步"""
        self.dr._sync_state()

        # 验证各组件同步方法被调用
        self.order_manager.sync_orders.assert_called()
        self.order_manager.sync_positions.assert_called()
        self.order_manager.sync_account.assert_called()
        self.data_manager.sync_latest_data.assert_called()

    def test_consistency_validation(self):
        """测试数据一致性验证"""
        # 模拟验证通过
        self.order_manager.validate_order_consistency.return_value = True
        self.order_manager.validate_position_consistency.return_value = True
        self.order_manager.validate_account_consistency.return_value = True

        self.assertTrue(self.dr._validate_consistency())

        # 模拟验证失败
        self.order_manager.validate_order_consistency.return_value = False
        self.assertFalse(self.dr._validate_consistency())

    def test_multi_level_failover(self):
        """测试多级灾备切换"""
        from src.infrastructure.disaster_recovery import MultiLevelDisasterRecovery

        # 创建多级灾备系统
        ml_dr = MultiLevelDisasterRecovery(self.dr)
        ml_dr.add_secondary_node(self.backup_node)

        # 测试一级灾备
        with patch.object(self.dr, '_activate_failover') as mock_failover:
            ml_dr.activate_failover("level1")
            mock_failover.assert_called()

        # 测试二级灾备
        with patch.object(self.backup_node, '_activate_failover') as mock_failover:
            ml_dr.activate_failover("level2")
            mock_failover.assert_called()

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 测试故障切换时间
        start = time.time()
        self.dr._activate_failover()
        elapsed = time.time() - start

        self.assertLess(elapsed, 1.0)  # 故障切换应在1秒内完成
        print(f"故障切换时间: {elapsed*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main()

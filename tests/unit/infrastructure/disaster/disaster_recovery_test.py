import unittest
import time
from unittest.mock import patch, MagicMock
from src.infrastructure.disaster import DisasterRecovery

class TestDisasterRecovery(unittest.TestCase):
    """灾备控制模块单元测试"""

    def setUp(self):
        self.dr = DisasterRecovery()
        # 模拟备份节点
        self.mock_backup = MagicMock()
        self.dr.backup_node = self.mock_backup
        # 模拟状态管理器
        self.mock_state = MagicMock()
        self.dr.state_manager = self.mock_state
        # 模拟数据验证器
        self.mock_validator = MagicMock()
        self.dr.data_validator = self.mock_validator
        # 模拟券商API
        self.mock_broker = MagicMock()
        self.dr.broker_api = self.mock_broker
        # 模拟告警管理器
        self.mock_alert = MagicMock()
        self.dr.alert_manager = self.mock_alert

    def test_heartbeat_monitoring(self):
        """测试心跳检测机制"""
        # 初始状态应为健康
        self.assertTrue(self.dr.is_healthy())

        # 模拟心跳超时
        with patch('time.time', return_value=time.time() + 30):
            self.assertFalse(self.dr.check_heartbeat())

        # 发送心跳后应恢复
        self.dr.send_heartbeat()
        self.assertTrue(self.dr.check_heartbeat())

    def test_failover_activation(self):
        """测试故障切换流程"""
        # 触发故障切换
        self.dr.activate_failover()

        # 验证流程执行顺序
        self.mock_backup.activate.assert_called_once()
        self.mock_state.restore_last_valid_state.assert_called_once()
        self.mock_validator.validate_consistency.assert_called_once()
        self.mock_broker.reconnect.assert_called_once()
        self.mock_alert.send_critical_alert.assert_called_once()

    def test_state_recovery(self):
        """测试状态恢复功能"""
        # 模拟有效状态点
        self.mock_state.get_last_valid_state.return_value = {"key": "value"}

        # 执行状态恢复
        self.dr.recover_state()

        # 验证状态恢复
        self.mock_state.restore_last_valid_state.assert_called_once_with({"key": "value"})

    def test_data_validation(self):
        """测试数据一致性校验"""
        # 模拟验证通过
        self.mock_validator.validate_consistency.return_value = True

        # 执行数据验证
        result = self.dr.validate_data()

        self.assertTrue(result)
        self.mock_validator.validate_consistency.assert_called_once()

    def test_auto_recovery(self):
        """测试自动恢复策略"""
        # 模拟故障场景
        self.dr.last_heartbeat = 0  # 模拟心跳丢失

        # 执行监控循环（单次迭代）
        with patch('time.sleep', side_effect=[None, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.dr.monitor_system()

        # 验证自动恢复触发
        self.mock_backup.activate.assert_called_once()

    def test_performance_metrics(self):
        """测试性能指标收集"""
        # 执行故障切换并记录耗时
        start_time = time.time()
        with patch('time.time', side_effect=[start_time, start_time + 0.5]):
            self.dr.activate_failover()

        # 验证性能指标
        metrics = self.dr.get_metrics()
        self.assertAlmostEqual(metrics['failover_time'], 0.5, places=2)

    def test_graceful_shutdown(self):
        """测试优雅关闭"""
        # 启动监控线程
        self.dr.start_monitoring()

        # 执行关闭
        self.dr.shutdown()

        # 验证线程已停止
        self.assertFalse(self.dr.monitor_thread.is_alive())

    def test_manual_failover(self):
        """测试手动触发故障切换"""
        # 初始状态应为活跃
        self.assertTrue(self.dr.is_active())

        # 手动触发切换
        self.dr.manual_failover()

        # 验证状态变为备用
        self.assertFalse(self.dr.is_active())
        self.mock_backup.activate.assert_called_once()

if __name__ == '__main__':
    unittest.main()

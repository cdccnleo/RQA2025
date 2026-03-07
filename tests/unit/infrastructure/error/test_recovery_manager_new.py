"""
基础设施层 - UnifiedRecoveryManager 单元测试

测试统一恢复管理器的核心功能，包括自动恢复、灾难恢复、降级服务等。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

from src.infrastructure.error.recovery.recovery import (
    UnifiedRecoveryManager,
    RecoveryStrategy,
    RecoveryAction,
    AutoRecoveryStrategy,
    DisasterRecoveryStrategy,
    ComponentStatus,
    ComponentHealth,
    RecoveryType,
    RecoveryPriority
)


class TestUnifiedRecoveryManager(unittest.TestCase):
    """UnifiedRecoveryManager 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.manager = UnifiedRecoveryManager()

    def tearDown(self):
        """测试后清理"""
        # 停止监控线程
        if hasattr(self.manager, '_monitoring_thread') and self.manager._monitoring_thread.is_alive():
            # 这里无法直接停止守护线程，依赖测试框架清理
            pass

    def test_initialization(self):
        """测试初始化"""
        manager = UnifiedRecoveryManager()

        # 验证组件健康状态字典已初始化
        self.assertIsInstance(manager._component_health, dict)
        self.assertIsInstance(manager._recovery_strategies, dict)
        self.assertIsInstance(manager._recovery_queue, list)

        # 验证默认策略已注册
        self.assertIn('auto', manager._recovery_strategies)
        self.assertIn('disaster', manager._recovery_strategies)
        self.assertIsInstance(manager._recovery_strategies['auto'], AutoRecoveryStrategy)
        self.assertIsInstance(manager._recovery_strategies['disaster'], DisasterRecoveryStrategy)

    def test_register_component(self):
        """测试注册组件"""
        component_name = "test_component"

        self.manager.register_component(component_name)

        # 验证组件已注册
        self.assertIn(component_name, self.manager._component_health)
        health = self.manager._component_health[component_name]
        self.assertIsInstance(health, ComponentHealth)
        self.assertEqual(health.component_name, component_name)
        self.assertEqual(health.status, ComponentStatus.HEALTHY)

    def test_update_component_health(self):
        """测试更新组件健康状态"""
        component_name = "test_component"

        # 注册组件
        self.manager.register_component(component_name)

        # 更新健康状态
        self.manager.update_component_health(component_name, ComponentStatus.FAILED, {"error": "Connection lost"})

        # 验证状态已更新
        health = self.manager._component_health[component_name]
        self.assertEqual(health.status, ComponentStatus.FAILED)
        self.assertEqual(health.last_failure, "Component test_component failed")
        self.assertEqual(health.failure_count, 1)

    def test_register_recovery_strategy(self):
        """测试注册恢复策略"""
        strategy_name = "custom_strategy"
        custom_strategy = AutoRecoveryStrategy()

        self.manager.register_recovery_strategy(strategy_name, custom_strategy)

        # 验证策略已注册
        self.assertIn(strategy_name, self.manager._recovery_strategies)
        self.assertEqual(self.manager._recovery_strategies[strategy_name], custom_strategy)

    def test_register_fallback_service(self):
        """测试注册降级服务"""
        service_name = "test_service"
        fallback_function = Mock(return_value="fallback_result")

        self.manager.register_fallback_service(service_name, fallback_function)

        # 验证降级服务已注册
        self.assertIn(service_name, self.manager._fallback_manager.fallback_services)
        self.assertEqual(self.manager._fallback_manager.fallback_services[service_name], fallback_function)

    def test_activate_fallback(self):
        """测试激活降级服务"""
        service_name = "test_service"
        fallback_function = Mock(return_value="fallback_result")

        # 注册降级服务
        self.manager.register_fallback_service(service_name, fallback_function)

        # 激活降级
        result = self.manager.activate_fallback(service_name)

        self.assertTrue(result)
        self.assertIn(service_name, self.manager._fallback_manager.active_fallbacks)
        self.assertTrue(self.manager._fallback_manager.active_fallbacks[service_name])

    def test_get_component_status(self):
        """测试获取组件状态"""
        component_name = "test_component"

        # 注册组件
        self.manager.register_component(component_name)
        status: Optional[ComponentHealth] = self.manager.get_component_status(component_name)

        self.assertIsNotNone(status)
        if status is not None:
            self.assertIsInstance(status, ComponentHealth)
            self.assertEqual(status.component_name, component_name)

    def test_get_all_component_status(self):
        """测试获取所有组件状态"""
        # 注册多个组件
        components = ["comp1", "comp2", "comp3"]
        for comp in components:
            self.manager.register_component(comp)

        all_status = self.manager.get_all_component_status()

        # 验证所有组件都在结果中
        for comp in components:
            self.assertIn(comp, all_status)
            self.assertIsInstance(all_status[comp], ComponentHealth)

    def test_get_recovery_stats(self):
        """测试获取恢复统计信息"""
        stats = self.manager.get_recovery_stats()

        # 验证统计信息结构
        self.assertIn('total_components', stats)
        self.assertIn('healthy_count', stats)
        self.assertIn('failed_count', stats)
        self.assertIn('recovery_strategies', stats)
        self.assertIn('recovery_queue_size', stats)

        # 验证初始状态
        self.assertEqual(stats['total_components'], 0)
        self.assertEqual(stats['healthy_count'], 0)
        self.assertEqual(stats['failed_count'], 0)
        self.assertEqual(stats['recovery_strategies'], 2)  # 默认2个策略
        self.assertEqual(stats['recovery_queue_size'], 0)

    def test_force_recovery(self):
        """测试强制恢复"""
        component_name = "test_component"

        # 注册组件并使其失败
        self.manager.register_component(component_name)
        self.manager.update_component_health(component_name, ComponentStatus.FAILED)

        # 强制恢复
        result = self.manager.force_recovery(component_name)

        # 验证恢复结果（取决于具体实现）
        self.assertIsInstance(result, bool)

    def test_auto_recovery_strategy_can_recover(self):
        """测试自动恢复策略的恢复判断"""
        strategy = AutoRecoveryStrategy()

        # 健康的组件
        healthy_component = ComponentHealth(
            component_name="healthy",
            status=ComponentStatus.HEALTHY,
            failure_count=0,
            last_check=time.time()
        )
        self.assertFalse(strategy.can_recover(healthy_component))

        # 已降级的组件（可以恢复）
        degraded_component = ComponentHealth(
            component_name="degraded",
            status=ComponentStatus.DEGRADED,
            failure_count=2,
            last_check=time.time() - 70  # 超过1分钟
        )
        self.assertTrue(strategy.can_recover(degraded_component))

    def test_auto_recovery_strategy_execute_recovery(self):
        """测试自动恢复策略执行恢复"""
        strategy = AutoRecoveryStrategy()
        component = ComponentHealth(
            component_name="test",
            status=ComponentStatus.FAILED,
            failure_count=1,
            last_check=time.time()
        )

        result = strategy.execute_recovery(component)

        self.assertTrue(result)
        self.assertEqual(component.status, ComponentStatus.HEALTHY)
        self.assertEqual(component.failure_count, 0)

    def test_disaster_recovery_strategy_can_recover(self):
        """测试灾难恢复策略的恢复判断"""
        strategy = DisasterRecoveryStrategy()
        strategy.backup_locations = ["/backup/location"]

        # 失败次数不足的组件
        failed_component = ComponentHealth(
            component_name="failed",
            status=ComponentStatus.FAILED,
            failure_count=3,
            last_check=time.time()
        )
        self.assertFalse(strategy.can_recover(failed_component))

        # 失败次数足够的组件
        failed_component.failure_count = 6
        self.assertTrue(strategy.can_recover(failed_component))

    def test_recovery_action_creation(self):
        """测试恢复动作创建"""
        mock_action_function = Mock(return_value=True)
        
        action = RecoveryAction(
            action_type="restart",
            component_name="database",
            priority=RecoveryPriority.CRITICAL,
            description="重启数据库服务",
            action_function=mock_action_function
        )

        self.assertEqual(action.action_type, "restart")
        self.assertEqual(action.component_name, "database")
        self.assertEqual(action.priority, RecoveryPriority.CRITICAL)
        self.assertEqual(action.description, "重启数据库服务")
        self.assertEqual(action.action_function, mock_action_function)

    def test_recovery_queue_management(self):
        """测试恢复队列管理"""
        # 验证初始队列为空
        self.assertEqual(len(self.manager._recovery_queue), 0)

        # 添加恢复动作
        mock_action_function = Mock(return_value=True)
        action = RecoveryAction(
            action_type="test",
            component_name="test_component",
            priority=RecoveryPriority.MEDIUM,
            description="测试恢复动作",
            action_function=mock_action_function
        )

        self.manager._recovery_queue.append(action)
        self.assertEqual(len(self.manager._recovery_queue), 1)

    def test_component_failure_counting(self):
        """测试组件失败计数"""
        component_name = "test_component"
        self.manager.register_component(component_name)

        # 多次失败
        for i in range(3):
            self.manager.update_component_health(component_name, ComponentStatus.FAILED, {"error": f"Error {i}"})

        health: Optional[ComponentHealth] = self.manager.get_component_status(component_name)
        if health is not None:
            self.assertEqual(health.failure_count, 3)
            self.assertEqual(health.last_failure, "Component test_component failed")

    def test_recovery_action_execution(self):
        """测试恢复动作执行"""
        mock_action_function = Mock(return_value=True)
        action = RecoveryAction(
            action_type="test",
            component_name="test_component",
            priority=RecoveryPriority.MEDIUM,
            description="测试恢复动作",
            action_function=mock_action_function
        )

        self.manager._execute_recovery_action(action)
        mock_action_function.assert_called_once()

    def test_health_check_performed(self):
        """测试健康检查执行"""
        component_name = "test_component"
        self.manager.register_component(component_name)

        component: Optional[ComponentHealth] = self.manager.get_component_status(component_name)
        if component is not None:
            initial_check_time = component.last_check

            # 执行健康检查
            self.manager._perform_health_check(component)

            # 验证检查时间已更新
            self.assertGreater(component.last_check, initial_check_time)

if __name__ == '__main__':
    unittest.main()
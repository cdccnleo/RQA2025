#!/usr/bin/env python3
"""
RQA2025 基础设施层集成测试框架

提供完整的集成测试环境，包括组件间通信测试、端到端流程测试、
性能基准测试和持续集成支持。
"""

import unittest
import time
import logging
import threading
import tempfile
import shutil
import json
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

from .component_registry import global_component_registry, ComponentRegistry
from .component_bus import global_component_bus, ComponentBus
from .performance_monitor import global_performance_monitor, PerformanceMonitor
from ..components.monitoring_coordinator import MonitoringCoordinator
from ..components.stats_collector import StatsCollector
from ..components.alert_manager import AlertManager
from ..components.metrics_exporter import MetricsExporter
from ..components.data_persistor import DataPersistor
from ..components.adaptive_configurator import AdaptiveConfigurator

logger = logging.getLogger(__name__)


class IntegrationTestEnvironment:
    """
    集成测试环境

    提供隔离的测试环境，包括独立的组件注册表、事件总线和性能监控器。
    """

    def __init__(self, test_name: str = "integration_test"):
        """
        初始化测试环境

        Args:
            test_name: 测试名称
        """
        self.test_name = test_name
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"rqa_integration_{test_name}_"))

        # 创建独立的组件实例
        self.component_registry = ComponentRegistry()
        self.component_bus = ComponentBus()
        self.performance_monitor = PerformanceMonitor()

        # 测试状态
        self.is_setup = False
        self.components_started = []
        self.test_data = {}

        logger.info(f"集成测试环境创建: {test_name}")

    def setup(self):
        """设置测试环境"""
        if self.is_setup:
            return

        # 创建必要的目录
        (self.temp_dir / "logs").mkdir(exist_ok=True)
        (self.temp_dir / "data").mkdir(exist_ok=True)
        (self.temp_dir / "config").mkdir(exist_ok=True)

        # 保存原始全局实例
        self._original_registry = global_component_registry
        self._original_bus = global_component_bus
        self._original_monitor = global_performance_monitor

        # 替换全局实例
        import src.infrastructure.monitoring.core.component_registry as reg_module
        import src.infrastructure.monitoring.core.component_bus as bus_module
        import src.infrastructure.monitoring.core.performance_monitor as mon_module

        reg_module.global_component_registry = self.component_registry
        bus_module.global_component_bus = self.component_bus
        mon_module.global_performance_monitor = self.performance_monitor

        self.is_setup = True
        logger.info(f"集成测试环境设置完成: {self.test_name}")

    def teardown(self):
        """清理测试环境"""
        if not self.is_setup:
            return

        # 停止所有组件
        for component_name in self.components_started:
            try:
                self.component_registry.stop_component(component_name)
            except Exception as e:
                logger.warning(f"停止组件失败 {component_name}: {e}")

        # 恢复原始全局实例
        import src.infrastructure.monitoring.core.component_registry as reg_module
        import src.infrastructure.monitoring.core.component_bus as bus_module
        import src.infrastructure.monitoring.core.performance_monitor as mon_module

        reg_module.global_component_registry = self._original_registry
        bus_module.global_component_bus = self._original_bus
        mon_module.global_performance_monitor = self._original_monitor

        # 清理临时目录
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

        self.is_setup = False
        logger.info(f"集成测试环境清理完成: {self.test_name}")

    def register_test_component(self, name: str, component_class: Type,
                              config: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册测试组件

        Args:
            name: 组件名称
            component_class: 组件类
            config: 配置

        Returns:
            bool: 是否成功注册
        """
        success = self.component_registry.register_component(
            name=name,
            component_class=component_class,
            config=config
        )

        if success:
            self.components_started.append(name)

        return success

    def start_component(self, name: str) -> bool:
        """
        启动组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功启动
        """
        return self.component_registry.start_component(name)

    def get_component_instance(self, name: str) -> Optional[Any]:
        """
        获取组件实例

        Args:
            name: 组件名称

        Returns:
            Optional[Any]: 组件实例
        """
        return self.component_registry.get_component_instance(name)

    def wait_for_event(self, event_type: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        等待事件发生

        Args:
            event_type: 事件类型
            timeout: 超时时间（秒）

        Returns:
            Optional[Dict[str, Any]]: 事件数据
        """
        start_time = time.time()
        events = []

        def event_handler(message):
            if message.type.value == event_type:
                events.append(message.payload)

        # 订阅事件
        self.component_bus.subscribe(event_type, event_handler)

        # 等待事件
        while time.time() - start_time < timeout:
            if events:
                return events[0]
            time.sleep(0.1)

        return None

    def inject_test_data(self, key: str, data: Any):
        """
        注入测试数据

        Args:
            key: 数据键
            data: 数据值
        """
        self.test_data[key] = data

    def get_test_data(self, key: str) -> Any:
        """
        获取测试数据

        Args:
            key: 数据键

        Returns:
            Any: 数据值
        """
        return self.test_data.get(key)

    def get_temp_file_path(self, filename: str) -> Path:
        """
        获取临时文件路径

        Args:
            filename: 文件名

        Returns:
            Path: 文件路径
        """
        return self.temp_dir / filename

    def simulate_system_load(self, cpu_usage: float = 50.0, memory_usage: float = 60.0):
        """
        模拟系统负载

        Args:
            cpu_usage: CPU 使用率
            memory_usage: 内存使用率
        """
        # 注入模拟的性能指标
        self.performance_monitor.record_metric("cpu_usage", cpu_usage)
        self.performance_monitor.record_metric("memory_usage", memory_usage)

        # 发布负载变化事件
        self.component_bus.publish_message("system.load.changed", {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': datetime.now().isoformat()
        })

    def get_environment_status(self) -> Dict[str, Any]:
        """
        获取环境状态

        Returns:
            Dict[str, Any]: 环境状态
        """
        return {
            'test_name': self.test_name,
            'temp_dir': str(self.temp_dir),
            'is_setup': self.is_setup,
            'components_count': len(self.components_started),
            'registry_health': self.component_registry.get_system_health(),
            'performance_metrics': len(self.performance_monitor.get_recent_metrics()),
            'test_data_keys': list(self.test_data.keys())
        }


class ComponentIntegrationTest(unittest.TestCase):
    """
    组件集成测试基类

    提供通用的集成测试功能和断言方法。
    """

    def setUp(self):
        """测试前准备"""
        self.env = IntegrationTestEnvironment(self.__class__.__name__)
        self.env.setup()

        # 注册核心组件
        self._register_core_components()

    def tearDown(self):
        """测试后清理"""
        self.env.teardown()

    def _register_core_components(self):
        """注册核心组件"""
        # 注册监控协调器
        self.env.register_test_component(
            "monitoring_coordinator",
            MonitoringCoordinator,
            config={"monitoring_interval": 5}
        )

        # 注册统计收集器
        self.env.register_test_component(
            "stats_collector",
            StatsCollector,
            config={"collection_interval": 5}
        )

        # 注册告警管理器
        self.env.register_test_component(
            "alert_manager",
            AlertManager,
            config={"check_interval": 5}
        )

        # 注册指标导出器
        self.env.register_test_component(
            "metrics_exporter",
            MetricsExporter,
            config={"export_interval": 10}
        )

        # 注册数据持久化器
        self.env.register_test_component(
            "data_persistor",
            DataPersistor,
            config={"storage_path": str(self.env.get_temp_file_path("monitoring_data.db"))}
        )

    def assertComponentRunning(self, component_name: str):
        """断言组件正在运行"""
        component = self.env.component_registry.get_component(component_name)
        self.assertIsNotNone(component, f"组件 {component_name} 未找到")
        self.assertTrue(component.is_running, f"组件 {component_name} 未运行")

    def assertEventReceived(self, event_type: str, timeout: int = 5) -> Dict[str, Any]:
        """断言收到事件"""
        event_data = self.env.wait_for_event(event_type, timeout)
        self.assertIsNotNone(event_data, f"未收到事件 {event_type}")
        return event_data

    def assertMetricsRecorded(self, metric_name: str, expected_value: Optional[float] = None):
        """断言指标已记录"""
        metrics = self.env.performance_monitor.get_recent_metrics()
        self.assertIn(metric_name, metrics, f"指标 {metric_name} 未记录")

        if expected_value is not None:
            self.assertAlmostEqual(metrics[metric_name], expected_value, places=2,
                                 msg=f"指标 {metric_name} 值不匹配")

    def assertSystemHealthy(self):
        """断言系统健康"""
        health = self.env.component_registry.get_system_health()
        self.assertGreaterEqual(health['running_components'], 1, "没有运行的组件")
        self.assertGreaterEqual(health['dependency_satisfaction'], 0.5, "依赖满足度太低")

    def waitForCondition(self, condition_func: Callable[[], bool], timeout: int = 10,
                        message: str = "条件未满足") -> bool:
        """
        等待条件满足

        Args:
            condition_func: 条件函数
            timeout: 超时时间
            message: 错误消息

        Returns:
            bool: 是否满足条件
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(0.1)

        self.fail(message)
        return False


class EndToEndTest(ComponentIntegrationTest):
    """
    端到端测试

    测试完整的监控流程从数据收集到告警触发的全过程。
    """

    def test_full_monitoring_cycle(self):
        """测试完整的监控周期"""
        # 启动所有组件
        for component_name in ["monitoring_coordinator", "stats_collector",
                             "alert_manager", "metrics_exporter", "data_persistor"]:
            self.assertTrue(self.env.start_component(component_name))
            self.assertComponentRunning(component_name)

        # 模拟系统负载
        self.env.simulate_system_load(cpu_usage=85.0, memory_usage=90.0)

        # 等待监控周期完成
        event_data = self.assertEventReceived("monitoring.cycle.completed", timeout=15)
        self.assertIn("stats_collected", event_data)
        self.assertIn("alerts_checked", event_data)

        # 验证指标已记录
        self.assertMetricsRecorded("cpu_usage", 85.0)
        self.assertMetricsRecorded("memory_usage", 90.0)

        # 验证数据已持久化
        persistor = self.env.get_component_instance("data_persistor")
        self.assertIsNotNone(persistor)

        # 系统应该健康
        self.assertSystemHealthy()

    def test_alert_triggering(self):
        """测试告警触发"""
        # 启动告警相关组件
        self.assertTrue(self.env.start_component("alert_manager"))
        self.assertTrue(self.env.start_component("stats_collector"))

        # 模拟高负载
        self.env.simulate_system_load(cpu_usage=95.0, memory_usage=95.0)

        # 等待告警事件
        alert_event = self.env.wait_for_event("alert.triggered", timeout=10)
        self.assertIsNotNone(alert_event, "未触发告警")

        # 验证告警内容
        self.assertIn("severity", alert_event)
        self.assertIn("message", alert_event)

    def test_adaptive_configuration(self):
        """测试自适应配置"""
        from ..components.adaptive_configurator import create_adaptive_configurator

        # 注册自适应配置器
        self.env.register_test_component(
            "adaptive_configurator",
            lambda: create_adaptive_configurator()
        )
        self.assertTrue(self.env.start_component("adaptive_configurator"))

        # 模拟持续高负载
        for _ in range(5):
            self.env.simulate_system_load(cpu_usage=90.0)
            time.sleep(1)

        # 等待配置调整
        config_event = self.env.wait_for_event("config.adaptive_change", timeout=15)
        if config_event:
            self.assertIn("component", config_event)
            self.assertIn("old_value", config_event)
            self.assertIn("new_value", config_event)


class PerformanceBenchmarkTest(ComponentIntegrationTest):
    """
    性能基准测试

    测试组件在不同负载下的性能表现。
    """

    def test_monitoring_throughput(self):
        """测试监控吞吐量"""
        # 启动核心组件
        self.assertTrue(self.env.start_component("stats_collector"))
        self.assertTrue(self.env.start_component("performance_monitor"))

        start_time = time.time()
        operations = 100

        # 执行多次监控操作
        for i in range(operations):
            self.env.simulate_system_load(
                cpu_usage=50.0 + (i % 50),
                memory_usage=60.0 + (i % 40)
            )

        end_time = time.time()
        duration = end_time - start_time

        throughput = operations / duration
        logger.info(".2f")

        # 验证吞吐量合理
        self.assertGreater(throughput, 10, "监控吞吐量太低")

    def test_memory_usage_stability(self):
        """测试内存使用稳定性"""
        import psutil
        import os

        # 启动所有组件
        for component_name in ["monitoring_coordinator", "stats_collector",
                             "alert_manager", "metrics_exporter"]:
            self.assertTrue(self.env.start_component(component_name))

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # 运行一段时间
        for i in range(50):
            self.env.simulate_system_load(cpu_usage=60.0, memory_usage=70.0)
            time.sleep(0.1)

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info(".2f")

        # 内存增长应该在合理范围内
        self.assertLess(memory_increase, 50, "内存增长过多")

    def test_component_response_times(self):
        """测试组件响应时间"""
        self.assertTrue(self.env.start_component("stats_collector"))

        response_times = []

        for _ in range(20):
            start_time = time.time()
            self.env.simulate_system_load(cpu_usage=70.0)
            event_data = self.env.wait_for_event("performance.metrics.updated", timeout=2)
            if event_data:
                response_time = time.time() - start_time
                response_times.append(response_time)

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)

            logger.info(".3f")
            logger.info(".3f")

            # 响应时间应该在合理范围内
            self.assertLess(avg_response_time, 1.0, "平均响应时间太长")
            self.assertLess(max_response_time, 2.0, "最大响应时间太长")


def create_integration_test_suite() -> unittest.TestSuite:
    """
    创建集成测试套件

    Returns:
        unittest.TestSuite: 测试套件
    """
    suite = unittest.TestSuite()

    # 添加端到端测试
    suite.addTest(EndToEndTest('test_full_monitoring_cycle'))
    suite.addTest(EndToEndTest('test_alert_triggering'))
    suite.addTest(EndToEndTest('test_adaptive_configuration'))

    # 添加性能基准测试
    suite.addTest(PerformanceBenchmarkTest('test_monitoring_throughput'))
    suite.addTest(PerformanceBenchmarkTest('test_memory_usage_stability'))
    suite.addTest(PerformanceBenchmarkTest('test_component_response_times'))

    return suite


@contextmanager
def integration_test_environment(test_name: str = "integration_test"):
    """
    集成测试环境上下文管理器

    Args:
        test_name: 测试名称
    """
    env = IntegrationTestEnvironment(test_name)
    try:
        env.setup()
        yield env
    finally:
        env.teardown()


def run_integration_tests(pattern: str = "*", verbose: bool = True) -> Dict[str, Any]:
    """
    运行集成测试

    Args:
        pattern: 测试模式
        verbose: 是否详细输出

    Returns:
        Dict[str, Any]: 测试结果
    """
    suite = create_integration_test_suite()

    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout if verbose else None
    )

    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'duration': end_time - start_time,
        'success': result.wasSuccessful()
    }


if __name__ == "__main__":
    import sys

    # 运行集成测试
    results = run_integration_tests()

    if results['success']:
        print("集成测试通过！")
        sys.exit(0)
    else:
        print(f"集成测试失败: {results['failures']} 失败, {results['errors']} 错误")
        sys.exit(1)

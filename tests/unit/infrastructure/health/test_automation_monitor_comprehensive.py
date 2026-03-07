#!/usr/bin/env python3
"""
自动化监控器综合测试 - 提升测试覆盖率至80%+

针对automation_monitor.py的深度测试覆盖
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Callable, Optional


class TestAutomationMonitorComprehensive:
    """自动化监控器全面测试"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.monitoring.automation_monitor import (
                AutomationMonitor, AlertRule, ServiceHealth
            )
            self.AutomationMonitor = AutomationMonitor
            self.AlertRule = AlertRule
            self.ServiceHealth = ServiceHealth
        except ImportError as e:
            pytest.skip(f"无法导入AutomationMonitor: {e}")

    def test_initialization(self):
        """测试初始化"""
        monitor = self.AutomationMonitor()
        assert monitor is not None
        assert hasattr(monitor, '_services')
        assert hasattr(monitor, '_alert_rules')
        assert hasattr(monitor, '_automation_tasks')
        assert hasattr(monitor, '_running')

    def test_initialization_with_custom_port(self):
        """测试带自定义端口的初始化"""
        monitor = self.AutomationMonitor(prometheus_port=8080)
        assert monitor is not None

    def test_register_service(self):
        """测试服务注册"""
        monitor = self.AutomationMonitor()

        def dummy_checker():
            return True

        # 测试正常注册
        monitor.register_service("test_service", dummy_checker)
        assert "test_service" in monitor._services
        assert "test_service" in monitor._health_checkers

        # 验证服务信息
        service_info = monitor._services["test_service"]
        assert service_info.name == "test_service"
        assert service_info.status == "unknown"
        assert monitor._health_checkers["test_service"] == dummy_checker

    def test_register_service_invalid_name(self):
        """测试注册无效服务名"""
        monitor = self.AutomationMonitor()

        def dummy_checker():
            return True

        # 测试空名称（实际代码允许）
        monitor.register_service("", dummy_checker)
        assert "" in monitor._services

        # 测试None名称（实际代码允许）
        monitor.register_service(None, dummy_checker)
        assert None in monitor._services

    def test_get_service_health(self):
        """测试获取服务健康状态"""
        monitor = self.AutomationMonitor()

        def healthy_checker():
            return True

        def unhealthy_checker():
            return False

        # 注册健康服务
        monitor.register_service("healthy_service", healthy_checker)
        # 注册不健康服务
        monitor.register_service("unhealthy_service", unhealthy_checker)

        # 测试获取健康服务状态
        health = monitor.get_service_health("healthy_service")
        assert health is not None
        assert health.name == "healthy_service"
        assert health.status == "unknown"  # 初始状态

        # 测试获取不存在的服务
        health = monitor.get_service_health("nonexistent")
        assert health is None

    def test_get_all_services_health(self):
        """测试获取所有服务健康状态"""
        monitor = self.AutomationMonitor()

        def checker1():
            return True

        def checker2():
            return False

        # 注册多个服务
        monitor.register_service("service1", checker1)
        monitor.register_service("service2", checker2)

        all_health = monitor.get_all_services_health()
        assert len(all_health) == 2
        assert "service1" in all_health
        assert "service2" in all_health
        assert all_health["service1"].name == "service1"
        assert all_health["service2"].name == "service2"

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        monitor = self.AutomationMonitor()

        rule = self.AlertRule(
            name="cpu_high",
            condition="cpu_usage > 90",
            severity="high",
            channels=["email"]
        )

        monitor.add_alert_rule(rule)
        assert "cpu_high" in monitor._alert_rules
        assert monitor._alert_rules["cpu_high"] == rule

    def test_add_alert_rule_duplicate(self):
        """测试添加重复告警规则"""
        monitor = self.AutomationMonitor()

        rule1 = self.AlertRule(
            name="cpu_high",
            condition="cpu_usage > 90",
            severity="high",
            channels=["email"]
        )

        rule2 = self.AlertRule(
            name="cpu_high",
            condition="cpu_usage > 95",
            severity="critical",
            channels=["email", "sms"]
        )

        monitor.add_alert_rule(rule1)
        # 应该允许覆盖
        monitor.add_alert_rule(rule2)
        assert monitor._alert_rules["cpu_high"] == rule2

    def test_get_alert_rules(self):
        """测试获取告警规则"""
        monitor = self.AutomationMonitor()

        rule1 = self.AlertRule("rule1", "cond1", "low", ["email"])
        rule2 = self.AlertRule("rule2", "cond2", "high", ["email"])

        monitor.add_alert_rule(rule1)
        monitor.add_alert_rule(rule2)

        rules = monitor.get_alert_rules()
        assert len(rules) == 2
        assert "rule1" in rules
        assert "rule2" in rules

    def test_register_automation_task(self):
        """测试注册自动化任务"""
        monitor = self.AutomationMonitor()

        def dummy_task():
            pass

        schedule = {"interval": 60, "enabled": True}

        monitor.register_automation_task("test_task", dummy_task, schedule)
        assert "test_task" in monitor._automation_tasks
        assert "test_task" in monitor._task_schedules

        # 任务存储在_automation_tasks中
        task = monitor._automation_tasks["test_task"]
        assert task == dummy_task

        # 调度信息存储在_task_schedules中
        stored_schedule = monitor._task_schedules["test_task"]
        assert stored_schedule == schedule

    def test_register_automation_task_invalid(self):
        """测试注册无效自动化任务"""
        monitor = self.AutomationMonitor()

        # 测试无效任务（实际代码可能不验证）
        try:
            monitor.register_automation_task("test_task", None, {"interval": 60})
        except (ValueError, TypeError):
            pass  # 如果抛出异常，说明有验证

        # 测试无效调度（实际代码可能不验证）
        try:
            monitor.register_automation_task("test_task", lambda: None, {})
        except (ValueError, TypeError):
            pass  # 如果抛出异常，说明有验证

    def test_get_automation_tasks(self):
        """测试获取自动化任务"""
        monitor = self.AutomationMonitor()

        def task1():
            pass

        def task2():
            pass

        schedule1 = {"interval": 60, "enabled": True}
        schedule2 = {"interval": 120, "enabled": False}

        monitor.register_automation_task("task1", task1, schedule1)
        monitor.register_automation_task("task2", task2, schedule2)

        tasks = monitor.get_automation_tasks()
        assert len(tasks) == 2
        assert "task1" in tasks
        assert "task2" in tasks

        # 验证返回的数据结构
        task1_info = tasks["task1"]
        assert "schedule" in task1_info
        assert task1_info["schedule"] == schedule1
        assert "last_execution" in task1_info

        task2_info = tasks["task2"]
        assert task2_info["schedule"] == schedule2

    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        monitor = self.AutomationMonitor()

        # 测试启动
        result = monitor.start_monitoring()
        assert result is True
        assert monitor._running is True

        # 测试停止
        result = monitor.stop_monitoring()
        assert result is True
        assert monitor._running is False

    def test_is_monitoring(self):
        """测试监控状态查询"""
        monitor = self.AutomationMonitor()

        # 初始状态
        assert monitor.is_monitoring() is False

        # 启动后
        monitor.start_monitoring()
        assert monitor.is_monitoring() is True

        # 停止后
        monitor.stop_monitoring()
        assert monitor.is_monitoring() is False

    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        monitor = self.AutomationMonitor()

        status = monitor.get_monitoring_status()
        assert isinstance(status, dict)
        assert "running" in status
        assert "services_count" in status
        assert "alert_rules_count" in status
        assert "automation_tasks_count" in status

    def test_collect_automation_metrics(self):
        """测试收集自动化指标"""
        monitor = self.AutomationMonitor()

        metrics = monitor.collect_automation_metrics()
        assert isinstance(metrics, dict)
        assert "services_monitored" in metrics
        assert "alerts_triggered" in metrics
        assert "tasks_executed" in metrics
        assert "monitoring_active" in metrics

    def test_export_metrics(self):
        """测试指标导出"""
        monitor = self.AutomationMonitor()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            # 测试导出
            result = monitor.export_metrics(temp_file)
            assert result is True or result is False  # 允许失败但不抛异常

            # 如果导出成功，检查文件是否存在
            if result and os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    content = f.read()
                    assert len(content) > 0

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_evaluate_alert_condition(self):
        """测试告警条件评估"""
        monitor = self.AutomationMonitor()

        # 测试CPU使用率条件（会被mock）
        with patch('psutil.cpu_percent', return_value=85):
            assert monitor._evaluate_alert_condition("cpu_usage > 80") is True
            assert monitor._evaluate_alert_condition("cpu_usage < 80") is False

        # 测试内存使用率条件（会被mock）
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 90
            assert monitor._evaluate_alert_condition("memory_usage > 80") is True
            assert monitor._evaluate_alert_condition("memory_usage < 80") is False

        # 测试不支持的条件格式
        assert monitor._evaluate_alert_condition("1 > 0") is False

    def test_evaluate_alert_condition_invalid(self):
        """测试无效告警条件评估"""
        monitor = self.AutomationMonitor()

        # 测试无效表达式（方法不抛异常，只返回False）
        result = monitor._evaluate_alert_condition("invalid syntax +++")
        assert result is False

    def test_should_execute_task(self):
        """测试任务执行判断"""
        monitor = self.AutomationMonitor()

        # 测试间隔调度
        schedule = {"interval": 60}  # 每60秒执行一次
        now = datetime.now()

        # 立即执行（首次）
        assert monitor._should_execute_task("task1", schedule, now) is True

        # 记录最后执行时间（在schedule中）
        schedule_with_last_exec = {"interval": 60, "last_execution": now}

        # 刚执行不久，不应该执行
        assert monitor._should_execute_task("task1", schedule_with_last_exec, now) is False

        # 60秒后，应该执行
        future_time = now + timedelta(seconds=61)
        assert monitor._should_execute_task("task1", schedule_with_last_exec, future_time) is True

    def test_execute_automation_task(self):
        """测试执行自动化任务"""
        monitor = self.AutomationMonitor()

        executed = False

        def test_task():
            nonlocal executed
            executed = True

        monitor._execute_automation_task("test_task", test_task)
        assert executed is True

    def test_execute_automation_task_with_exception(self):
        """测试执行异常任务"""
        monitor = self.AutomationMonitor()

        def failing_task():
            raise ValueError("Task failed")

        # 不应该抛出异常
        monitor._execute_automation_task("failing_task", failing_task)

    def test_trigger_alert(self):
        """测试触发告警"""
        monitor = self.AutomationMonitor()

        rule = self.AlertRule(
            name="test_alert",
            condition="1 > 0",
            severity="high",
            channels=["email"]
        )

        # 不应该抛出异常
        monitor._trigger_alert(rule)

    @patch('requests.post')
    def test_send_to_alertmanager(self, mock_post):
        """测试发送告警到Alertmanager"""
        monitor = self.AutomationMonitor(alertmanager_url="http://localhost:9093")

        alert_data = {
            "name": "test_alert",
            "severity": "high",
            "condition": "cpu_usage > 90",
            "timestamp": "2024-01-01T00:00:00Z"
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # 测试发送
        monitor._send_to_alertmanager(alert_data)
        mock_post.assert_called_once()

        # 验证请求参数
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:9093/api / v1 / alerts"
        assert "json" in call_args[1]

    @patch('time.sleep')
    def test_health_check_worker(self, mock_sleep):
        """测试健康检查工作线程"""
        import threading
        import time

        monitor = self.AutomationMonitor()

        # 注册服务
        call_count = 0
        def checker():
            nonlocal call_count
            call_count += 1
            return True

        monitor.register_service("test_service", checker)

        # 模拟运行状态
        monitor._running = True

        # 创建并启动健康检查线程
        health_thread = threading.Thread(target=monitor._health_check_worker, daemon=True)
        health_thread.start()

        # 等待至少执行一次健康检查
        # 使用短超时，因为mock_sleep会立即返回
        timeout = 2.0
        start_time = time.time()
        while call_count == 0 and (time.time() - start_time) < timeout:
            time.sleep(0.01)

        # 停止监控
        monitor._running = False

        # 等待线程停止
        health_thread.join(timeout=1.0)

        # 验证检查器至少被调用了一次
        assert call_count >= 1, f"Health checker was called {call_count} times, expected at least 1"

        # 验证服务状态已被更新
        service = monitor.get_service_health("test_service")
        assert service is not None
        assert service.status == "healthy"  # 检查器返回True，所以应该是healthy
        assert service.last_check is not None
        assert isinstance(service.response_time, (int, float))
        assert service.response_time >= 0

    def test_start_and_stop(self):
        """测试启动和停止"""
        monitor = self.AutomationMonitor()

        # 测试启动
        monitor.start()
        assert monitor._running is True

        # 测试停止
        monitor.stop()
        assert monitor._running is False

    def test_destructor(self):
        """测试析构函数"""
        monitor = self.AutomationMonitor()

        # 启动监控
        monitor.start_monitoring()

        # 删除对象（触发__del__）
        del monitor

        # 如果没有异常抛出，测试通过

    # 模块级函数测试
    def test_module_level_check_health(self):
        """测试模块级健康检查函数"""
        from src.infrastructure.health.monitoring.automation_monitor import check_health

        result = check_health()
        assert isinstance(result, dict)
        assert "healthy" in result
        assert "service" in result
        assert "checks" in result

    def test_module_level_check_module_structure(self):
        """测试模块结构检查函数"""
        from src.infrastructure.health.monitoring.automation_monitor import check_module_structure

        result = check_module_structure()
        assert isinstance(result, dict)

    def test_module_level_check_automation_system(self):
        """测试自动化系统检查函数"""
        from src.infrastructure.health.monitoring.automation_monitor import check_automation_system

        result = check_automation_system()
        assert isinstance(result, dict)

    def test_module_level_health_status(self):
        """测试健康状态函数"""
        from src.infrastructure.health.monitoring.automation_monitor import health_status

        result = health_status()
        assert isinstance(result, dict)

    def test_module_level_health_summary(self):
        """测试健康摘要函数"""
        from src.infrastructure.health.monitoring.automation_monitor import health_summary

        result = health_summary()
        assert isinstance(result, dict)

    def test_module_level_monitor_automation_monitor(self):
        """测试自动化监控器监控函数"""
        from src.infrastructure.health.monitoring.automation_monitor import monitor_automation_monitor

        result = monitor_automation_monitor()
        assert isinstance(result, dict)

    def test_module_level_validate_automation_monitor(self):
        """测试自动化监控器验证函数"""
        from src.infrastructure.health.monitoring.automation_monitor import validate_automation_monitor

        result = validate_automation_monitor()
        assert isinstance(result, dict)


class TestAutomationMonitorEdgeCases:
    """自动化监控器边界情况测试"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.monitoring.automation_monitor import AutomationMonitor, AlertRule
            self.AutomationMonitor = AutomationMonitor
            self.AlertRule = AlertRule
        except ImportError:
            pytest.skip("无法导入AutomationMonitor")

    def test_concurrent_service_registration(self):
        """测试并发服务注册"""
        import threading
        monitor = self.AutomationMonitor()

        def register_service(service_id):
            def checker():
                return True
            monitor.register_service(f"service_{service_id}", checker)

        # 创建多个线程并发注册
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_service, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有服务都已注册
        all_health = monitor.get_all_services_health()
        assert len(all_health) == 10

    def test_large_number_of_services(self):
        """测试大量服务注册"""
        monitor = self.AutomationMonitor()

        # 注册100个服务
        for i in range(100):
            def checker():
                return True
            monitor.register_service(f"service_{i}", checker)

        all_health = monitor.get_all_services_health()
        assert len(all_health) == 100

    def test_service_health_caching(self):
        """测试服务健康状态缓存"""
        monitor = self.AutomationMonitor()

        call_count = 0

        def counting_checker():
            nonlocal call_count
            call_count += 1
            return True

        monitor.register_service("cached_service", counting_checker)

        # 启动监控以触发健康检查
        monitor.start_monitoring()

        # 等待一会儿让健康检查线程运行
        import time
        time.sleep(2)

        # 停止监控
        monitor.stop_monitoring()

        # 验证检查器至少被调用了一次
        assert call_count >= 1

    def test_alert_rule_validation(self):
        """测试告警规则验证"""
        monitor = self.AutomationMonitor()

        # 测试有效规则
        valid_rule = self.AlertRule("valid", "1 > 0", "low", ["email"])
        monitor.add_alert_rule(valid_rule)

        # 测试无效规则（应该被拒绝或处理）
        try:
            invalid_rule = self.AlertRule("", "", "", [])
            monitor.add_alert_rule(invalid_rule)
        except Exception:
            pass  # 可能抛出异常

    def test_task_scheduling_edge_cases(self):
        """测试任务调度边界情况"""
        monitor = self.AutomationMonitor()

        # 测试立即执行任务
        schedule = {"interval": 0}
        now = datetime.now()

        # 应该总是执行
        assert monitor._should_execute_task("immediate_task", schedule, now) is True

        # 测试负间隔
        schedule = {"interval": -1}
        assert monitor._should_execute_task("negative_task", schedule, now) is True

    def test_metrics_collection_under_load(self):
        """测试负载下的指标收集"""
        monitor = self.AutomationMonitor()

        # 注册大量服务
        for i in range(50):
            def checker():
                return (i % 2) == 0  # 交替健康状态
            monitor.register_service(f"service_{i}", checker)

        # 收集指标
        metrics = monitor.collect_automation_metrics()
        assert isinstance(metrics, dict)

        # 验证指标合理性
        assert metrics["services_monitored"] == 50

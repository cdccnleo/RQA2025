"""
test_execution_monitor_component 模块

提供 test_execution_monitor_component 相关功能和接口。

测试执行监控组件

实现测试执行过程的监控和跟踪功能：
- 测试注册和状态跟踪
- 超时检测和处理
- 测试历史记录管理
- 性能指标关联
"""

import logging
import threading
import time

from ..alert_dataclasses import TestExecutionInfo, PerformanceMetrics
from ..alert_enums import MonitoringEvent
from ..shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any

class MonitoringTestExecutionMonitor:
    """Test execution monitor class"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.active_tests: Dict[str, TestExecutionInfo] = {}
        self.test_history: List[TestExecutionInfo] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 配置参数
        self.timeout_seconds = 300  # 默认5分钟超时
        self.check_interval = 1  # 检查间隔(秒)

        # 事件处理器
        self.event_handlers: Dict[MonitoringEvent, List[Callable]] = {}

        # 配置日志和错误处理
        self.logger: ILogger = StandardLogger(f"{self.__class__.__name__}")
        self.error_handler: IErrorHandler = BaseErrorHandler()

        # 应用配置
        if config:
            self._apply_config(config)

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置"""
        self.timeout_seconds = config.get('timeout_seconds', self.timeout_seconds)
        self.check_interval = config.get('check_interval', self.check_interval)

    def register_event_handler(self, event: MonitoringEvent, handler: Callable):
        """注册事件处理器"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
        self.logger.log_info(f"注册事件处理器: {event.value}")

    def _trigger_event(self, event: MonitoringEvent, test_info: TestExecutionInfo):
        """触发事件"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(test_info)
                except Exception as e:
                    self.error_handler.handle_error(e, f"事件处理器执行错误: {event.value}")

    def start_monitoring(self):
        """Start monitoring"""
        if self.monitoring:
            return

        try:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.log_info("测试执行监控已启动")
        except Exception as e:
            self.error_handler.handle_error(e, "启动测试执行监控失败")
            self.monitoring = False

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.log_info("测试执行监控已停止")

    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                self._check_test_timeouts()
                self._check_test_status()
                time.sleep(self.check_interval)
            except Exception as e:
                self.error_handler.handle_error(e, "测试监控循环出错")
                time.sleep(1)  # 短暂延迟后重试

    def _check_test_timeouts(self):
        """Check test timeouts"""
        current_time = datetime.now()
        timeout_tests = []

        with self._lock:
            for test_id, test_info in self.active_tests.items():
                if test_info.status == "running":
                    # 检查是否超时
                    elapsed = (current_time - test_info.start_time).total_seconds()
                    if elapsed > self.timeout_seconds:
                        test_info.status = "timeout"
                        test_info.end_time = current_time
                        test_info.error_message = f"测试执行超时 ({self.timeout_seconds}秒)"
                        timeout_tests.append(test_id)

        # 处理超时测试
        for test_id in timeout_tests:
            self._handle_test_timeout(test_id)

    def _check_test_status(self):
        """检查测试状态变化"""
        # 这里可以添加其他状态检查逻辑
        pass

    def _handle_test_timeout(self, test_id: str):
        """处理测试超时"""
        with self._lock:
            if test_id in self.active_tests:
                test_info = self.active_tests[test_id]
                self.logger.warning(f"测试超时: {test_id} ({test_info.test_name})")

                # 触发超时事件
                self._trigger_event(MonitoringEvent.TEST_TIMEOUT, test_info)

                # 移动到历史记录
                self.test_history.append(test_info)
                del self.active_tests[test_id]

    def register_test(self, test_id: str, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """注册测试"""
        test_info = TestExecutionInfo(
            test_id=test_id,
            test_name=test_name,
            start_time=datetime.now()
        )

        # 添加元数据
        if metadata:
            test_info.details = metadata

        with self._lock:
            self.active_tests[test_id] = test_info

        self.logger.log_info(f"注册测试: {test_id} ({test_name})")

        # 触发测试开始事件
        self._trigger_event(MonitoringEvent.TEST_STARTED, test_info)

        return test_id

    def update_test_status(self, test_id: str, status: str,
                          execution_time: Optional[float] = None,
                          error_message: Optional[str] = None,
                          performance_metrics: Optional[PerformanceMetrics] = None):
        """Update test status"""
        with self._lock:
            if test_id in self.active_tests:
                test_info = self.active_tests[test_id]
                test_info.status = status
                test_info.end_time = datetime.now()
                test_info.execution_time = execution_time
                test_info.error_message = error_message
                test_info.performance_metrics = performance_metrics

                # 移动到历史记录
                self.test_history.append(test_info)
                del self.active_tests[test_id]

                # 触发相应事件
                if status == "completed":
                    self._trigger_event(MonitoringEvent.TEST_COMPLETED, test_info)
                elif status == "failed":
                    self._trigger_event(MonitoringEvent.TEST_FAILED, test_info)

                self.logger.log_info(f"测试状态更新: {test_id} -> {status}")

    def get_active_tests(self) -> List[TestExecutionInfo]:
        """获取活跃测试"""
        with self._lock:
            return list(self.active_tests.values())

    def get_test_history(self, hours: int = 24) -> List[TestExecutionInfo]:
        """获取测试历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [t for t in self.test_history if t.start_time > cutoff_time]

    def get_test_by_id(self, test_id: str) -> Optional[TestExecutionInfo]:
        """根据ID获取测试信息"""
        with self._lock:
            # 先检查活跃测试
            if test_id in self.active_tests:
                return self.active_tests[test_id]

            # 再检查历史记录
            for test_info in self.test_history:
                if test_info.test_id == test_id:
                    return test_info

        return None

    def get_test_statistics(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        with self._lock:
            total_tests = len(self.test_history)
            active_tests = len(self.active_tests)

            # 状态统计
            status_counts = {}
            for test_info in self.test_history:
                status = test_info.status
                status_counts[status] = status_counts.get(status, 0) + 1

            # 成功率计算
            completed_tests = [t for t in self.test_history if t.status == "completed"]
            success_rate = len(completed_tests) / total_tests if total_tests > 0 else 0

            # 平均执行时间
            avg_execution_time = 0
            timed_tests = [t for t in self.test_history if t.execution_time is not None]
            if timed_tests:
                avg_execution_time = sum(t.execution_time for t in timed_tests) / len(timed_tests)

            return {
                "total_tests": total_tests,
                "active_tests": active_tests,
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "status_distribution": status_counts
            }

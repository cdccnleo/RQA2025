"""
测试执行管理器

职责：管理测试执行、监控和报告
"""

import threading
import time

from ..alert_dataclasses import TestExecutionInfo
from ..shared_interfaces import ILogger, StandardLogger
from typing import Dict, List, Optional, Any


class TestExecutionManager:
    """
    测试执行管理器

    职责：管理测试执行、监控和报告
    """

    def __init__(self, test_monitor, logger: Optional[ILogger] = None):
        self.test_monitor = test_monitor
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._lock = threading.Lock()
        self._active_tests: Dict[str, TestExecutionInfo] = {}

    def register_test(self, test_id: str, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """注册测试"""
        with self._lock:
            try:
                execution_id = f"{test_id}_{int(time.time())}"
                test_info = TestExecutionInfo(
                    execution_id=execution_id,
                    test_id=test_id,
                    test_name=test_name,
                    status="registered",
                    start_time=time.time(),
                    metadata=metadata or {}
                )
                self._active_tests[execution_id] = test_info
                self.logger.log_info(f"测试已注册: {execution_id}")
                return execution_id
            except Exception as e:
                self.logger.log_error(f"注册测试失败: {e}")
                return ""

    def update_test_status(self, execution_id: str, status: str,
                          message: Optional[str] = None,
                          metrics: Optional[Dict[str, Any]] = None):
        """更新测试状态"""
        with self._lock:
            try:
                if execution_id in self._active_tests:
                    test_info = self._active_tests[execution_id]
                    test_info.status = status
                    test_info.end_time = time.time()
                    if message:
                        test_info.metadata['message'] = message
                    if metrics:
                        test_info.metadata['metrics'] = metrics
                    self.logger.log_info(f"测试状态已更新: {execution_id} -> {status}")
                else:
                    self.logger.log_warning(f"未找到测试执行: {execution_id}")
            except Exception as e:
                self.logger.log_error(f"更新测试状态失败: {e}")

    def get_active_tests(self) -> List[TestExecutionInfo]:
        """获取活跃测试"""
        with self._lock:
            return list(self._active_tests.values())

    def get_test_history(self, hours: int = 24) -> List[TestExecutionInfo]:
        """获取测试历史"""
        # 简化的实现，返回活跃测试
        return self.get_active_tests()
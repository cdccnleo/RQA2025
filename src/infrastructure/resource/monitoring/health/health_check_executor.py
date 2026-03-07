
import threading
import time

from ...core.shared_interfaces import ILogger, StandardLogger
from .health_check_manager import HealthCheck
from typing import Dict, List, Any, Optional
"""
健康检查执行器

职责：执行具体的健康检查逻辑
"""


class HealthCheckExecutor:
    """
    健康检查执行器

    职责：执行具体的健康检查逻辑
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._lock = threading.RLock()

    def execute_check(self, check: HealthCheck) -> Dict[str, Any]:
        """执行单个健康检查"""
        start_time = time.time()

        try:
            # 设置超时
            result = self._execute_with_timeout(check.check_function, check.timeout)

            execution_time = time.time() - start_time

            return {
                'check_name': check.name,
                'status': 'success',
                'result': result,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'error': None
            }

        except Exception as e:
            execution_time = time.time() - start_time

            return {
                'check_name': check.name,
                'status': 'failed',
                'result': None,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'error': str(e)
            }

    def execute_checks(self, checks: List[HealthCheck]) -> List[Dict[str, Any]]:
        """批量执行健康检查"""
        results = []

        for check in checks:
            if check.enabled:
                result = self.execute_check(check)
                results.append(result)
            else:
                results.append({
                    'check_name': check.name,
                    'status': 'disabled',
                    'result': None,
                    'execution_time': 0,
                    'timestamp': time.time(),
                    'error': None
                })

        return results

    def _execute_with_timeout(self, func: callable, timeout: int) -> Any:
        """带超时的函数执行"""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"检查执行超时: {timeout}秒")
        elif exception[0]:
            raise exception[0]
        else:
            return result[0]

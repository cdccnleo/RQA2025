
import threading
import time

from ...core.shared_interfaces import ILogger, StandardLogger
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
"""
健康状态报告器

职责：处理健康状态报告和查询
"""


class HealthStatusReporter:
    """
    健康状态报告器

    职责：处理健康状态报告和查询
    """

    def __init__(self, max_history: int = 1000, retention_hours: int = 24,
                 logger: Optional[ILogger] = None):
        self.max_history = max_history
        self.retention_period = timedelta(hours=retention_hours)
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

        self._results_history: deque = deque(maxlen=max_history)
        self._current_status: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def report_result(self, result: Dict[str, Any]) -> None:
        """报告检查结果"""
        with self._lock:
            # 添加时间戳
            result['reported_at'] = time.time()
            self._results_history.append(result)

            # 更新当前状态
            self._current_status[result['check_name']] = result

            self.logger.log_debug(f"已报告检查结果: {result['check_name']}")

    def get_current_status(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """获取当前状态"""
        with self._lock:
            if check_name:
                return self._current_status.get(check_name, {})

            # 汇总所有检查的状态
            all_statuses = list(self._current_status.values())
            if not all_statuses:
                return {'overall_status': 'unknown', 'checks': {}}

            # 计算整体状态
            failed_checks = [s for s in all_statuses if s.get('status') == 'failed']
            warning_checks = [s for s in all_statuses if s.get('status') == 'warning']

            if failed_checks:
                overall_status = 'critical'
            elif warning_checks:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'

            return {
                'overall_status': overall_status,
                'total_checks': len(all_statuses),
                'failed_checks': len(failed_checks),
                'warning_checks': len(warning_checks),
                'healthy_checks': len([s for s in all_statuses if s.get('status') == 'success']),
                'last_updated': max((s.get('timestamp', 0) for s in all_statuses), default=time.time()),
                'checks': dict(self._current_status)
            }

    def get_health_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取健康报告"""
        with self._lock:
            # 清理过期数据
            self._cleanup_expired_results()

            # 获取指定时间范围内的结果
            since_time = time.time() - hours * 3600
            recent_results = [r for r in self._results_history if r.get(
                'timestamp', 0) >= since_time]

            if not recent_results:
                return {
                    'period_hours': hours,
                    'total_results': 0,
                    'success_rate': 0.0,
                    'avg_execution_time': 0.0,
                    'error_summary': {}
                }

            # 统计分析
            total_results = len(recent_results)
            successful_results = [r for r in recent_results if r.get('status') == 'success']
            success_rate = len(successful_results) / total_results

            execution_times = [r.get('execution_time', 0) for r in successful_results]
            avg_execution_time = sum(execution_times) / \
                len(execution_times) if execution_times else 0

            # 错误统计
            error_summary = {}
            for result in recent_results:
                if result.get('error'):
                    error_type = type(result['error']).__name__
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1

            return {
                'period_hours': hours,
                'total_results': total_results,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'error_summary': error_summary,
                'timestamp': time.time()
            }

    def get_failed_checks(self, hours: int = 1) -> List[Dict[str, Any]]:
        """获取失败的检查"""
        with self._lock:
            since_time = time.time() - hours * 3600
            return [r for r in self._results_history
                    if r.get('timestamp', 0) >= since_time and r.get('status') == 'failed']

    def _cleanup_expired_results(self) -> None:
        """清理过期结果"""
        cutoff_time = datetime.now() - self.retention_period

        # 从历史记录中移除过期项目
        while self._results_history:
            oldest = self._results_history[0]
            if isinstance(oldest.get('timestamp'), (int, float)):
                if datetime.fromtimestamp(oldest['timestamp']) < cutoff_time:
                    self._results_history.popleft()
                else:
                    break
            else:
                self._results_history.popleft()

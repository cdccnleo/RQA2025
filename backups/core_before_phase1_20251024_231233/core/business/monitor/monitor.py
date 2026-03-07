"""
业务流程监控
提供流程执行监控、性能统计和健康检查功能
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict, deque
import psutil

from ..config.enums import BusinessProcessState
from .business_process_models import ProcessInstance

logger = logging.getLogger(__name__)


class ProcessMonitor:

    """流程监控器 - 增强版"""

    def __init__(self, monitor_interval: float = 5.0):
        self.monitor_interval = monitor_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._process_instances: Dict[str, ProcessInstance] = {}
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        # 监控配置
        self._performance_thresholds = {
            'max_duration': 600,  # 10分钟
            'max_memory_mb': 1024,  # 1GB
            'max_cpu_percent': 80.0,
            'error_rate_threshold': 0.1  # 10%
        }

        self._alert_callbacks: List[Callable] = []

    def start_monitoring(self) -> None:
        """启动监控"""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("流程监控已启动")

    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("流程监控已停止")

    def register_process(self, process_instance: ProcessInstance) -> None:
        """注册流程实例进行监控"""
        with self._lock:
            self._process_instances[process_instance.instance_id] = process_instance
            logger.debug(f"注册流程监控: {process_instance.instance_id}")

    def unregister_process(self, process_id: str) -> None:
        """取消注册流程实例"""
        with self._lock:
            if process_id in self._process_instances:
                del self._process_instances[process_id]
                logger.debug(f"取消注册流程监控: {process_id}")

    def add_alert_callback(self, callback: Callable) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                self._perform_monitoring()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")

    def _perform_monitoring(self) -> None:
        """执行监控"""
        current_time = time.time()
        system_metrics = self._collect_system_metrics()

        with self._lock:
            active_processes = list(self._process_instances.values())

        for process in active_processes:
            try:
                self._monitor_single_process(process, current_time, system_metrics)
            except Exception as e:
                logger.error(f"监控流程 {process.instance_id} 异常: {e}")

        # 清理已完成的旧流程
        self._cleanup_completed_processes()

    def _monitor_single_process(self, process: ProcessInstance, current_time: float,
                                system_metrics: Dict[str, Any]) -> None:
        """监控单个流程"""
        process_id = process.instance_id

        # 收集流程指标
        process_metrics = self._collect_process_metrics(process, system_metrics)

        # 检查超时
        if self._check_process_timeout(process, current_time):
            self._trigger_timeout_alert(process)

        # 检查性能阈值
        if self._check_performance_thresholds(process_metrics):
            self._trigger_performance_alert(process, process_metrics)

        # 检查错误率
        if self._check_error_rate(process):
            self._trigger_error_rate_alert(process)

        # 存储性能历史
        self._performance_history[process_id].append({
            'timestamp': current_time,
            'metrics': process_metrics,
            'state': process.current_state
        })

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections())
            }
        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")
            return {}

    def _collect_process_metrics(self, process: ProcessInstance,
                                 system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """收集流程指标"""
        metrics = {
            'process_id': process.instance_id,
            'state': process.current_state.name,
            'duration': process.duration_seconds or 0,
            'error_count': len(process.errors),
            'retry_count': process.retry_count,
            'system_metrics': system_metrics,
            'timestamp': time.time()
        }

        # 计算流程特定指标
        if process.start_time:
            elapsed = (datetime.now() - process.start_time).total_seconds()
            metrics['elapsed_seconds'] = elapsed

            # 预期完成时间估计
            if process.current_state != BusinessProcessState.IDLE:
                progress_ratio = self._estimate_progress_ratio(process)
                metrics['estimated_completion'] = elapsed / max(progress_ratio, 0.01)

        return metrics

    def _estimate_progress_ratio(self, process: ProcessInstance) -> float:
        """估算流程进度比例"""
        state_weights = {
            BusinessProcessState.IDLE: 0.0,
            BusinessProcessState.DATA_COLLECTING: 0.1,
            BusinessProcessState.DATA_QUALITY_CHECKING: 0.2,
            BusinessProcessState.FEATURE_EXTRACTING: 0.4,
            BusinessProcessState.GPU_ACCELERATING: 0.5,
            BusinessProcessState.MODEL_PREDICTING: 0.7,
            BusinessProcessState.MODEL_ENSEMBLING: 0.8,
            BusinessProcessState.SIGNAL_GENERATING: 0.85,
            BusinessProcessState.STRATEGY_DECIDING: 0.9,
            BusinessProcessState.RISK_CHECKING: 0.92,
            BusinessProcessState.ORDER_GENERATING: 0.94,
            BusinessProcessState.ORDER_ROUTING: 0.96,
            BusinessProcessState.EXECUTING: 0.98,
            BusinessProcessState.MONITORING: 0.99,
            BusinessProcessState.COMPLETED: 1.0,
        }

        return state_weights.get(process.current_state, 0.0)

    def _check_process_timeout(self, process: ProcessInstance, current_time: float) -> bool:
        """检查流程是否超时"""
        if not process.start_time:
            return False

        # 根据流程配置检查超时
        timeout_seconds = getattr(process.process_config, 'timeout_seconds', 300)
        elapsed = current_time - process.start_time.timestamp()

        return elapsed > timeout_seconds

    def _check_performance_thresholds(self, metrics: Dict[str, Any]) -> bool:
        """检查性能阈值"""
        memory_mb = metrics.get('system_metrics', {}).get('memory_mb', 0)
        cpu_percent = metrics.get('system_metrics', {}).get('cpu_percent', 0)

        return (memory_mb > self._performance_thresholds['max_memory_mb'] or
                cpu_percent > self._performance_thresholds['max_cpu_percent'])

    def _check_error_rate(self, process: ProcessInstance) -> bool:
        """检查错误率"""
        total_operations = len(process.errors) + 1  # 加1避免除零
        error_rate = len(process.errors) / total_operations

        return error_rate > self._performance_thresholds['error_rate_threshold']

    def _trigger_timeout_alert(self, process: ProcessInstance) -> None:
        """触发超时告警"""
        alert = {
            'type': 'timeout',
            'process_id': process.instance_id,
            'message': f'流程 {process.instance_id} 执行超时',
            'timestamp': datetime.now(),
            'details': {
                'state': process.current_state.name,
                'start_time': process.start_time,
                'timeout_seconds': getattr(process.process_config, 'timeout_seconds', 300)
            }
        }
        self._alerts.append(alert)
        self._notify_alert_callbacks(alert)

    def _trigger_performance_alert(self, process: ProcessInstance, metrics: Dict[str, Any]) -> None:
        """触发性能告警"""
        alert = {
            'type': 'performance',
            'process_id': process.instance_id,
            'message': f'流程 {process.instance_id} 性能异常',
            'timestamp': datetime.now(),
            'details': metrics
        }
        self._alerts.append(alert)
        self._notify_alert_callbacks(alert)

    def _trigger_error_rate_alert(self, process: ProcessInstance) -> None:
        """触发错误率告警"""
        alert = {
            'type': 'error_rate',
            'process_id': process.instance_id,
            'message': f'流程 {process.instance_id} 错误率过高',
            'timestamp': datetime.now(),
            'details': {
                'error_count': len(process.errors),
                'errors': process.errors[-5:]  # 最近5个错误
            }
        }
        self._alerts.append(alert)
        self._notify_alert_callbacks(alert)

    def _notify_alert_callbacks(self, alert: Dict[str, Any]) -> None:
        """通知告警回调"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def _cleanup_completed_processes(self) -> None:
        """清理已完成的旧流程"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24小时前

        with self._lock:
            to_remove = []
            for process_id, process in self._process_instances.items():
                if (process.is_completed() and
                    process.end_time and
                        process.end_time.timestamp() < cutoff_time):
                    to_remove.append(process_id)

            for process_id in to_remove:
                del self._process_instances[process_id]
                logger.debug(f"清理已完成流程: {process_id}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        with self._lock:
            return {
                'active_processes': len(self._process_instances),
                'total_alerts': len(self._alerts),
                'recent_alerts': self._alerts[-10:],  # 最近10个告警
                'performance_history_size': sum(len(history) for history in self._performance_history.values())
            }

    def get_process_performance_history(self, process_id: str, hours: int = 1) -> List[Dict[str, Any]]:
        """获取流程性能历史"""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            history = self._performance_history.get(process_id, deque())
            return [entry for entry in history if entry['timestamp'] >= cutoff_time]


# 别名定义
BusinessProcessMonitor = ProcessMonitor

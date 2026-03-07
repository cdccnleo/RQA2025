"""
performance_monitor_component 模块

提供 performance_monitor_component 相关功能和接口。
"""

import requests

import psutil
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable

# 使用try-except来处理可能缺失的导入
try:
    from ..alert_dataclasses import PerformanceMetrics
except ImportError:
    try:
        from ...models.alert_dataclasses import PerformanceMetrics
    except ImportError:
        PerformanceMetrics = None

try:
    from ..shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
except ImportError:
    # 如果导入失败，创建基本的mock类
    class ILogger:
        pass
    class IErrorHandler:
        pass
    class StandardLogger:
        def __init__(self, name):
            self.name = name
        def log_info(self, msg):
            print(f"INFO: {msg}")
    class BaseErrorHandler:
        def handle_error(self, error, msg):
            print(f"ERROR: {msg} - {error}")
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
"""
性能监控组件

实现系统性能指标的收集和监控功能：
- 实时性能指标收集
- 历史数据存储
- 网络延迟测试
- 多线程安全
"""


class MonitoringPerformanceMonitor:
    """Performance monitor class"""

    def __init__(self, update_interval: int = 5, config: Optional[Dict[str, Any]] = None):
        self.update_interval = update_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics() if PerformanceMetrics else None
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 告警回调列表
        self.alert_callbacks: List[Callable] = []

        # 配置日志和错误处理
        self.logger: ILogger = StandardLogger(f"{self.__class__.__name__}")
        self.error_handler: IErrorHandler = BaseErrorHandler()

        # 应用配置
        if config:
            self.update_interval = config.get('update_interval', self.update_interval)

    def start_monitoring(self):
        """Start monitoring"""
        if self.monitoring:
            return

        try:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.log_info("性能监控已启动")
        except Exception as e:
            self.error_handler.handle_error(e, "启动性能监控失败")
            self.monitoring = False

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.log_info("性能监控已停止")

    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                self.error_handler.handle_error(e, "性能监控循环出错")
                time.sleep(1)  # 短暂延迟后重试

    def _collect_metrics(self):
        """Collect performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            with self._lock:
                if PerformanceMetrics:
                    self.current_metrics = PerformanceMetrics(
                        cpu_usage=cpu_percent,
                        memory_usage=memory.percent,
                        disk_usage=disk.percent,
                        network_latency=self._get_network_latency(),
                        active_threads=threading.active_count(),
                        timestamp=datetime.now()
                    )

                self.metrics_history.append(self.current_metrics)
                
            return self.current_metrics

        except ImportError:
            # psutil不可用时使用模拟数据
            return self._collect_simulated_metrics()

    def _collect_simulated_metrics(self):
        """Collect simulated performance metrics"""
        with self._lock:
            if PerformanceMetrics:
                self.current_metrics = PerformanceMetrics(
                    cpu_usage=time.time() % 100,
                    memory_usage=50 + (time.time() % 30),
                    disk_usage=60 + (time.time() % 20),
                    network_latency=10 + (time.time() % 50),
                    active_threads=threading.active_count(),
                    timestamp=datetime.now()
                )

            self.metrics_history.append(self.current_metrics)
            return self.current_metrics

    def _get_network_latency(self) -> float:
        """Get network latency"""
        try:
            start_time = time.time()
            requests.get("http://www.baidu.com", timeout=5)
            return (time.time() - start_time) * 1000
        except Exception:
            return 100.0  # 默认延迟

    def get_current_metrics(self) -> Optional[Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.current_metrics

    def get_metrics_history(self, minutes: int = 60) -> List[Any]:
        """Get historical performance metrics"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self.metrics_history if hasattr(m, 'timestamp') and m.timestamp > cutoff_time]

    def get_average_metrics(self, minutes: int = 60) -> Optional[Any]:
        """Get average performance metrics over time period"""
        history = self.get_metrics_history(minutes)
        if not history or not PerformanceMetrics:
            return PerformanceMetrics() if PerformanceMetrics else None

        avg_cpu = sum(m.cpu_usage for m in history if hasattr(m, 'cpu_usage')) / len(history)
        avg_memory = sum(m.memory_usage for m in history if hasattr(m, 'memory_usage')) / len(history)
        avg_disk = sum(m.disk_usage for m in history if hasattr(m, 'disk_usage')) / len(history)
        avg_network = sum(m.network_latency for m in history if hasattr(m, 'network_latency')) / len(history)
        avg_threads = sum(m.active_threads for m in history if hasattr(m, 'active_threads')) / len(history)

        return PerformanceMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            disk_usage=avg_disk,
            network_latency=avg_network,
            active_threads=int(avg_threads),
            timestamp=datetime.now()
        )

    def add_alert_callback(self, callback: Callable) -> None:
        """添加告警回调函数"""
        with self._lock:
            if callback is not None and callback not in self.alert_callbacks:
                self.alert_callbacks.append(callback)

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        """触发告警"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # 调用所有注册的回调函数
        with self._lock:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.error_handler.handle_error(e, f"告警回调执行失败: {callback}")

    def _check_smart_alerts(self, metrics) -> None:
        """检查智能告警"""
        if not metrics:
            return
        
        # CPU使用率告警
        if hasattr(metrics, 'cpu_usage') and metrics.cpu_usage > 80.0:
            self._trigger_alert("high_cpu_usage", f"CPU使用率过高: {metrics.cpu_usage}%")
        
        # 内存使用率告警
        if hasattr(metrics, 'memory_usage') and metrics.memory_usage > 80.0:
            self._trigger_alert("high_memory_usage", f"内存使用率过高: {metrics.memory_usage}%")
        
        # 磁盘使用率告警
        if hasattr(metrics, 'disk_usage') and metrics.disk_usage > 90.0:
            self._trigger_alert("high_disk_usage", f"磁盘使用率过高: {metrics.disk_usage}%")
        
        # 网络延迟告警
        if hasattr(metrics, 'network_latency') and metrics.network_latency > 100.0:
            self._trigger_alert("high_network_latency", f"网络延迟过高: {metrics.network_latency}ms")
        
        # 活跃线程数告警
        if hasattr(metrics, 'active_threads') and metrics.active_threads > 100:
            self._trigger_alert("high_thread_count", f"活跃线程数过多: {metrics.active_threads}")

    def add_performance_data(self, metrics) -> None:
        """添加性能数据"""
        if metrics:
            with self._lock:
                self.metrics_history.append(metrics)

    def predict_performance(self, time_window: int = 60) -> Dict[str, Any]:
        """预测性能指标"""
        history = self.get_metrics_history(time_window)
        
        if not history:
            return {
                'predicted_hit_rate': 0.8,
                'predicted_response_time': 100.0,
                'confidence': 0.0
            }
        
        # 简单的线性预测（实际项目中可以使用更复杂的算法）
        recent_data = history[-min(5, len(history)):]  # 取最近5个数据点
        
        if len(recent_data) < 2:
            return {
                'predicted_hit_rate': recent_data[0].test_success_rate if recent_data else 0.8,
                'predicted_response_time': recent_data[0].test_execution_time if recent_data else 100.0,
                'confidence': 0.3
            }
        
        # 计算趋势
        hit_rates = [m.test_success_rate for m in recent_data if hasattr(m, 'test_success_rate')]
        response_times = [m.test_execution_time for m in recent_data if hasattr(m, 'test_execution_time')]
        
        # 简单平均作为预测
        predicted_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.8
        predicted_response_time = sum(response_times) / len(response_times) if response_times else 100.0
        
        # 基于数据点数量的置信度
        confidence = min(0.9, 0.3 + (len(recent_data) - 1) * 0.15)
        
        return {
            'predicted_hit_rate': predicted_hit_rate,
            'predicted_response_time': predicted_response_time,
            'confidence': confidence
        }

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        with self._lock:
            return {
                'total_metrics': len(self.metrics_history),
                'alert_count': len(self.alert_callbacks),  # 使用回调数量作为告警计数的替代
                'monitoring_active': self.monitoring,
                'update_interval': self.update_interval,
                'metrics_history_size': self.metrics_history.maxlen if hasattr(self.metrics_history, 'maxlen') else len(self.metrics_history)
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self._lock:
            current_metrics = self.current_metrics
            history = list(self.metrics_history)
            
            # 生成摘要
            summary = {}
            if current_metrics:
                summary = {
                    'current_cpu_usage': getattr(current_metrics, 'cpu_usage', 0.0),
                    'current_memory_usage': getattr(current_metrics, 'memory_usage', 0.0),
                    'current_disk_usage': getattr(current_metrics, 'disk_usage', 0.0),
                    'current_network_latency': getattr(current_metrics, 'network_latency', 0.0),
                    'active_threads': getattr(current_metrics, 'active_threads', 0)
                }
            
            # 计算趋势
            trends = {}
            if len(history) >= 2:
                recent = history[-min(5, len(history)):]
                
                cpu_values = [getattr(m, 'cpu_usage', 0.0) for m in recent if hasattr(m, 'cpu_usage')]
                memory_values = [getattr(m, 'memory_usage', 0.0) for m in recent if hasattr(m, 'memory_usage')]
                
                trends = {
                    'cpu_trend': 'increasing' if len(cpu_values) > 1 and cpu_values[-1] > cpu_values[0] else 'stable',
                    'memory_trend': 'increasing' if len(memory_values) > 1 and memory_values[-1] > memory_values[0] else 'stable',
                    'data_points': len(recent)
                }
            
            # 生成建议
            recommendations = []
            if current_metrics:
                if getattr(current_metrics, 'cpu_usage', 0.0) > 80.0:
                    recommendations.append("考虑优化CPU密集型操作")
                if getattr(current_metrics, 'memory_usage', 0.0) > 80.0:
                    recommendations.append("考虑内存清理和优化")
                if getattr(current_metrics, 'disk_usage', 0.0) > 90.0:
                    recommendations.append("考虑磁盘空间清理")
                if getattr(current_metrics, 'network_latency', 0.0) > 100.0:
                    recommendations.append("检查网络连接状况")
            
            if not recommendations:
                recommendations.append("系统运行状态良好")
            
            return {
                'summary': summary,
                'trends': trends,
                'recommendations': recommendations
            }

    def _get_hit_rate(self) -> float:
        """获取命中率"""
        if not self.current_metrics:
            return 0.8  # 默认命中率
        return getattr(self.current_metrics, 'test_success_rate', 0.8)

    def _get_response_time(self) -> float:
        """获取响应时间"""
        if not self.current_metrics:
            return 100.0  # 默认响应时间
        return getattr(self.current_metrics, 'test_execution_time', 100.0)

    def _get_throughput(self) -> float:
        """获取吞吐量"""
        if not self.current_metrics:
            return 100.0  # 默认吞吐量
        return getattr(self.current_metrics, 'test_execution_time', 100.0)

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        if not self.current_metrics:
            return 50.0  # 默认内存使用率
        return getattr(self.current_metrics, 'memory_usage', 50.0)

    def _get_eviction_rate(self) -> float:
        """获取驱逐率"""
        # 基于内存使用率计算驱逐率
        memory_usage = self._get_memory_usage()
        return max(0.0, (memory_usage - 80.0) / 20.0) if memory_usage > 80.0 else 0.0

    def _get_miss_penalty(self) -> float:
        """获取未命中惩罚"""
        hit_rate = self._get_hit_rate()
        response_time = self._get_response_time()
        # 未命中惩罚与命中率成反比，与响应时间成正比
        return response_time * (1.0 - hit_rate)

    def add_metric(self, metrics) -> None:
        """添加指标数据 - add_performance_data的别名"""
        self.add_performance_data(metrics)

    def detect_anomaly(self, metrics) -> Dict[str, Any]:
        """检测异常"""
        if not metrics:
            return {
                'severity': 'unknown',
                'description': '无效的指标数据'
            }
        
        severity = 'normal'
        descriptions = []
        
        # 检查CPU使用率异常
        cpu_usage = getattr(metrics, 'cpu_usage', 0.0)
        if cpu_usage > 90.0:
            severity = 'critical'
            descriptions.append(f"CPU使用率异常高: {cpu_usage}%")
        elif cpu_usage > 80.0:
            severity = 'warning' if severity == 'normal' else severity
            descriptions.append(f"CPU使用率偏高: {cpu_usage}%")
        
        # 检查内存使用率异常
        memory_usage = getattr(metrics, 'memory_usage', 0.0)
        if memory_usage > 90.0:
            severity = 'critical'
            descriptions.append(f"内存使用率异常高: {memory_usage}%")
        elif memory_usage > 80.0:
            severity = 'warning' if severity == 'normal' else severity
            descriptions.append(f"内存使用率偏高: {memory_usage}%")
        
        # 检查磁盘使用率异常
        disk_usage = getattr(metrics, 'disk_usage', 0.0)
        if disk_usage > 95.0:
            severity = 'critical'
            descriptions.append(f"磁盘使用率异常高: {disk_usage}%")
        elif disk_usage > 85.0:
            severity = 'warning' if severity == 'normal' else severity
            descriptions.append(f"磁盘使用率偏高: {disk_usage}%")
        
        # 检查网络延迟异常
        network_latency = getattr(metrics, 'network_latency', 0.0)
        if network_latency > 200.0:
            severity = 'critical'
            descriptions.append(f"网络延迟异常高: {network_latency}ms")
        elif network_latency > 100.0:
            severity = 'warning' if severity == 'normal' else severity
            descriptions.append(f"网络延迟偏高: {network_latency}ms")
        
        # 检查测试成功率异常
        test_success_rate = getattr(metrics, 'test_success_rate', 1.0)
        if test_success_rate < 0.3:
            severity = 'critical'
            descriptions.append(f"测试成功率异常低: {test_success_rate:.2%}")
        elif test_success_rate < 0.6:
            severity = 'warning' if severity == 'normal' else severity
            descriptions.append(f"测试成功率偏低: {test_success_rate:.2%}")
        
        description = '; '.join(descriptions) if descriptions else '系统运行正常'
        
        return {
            'severity': severity,
            'description': description
        }

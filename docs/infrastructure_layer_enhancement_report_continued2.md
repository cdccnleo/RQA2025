# RQA2025 基础设施层功能增强分析报告（续2）

## 2. 功能分析（续）

### 2.4 监控系统增强（续）

#### 2.4.1 系统监控（续）

**实现建议**（续）：

```python
    def stop_monitoring(self) -> None:
        """停止系统监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval + 1)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """系统监控循环"""
        while self.monitoring:
            try:
                # 收集系统状态
                status = self._collect_system_status()
                self.monitoring_data.append(status)
                
                # 检查系统状态
                self._check_system_status(status)
                
                # 限制监控数据数量
                if len(self.monitoring_data) > 1000:
                    self.monitoring_data = self.monitoring_data[-1000:]
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def _collect_system_status(self) -> Dict:
        """
        收集系统状态
        
        Returns:
            Dict: 系统状态
        """
        import psutil
        
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 获取网络IO统计
        net_io = psutil.net_io_counters()
        
        # 获取进程数量
        process_count = len(psutil.pids())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'process': {
                'count': process_count
            }
        }
    
    def _check_system_status(self, status: Dict) -> None:
        """
        检查系统状态
        
        Args:
            status: 系统状态
        """
        alerts = []
        
        # 检查CPU使用率
        if status['cpu']['percent'] > 90:
            alert = {
                'level': 'warning',
                'message': f"High CPU usage: {status['cpu']['percent']}%",
                'timestamp': status['timestamp']
            }
            alerts.append(alert)
        
        # 检查内存使用率
        if status['memory']['percent'] > 90:
            alert = {
                'level': 'warning',
                'message': f"High memory usage: {status['memory']['percent']}%",
                'timestamp': status['timestamp']
            }
            alerts.append(alert)
        
        # 检查磁盘使用率
        if status['disk']['percent'] > 90:
            alert = {
                'level': 'warning',
                'message': f"High disk usage: {status['disk']['percent']}%",
                'timestamp': status['timestamp']
            }
            alerts.append(alert)
        
        # 发送告警
        for alert in alerts:
            self._send_alert(alert['level'], alert)
    
    def _send_alert(self, level: str, alert_data: Dict) -> None:
        """
        发送告警
        
        Args:
            level: 告警级别
            alert_data: 告警数据
        """
        # 记录告警日志
        log_method = getattr(logger, level, logger.warning)
        log_method(alert_data['message'])
        
        # 调用告警回调函数
        for callback in self.alert_callbacks:
            try:
                callback(level, alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_monitoring_data(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取监控数据
        
        Args:
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: 监控数据列表
        """
        if not start_time and not end_time:
            return self.monitoring_data
        
        filtered_data = self.monitoring_data
        
        if start_time:
            filtered_data = [d for d in filtered_data if d['timestamp'] >= start_time]
        
        if end_time:
            filtered_data = [d for d in filtered_data if d['timestamp'] <= end_time]
        
        return filtered_data
    
    def get_system_summary(self) -> Dict:
        """
        获取系统摘要
        
        Returns:
            Dict: 系统摘要
        """
        if not self.monitoring_data:
            return {
                'cpu': {'avg': 0, 'max': 0},
                'memory': {'avg': 0, 'max': 0},
                'disk': {'avg': 0, 'max': 0}
            }
        
        # 计算CPU使用率统计
        cpu_values = [d['cpu']['percent'] for d in self.monitoring_data]
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)
        
        # 计算内存使用率统计
        memory_values = [d['memory']['percent'] for d in self.monitoring_data]
        memory_avg = sum(memory_values) / len(memory_values)
        memory_max = max(memory_values)
        
        # 计算磁盘使用率统计
        disk_values = [d['disk']['percent'] for d in self.monitoring_data]
        disk_avg = sum(disk_values) / len(disk_values)
        disk_max = max(disk_values)
        
        return {
            'cpu': {
                'avg': cpu_avg,
                'max': cpu_max
            },
            'memory': {
                'avg': memory_avg,
                'max': memory_max
            },
            'disk': {
                'avg': disk_avg,
                'max': disk_max
            }
        }
```

#### 2.4.2 应用监控

**现状分析**：
缺乏对应用运行状态的监控，无法及时发现应用异常。

**实现建议**：
实现一个 `ApplicationMonitor` 类，提供应用监控功能：

```python
import time
import threading
from typing import Dict, List, Optional, Callable, Any
import logging
from datetime import datetime
import traceback
import functools

logger = logging.getLogger(__name__)

class ApplicationMonitor:
    """应用监控器"""
    
    def __init__(
        self,
        app_name: str = 'rqa2025',
        alert_callbacks: Optional[List[Callable[[str, Dict], None]]] = None
    ):
        """
        初始化应用监控器
        
        Args:
            app_name: 应用名称
            alert_callbacks: 告警回调函数列表
        """
        self.app_name = app_name
        self.alert_callbacks = alert_callbacks or []
        
        # 应用指标
        self.metrics: Dict[str, List[Dict]] = {
            'function_calls': [],
            'errors': [],
            'custom_metrics': []
        }
    
    def monitor_function(self, name: Optional[str] = None) -> Callable:
        """
        监控函数装饰器
        
        Args:
            name: 函数名称，默认为函数的__name__
            
        Returns:
            Callable: 装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    self.record_error(
                        func_name,
                        str(e),
                        traceback.format_exc()
                    )
                    raise e
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self.record_function_call(
                        func_name,
                        execution_time,
                        success
                    )
                
                return result
            
            return wrapper
        
        return decorator
    
    def record_function_call(
        self,
        name: str,
        execution_time: float,
        success: bool
    ) -> None:
        """
        记录函数调用
        
        Args:
            name: 函数名称
            execution_time: 执行时间（秒）
            success: 是否成功
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'execution_time': execution_time,
            'success': success
        }
        
        self.metrics['function_calls'].append(metric)
        
        # 限制指标数量
        if len(self.metrics['function_calls']) > 1000:
            self.metrics['function_calls'] = self.metrics['function_calls'][-1000:]
        
        # 检查执行时间是否过长
        if execution_time > 10.0:  # 10秒
            self._send_alert(
                'warning',
                {
                    'level': 'warning',
                    'message': f"Function {name} execution time too long: {execution_time:.2f}s",
                    'timestamp': metric['timestamp'],
                    'details': metric
                }
            )
    
    def record_error(
        self,
        source: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> None:
        """
        记录错误
        
        Args:
            source: 错误来源
            error_message: 错误消息
            stack_trace: 堆栈跟踪
        """
        error = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'message': error_message,
            'stack_trace': stack_trace
        }
        
        self.metrics['errors'].append(error)
        
        # 限制指标数量
        if len(self.metrics['errors']) > 1000:
            self.metrics['errors'] = self.metrics['errors'][-1000:]
        
        # 发送告警
        self._send_alert(
            'error',
            {
                'level': 'error',
                'message': f"Error in {source}: {error_message}",
                'timestamp': error['timestamp'],
                'details': error
            }
        )
    
    def record_custom_metric(
        self,
        name: str,
        value: Any,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录自定义指标
        
        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'tags': tags or {}
        }
        
        self.metrics['custom_metrics'].append(metric)
        
        # 限制指标数量
        if len(self.metrics['custom_metrics']) > 1000:
            self.metrics['custom_metrics'] = self.metrics['custom_metrics'][-1000:]
    
    def _send_alert(self, level: str, alert_data: Dict) -> None:
        """
        发送告警
        
        Args:
            level: 告警级别
            alert_data: 告警数据
        """
        # 记录告警日志
        log_method = getattr(logger, level, logger.warning)
        log_method(alert_data['message'])
        
        # 调用告警回调函数
        for callback in self.alert_callbacks:
            try:
                callback(level, alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_function_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取函数指标
        
        Args:
            name: 函数名称
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: 函数指标列表
        """
        metrics = self.metrics['function_calls']
        
        if name:
            metrics = [m for m in metrics if m['name'] == name]
        
        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]
        
        return metrics
    
    def get_error_metrics(
        self,
        source: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取错误指标
        
        Args:
            source: 错误来源
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: 错误指标列表
        """
        metrics = self.metrics['errors']
        
        if source:
            metrics = [m for m in metrics if m['source'] == source]
        
        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]
        
        return metrics
    
    def get_custom_metrics(
        self
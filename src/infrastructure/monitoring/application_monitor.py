import threading
import time
import functools
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class ApplicationMonitor:
    """应用性能监控器"""

    def __init__(
        self,
        app_name: str = "rqa2025",
        alert_handlers: Optional[List[Callable[[str, Dict], None]]] = None,
        influx_config: Optional[Dict] = None,
        sample_rate: float = 1.0,
        retention_policy: str = "30d",
        influx_client_mock: Any = None,
        skip_thread: bool = False
    ):
        """
        初始化应用监控器

        Args:
            app_name: 应用名称
            alert_handlers: 告警处理器列表
            influx_config: InfluxDB配置字典
            sample_rate: 采样率
            retention_policy: 保留策略
            influx_client_mock: 测试用mock的influx_client
            skip_thread: 测试时跳过后台线程
        """
        self.app_name = app_name
        self.alert_handlers = alert_handlers or []

        # 监控数据存储
        self._metrics: Dict[str, List[Dict]] = {
            'functions': [],  # 函数执行指标
            'errors': [],     # 错误记录
            'custom': []      # 自定义指标
        }
        
        # 查询缓存
        self._cache: Dict[str, Dict] = {
            'functions': {},
            'errors': {},
            'custom': {}
        }

        # 监控配置
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.retention_policy = retention_policy
        self._last_compaction = time.time()
        self._last_aggregation = time.time()
        
        # 添加测试期望的属性
        self._error_counter = None
        self._client = None
        self._default_tags = {}
        self._health_checks = {}
        self._alert_rules = {}
        
        # 启动后台压缩线程（可跳过）
        self._compaction_thread = None
        if not skip_thread:
            self._compaction_thread = threading.Thread(
                target=self._auto_compact,
                daemon=True
            )
            self._compaction_thread.start()
        
        # 初始化InfluxDB客户端（可注入mock）
        self.influx_client = None
        if influx_client_mock is not None:
            self.influx_client = influx_client_mock
            self.influx_bucket = influx_config['bucket'] if influx_config and 'bucket' in influx_config else None
        elif influx_config:
            influx_config['retention_policy'] = retention_policy
            try:
                from influxdb_client.client.influxdb_client import InfluxDBClient
                self.influx_client = InfluxDBClient(
                    url=influx_config['url'],
                    token=influx_config['token'],
                    org=influx_config['org']
                )
                self.influx_bucket = influx_config['bucket']
                logger.info(f"InfluxDB connected to {influx_config['url']}")
            except ImportError:
                logger.warning("influxdb_client not installed, metrics persistence disabled")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB client: {e}")

    def monitor(self, name: Optional[str] = None, slow_threshold: float = 5.0):
        """
        函数监控装饰器

        Args:
            name: 监控名称(默认使用函数名)
            slow_threshold: 慢执行阈值(秒)
        """
        def decorator(func):
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
                        source=func_name,
                        message=str(e),
                        stack_trace=traceback.format_exc(),
                        context=None
                    )
                    raise
                finally:
                    # 记录执行指标
                    execution_time = time.time() - start_time
                    self.record_function(
                        name=func_name,
                        execution_time=execution_time,
                        success=success
                    )

                    # 慢执行告警
                    if execution_time > slow_threshold:
                        self._trigger_alert(
                            'performance',
                            {
                                'level': 'warning',
                                'message': f"Slow execution: {func_name} took {execution_time:.2f}s",
                                'value': execution_time,
                                'threshold': slow_threshold,
                                'timestamp': datetime.now().isoformat()
                            }
                        )

                return result
            return wrapper
        return decorator

    def record_function(
        self,
        name: str,
        execution_time: float,
        success: bool = True
    ) -> None:
        """
        记录函数执行指标

        Args:
            name: 函数名称
            execution_time: 执行时间(秒)
            success: 是否成功
        """
        timestamp = datetime.now().isoformat()
        metric = {
            'timestamp': timestamp,
            'name': name,
            'execution_time': execution_time,
            'success': success
        }
        self._metrics['functions'].append(metric)

        # 写入InfluxDB
        if self.influx_client and self.influx_bucket:
            try:
                from influxdb_client.client.write_api import SYNCHRONOUS
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

                point = {
                    "measurement": "function_metrics",
                    "tags": {
                        "app": self.app_name,
                        "function": name
                    },
                    "fields": {
                        "execution_time": execution_time,
                        "success": success
                    },
                    "time": timestamp
                }
                write_api.write(bucket=self.influx_bucket, record=point)
            except Exception as e:
                logger.error(f"Failed to write function metric to InfluxDB: {e}")

        # 限制数据量
        if len(self._metrics['functions']) > 10000:
            self._metrics['functions'] = self._metrics['functions'][-10000:]

    def record_error(self, source: str, message: str, stack_trace: Optional[str] = None, context: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None) -> None:
        """
        记录错误
        
        Args:
            source: 错误来源
            message: 错误消息
            stack_trace: 堆栈跟踪(可选)
            context: 上下文信息(可选)
            error: 异常对象(可选)
        """
        timestamp = datetime.now().isoformat()
        error_record = {
            'timestamp': timestamp,
            'source': source,
            'message': message,
            'stack_trace': stack_trace,
            'context': context or {}
        }
        
        # 如果有异常对象，提取额外信息
        if error:
            error_record['error_type'] = type(error).__name__
            error_record['error_message'] = str(error)
        
        self._metrics['errors'].append(error_record)

        # 写入InfluxDB
        if self.influx_client and self.influx_bucket:
            try:
                from influxdb_client.client.write_api import SYNCHRONOUS
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

                point = {
                    "measurement": "error_metrics",
                    "tags": {
                        "app": self.app_name,
                        "source": source,
                        "error_type": error_record.get('error_type', 'Unknown')
                    },
                    "fields": {
                        "message": message,
                        "has_stack_trace": bool(stack_trace)
                    },
                    "time": timestamp
                }
                write_api.write(bucket=self.influx_bucket, record=point)
            except Exception as e:
                logger.error(f"Failed to write error metric to InfluxDB: {e}")

        # 限制数据量
        if len(self._metrics['errors']) > 10000:
            self._metrics['errors'] = self._metrics['errors'][-10000:]

        # 触发告警
        self._trigger_alert(
            'error',
            {
                'level': 'error',
                'message': f"Error in {source}: {message}",
                'source': source,
                'timestamp': timestamp
            }
        )

    def record_metric(
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
            tags: 标签字典
        """
        timestamp = datetime.now().isoformat()
        metric = {
            'timestamp': timestamp,
            'name': name,
            'value': value,
            'tags': tags or {}
        }
        self._metrics['custom'].append(metric)

        # 写入InfluxDB
        if self.influx_client and self.influx_bucket:
            try:
                from influxdb_client.client.write_api import SYNCHRONOUS
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

                point = {
                    "measurement": name,
                    "tags": tags or {},
                    "fields": {
                        "value": value
                    },
                    "time": timestamp
                }
                write_api.write(bucket=self.influx_bucket, record=point)
            except Exception as e:
                logger.error(f"Failed to write custom metric to InfluxDB: {e}")

        # 限制数据量
        if len(self._metrics['custom']) > 10000:
            self._metrics['custom'] = self._metrics['custom'][-10000:]

    def add_health_check(self, name: str, check_func: Callable) -> None:
        """添加健康检查"""
        self._health_checks[name] = check_func

    def run_health_checks(self) -> Dict[str, bool]:
        """运行所有健康检查"""
        results = {}
        for name, check_func in self._health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = False
        return results

    def add_alert_rule(self, name: str, condition: Callable, handler: Callable) -> None:
        """添加告警规则"""
        self._alert_rules[name] = {
            'condition': condition,
            'handler': handler
        }

    def set_default_tags(self, tags: Dict[str, str]) -> None:
        """设置默认标签"""
        self._default_tags.update(tags)

    def _trigger_alert(self, alert_type: str, alert: Dict) -> None:
        """触发告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert_type, alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_function_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取函数执行指标

        Args:
            name: 函数名称
            start_time: 开始时间(ISO格式)
            end_time: 结束时间(ISO格式)

        Returns:
            List[Dict]: 函数指标列表
        """
        metrics = self._metrics['functions']

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
        level: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取错误指标

        Args:
            source: 错误来源
            level: 错误级别
            start_time: 开始时间(ISO格式)
            end_time: 结束时间(ISO格式)

        Returns:
            List[Dict]: 错误指标列表
        """
        errors = self._metrics['errors']

        if source:
            errors = [e for e in errors if e['source'] == source]

        if level:
            errors = [e for e in errors if e['level'] == level]

        if start_time:
            errors = [e for e in errors if e['timestamp'] >= start_time]

        if end_time:
            errors = [e for e in errors if e['timestamp'] <= end_time]

        return errors

    def get_custom_metrics(
        self,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取自定义指标

        Args:
            name: 指标名称
            tags: 标签过滤
            start_time: 开始时间(ISO格式)
            end_time: 结束时间(ISO格式)

        Returns:
            List[Dict]: 自定义指标列表
        """
        metrics = self._metrics['custom']

        if name:
            metrics = [m for m in metrics if m['name'] == name]

        if tags:
            metrics = [
                m for m in metrics if all(
                    m['tags'].get(k) == v for k, v in tags.items()
                )
            ]

        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]

        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]

        return metrics

    def get_function_summary(
        self,
        name: Optional[str] = None
    ) -> Dict:
        """
        获取函数执行摘要

        Args:
            name: 函数名称

        Returns:
            Dict: 函数执行摘要
        """
        metrics = self.get_function_metrics(name=name)

        if not metrics:
            return {
                'count': 0,
                'success_rate': 0,
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0
            }

        total = len(metrics)
        success = sum(1 for m in metrics if m['success'])
        times = [m['execution_time'] for m in metrics]

        return {
            'count': total,
            'success_rate': success / total if total > 0 else 0,
            'avg_time': sum(times) / len(times) if times else 0,
            'min_time': min(times) if times else 0,
            'max_time': max(times) if times else 0
        }

    def get_error_summary(self) -> Dict:
        """
        获取错误摘要

        Returns:
            Dict: 错误摘要
        """
        errors = self._metrics['errors']

        if not errors:
            return {
                'total': 0,
                'levels': {},
                'sources': {}
            }

        # 按级别统计
        levels = {}
        for e in errors:
            if e['level'] not in levels:
                levels[e['level']] = 0
            levels[e['level']] += 1

        # 按来源统计
        sources = {}
        for e in errors:
            if e['source'] not in sources:
                sources[e['source']] = 0
            sources[e['source']] += 1

        return {
            'total': len(errors),
            'levels': levels,
            'sources': sources
        }

    def close(self):
        """关闭监控器并释放资源"""
        if self.influx_client:
            try:
                self.influx_client.close()
                logger.info("InfluxDB connection closed")
            except Exception as e:
                logger.error(f"Failed to close InfluxDB client: {e}")



    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def _auto_compact(self):
        """自动压缩监控数据"""
        while True:
            time.sleep(60)  # 每分钟检查一次
            now = time.time()
            
            # 压缩函数指标
            if len(self._metrics['functions']) > 1000:
                self._metrics['functions'] = self._metrics['functions'][-1000:]
            
            # 压缩错误记录
            if len(self._metrics['errors']) > 1000:
                self._metrics['errors'] = self._metrics['errors'][-1000:]
            
            # 压缩自定义指标
            if len(self._metrics['custom']) > 1000:
                self._metrics['custom'] = self._metrics['custom'][-1000:]
            
            self._last_compaction = now

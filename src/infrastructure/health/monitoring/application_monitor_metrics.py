"""
application_monitor_metrics 模块

提供 application_monitor_metrics 相关功能和接口。
"""

import logging

import platform
import psutil
import time

from datetime import datetime
from src.infrastructure.health.core.interfaces import IUnifiedInfrastructureInterface
from typing import Dict, List, Any, Optional
"""
基础设施层 - 应用监控指标组件

application_monitor_metrics 模块

应用监控器的指标管理功能实现，包含数据查询、指标记录等功能。
"""

logger = logging.getLogger(__name__)


class ApplicationMonitorMetricsMixin(IUnifiedInfrastructureInterface):
    """
    应用监控指标混入类

    提供指标记录、数据查询等指标管理功能。
    应与ApplicationMonitorCoreMixin一起使用。
    """

    def record_metric(self,
                      name: str,
                      value: Any,
                      tags: Optional[Dict[str, str]] = None,
                      timestamp: Optional[datetime] = None) -> None:
        """
        记录自定义指标

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签字典
            timestamp: 时间戳(默认当前时间)
        """
        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            'name': name,
            'value': value,
            'tags': tags or {},
            'timestamp': timestamp.isoformat()
        }

        # 合并默认标签
        if hasattr(self, '_default_tags'):
            metric['tags'].update(self._default_tags)

        # 添加到存储
        self._metrics['custom'].append(metric)

        # 限制数据量
        if len(self._metrics['custom']) > 5000:  # 自定义指标限制更小
            self._metrics['custom'] = self._metrics['custom'][-5000:]

        # 写入InfluxDB
        self._write_custom_metric_to_influxdb(metric)

    def _write_custom_metric_to_influxdb(self, metric: Dict[str, Any]):
        """写入自定义指标到InfluxDB"""
        if not (hasattr(self, 'influx_client') and self.influx_client
                and hasattr(self, 'influx_bucket') and self.influx_bucket
                and hasattr(self, 'SYNCHRONOUS') and self.SYNCHRONOUS is not None):
            return

        try:
            write_api = self.influx_client.write_api(write_options=self.SYNCHRONOUS)

            # 构建标签
            influx_tags = {
                "app": getattr(self, 'app_name', 'unknown'),
                "metric_name": metric['name']
            }
            influx_tags.update(metric.get('tags', {}))

            point = {
                "measurement": "custom_metrics",
                "tags": influx_tags,
                "fields": {
                    "value": metric['value']
                },
                "time": metric['timestamp']
            }

            write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.error(f"Failed to write custom metric to InfluxDB: {e}")

    def record_prometheus_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        记录Prometheus指标

        Args:
            name: 指标名称
            value: 指标值
            labels: 标签字典
        """
        # 这里可以扩展为支持更多类型的Prometheus指标
        # 目前主要用于记录Gauge类型的指标

        try:
            # 如果有自定义Prometheus指标注册，可以在这里处理
            # 这里只是示例，实际实现可能需要根据具体需求定制

            metric_data = {
                'name': name,
                'value': value,
                'labels': labels or {},
                'timestamp': datetime.now().isoformat()
            }

            # 可以选择存储到本地或直接上报到Prometheus Pushgateway
            logger.debug(f"Recording Prometheus metric: {metric_data}")

        except Exception as e:
            logger.error(f"Failed to record Prometheus metric: {e}")

    def get_function_metrics(self, name: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 1000) -> List[Dict[str, Any]]:
        """
        获取函数执行指标

        Args:
            name: 函数名称过滤器
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制

        Returns:
            函数执行指标列表
        """
        metrics = self._metrics['functions']

        # 应用过滤器
        if name:
            metrics = [m for m in metrics if m.get('name') == name]

        if start_time:
            start_iso = start_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') >= start_iso]

        if end_time:
            end_iso = end_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') <= end_iso]

        # 按时间排序（最新的在前）
        metrics = sorted(metrics, key=lambda x: x.get('timestamp', ''), reverse=True)

        return metrics[:limit]

    def get_error_metrics(self, source: Optional[str] = None,
                          error_type: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """
        获取错误指标

        Args:
            source: 错误来源过滤器
            error_type: 错误类型过滤器
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制

        Returns:
            错误指标列表
        """
        metrics = self._metrics['errors']

        # 应用过滤器
        if source:
            metrics = [m for m in metrics if m.get('source') == source]

        if error_type:
            metrics = [m for m in metrics if m.get('error_type') == error_type]

        if start_time:
            start_iso = start_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') >= start_iso]

        if end_time:
            end_iso = end_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') <= end_iso]

        # 按时间排序（最新的在前）
        metrics = sorted(metrics, key=lambda x: x.get('timestamp', ''), reverse=True)

        return metrics[:limit]

    def get_custom_metrics(self, name: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """
        获取自定义指标

        Args:
            name: 指标名称过滤器
            tags: 标签过滤器
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回记录数量限制

        Returns:
            自定义指标列表
        """
        metrics = self._metrics['custom']

        # 应用过滤器
        if name:
            metrics = [m for m in metrics if m.get('name') == name]

        if tags:
            def matches_tags(metric_tags):
                metric_tags = metric_tags or {}
                return all(metric_tags.get(k) == v for k, v in tags.items())

            metrics = [m for m in metrics if matches_tags(m.get('tags'))]

        if start_time:
            start_iso = start_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') >= start_iso]

        if end_time:
            end_iso = end_time.isoformat()
            metrics = [m for m in metrics if m.get('timestamp', '') <= end_iso]

        # 按时间排序（最新的在前）
        metrics = sorted(metrics, key=lambda x: x.get('timestamp', ''), reverse=True)

        return metrics[:limit]

    def get_function_summary(self, start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取函数执行摘要

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            函数执行统计摘要
        """
        metrics = self.get_function_metrics(start_time=start_time, end_time=end_time, limit=10000)

        if not metrics:
            return {
                'total_calls': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'functions': {}
            }

        # 基础统计
        total_calls = len(metrics)
        successful_calls = sum(1 for m in metrics if m.get('success', False))
        success_rate = successful_calls / total_calls if total_calls > 0 else 0

        # 执行时间统计
        execution_times = [m.get('execution_time', 0)
                           for m in metrics if m.get('execution_time', 0) > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        # 按函数分组统计
        function_stats = {}
        for metric in metrics:
            func_name = metric.get('name', 'unknown')
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'calls': 0,
                    'successes': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0
                }

            stats = function_stats[func_name]
            stats['calls'] += 1
            if metric.get('success', False):
                stats['successes'] += 1
            stats['total_time'] += metric.get('execution_time', 0)

        # 计算平均时间
        for stats in function_stats.values():
            if stats['calls'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['calls']

        return {
            'total_calls': total_calls,
            'success_rate': round(success_rate, 3),
            'avg_execution_time': round(avg_execution_time, 3),
            'functions': function_stats
        }

    def get_error_summary(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取错误摘要

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            错误统计摘要
        """
        metrics = self.get_error_metrics(start_time=start_time, end_time=end_time, limit=10000)

        if not metrics:
            return {
                'total_errors': 0,
                'error_types': {},
                'error_sources': {}
            }

        # 按错误类型统计
        error_types = {}
        error_sources = {}

        for metric in metrics:
            # 错误类型统计
            err_type = metric.get('error_type', 'Unknown')
            error_types[err_type] = error_types.get(err_type, 0) + 1

            # 错误来源统计
            source = metric.get('source', 'Unknown')
            error_sources[source] = error_sources.get(source, 0) + 1

        return {
            'total_errors': len(metrics),
            'error_types': error_types,
            'error_sources': error_sources
        }

    def get_metrics(self, include_system: bool = True) -> Dict[str, Any]:
        """
        获取完整的指标数据

        Args:
            include_system: 是否包含系统指标

        Returns:
            完整的指标数据字典
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'functions': self.get_function_summary(),
            'errors': self.get_error_summary(),
            'custom_metrics_count': len(self._metrics['custom'])
        }

        if include_system and hasattr(self, '_get_system_metrics'):
            try:
                result['system'] = self._get_system_metrics()
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
                result['system'] = {}

        return result

    def _get_system_metrics(self) -> Dict[str, Any]:
        """
        获取系统指标（需要具体实现）

        Returns:
            系统指标字典
        """
        # 这是一个占位符方法，实际实现可能需要根据具体需求来获取系统指标
        # 例如：CPU使用率、内存使用率、磁盘空间等

        try:
            return {
                'platform': platform.platform(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'uptime': time.time()  # 可以计算实际运行时间
            }
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {
                'platform': platform.platform(),
                'error': str(e)
            }

    # =========================================================================
    # 统一基础设施接口实现
    # =========================================================================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化组件"""
        try:
            if config:
                # 更新配置
                if hasattr(self, 'config'):
                    self.config.update(config)
                else:
                    self.config = config

            logger.info("ApplicationMonitorMetricsMixin 初始化成功")
            return True
        except Exception as e:
            logger.error(f"ApplicationMonitorMetricsMixin 初始化失败: {e}")
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "component_type": "ApplicationMonitorMetricsMixin",
            "version": "1.0.0",
            "capabilities": ["metrics_collection", "performance_monitoring", "data_analysis"],
            "status": "active"
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态"""
        try:
            # 检查基本属性是否存在
            if not hasattr(self, 'metrics_data'):
                return False

            # 检查指标数据是否正常
            if hasattr(self, 'metrics_data') and len(self.metrics_data) > 10000:  # 防止内存溢出
                return False

            return True
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标"""
        try:
            metrics_count = len(self.metrics_data) if hasattr(self, 'metrics_data') else 0
            functions_count = len(self.function_metrics) if hasattr(self, 'function_metrics') else 0

            return {
                "metrics_count": metrics_count,
                "functions_count": functions_count,
                "memory_usage": metrics_count * 100,  # 估算内存使用
                "data_retention_days": 7,  # 默认保留7天
                "collection_interval": 60  # 默认60秒间隔
            }
        except Exception as e:
            logger.error(f"获取指标失败: {e}")
            return {"error": str(e)}

    def cleanup(self) -> bool:
        """清理组件资源"""
        try:
            # 清理指标数据
            if hasattr(self, 'metrics_data'):
                self.metrics_data.clear()

            if hasattr(self, 'function_metrics'):
                self.function_metrics.clear()

            logger.info("ApplicationMonitorMetricsMixin 资源清理完成")
            return True
        except Exception as e:
            logger.error(f"ApplicationMonitorMetricsMixin 资源清理失败: {e}")
            return False

    # ============================================================================
    # 标准化健康检查方法
    # ============================================================================

    def check_health(self) -> Dict[str, Any]:
        """检查应用监控指标组件整体健康状态

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始执行应用监控指标健康检查")

            # 检查指标数据状态
            data_status = self.check_metrics_data_health()

            # 检查配置状态
            config_status = self.check_metrics_config_health()

            # 检查性能状态
            performance_status = self.check_metrics_performance_health()

            # 综合判断整体健康状态
            overall_healthy = all([
                data_status.get('healthy', False),
                config_status.get('healthy', False),
                performance_status.get('healthy', False)
            ])

            result = {
                'healthy': overall_healthy,
                'timestamp': datetime.now().isoformat(),
                'component': 'application_monitor_metrics',
                'details': {
                    'data_status': data_status,
                    'config_status': config_status,
                    'performance_status': performance_status
                }
            }

            logger.info(f"应用监控指标健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"应用监控指标健康检查失败: {e}")
            return {
                'healthy': False,
                'timestamp': datetime.now().isoformat(),
                'component': 'application_monitor_metrics',
                'error': str(e)
            }

    def check_metrics_data_health(self) -> Dict[str, Any]:
        """检查指标数据健康状态

        Returns:
            Dict[str, Any]: 指标数据健康检查结果
        """
        try:
            issues = []

            # 检查指标数据存储
            if hasattr(self, 'metrics_data'):
                if not isinstance(self.metrics_data, (list, dict)):
                    issues.append("指标数据存储类型异常")
                else:
                    data_count = len(self.metrics_data) if hasattr(
                        self.metrics_data, '__len__') else 0
                    if data_count > 50000:  # 防止内存溢出
                        issues.append(f"指标数据量过大: {data_count}条记录")
            else:
                issues.append("缺少指标数据存储")

            # 检查函数指标数据
            if hasattr(self, 'function_metrics'):
                if not isinstance(self.function_metrics, dict):
                    issues.append("函数指标数据类型异常")
                else:
                    func_count = len(self.function_metrics)
                    if func_count > 1000:  # 防止过度记录
                        issues.append(f"函数指标数量过多: {func_count}个函数")
            else:
                issues.append("缺少函数指标存储")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'metrics_count': len(self.metrics_data) if hasattr(self, 'metrics_data') and hasattr(self.metrics_data, '__len__') else 0,
                'functions_count': len(self.function_metrics) if hasattr(self, 'function_metrics') else 0
            }

        except Exception as e:
            logger.error(f"指标数据健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def check_metrics_config_health(self) -> Dict[str, Any]:
        """检查指标配置健康状态

        Returns:
            Dict[str, Any]: 指标配置健康检查结果
        """
        try:
            issues = []

            # 检查配置对象
            if hasattr(self, 'config'):
                if not isinstance(self.config, dict):
                    issues.append("配置对象类型异常")
            else:
                issues.append("缺少配置对象")

            # 检查必要的配置项
            required_configs = ['collection_interval', 'retention_days', 'max_metrics_count']
            for config_key in required_configs:
                if hasattr(self, 'config') and isinstance(self.config, dict):
                    if config_key not in self.config:
                        issues.append(f"缺少必要配置项: {config_key}")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'config_keys': list(self.config.keys()) if hasattr(self, 'config') and isinstance(self.config, dict) else []
            }

        except Exception as e:
            logger.error(f"指标配置健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def check_metrics_performance_health(self) -> Dict[str, Any]:
        """检查指标性能健康状态

        Returns:
            Dict[str, Any]: 指标性能健康检查结果
        """
        try:
            issues = []
            warnings = []

            # 检查记录性能
            start_time = datetime.now()
            test_metric = {'name': 'health_check_test', 'value': 1,
                           'timestamp': datetime.now().isoformat()}
            if hasattr(self, 'record_metric'):
                try:
                    self.record_metric('health_check_test', 1)
                except Exception as e:
                    issues.append(f"指标记录功能异常: {str(e)}")

            record_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒
            if record_time > 100:  # 记录耗时超过100ms
                warnings.append(f"指标记录性能较慢: {record_time:.2f}ms")

            # 检查查询性能
            start_time = datetime.now()
            if hasattr(self, 'get_function_summary'):
                try:
                    summary = self.get_function_summary()
                except Exception as e:
                    issues.append(f"指标查询功能异常: {str(e)}")

            query_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒
            if query_time > 500:  # 查询耗时超过500ms
                warnings.append(f"指标查询性能较慢: {query_time:.2f}ms")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'record_time_ms': record_time,
                'query_time_ms': query_time
            }

        except Exception as e:
            logger.error(f"指标性能健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查（别名方法）

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        return self.check_health()

    def monitor_metrics_status(self) -> Dict[str, Any]:
        """监控指标系统状态

        Returns:
            Dict[str, Any]: 指标系统状态信息
        """
        try:
            return {
                'component': 'application_monitor_metrics',
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'data_stats': {
                    'metrics_count': len(self.metrics_data) if hasattr(self, 'metrics_data') and hasattr(self.metrics_data, '__len__') else 0,
                    'functions_count': len(self.function_metrics) if hasattr(self, 'function_metrics') else 0,
                    'custom_count': len(self.custom_metrics) if hasattr(self, 'custom_metrics') else 0
                },
                'performance': {
                    'last_collection_time': getattr(self, '_last_collection_time', None),
                    'collection_interval': getattr(self, 'config', {}).get('collection_interval', 60)
                },
                'health': self.check_health()
            }
        except Exception as e:
            logger.error(f"获取指标系统状态失败: {e}")
            return {
                'component': 'application_monitor_metrics',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def validate_metrics_config(self) -> Dict[str, Any]:
        """验证指标配置有效性

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            config_status = self.check_metrics_config_health()

            # 额外的配置验证逻辑
            additional_issues = []

            if hasattr(self, 'config') and isinstance(self.config, dict):
                # 验证收集间隔
                interval = self.config.get('collection_interval', 60)
                if not isinstance(interval, (int, float)) or interval <= 0:
                    additional_issues.append("收集间隔配置无效")

                # 验证保留天数
                retention = self.config.get('retention_days', 7)
                if not isinstance(retention, (int, float)) or retention <= 0:
                    additional_issues.append("保留天数配置无效")

                # 验证最大指标数量
                max_count = self.config.get('max_metrics_count', 10000)
                if not isinstance(max_count, int) or max_count <= 0:
                    additional_issues.append("最大指标数量配置无效")

            config_status['issues'].extend(additional_issues)
            config_status['healthy'] = len(config_status['issues']) == 0

            return config_status

        except Exception as e:
            logger.error(f"指标配置验证失败: {e}")
            return {
                'healthy': False,
                'issues': [f"验证过程异常: {str(e)}"]
            }

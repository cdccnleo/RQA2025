#!/usr/bin/env python3
"""
RQA2025 基础设施层指标导出器

负责将监控指标导出为各种格式，支持Prometheus、JSON等多种格式。
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .performance_monitor import monitor_performance

from ..core.parameter_objects import PrometheusExportConfig


class MetricsExporter:
    """
    指标导出器

    支持多种格式的指标导出，包括Prometheus格式、JSON格式等。
    """

    def __init__(
        self,
        pool_name: str = "default_pool",
        config: Optional[PrometheusExportConfig] = None,
    ):
        """
        初始化指标导出器

        Args:
            pool_name: 池名称
            config: 导出配置
        """
        self.pool_name = pool_name
        self.config = config or PrometheusExportConfig()

        # 导出缓存
        self._last_export_time: Optional[datetime] = None
        self._export_cache: Dict[str, str] = {}

    @monitor_performance("MetricsExporter", "export_metrics")
    def export_metrics(self, stats: Dict[str, Any]) -> bool:
        """
        导出指标

        Args:
            stats: 统计信息

        Returns:
            bool: 是否成功导出
        """
        try:
            # Prometheus格式
            if self.config.enable_compression:
                prometheus_data = self._generate_prometheus_compressed(stats)
            else:
                prometheus_data = self._generate_prometheus_format(stats)

            self._export_cache['prometheus'] = prometheus_data

            # JSON格式
            json_data = self._generate_json_format(stats)
            self._export_cache['json'] = json_data

            # 更新导出时间
            self._last_export_time = datetime.now()

            return True

        except Exception as e:
            print(f"导出指标失败: {e}")
            return False

    def get_prometheus_metrics(self) -> str:
        """
        获取Prometheus格式的指标

        Returns:
            str: Prometheus格式的指标字符串
        """
        return self._export_cache.get('prometheus', '')

    def get_json_metrics(self) -> str:
        """
        获取JSON格式的指标

        Returns:
            str: JSON格式的指标字符串
        """
        return self._export_cache.get('json', '{}')

    def get_export_status(self) -> Dict[str, Any]:
        """
        获取导出状态

        Returns:
            Dict[str, Any]: 导出状态信息
        """
        return {
            'last_export_time': self._last_export_time.isoformat() if self._last_export_time else None,
            'available_formats': list(self._export_cache.keys()),
            'cache_size': len(self._export_cache),
            'pool_name': self.pool_name
        }

    def clear_cache(self):
        """清空导出缓存"""
        self._export_cache.clear()
        self._last_export_time = None

    def _generate_prometheus_format(self, stats: Dict[str, Any]) -> str:
        """
        生成Prometheus格式

        Args:
            stats: 统计信息

        Returns:
            str: Prometheus格式字符串
        """
        lines = []
        labels = self._generate_labels()

        # 生成所有基础指标
        metrics_definitions = self._get_metrics_definitions()

        for metric_def in metrics_definitions:
            metric_lines = self._generate_single_metric(
                metric_def, stats, labels
            )
            lines.extend(metric_lines)

        return '\n'.join(lines)

    def _get_metrics_definitions(self) -> List[Dict[str, Any]]:
        """
        获取指标定义列表

        Returns:
            List[Dict[str, Any]]: 指标定义列表
        """
        return [
            {
                'name': 'pool_size',
                'help': 'Logger pool current size',
                'type': 'gauge',
                'key': 'pool_size',
                'unit': None
            },
            {
                'name': 'max_size',
                'help': 'Logger pool maximum size',
                'type': 'gauge',
                'key': 'max_size',
                'unit': None
            },
            {
                'name': 'created_count',
                'help': 'Total loggers created',
                'type': 'counter',
                'key': 'created_count',
                'unit': None
            },
            {
                'name': 'hit_count',
                'help': 'Total cache hits',
                'type': 'counter',
                'key': 'hit_count',
                'unit': None
            },
            {
                'name': 'hit_rate',
                'help': 'Cache hit rate',
                'type': 'gauge',
                'key': 'hit_rate',
                'unit': None
            },
            {
                'name': 'memory_usage_mb',
                'help': 'Memory usage in MB',
                'type': 'gauge',
                'key': 'memory_usage_mb',
                'unit': None
            },
            {
                'name': 'avg_access_time_ms',
                'help': 'Average access time in milliseconds',
                'type': 'gauge',
                'key': 'avg_access_time',
                'unit': 'ms',
                'converter': lambda x: x * 1000  # 转换为毫秒
            }
        ]

    def _generate_single_metric(self, metric_def: Dict[str, Any],
                               stats: Dict[str, Any], labels: str) -> List[str]:
        """
        生成单个指标的Prometheus格式

        Args:
            metric_def: 指标定义
            stats: 统计信息
            labels: Prometheus标签

        Returns:
            List[str]: Prometheus格式行列表
        """
        lines = []
        metric_name = f"{self.config.metric_prefix}_{metric_def['name']}"

        # 添加HELP注释
        if self.config.include_help_text:
            lines.append(f'# HELP {metric_name} {metric_def["help"]}')

        # 添加TYPE注释
        if self.config.include_type_info:
            lines.append(f'# TYPE {metric_name} {metric_def["type"]}')

        # 获取指标值
        value = stats.get(metric_def['key'], 0.0)

        # 应用转换器（如需要）
        if 'converter' in metric_def:
            value = metric_def['converter'](value)

        # 添加指标行
        lines.append(f'{metric_name}{{{labels}}} {value}')

        return lines

    def _generate_prometheus_compressed(self, stats: Dict[str, Any]) -> str:
        """
        生成压缩的Prometheus格式

        Args:
            stats: 统计信息

        Returns:
            str: 压缩的Prometheus格式字符串
        """
        # 简化的实现，实际可以根据配置进行更复杂的压缩
        return self._generate_prometheus_format(stats)

    def _generate_json_format(self, stats: Dict[str, Any]) -> str:
        """
        生成JSON格式

        Args:
            stats: 统计信息

        Returns:
            str: JSON格式字符串
        """
        try:
            # 添加元数据
            export_data = {
                'metadata': {
                    'pool_name': self.pool_name,
                    'export_time': datetime.now().isoformat(),
                    'format': 'json',
                    'version': '1.0'
                },
                'metrics': stats
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"生成JSON格式失败: {e}")
            return '{"error": "Failed to generate JSON format"}'

    def _generate_labels(self) -> str:
        """
        生成Prometheus标签字符串

        Returns:
            str: 标签字符串
        """
        labels = [f'pool="{self.pool_name}"']

        # 添加默认标签
        for key, value in self.config.default_labels.items():
            labels.append(f'{key}="{value}"')

        return ','.join(labels)

    def export_to_file(self, format_type: str = 'prometheus', file_path: Optional[str] = None) -> bool:
        """
        导出到文件

        Args:
            format_type: 导出格式 ('prometheus' 或 'json')
            file_path: 文件路径，如果为None则使用默认路径

        Returns:
            bool: 是否成功导出
        """
        try:
            if format_type not in self._export_cache:
                print(f"不支持的导出格式: {format_type}")
                return False

            data = self._export_cache[format_type]

            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"metrics_export_{self.pool_name}_{timestamp}.{format_type}"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)

            print(f"指标已导出到文件: {file_path}")
            return True

        except Exception as e:
            print(f"导出到文件失败: {e}")
            return False

    def get_supported_formats(self) -> list:
        """
        获取支持的导出格式

        Returns:
            list: 支持的格式列表
        """
        return ['prometheus', 'json']

    def validate_export_data(self, format_type: str) -> bool:
        """
        验证导出数据

        Args:
            format_type: 格式类型

        Returns:
            bool: 数据是否有效
        """
        if format_type not in self._export_cache:
            return False

        data = self._export_cache[format_type]

        try:
            if format_type == 'json':
                json.loads(data)
                return True
            elif format_type == 'prometheus':
                # 简单的Prometheus格式验证
                return bool(data.strip())
            else:
                return False

        except Exception:
            return False

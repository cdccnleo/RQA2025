import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层指标收集器

提供统一的指标收集接口和缓存机制，支持多种指标类型和收集策略。
"""

import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta
import numpy as np


logger = logging.getLogger(__name__)


class MetricCategory(Enum):

    """指标类别枚举"""
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    CUSTOM = "custom"


class MetricType(Enum):

    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:

    """指标数据结构"""
    name: str
    value: float
    timestamp: float
    category: MetricCategory
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:

    """
    指标收集器

    提供统一的指标收集接口和缓存机制，支持：
    - 多种指标类型
    - 自动缓存管理
    - 批量收集
    - 指标聚合
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化指标收集器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 指标存储
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metrics_lock = threading.Lock()
        self._metric_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.collection_history: List[Dict[str, Any]] = []
        self.aggregation_rules: Dict[str, Dict[str, Any]] = {}

        # 收集器注册表
        self.collectors: Dict[str, Callable] = {}
        self.collector_configs: Dict[str, Dict] = {}

        # 缓存配置
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟
        self.cache: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()

        # 聚合配置
        self.aggregation_enabled = self.config.get('aggregation_enabled', True)
        self.aggregation_window = self.config.get('aggregation_window', 60)  # 1分钟

        # 预定义收集器
        self._register_default_collectors()

        logger.info("指标收集器初始化完成")

    def _register_default_collectors(self) -> None:
        """注册预定义收集器"""
        # 特征生成收集器
        self.register_collector("feature_generation", self._collect_feature_generation_metrics)

        # 特征处理收集器
        self.register_collector("feature_processing", self._collect_feature_processing_metrics)

        # 缓存收集器
        self.register_collector("cache_metrics", self._collect_cache_metrics)

        # 内存收集器
        self.register_collector("memory_metrics", self._collect_memory_metrics)

    def register_collector(self, name: str, collector_func: Callable,


                           config: Optional[Dict] = None) -> None:
        """
        注册指标收集器

        Args:
            name: 收集器名称
            collector_func: 收集函数
            config: 收集器配置
        """
        self.collectors[name] = collector_func
        self.collector_configs[name] = config or {}

        logger.info(f"指标收集器 {name} 注册成功")

    def unregister_collector(self, name: str) -> None:
        """
        注销指标收集器

        Args:
            name: 收集器名称
        """
        if name in self.collectors:
            del self.collectors[name]
            del self.collector_configs[name]
            logger.info(f"指标收集器 {name} 注销成功")
        else:
            logger.warning(f"指标收集器 {name} 不存在")

    def collect_metric(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.CUSTOM,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        收集单个指标

        Args:
            name: 指标名称
            value: 指标值
            category: 指标类别
            metric_type: 指标类型
            labels: 标签
            metadata: 元数据
        """
        resolved_labels = labels or tags or {}
        if timestamp is None:
            timestamp_dt = datetime.now()
        elif isinstance(timestamp, (int, float)):
            timestamp_dt = datetime.fromtimestamp(timestamp)
        else:
            timestamp_dt = timestamp

        metric_data = MetricData(
            name=name,
            value=value,
            timestamp=timestamp_dt.timestamp(),
            category=category,
            metric_type=metric_type,
            labels=resolved_labels,
            metadata=metadata or {}
        )

        history_entry = {
            'name': name,
            'value': value,
            'timestamp': timestamp_dt,
            'tags': resolved_labels,
            'category': category.value,
            'metric_type': metric_type.value,
            'metadata': metadata or {},
        }

        with self.metrics_lock:
            self.metrics[name].append(metric_data)
            self._metric_records[name].append(history_entry)
            self.collection_history.append(history_entry)

        # 更新缓存
        if self.cache_enabled:
            self._update_cache(name, metric_data)

        logger.debug(f"指标 {name} 收集完成: {value}")
        return True

    def collect_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """
        批量收集指标

        Args:
            metrics: 指标列表，每个指标包含name, value, category, metric_type, labels, metadata
        """
        for metric in metrics:
            self.collect_metric(
                name=metric['name'],
                value=metric['value'],
                category=metric.get('category', MetricCategory.CUSTOM),
                metric_type=metric.get('metric_type', MetricType.GAUGE),
                labels=metric.get('labels', {}),
                metadata=metric.get('metadata', {}),
                timestamp=metric.get('timestamp'),
            )

    def collect_from_collector(self, collector_name: str,


                               context: Optional[Dict] = None) -> None:
        """
        从指定收集器收集指标

        Args:
            collector_name: 收集器名称
            context: 上下文信息
        """
        if collector_name not in self.collectors:
            logger.warning(f"收集器 {collector_name} 不存在")
            return

        try:
            collector_func = self.collectors[collector_name]
            config = self.collector_configs[collector_name]

            # 执行收集器
            metrics = collector_func(config, context or {})

            if metrics:
                self.collect_metrics(metrics)

        except Exception as e:
            logger.error(f"从收集器 {collector_name} 收集指标失败: {str(e)}")

    def collect_all(self, context: Optional[Dict] = None) -> None:
        """
        从所有收集器收集指标

        Args:
            context: 上下文信息
        """
        for collector_name in self.collectors:
            self.collect_from_collector(collector_name, context)

    def get_metric(self, name: str,


                   window: Optional[int] = None) -> List[MetricData]:
        """
        获取指标数据

        Args:
            name: 指标名称
            window: 时间窗口（秒）

        Returns:
            指标数据列表
        """
        if name not in self.metrics:
            return []

        metrics = list(self.metrics[name])

        if window:
            cutoff_time = time.time() - window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        return metrics

    def get_metrics_by_category(self, category: MetricCategory,
                                window: Optional[int] = None) -> Dict[str, List[MetricData]]:
        """
        按类别获取指标

        Args:
            category: 指标类别
            window: 时间窗口（秒）

        Returns:
            指标数据字典
        """
        result = {}

        with self.metrics_lock:
            for name, metric_queue in self.metrics.items():
                metrics = list(metric_queue)

                if window:
                    cutoff_time = time.time() - window
                    metrics = [m for m in metrics if m.timestamp >= cutoff_time]

                # 过滤指定类别
                category_metrics = [m for m in metrics if m.category == category]

                if category_metrics:
                    result[name] = category_metrics

        return result

    def get_metric_values(
        self,
        name: str,
        time_range: Optional[tuple[datetime, datetime]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """查询某个指标的原始记录"""
        with self.metrics_lock:
            records = list(self._metric_records.get(name, []))

        if time_range:
            start, end = time_range
            records = [r for r in records if start <= r['timestamp'] <= end]

        if tags:
            records = [
                r for r in records
                if all(r['tags'].get(k) == v for k, v in tags.items())
            ]

        return [record.copy() for record in records]

    def get_latest_metrics(self, names: Optional[List[str]] = None) -> Dict[str, MetricData]:
        """
        获取最新指标

        Args:
            names: 指标名称列表，None表示所有指标

        Returns:
            最新指标字典
        """
        result = {}

        with self.metrics_lock:
            target_names = names or list(self.metrics.keys())

            for name in target_names:
                if name in self.metrics and self.metrics[name]:
                    result[name] = self.metrics[name][-1]

        return result

    def aggregate_metrics(self, name: str,
                          aggregation_type: str = "mean",
                          window: int = 60) -> Optional[float]:
        """
        聚合指标

        Args:
            name: 指标名称
            aggregation_type: 聚合类型 (mean, sum, min, max, count)
            window: 时间窗口（秒）

        Returns:
            聚合结果
        """
        metrics = self.get_metric(name, window)

        if not metrics:
            return None

        values = [m.value for m in metrics]

        agg = aggregation_type.lower()

        if agg in ("mean", "avg"):
            return np.mean(values)
        elif agg == "sum":
            return np.sum(values)
        elif agg == "min":
            return np.min(values)
        elif agg == "max":
            return np.max(values)
        elif agg == "count":
            return len(values)
        else:
            logger.warning(f"不支持的聚合类型: {aggregation_type}")
            return None

    def add_aggregation_rule(
        self,
        rule_name: str,
        metric_name: str,
        aggregation_type: str,
        interval_seconds: int = 60,
    ) -> bool:
        """注册指标聚合规则，便于外部调度器定期执行"""
        with self.metrics_lock:
            self.aggregation_rules[rule_name] = {
                'metric_name': metric_name,
                'aggregation_type': aggregation_type,
                'interval_seconds': interval_seconds,
                'last_aggregation': None,
            }
        return True

    def get_metrics_summary(self, window: int = 3600) -> Dict[str, Any]:
        """
        获取指标摘要

        Args:
            window: 时间窗口（秒）

        Returns:
            指标摘要字典
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'window': window,
            'total_metrics': 0,
            'categories': {},
            'top_metrics': []
        }

        cutoff_time = datetime.now() - timedelta(seconds=window)
        with self.metrics_lock:
            for name, records in self._metric_records.items():
                recent_records = [r for r in records if r['timestamp'] >= cutoff_time]
                if not recent_records:
                    continue

                summary['total_metrics'] += len(recent_records)

                for record in recent_records:
                    category = record['category']
                    summary['categories'][category] = summary['categories'].get(category, 0) + 1

                values = [r['value'] for r in recent_records if isinstance(r['value'], (int, float))]
                if values:
                    summary[name] = {
                        'count': len(recent_records),
                        'avg': float(np.mean(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'latest': values[-1],
                        'latest_timestamp': recent_records[-1]['timestamp'],
                    }

                summary['top_metrics'].append({
                    'name': name,
                    'latest_value': recent_records[-1]['value'],
                    'count': len(recent_records)
                })

        # 按最新值排序
        summary['top_metrics'].sort(key=lambda x: x['latest_value'], reverse=True)
        summary['top_metrics'] = summary['top_metrics'][:10]  # 只保留前10个

        return summary

    def clear_metrics(self, names: Optional[List[str]] = None) -> None:
        """
        清除指标数据

        Args:
            names: 指标名称列表，None表示清除所有
        """
        with self.metrics_lock:
            if names is None:
                self.metrics.clear()
                self._metric_records.clear()
                self.collection_history.clear()
            else:
                for name in names:
                    if name in self.metrics:
                        self.metrics[name].clear()
                    if name in self._metric_records:
                        self._metric_records[name].clear()

        logger.info(f"指标数据已清除: {names or 'all'}")

    def export_metrics(
        self,
        format: str = 'json',
        names: Optional[List[str]] = None,
        window: Optional[int] = None,
        file_path: Optional[str] = None,
    ):
        """
        导出指标数据；兼容旧版直接返回 JSON/CSV 数据，也支持写入文件。

        Args:
            format: 'json' or 'csv'，也可以直接传入文件路径
            names: 指标名称列表
            window: 时间窗口（秒）
            file_path: 若提供则写入文件
        """
        try:
            export_format = 'json'
            destination_path: Optional[Path] = Path(file_path) if file_path else None

            if destination_path is None and isinstance(format, str) and format.lower() not in {'json', 'csv'}:
                potential_path = Path(format)
                if potential_path.suffix:
                    destination_path = potential_path
                    export_format = potential_path.suffix.lstrip('.').lower() or 'json'
                else:
                    export_format = 'json'
            else:
                export_format = format.lower() if isinstance(format, str) else 'json'

            if destination_path and export_format not in {'json', 'csv'}:
                export_format = destination_path.suffix.lstrip('.').lower() or 'json'

            if export_format not in {'json', 'csv'}:
                export_format = 'json'

            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {}
            }

            cutoff = datetime.now() - timedelta(seconds=window) if window else None

            with self.metrics_lock:
                target_names = names or list(self._metric_records.keys())
                for name in target_names:
                    records = self._metric_records.get(name, [])
                    if not records:
                        continue
                    filtered = [
                        record for record in records
                        if cutoff is None or record['timestamp'] >= cutoff
                    ]
                    if not filtered:
                        continue
                    export_data['metrics'][name] = [
                        {
                            'value': record['value'],
                            'timestamp': record['timestamp'].isoformat(),
                            'category': record['category'],
                            'metric_type': record['metric_type'],
                            'labels': record['tags'],
                            'metadata': record['metadata'],
                        }
                        for record in filtered
                    ]

            if export_format == 'json':
                payload = json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                rows = []
                for name, metrics in export_data['metrics'].items():
                    for record in metrics:
                        rows.append({
                            'metric_name': name,
                            'value': record['value'],
                            'timestamp': record['timestamp'],
                            'category': record['category'],
                            'metric_type': record['metric_type'],
                            'labels': json.dumps(record['labels'], ensure_ascii=False),
                        })
                payload = rows

            if destination_path:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                with destination_path.open('w', encoding='utf-8') as f:
                    if isinstance(payload, str):
                        f.write(payload)
                    else:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info("指标数据已导出到: %s", destination_path)
                return destination_path

            return payload

        except Exception as e:
            logger.error(f"导出指标数据失败: {str(e)}")
            return None

    def _update_cache(self, name: str, metric_data: MetricData) -> None:
        """更新缓存"""
        with self.cache_lock:
            self.cache[name] = {
                'value': metric_data.value,
                'timestamp': metric_data.timestamp,
                'expires_at': time.time() + self.cache_ttl
            }

    def _collect_feature_generation_metrics(self, config: Dict, context: Dict) -> List[Dict]:
        """收集特征生成指标"""
        metrics = []

        # 特征生成时间
        if 'generation_time' in context:
            metrics.append({
                'name': 'feature_generation_time',
                'value': context['generation_time'],
                'category': MetricCategory.PERFORMANCE,
                'metric_type': MetricType.HISTOGRAM,
                'labels': context.get('labels', {}),
                'metadata': {'feature_count': context.get('feature_count', 0)}
            })

        # 特征生成数量
        if 'feature_count' in context:
            metrics.append({
                'name': 'feature_generation_count',
                'value': context['feature_count'],
                'category': MetricCategory.BUSINESS,
                'metric_type': MetricType.COUNTER,
                'labels': context.get('labels', {}),
                'metadata': {'symbol': context.get('symbol', 'unknown')}
            })

        return metrics

    def _collect_feature_processing_metrics(self, config: Dict, context: Dict) -> List[Dict]:
        """收集特征处理指标"""
        metrics = []

        # 处理时间
        if 'processing_time' in context:
            metrics.append({
                'name': 'feature_processing_time',
                'value': context['processing_time'],
                'category': MetricCategory.PERFORMANCE,
                'metric_type': MetricType.HISTOGRAM,
                'labels': context.get('labels', {}),
                'metadata': {'processor': context.get('processor', 'unknown')}
            })

        # 处理成功率
        if 'success_rate' in context:
            metrics.append({
                'name': 'feature_processing_success_rate',
                'value': context['success_rate'],
                'category': MetricCategory.BUSINESS,
                'metric_type': MetricType.GAUGE,
                'labels': context.get('labels', {}),
                'metadata': {'total_processed': context.get('total_processed', 0)}
            })

        return metrics

    def _collect_cache_metrics(self, config: Dict, context: Dict) -> List[Dict]:
        """收集缓存指标"""
        metrics = []

        # 缓存命中率
        if 'cache_hit_rate' in context:
            metrics.append({
                'name': 'cache_hit_rate',
                'value': context['cache_hit_rate'],
                'category': MetricCategory.PERFORMANCE,
                'metric_type': MetricType.GAUGE,
                'labels': context.get('labels', {}),
                'metadata': {'cache_size': context.get('cache_size', 0)}
            })

        # 缓存大小
        if 'cache_size' in context:
            metrics.append({
                'name': 'cache_size',
                'value': context['cache_size'],
                'category': MetricCategory.SYSTEM,
                'metric_type': MetricType.GAUGE,
                'labels': context.get('labels', {}),
                'metadata': {'max_cache_size': context.get('max_cache_size', 0)}
            })

        return metrics

    def _collect_memory_metrics(self, config: Dict, context: Dict) -> List[Dict]:
        """收集内存指标"""
        metrics = []

        try:
            import psutil

            # 内存使用率
            memory = psutil.virtual_memory()
            metrics.append({
                'name': 'memory_usage_percent',
                'value': memory.percent,
                'category': MetricCategory.SYSTEM,
                'metric_type': MetricType.GAUGE,
                'labels': {},
                'metadata': {
                    'total_memory': memory.total,
                    'available_memory': memory.available
                }
            })

            # 内存使用量
            metrics.append({
                'name': 'memory_usage_bytes',
                'value': memory.used,
                'category': MetricCategory.SYSTEM,
                'metric_type': MetricType.GAUGE,
                'labels': {},
                'metadata': {'total_memory': memory.total}
            })

        except ImportError:
            logger.warning("psutil未安装，无法收集内存指标")
        except Exception as e:
            logger.error(f"收集内存指标失败: {str(e)}")

        return metrics


# 全局指标收集器实例
_global_collector: Optional[MetricsCollector] = None


def get_collector(config: Optional[Dict] = None) -> MetricsCollector:
    """
    获取全局指标收集器实例

    Args:
        config: 收集器配置

    Returns:
        指标收集器实例
    """
    global _global_collector

    if _global_collector is None:
        _global_collector = MetricsCollector(config)

    return _global_collector


def collect_metric(name: str, value: float, **kwargs) -> None:
    """
    收集指标的便捷函数

    Args:
        name: 指标名称
        value: 指标值
        **kwargs: 其他参数
    """
    collector = get_collector()
    collector.collect_metric(name, value, **kwargs)

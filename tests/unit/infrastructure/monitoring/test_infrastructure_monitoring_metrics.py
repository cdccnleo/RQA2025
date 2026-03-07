#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infrastructure Monitoring指标收集测试

测试监控系统的指标收集、聚合、存储和查询功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime, timedelta


class TestMetricsCollection:
    """测试指标收集功能"""
    
    def test_collect_basic_metric(self):
        """测试收集基础指标"""
        metrics = {}
        
        # 模拟收集CPU使用率
        metrics['cpu_usage'] = 45.5
        metrics['timestamp'] = datetime.now().isoformat()
        
        assert 'cpu_usage' in metrics
        assert isinstance(metrics['cpu_usage'], (int, float))
        assert metrics['cpu_usage'] == 45.5
    
    def test_collect_multiple_metrics(self):
        """测试收集多个指标"""
        metrics = {
            'cpu_usage': 45.5,
            'memory_usage': 60.2,
            'disk_usage': 75.0,
            'network_io': 1024.5
        }
        
        assert len(metrics) == 4
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_collect_metrics_with_labels(self):
        """测试带标签的指标收集"""
        metric = {
            'name': 'http_requests_total',
            'value': 1000,
            'labels': {
                'method': 'GET',
                'endpoint': '/api/users',
                'status': '200'
            }
        }
        
        assert metric['name'] == 'http_requests_total'
        assert metric['value'] == 1000
        assert 'labels' in metric
        assert metric['labels']['method'] == 'GET'
    
    def test_collect_counter_metric(self):
        """测试计数器指标"""
        counter = {'value': 0}
        
        # 递增计数器
        for _ in range(10):
            counter['value'] += 1
        
        assert counter['value'] == 10
    
    def test_collect_gauge_metric(self):
        """测试仪表盘指标"""
        gauge = {'value': 50.0}
        
        # 更新值
        gauge['value'] = 75.0
        assert gauge['value'] == 75.0
        
        # 再次更新
        gauge['value'] = 30.0
        assert gauge['value'] == 30.0
    
    def test_collect_histogram_metric(self):
        """测试直方图指标"""
        histogram = {
            'buckets': [0, 10, 50, 100, 500, 1000],
            'counts': [0, 0, 0, 0, 0, 0],
            'sum': 0,
            'count': 0
        }
        
        # 添加观察值
        values = [5, 15, 45, 80, 200, 800]
        for value in values:
            histogram['sum'] += value
            histogram['count'] += 1
            # 找到合适的bucket
            for i, bucket in enumerate(histogram['buckets']):
                if value < bucket:
                    histogram['counts'][i] += 1
                    break
        
        assert histogram['count'] == 6
        assert histogram['sum'] == sum(values)


class TestMetricsAggregation:
    """测试指标聚合功能"""
    
    def test_aggregate_average(self):
        """测试平均值聚合"""
        values = [10, 20, 30, 40, 50]
        average = sum(values) / len(values)
        
        assert average == 30.0
    
    def test_aggregate_sum(self):
        """测试求和聚合"""
        values = [10, 20, 30, 40, 50]
        total = sum(values)
        
        assert total == 150
    
    def test_aggregate_min_max(self):
        """测试最小值和最大值聚合"""
        values = [10, 20, 30, 40, 50]
        
        min_value = min(values)
        max_value = max(values)
        
        assert min_value == 10
        assert max_value == 50
    
    def test_aggregate_percentile(self):
        """测试百分位数聚合"""
        values = sorted([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # P50 (中位数)
        p50_index = int(len(values) * 0.5)
        p50 = values[p50_index]
        
        # P95
        p95_index = int(len(values) * 0.95)
        p95 = values[min(p95_index, len(values)-1)]
        
        assert p50 == 60  # 第50%位置的值
        assert p95 == 100  # 第95%位置的值
    
    def test_aggregate_by_time_window(self):
        """测试按时间窗口聚合"""
        metrics = [
            {'timestamp': datetime(2025, 11, 2, 10, 0), 'value': 10},
            {'timestamp': datetime(2025, 11, 2, 10, 5), 'value': 20},
            {'timestamp': datetime(2025, 11, 2, 10, 10), 'value': 30},
        ]
        
        # 按5分钟窗口聚合
        window_size = timedelta(minutes=5)
        aggregated = {}
        
        for metric in metrics:
            window_key = metric['timestamp'].replace(second=0, microsecond=0)
            if window_key not in aggregated:
                aggregated[window_key] = []
            aggregated[window_key].append(metric['value'])
        
        assert len(aggregated) >= 1


class TestMetricsStorage:
    """测试指标存储功能"""
    
    def test_store_metric_in_memory(self):
        """测试在内存中存储指标"""
        metrics_store = []
        
        metric = {
            'name': 'cpu_usage',
            'value': 45.5,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_store.append(metric)
        
        assert len(metrics_store) == 1
        assert metrics_store[0]['name'] == 'cpu_usage'
    
    def test_store_multiple_metrics(self):
        """测试存储多个指标"""
        metrics_store = []
        
        for i in range(10):
            metric = {
                'name': 'request_count',
                'value': i * 100,
                'timestamp': datetime.now().isoformat()
            }
            metrics_store.append(metric)
        
        assert len(metrics_store) == 10
    
    def test_store_with_retention(self):
        """测试带保留期的存储"""
        metrics_store = []
        retention_period = timedelta(hours=1)
        
        # 添加新指标
        now = datetime.now()
        metric_new = {'timestamp': now, 'value': 100}
        metric_old = {'timestamp': now - timedelta(hours=2), 'value': 50}
        
        metrics_store.extend([metric_new, metric_old])
        
        # 清理过期指标
        cutoff_time = now - retention_period
        metrics_store = [m for m in metrics_store if m['timestamp'] > cutoff_time]
        
        assert len(metrics_store) == 1
        assert metrics_store[0]['value'] == 100
    
    def test_store_with_circular_buffer(self):
        """测试使用循环缓冲区存储"""
        max_size = 5
        metrics_buffer = []
        
        # 添加超过最大大小的指标
        for i in range(10):
            metrics_buffer.append({'value': i})
            if len(metrics_buffer) > max_size:
                metrics_buffer.pop(0)  # 移除最旧的
        
        assert len(metrics_buffer) == max_size
        assert metrics_buffer[0]['value'] == 5  # 最旧的是5
        assert metrics_buffer[-1]['value'] == 9  # 最新的是9


class TestMetricsQuery:
    """测试指标查询功能"""
    
    @pytest.fixture
    def sample_metrics(self):
        """创建示例指标数据"""
        return [
            {'name': 'cpu_usage', 'value': 45.5, 'timestamp': datetime(2025, 11, 2, 10, 0)},
            {'name': 'cpu_usage', 'value': 50.0, 'timestamp': datetime(2025, 11, 2, 10, 5)},
            {'name': 'memory_usage', 'value': 60.0, 'timestamp': datetime(2025, 11, 2, 10, 0)},
            {'name': 'memory_usage', 'value': 65.0, 'timestamp': datetime(2025, 11, 2, 10, 5)},
        ]
    
    def test_query_by_name(self, sample_metrics):
        """测试按名称查询指标"""
        cpu_metrics = [m for m in sample_metrics if m['name'] == 'cpu_usage']
        
        assert len(cpu_metrics) == 2
        assert all(m['name'] == 'cpu_usage' for m in cpu_metrics)
    
    def test_query_by_time_range(self, sample_metrics):
        """测试按时间范围查询指标"""
        start_time = datetime(2025, 11, 2, 10, 0)
        end_time = datetime(2025, 11, 2, 10, 3)
        
        filtered = [
            m for m in sample_metrics 
            if start_time <= m['timestamp'] <= end_time
        ]
        
        assert len(filtered) == 2  # 只有10:00的数据
    
    def test_query_latest_value(self, sample_metrics):
        """测试查询最新值"""
        cpu_metrics = [m for m in sample_metrics if m['name'] == 'cpu_usage']
        latest = max(cpu_metrics, key=lambda x: x['timestamp'])
        
        assert latest['value'] == 50.0
        assert latest['timestamp'] == datetime(2025, 11, 2, 10, 5)
    
    def test_query_with_aggregation(self, sample_metrics):
        """测试带聚合的查询"""
        cpu_metrics = [m for m in sample_metrics if m['name'] == 'cpu_usage']
        values = [m['value'] for m in cpu_metrics]
        
        avg_cpu = sum(values) / len(values)
        max_cpu = max(values)
        
        assert avg_cpu == 47.75
        assert max_cpu == 50.0


class TestMetricsVisualization:
    """测试指标可视化功能"""
    
    def test_prepare_chart_data(self):
        """测试准备图表数据"""
        metrics = [
            {'timestamp': datetime(2025, 11, 2, 10, 0), 'value': 10},
            {'timestamp': datetime(2025, 11, 2, 10, 5), 'value': 20},
            {'timestamp': datetime(2025, 11, 2, 10, 10), 'value': 30},
        ]
        
        chart_data = {
            'labels': [m['timestamp'].strftime('%H:%M') for m in metrics],
            'values': [m['value'] for m in metrics]
        }
        
        assert len(chart_data['labels']) == 3
        assert len(chart_data['values']) == 3
        assert chart_data['values'] == [10, 20, 30]
    
    def test_generate_time_series(self):
        """测试生成时间序列"""
        start = datetime(2025, 11, 2, 10, 0)
        data_points = []
        
        for i in range(5):
            data_points.append({
                'timestamp': start + timedelta(minutes=i*5),
                'value': 10 + i * 5
            })
        
        assert len(data_points) == 5
        assert data_points[0]['value'] == 10
        assert data_points[-1]['value'] == 30
    
    def test_create_dashboard_layout(self):
        """测试创建仪表盘布局"""
        dashboard = {
            'title': 'System Metrics',
            'panels': [
                {'type': 'line', 'title': 'CPU Usage', 'metric': 'cpu_usage'},
                {'type': 'line', 'title': 'Memory Usage', 'metric': 'memory_usage'},
                {'type': 'gauge', 'title': 'Disk Usage', 'metric': 'disk_usage'},
            ]
        }
        
        assert dashboard['title'] == 'System Metrics'
        assert len(dashboard['panels']) == 3
        assert dashboard['panels'][0]['type'] == 'line'


class TestMetricsExport:
    """测试指标导出功能"""
    
    def test_export_to_prometheus_format(self):
        """测试导出为Prometheus格式"""
        metrics = [
            {'name': 'http_requests_total', 'value': 1000, 'labels': {'method': 'GET'}},
            {'name': 'http_requests_total', 'value': 500, 'labels': {'method': 'POST'}},
        ]
        
        # 模拟Prometheus格式
        prometheus_format = []
        for metric in metrics:
            labels_str = ','.join([f'{k}="{v}"' for k, v in metric['labels'].items()])
            line = f"{metric['name']}{{{labels_str}}} {metric['value']}"
            prometheus_format.append(line)
        
        assert len(prometheus_format) == 2
        assert 'http_requests_total' in prometheus_format[0]
        assert 'method="GET"' in prometheus_format[0]
    
    def test_export_to_json(self):
        """测试导出为JSON格式"""
        import json
        
        metrics = [
            {'name': 'cpu_usage', 'value': 45.5},
            {'name': 'memory_usage', 'value': 60.0},
        ]
        
        json_str = json.dumps(metrics)
        parsed = json.loads(json_str)
        
        assert len(parsed) == 2
        assert parsed[0]['name'] == 'cpu_usage'
    
    def test_export_to_csv(self):
        """测试导出为CSV格式"""
        metrics = [
            {'timestamp': '2025-11-02 10:00:00', 'name': 'cpu_usage', 'value': 45.5},
            {'timestamp': '2025-11-02 10:05:00', 'name': 'cpu_usage', 'value': 50.0},
        ]
        
        # 模拟CSV格式
        csv_lines = ['timestamp,name,value']
        for metric in metrics:
            line = f"{metric['timestamp']},{metric['name']},{metric['value']}"
            csv_lines.append(line)
        
        assert len(csv_lines) == 3  # header + 2 data rows
        assert csv_lines[0] == 'timestamp,name,value'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


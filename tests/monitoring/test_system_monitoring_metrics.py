#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统监控指标收集测试
System Monitoring Metrics Collection Tests

测试系统监控指标的完整性和正确性，包括：
1. 系统资源指标收集
2. 应用性能指标收集
3. 业务指标收集
4. 自定义指标收集
5. 指标聚合和计算
6. 指标存储和持久化
7. 指标数据质量验证
8. 指标采集性能测试
"""

import pytest
import time
import psutil
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import json
import statistics
from datetime import datetime, timedelta

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestSystemResourceMetrics:
    """测试系统资源指标收集"""

    def setup_method(self):
        """测试前准备"""
        self.metrics_collector = Mock()
        self.system_monitor = Mock()

    def test_cpu_usage_metrics_collection(self):
        """测试CPU使用率指标收集"""
        # 模拟CPU使用率数据收集
        cpu_metrics = []

        def collect_cpu_metrics(duration_seconds: int = 5) -> List[Dict]:
            """收集CPU指标"""
            metrics = []

            for _ in range(duration_seconds):
                # 模拟CPU使用率（实际应该使用psutil）
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_times = psutil.cpu_times_percent(interval=0)

                metric = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'cpu_user': cpu_times.user,
                    'cpu_system': cpu_times.system,
                    'cpu_idle': cpu_times.idle,
                    'cpu_iowait': getattr(cpu_times, 'iowait', 0),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
                }

                metrics.append(metric)
                time.sleep(0.1)  # 短暂延迟

            return metrics

        # 收集CPU指标
        cpu_data = collect_cpu_metrics(3)

        # 验证指标收集
        assert len(cpu_data) >= 3, "应该收集到足够的CPU指标数据"
        assert all('cpu_percent' in m for m in cpu_data), "每个指标应该包含CPU使用率"
        assert all('timestamp' in m for m in cpu_data), "每个指标应该包含时间戳"

        # 验证数据合理性
        cpu_percents = [m['cpu_percent'] for m in cpu_data]
        assert all(0 <= p <= 100 for p in cpu_percents), "CPU使用率应该在0-100范围内"

        # 计算统计信息
        avg_cpu = statistics.mean(cpu_percents)
        max_cpu = max(cpu_percents)
        cpu_volatility = statistics.stdev(cpu_percents) if len(cpu_percents) > 1 else 0

        # 验证统计合理性
        assert 0 <= avg_cpu <= 100, f"平均CPU使用率不合理: {avg_cpu}"
        assert 0 <= max_cpu <= 100, f"最大CPU使用率不合理: {max_cpu}"

        # CPU波动应该相对稳定（测试环境）
        assert cpu_volatility < 50, f"CPU波动过大: {cpu_volatility}"

    def test_memory_usage_metrics_collection(self):
        """测试内存使用率指标收集"""
        # 模拟内存使用率数据收集
        memory_metrics = []

        def collect_memory_metrics(duration_seconds: int = 3) -> List[Dict]:
            """收集内存指标"""
            metrics = []

            for _ in range(duration_seconds):
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                metric = {
                    'timestamp': datetime.now(),
                    'memory_total': memory.total,
                    'memory_available': memory.available,
                    'memory_used': memory.used,
                    'memory_percent': memory.percent,
                    'memory_free': memory.free,
                    'swap_total': swap.total,
                    'swap_used': swap.used,
                    'swap_percent': swap.percent,
                    'swap_free': swap.free
                }

                metrics.append(metric)
                time.sleep(0.2)  # 短暂延迟

            return metrics

        # 收集内存指标
        memory_data = collect_memory_metrics(3)

        # 验证指标收集
        assert len(memory_data) >= 3, "应该收集到足够的内存指标数据"

        required_fields = ['memory_total', 'memory_used', 'memory_percent', 'swap_total', 'swap_used']
        for field in required_fields:
            assert all(field in m for m in memory_data), f"每个指标应该包含{field}"

        # 验证数据合理性
        memory_percents = [m['memory_percent'] for m in memory_data]
        swap_percents = [m['swap_percent'] for m in memory_data]

        assert all(0 <= p <= 100 for p in memory_percents), "内存使用率应该在0-100范围内"
        assert all(0 <= p <= 100 for p in swap_percents), "交换空间使用率应该在0-100范围内"

        # 验证内存总量一致性
        memory_totals = [m['memory_total'] for m in memory_data]
        assert len(set(memory_totals)) == 1, "内存总量应该保持一致"

        # 计算内存使用趋势
        avg_memory_percent = statistics.mean(memory_percents)
        memory_trend = 'stable'

        if len(memory_percents) > 1:
            first_half = memory_percents[:len(memory_percents)//2]
            second_half = memory_percents[len(memory_percents)//2:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg + 5:
                memory_trend = 'increasing'
            elif second_avg < first_avg - 5:
                memory_trend = 'decreasing'

        # 验证内存趋势合理性（测试环境应该相对稳定）
        assert memory_trend in ['stable', 'increasing', 'decreasing'], f"内存趋势异常: {memory_trend}"

    def test_disk_io_metrics_collection(self):
        """测试磁盘I/O指标收集"""
        # 模拟磁盘I/O指标收集
        disk_metrics = []

        def collect_disk_metrics(duration_seconds: int = 3) -> List[Dict]:
            """收集磁盘I/O指标"""
            metrics = []

            # 获取磁盘分区信息
            disk_partitions = psutil.disk_partitions()

            for _ in range(duration_seconds):
                timestamp = datetime.now()

                # 收集所有分区的使用情况
                partitions_usage = {}
                for partition in disk_partitions:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        partitions_usage[partition.mountpoint] = {
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percent': usage.percent
                        }
                    except:
                        continue  # 跳过无法访问的分区

                # 收集磁盘I/O统计
                try:
                    disk_io = psutil.disk_io_counters(perdisk=True) or {}
                    io_stats = {}
                    for disk_name, counters in disk_io.items():
                        io_stats[disk_name] = {
                            'read_count': counters.read_count,
                            'write_count': counters.write_count,
                            'read_bytes': counters.read_bytes,
                            'write_bytes': counters.write_bytes,
                            'read_time': counters.read_time,
                            'write_time': counters.write_time
                        }
                except:
                    io_stats = {}

                metric = {
                    'timestamp': timestamp,
                    'partitions': partitions_usage,
                    'io_stats': io_stats
                }

                metrics.append(metric)
                time.sleep(0.5)  # 短暂延迟

            return metrics

        # 收集磁盘指标
        disk_data = collect_disk_metrics(3)

        # 验证指标收集
        assert len(disk_data) >= 3, "应该收集到足够的磁盘指标数据"

        for metric in disk_data:
            assert 'partitions' in metric, "应该包含分区信息"
            assert 'io_stats' in metric, "应该包含I/O统计"

            # 验证分区信息
            if metric['partitions']:
                for mount_point, usage in metric['partitions'].items():
                    assert 'total' in usage, f"分区 {mount_point} 应该包含total"
                    assert 'used' in usage, f"分区 {mount_point} 应该包含used"
                    assert 'percent' in usage, f"分区 {mount_point} 应该包含percent"

                    # 验证数据合理性
                    assert usage['total'] > 0, f"分区 {mount_point} 总容量应该大于0"
                    assert 0 <= usage['percent'] <= 100, f"分区 {mount_point} 使用率应该在0-100范围内"

        # 验证数据一致性（分区信息应该相对稳定）
        if len(disk_data) > 1:
            first_partitions = disk_data[0]['partitions']
            last_partitions = disk_data[-1]['partitions']

            for mount_point in first_partitions:
                if mount_point in last_partitions:
                    first_total = first_partitions[mount_point]['total']
                    last_total = last_partitions[mount_point]['total']
                    assert first_total == last_total, f"分区 {mount_point} 总容量不应该变化"

    def test_network_io_metrics_collection(self):
        """测试网络I/O指标收集"""
        # 模拟网络I/O指标收集
        network_metrics = []

        def collect_network_metrics(duration_seconds: int = 3) -> List[Dict]:
            """收集网络I/O指标"""
            metrics = []

            for _ in range(duration_seconds):
                timestamp = datetime.now()

                # 收集网络接口统计
                net_io = psutil.net_io_counters(pernic=True) or {}
                interfaces_stats = {}

                for interface_name, counters in net_io.items():
                    interfaces_stats[interface_name] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv,
                        'packets_sent': counters.packets_sent,
                        'packets_recv': counters.packets_recv,
                        'errin': counters.errin,
                        'errout': counters.errout,
                        'dropin': counters.dropin,
                        'dropout': counters.dropout
                    }

                # 收集网络连接统计
                connections = psutil.net_connections(kind='inet')
                connection_stats = {
                    'total': len(connections),
                    'listening': len([c for c in connections if c.status == 'LISTEN']),
                    'established': len([c for c in connections if c.status == 'ESTABLISHED']),
                    'close_wait': len([c for c in connections if c.status == 'CLOSE_WAIT']),
                    'time_wait': len([c for c in connections if c.status == 'TIME_WAIT'])
                }

                metric = {
                    'timestamp': timestamp,
                    'interfaces': interfaces_stats,
                    'connections': connection_stats
                }

                metrics.append(metric)
                time.sleep(0.3)  # 短暂延迟

            return metrics

        # 收集网络指标
        network_data = collect_network_metrics(3)

        # 验证指标收集
        assert len(network_data) >= 3, "应该收集到足够的网络指标数据"

        for metric in network_data:
            assert 'interfaces' in metric, "应该包含接口统计"
            assert 'connections' in metric, "应该包含连接统计"

            # 验证连接统计
            connections = metric['connections']
            required_conn_fields = ['total', 'listening', 'established']
            for field in required_conn_fields:
                assert field in connections, f"连接统计应该包含{field}"
                assert connections[field] >= 0, f"{field} 应该大于等于0"

            # 验证接口统计
            if metric['interfaces']:
                for interface_name, stats in metric['interfaces'].items():
                    required_io_fields = ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv']
                    for field in required_io_fields:
                        assert field in stats, f"接口 {interface_name} 应该包含{field}"
                        assert stats[field] >= 0, f"接口 {interface_name} {field} 应该大于等于0"

        # 验证数据趋势（网络流量通常是递增的）
        if len(network_data) > 1:
            first_metric = network_data[0]
            last_metric = network_data[-1]

            # 检查是否有活跃的网络接口
            active_interfaces = []
            for interface_name in first_metric['interfaces']:
                if interface_name in last_metric['interfaces']:
                    first_bytes = first_metric['interfaces'][interface_name]['bytes_sent'] + \
                                 first_metric['interfaces'][interface_name]['bytes_recv']
                    last_bytes = last_metric['interfaces'][interface_name]['bytes_sent'] + \
                                last_metric['interfaces'][interface_name]['bytes_recv']

                    if last_bytes >= first_bytes:  # 流量应该不减少
                        active_interfaces.append(interface_name)

            # 应该至少有一个网络接口
            assert len(active_interfaces) > 0, "应该至少有一个活跃的网络接口"


class TestApplicationPerformanceMetrics:
    """测试应用性能指标收集"""

    def setup_method(self):
        """测试前准备"""
        self.app_monitor = Mock()
        self.performance_collector = Mock()

    def test_http_request_metrics_collection(self):
        """测试HTTP请求指标收集"""
        # 模拟HTTP请求指标收集
        http_metrics = []

        def collect_http_metrics(duration_seconds: int = 5) -> List[Dict]:
            """收集HTTP请求指标"""
            metrics = []

            # 模拟HTTP请求处理
            request_patterns = [
                {'endpoint': '/api/users', 'method': 'GET', 'status': 200, 'duration': 0.05},
                {'endpoint': '/api/orders', 'method': 'POST', 'status': 201, 'duration': 0.08},
                {'endpoint': '/api/products', 'method': 'GET', 'status': 200, 'duration': 0.03},
                {'endpoint': '/api/inventory', 'method': 'PUT', 'status': 200, 'duration': 0.12},
                {'endpoint': '/api/reports', 'method': 'GET', 'status': 500, 'duration': 2.5},  # 慢请求
            ]

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                # 随机选择请求模式
                pattern = request_patterns[int(time.time() * 1000) % len(request_patterns)]

                # 添加一些随机性
                duration_variation = pattern['duration'] * (0.8 + 0.4 * (time.time() % 1))

                metric = {
                    'timestamp': datetime.now(),
                    'endpoint': pattern['endpoint'],
                    'method': pattern['method'],
                    'status_code': pattern['status'],
                    'duration_seconds': duration_variation,
                    'response_size_bytes': int(1000 + 9000 * (time.time() % 1)),
                    'user_agent': 'TestClient/1.0',
                    'remote_ip': f'192.168.1.{int(time.time() * 100) % 255}'
                }

                http_metrics.append(metric)
                time.sleep(0.1)  # 模拟请求间隔

            return http_metrics

        # 收集HTTP指标
        http_data = collect_http_metrics(3)

        # 验证指标收集
        assert len(http_data) >= 10, "应该收集到足够的HTTP请求指标"

        required_fields = ['endpoint', 'method', 'status_code', 'duration_seconds', 'timestamp']
        for metric in http_data:
            for field in required_fields:
                assert field in metric, f"HTTP指标应该包含{field}"

        # 验证数据合理性
        status_codes = [m['status_code'] for m in http_data]
        durations = [m['duration_seconds'] for m in http_data]

        # HTTP状态码应该在有效范围内
        valid_status_codes = [200, 201, 400, 404, 500]
        for code in status_codes:
            assert code in valid_status_codes, f"无效的HTTP状态码: {code}"

        # 请求持续时间应该为正
        assert all(d > 0 for d in durations), "所有请求持续时间应该为正"

        # 计算性能统计
        avg_duration = statistics.mean(durations)
        p95_duration = sorted(durations)[int(len(durations) * 0.95)]
        error_rate = len([s for s in status_codes if s >= 400]) / len(status_codes)

        # 验证性能指标
        assert avg_duration < 1.0, f"平均响应时间过长: {avg_duration:.3f}s"
        assert p95_duration < 3.0, f"95%响应时间过长: {p95_duration:.3f}s"
        assert error_rate < 0.2, f"错误率过高: {error_rate:.3f}"

    def test_database_query_metrics_collection(self):
        """测试数据库查询指标收集"""
        # 模拟数据库查询指标收集
        db_metrics = []

        def collect_db_metrics(duration_seconds: int = 4) -> List[Dict]:
            """收集数据库查询指标"""
            metrics = []

            query_patterns = [
                {'query_type': 'SELECT', 'table': 'users', 'duration': 0.02, 'rows_affected': 1},
                {'query_type': 'SELECT', 'table': 'orders', 'duration': 0.05, 'rows_affected': 50},
                {'query_type': 'INSERT', 'table': 'transactions', 'duration': 0.08, 'rows_affected': 1},
                {'query_type': 'UPDATE', 'table': 'inventory', 'duration': 0.03, 'rows_affected': 1},
                {'query_type': 'SELECT', 'table': 'reports', 'duration': 1.2, 'rows_affected': 1000},  # 慢查询
            ]

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                pattern = query_patterns[int(time.time() * 1000) % len(query_patterns)]

                # 添加随机性
                duration_variation = pattern['duration'] * (0.7 + 0.6 * (time.time() % 1))

                metric = {
                    'timestamp': datetime.now(),
                    'query_type': pattern['query_type'],
                    'table': pattern['table'],
                    'duration_seconds': duration_variation,
                    'rows_affected': pattern['rows_affected'],
                    'connection_id': f'conn_{int(time.time() * 100) % 10}',
                    'success': duration_variation < 5.0  # 5秒超时
                }

                db_metrics.append(metric)
                time.sleep(0.15)  # 模拟查询间隔

            return db_metrics

        # 收集数据库指标
        db_data = collect_db_metrics(3)

        # 验证指标收集
        assert len(db_data) >= 10, "应该收集到足够的数据库查询指标"

        required_fields = ['query_type', 'table', 'duration_seconds', 'rows_affected', 'success']
        for metric in db_data:
            for field in required_fields:
                assert field in metric, f"数据库指标应该包含{field}"

        # 验证数据合理性
        query_types = [m['query_type'] for m in db_data]
        durations = [m['duration_seconds'] for m in db_data]
        rows_affected = [m['rows_affected'] for m in db_data]

        # 查询类型应该有效
        valid_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
        for qtype in query_types:
            assert qtype in valid_types, f"无效的查询类型: {qtype}"

        # 持续时间和影响行数应该为正
        assert all(d > 0 for d in durations), "所有查询持续时间应该为正"
        assert all(r >= 0 for r in rows_affected), "影响行数应该大于等于0"

        # 计算数据库性能统计
        avg_duration = statistics.mean(durations)
        p95_duration = sorted(durations)[int(len(durations) * 0.95)]
        success_rate = len([m for m in db_data if m['success']]) / len(db_data)

        # 按查询类型分组统计
        type_stats = {}
        for metric in db_data:
            qtype = metric['query_type']
            if qtype not in type_stats:
                type_stats[qtype] = []
            type_stats[qtype].append(metric['duration_seconds'])

        # SELECT查询通常最快
        if 'SELECT' in type_stats:
            select_avg = statistics.mean(type_stats['SELECT'])
            assert select_avg < avg_duration, "SELECT查询应该相对较快"

        # 验证性能指标
        assert avg_duration < 1.0, f"平均查询时间过长: {avg_duration:.3f}s"
        assert p95_duration < 2.0, f"95%查询时间过长: {p95_duration:.3f}s"
        assert success_rate > 0.95, f"查询成功率过低: {success_rate:.3f}"

    def test_cache_performance_metrics_collection(self):
        """测试缓存性能指标收集"""
        # 模拟缓存性能指标收集
        cache_metrics = []

        def collect_cache_metrics(duration_seconds: int = 4) -> List[Dict]:
            """收集缓存性能指标"""
            metrics = []

            cache_operations = ['GET', 'SET', 'DELETE', 'EXISTS']

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                operation = cache_operations[int(time.time() * 1000) % len(cache_operations)]

                # 模拟操作结果
                if operation == 'GET':
                    hit = (time.time() * 1000) % 10 < 7  # 70% 命中率
                    duration = 0.001 + 0.002 * (time.time() % 1)
                elif operation == 'SET':
                    hit = True  # SET总是"命中"
                    duration = 0.002 + 0.003 * (time.time() % 1)
                else:
                    hit = True
                    duration = 0.001 + 0.001 * (time.time() % 1)

                metric = {
                    'timestamp': datetime.now(),
                    'operation': operation,
                    'duration_seconds': duration,
                    'hit': hit,
                    'key': f'cache_key_{int(time.time() * 100) % 100}',
                    'size_bytes': int(100 + 900 * (time.time() % 1))
                }

                cache_metrics.append(metric)
                time.sleep(0.05)  # 模拟操作间隔

            return cache_metrics

        # 收集缓存指标
        cache_data = collect_cache_metrics(3)

        # 验证指标收集
        assert len(cache_data) >= 20, "应该收集到足够的缓存操作指标"

        required_fields = ['operation', 'duration_seconds', 'hit', 'timestamp']
        for metric in cache_data:
            for field in required_fields:
                assert field in metric, f"缓存指标应该包含{field}"

        # 验证数据合理性
        operations = [m['operation'] for m in cache_data]
        durations = [m['duration_seconds'] for m in cache_data]
        hits = [m['hit'] for m in cache_data]

        # 操作类型应该有效
        valid_operations = ['GET', 'SET', 'DELETE', 'EXISTS']
        for op in operations:
            assert op in valid_operations, f"无效的缓存操作: {op}"

        # 持续时间应该很短（缓存操作）
        assert all(0 < d < 0.01 for d in durations), "缓存操作时间应该很短"

        # 计算缓存性能统计
        hit_rate = sum(hits) / len(hits)
        avg_duration = statistics.mean(durations)

        # 按操作类型分组统计
        op_stats = {}
        for metric in cache_data:
            op = metric['operation']
            if op not in op_stats:
                op_stats[op] = {'durations': [], 'hits': []}
            op_stats[op]['durations'].append(metric['duration_seconds'])
            op_stats[op]['hits'].append(metric['hit'])

        # GET操作的命中率统计
        if 'GET' in op_stats:
            get_hits = op_stats['GET']['hits']
            get_hit_rate = sum(get_hits) / len(get_hits)
            assert 0.5 <= get_hit_rate <= 0.9, f"GET操作命中率不合理: {get_hit_rate:.3f}"

        # 验证性能指标
        assert hit_rate > 0.6, f"缓存命中率过低: {hit_rate:.3f}"
        assert avg_duration < 0.005, f"平均缓存操作时间过长: {avg_duration:.6f}s"


class TestBusinessMetricsCollection:
    """测试业务指标收集"""

    def setup_method(self):
        """测试前准备"""
        self.business_monitor = Mock()
        self.kpi_collector = Mock()

    def test_user_activity_metrics_collection(self):
        """测试用户活动指标收集"""
        # 模拟用户活动指标收集
        user_activity_metrics = []

        def collect_user_activity_metrics(duration_seconds: int = 5) -> List[Dict]:
            """收集用户活动指标"""
            metrics = []

            activity_types = ['login', 'page_view', 'api_call', 'purchase', 'logout']

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                activity = activity_types[int(time.time() * 1000) % len(activity_types)]

                # 模拟用户活动数据
                if activity == 'login':
                    duration = None
                    success = True
                elif activity == 'page_view':
                    duration = 0.5 + 2.0 * (time.time() % 1)
                    success = True
                elif activity == 'api_call':
                    duration = 0.1 + 0.9 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 9  # 90% 成功率
                elif activity == 'purchase':
                    duration = 1.0 + 3.0 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 8  # 80% 成功率
                else:  # logout
                    duration = 0.1
                    success = True

                metric = {
                    'timestamp': datetime.now(),
                    'user_id': f'user_{int(time.time() * 100) % 1000}',
                    'session_id': f'session_{int(time.time() * 10) % 100}',
                    'activity_type': activity,
                    'duration_seconds': duration,
                    'success': success,
                    'device_type': ['mobile', 'desktop', 'tablet'][int(time.time() * 100) % 3],
                    'location': ['US', 'EU', 'AS'][int(time.time() * 10) % 3]
                }

                user_activity_metrics.append(metric)
                time.sleep(0.08)  # 模拟活动间隔

            return user_activity_metrics

        # 收集用户活动指标
        activity_data = collect_user_activity_metrics(4)

        # 验证指标收集
        assert len(activity_data) >= 20, "应该收集到足够的用户活动指标"

        required_fields = ['user_id', 'activity_type', 'timestamp', 'success']
        for metric in activity_data:
            for field in required_fields:
                assert field in metric, f"用户活动指标应该包含{field}"

        # 验证数据合理性
        activity_types = [m['activity_type'] for m in activity_data]
        success_flags = [m['success'] for m in activity_data]
        durations = [m['duration_seconds'] for m in activity_data if m['duration_seconds'] is not None]

        # 活动类型应该有效
        valid_activities = ['login', 'page_view', 'api_call', 'purchase', 'logout']
        for activity in activity_types:
            assert activity in valid_activities, f"无效的活动类型: {activity}"

        # 持续时间应该为正（如果存在）
        assert all(d > 0 for d in durations), "活动持续时间应该为正"

        # 计算用户活动统计
        total_activities = len(activity_data)
        success_rate = sum(success_flags) / total_activities

        # 按活动类型分组统计
        activity_stats = {}
        for metric in activity_data:
            activity = metric['activity_type']
            if activity not in activity_stats:
                activity_stats[activity] = {'count': 0, 'successes': 0, 'durations': []}
            activity_stats[activity]['count'] += 1
            if metric['success']:
                activity_stats[activity]['successes'] += 1
            if metric['duration_seconds']:
                activity_stats[activity]['durations'].append(metric['duration_seconds'])

        # 验证统计结果
        assert success_rate > 0.8, f"用户活动成功率过低: {success_rate:.3f}"

        # 不同活动的统计
        if 'api_call' in activity_stats:
            api_success_rate = activity_stats['api_call']['successes'] / activity_stats['api_call']['count']
            assert api_success_rate > 0.85, f"API调用成功率过低: {api_success_rate:.3f}"

        if 'purchase' in activity_stats:
            purchase_success_rate = activity_stats['purchase']['successes'] / activity_stats['purchase']['count']
            assert purchase_success_rate > 0.75, f"购买成功率过低: {purchase_success_rate:.3f}"

    def test_business_transaction_metrics_collection(self):
        """测试业务交易指标收集"""
        # 模拟业务交易指标收集
        transaction_metrics = []

        def collect_transaction_metrics(duration_seconds: int = 5) -> List[Dict]:
            """收集业务交易指标"""
            metrics = []

            transaction_types = ['order', 'payment', 'refund', 'transfer']

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                tx_type = transaction_types[int(time.time() * 1000) % len(transaction_types)]

                # 模拟交易数据
                if tx_type == 'order':
                    amount = 50 + 200 * (time.time() % 1)
                    duration = 0.5 + 1.5 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 9  # 90% 成功率
                elif tx_type == 'payment':
                    amount = 25 + 150 * (time.time() % 1)
                    duration = 1.0 + 2.0 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 8  # 80% 成功率
                elif tx_type == 'refund':
                    amount = 10 + 100 * (time.time() % 1)
                    duration = 0.8 + 1.2 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 7  # 70% 成功率
                else:  # transfer
                    amount = 100 + 900 * (time.time() % 1)
                    duration = 2.0 + 3.0 * (time.time() % 1)
                    success = (time.time() * 1000) % 10 < 9  # 90% 成功率

                metric = {
                    'timestamp': datetime.now(),
                    'transaction_id': f'tx_{int(time.time() * 1000000)}',
                    'transaction_type': tx_type,
                    'amount_usd': amount,
                    'duration_seconds': duration,
                    'success': success,
                    'user_id': f'user_{int(time.time() * 100) % 1000}',
                    'currency': 'USD',
                    'payment_method': ['credit_card', 'paypal', 'bank_transfer'][int(time.time() * 10) % 3]
                }

                transaction_metrics.append(metric)
                time.sleep(0.12)  # 模拟交易间隔

            return transaction_metrics

        # 收集交易指标
        transaction_data = collect_transaction_metrics(4)

        # 验证指标收集
        assert len(transaction_data) >= 15, "应该收集到足够的交易指标"

        required_fields = ['transaction_id', 'transaction_type', 'amount_usd', 'success', 'timestamp']
        for metric in transaction_data:
            for field in required_fields:
                assert field in metric, f"交易指标应该包含{field}"

        # 验证数据合理性
        tx_types = [m['transaction_type'] for m in transaction_data]
        amounts = [m['amount_usd'] for m in transaction_data]
        durations = [m['duration_seconds'] for m in transaction_data]
        success_flags = [m['success'] for m in transaction_data]

        # 交易类型和金额应该有效
        valid_types = ['order', 'payment', 'refund', 'transfer']
        for tx_type in tx_types:
            assert tx_type in valid_types, f"无效的交易类型: {tx_type}"

        assert all(a > 0 for a in amounts), "所有交易金额应该为正"
        assert all(d > 0 for d in durations), "所有交易持续时间应该为正"

        # 计算交易统计
        total_transactions = len(transaction_data)
        success_rate = sum(success_flags) / total_transactions
        total_volume = sum(amounts)
        avg_transaction_time = statistics.mean(durations)

        # 按交易类型分组统计
        type_stats = {}
        for metric in transaction_data:
            tx_type = metric['transaction_type']
            if tx_type not in type_stats:
                type_stats[tx_type] = {'count': 0, 'volume': 0, 'successes': 0}
            type_stats[tx_type]['count'] += 1
            type_stats[tx_type]['volume'] += metric['amount_usd']
            if metric['success']:
                type_stats[tx_type]['successes'] += 1

        # 验证业务指标
        assert success_rate > 0.75, f"交易成功率过低: {success_rate:.3f}"
        assert total_volume > 0, "总交易量应该大于0"
        assert avg_transaction_time < 3.0, f"平均交易时间过长: {avg_transaction_time:.3f}s"

        # 不同交易类型的验证
        if 'payment' in type_stats:
            payment_success_rate = type_stats['payment']['successes'] / type_stats['payment']['count']
            assert payment_success_rate > 0.75, f"支付成功率过低: {payment_success_rate:.3f}"

        if 'order' in type_stats:
            order_volume = type_stats['order']['volume']
            assert order_volume > 0, "订单交易量应该大于0"

    def test_service_level_metrics_collection(self):
        """测试服务水平指标收集"""
        # 模拟服务水平指标收集
        sl_metrics = []

        def collect_sl_metrics(duration_seconds: int = 6) -> List[Dict]:
            """收集服务水平指标"""
            metrics = []

            services = ['api_gateway', 'user_service', 'order_service', 'payment_service', 'notification_service']

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                service = services[int(time.time() * 1000) % len(services)]

                # 模拟服务指标
                response_time = 0.1 + 0.4 * (time.time() % 1)
                availability = 0.98 + 0.04 * (time.time() % 1)  # 98-100% 可用性
                error_rate = 0.01 + 0.04 * (time.time() % 1)   # 1-5% 错误率
                throughput = 100 + 200 * (time.time() % 1)     # 100-300 req/s

                metric = {
                    'timestamp': datetime.now(),
                    'service_name': service,
                    'response_time_p95': response_time,
                    'availability_percentage': availability * 100,
                    'error_rate_percentage': error_rate * 100,
                    'throughput_req_per_sec': throughput,
                    'active_connections': int(50 + 150 * (time.time() % 1)),
                    'queue_depth': int(5 + 15 * (time.time() % 1))
                }

                sl_metrics.append(metric)
                time.sleep(0.2)  # 模拟指标收集间隔

            return sl_metrics

        # 收集服务水平指标
        sl_data = collect_sl_metrics(4)

        # 验证指标收集
        assert len(sl_data) >= 10, "应该收集到足够的服务水平指标"

        required_fields = ['service_name', 'response_time_p95', 'availability_percentage', 'error_rate_percentage']
        for metric in sl_data:
            for field in required_fields:
                assert field in metric, f"服务水平指标应该包含{field}"

        # 验证数据合理性
        response_times = [m['response_time_p95'] for m in sl_data]
        availabilities = [m['availability_percentage'] for m in sl_data]
        error_rates = [m['error_rate_percentage'] for m in sl_data]
        throughputs = [m['throughput_req_per_sec'] for m in sl_data]

        # 验证指标范围
        assert all(0 < rt < 1.0 for rt in response_times), "响应时间应该在合理范围内"
        assert all(95 <= a <= 100 for a in availabilities), "可用性应该在95-100%范围内"
        assert all(0 <= e <= 10 for e in error_rates), "错误率应该在0-10%范围内"
        assert all(t > 0 for t in throughputs), "吞吐量应该大于0"

        # 计算SLA统计
        avg_availability = statistics.mean(availabilities)
        avg_response_time = statistics.mean(response_times)
        max_error_rate = max(error_rates)
        avg_throughput = statistics.mean(throughputs)

        # 按服务分组统计
        service_stats = {}
        for metric in sl_data:
            service = metric['service_name']
            if service not in service_stats:
                service_stats[service] = {'response_times': [], 'availabilities': [], 'error_rates': []}
            service_stats[service]['response_times'].append(metric['response_time_p95'])
            service_stats[service]['availabilities'].append(metric['availability_percentage'])
            service_stats[service]['error_rates'].append(metric['error_rate_percentage'])

        # 计算各服务的SLA指标
        for service, stats in service_stats.items():
            service_avg_availability = statistics.mean(stats['availabilities'])
            service_avg_response_time = statistics.mean(stats['response_times'])
            service_max_error_rate = max(stats['error_rates'])

            # 验证服务SLA
            assert service_avg_availability >= 99.0, f"服务 {service} 可用性不符合SLA: {service_avg_availability:.2f}%"
            assert service_avg_response_time < 0.5, f"服务 {service} 响应时间不符合SLA: {service_avg_response_time:.3f}s"
            assert service_max_error_rate < 5.0, f"服务 {service} 错误率不符合SLA: {service_max_error_rate:.2f}%"

        # 验证整体SLA
        assert avg_availability >= 99.5, f"整体可用性不符合SLA: {avg_availability:.2f}%"
        assert avg_response_time < 0.3, f"整体响应时间不符合SLA: {avg_response_time:.3f}s"
        assert max_error_rate < 3.0, f"最大错误率不符合SLA: {max_error_rate:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__])

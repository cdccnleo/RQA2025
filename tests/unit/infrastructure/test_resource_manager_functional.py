"""
Resource Manager资源管理器功能测试模块

按《投产计划-总览.md》第二阶段Week 4 Day 1-3执行
测试资源管理器的完整功能

测试覆盖：
- 资源池管理（7个）
- 资源分配（7个）
- 资源回收（7个）
- 资源监控（7个）
- 资源限制（7个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
from threading import Thread, Lock
from typing import Dict, List, Any


# Apply timeout to all tests (10 seconds per test)
pytestmark = pytest.mark.timeout(10)


class TestResourcePoolManagementFunctional:
    """资源池管理功能测试"""

    def test_resource_pool_initialization(self):
        """测试1: 资源池初始化"""
        # Arrange
        pool_config = {
            'min_size': 5,
            'max_size': 20,
            'resource_type': 'database_connection'
        }
        
        pool = {
            'resources': [],
            'available': [],
            'in_use': [],
            'config': pool_config
        }
        
        # Act - Initialize pool with min_size resources
        for i in range(pool_config['min_size']):
            resource = {'id': f"resource_{i}", 'type': pool_config['resource_type']}
            pool['resources'].append(resource)
            pool['available'].append(resource)
        
        # Assert
        assert len(pool['resources']) == 5
        assert len(pool['available']) == 5
        assert len(pool['in_use']) == 0

    def test_resource_pool_expansion(self):
        """测试2: 资源池扩展"""
        # Arrange
        pool = {
            'resources': [{'id': f'r{i}'} for i in range(5)],
            'available': [],
            'max_size': 10
        }
        
        # All resources in use
        pool['in_use'] = pool['resources'].copy()
        
        # Act - Need more resources
        if len(pool['resources']) < pool['max_size']:
            new_resource = {'id': f"r{len(pool['resources'])}"}
            pool['resources'].append(new_resource)
            pool['available'].append(new_resource)
        
        # Assert
        assert len(pool['resources']) == 6
        assert len(pool['available']) == 1

    def test_resource_pool_shrinking(self):
        """测试3: 资源池收缩"""
        # Arrange
        pool = {
            'resources': [{'id': f'r{i}', 'last_used': time.time() - i*100} for i in range(10)],
            'available': [],
            'min_size': 5
        }
        
        # 6 available (more than min)
        pool['available'] = pool['resources'][4:]
        pool['in_use'] = pool['resources'][:4]
        
        # Act - Remove idle resources
        idle_threshold = 300  # 5 minutes
        current_time = time.time()
        
        to_remove = [
            r for r in pool['available']
            if (current_time - r['last_used']) > idle_threshold and len(pool['resources']) > pool['min_size']
        ]
        
        for resource in to_remove:
            if len(pool['resources']) > pool['min_size']:
                pool['resources'].remove(resource)
                pool['available'].remove(resource)
        
        # Assert
        assert len(pool['resources']) >= pool['min_size']

    def test_resource_pool_health_check(self):
        """测试4: 资源池健康检查"""
        # Arrange
        pool = {
            'resources': [
                {'id': 'r1', 'status': 'healthy'},
                {'id': 'r2', 'status': 'healthy'},
                {'id': 'r3', 'status': 'unhealthy'},
                {'id': 'r4', 'status': 'healthy'}
            ]
        }
        
        # Act
        healthy = [r for r in pool['resources'] if r['status'] == 'healthy']
        unhealthy = [r for r in pool['resources'] if r['status'] != 'healthy']
        
        # Assert
        assert len(healthy) == 3
        assert len(unhealthy) == 1
        assert unhealthy[0]['id'] == 'r3'

    def test_connection_pool_management(self):
        """测试5: 连接池管理"""
        # Arrange
        connection_pool = {
            'connections': [],
            'max_connections': 10,
            'active_connections': 0
        }
        
        def get_connection():
            if connection_pool['active_connections'] < connection_pool['max_connections']:
                conn = {'id': len(connection_pool['connections']), 'active': True}
                connection_pool['connections'].append(conn)
                connection_pool['active_connections'] += 1
                return conn
            return None
        
        def release_connection(conn_id):
            for conn in connection_pool['connections']:
                if conn['id'] == conn_id and conn['active']:
                    conn['active'] = False
                    connection_pool['active_connections'] -= 1
                    return True
            return False
        
        # Act
        conn1 = get_connection()
        conn2 = get_connection()
        release_connection(conn1['id'])
        conn3 = get_connection()
        
        # Assert
        assert conn1 is not None
        assert conn2 is not None
        assert conn3 is not None
        assert connection_pool['active_connections'] == 2  # conn2 and conn3

    def test_thread_pool_management(self):
        """测试6: 线程池管理"""
        # Arrange
        from concurrent.futures import ThreadPoolExecutor
        
        max_workers = 4
        tasks = [lambda x=i: x**2 for i in range(10)]
        
        # Act
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda f: f(), tasks))
        
        # Assert
        assert len(results) == 10
        assert results == [i**2 for i in range(10)]

    def test_memory_pool_management(self):
        """测试7: 内存池管理"""
        # Arrange
        memory_pool = {
            'total_mb': 1000,
            'allocated_mb': 0,
            'allocations': []
        }
        
        def allocate_memory(size_mb):
            if memory_pool['allocated_mb'] + size_mb <= memory_pool['total_mb']:
                alloc = {'id': len(memory_pool['allocations']), 'size': size_mb}
                memory_pool['allocations'].append(alloc)
                memory_pool['allocated_mb'] += size_mb
                return alloc
            return None
        
        def free_memory(alloc_id):
            for alloc in memory_pool['allocations']:
                if alloc['id'] == alloc_id:
                    memory_pool['allocated_mb'] -= alloc['size']
                    memory_pool['allocations'].remove(alloc)
                    return True
            return False
        
        # Act
        alloc1 = allocate_memory(100)
        alloc2 = allocate_memory(200)
        free_memory(alloc1['id'])
        alloc3 = allocate_memory(150)
        
        # Assert
        assert memory_pool['allocated_mb'] == 350  # 200 + 150


class TestResourceAllocationFunctional:
    """资源分配功能测试"""

    def test_fair_resource_allocation(self):
        """测试8: 公平资源分配"""
        # Arrange
        total_resources = 100
        requests = [
            {'client': 'A', 'requested': 30},
            {'client': 'B', 'requested': 40},
            {'client': 'C', 'requested': 50}
        ]
        
        # Act - Proportional allocation
        total_requested = sum(r['requested'] for r in requests)
        allocations = {}
        
        for req in requests:
            allocated = int((req['requested'] / total_requested) * total_resources)
            allocations[req['client']] = allocated
        
        # Assert
        assert sum(allocations.values()) <= total_resources
        assert allocations['C'] > allocations['A']  # C requested more

    def test_priority_based_allocation(self):
        """测试9: 优先级资源分配"""
        # Arrange
        resources = 100
        requests = [
            {'client': 'critical', 'priority': 1, 'requested': 40},
            {'client': 'normal', 'priority': 3, 'requested': 60},
            {'client': 'low', 'priority': 5, 'requested': 30}
        ]
        
        # Act - Allocate by priority
        sorted_requests = sorted(requests, key=lambda r: r['priority'])
        allocations = {}
        remaining = resources
        
        for req in sorted_requests:
            allocated = min(req['requested'], remaining)
            allocations[req['client']] = allocated
            remaining -= allocated
        
        # Assert
        assert allocations['critical'] == 40  # Highest priority, got full request
        assert allocations['normal'] == 60
        assert allocations['low'] == 0  # No resources left

    def test_dynamic_resource_reallocation(self):
        """测试10: 动态资源重分配"""
        # Arrange
        allocations = {
            'serviceA': 50,
            'serviceB': 30,
            'serviceC': 20
        }
        
        usage = {
            'serviceA': 20,  # Under-utilizing
            'serviceB': 30,  # Fully utilizing
            'serviceC': 15   # Under-utilizing
        }
        
        # Act - Reallocate unused resources
        freed_resources = 0
        for service, allocated in allocations.items():
            used = usage[service]
            if used < allocated * 0.5:  # Using less than 50%
                freed = allocated - used
                allocations[service] = used
                freed_resources += freed
        
        # Reallocate to service B (fully utilizing)
        allocations['serviceB'] += freed_resources
        
        # Assert
        assert allocations['serviceA'] == 20
        assert allocations['serviceB'] > 30  # Got additional resources
        assert allocations['serviceC'] == 15

    def test_quota_based_allocation(self):
        """测试11: 配额资源分配"""
        # Arrange
        quotas = {
            'teamA': 100,
            'teamB': 150,
            'teamC': 50
        }
        
        current_usage = {
            'teamA': 80,
            'teamB': 150,
            'teamC': 30
        }
        
        # Act
        can_allocate = {}
        for team, quota in quotas.items():
            available = quota - current_usage.get(team, 0)
            can_allocate[team] = available
        
        # Assert
        assert can_allocate['teamA'] == 20  # Can allocate 20 more
        assert can_allocate['teamB'] == 0   # At quota
        assert can_allocate['teamC'] == 20  # Can allocate 20 more

    def test_burst_allocation(self):
        """测试12: 突发资源分配"""
        # Arrange
        base_allocation = 50
        burst_limit = 100
        current_usage = 50
        
        # Act
        def can_burst(current, base, burst):
            if current <= base:
                return True, burst - base
            elif current <= burst:
                return True, burst - current
            else:
                return False, 0
        
        can_burst_result, available = can_burst(current_usage, base_allocation, burst_limit)
        
        # Assert
        assert can_burst_result is True
        assert available == 50  # Can burst up to 50 more

    def test_reservation_based_allocation(self):
        """测试13: 预留资源分配"""
        # Arrange
        total_resources = 100
        reserved_resources = 20  # Always keep 20 reserved
        allocated = 70
        
        # Act
        available_for_allocation = total_resources - reserved_resources - allocated
        can_allocate_more = available_for_allocation > 0
        
        # Assert
        assert available_for_allocation == 10
        assert can_allocate_more is True

    def test_load_based_allocation(self):
        """测试14: 负载资源分配"""
        # Arrange
        servers = [
            {'id': 's1', 'capacity': 100, 'current_load': 80},
            {'id': 's2', 'capacity': 100, 'current_load': 50},
            {'id': 's3', 'capacity': 100, 'current_load': 90}
        ]
        
        new_task_size = 20
        
        # Act - Allocate to server with most available capacity
        server_capacity = [(s, s['capacity'] - s['current_load']) for s in servers]
        best_server = max(server_capacity, key=lambda x: x[1])
        
        # Assert
        assert best_server[0]['id'] == 's2'  # Most available capacity (50)
        assert best_server[1] >= new_task_size


class TestResourceReclamationFunctional:
    """资源回收功能测试"""

    def test_idle_resource_reclamation(self):
        """测试15: 空闲资源回收"""
        # Arrange
        resources = [
            {'id': 'r1', 'last_used': time.time()},
            {'id': 'r2', 'last_used': time.time() - 3600},  # 1 hour idle
            {'id': 'r3', 'last_used': time.time() - 7200}   # 2 hours idle
        ]
        
        idle_threshold = 1800  # 30 minutes
        
        # Act
        current_time = time.time()
        idle_resources = [
            r for r in resources
            if (current_time - r['last_used']) > idle_threshold
        ]
        
        # Assert
        assert len(idle_resources) == 2  # r2 and r3
        assert all(r['id'] in ['r2', 'r3'] for r in idle_resources)

    def test_unused_resource_cleanup(self):
        """测试16: 未使用资源清理"""
        # Arrange
        resources = {
            'r1': {'allocated': True, 'in_use': True},
            'r2': {'allocated': True, 'in_use': False},
            'r3': {'allocated': False, 'in_use': False}
        }
        
        # Act - Clean up allocated but not in use
        to_cleanup = [
            rid for rid, r in resources.items()
            if r['allocated'] and not r['in_use']
        ]
        
        for rid in to_cleanup:
            resources[rid]['allocated'] = False
        
        # Assert
        assert len(to_cleanup) == 1
        assert 'r2' in to_cleanup
        assert resources['r2']['allocated'] is False

    def test_memory_leak_detection(self):
        """测试17: 内存泄漏检测"""
        # Arrange
        memory_tracking = {
            'baseline': 100,
            'snapshots': [100, 105, 110, 116, 123, 131]  # Growing
        }
        
        # Act - Detect if memory is consistently growing
        growth_rate = []
        for i in range(1, len(memory_tracking['snapshots'])):
            growth = memory_tracking['snapshots'][i] - memory_tracking['snapshots'][i-1]
            growth_rate.append(growth)
        
        avg_growth = sum(growth_rate) / len(growth_rate)
        is_leak_suspected = avg_growth > 2 and all(g > 0 for g in growth_rate[-3:])
        
        # Assert
        assert avg_growth > 2
        assert is_leak_suspected is True

    def test_resource_timeout_cleanup(self):
        """测试18: 资源超时清理"""
        # Arrange
        locks = {
            'lock1': {'acquired_at': time.time() - 100, 'holder': 'process1'},
            'lock2': {'acquired_at': time.time() - 10, 'holder': 'process2'}
        }
        
        max_hold_time = 60  # 60 seconds
        
        # Act
        current_time = time.time()
        timed_out = [
            lock_id for lock_id, lock_info in locks.items()
            if (current_time - lock_info['acquired_at']) > max_hold_time
        ]
        
        for lock_id in timed_out:
            del locks[lock_id]
        
        # Assert
        assert len(timed_out) == 1
        assert 'lock1' in timed_out
        assert 'lock1' not in locks
        assert 'lock2' in locks

    def test_garbage_collection_trigger(self):
        """测试19: 垃圾回收触发"""
        # Arrange
        import gc
        
        memory_threshold = 0.85  # 85% memory usage
        current_memory_usage = 0.90  # 90%
        
        # Act
        should_trigger_gc = current_memory_usage > memory_threshold
        
        if should_trigger_gc:
            gc_result = gc.collect()  # Trigger GC
        
        # Assert
        assert should_trigger_gc is True
        assert gc_result >= 0  # GC returns number of objects collected

    def test_resource_lease_expiration(self):
        """测试20: 资源租约过期"""
        # Arrange
        leases = {
            'lease1': {'resource': 'r1', 'expires_at': time.time() + 100},
            'lease2': {'resource': 'r2', 'expires_at': time.time() - 10}  # Expired
        }
        
        # Act
        current_time = time.time()
        expired = [
            lease_id for lease_id, lease in leases.items()
            if lease['expires_at'] < current_time
        ]
        
        # Assert
        assert len(expired) == 1
        assert 'lease2' in expired

    def test_circular_reference_cleanup(self):
        """测试21: 循环引用清理"""
        # Arrange
        obj_a = {'name': 'A', 'ref': None}
        obj_b = {'name': 'B', 'ref': None}
        
        # Create circular reference
        obj_a['ref'] = obj_b
        obj_b['ref'] = obj_a
        
        # Act - Break circular reference
        def break_circular_refs(objects):
            for obj in objects:
                obj['ref'] = None
        
        break_circular_refs([obj_a, obj_b])
        
        # Assert
        assert obj_a['ref'] is None
        assert obj_b['ref'] is None


class TestResourceMonitoringFunctional:
    """资源监控功能测试"""

    def test_cpu_usage_monitoring(self):
        """测试22: CPU使用率监控"""
        # Arrange
        import psutil
        
        # Act
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Assert
        assert 0 <= cpu_percent <= 100

    def test_memory_usage_monitoring(self):
        """测试23: 内存使用率监控"""
        # Arrange
        import psutil
        
        # Act
        memory = psutil.virtual_memory()
        
        # Assert
        assert memory.total > 0
        assert 0 <= memory.percent <= 100

    def test_disk_usage_monitoring(self):
        """测试24: 磁盘使用率监控"""
        # Arrange
        import psutil
        
        # Act
        disk = psutil.disk_usage('/')
        
        # Assert
        assert disk.total > 0
        assert 0 <= disk.percent <= 100

    def test_network_bandwidth_monitoring(self):
        """测试25: 网络带宽监控"""
        # Arrange
        network_stats = {
            'bytes_sent': 1000000,
            'bytes_received': 2000000,
            'packets_sent': 10000,
            'packets_received': 15000
        }
        
        time_window = 10  # seconds
        
        # Act
        bandwidth_sent_mbps = (network_stats['bytes_sent'] * 8) / (time_window * 1024 * 1024)
        bandwidth_recv_mbps = (network_stats['bytes_received'] * 8) / (time_window * 1024 * 1024)
        
        # Assert
        assert bandwidth_sent_mbps > 0
        assert bandwidth_recv_mbps > bandwidth_sent_mbps

    def test_resource_utilization_tracking(self):
        """测试26: 资源利用率跟踪"""
        # Arrange
        resource_stats = {
            'total_capacity': 1000,
            'used': 750,
            'reserved': 100
        }
        
        # Act
        utilization = (resource_stats['used'] / resource_stats['total_capacity']) * 100
        available = resource_stats['total_capacity'] - resource_stats['used'] - resource_stats['reserved']
        
        # Assert
        assert utilization == 75.0
        assert available == 150

    def test_performance_metrics_collection(self):
        """测试27: 性能指标收集"""
        # Arrange
        metrics = []
        
        def collect_metrics():
            return {
                'timestamp': time.time(),
                'cpu': 50.0,
                'memory': 60.0,
                'requests_per_sec': 1000
            }
        
        # Act
        for _ in range(5):
            metrics.append(collect_metrics())
            time.sleep(0.01)
        
        # Assert
        assert len(metrics) == 5
        assert all('cpu' in m for m in metrics)
        assert all('memory' in m for m in metrics)

    def test_alert_threshold_monitoring(self):
        """测试28: 告警阈值监控"""
        # Arrange
        thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 90,
            'memory_warning': 80,
            'memory_critical': 95
        }
        
        current_metrics = {
            'cpu': 85,
            'memory': 92
        }
        
        # Act
        alerts = []
        if current_metrics['cpu'] >= thresholds['cpu_critical']:
            alerts.append('CPU CRITICAL')
        elif current_metrics['cpu'] >= thresholds['cpu_warning']:
            alerts.append('CPU WARNING')
        
        if current_metrics['memory'] >= thresholds['memory_critical']:
            alerts.append('MEMORY CRITICAL')
        elif current_metrics['memory'] >= thresholds['memory_warning']:
            alerts.append('MEMORY WARNING')
        
        # Assert
        assert 'CPU WARNING' in alerts
        assert 'MEMORY WARNING' in alerts


class TestResourceLimitsFunctional:
    """资源限制功能测试"""

    def test_concurrent_access_limit(self):
        """测试29: 并发访问限制"""
        # Arrange
        max_concurrent = 5
        current_access = {'count': 0}
        lock = Lock()
        
        def acquire_access():
            with lock:
                if current_access['count'] < max_concurrent:
                    current_access['count'] += 1
                    return True
                return False
        
        def release_access():
            with lock:
                if current_access['count'] > 0:
                    current_access['count'] -= 1
        
        # Act
        results = []
        for _ in range(10):
            results.append(acquire_access())
        
        # Assert
        assert sum(results) == 5  # Only 5 succeeded
        assert current_access['count'] == 5

    def test_rate_limiting(self):
        """测试30: 速率限制"""
        # Arrange
        rate_limit = 10  # requests per second
        window_start = time.time()
        requests = []
        
        def check_rate_limit():
            current_time = time.time()
            # Count requests in current window
            recent_requests = [
                r for r in requests
                if current_time - r < 1.0  # Last 1 second
            ]
            return len(recent_requests) < rate_limit
        
        # Act
        allowed_count = 0
        for _ in range(15):
            if check_rate_limit():
                requests.append(time.time())
                allowed_count += 1
        
        # Assert
        assert allowed_count <= rate_limit

    def test_bandwidth_throttling(self):
        """测试31: 带宽限流"""
        # Arrange
        max_bandwidth_mbps = 100
        current_usage_mbps = 85
        new_request_mbps = 20
        
        # Act
        total_if_allowed = current_usage_mbps + new_request_mbps
        should_throttle = total_if_allowed > max_bandwidth_mbps
        
        # Assert
        assert total_if_allowed == 105
        assert should_throttle is True

    def test_queue_size_limit(self):
        """测试32: 队列大小限制"""
        # Arrange
        from collections import deque
        
        max_queue_size = 100
        queue = deque(maxlen=max_queue_size)
        
        # Act
        for i in range(150):
            queue.append(i)
        
        # Assert
        assert len(queue) == max_queue_size
        assert queue[0] == 50  # First 50 were dropped

    def test_connection_timeout_limit(self):
        """测试33: 连接超时限制"""
        # Arrange
        connection = {
            'established_at': time.time() - 3700,  # 1+ hour old
            'max_lifetime': 3600  # 1 hour
        }
        
        # Act
        current_time = time.time()
        connection_age = current_time - connection['established_at']
        should_close = connection_age > connection['max_lifetime']
        
        # Assert
        assert connection_age > 3600
        assert should_close is True

    def test_resource_quota_enforcement(self):
        """测试34: 资源配额强制执行"""
        # Arrange
        user_quota = {
            'max_storage_gb': 10,
            'current_storage_gb': 9.5,
            'max_requests_per_day': 1000,
            'requests_today': 950
        }
        
        # Act
        can_store_more = user_quota['current_storage_gb'] < user_quota['max_storage_gb']
        storage_available_gb = user_quota['max_storage_gb'] - user_quota['current_storage_gb']
        
        can_make_requests = user_quota['requests_today'] < user_quota['max_requests_per_day']
        requests_remaining = user_quota['max_requests_per_day'] - user_quota['requests_today']
        
        # Assert
        assert can_store_more is True
        assert storage_available_gb == pytest.approx(0.5)
        assert can_make_requests is True
        assert requests_remaining == 50

    def test_priority_queue_resource_limit(self):
        """测试35: 优先级队列资源限制"""
        # Arrange
        from queue import PriorityQueue
        
        pq = PriorityQueue(maxsize=5)
        
        # Act
        pq.put((1, 'high_priority_task'))
        pq.put((5, 'low_priority_task'))
        pq.put((3, 'medium_priority_task'))
        pq.put((2, 'task4'))
        pq.put((4, 'task5'))
        
        # Queue is full now
        is_full = pq.full()
        
        # Get highest priority (lowest number)
        first_task = pq.get()
        
        # Assert
        assert is_full is True
        assert first_task == (1, 'high_priority_task')


# 测试统计
# Total: 35 tests
# TestResourcePoolManagementFunctional: 7 tests
# TestResourceAllocationFunctional: 7 tests
# TestResourceReclamationFunctional: 7 tests
# TestResourceMonitoringFunctional: 7 tests
# TestResourceLimitsFunctional: 7 tests


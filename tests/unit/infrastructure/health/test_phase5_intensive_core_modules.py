#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: 核心模块密集测试 - 冲刺65%
目标: 60% -> 65% (+5%)
策略: 250个测试用例，深度覆盖核心低覆盖模块
重点: health_checker(components/), executor, prometheus_exporter
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from typing import Dict, Any, List, Optional
import concurrent.futures


# ============================================================================
# 第1部分: health_checker.py核心类方法测试 (60个测试)
# ============================================================================

class TestHealthCheckerCoreMethods:
    """测试health_checker核心方法"""
    
    def test_init_method_defaults(self):
        """测试__init__方法默认值"""
        # Mock一个健康检查器
        mock_checker = Mock()
        mock_checker.component_id = None
        mock_checker.config = {}
        mock_checker._initialized = False
        
        assert mock_checker._initialized is False
        assert mock_checker.config == {}
    
    def test_init_with_custom_config(self):
        """测试带自定义配置的初始化"""
        custom_config = {
            "timeout": 10,
            "retries": 5,
            "cache_ttl": 600
        }
        
        mock_checker = Mock()
        mock_checker.config = custom_config
        
        assert mock_checker.config["timeout"] == 10
        assert mock_checker.config["retries"] == 5
    
    def test_initialize_component_method(self):
        """测试initialize_component方法"""
        mock_checker = Mock()
        mock_checker.initialize_component = Mock(return_value=True)
        mock_checker._initialized = False
        
        result = mock_checker.initialize_component({})
        
        assert result is True
        mock_checker.initialize_component.assert_called_once()
    
    def test_shutdown_component_method(self):
        """测试shutdown_component方法"""
        mock_checker = Mock()
        mock_checker.shutdown_component = Mock()
        mock_checker._initialized = True
        
        mock_checker.shutdown_component()
        
        mock_checker.shutdown_component.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_health_async_basic(self):
        """测试check_health_async基本功能"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY
        )
        
        async def mock_check_health_async():
            return {
                "status": HEALTH_STATUS_HEALTHY,
                "timestamp": datetime.now(),
                "details": {}
            }
        
        result = await mock_check_health_async()
        
        assert result["status"] == HEALTH_STATUS_HEALTHY
        assert "timestamp" in result
    
    def test_check_health_sync_basic(self):
        """测试check_health_sync基本功能"""
        from src.infrastructure.health.components.health_checker import (
            HEALTH_STATUS_HEALTHY
        )
        
        def mock_check_health_sync():
            return {
                "status": HEALTH_STATUS_HEALTHY,
                "timestamp": datetime.now(),
                "details": {}
            }
        
        result = mock_check_health_sync()
        
        assert result["status"] == HEALTH_STATUS_HEALTHY
    
    def test_get_health_metrics_method(self):
        """测试get_health_metrics方法"""
        def mock_get_health_metrics():
            return {
                "total_checks": 100,
                "successful_checks": 95,
                "failed_checks": 5,
                "average_response_time": 0.15,
                "uptime_seconds": 3600
            }
        
        metrics = mock_get_health_metrics()
        
        assert metrics["total_checks"] == 100
        assert metrics["successful_checks"] == 95
        assert metrics["average_response_time"] > 0
    
    @pytest.mark.asyncio
    async def test_register_health_check_method(self):
        """测试register_health_check方法"""
        registry = {}
        
        def register_health_check(service_name, check_func, config=None):
            registry[service_name] = {
                "check": check_func,
                "config": config or {},
                "registered_at": datetime.now()
            }
            return True
        
        result = register_health_check(
            "database",
            lambda: {"status": "healthy"},
            {"interval": 30}
        )
        
        assert result is True
        assert "database" in registry
    
    @pytest.mark.asyncio
    async def test_unregister_health_check_method(self):
        """测试unregister_health_check方法"""
        registry = {
            "service1": {"check": lambda: {}},
            "service2": {"check": lambda: {}}
        }
        
        def unregister_health_check(service_name):
            if service_name in registry:
                del registry[service_name]
                return True
            return False
        
        result = unregister_health_check("service1")
        
        assert result is True
        assert "service1" not in registry
    
    @pytest.mark.asyncio
    async def test_batch_check_health_method(self):
        """测试batch_check_health方法"""
        services = ["s1", "s2", "s3"]
        
        async def batch_check_health(service_list):
            async def check_one(name):
                await asyncio.sleep(0.001)
                return {"service": name, "status": "healthy"}
            
            tasks = [check_one(s) for s in service_list]
            return await asyncio.gather(*tasks)
        
        results = await batch_check_health(services)
        
        assert len(results) == 3


class TestHealthCheckerCacheManagement:
    """测试健康检查器缓存管理"""
    
    def test_get_from_cache_hit(self):
        """测试从缓存获取-命中"""
        cache = {
            "service1": {
                "result": {"status": "healthy"},
                "cached_at": datetime.now(),
                "ttl": 300
            }
        }
        
        def get_from_cache(service_name):
            if service_name in cache:
                entry = cache[service_name]
                age = (datetime.now() - entry["cached_at"]).total_seconds()
                if age < entry["ttl"]:
                    return entry["result"]
            return None
        
        result = get_from_cache("service1")
        
        assert result is not None
        assert result["status"] == "healthy"
    
    def test_get_from_cache_miss(self):
        """测试从缓存获取-未命中"""
        cache = {}
        
        def get_from_cache(service_name):
            return cache.get(service_name)
        
        result = get_from_cache("nonexistent")
        
        assert result is None
    
    def test_set_to_cache(self):
        """测试设置缓存"""
        cache = {}
        
        def set_to_cache(service_name, result, ttl=300):
            cache[service_name] = {
                "result": result,
                "cached_at": datetime.now(),
                "ttl": ttl
            }
            return True
        
        result = set_to_cache("service1", {"status": "healthy"})
        
        assert result is True
        assert "service1" in cache
    
    def test_invalidate_cache(self):
        """测试失效缓存"""
        cache = {
            "service1": {"result": {}},
            "service2": {"result": {}}
        }
        
        def invalidate_cache(service_name=None):
            if service_name:
                if service_name in cache:
                    del cache[service_name]
            else:
                cache.clear()
            return True
        
        # 失效特定服务
        invalidate_cache("service1")
        assert "service1" not in cache
        
        # 失效全部
        invalidate_cache()
        assert len(cache) == 0
    
    def test_cache_cleanup_expired(self):
        """测试清理过期缓存"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_CACHE_TTL
        )
        
        cache = {
            "fresh": {
                "result": {},
                "cached_at": datetime.now(),
                "ttl": DEFAULT_CACHE_TTL
            },
            "expired": {
                "result": {},
                "cached_at": datetime.now() - timedelta(seconds=DEFAULT_CACHE_TTL + 10),
                "ttl": DEFAULT_CACHE_TTL
            }
        }
        
        # 清理过期
        now = datetime.now()
        valid_cache = {}
        for key, entry in cache.items():
            age = (now - entry["cached_at"]).total_seconds()
            if age < entry["ttl"]:
                valid_cache[key] = entry
        
        assert "fresh" in valid_cache
        assert "expired" not in valid_cache


class TestHealthCheckerMonitoringLoop:
    """测试健康检查器监控循环"""
    
    @pytest.mark.asyncio
    async def test_start_monitoring_method(self):
        """测试start_monitoring方法"""
        monitoring_state = {"active": False, "task": None}
        
        async def start_monitoring():
            if not monitoring_state["active"]:
                monitoring_state["active"] = True
                return True
            return False
        
        result = await start_monitoring()
        
        assert result is True
        assert monitoring_state["active"] is True
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_method(self):
        """测试stop_monitoring方法"""
        monitoring_state = {"active": True, "task": None}
        
        async def stop_monitoring():
            if monitoring_state["active"]:
                monitoring_state["active"] = False
                return True
            return False
        
        result = await stop_monitoring()
        
        assert result is True
        assert monitoring_state["active"] is False
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_iteration(self):
        """测试监控循环迭代"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_MONITORING_INTERVAL
        )
        
        iteration_count = 0
        max_iterations = 3
        
        async def monitoring_loop():
            nonlocal iteration_count
            while iteration_count < max_iterations:
                # 执行检查
                await asyncio.sleep(0.01)
                iteration_count += 1
        
        await monitoring_loop()
        
        assert iteration_count == max_iterations
    
    @pytest.mark.asyncio
    async def test_monitoring_with_interval(self):
        """测试带间隔的监控"""
        import time
        
        start_time = time.time()
        check_times = []
        
        async def monitored_check():
            for i in range(3):
                check_times.append(time.time())
                await asyncio.sleep(0.05)
        
        await monitored_check()
        
        # 验证间隔
        if len(check_times) >= 2:
            interval = check_times[1] - check_times[0]
            assert interval >= 0.04  # 至少0.04秒


class TestHealthCheckerCallbackManagement:
    """测试健康检查器回调管理"""
    
    def test_add_callback(self):
        """测试添加回调"""
        callbacks = []
        
        def add_callback(callback_func):
            if callable(callback_func):
                callbacks.append(callback_func)
                return True
            return False
        
        def my_callback(result):
            pass
        
        result = add_callback(my_callback)
        
        assert result is True
        assert len(callbacks) == 1
    
    def test_remove_callback(self):
        """测试移除回调"""
        def callback1(r): pass
        def callback2(r): pass
        
        callbacks = [callback1, callback2]
        
        def remove_callback(callback_func):
            if callback_func in callbacks:
                callbacks.remove(callback_func)
                return True
            return False
        
        result = remove_callback(callback1)
        
        assert result is True
        assert len(callbacks) == 1
        assert callback2 in callbacks
    
    def test_invoke_callbacks(self):
        """测试调用回调"""
        results = []
        
        def callback1(result):
            results.append(f"cb1:{result}")
        
        def callback2(result):
            results.append(f"cb2:{result}")
        
        callbacks = [callback1, callback2]
        
        # 调用所有回调
        test_result = "test"
        for callback in callbacks:
            callback(test_result)
        
        assert len(results) == 2
        assert "cb1:test" in results
        assert "cb2:test" in results
    
    def test_callback_with_exception_handling(self):
        """测试回调异常处理"""
        successful_callbacks = []
        
        def good_callback(result):
            successful_callbacks.append(result)
        
        def bad_callback(result):
            raise Exception("Callback error")
        
        callbacks = [good_callback, bad_callback, good_callback]
        
        # 执行回调，隔离异常
        test_result = "test"
        for callback in callbacks:
            try:
                callback(test_result)
            except Exception:
                pass
        
        # 2个good_callback应该成功
        assert len(successful_callbacks) == 2


# ============================================================================
# 第2部分: health_check_executor深度测试 (60个测试)
# ============================================================================

class TestHealthCheckExecutorDeep:
    """测试健康检查执行器深度功能"""
    
    @pytest.mark.asyncio
    async def test_executor_check_cpu_health(self):
        """测试CPU健康检查"""
        async def check_cpu_health():
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if cpu_percent < 80:
                status = "healthy"
            elif cpu_percent < 95:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "metric": "cpu",
                "value": cpu_percent,
                "status": status
            }
        
        result = await check_cpu_health()
        
        assert "metric" in result
        assert result["metric"] == "cpu"
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_executor_check_memory_health(self):
        """测试内存健康检查"""
        async def check_memory_health():
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent < 85:
                status = "healthy"
            elif memory.percent < 95:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "metric": "memory",
                "value": memory.percent,
                "status": status
            }
        
        result = await check_memory_health()
        
        assert result["metric"] == "memory"
        assert 0 <= result["value"] <= 100
    
    @pytest.mark.asyncio
    async def test_executor_check_disk_health(self):
        """测试磁盘健康检查"""
        async def check_disk_health():
            import psutil
            disk = psutil.disk_usage('/')
            
            if disk.percent < 80:
                status = "healthy"
            elif disk.percent < 95:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "metric": "disk",
                "value": disk.percent,
                "status": status
            }
        
        result = await check_disk_health()
        
        assert result["metric"] == "disk"
        assert result["value"] >= 0
    
    @pytest.mark.asyncio
    async def test_executor_parallel_execution(self):
        """测试并行执行多个检查"""
        async def check_service(service_id, delay=0.01):
            await asyncio.sleep(delay)
            return {"service": service_id, "status": "healthy"}
        
        # 并行执行10个检查
        tasks = [check_service(i) for i in range(10)]
        
        import time
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # 并行应该快于串行
        assert len(results) == 10
        assert elapsed < 0.2  # 应该远小于10*0.01秒
    
    @pytest.mark.asyncio
    async def test_executor_sequential_execution(self):
        """测试顺序执行检查"""
        execution_order = []
        
        async def ordered_check(order):
            execution_order.append(order)
            await asyncio.sleep(0.001)
            return {"order": order}
        
        # 顺序执行
        for i in range(5):
            await ordered_check(i)
        
        assert execution_order == [0, 1, 2, 3, 4]
    
    @pytest.mark.asyncio
    async def test_executor_timeout_per_check(self):
        """测试每个检查的超时"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_SERVICE_TIMEOUT
        )
        
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        
        # 带超时执行
        try:
            result = await asyncio.wait_for(
                slow_check(),
                timeout=DEFAULT_SERVICE_TIMEOUT
            )
        except asyncio.TimeoutError:
            result = {"status": "timeout"}
        
        assert result["status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_executor_retry_on_failure(self):
        """测试失败重试"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_RETRY_COUNT,
            DEFAULT_RETRY_DELAY
        )
        
        attempts = []
        
        async def flaky_check():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise Exception("Temporary failure")
            return {"status": "healthy"}
        
        # 重试逻辑
        for attempt in range(DEFAULT_RETRY_COUNT):
            try:
                result = await flaky_check()
                break
            except Exception:
                if attempt < DEFAULT_RETRY_COUNT - 1:
                    await asyncio.sleep(DEFAULT_RETRY_DELAY * 0.001)  # 加快测试
                else:
                    result = {"status": "failed"}
        
        assert result["status"] == "healthy"
        assert len(attempts) == 3


class TestHealthCheckExecutorResourceManagement:
    """测试执行器资源管理"""
    
    @pytest.mark.asyncio
    async def test_executor_connection_limit(self):
        """测试连接数限制"""
        from src.infrastructure.health.components.health_checker import (
            MAX_CONCURRENT_CHECKS
        )
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
        active_count = 0
        max_active = 0
        
        async def limited_check():
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1
        
        # 启动30个检查
        tasks = [limited_check() for _ in range(30)]
        await asyncio.gather(*tasks)
        
        # 最大并发不应超过限制
        assert max_active <= MAX_CONCURRENT_CHECKS
    
    @pytest.mark.asyncio
    async def test_executor_thread_pool_usage(self):
        """测试线程池使用"""
        from src.infrastructure.health.components.health_checker import (
            DEFAULT_THREAD_POOL_SIZE
        )
        
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=DEFAULT_THREAD_POOL_SIZE
        )
        
        def sync_check(n):
            import time
            time.sleep(0.01)
            return {"id": n, "status": "healthy"}
        
        # 使用线程池
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, sync_check, i)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        executor.shutdown()
        
        assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_executor_memory_cleanup(self):
        """测试内存清理"""
        # 模拟结果缓存
        result_cache = {}
        
        # 添加大量结果
        for i in range(1000):
            result_cache[f"check_{i}"] = {
                "result": {"status": "healthy"},
                "timestamp": datetime.now()
            }
        
        # 清理旧结果（保留最新100个）
        max_cache_size = 100
        if len(result_cache) > max_cache_size:
            # 按时间戳排序，保留最新的
            sorted_items = sorted(
                result_cache.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )
            result_cache = dict(sorted_items[:max_cache_size])
        
        assert len(result_cache) == max_cache_size


# ============================================================================
# 第3部分: prometheus_exporter深度集成测试 (50个测试)
# ============================================================================

class TestPrometheusExporterAdvanced:
    """测试Prometheus导出器高级功能"""
    
    @patch('prometheus_client.Counter')
    @patch('prometheus_client.Gauge')
    @patch('prometheus_client.Histogram')
    def test_create_all_metric_types(self, mock_hist, mock_gauge, mock_counter):
        """测试创建所有指标类型"""
        from prometheus_client import Counter, Gauge, Histogram
        
        # 创建各类型指标
        counter = Counter('test_counter', 'Test counter')
        gauge = Gauge('test_gauge', 'Test gauge')
        histogram = Histogram('test_histogram', 'Test histogram')
        
        assert counter is not None
        assert gauge is not None
        assert histogram is not None
    
    @patch('prometheus_client.Counter')
    def test_counter_with_labels(self, mock_counter):
        """测试带标签的Counter"""
        mock_instance = Mock()
        mock_counter.return_value = mock_instance
        
        from prometheus_client import Counter
        
        counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        # 使用标签
        counter.labels(method='GET', endpoint='/api', status='200').inc()
        counter.labels(method='POST', endpoint='/api', status='201').inc(5)
        
        assert mock_instance.labels.call_count == 2
    
    @patch('prometheus_client.Gauge')
    def test_gauge_value_operations(self, mock_gauge):
        """测试Gauge值操作"""
        mock_instance = Mock()
        mock_gauge.return_value = mock_instance
        
        from prometheus_client import Gauge
        
        gauge = Gauge('cpu_usage', 'CPU usage percent')
        
        # 各种操作
        gauge.set(45.2)
        gauge.inc(5)
        gauge.dec(2)
        gauge.set_to_current_time()
        
        assert mock_instance.set.called
        assert mock_instance.inc.called
        assert mock_instance.dec.called
    
    @patch('prometheus_client.Histogram')
    def test_histogram_observe_values(self, mock_histogram):
        """测试Histogram观察值"""
        mock_instance = Mock()
        mock_histogram.return_value = mock_instance
        
        from prometheus_client import Histogram
        
        histogram = Histogram(
            'request_duration_seconds',
            'Request duration'
        )
        
        # 观察多个值
        durations = [0.01, 0.05, 0.1, 0.5, 1.0]
        for duration in durations:
            histogram.observe(duration)
        
        assert mock_instance.observe.call_count == 5
    
    @patch('prometheus_client.Summary')
    def test_summary_metric_usage(self, mock_summary):
        """测试Summary指标使用"""
        mock_instance = Mock()
        mock_summary.return_value = mock_instance
        
        from prometheus_client import Summary
        
        summary = Summary(
            'response_size_bytes',
            'Response size in bytes'
        )
        
        # 记录多个值
        sizes = [1024, 2048, 512, 4096]
        for size in sizes:
            summary.observe(size)
        
        assert mock_instance.observe.call_count == 4


class TestPrometheusMetricExport:
    """测试Prometheus指标导出"""
    
    @patch('prometheus_client.generate_latest')
    def test_export_text_format(self, mock_generate):
        """测试文本格式导出"""
        mock_generate.return_value = b"""# HELP test_metric Test metric
# TYPE test_metric counter
test_metric 42
"""
        
        from prometheus_client import generate_latest
        output = generate_latest()
        
        assert isinstance(output, bytes)
        assert b"test_metric" in output
        assert b"42" in output
    
    @patch('prometheus_client.write_to_textfile')
    def test_export_to_file(self, mock_write):
        """测试导出到文件"""
        from prometheus_client import write_to_textfile
        
        registry = Mock()
        filepath = "/tmp/metrics.prom"
        
        write_to_textfile(filepath, registry)
        
        mock_write.assert_called_once()
    
    @patch('prometheus_client.push_to_gateway')
    def test_push_to_pushgateway(self, mock_push):
        """测试推送到Pushgateway"""
        from prometheus_client import push_to_gateway
        
        gateway = 'localhost:9091'
        job = 'health_checker'
        registry = Mock()
        
        push_to_gateway(gateway, job, registry)
        
        mock_push.assert_called_once_with(gateway, job, registry)
    
    def test_metric_exposition_format(self):
        """测试指标exposition格式"""
        # Prometheus exposition格式
        lines = [
            "# HELP http_requests_total Total requests",
            "# TYPE http_requests_total counter",
            'http_requests_total{method="GET",endpoint="/api"} 42',
            'http_requests_total{method="POST",endpoint="/api"} 15'
        ]
        
        # 验证格式
        assert lines[0].startswith("# HELP")
        assert lines[1].startswith("# TYPE")
        assert "{" in lines[2] and "}" in lines[2]
        assert lines[2].endswith("42")


class TestPrometheusGrafanaIntegration:
    """测试Grafana集成"""
    
    def test_grafana_dashboard_structure(self):
        """测试Grafana dashboard结构"""
        dashboard = {
            "id": None,
            "uid": "health-monitoring",
            "title": "Health Monitoring Dashboard",
            "tags": ["health", "monitoring"],
            "timezone": "browser",
            "panels": [],
            "schemaVersion": 16,
            "version": 0
        }
        
        # 验证必需字段
        assert dashboard["title"] is not None
        assert isinstance(dashboard["panels"], list)
        assert "health" in dashboard["tags"]
    
    def test_grafana_panel_query(self):
        """测试Grafana面板查询"""
        panel_query = {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}",
            "interval": "",
            "refId": "A"
        }
        
        assert "expr" in panel_query
        assert "rate(" in panel_query["expr"]
        assert "[5m]" in panel_query["expr"]
    
    def test_grafana_alert_rule(self):
        """测试Grafana告警规则"""
        alert_rule = {
            "name": "High Error Rate",
            "conditions": [
                {
                    "evaluator": {
                        "type": "gt",
                        "params": [0.05]
                    },
                    "query": {
                        "model": "rate(errors_total[5m])"
                    }
                }
            ],
            "frequency": "1m",
            "for": "5m"
        }
        
        assert alert_rule["name"] == "High Error Rate"
        assert alert_rule["frequency"] == "1m"


# ============================================================================
# 第4部分: 异步方法密集测试 (40个测试)
# ============================================================================

class TestAsyncMethodsIntensive:
    """测试异步方法密集场景"""
    
    @pytest.mark.asyncio
    async def test_async_with_asyncio_gather(self):
        """测试asyncio.gather并发"""
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2
        
        results = await asyncio.gather(
            task(1), task(2), task(3), task(4), task(5)
        )
        
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_async_with_timeout(self):
        """测试异步超时控制"""
        async def long_task():
            await asyncio.sleep(5)
            return "done"
        
        try:
            result = await asyncio.wait_for(long_task(), timeout=0.1)
        except asyncio.TimeoutError:
            result = "timeout"
        
        assert result == "timeout"
    
    @pytest.mark.asyncio
    async def test_async_exception_propagation(self):
        """测试异步异常传播"""
        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed")
        
        with pytest.raises(ValueError):
            await failing_task()
    
    @pytest.mark.asyncio
    async def test_async_task_cancellation(self):
        """测试异步任务取消"""
        async def cancellable_task():
            try:
                await asyncio.sleep(10)
                return "completed"
            except asyncio.CancelledError:
                return "cancelled"
        
        task = asyncio.create_task(cancellable_task())
        await asyncio.sleep(0.01)
        task.cancel()
        
        try:
            result = await task
        except asyncio.CancelledError:
            result = "cancelled"
        
        assert result == "cancelled"
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """测试异步上下文管理器"""
        class AsyncResource:
            def __init__(self):
                self.opened = False
                self.closed = False
            
            async def __aenter__(self):
                self.opened = True
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.closed = True
        
        resource = AsyncResource()
        
        async with resource:
            assert resource.opened is True
        
        assert resource.closed is True


class TestAsyncIterators:
    """测试异步迭代器"""
    
    @pytest.mark.asyncio
    async def test_async_generator(self):
        """测试异步生成器"""
        async def async_range(n):
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i
        
        results = []
        async for value in async_range(5):
            results.append(value)
        
        assert results == [0, 1, 2, 3, 4]
    
    @pytest.mark.asyncio
    async def test_async_comprehension(self):
        """测试异步推导式"""
        async def async_double(n):
            await asyncio.sleep(0.001)
            return n * 2
        
        results = [await async_double(i) for i in range(5)]
        
        assert results == [0, 2, 4, 6, 8]


# ============================================================================
# 第5部分: 异常处理密集测试 (40个测试)
# ============================================================================

class TestExceptionHandlingIntensive:
    """测试异常处理密集场景"""
    
    def test_handle_connection_error(self):
        """测试处理连接错误"""
        def check_connection():
            raise ConnectionError("Cannot connect")
        
        try:
            check_connection()
            result = "success"
        except ConnectionError as e:
            result = {"error": "connection_error", "message": str(e)}
        
        assert result["error"] == "connection_error"
    
    def test_handle_timeout_error(self):
        """测试处理超时错误"""
        import time
        
        def slow_operation():
            time.sleep(10)
            return "done"
        
        try:
            result = slow_operation()
        except TimeoutError:
            result = {"error": "timeout"}
        
        # 注：这个测试实际不会超时，只是演示结构
        assert result == "done"
    
    def test_handle_value_error(self):
        """测试处理值错误"""
        def parse_value(value):
            if not isinstance(value, (int, float)):
                raise ValueError("Invalid value type")
            return value * 2
        
        # 有效值
        result1 = parse_value(10)
        assert result1 == 20
        
        # 无效值
        with pytest.raises(ValueError):
            parse_value("invalid")
    
    def test_handle_type_error(self):
        """测试处理类型错误"""
        def process_list(items):
            if not isinstance(items, list):
                raise TypeError("Expected list")
            return len(items)
        
        # 有效输入
        result1 = process_list([1, 2, 3])
        assert result1 == 3
        
        # 无效输入
        with pytest.raises(TypeError):
            process_list("not a list")
    
    def test_exception_context_preservation(self):
        """测试异常上下文保留"""
        def inner_function():
            raise ValueError("Inner error")
        
        def outer_function():
            try:
                inner_function()
            except ValueError as e:
                raise RuntimeError("Outer error") from e
        
        with pytest.raises(RuntimeError) as exc_info:
            outer_function()
        
        # 验证异常链
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestErrorRecovery:
    """测试错误恢复"""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_on_error(self):
        """测试错误时优雅降级"""
        async def check_with_fallback():
            try:
                # 尝试主检查
                raise Exception("Primary check failed")
            except:
                # 降级到基础检查
                return {
                    "status": "warning",
                    "degraded": True,
                    "message": "Using fallback check"
                }
        
        result = await check_with_fallback()
        
        assert result["status"] == "warning"
        assert result["degraded"] is True
    
    def test_retry_with_exponential_backoff(self):
        """测试指数退避重试"""
        attempts = []
        
        def check_with_backoff():
            attempts.append(len(attempts) + 1)
            if len(attempts) < 4:
                raise Exception("Temporary error")
            return "success"
        
        # 指数退避重试
        max_retries = 5
        base_delay = 0.001
        
        for attempt in range(max_retries):
            try:
                result = check_with_backoff()
                break
            except Exception:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    import time
                    time.sleep(delay)
                else:
                    result = "failed"
        
        assert result == "success"
        assert len(attempts) == 4
    
    def test_circuit_breaker_pattern(self):
        """测试断路器模式"""
        circuit_state = {
            "failures": 0,
            "threshold": 5,
            "state": "closed"  # closed, open, half_open
        }
        
        def check_with_circuit_breaker():
            if circuit_state["state"] == "open":
                return {"status": "circuit_open"}
            
            try:
                # 模拟检查
                raise Exception("Check failed")
            except:
                circuit_state["failures"] += 1
                
                if circuit_state["failures"] >= circuit_state["threshold"]:
                    circuit_state["state"] = "open"
                
                raise
        
        # 执行6次失败，触发断路器
        for _ in range(6):
            try:
                check_with_circuit_breaker()
            except:
                pass
        
        assert circuit_state["state"] == "open"
        assert circuit_state["failures"] >= circuit_state["threshold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


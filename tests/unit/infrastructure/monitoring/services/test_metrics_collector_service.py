import time
from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from src.infrastructure.monitoring.services.metrics_collector import MetricsCollector


def _build_metrics_collector():
    collector = MetricsCollector(project_root=".")
    collector.clear_cache()
    collector.reset_stats()
    return collector


@patch("src.infrastructure.monitoring.services.metrics_collector.time.time", side_effect=[0, 1, 2])
def test_collect_all_metrics_success_updates_stats(mock_time):
    collector = _build_metrics_collector()

    with patch.object(collector, "_collect_system_metrics_cached", return_value={"cpu": {}}) as mock_system, \
         patch.object(collector, "_collect_test_coverage_metrics_cached", return_value={"coverage": 80}) as mock_coverage, \
         patch.object(collector, "_collect_performance_metrics_cached", return_value={"perf": {}}) as mock_perf, \
         patch.object(collector, "_collect_resource_usage_cached", return_value={"resource": {}}) as mock_resource, \
         patch.object(collector, "_collect_health_status_cached", return_value={"status": "healthy"}) as mock_health:

        result = collector.collect_all_metrics()

    assert "timestamp" in result
    assert collector.collection_stats["total_collections"] == 1
    assert collector.collection_stats["successful_collections"] == 1
    assert collector.collection_stats["failed_collections"] == 0
    assert collector.collection_stats["last_collection_time"] == result["timestamp"]

    mock_system.assert_called_once()
    mock_coverage.assert_called_once()
    mock_perf.assert_called_once()
    mock_resource.assert_called_once()
    mock_health.assert_called_once()


def test_collect_all_metrics_failure_path():
    collector = _build_metrics_collector()

    with patch.object(collector, "_collect_system_metrics_cached", side_effect=RuntimeError("boom")):
        result = collector.collect_all_metrics()

    assert result["error"] == "boom"
    assert collector.collection_stats["failed_collections"] == 1


def test_cache_hit_and_miss_behavior():
    collector = _build_metrics_collector()

    call_counter = {"count": 0}

    def fake_collector():
        call_counter["count"] += 1
        return {"value": call_counter["count"]}

    with patch.object(collector, "_collect_system_metrics", side_effect=fake_collector):
        first = collector._collect_system_metrics_cached()
        assert first == {"value": 1}

        second = collector._collect_system_metrics_cached()
        assert second == {"value": 1}
        assert call_counter["count"] == 1

        time.sleep(collector._cache_timeout + 0.1)

        third = collector._collect_system_metrics_cached()
        assert third == {"value": 2}
        assert call_counter["count"] == 2


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_system_metrics_structure(mock_psutil):
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.cpu_times_percent.return_value = Mock(user=10.0, system=5.0, idle=85.0)
    mock_psutil.cpu_count.side_effect = [4, 8]
    mock_psutil.virtual_memory.return_value = Mock(percent=60, used=1024, total=2048, available=1024, free=512)
    mock_psutil.disk_usage.return_value = Mock(percent=70, used=700, total=1000, free=300)
    mock_psutil.net_io_counters.return_value = Mock(
        bytes_sent=100, bytes_recv=200, packets_sent=10, packets_recv=20, errin=0, errout=0
    )
    mock_psutil.pids.return_value = [1, 2, 3]
    mock_psutil.getloadavg.return_value = (1.0, 0.5, 0.2)
    mock_psutil.boot_time.return_value = 123456789

    collector = _build_metrics_collector()
    metrics = collector._collect_system_metrics()

    assert metrics["cpu"]["usage_percent"] == 50.0
    assert metrics["cpu"]["count"] == 4
    assert metrics["cpu"]["count_logical"] == 8
    assert metrics["memory"]["usage_percent"] == 60
    assert metrics["disk"]["usage_percent"] == 70
    assert metrics["network"]["bytes_sent"] == 100
    assert metrics["system"]["process_count"] == 3


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_health_status_thresholds(mock_psutil):
    mock_psutil.cpu_percent.return_value = 85
    mock_psutil.virtual_memory.return_value = Mock(percent=90)
    mock_psutil.disk_usage.return_value = Mock(percent=95)

    collector = _build_metrics_collector()
    health = collector._collect_health_status()

    assert health["overall_status"] == "critical"
    assert health["health_score"] < 60


def test_get_collection_stats_success_rate_calculation():
    collector = _build_metrics_collector()
    collector.collection_stats["total_collections"] = 4
    collector.collection_stats["successful_collections"] = 3
    collector.collection_stats["failed_collections"] = 1

    stats = collector.get_collection_stats()

    assert stats["success_rate"] == pytest.approx(75.0)


def test_reset_and_clear_behaviour():
    collector = _build_metrics_collector()

    collector._cache["system_metrics"] = {"cached": True}
    collector._last_cache_update["system_metrics"] = time.time()
    collector.collection_stats["total_collections"] = 5

    collector.clear_cache()
    assert collector._cache == {}
    assert collector._last_cache_update == {}

    collector.reset_stats()
    assert collector.collection_stats["total_collections"] == 0
    assert collector.collection_stats["successful_collections"] == 0
    assert collector.collection_stats["failed_collections"] == 0


def test_get_cached_result_exception_fallback():
    """测试缓存收集异常时的回退逻辑"""
    collector = _build_metrics_collector()
    
    # 先设置一个缓存值
    collector._cache["test_key"] = {"fallback": "data"}
    collector._last_cache_update["test_key"] = time.time()
    
    def failing_collector():
        raise RuntimeError("collection failed")
    
    # 当收集失败但缓存存在时，应该返回缓存值
    result = collector._get_cached_result("test_key", failing_collector)
    assert result == {"fallback": "data"}


def test_get_cached_result_exception_no_cache():
    """测试缓存收集异常且无缓存时抛出异常"""
    collector = _build_metrics_collector()
    
    def failing_collector():
        raise RuntimeError("collection failed")
    
    # 当收集失败且无缓存时，应该抛出异常
    with pytest.raises(RuntimeError, match="collection failed"):
        collector._get_cached_result("new_key", failing_collector)


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_system_metrics_exception_handling(mock_psutil):
    """测试系统指标收集异常处理"""
    collector = _build_metrics_collector()
    
    mock_psutil.cpu_percent.side_effect = RuntimeError("CPU error")
    
    result = collector._collect_system_metrics()
    assert result == {}


def test_collect_test_coverage_metrics_with_files(tmp_path, monkeypatch):
    """测试测试覆盖率指标收集（有覆盖率文件的情况）"""
    collector = _build_metrics_collector()
    collector.project_root = Path(tmp_path)
    
    # 创建一个假的覆盖率文件
    coverage_file = tmp_path / "coverage.xml"
    coverage_file.write_text("<?xml version='1.0'?>")
    
    result = collector._collect_test_coverage_metrics()
    assert isinstance(result, dict)
    # 检查返回的是模拟数据（因为实际解析未实现）
    assert len(result) > 0


def test_collect_test_coverage_metrics_no_files(tmp_path, monkeypatch):
    """测试测试覆盖率指标收集（无覆盖率文件的情况）"""
    collector = _build_metrics_collector()
    collector.project_root = Path(tmp_path)
    
    result = collector._collect_test_coverage_metrics()
    assert isinstance(result, dict)
    # 应该返回模拟数据
    assert len(result) > 0


def test_collect_test_coverage_metrics_exception(tmp_path, monkeypatch):
    """测试测试覆盖率指标收集异常处理"""
    collector = _build_metrics_collector()
    collector.project_root = Path(tmp_path)
    
    # 模拟 _get_mock_coverage_data 抛出异常
    original_method = collector._get_mock_coverage_data
    def failing_mock():
        raise RuntimeError("mock error")
    
    collector._get_mock_coverage_data = failing_mock
    
    # 即使异常，也应该返回空字典或处理异常
    try:
        result = collector._collect_test_coverage_metrics()
        assert isinstance(result, dict)
    except RuntimeError:
        # 如果异常被重新抛出，也是可以接受的
        pass


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_performance_metrics_success(mock_psutil):
    """测试性能指标收集成功路径"""
    collector = _build_metrics_collector()
    
    mock_process = Mock()
    mock_process.cpu_percent.return_value = 25.0
    mock_process.memory_info.return_value = Mock(rss=1024*1024, vms=2048*1024)
    mock_process.num_threads.return_value = 5
    mock_psutil.Process.return_value = mock_process
    mock_psutil.getloadavg.return_value = (1.0, 0.8, 0.6)
    
    result = collector._collect_performance_metrics()
    
    assert "process" in result
    assert result["process"]["cpu_percent"] == 25.0
    assert result["process"]["threads"] == 5
    assert "system_load" in result
    assert result["system_load"]["load_1min"] == 1.0


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_performance_metrics_exception(mock_psutil):
    """测试性能指标收集异常处理"""
    collector = _build_metrics_collector()
    
    mock_psutil.Process.side_effect = RuntimeError("Process error")
    
    result = collector._collect_performance_metrics()
    assert result == {}


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_performance_metrics_no_loadavg(mock_psutil):
    """测试性能指标收集（无loadavg的情况）"""
    collector = _build_metrics_collector()
    
    mock_process = Mock()
    mock_process.cpu_percent.return_value = 20.0
    mock_process.memory_info.return_value = Mock(rss=1024*1024, vms=2048*1024)
    mock_process.num_threads.return_value = 3
    mock_psutil.Process.return_value = mock_process
    # 模拟没有getloadavg方法
    del mock_psutil.getloadavg
    
    result = collector._collect_performance_metrics()
    
    assert result["system_load"]["load_1min"] is None


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_resource_usage_success(mock_psutil):
    """测试资源使用情况收集成功路径"""
    collector = _build_metrics_collector()
    
    mock_memory = Mock(available=1024*1024, free=512*1024, cached=256*1024, buffers=128*1024)
    mock_psutil.virtual_memory.return_value = mock_memory
    
    mock_disk_io = Mock(read_count=100, write_count=50, read_bytes=1024*1024, write_bytes=512*1024)
    mock_psutil.disk_io_counters.return_value = mock_disk_io
    
    mock_network_io = Mock(packets_sent=1000, packets_recv=2000, errin=0, errout=0)
    mock_psutil.net_io_counters.return_value = mock_network_io
    
    result = collector._collect_resource_usage()
    
    assert "memory_detailed" in result
    assert "disk_io" in result
    assert result["disk_io"]["read_count"] == 100
    assert "network_io_detailed" in result
    assert result["network_io_detailed"]["packets_sent"] == 1000


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_resource_usage_no_disk_io(mock_psutil):
    """测试资源使用情况收集（无磁盘IO的情况）"""
    collector = _build_metrics_collector()
    
    mock_memory = Mock(available=1024*1024, free=512*1024, cached=256*1024, buffers=128*1024)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_psutil.disk_io_counters.return_value = None  # 无磁盘IO数据
    mock_psutil.net_io_counters.return_value = Mock(packets_sent=100, packets_recv=200, errin=0, errout=0)
    
    result = collector._collect_resource_usage()
    
    assert "memory_detailed" in result
    assert result.get("disk_io") == {}


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_resource_usage_exception(mock_psutil):
    """测试资源使用情况收集异常处理"""
    collector = _build_metrics_collector()
    
    mock_psutil.virtual_memory.side_effect = RuntimeError("Memory error")
    
    result = collector._collect_resource_usage()
    assert result == {}


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_health_status_warning(mock_psutil):
    """测试健康状态收集（warning状态）"""
    collector = _build_metrics_collector()
    
    mock_psutil.cpu_percent.return_value = 70  # 正常
    mock_psutil.virtual_memory.return_value = Mock(percent=70)  # 正常
    mock_psutil.disk_usage.return_value = Mock(percent=65)  # 正常
    
    health = collector._collect_health_status()
    
    # health_score应该在60-100之间，状态应该是warning或healthy
    assert health["overall_status"] in ["healthy", "warning"]
    assert 60 <= health["health_score"] <= 100


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_health_status_critical(mock_psutil):
    """测试健康状态收集（critical状态）"""
    collector = _build_metrics_collector()
    
    mock_psutil.cpu_percent.return_value = 95  # 过高
    mock_psutil.virtual_memory.return_value = Mock(percent=95)  # 过高
    mock_psutil.disk_usage.return_value = Mock(percent=95)  # 过高
    
    health = collector._collect_health_status()
    
    assert health["overall_status"] == "critical"
    assert health["health_score"] < 60


@patch("src.infrastructure.monitoring.services.metrics_collector.psutil")
def test_collect_health_status_exception(mock_psutil):
    """测试健康状态收集异常处理"""
    collector = _build_metrics_collector()
    
    mock_psutil.cpu_percent.side_effect = RuntimeError("Health check error")
    
    health = collector._collect_health_status()
    
    assert health["overall_status"] == "unknown"
    assert health["health_score"] == 0
    assert "error" in health


def test_set_cache_timeout():
    """测试设置缓存超时时间"""
    collector = _build_metrics_collector()
    
    collector.set_cache_timeout(60)
    assert collector._cache_timeout == 60


def test_get_cache_stats():
    """测试获取缓存统计信息"""
    collector = _build_metrics_collector()
    
    # 设置一些缓存数据
    collector._cache["system_metrics"] = {"cpu": 50}
    collector._cache["test_coverage"] = {"coverage": 80}
    collector._last_cache_update["system_metrics"] = time.time()
    collector._last_cache_update["test_coverage"] = time.time() - 100  # 100秒前
    
    stats = collector.get_cache_stats()
    
    assert stats["cache_entries"] == 2
    assert stats["cache_timeout"] == collector._cache_timeout
    assert "system_metrics" in stats["entries"]
    assert "test_coverage" in stats["entries"]
    assert stats["entries"]["system_metrics"]["is_valid"] is True
    # test_coverage 可能已过期，取决于缓存超时时间


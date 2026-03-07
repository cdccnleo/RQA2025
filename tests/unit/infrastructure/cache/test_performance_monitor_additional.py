from src.infrastructure.cache.monitoring.performance_monitor import PerformanceMonitor


def test_performance_monitor_records_hits_and_latency():
    monitor = PerformanceMonitor()
    monitor.record_hit("cache1")
    monitor.record_hit("cache1")
    monitor.record_miss("cache1")
    monitor.record_operation_time("get", 0.002)
    monitor.record_operation_time("get", 0.004)
    monitor.record_operation_time("set", 0.006)

    assert monitor.get_hit_rate("cache1") == 2 / 3
    assert monitor.get_hit_rate() == 2 / 3
    assert monitor.get_average_latency("get") == 0.003
    stats = monitor.get_statistics("cache1")
    assert stats["hits"] == 2 and stats["misses"] == 1


def test_performance_monitor_operation_duration():
    monitor = PerformanceMonitor()
    monitor.start_operation("refresh")
    duration = monitor.end_operation("refresh")
    assert duration >= 0.0
    assert monitor.get_operation_duration("refresh") >= 0.0


def test_performance_monitor_global_stats_and_validation():
    monitor = PerformanceMonitor()
    monitor.record_metric("cache1", 99)
    monitor.record_hit("cache1")
    monitor.record_miss("cache2")
    monitor.record_operation_time("get", "bad-input")
    monitor.record_operation_time("get", -1)

    all_stats = monitor.get_statistics()
    assert set(all_stats.keys()) == {"cache1", "cache2"}
    assert monitor.get_hit_rate("cache2") == 0.0
    assert monitor.get_average_latency("get") == 0.0
    assert monitor.metrics["cache1"] == 99


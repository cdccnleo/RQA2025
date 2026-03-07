import json

import pytest

from src.infrastructure.distributed.performance_monitor import PerformanceMonitor


class _DummyMemory:
    percent = 68.5
    used = 512 * 1024 * 1024  # 512 MB


class _DummyDisk:
    percent = 42.0


class _DummyNet:
    bytes_sent = 1024
    bytes_recv = 2048


@pytest.fixture
def patched_psutil(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.distributed.performance_monitor.psutil.cpu_percent",
        lambda interval=None: 37.5,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.performance_monitor.psutil.virtual_memory",
        lambda: _DummyMemory,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.performance_monitor.psutil.disk_usage",
        lambda path="/": _DummyDisk,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.performance_monitor.psutil.net_io_counters",
        lambda: _DummyNet,
    )


def test_performance_monitor_end_to_end(patched_psutil):
    monitor = PerformanceMonitor(analysis_interval=3600)

    # 生成足够的数据点以覆盖统计与趋势预测逻辑
    for idx in range(12):
        duration = 1.5 if idx % 2 == 0 else 0.4
        monitor.record_metric(
            "order.processing",
            duration,
            tags={"stage": "checkout"},
            metadata={"attempt": idx},
            is_error=(idx % 5 == 0),
        )

    stats = monitor.get_stats("order.processing")["order.processing"]
    assert stats["count"] == 12
    assert stats["max_time"] >= 1.5

    system_perf = monitor.get_system_performance()
    assert system_perf["cpu_usage"] == pytest.approx(37.5)

    bottlenecks = monitor.identify_bottlenecks()
    assert bottlenecks and bottlenecks[0]["issues"]

    trend = monitor.predict_trends("order.processing")
    assert "trend" in trend

    json_payload = monitor.export_metrics("json")
    parsed = json.loads(json_payload)
    assert "stats" in parsed

    csv_payload = monitor.export_metrics("csv")
    assert "order.processing" in csv_payload

    prometheus_payload = monitor.export_metrics("prometheus")
    assert "order.processing_count" in prometheus_payload


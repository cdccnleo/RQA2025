import pytest

from src.infrastructure.distributed.distributed_monitoring import (
    AlertLevel,
    AlertRule,
    DistributedMonitoring,
    DistributedMonitoringManager,
)


class _MiniMemory:
    percent = 55.0
    used = 256 * 1024 * 1024


class _MiniDisk:
    percent = 33.0


class _MiniNet:
    bytes_sent = 4096
    bytes_recv = 8192


@pytest.fixture
def patched_monitoring_psutil(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.distributed.distributed_monitoring.psutil.cpu_percent",
        lambda interval=None: 48.0,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.distributed_monitoring.psutil.virtual_memory",
        lambda: _MiniMemory,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.distributed_monitoring.psutil.disk_usage",
        lambda path="/": _MiniDisk,
    )
    monkeypatch.setattr(
        "src.infrastructure.distributed.distributed_monitoring.psutil.net_io_counters",
        lambda: _MiniNet,
    )


def test_distributed_monitoring_manager_core(patched_monitoring_psutil):
    manager = DistributedMonitoringManager({"collection_interval": 5})

    manager.record_metric("latency", 120.0)

    metric_value = manager.get_metric("latency")
    assert metric_value == pytest.approx(120.0)

    manager.add_alert_rule(
        AlertRule(
            name="latency_alert",
            condition="value > 100",
            threshold=100.0,
            level=AlertLevel.CRITICAL,
            message="Latency above threshold",
        )
    )
    assert manager.get_alert_rules()

    system_metrics = manager.system_monitor.collect_system_metrics()
    manager.collect_system_metrics()
    assert system_metrics and system_metrics[0].name.startswith("system.")

    node_status = manager.node_status_manager.get_node_status()
    assert node_status["status"] in {"healthy", "error"}


def test_distributed_monitoring_facade(patched_monitoring_psutil):
    monitoring = DistributedMonitoring()

    collected = monitoring.collect_metrics()
    assert collected["service"] == "default_service"

    aggregated = monitoring.aggregate_metrics(["svc-blue", "svc-green"])
    assert aggregated["count"] == 2

    monitoring.send_alert("svc-blue", "All good", level=AlertLevel.INFO)
    status = monitoring.get_service_status("svc-blue")
    assert status["alerts"] >= 1


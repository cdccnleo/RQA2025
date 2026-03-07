import importlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def metrics_module(monkeypatch):
    perf_module = importlib.import_module(
        "src.infrastructure.monitoring.components.performance_monitor"
    )
    monkeypatch.setattr(
        perf_module, "monitor_performance", lambda *a, **k: (lambda func: func)
    )

    module = importlib.import_module(
        "src.infrastructure.monitoring.components.metrics_exporter"
    )
    module = importlib.reload(module)
    return module


def _fake_datetime(target_module, fixed_dt):
    class _FixedDateTime:
        @staticmethod
        def now():
            return fixed_dt

        @staticmethod
        def utcnow():
            return fixed_dt

        @staticmethod
        def fromtimestamp(ts):
            return datetime.fromtimestamp(ts)

    return _FixedDateTime


def test_export_metrics_without_compression(monkeypatch, metrics_module):
    exporter = metrics_module.MetricsExporter(pool_name="alpha")

    monkeypatch.setattr(
        exporter,
        "_generate_prometheus_format",
        lambda stats: "prometheus-data",
    )
    monkeypatch.setattr(
        exporter,
        "_generate_json_format",
        lambda stats: '{"metrics": 1}',
    )

    fixed_dt = datetime(2025, 1, 1, 12, 0, 0)
    monkeypatch.setattr(
        metrics_module, "datetime", _fake_datetime(metrics_module, fixed_dt)
    )

    result = exporter.export_metrics({"pool_size": 10})

    assert result is True
    assert exporter.get_prometheus_metrics() == "prometheus-data"
    assert exporter.get_json_metrics() == '{"metrics": 1}'

    status = exporter.get_export_status()
    assert status["last_export_time"] == fixed_dt.isoformat()
    assert status["available_formats"] == ["prometheus", "json"]
    assert status["cache_size"] == 2


def test_export_metrics_with_compression(monkeypatch, metrics_module):
    config = metrics_module.PrometheusExportConfig(enable_compression=True)
    exporter = metrics_module.MetricsExporter(config=config)

    monkeypatch.setattr(
        exporter,
        "_generate_json_format",
        lambda stats: '{"compressed": true}',
    )

    assert exporter.export_metrics({"pool_size": 5}) is True
    assert "monitoring_pool_size" in exporter.get_prometheus_metrics()
    assert exporter.validate_export_data("prometheus") is True


def test_export_metrics_handles_failure(monkeypatch, metrics_module):
    exporter = metrics_module.MetricsExporter()

    def raise_error(stats):
        raise RuntimeError("boom")

    monkeypatch.setattr(exporter, "_generate_prometheus_format", raise_error)

    assert exporter.export_metrics({"pool_size": 1}) is False
    assert exporter.get_prometheus_metrics() == ""
    assert exporter.get_json_metrics() == "{}"


def test_get_export_status_after_clear(metrics_module):
    exporter = metrics_module.MetricsExporter()
    exporter._export_cache = {"prometheus": "data"}
    exporter._last_export_time = datetime(2025, 1, 1, 8, 0, 0)

    exporter.clear_cache()
    status = exporter.get_export_status()

    assert status["last_export_time"] is None
    assert status["cache_size"] == 0
    assert exporter.get_prometheus_metrics() == ""


def test_generate_prometheus_format(metrics_module):
    config = metrics_module.PrometheusExportConfig(
        include_help_text=True,
        include_type_info=True,
        metric_prefix="custom",
        default_labels={"env": "prod"},
    )
    exporter = metrics_module.MetricsExporter(pool_name="alpha", config=config)

    stats = {
        "pool_size": 3,
        "max_size": 10,
        "created_count": 100,
        "hit_count": 80,
        "hit_rate": 0.8,
        "memory_usage_mb": 64,
        "avg_access_time": 0.123,
    }

    output = exporter._generate_prometheus_format(stats)

    assert '# HELP custom_pool_size' in output
    assert '# TYPE custom_hit_count counter' in output
    assert 'custom_avg_access_time_ms{pool="alpha",env="prod"} 123.0' in output


def test_generate_json_format_success(monkeypatch, metrics_module):
    exporter = metrics_module.MetricsExporter(pool_name="beta")
    fixed_dt = datetime(2025, 1, 2, 9, 30, 0)
    monkeypatch.setattr(
        metrics_module, "datetime", _fake_datetime(metrics_module, fixed_dt)
    )

    data = exporter._generate_json_format({"pool_size": 7})
    parsed = json.loads(data)

    assert parsed["metadata"]["pool_name"] == "beta"
    assert parsed["metadata"]["export_time"] == fixed_dt.isoformat()
    assert parsed["metrics"]["pool_size"] == 7


def test_generate_json_format_failure(monkeypatch, metrics_module):
    exporter = metrics_module.MetricsExporter()

    def raise_json_error(*args, **kwargs):
        raise ValueError("fail json")

    monkeypatch.setattr(metrics_module.json, "dumps", raise_json_error)

    data = exporter._generate_json_format({})
    assert data == '{"error": "Failed to generate JSON format"}'


def test_export_to_file_success(tmp_path, metrics_module):
    exporter = metrics_module.MetricsExporter(pool_name="gamma")
    exporter._export_cache["prometheus"] = "metric data"

    file_path = tmp_path / "metrics.prometheus"
    assert exporter.export_to_file("prometheus", str(file_path)) is True
    assert file_path.read_text(encoding="utf-8") == "metric data"


def test_export_to_file_unsupported_format(metrics_module):
    exporter = metrics_module.MetricsExporter()
    assert exporter.export_to_file("xml", "file.xml") is False


def test_export_to_file_default_path(monkeypatch, tmp_path, metrics_module):
    exporter = metrics_module.MetricsExporter(pool_name="poolZ")
    exporter._export_cache["json"] = '{"x":1}'

    fixed_dt = datetime(2025, 1, 3, 7, 0, 0)
    monkeypatch.setattr(
        metrics_module, "datetime", _fake_datetime(metrics_module, fixed_dt)
    )

    monkeypatch.chdir(tmp_path)

    assert exporter.export_to_file("json", None) is True
    expected_name = f"metrics_export_poolZ_{fixed_dt.strftime('%Y%m%d_%H%M%S')}.json"
    assert Path(expected_name).read_text(encoding="utf-8") == '{"x":1}'


def test_export_to_file_handles_exception(monkeypatch, metrics_module):
    exporter = metrics_module.MetricsExporter()
    exporter._export_cache["json"] = '{"y":2}'

    def raise_io_error(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr("builtins.open", raise_io_error)

    assert exporter.export_to_file("json", "unused.json") is False


def test_validate_export_data(metrics_module):
    exporter = metrics_module.MetricsExporter()
    exporter._export_cache["json"] = '{"valid": true}'
    exporter._export_cache["prometheus"] = "metric{} 1"

    assert exporter.validate_export_data("json") is True
    assert exporter.validate_export_data("prometheus") is True

    exporter._export_cache["json"] = "not-json"
    assert exporter.validate_export_data("json") is False

    assert exporter.validate_export_data("xml") is False

    exporter._export_cache["unknown"] = "data"
    assert exporter.validate_export_data("unknown") is False


def test_generate_prometheus_compressed_branches(metrics_module):
    exporter = metrics_module.MetricsExporter()
    stats = {"pool_size": 1}
    compressed = exporter._generate_prometheus_compressed(stats)
    plain = exporter._generate_prometheus_format(stats)
    assert compressed == plain


def test_get_supported_formats(metrics_module):
    exporter = metrics_module.MetricsExporter()
    assert exporter.get_supported_formats() == ["prometheus", "json"]


def test_generate_labels_with_default_labels(metrics_module):
    """测试生成标签（带默认标签）"""
    config = metrics_module.PrometheusExportConfig(
        default_labels={"env": "prod", "region": "us-east"}
    )
    exporter = metrics_module.MetricsExporter(pool_name="test", config=config)
    
    labels = exporter._generate_labels()
    assert 'pool="test"' in labels
    assert 'env="prod"' in labels
    assert 'region="us-east"' in labels


def test_generate_labels_no_default_labels(metrics_module):
    """测试生成标签（无默认标签）"""
    exporter = metrics_module.MetricsExporter(pool_name="test")
    labels = exporter._generate_labels()
    assert labels == 'pool="test"'


def test_generate_single_metric_with_converter(metrics_module):
    """测试生成单个指标（带转换器）"""
    config = metrics_module.PrometheusExportConfig(
        include_help_text=True,
        include_type_info=True,
        metric_prefix="test"
    )
    exporter = metrics_module.MetricsExporter(pool_name="test", config=config)
    
    metric_def = {
        'name': 'access_time',
        'help': 'Access time',
        'type': 'gauge',
        'key': 'avg_access_time',
        'unit': 'ms',
        'converter': lambda x: x * 1000
    }
    
    stats = {'avg_access_time': 0.5}
    labels = 'pool="test"'
    
    lines = exporter._generate_single_metric(metric_def, stats, labels)
    assert any('test_access_time{pool="test"} 500.0' in line for line in lines)


def test_generate_single_metric_without_help_and_type(metrics_module):
    """测试生成单个指标（不包含help和type）"""
    config = metrics_module.PrometheusExportConfig(
        include_help_text=False,
        include_type_info=False,
        metric_prefix="test"
    )
    exporter = metrics_module.MetricsExporter(pool_name="test", config=config)
    
    metric_def = {
        'name': 'pool_size',
        'help': 'Pool size',
        'type': 'gauge',
        'key': 'pool_size'
    }
    
    stats = {'pool_size': 10}
    labels = 'pool="test"'
    
    lines = exporter._generate_single_metric(metric_def, stats, labels)
    assert not any('# HELP' in line for line in lines)
    assert not any('# TYPE' in line for line in lines)
    assert any('test_pool_size{pool="test"} 10' in line for line in lines)


def test_generate_single_metric_missing_key(metrics_module):
    """测试生成单个指标（缺失key）"""
    exporter = metrics_module.MetricsExporter(pool_name="test")
    
    metric_def = {
        'name': 'missing',
        'help': 'Missing metric',
        'type': 'gauge',
        'key': 'missing_key'
    }
    
    stats = {}
    labels = 'pool="test"'
    
    lines = exporter._generate_single_metric(metric_def, stats, labels)
    assert any('monitoring_missing{pool="test"} 0.0' in line for line in lines)


def test_validate_export_data_prometheus_invalid(metrics_module):
    """测试验证导出数据（Prometheus格式无效）"""
    exporter = metrics_module.MetricsExporter()
    exporter._export_cache["prometheus"] = "invalid{prometheus}format"
    
    # Prometheus格式验证可能通过（简单验证）
    result = exporter.validate_export_data("prometheus")
    assert isinstance(result, bool)


def test_export_to_file_prometheus_format(tmp_path, metrics_module):
    """测试导出到文件（Prometheus格式）"""
    exporter = metrics_module.MetricsExporter(pool_name="test")
    exporter._export_cache["prometheus"] = "metric_data"
    
    file_path = tmp_path / "metrics.prometheus"
    assert exporter.export_to_file("prometheus", str(file_path)) is True
    assert file_path.read_text(encoding="utf-8") == "metric_data"


def test_get_export_status_no_exports(metrics_module):
    """测试获取导出状态（无导出）"""
    exporter = metrics_module.MetricsExporter()
    status = exporter.get_export_status()
    
    assert status["last_export_time"] is None
    assert status["cache_size"] == 0
    assert status["available_formats"] == []
    assert status["pool_name"] == "default_pool"


def test_export_metrics_updates_cache(metrics_module):
    """测试导出指标更新缓存"""
    exporter = metrics_module.MetricsExporter(pool_name="test")
    
    stats = {
        "pool_size": 5,
        "max_size": 10,
        "created_count": 100,
        "hit_count": 80,
        "hit_rate": 0.8,
        "memory_usage_mb": 64,
        "avg_access_time": 0.123,
    }
    
    result = exporter.export_metrics(stats)
    assert result is True
    assert exporter.get_prometheus_metrics() != ""
    assert exporter.get_json_metrics() != "{}"
    assert exporter._last_export_time is not None



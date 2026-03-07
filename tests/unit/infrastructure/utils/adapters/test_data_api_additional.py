import pytest
from fastapi import HTTPException

from src.infrastructure.utils.adapters import data_api


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """为每个测试重置 data_api 模块的全局状态。"""
    monkeypatch.setattr(data_api, "loaders", {}, raising=False)
    monkeypatch.setattr(data_api, "performance_monitor", None, raising=False)
    monkeypatch.setattr(data_api, "advanced_quality_monitor", None, raising=False)
    monkeypatch.setattr(data_api, "data_manager", None, raising=False)
    yield


class DummyLoader:
    def __init__(self, payload=None):
        self.payload = payload or [{"value": 1}]
        self.calls = []

    async def load_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class DummyPerformanceMonitor:
    def __init__(self):
        self.recorded = []

    def record_load_time(self, value):
        self.recorded.append(value)

    def get_metrics(self):
        return {
            "cache_hit_rate": 0.8,
            "avg_load_time": 0.12,
            "memory_usage": 0.45,
            "error_rate": 0.01,
        }

    def get_alerts(self):
        return ["perf-warning"]


class ExplodingPerformanceMonitor(DummyPerformanceMonitor):
    def record_load_time(self, value):
        raise RuntimeError("monitor explode")

    def get_metrics(self):
        raise RuntimeError("metrics explode")


class DummyQualityMonitor:
    def evaluate_data_quality(self, data):
        return {
            "completeness": 0.9,
            "accuracy": 0.8,
            "consistency": 0.7,
            "timeliness": 0.6,
            "validity": 0.5,
            "reliability": 0.4,
            "uniqueness": 0.3,
            "integrity": 0.2,
            "precision": 0.1,
            "availability": 0.95,
        }

    def generate_quality_report(self, days, source_type=None):
        return {"days": days, "source": source_type or "all"}

    def get_alerts(self):
        return ["quality-warning"]

    def get_current_metrics(self):
        return {"quality_score": 0.88}


class DummyDataManager:
    def __init__(self):
        self.cleared = False

    def get_cache_statistics(self):
        return {"entries": 5}

    def clear_cache(self):
        self.cleared = True


class ExplodingDataManager(DummyDataManager):
    def get_cache_statistics(self):
        raise RuntimeError("cache explode")

    def clear_cache(self):
        raise RuntimeError("clear explode")


class ExplodingQualityMonitor(DummyQualityMonitor):
    def generate_quality_report(self, days, source_type=None):
        raise RuntimeError("report explode")

    def get_alerts(self):
        raise RuntimeError("quality alerts explode")

    def get_current_metrics(self):
        raise RuntimeError("quality metrics explode")


@pytest.mark.asyncio
async def test_health_check_returns_basic_status():
    result = await data_api.health_check()
    assert result["status"] == "healthy"
    assert result["available_sources"] == []


@pytest.mark.asyncio
async def test_health_check_handles_internal_error(monkeypatch):
    class FailingLoaders:
        def keys(self):
            raise RuntimeError("keys explode")

    monkeypatch.setattr(data_api, "loaders", FailingLoaders(), raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await data_api.health_check()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_list_data_sources_includes_registered_loader(monkeypatch):
    loader = DummyLoader()
    monkeypatch.setitem(data_api.loaders, "demo", loader)

    result = await data_api.list_data_sources()
    assert result["total"] == 1
    assert result["sources"][0]["name"] == "demo"


@pytest.mark.asyncio
async def test_list_data_sources_handles_exception(monkeypatch):
    class FailingLoaders:
        def items(self):
            raise RuntimeError("items explode")

    monkeypatch.setattr(data_api, "loaders", FailingLoaders(), raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await data_api.list_data_sources()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_get_data_source_info_not_found():
    with pytest.raises(HTTPException) as exc_info:
        await data_api.get_data_source_info("missing")
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_NOT_FOUND


@pytest.mark.asyncio
async def test_get_data_source_info_success(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())
    info = await data_api.get_data_source_info("demo")
    assert info["name"] == "demo"
    assert info["status"] == "active"


@pytest.mark.asyncio
async def test_load_data_records_performance(monkeypatch):
    loader = DummyLoader()
    monitor = DummyPerformanceMonitor()
    monkeypatch.setitem(data_api.loaders, "demo", loader)
    monkeypatch.setattr(data_api, "performance_monitor", monitor, raising=False)

    request = data_api.DataSourceRequest(source="demo", symbol="BTC")
    result = await data_api.load_data(request)

    assert result["status"] == "success"
    assert loader.calls, "加载器应在调用过程中被触发"
    assert monitor.recorded, "性能监控应记录加载时间"


@pytest.mark.asyncio
async def test_load_data_requires_loader_support(monkeypatch):
    class LoaderWithoutMethod:
        pass

    monkeypatch.setitem(data_api.loaders, "demo", LoaderWithoutMethod())
    request = data_api.DataSourceRequest(source="demo")
    with pytest.raises(HTTPException) as exc_info:
        await data_api.load_data(request)
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_load_data_records_monitor_failure(monkeypatch):
    loader = DummyLoader()
    monitor = ExplodingPerformanceMonitor()
    monkeypatch.setitem(data_api.loaders, "demo", loader)
    monkeypatch.setattr(data_api, "performance_monitor", monitor, raising=False)
    request = data_api.DataSourceRequest(source="demo")

    result = await data_api.load_data(request)
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_get_performance_metrics_returns_defaults_when_monitor_missing():
    metrics = await data_api.get_performance_metrics()
    assert metrics["cache_hit_rate"] == 0.0
    assert metrics["load_time"] == 0.0


@pytest.mark.asyncio
async def test_get_performance_metrics_handles_monitor_error(monkeypatch):
    monkeypatch.setattr(data_api, "performance_monitor", ExplodingPerformanceMonitor(), raising=False)
    metrics = await data_api.get_performance_metrics()
    assert metrics["cache_hit_rate"] == 0.0
    assert metrics["memory_usage"] == 0.0


@pytest.mark.asyncio
async def test_check_data_quality_requires_quality_monitor(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())
    request = data_api.DataQualityRequest(source_type="demo", symbol="ETH", metrics=["accuracy"])

    with pytest.raises(HTTPException) as exc_info:
        await data_api.check_data_quality(request)
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_check_data_quality_success(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())
    monkeypatch.setattr(data_api, "advanced_quality_monitor", DummyQualityMonitor(), raising=False)

    request = data_api.DataQualityRequest(source_type="demo", symbol="ETH", metrics=["accuracy"])
    result = await data_api.check_data_quality(request)

    assert pytest.approx(result["completeness"]) == 0.9
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_generate_quality_report_requires_monitor():
    with pytest.raises(HTTPException) as exc_info:
        await data_api.generate_quality_report()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_generate_quality_report_success(monkeypatch):
    monkeypatch.setattr(data_api, "advanced_quality_monitor", DummyQualityMonitor(), raising=False)

    result = await data_api.generate_quality_report(days=3, source_type="demo")
    assert result["report"]["days"] == 3
    assert result["report"]["source"] == "demo"


@pytest.mark.asyncio
async def test_clear_cache_requires_manager():
    with pytest.raises(HTTPException) as exc_info:
        await data_api.clear_cache()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_clear_cache_success(monkeypatch):
    manager = DummyDataManager()
    monkeypatch.setattr(data_api, "data_manager", manager, raising=False)

    result = await data_api.clear_cache()
    assert result["status"] == "success"
    assert manager.cleared is True


@pytest.mark.asyncio
async def test_generate_quality_report_handles_monitor_error(monkeypatch):
    monkeypatch.setattr(data_api, "advanced_quality_monitor", ExplodingQualityMonitor(), raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await data_api.generate_quality_report()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_get_cache_statistics_handles_manager_error(monkeypatch):
    monkeypatch.setattr(data_api, "data_manager", ExplodingDataManager(), raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await data_api.get_cache_statistics()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_clear_cache_handles_manager_error(monkeypatch):
    monkeypatch.setattr(data_api, "data_manager", ExplodingDataManager(), raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await data_api.clear_cache()
    assert exc_info.value.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_get_alerts_returns_empty_when_monitors_missing():
    result = await data_api.get_alerts()
    assert result["performance_alerts"] == []
    assert result["quality_alerts"] == []


@pytest.mark.asyncio
async def test_get_alerts_collects_monitor_results(monkeypatch):
    monkeypatch.setattr(data_api, "performance_monitor", DummyPerformanceMonitor(), raising=False)
    monkeypatch.setattr(data_api, "advanced_quality_monitor", DummyQualityMonitor(), raising=False)

    result = await data_api.get_alerts()
    assert "perf-warning" in result["performance_alerts"]
    assert "quality-warning" in result["quality_alerts"]


@pytest.mark.asyncio
async def test_get_dashboard_metrics_uses_defaults(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())

    dashboard = await data_api.get_dashboard_metrics()
    assert "performance" in dashboard and dashboard["performance"] == {}
    assert "quality" in dashboard and dashboard["quality"] == {}
    assert "sources" in dashboard and "demo" in dashboard["sources"]


@pytest.mark.asyncio
async def test_get_dashboard_metrics_with_monitors(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())
    monkeypatch.setattr(data_api, "performance_monitor", DummyPerformanceMonitor(), raising=False)
    monkeypatch.setattr(data_api, "advanced_quality_monitor", DummyQualityMonitor(), raising=False)

    dashboard = await data_api.get_dashboard_metrics()
    assert dashboard["performance"]["cache_hit_rate"] == 0.8
    assert dashboard["quality"]["quality_score"] == 0.88
    assert dashboard["sources"]["demo"]["status"] == "active"


@pytest.mark.asyncio
async def test_get_alerts_handles_monitor_errors(monkeypatch):
    class PerfMonitor(DummyPerformanceMonitor):
        def get_alerts(self):
            raise RuntimeError("perf alerts explode")

    monkeypatch.setattr(data_api, "performance_monitor", PerfMonitor(), raising=False)
    monkeypatch.setattr(data_api, "advanced_quality_monitor", ExplodingQualityMonitor(), raising=False)

    result = await data_api.get_alerts()
    assert result["performance_alerts"] == []
    assert result["quality_alerts"] == []


@pytest.mark.asyncio
async def test_get_dashboard_metrics_handles_monitor_errors(monkeypatch):
    monkeypatch.setitem(data_api.loaders, "demo", DummyLoader())

    class PerfMonitor(DummyPerformanceMonitor):
        def get_metrics(self):
            raise RuntimeError("perf metrics explode")

    monkeypatch.setattr(data_api, "performance_monitor", PerfMonitor(), raising=False)
    monkeypatch.setattr(data_api, "advanced_quality_monitor", ExplodingQualityMonitor(), raising=False)

    dashboard = await data_api.get_dashboard_metrics()
    assert dashboard["performance"] == {}
    assert dashboard["quality"] == {}


@pytest.mark.asyncio
async def test_create_exception_handler_returns_json_response():
    handler = data_api.create_exception_handler()
    response = await handler(None, RuntimeError("boom"))
    assert response.status_code == data_api.DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR
    assert "boom" in response.body.decode()
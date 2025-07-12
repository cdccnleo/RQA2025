import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from src.infrastructure.monitoring.resource_api import ResourceAPI
from fastapi.testclient import TestClient

# 统一mock prometheus_client的指标对象
@pytest.fixture(autouse=True)
def mock_prometheus():
    with patch('prometheus_client.Counter', MagicMock()), \
         patch('prometheus_client.Gauge', MagicMock()), \
         patch('prometheus_client.Histogram', MagicMock()):
        yield

@pytest.fixture
def mock_managers():
    """模拟资源管理器"""
    resource_mgr = MagicMock()
    gpu_mgr = MagicMock()

    # 模拟系统资源数据
    resource_mgr.get_stats.return_value = [{
        "timestamp": "2023-07-20T10:00:00",
        "cpu": {"percent": 30.5},
        "memory": {"percent": 45.2},
        "disk": {"percent": 25.1}
    }]

    resource_mgr.get_summary.return_value = {
        "cpu": {"avg": 25.3, "max": 50.0},
        "memory": {"avg": 40.2, "max": 60.0},
        "disk": {"avg": 20.1, "max": 30.0}
    }

    # 模拟GPU数据
    gpu_mgr.has_gpu = True
    gpu_mgr.get_stats.return_value = [{
        "timestamp": "2023-07-20T10:00:00",
        "gpus": [{
            "index": 0,
            "name": "NVIDIA RTX 3090",
            "memory": {
                "allocated": 1024 * 1024 * 1024,  # 1GB
                "total": 24 * 1024 * 1024 * 1024,  # 24GB
                "percent": 4.17
            },
            "utilization": 15.5
        }]
    }]

    gpu_mgr.get_summary.return_value = {
        "gpus": [{
            "index": 0,
            "name": "NVIDIA RTX 3090",
            "memory": {
                "avg": 5.0,
                "max": 10.0
            },
            "utilization": {
                "avg": 12.0,
                "max": 20.0
            }
        }]
    }

    return resource_mgr, gpu_mgr

@pytest.fixture
def resource_api(mock_managers):
    """资源API测试实例"""
    resource_mgr, gpu_mgr = mock_managers
    return ResourceAPI(resource_mgr, gpu_mgr)

@pytest.fixture
def test_client(resource_api):
    """测试客户端"""
    return TestClient(resource_api.router)

def test_get_system_usage(test_client):
    """测试获取系统资源使用情况"""
    response = test_client.get("/system")
    assert response.status_code == 200
    data = response.json()

    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data
    assert data["cpu"]["current"] == 30.5
    assert data["memory"]["avg"] == 40.2

def test_get_gpu_usage(test_client):
    """测试获取GPU使用情况"""
    response = test_client.get("/gpu")
    assert response.status_code == 200
    data = response.json()

    assert len(data["gpus"]) == 1
    assert data["gpus"][0]["name"] == "NVIDIA RTX 3090"
    assert data["gpus"][0]["memory"]["percent"] == 4.17

def test_get_usage_history(test_client, mock_managers):
    """测试获取资源使用历史数据"""
    resource_mgr, gpu_mgr = mock_managers

    # 模拟历史数据
    history_data = []
    for i in range(60):  # 60分钟数据
        history_data.append({
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "cpu": {"percent": 20 + i % 10},
            "memory": {"percent": 30 + i % 15},
            "disk": {"percent": 10 + i % 5}
        })

    resource_mgr.get_stats.return_value = history_data

    # 测试不同时间范围和分辨率
    for duration in ["1h", "6h", "24h"]:
        for resolution in ["1m", "5m", "15m"]:
            response = test_client.get(
                "/history",
                params={"duration": duration, "resolution": resolution}
            )
            assert response.status_code == 200
            data = response.json()

            assert len(data["system"]) <= 60  # 采样后数据量应减少
            assert data["duration"] == duration
            assert data["resolution"] == resolution

def test_no_gpu_support(test_client, mock_managers):
    """测试无GPU支持的情况"""
    _, gpu_mgr = mock_managers
    gpu_mgr.has_gpu = False

    response = test_client.get("/gpu")
    assert response.status_code == 200
    assert response.json()["gpus"] == []

def test_empty_history(test_client, mock_managers):
    """测试空历史数据情况"""
    resource_mgr, _ = mock_managers
    resource_mgr.get_stats.return_value = []

    response = test_client.get("/history")
    assert response.status_code == 200
    assert response.json()["system"] == []

def test_get_strategy_usage_empty(test_client, mock_managers):
    """测试策略资源为空的情况"""
    resource_mgr, _ = mock_managers
    # 不设置strategy_resources属性
    response = test_client.get("/strategies")
    assert response.status_code == 200
    data = response.json()
    assert data["strategies"] == []

def test_get_strategy_usage_normal(test_client, mock_managers):
    """测试策略资源正常返回"""
    resource_mgr, _ = mock_managers
    resource_mgr.strategy_resources = {
        "alpha": {"workers": [1, 2]},
        "beta": {"workers": [3]}
    }
    resource_mgr.quota_map = {
        "alpha": {"max_workers": 5, "cpu": 2, "gpu_memory": 4096},
        "beta": {"max_workers": 3, "cpu": 1, "gpu_memory": 2048}
    }
    response = test_client.get("/strategies")
    assert response.status_code == 200
    data = response.json()
    assert len(data["strategies"]) == 2
    assert data["strategies"][0]["name"] == "alpha"
    assert data["strategies"][0]["workers"] == 2
    assert data["strategies"][0]["quota"]["max_workers"] == 5
    assert data["strategies"][1]["name"] == "beta"
    assert data["strategies"][1]["workers"] == 1
    assert data["strategies"][1]["quota"]["cpu_limit"] == 1

def test_get_strategy_usage_quota_missing(test_client, mock_managers):
    """测试quota缺失的情况"""
    resource_mgr, _ = mock_managers
    resource_mgr.strategy_resources = {
        "gamma": {"workers": []}
    }
    resource_mgr.quota_map = {}
    response = test_client.get("/strategies")
    assert response.status_code == 200
    data = response.json()
    assert len(data["strategies"]) == 1
    assert data["strategies"][0]["name"] == "gamma"
    assert data["strategies"][0]["quota"]["max_workers"] == 0
    assert data["strategies"][0]["quota"]["cpu_limit"] == 0
    assert data["strategies"][0]["quota"]["gpu_memory_limit"] == 0

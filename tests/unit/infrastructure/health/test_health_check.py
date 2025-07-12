import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.infrastructure.health.health_check import HealthCheck
import psutil
from datetime import datetime, timedelta

@pytest.fixture
def health_check():
    """健康检查服务实例"""
    return HealthCheck()

@pytest.fixture
def test_client(health_check):
    """测试客户端"""
    return TestClient(health_check.router)

def test_health_endpoint(test_client):
    """测试/health端点基本响应"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "system" in data
    assert "dependencies" in data

def test_ready_endpoint(test_client):
    """测试/ready端点响应"""
    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"

def test_health_with_dependencies(health_check, test_client):
    """测试带依赖服务的健康检查"""
    # 添加模拟依赖检查
    mock_check = MagicMock(return_value=True)
    health_check.add_dependency_check("mock_service", mock_check)

    response = test_client.get("/health")
    data = response.json()

    assert len(data["dependencies"]) == 1
    assert data["dependencies"][0]["name"] == "mock_service"
    assert data["dependencies"][0]["status"] == "healthy"

def test_health_with_failing_dependency(health_check, test_client):
    """测试依赖服务失败的健康检查"""
    # 添加模拟失败依赖
    def failing_check():
        raise Exception("Service unavailable")

    health_check.add_dependency_check("failing_service", failing_check)

    response = test_client.get("/health")
    data = response.json()

    assert data["status"] == "degraded"
    assert data["dependencies"][0]["status"] == "error"

@patch('psutil.cpu_percent')
@patch('psutil.virtual_memory')
def test_system_health(mock_mem, mock_cpu, test_client):
    """测试系统健康状态报告"""
    # 设置模拟返回值
    mock_cpu.return_value = 25.5
    mock_mem.return_value = MagicMock(percent=65.2)

    response = test_client.get("/health")
    data = response.json()

    assert data["system"]["cpu"] == "25.5%"
    assert data["system"]["memory"] == "65.2%"
    assert "disk" in data["system"]

def test_health_status_degraded(health_check, test_client):
    """测试降级状态报告"""
    # 添加模拟不健康依赖
    health_check.add_dependency_check("unhealthy_service", lambda: False)

    response = test_client.get("/health")
    data = response.json()

    assert data["status"] == "degraded"

@patch('psutil.virtual_memory')
def test_health_system_error(mock_mem, test_client):
    """测试系统监控错误处理"""
    # 模拟内存检查出错
    mock_mem.side_effect = Exception("Memory check failed")

    response = test_client.get("/health")
    data = response.json()

    assert "error" in data["system"]

def test_dependency_order_preserved(health_check, test_client):
    """测试依赖检查顺序保持"""
    # 添加多个依赖检查
    health_check.add_dependency_check("service1", lambda: True)
    health_check.add_dependency_check("service2", lambda: True)

    response = test_client.get("/health")
    data = response.json()

    assert [d["name"] for d in data["dependencies"]] == ["service1", "service2"]

def test_process_uptime_calculation(test_client):
    """测试进程运行时间计算"""
    with patch('psutil.Process') as mock_proc:
        # 模拟进程启动时间(1小时前)
        mock_proc.return_value.create_time.return_value = (
            datetime.now() - timedelta(hours=1)
        ).timestamp()

        response = test_client.get("/health")
        data = response.json()

        assert "uptime" in data["system"]["process"]
        assert "1:00:00" in data["system"]["process"]["uptime"]

"""
测试调度器监控功能
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.gateway.web.datasource_routes import router
from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler


class TestSchedulerMonitoring:
    """调度器监控测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1/data")
        return TestClient(app)

    def test_scheduler_dashboard_endpoint_exists(self, client):
        """测试调度器监控面板端点存在"""
        response = client.get("/api/v1/data/scheduler/dashboard")

        # 即使调度器没有运行，也应该返回200状态码（返回默认数据）
        assert response.status_code == 200

        data = response.json()
        assert "scheduler" in data
        assert "performance" in data
        assert "recent_activity" in data

    def test_scheduler_control_endpoint_exists(self, client):
        """测试调度器控制端点存在"""
        response = client.post("/api/v1/data/scheduler/control", json={"action": "status"})

        # 即使调度器控制失败，也应该返回200状态码
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "action" in data

    @patch('src.core.orchestration.business_process.service_scheduler.get_data_collection_scheduler')
    def test_scheduler_dashboard_with_mock_scheduler(self, mock_get_scheduler, client):
        """测试调度器监控面板返回正确数据结构"""
        # 创建模拟调度器
        mock_scheduler = MagicMock()
        mock_scheduler.is_running.return_value = True
        mock_scheduler.get_status.return_value = {
            "running": True,
            "startup_time": "2024-01-18T10:00:00",
            "enabled_sources_count": 5,
            "check_interval": 30,
            "last_collection_times": {
                "source1": 1705569600,  # 2024-01-18 10:00:00
                "source2": 1705569660   # 2024-01-18 10:01:00
            }
        }
        mock_scheduler.max_concurrent_tasks = 3
        mock_scheduler.active_tasks = {"task1", "task2"}
        mock_get_scheduler.return_value = mock_scheduler

        # 模拟psutil
        with patch('psutil.cpu_percent', return_value=45.2), \
             patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 67.8

            response = client.get("/api/v1/data/scheduler/dashboard")

            assert response.status_code == 200
            data = response.json()

            # 验证调度器数据
            scheduler_data = data["scheduler"]
            assert scheduler_data["running"] == True
            assert scheduler_data["active_sources"] == 5
            assert scheduler_data["active_tasks"] == 2
            assert scheduler_data["concurrent_limit"] == 3
            assert scheduler_data["last_check"] == 30
            assert "uptime" in scheduler_data

            # 验证性能数据
            performance_data = data["performance"]
            assert performance_data["cpu_usage"] == 45.2
            assert performance_data["memory_usage"] == 67.8

            # 验证最近活动
            assert "recent_activity" in data

    @patch('src.core.orchestration.business_process.service_scheduler.get_data_collection_scheduler')
    def test_scheduler_control_start_stop(self, mock_get_scheduler, client):
        """测试调度器启动和停止控制"""
        # 创建模拟调度器
        mock_scheduler = MagicMock()
        mock_scheduler.is_running.return_value = False
        mock_get_scheduler.return_value = mock_scheduler

        # 测试启动
        with patch('src.core.orchestration.business_process.service_scheduler.start_data_collection_scheduler') as mock_start:
            mock_start.return_value = True

            response = client.post("/api/v1/data/scheduler/control", json={"action": "start"})
            assert response.status_code == 200

            data = response.json()
            assert data["success"] == True
            assert data["action"] == "start"
            mock_start.assert_called_once()

        # 测试停止
        with patch('src.core.orchestration.business_process.service_scheduler.stop_data_collection_scheduler') as mock_stop:
            mock_stop.return_value = True

            response = client.post("/api/v1/data/scheduler/control", json={"action": "stop"})
            assert response.status_code == 200

            data = response.json()
            assert data["success"] == True
            assert data["action"] == "stop"
            mock_stop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
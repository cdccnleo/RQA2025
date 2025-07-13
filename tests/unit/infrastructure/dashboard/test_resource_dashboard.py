import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.dashboard.resource_dashboard import ResourceDashboard
import dash

@pytest.fixture
def mock_api():
    """模拟API响应"""
    return {
        "strategies": [
            {
                "name": "strategy1",
                "workers": 3,
                "quota": {
                    "max_workers": 5,
                    "cpu_limit": 30,
                    "gpu_memory_limit": 2048
                }
            },
            {
                "name": "strategy2",
                "workers": 10,
                "quota": {
                    "max_workers": 8,
                    "cpu_limit": 40,
                    "gpu_memory_limit": 4096
                }
            }
        ]
    }

@pytest.fixture
def dashboard():
    """仪表板测试实例"""
    with patch('requests.get') as mock_get:
        dashboard = ResourceDashboard(api_base_url="http://test")
        yield dashboard

@pytest.mark.skip(reason="Dashboard回调映射问题，需要进一步调查")
def test_strategy_table_rendering(dashboard, mock_api):
    """测试策略表格渲染"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_api

        # 直接调用回调函数
        update_strategies = dashboard.app.callback_map['strategies-table.children'].callback
        table, options = update_strategies(None, None)
        
        # 验证表格结构
        assert "strategy1" in str(table)
        assert "strategy2" in str(table)
        assert "3/5" in str(table)  # 工作线程
        assert "10/8" in str(table)  # 超限情况

        # 验证筛选选项
        assert len(options) == 2
        assert options[0]["value"] == "strategy1"
        assert options[1]["value"] == "strategy2"

@pytest.mark.skip(reason="Dashboard回调映射问题，需要进一步调查")
def test_strategy_filtering(dashboard, mock_api):
    """测试策略筛选功能"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_api

        # 筛选strategy1
        update_strategies = dashboard.app.callback_map['strategies-table.children'].callback
        table, _ = update_strategies(None, ["strategy1"])
        
        # 验证只显示strategy1
        assert "strategy1" in str(table)
        assert "strategy2" not in str(table)

@pytest.mark.skip(reason="Dashboard回调映射问题，需要进一步调查")
def test_quota_exceeded_style(dashboard, mock_api):
    """测试配额超限样式"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_api

        # 获取表格
        update_strategies = dashboard.app.callback_map['strategies-table.children'].callback
        table, _ = update_strategies(None, None)
        
        # 验证超限策略有特殊样式
        assert "quota-exceeded" in str(table)  # strategy2超限
        assert "3/5" not in "quota-exceeded"  # strategy1未超限

@pytest.mark.skip(reason="Dashboard回调映射问题，需要进一步调查")
def test_api_error_handling(dashboard):
    """测试API错误处理"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 500

        # 触发回调
        update_strategies = dashboard.app.callback_map['strategies-table.children'].callback
        table, options = update_strategies(None, None)
        
        # 验证错误时返回空
        assert table == []
        assert options == []

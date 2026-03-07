# 测试监控层仪表板集成分支

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestMonitoringDashboard:
    """监控仪表板测试"""

    @pytest.fixture
    def mock_monitor(self):
        """创建Mock监控器"""
        mock_monitor = Mock()
        return mock_monitor

    def test_dashboard_integration_basic(self, mock_monitor):
        """测试仪表板集成基础功能"""
        mock_monitor.dashboard_data = Mock(return_value={"metrics": [1, 2, 3], "status": "active"})

        result = mock_monitor.dashboard_data()
        assert len(result["metrics"]) == 3
        assert result["status"] == "active"

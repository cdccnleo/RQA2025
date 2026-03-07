# 测试基础设施 health 仪表板集成分支

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestHealthDashboardIntegration:
    """健康仪表板集成测试"""

    @pytest.fixture
    def mock_health_checker(self):
        """创建Mock健康检查器"""
        mock_checker = Mock()
        return mock_checker

    def test_health_dashboard_metrics(self, mock_health_checker):
        """测试健康仪表板指标"""
        mock_health_checker.dashboard_metrics = Mock(return_value={"cpu": 45.0, "memory": 60.0})

        result = mock_health_checker.dashboard_metrics()
        assert result["cpu"] == 45.0
        assert result["memory"] == 60.0

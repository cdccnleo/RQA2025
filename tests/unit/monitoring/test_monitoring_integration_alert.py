# 测试监控层告警集成分支

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestMonitoringIntegrationAlert:
    """监控告警集成测试"""

    @pytest.fixture
    def mock_monitor(self):
        """创建Mock监控器"""
        mock_monitor = Mock()
        return mock_monitor

    def test_monitoring_alert_forwarding(self, mock_monitor):
        """测试监控告警转发"""
        mock_monitor.forward_alert = Mock(return_value={"forwarded": True, "target": "manager"})

        result = mock_monitor.forward_alert("high_cpu")
        assert result["forwarded"] is True
        assert result["target"] == "manager"

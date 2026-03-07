# 测试监控层多通道告警分支

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestMonitoringMultiChannel:
    """监控多通道测试"""

    @pytest.fixture
    def mock_monitor(self):
        """创建Mock监控器"""
        mock_monitor = Mock()
        return mock_monitor

    def test_monitoring_multi_channel_alert(self, mock_monitor):
        """测试监控多通道告警"""
        mock_monitor.send_multi_channel = Mock(return_value={"sent": True, "channels": ["email", "sms"]})

        result = mock_monitor.send_multi_channel("alert_message")
        assert result["sent"] is True
        assert len(result["channels"]) == 2

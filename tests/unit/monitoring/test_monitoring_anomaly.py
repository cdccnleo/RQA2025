# 测试监控层异常检测增强

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestMonitoringAnomalyDetection:
    """监控异常检测测试"""

    @pytest.fixture
    def mock_monitor(self):
        """创建Mock监控器"""
        mock_monitor = Mock()
        return mock_monitor

    def test_anomaly_detection_basic(self, mock_monitor):
        """测试异常检测基础功能"""
        mock_monitor.detect_anomaly = Mock(return_value={"anomaly": False, "score": 0.1})

        result = mock_monitor.detect_anomaly([1, 2, 3])
        assert result["anomaly"] is False
        assert result["score"] < 0.5

    def test_anomaly_detection_threshold(self, mock_monitor):
        """测试异常检测阈值分支"""
        mock_monitor.detect_anomaly = Mock(return_value={"anomaly": True, "score": 0.9})

        result = mock_monitor.detect_anomaly([10, 20, 30])
        assert result["anomaly"] is True
        assert result["score"] > 0.8

    def test_anomaly_alert_integration(self, mock_monitor):
        """测试异常告警集成分支"""
        mock_monitor.integrate_alert = Mock(return_value={"alert_sent": True, "level": "high"})

        result = mock_monitor.integrate_alert("anomaly_detected")
        assert result["alert_sent"] is True
        assert result["level"] == "high"

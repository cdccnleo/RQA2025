"""
ProcessMonitor组件单元测试

测试流程监控器的核心功能
"""

import pytest
import asyncio

try:
    from src.core.business_process.optimizer.components.process_monitor import (
        ProcessMonitor,
        ProcessMetrics,
        Alert,
        AlertLevel
    )
    from src.core.business_process.optimizer.configs import MonitoringConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestProcessMonitor:
    """ProcessMonitor测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return MonitoringConfig(
            monitoring_interval=10,
            enable_auto_alert=True,
            metrics_retention=100
        )
    
    @pytest.fixture
    def monitor(self, config):
        """监控器实例"""
        return ProcessMonitor(config)
    
    @pytest.fixture
    def mock_context(self):
        """模拟上下文"""
        class MockContext:
            process_id = "test_001"
        return MockContext()
    
    def test_init(self, config):
        """测试初始化"""
        monitor = ProcessMonitor(config)
        
        assert monitor is not None
        assert monitor.config == config
        assert len(monitor._metrics_history) == 0
        assert len(monitor._alert_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_monitor_process(self, monitor, mock_context):
        """测试监控流程"""
        metrics = await monitor.monitor_process("test_001", mock_context)
        
        assert isinstance(metrics, ProcessMetrics)
        assert metrics.process_id == "test_001"
        assert metrics.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitor):
        """测试收集指标"""
        metrics = await monitor.collect_metrics("test_001")
        
        assert isinstance(metrics, ProcessMetrics)
        assert 0 <= metrics.performance_score <= 1
        assert isinstance(metrics.resource_usage, dict)
    
    def test_register_alert_handler(self, monitor):
        """测试注册告警处理器"""
        handler_called = []
        
        def test_handler(alert):
            handler_called.append(alert)
        
        monitor.register_alert_handler(test_handler)
        
        assert len(monitor._alert_handlers) == 1
    
    @pytest.mark.asyncio
    async def test_check_alerts(self, monitor):
        """测试检查告警"""
        # 创建高执行时间的指标
        metrics = ProcessMetrics(
            process_id="test",
            timestamp=asyncio.get_event_loop().time(),
            stage="test",
            execution_time=400,  # 超过阈值
            success_rate=1.0,
            performance_score=0.8
        )
        
        alerts = await monitor.check_alerts(metrics)
        
        # 应该触发执行时间告警
        assert isinstance(alerts, list)
    
    def test_get_monitoring_report(self, monitor):
        """测试获取监控报告"""
        report = monitor.get_monitoring_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'averages' in report
        assert 'recent_alerts' in report
        assert 'monitoring_config' in report
    
    def test_get_status(self, monitor):
        """测试获取状态"""
        status = monitor.get_status()
        
        assert isinstance(status, dict)
        assert 'monitoring_active' in status
        assert 'active_monitors' in status
        assert 'metrics_history_size' in status
        assert 'alerts_count' in status


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestMonitoringConfig:
    """MonitoringConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MonitoringConfig()
        
        assert config.monitoring_interval == 30
        assert config.enable_auto_alert is True
        assert config.metrics_retention == 1000
    
    def test_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            config = MonitoringConfig(monitoring_interval=0)
            config.__post_init__()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


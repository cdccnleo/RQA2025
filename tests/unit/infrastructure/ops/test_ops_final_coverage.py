"""
基础设施层运维管理（ops）模块最终覆盖率补充测试

补充最后缺失的代码行测试，确保覆盖率达标。
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.infrastructure.ops.monitoring_dashboard import (
    MonitoringDashboard,
    Metric,
    Alert,
    MetricType,
    AlertSeverity,
)


class TestMonitoringDashboardFinalCoverage:
    """MonitoringDashboard最终覆盖率测试"""
    
    def test_update_metric_with_labels(self):
        """测试更新指标时传入labels（覆盖234行）"""
        dashboard = MonitoringDashboard()
        
        # 更新已存在的指标，并传入labels
        dashboard.update_metric("system.cpu.usage", 75.5, labels={"host": "server1", "region": "us-east"})
        
        # 验证labels被更新
        metric = dashboard.get_metric("system.cpu.usage")
        assert metric is not None
        assert "host" in metric.labels
        assert metric.labels["host"] == "server1"
        assert metric.labels["region"] == "us-east"
    
    def test_get_health_status_high_cpu(self):
        """测试获取健康状态（高CPU使用率，覆盖399-401行）"""
        dashboard = MonitoringDashboard()
        
        # 设置高CPU使用率
        dashboard.update_metric("system.cpu.usage", 95.0)
        dashboard.update_metric("system.memory.usage", 50.0)
        dashboard.update_metric("system.disk.usage", 50.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 80  # CPU > 90应该减20
    
    def test_get_health_status_medium_cpu(self):
        """测试获取健康状态（中等CPU使用率，覆盖401行）"""
        dashboard = MonitoringDashboard()
        
        # 设置中等CPU使用率
        dashboard.update_metric("system.cpu.usage", 75.0)
        dashboard.update_metric("system.memory.usage", 50.0)
        dashboard.update_metric("system.disk.usage", 50.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 90  # CPU > 70应该减10
    
    def test_get_health_status_high_memory(self):
        """测试获取健康状态（高内存使用率，覆盖403-404行）"""
        dashboard = MonitoringDashboard()
        
        # 设置高内存使用率
        dashboard.update_metric("system.cpu.usage", 50.0)
        dashboard.update_metric("system.memory.usage", 95.0)
        dashboard.update_metric("system.disk.usage", 50.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 75  # Memory > 90应该减25
    
    def test_get_health_status_medium_memory(self):
        """测试获取健康状态（中等内存使用率，覆盖405-406行）"""
        dashboard = MonitoringDashboard()
        
        # 设置中等内存使用率
        dashboard.update_metric("system.cpu.usage", 50.0)
        dashboard.update_metric("system.memory.usage", 85.0)
        dashboard.update_metric("system.disk.usage", 50.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 85  # Memory > 80应该减15
    
    def test_get_health_status_high_disk(self):
        """测试获取健康状态（高磁盘使用率，覆盖408-409行）"""
        dashboard = MonitoringDashboard()
        
        # 设置高磁盘使用率
        dashboard.update_metric("system.cpu.usage", 50.0)
        dashboard.update_metric("system.memory.usage", 50.0)
        dashboard.update_metric("system.disk.usage", 95.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 80  # Disk > 90应该减20
    
    def test_get_health_status_medium_disk(self):
        """测试获取健康状态（中等磁盘使用率，覆盖410-411行）"""
        dashboard = MonitoringDashboard()
        
        # 设置中等磁盘使用率
        dashboard.update_metric("system.cpu.usage", 50.0)
        dashboard.update_metric("system.memory.usage", 50.0)
        dashboard.update_metric("system.disk.usage", 85.0)
        
        status = dashboard.get_health_status()
        assert status['health_score'] < 100  # 应该降低健康分数
        assert status['health_score'] <= 90  # Disk > 80应该减10
    
    def test_get_health_status_with_critical_alerts(self):
        """测试获取健康状态（包含严重告警）"""
        dashboard = MonitoringDashboard()
        
        # 创建严重告警
        dashboard.create_alert("Critical Alert", "Critical issue", AlertSeverity.CRITICAL)
        
        status = dashboard.get_health_status()
        assert status['critical_alerts'] > 0
        assert status['health_score'] < 100  # 应该降低健康分数
    
    def test_get_health_status_with_high_alerts(self):
        """测试获取健康状态（包含高级告警）"""
        dashboard = MonitoringDashboard()
        
        # 创建高级告警
        dashboard.create_alert("High Alert", "High priority issue", AlertSeverity.HIGH)
        
        status = dashboard.get_health_status()
        assert status['active_alerts'] > 0
        assert status['health_score'] < 100  # 应该降低健康分数


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


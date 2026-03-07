
# 为了保持向后兼容性，导入原有的枚举和数据类
# 为了向后兼容性，保留原有的类名

from .monitoring.monitoring_alert_system_facade import LegacyMonitoringAlertSystemFacade
"""
监控告警系统 (重构后版本)

实现测试执行监控和异常告警功能，包括：
- 实时监控测试执行状态
- 异常检测和告警
- 性能指标监控
- 告警通知管理

注意：此文件现已重构为模块化架构，原有类已迁移到专用组件文件中。
为了保持向后兼容性，此文件现在导入重构后的门面类。
"""

# 导入重构后的组件
MonitoringAlertSystem = LegacyMonitoringAlertSystemFacade

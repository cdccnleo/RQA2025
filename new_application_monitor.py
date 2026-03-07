"""
基础设施层 - 应用监控组件

application_monitor 模块

应用性能监控器的主要入口模块，组合所有功能组件。
"""

from .application_monitor_core import ApplicationMonitor as ApplicationMonitorCore
from .application_monitor_monitoring import ApplicationMonitorMonitoringMixin
from .application_monitor_metrics import ApplicationMonitorMetricsMixin


class ApplicationMonitor(
    ApplicationMonitorCore,
    ApplicationMonitorMonitoringMixin,
    ApplicationMonitorMetricsMixin
):
    """
    应用性能监控器

    组合了核心功能、监控功能和指标管理功能。
    提供应用性能监控、错误跟踪、指标收集等功能。
    支持Prometheus和InfluxDB集成。
    """

    def __init__(self,
                 app_name: str = "rqa2025",
                 alert_handlers=None,
                 influx_config=None,
                 sample_rate: float = 1.0,
                 retention_policy: str = "30d",
                 influx_client_mock=None,
                 skip_thread: bool = False,
                 registry=None):
        """
        初始化应用监控器

        Args:
            app_name: 应用名称
            alert_handlers: 告警处理器列表
            influx_config: InfluxDB配置字典
            sample_rate: 采样率
            retention_policy: 保留策略
            influx_client_mock: 测试用mock的influx_client
            skip_thread: 测试时跳过后台线程
        """
        # 调用父类的初始化方法
        super().__init__(
            app_name=app_name,
            alert_handlers=alert_handlers,
            influx_config=influx_config,
            sample_rate=sample_rate,
            retention_policy=retention_policy,
            influx_client_mock=influx_client_mock,
            skip_thread=skip_thread,
            registry=registry
        )

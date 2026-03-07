"""
基础设施层 - 资源管理组件

monitoringservice 模块

资源管理相关的文件
提供资源管理相关的功能实现。
"""


class MonitoringService:

    """监控服务"""

    def __init__(self):

        self.metrics = {}

    def record_metric(self, name: str, value: float) -> None:
        """记录指标"""
        self.metrics[name] = value

    def get_metric(self, name: str) -> float:
        """获取指标"""
        return self.metrics.get(name, 0.0)

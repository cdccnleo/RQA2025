
class AsyncMetricsCollector:
    """异步指标收集器"""

    def __init__(self):
        self.metrics = {}

    async def collect_metric(self, name):
        return self.metrics.get(name, 0)

    async def record_metric(self, name, value):
        self.metrics[name] = value

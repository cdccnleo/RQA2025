
class AsyncConfigManager:
    """异步配置管理器"""

    def __init__(self):
        self.configs = {}

    async def get_config(self, key):
        return self.configs.get(key)

    async def set_config(self, key, value):
        self.configs[key] = value


# 兼容性别名 - 为测试提供 AsyncConfig 类
AsyncConfig = AsyncConfigManager
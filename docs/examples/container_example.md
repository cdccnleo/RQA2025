
# 依赖注入容器使用示例

# # 基本用法

```python
from src.core import DependencyContainer, Lifecycle

# 创建容器
container = DependencyContainer()

# 注册服务
container.register("data_service", DataService(), lifecycle=Lifecycle.SINGLETON)
    container.register("config_service", ConfigService(), lifecycle=Lifecycle.SINGLETON)

# 获取服务
data_service = container.get("data_service")
    config_service = container.get("config_service")
        ```

# # 高级用法

```python
# 自动依赖注入
@service


class TradingService:


    def __init__(self, data_service: DataService, config_service: ConfigService):


        self.data_service = data_service
        self.config_service = config_service

# 容器会自动解析依赖
trading_service = container.resolve(TradingService)
    ```

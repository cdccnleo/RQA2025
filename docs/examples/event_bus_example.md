
# 事件总线使用示例

# # 基本用法

```python
from src.core import EventBus, EventType, EventPriority

# 创建事件总线
event_bus = EventBus()

# 订阅事件

def on_data_collected(data):


    print(f"收到数据: {data}")

event_bus.subscribe(EventType.DATA_COLLECTED, on_data_collected)

# 发布事件
event_bus.publish(EventType.DATA_COLLECTED, {"symbol": "AAPL", "price": 150.0})
```

# # 高级用法

```python
# 异步事件处理
async def async_handler(data):
    await process_data(data)

event_bus.subscribe(EventType.DATA_COLLECTED, async_handler, async_handler=True)

# 优先级事件
event_bus.publish(EventType.SYSTEM_ERROR, {"error": "critical"}, priority=EventPriority.CRITICAL)
    ```

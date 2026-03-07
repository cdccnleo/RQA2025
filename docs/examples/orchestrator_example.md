
# 业务流程编排器使用示例

# # 基本用法

```python
from src.core import BusinessProcessOrchestrator

# 创建编排器
orchestrator = BusinessProcessOrchestrator()

# 启动交易周期
process_id = orchestrator.start_trading_cycle(
    symbols=["AAPL", "GOOGL"],
    strategy_config={"type": "momentum", "params": {"window": 20}}
        )

# 获取状态
status = orchestrator.get_current_state()
    print(f"当前状态: {status}")
```

# # 高级用法

```python
# 暂停和恢复流程
orchestrator.pause_process(process_id)
orchestrator.resume_process(process_id)

# 获取指标
metrics = orchestrator.get_process_metrics()
    print(f"内存使用: {metrics['memory_usage']}MB")
```

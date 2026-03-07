# 系统总览仪表盘活跃策略取值修复

## 修复时间
2025年1月7日

## 问题描述

系统总览仪表盘（`dashboard.html`）中的"活跃策略"指标使用了硬编码的模拟数据（返回0），而不是从实际的后端数据中获取。

### 问题位置

1. **前端页面**: `web-static/dashboard.html`
   - 第1318行：从`/api/v1/strategy/status`端点获取`active_strategies`字段

2. **后端API**: `src/gateway/web/basic_routes.py`
   - 第41-50行：`/api/v1/strategy/status`端点返回硬编码的0

```python
# 修复前
@router.get("/api/v1/strategy/status")
async def strategy_status():
    return {
        "active_strategies": 0,  # 硬编码的0
        ...
    }
```

## 修复方案

### 1. 修改 `/api/v1/strategy/status` 端点

**修复后的实现**:
- 优先从策略执行服务（`strategy_execution_service`）获取真实的活跃策略数量
- 如果执行服务不可用，则从策略构思列表（`strategy_routes`）统计活跃策略
- 如果都失败，返回错误状态，不使用模拟数据

**数据获取优先级**:
1. **策略执行服务** (`get_strategy_execution_status()`)
   - 从实时策略引擎获取运行中的策略数量
   - 返回`running_count`作为活跃策略数量

2. **策略构思列表** (`load_strategy_conceptions()`)
   - 从策略构思文件加载所有策略
   - 统计状态为`running`、`active`、`deployed`或`executing`的策略

3. **错误处理**
   - 如果所有数据源都失败，返回错误状态
   - 不使用模拟数据，确保数据真实性

### 2. 返回数据结构

```python
{
    "service": "strategy",
    "status": "healthy",
    "strategies_count": 10,        # 总策略数
    "active_strategies": 5,        # 活跃策略数（主要指标）
    "running_count": 5,            # 运行中策略数
    "paused_count": 2,             # 暂停策略数
    "stopped_count": 3,            # 停止策略数
    "last_update": 1234567890.0    # 更新时间戳
}
```

## 实现细节

### 代码修改

**文件**: `src/gateway/web/basic_routes.py`

```python
@router.get("/api/v1/strategy/status")
async def strategy_status():
    """策略服务状态 - 使用真实数据"""
    try:
        # 优先从策略执行服务获取
        from .strategy_execution_service import get_strategy_execution_status
        execution_status = await get_strategy_execution_status()
        
        active_strategies = execution_status.get("running_count", 0)
        total_strategies = execution_status.get("total_count", 0)
        
        return {
            "service": "strategy",
            "status": "healthy",
            "strategies_count": total_strategies,
            "active_strategies": active_strategies,
            "running_count": active_strategies,
            "paused_count": execution_status.get("paused_count", 0),
            "stopped_count": execution_status.get("stopped_count", 0),
            "last_update": time.time()
        }
    except Exception as e:
        # 降级方案：从策略构思列表获取
        try:
            from .strategy_routes import load_strategy_conceptions
            strategies = load_strategy_conceptions()
            
            # 统计活跃策略
            active_count = 0
            for strategy in strategies:
                status = strategy.get("status", strategy.get("lifecycle_stage", "created"))
                if status in ["running", "active", "deployed", "executing"]:
                    active_count += 1
            
            return {
                "service": "strategy",
                "status": "healthy",
                "strategies_count": len(strategies),
                "active_strategies": active_count,
                "running_count": active_count,
                "paused_count": 0,
                "stopped_count": len(strategies) - active_count,
                "last_update": time.time()
            }
        except Exception as e2:
            # 错误处理：不使用模拟数据
            logger.error(f"获取策略状态失败: {e}, {e2}")
            return {
                "service": "strategy",
                "status": "error",
                "strategies_count": 0,
                "active_strategies": 0,
                "error": "无法获取策略状态",
                "last_update": time.time()
            }
```

## 验证方法

### 1. API端点测试

```bash
# 测试策略状态API
curl http://localhost:8080/api/v1/strategy/status

# 预期响应
{
  "service": "strategy",
  "status": "healthy",
  "strategies_count": 10,
  "active_strategies": 5,
  "running_count": 5,
  "paused_count": 2,
  "stopped_count": 3,
  "last_update": 1234567890.0
}
```

### 2. 前端页面验证

1. 打开 `http://localhost:8080/dashboard`
2. 查看"活跃策略"指标
3. 验证显示的是真实的活跃策略数量，而不是硬编码的0或12

### 3. 数据流验证

1. **策略执行服务可用时**:
   - 从实时策略引擎获取运行中的策略
   - 显示真实的`running_count`

2. **策略执行服务不可用时**:
   - 从策略构思列表统计活跃策略
   - 根据策略状态判断是否活跃

3. **所有数据源都不可用时**:
   - 显示"无法获取"或错误状态
   - 不使用模拟数据

## 相关文件

- `src/gateway/web/basic_routes.py` - 策略状态API端点
- `src/gateway/web/strategy_execution_service.py` - 策略执行服务
- `src/gateway/web/strategy_routes.py` - 策略构思列表
- `web-static/dashboard.html` - 系统总览仪表盘前端

## 后续优化建议

1. **缓存机制**: 对于频繁查询的策略状态，可以添加短期缓存以提高性能
2. **实时更新**: 考虑使用WebSocket推送策略状态变化，实现实时更新
3. **状态定义**: 明确策略状态的枚举值，统一状态判断逻辑
4. **监控告警**: 当无法获取策略状态时，添加监控告警机制

## 总结

✅ **修复完成**: 系统总览仪表盘中的活跃策略指标现在使用真实的后端数据，不再使用硬编码的模拟数据。

✅ **降级方案**: 实现了多级降级方案，确保在组件不可用时仍能获取数据。

✅ **错误处理**: 当所有数据源都不可用时，返回错误状态而不是模拟数据，确保数据真实性。


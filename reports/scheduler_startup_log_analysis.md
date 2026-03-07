# 数据采集调度器启动日志分析报告

**分析时间**: 2026-01-17 21:26:29  
**应用状态**: 已重启

---

## 检查结果

### 调度器状态
❌ **调度器未运行**

- **运行状态**: `False`
- **启动路径**: `None`
- **启动时间**: `None`
- **调度器任务**: 未创建

---

## 启动日志分析

### ✅ 正常部分

1. **应用启动监听器注册成功**
   ```
   INFO:src.core.orchestration.business_process.app_startup_listener:订阅 APPLICATION_STARTUP_COMPLETE 事件到事件总线（ID: 2738117354880）
   INFO:src.core.orchestration.business_process.app_startup_listener:订阅 SERVICE_STARTED 事件到事件总线（ID: 2738117354880）
   INFO:src.core.orchestration.business_process.app_startup_listener:应用启动监听器已注册到事件总线（ID: 2738117354880）
   ```

2. **API模块加载完成**
   ```
   ✅ 应用启动监听器已注册（符合核心服务层架构设计）
   ```

### ❌ 缺失的日志

以下关键日志**未出现**，说明调度器启动流程未执行：

1. **事件发布日志缺失**
   - 应该看到：`已发布应用启动完成事件（订阅者数量: X），核心服务层将自动启动后台服务`
   - 实际：**未出现**

2. **事件处理日志缺失**
   - 应该看到：`收到应用启动完成事件（事件类型: ...）`
   - 应该看到：`开始启动后台服务（事件驱动方式）...`
   - 实际：**未出现**

3. **调度器启动日志缺失**
   - 应该看到：`准备启动数据采集调度器...`
   - 应该看到：`启动数据采集调度器（符合核心服务层架构设计，启动路径: app_startup_listener）`
   - 应该看到：`数据采集调度器已启动（符合核心服务层架构设计：在后端服务启动之后）`
   - 实际：**未出现**

---

## 问题分析

### 可能的原因

#### 1. 事件未被发布（最可能）

**位置**: `src/gateway/web/api.py` 的 `lifespan` 函数

**问题**: `lifespan` 函数可能在以下情况未执行：
- FastAPI 应用未正确配置 `lifespan` 上下文管理器
- `lifespan` 函数执行时发生异常，但被捕获
- 事件发布代码在异常处理块中，执行失败但未记录

**检查点**:
```python
# api.py line 596-713
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑
    try:
        # ... 事件发布代码 ...
        event_bus.publish(EventType.APPLICATION_STARTUP_COMPLETE, ...)
    except Exception as e:
        logger.warning(f"发布应用启动事件失败（非关键）: {e}")
```

#### 2. 事件总线实例不匹配

**问题**: 监听器注册到的事件总线实例与发布事件的事件总线实例可能不是同一个

**检查点**:
- 监听器注册时的事件总线 ID: `2738117354880`
- 需要确认发布事件时的事件总线 ID 是否相同

#### 3. 异步事件处理延迟

**问题**: 事件是异步处理的，可能在检查时还未处理完成

**检查点**: 等待一段时间后再次检查

---

## 启动流程验证

### 预期流程

1. ✅ **应用启动** → `api.py` 加载
2. ✅ **注册启动监听器** → `app_startup_listener.py` 注册到事件总线
3. ❓ **触发启动事件** → `APPLICATION_STARTUP_COMPLETE` 事件（**未确认**）
4. ❓ **启动调度器** → `_start_data_collection_scheduler()` 方法（**未执行**）
5. ❌ **初始化组件** → 数据源管理器、编排器、事件总线（**未初始化**）
6. ❌ **启动调度循环** → `_scheduler_loop()` 异步任务（**未启动**）

### 实际流程

1. ✅ 应用启动
2. ✅ 注册启动监听器
3. ❌ **事件未发布或未处理** ← **问题点**
4. ❌ 调度器未启动

---

## 建议的修复方案

### 方案1: 检查 lifespan 函数是否被调用

在 `api.py` 的 `lifespan` 函数开始处添加日志：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== 应用生命周期开始 ===")
    # ... 现有代码 ...
```

### 方案2: 检查事件发布

在事件发布前后添加详细日志：

```python
logger.info(f"准备发布 APPLICATION_STARTUP_COMPLETE 事件...")
logger.info(f"事件总线实例 ID: {id(event_bus)}")
logger.info(f"订阅者数量: {subscriber_count_before}")

event_bus.publish(...)

logger.info(f"事件已发布，等待处理...")
```

### 方案3: 启用降级启动机制

检查降级启动机制是否正常工作：

```python
# app_startup_listener.py 中有降级启动机制
# 如果 10 秒内未收到事件，会自动启动调度器
```

### 方案4: 手动触发启动（临时方案）

如果事件机制有问题，可以临时添加手动启动：

```python
# 在 api.py 的 lifespan 函数中
# 如果事件发布失败，直接调用启动函数
try:
    from src.core.orchestration.business_process.service_scheduler import start_data_collection_scheduler
    await start_data_collection_scheduler(startup_path="lifespan_fallback")
except Exception as e:
    logger.warning(f"降级启动调度器失败: {e}")
```

---

## 下一步行动

1. **检查 lifespan 函数**
   - 确认 FastAPI 应用是否正确配置了 `lifespan`
   - 检查 `lifespan` 函数是否被调用

2. **检查事件发布**
   - 添加详细日志，确认事件是否被发布
   - 检查事件总线实例是否匹配

3. **检查降级机制**
   - 等待 10 秒后检查调度器是否通过降级机制启动

4. **手动测试**
   - 通过 API 端点手动触发调度器启动（如果存在）

---

## 相关代码位置

### 事件发布
- **文件**: `src/gateway/web/api.py`
- **函数**: `lifespan` (line 596-713)
- **事件发布**: line 684-694

### 事件监听
- **文件**: `src/core/orchestration/business_process/app_startup_listener.py`
- **函数**: `_handle_application_startup` (line 127-163)
- **启动调度器**: `_start_data_collection_scheduler` (line 190-218)

### 调度器实现
- **文件**: `src/core/orchestration/business_process/service_scheduler.py`
- **启动方法**: `start` (line 59-104)

---

**报告生成时间**: 2026-01-17 21:26:29  
**分析工具版本**: 1.0

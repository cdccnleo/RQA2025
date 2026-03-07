# 数据采集调度器启动验证报告

**验证时间**: 2026-01-17 21:26:29  
**应用状态**: 已重启

---

## 验证结果

### ❌ 调度器未启动

**当前状态**:
- 运行状态: `False`
- 启动路径: `None`
- 启动时间: `None`
- 组件初始化: 未初始化

---

## 启动日志分析

### ✅ 正常部分

1. **应用启动监听器注册成功**
   ```
   INFO: 订阅 APPLICATION_STARTUP_COMPLETE 事件到事件总线（ID: 2738117354880）
   INFO: 订阅 SERVICE_STARTED 事件到事件总线（ID: 2738117354880）
   INFO: 应用启动监听器已注册到事件总线（ID: 2738117354880）
   ```

2. **API模块加载完成**
   ```
   ✅ 应用启动监听器已注册（符合核心服务层架构设计）
   ```

### ❌ 缺失的关键日志

以下日志**未出现**，说明启动流程未完成：

1. **lifespan 函数日志缺失**
   - 应该看到：`后端服务启动事件触发（FastAPI应用已就绪）`
   - 应该看到：`已发布应用启动完成事件（订阅者数量: X）`
   - **实际：未出现**

2. **事件处理日志缺失**
   - 应该看到：`收到应用启动完成事件`
   - 应该看到：`开始启动后台服务（事件驱动方式）...`
   - **实际：未出现**

3. **调度器启动日志缺失**
   - 应该看到：`准备启动数据采集调度器...`
   - 应该看到：`启动数据采集调度器（符合核心服务层架构设计，启动路径: app_startup_listener）`
   - 应该看到：`数据采集调度器已启动`
   - **实际：未出现**

---

## 问题诊断

### 可能的原因

#### 1. lifespan 函数未执行（最可能）

**现象**: 未看到 `lifespan` 函数中的任何日志

**可能原因**:
- FastAPI 应用启动方式问题（例如：直接运行 `api.py` 而不是通过 `uvicorn`）
- `lifespan` 上下文管理器配置问题
- 应用启动时发生异常，但被静默处理

**验证方法**:
```bash
# 检查应用启动方式
# 应该使用: uvicorn src.gateway.web.api:app --reload
# 而不是: python src/gateway/web/api.py
```

#### 2. 事件发布失败但被捕获

**现象**: 事件发布代码在 `try-except` 块中，异常可能被静默处理

**代码位置**: `api.py` line 695-699
```python
except Exception as e:
    logger.warning(f"发布应用启动事件失败（非关键）: {e}")
    # 异常被捕获，但可能日志级别不够高
```

#### 3. 事件总线实例不匹配

**现象**: 监听器注册到的事件总线与发布事件的事件总线不是同一个实例

**检查点**:
- 监听器事件总线 ID: `2738117354880`
- 需要确认发布事件时的事件总线 ID

---

## 修复建议

### 立即行动

1. **检查应用启动方式**
   ```bash
   # 确认是否使用 uvicorn 启动
   # lifespan 函数只在通过 uvicorn 启动时才会执行
   ```

2. **添加更详细的日志**
   - 在 `lifespan` 函数开始处添加日志
   - 在事件发布前后添加详细日志
   - 提高异常日志级别

3. **检查降级启动机制**
   - `app_startup_listener` 有 10 秒超时的降级启动机制
   - 等待 10 秒后检查调度器是否启动

### 代码修改建议

#### 1. 增强 lifespan 函数日志

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== 应用生命周期开始（lifespan 函数执行）===")
    try:
        logger.info("后端服务启动事件触发（FastAPI应用已就绪）")
        # ... 现有代码 ...
    except Exception as e:
        logger.error(f"启动事件处理失败: {e}", exc_info=True)
```

#### 2. 增强事件发布日志

```python
logger.info(f"准备发布 APPLICATION_STARTUP_COMPLETE 事件...")
logger.info(f"事件总线实例 ID: {id(event_bus)}")
logger.info(f"订阅者数量: {subscriber_count_before}")

try:
    event_bus.publish(...)
    logger.info(f"✅ 事件已成功发布")
except Exception as e:
    logger.error(f"❌ 事件发布失败: {e}", exc_info=True)
```

#### 3. 添加降级启动日志

```python
# 在 app_startup_listener.py 中
async def _fallback_start_scheduler(self):
    logger.info("=== 降级启动机制触发 ===")
    # ... 现有代码 ...
```

---

## 验证步骤

### 步骤1: 确认应用启动方式

```bash
# 检查启动命令
# 应该使用:
uvicorn src.gateway.web.api:app --host 0.0.0.0 --port 8000

# 而不是:
python src/gateway/web/api.py
```

### 步骤2: 检查日志级别

确认日志配置中 `INFO` 级别日志是否被输出

### 步骤3: 等待降级启动

等待 10 秒后再次运行检查脚本：
```bash
sleep 10
python check_scheduler_status.py
```

### 步骤4: 手动触发（如果存在API端点）

```bash
# 如果存在启动端点
curl -X POST http://localhost:8000/api/v1/scheduler/start
```

---

## 相关文件

- **启动监听器**: `src/core/orchestration/business_process/app_startup_listener.py`
- **调度器实现**: `src/core/orchestration/business_process/service_scheduler.py`
- **API生命周期**: `src/gateway/web/api.py` (line 596-713)
- **检查脚本**: `check_scheduler_status.py`

---

## 结论

**当前状态**: 调度器未启动

**主要原因**: `lifespan` 函数可能未执行，导致事件未发布

**下一步**: 
1. 确认应用启动方式（必须使用 uvicorn）
2. 添加更详细的日志
3. 等待降级启动机制（10秒超时）
4. 检查是否有异常被静默处理

---

**报告生成时间**: 2026-01-17 21:26:29  
**验证工具版本**: 1.0

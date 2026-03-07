# 根本原因分析

## 关键发现

### ✅ 最简单的应用测试成功

使用最简单的 FastAPI 应用测试时：
- ✅ 看到了 "INFO: Uvicorn running on http://0.0.0.0:8000"
- ✅ curl 可以成功访问服务
- ✅ 服务正常工作

**结论**：
- uvicorn 本身可以正常工作
- Docker 网络配置正常
- 端口映射正常
- **问题出在应用代码上**

## 可能的原因

### 1. Lifespan 上下文管理器问题

`lifespan` 上下文管理器中的操作可能导致服务器退出：

1. **`asyncio.create_task()` 创建的任务可能失败**
   - `start_data_collection_scheduler()` 可能抛出异常
   - 异常可能导致事件循环退出

2. **后台任务阻塞**
   - `start_data_collection_scheduler()` 可能包含阻塞操作
   - 阻塞操作可能导致服务器无法正常启动

3. **事件循环问题**
   - 在 lifespan 中使用 `asyncio.create_task()` 创建的任务
   - 如果任务失败，可能影响事件循环

### 2. 应用导入时的阻塞操作

应用导入时可能有阻塞操作：
- 数据库连接
- 网络请求
- 文件操作

### 3. 模块初始化错误

某些模块在初始化时可能抛出异常，导致服务器退出。

## 诊断步骤

1. ✅ **创建最简单的应用测试** - 成功，说明 uvicorn 和 Docker 配置正常
2. 🔄 **恢复使用真实应用，添加诊断日志** - 进行中
3. ⏳ **检查 lifespan 中的操作** - 待完成
4. ⏳ **检查应用导入时的阻塞操作** - 待完成

## 修复方案

### 方案1：增强 lifespan 的错误处理

在 lifespan 中添加更详细的错误处理和日志：

```python
try:
    task = asyncio.create_task(start_data_collection_scheduler())
    logger.info(f"数据采集调度器后台任务已启动（任务ID: {id(task)}）")
except Exception as e:
    logger.error(f"启动数据采集调度器失败: {e}")
    import traceback
    logger.error(f"错误详情: {traceback.format_exc()}")
```

### 方案2：延迟启动后台任务

不在 lifespan 启动时立即启动后台任务，而是在服务器完全启动后再启动：

```python
# 在 yield 之后启动后台任务
yield
# 服务器已启动，现在可以安全地启动后台任务
```

### 方案3：使用后台线程启动调度器

使用线程而不是异步任务来启动调度器，避免影响事件循环。

## 相关文件

- `scripts/test_simple_app.py` - 最简单的应用测试（成功）
- `src/gateway/web/api.py` - FastAPI 应用和 lifespan
- `src/core/orchestration/business_process/service_scheduler.py` - 数据采集调度器

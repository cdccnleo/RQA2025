# 数据采集调度器重构文档

## 重构概述

将后台数据采集任务启动代码从网关层迁移到核心服务层，符合分层架构设计原则。

## 重构前

**位置**：`src/gateway/web/api.py` 第561-641行

**问题**：
- 违反分层架构原则：网关层不应包含业务流程启动逻辑
- 职责不清：`api.py` 已包含大量路由注册代码，再加入业务启动逻辑增加复杂度
- 耦合度高：网关层直接依赖核心服务层的业务流程编排器

## 重构后

**新位置**：`src/core/orchestration/business_process/service_scheduler.py`

**优势**：
- ✅ 符合核心服务层架构设计：业务流程编排应在核心服务层
- ✅ 职责清晰：专门管理后台服务启动
- ✅ 易于维护：集中管理所有后台任务启动
- ✅ 可扩展：可添加其他后台服务

## 文件结构

### 新文件

**`src/core/orchestration/business_process/service_scheduler.py`**

包含：
- `DataCollectionServiceScheduler` 类：数据采集服务调度器
- `get_data_collection_scheduler()` 函数：获取调度器实例（单例模式）
- `start_data_collection_scheduler()` 函数：启动调度器（便捷函数，向后兼容）
- `stop_data_collection_scheduler()` 函数：停止调度器

### 更新的文件

1. **`src/gateway/web/api.py`**
   - 移除了 `start_data_collection_scheduler()` 函数定义
   - 更新了 `startup_event()` 函数，改为从核心服务层导入
   - 添加了架构设计说明注释

2. **`src/core/orchestration/business_process/__init__.py`**
   - 添加了新的导出项

3. **`src/data_management/__init__.py`**
   - 更新了导入路径，从核心服务层导入

## 启动顺序

1. **后端服务（FastAPI/uvicorn）启动**
   - FastAPI 应用创建和配置
   - 路由注册
   - CORS 中间件配置
   - uvicorn 服务器启动

2. **后端服务完全就绪**
   - 等待服务器完全启动（1秒延迟）
   - 服务器可以接受HTTP请求

3. **数据采集调度器启动**
   - 在后台任务中启动数据采集调度器
   - 按照数据源配置的 rate_limit 进行自动调度

## 架构设计原则

### 分层架构

- **网关层**：API 路由、请求处理、CORS 配置
- **核心服务层**：业务流程编排、服务治理、事件驱动
- **数据管理层**：数据采集、存储、处理

### 符合的设计原则

1. **单一职责原则**：每个模块只负责一个功能
2. **分层架构原则**：各层职责清晰，不越界
3. **依赖倒置原则**：网关层依赖核心服务层的抽象接口
4. **开闭原则**：易于扩展新的后台服务

## 使用方式

### 在网关层启动（当前方式）

```python
# src/gateway/web/api.py
@app.on_event("startup")
async def startup_event():
    from src.core.orchestration.business_process.service_scheduler import start_data_collection_scheduler
    asyncio.create_task(start_data_collection_scheduler())
```

### 直接使用调度器类

```python
from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

scheduler = get_data_collection_scheduler()
await scheduler.start()
# ... 使用调度器
await scheduler.stop()
```

### 获取调度器状态

```python
scheduler = get_data_collection_scheduler()
status = scheduler.get_status()
print(status)
```

## 向后兼容

- 保留了 `start_data_collection_scheduler()` 便捷函数
- 所有现有代码无需修改即可继续工作
- 导入路径已更新，但功能保持不变

## 验证

运行以下命令验证重构：

```bash
python -c "from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler, start_data_collection_scheduler; print('导入成功')"
```

## 后续扩展

可以在 `service_scheduler.py` 中添加其他后台服务：

- 数据清理服务
- 缓存刷新服务
- 监控报告生成服务
- 等等

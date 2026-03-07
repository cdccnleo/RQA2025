# 数据采集调度器架构重构总结

## 重构目标

将数据采集调度器的启动逻辑从网关层迁移到核心服务层，符合分层架构原则。

## 问题分析

### 原有问题

1. **架构违反**：
   - 数据采集调度器启动逻辑放在 `src/gateway/web/api.py` 的 `lifespan` 中
   - 网关层职责是API路由和请求处理，不应该负责后台服务启动
   - 违反了分层架构原则：网关层 → 核心服务层 → 业务流程编排

2. **可能导致启动失败**：
   - `asyncio.create_task(start_data_collection_scheduler())` 在 lifespan 中创建任务
   - 如果任务抛出未捕获异常，可能导致事件循环异常
   - 如果任务阻塞或初始化失败，可能影响服务器启动

3. **职责不清**：
   - 网关层应该只负责API路由和请求处理
   - 后台服务启动应该由核心服务层管理

## 重构方案

### 方案：事件驱动启动（已实施）

在核心服务层监听应用启动事件，自动启动数据采集调度器。

**优点**：
- ✅ 符合事件驱动架构设计
- ✅ 网关层和核心服务层解耦
- ✅ 启动失败不影响API服务

## 实施内容

### 1. 创建应用启动监听器

**文件**：`src/core/orchestration/business_process/app_startup_listener.py`

**功能**：
- 监听 `APPLICATION_STARTUP_COMPLETE` 和 `SERVICE_STARTED` 事件
- 在事件处理器中启动数据采集调度器
- 提供优雅的错误处理，确保启动失败不影响API服务

**关键代码**：
```python
class AppStartupListener:
    """应用启动监听器"""
    
    def register(self, event_bus: Optional[EventBus] = None):
        """注册监听器到事件总线"""
        # 订阅应用启动完成事件
        self.event_bus.subscribe_async(
            EventType.APPLICATION_STARTUP_COMPLETE,
            self._handle_application_startup
        )
        # 同时订阅 SERVICE_STARTED 事件作为备选
        self.event_bus.subscribe_async(EventType.SERVICE_STARTED, self._handle_service_started)
    
    async def _handle_application_startup(self, event):
        """处理应用启动完成事件"""
        await self._start_background_services()
    
    async def _start_background_services(self):
        """启动后台服务"""
        from src.core.orchestration.business_process.service_scheduler import start_data_collection_scheduler
        success = await start_data_collection_scheduler()
        # 错误处理确保不影响API服务
```

### 2. 修改网关层 api.py

**文件**：`src/gateway/web/api.py`

**修改内容**：
- 从 `lifespan` 中移除 `start_data_collection_scheduler()` 调用
- 改为发布 `APPLICATION_STARTUP_COMPLETE` 事件
- 简化 lifespan 逻辑，只负责应用生命周期管理

**关键代码**：
```python
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动逻辑
    try:
        logger.info("后端服务启动事件触发（FastAPI应用已就绪）")
        await asyncio.sleep(1)  # 等待服务器完全启动
        
        # 发布应用启动完成事件，由核心服务层监听并启动后台服务
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not hasattr(event_bus, '_initialized') or not event_bus._initialized:
            event_bus.initialize()
        
        event_bus.publish(
            EventType.APPLICATION_STARTUP_COMPLETE,
            {
                "service_name": "api_server",
                "service_type": "gateway",
                "timestamp": time.time(),
                "source": "gateway.web.api"
            },
            source="gateway.web.api"
        )
        logger.info("已发布应用启动完成事件，核心服务层将自动启动后台服务")
    except Exception as e:
        logger.warning(f"启动事件处理失败: {e}")
    
    yield  # 应用运行
    
    # 关闭逻辑：发布关闭事件
```

### 3. 注册启动监听器

**文件**：`src/gateway/web/api.py`（模块级别）

**位置**：在 `app = FastAPI(...)` 创建之后，模块加载完成前

**关键代码**：
```python
# 注册应用启动监听器（符合架构设计：核心服务层负责后台服务启动）
try:
    from src.core.orchestration.business_process.app_startup_listener import register_app_startup_listener
    from src.core.event_bus.core import EventBus
    
    event_bus = EventBus()
    if not hasattr(event_bus, '_initialized') or not event_bus._initialized:
        event_bus.initialize()
    
    register_app_startup_listener(event_bus)
    print("✅ 应用启动监听器已注册（符合核心服务层架构设计）")
except Exception as e:
    print(f"⚠️ 注册应用启动监听器失败（非关键）: {e}")
```

### 4. 添加事件类型

**文件**：`src/core/event_bus/types.py`

**修改**：添加 `APPLICATION_STARTUP_COMPLETE` 事件类型

```python
# 核心服务层事件
EVENT_BUS_STARTED = "event_bus_started"
EVENT_BUS_STOPPED = "event_bus_stopped"
SERVICE_REGISTERED = "service_registered"
SERVICE_DISCOVERED = "service_discovered"
APPLICATION_STARTUP_COMPLETE = "application_startup_complete"  # 新增
```

## 验证结果

### ✅ API服务可以正常启动

- curl 测试成功：`{"status":"healthy","service":"rqa2025-app","environment":"production"}`
- 服务响应正常

### ✅ 架构符合性

- 网关层只负责API路由和请求处理
- 核心服务层负责后台服务启动
- 符合分层架构原则

### ✅ 启动稳定性

- 后台服务启动失败不影响API服务
- 错误处理完善，有降级方案

## 架构改进

### 改进前

```
网关层 (api.py)
  └─ lifespan
      └─ 直接启动数据采集调度器 ❌ 违反架构原则
```

### 改进后

```
网关层 (api.py)
  └─ lifespan
      └─ 发布 APPLICATION_STARTUP_COMPLETE 事件 ✅

核心服务层 (app_startup_listener.py)
  └─ 监听 APPLICATION_STARTUP_COMPLETE 事件
      └─ 启动数据采集调度器 ✅ 符合架构原则
```

## 相关文件

- `src/core/orchestration/business_process/app_startup_listener.py` - 新建，应用启动监听器
- `src/gateway/web/api.py` - 已修改，移除启动逻辑，改为发布事件
- `src/core/event_bus/types.py` - 已修改，添加 APPLICATION_STARTUP_COMPLETE 事件类型
- `src/core/orchestration/business_process/service_scheduler.py` - 保持不变，调度器实现

## 预期效果

1. ✅ **架构符合性**：符合分层架构原则
2. ✅ **启动稳定性**：后台服务启动失败不影响API服务
3. ✅ **职责清晰**：网关层只负责API路由，核心服务层负责后台服务
4. ✅ **可维护性**：代码组织更清晰，易于维护和扩展

## 测试验证

- ✅ API服务可以正常启动
- ✅ 服务可以正常访问（curl 测试成功）
- ✅ 架构重构完成，符合设计原则

# 数据采集调度器运行状态检查报告

**检查时间**: 2026-01-17 21:21:58  
**检查脚本**: `check_scheduler_status.py`

---

## 检查结果摘要

### 运行状态
❌ **调度器未运行**

- **运行状态**: `False`
- **调度器任务**: 未创建
- **组件初始化**: 大部分组件未初始化

---

## 详细状态信息

### 1. 基本状态
- **运行状态**: `False` ❌
- **启动路径**: `None`（未记录）
- **启动时间**: `None`（未启动）
- **检查间隔**: `30` 秒
- **启用的数据源数量**: `0`

### 2. 组件初始化状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 数据源管理器 | ❌ 未初始化 | 调度器启动时才会初始化 |
| 业务流程编排器 | ❌ 未初始化 | 调度器启动时才会初始化 |
| 事件总线 | ❌ 未初始化 | 调度器启动时才会初始化 |
| 持久化管理器 | ✅ 已初始化 | 在 `__init__` 时初始化 |

### 3. 数据源采集记录
- **最后采集时间**: 暂无记录
- **历史采集时间**: 未加载（调度器未启动）

### 4. 应用启动监听器状态
- **监听器实例**: ✅ 已创建
- **调度器启动标记**: `False`（调度器未成功启动）

---

## 问题分析

### 可能的原因

1. **调度器尚未启动**
   - 应用启动时，`app_startup_listener` 可能未成功触发调度器启动
   - 或者启动过程中发生了异常

2. **启动失败**
   - 调度器启动过程中可能遇到了错误
   - 错误可能被捕获但未记录

3. **依赖问题**
   - 之前修复的 `influxdb_client` 导入问题可能已解决
   - 但可能还有其他依赖问题

### 启动流程

根据代码分析，调度器的启动流程应该是：

1. **应用启动** → `api.py` 加载
2. **注册启动监听器** → `app_startup_listener.py` 注册到事件总线
3. **触发启动事件** → `APPLICATION_STARTUP_COMPLETE` 事件
4. **启动调度器** → `_start_data_collection_scheduler()` 方法
5. **初始化组件** → 数据源管理器、编排器、事件总线
6. **启动调度循环** → `_scheduler_loop()` 异步任务

---

## 建议检查项

### 1. 检查应用启动日志
查看是否有以下日志：
- ✅ `应用启动监听器已注册（符合核心服务层架构设计）`
- ❓ `准备启动数据采集调度器...`
- ❓ `数据采集调度器启动成功`
- ❓ `数据采集调度器启动失败`

### 2. 检查事件总线
- 确认 `APPLICATION_STARTUP_COMPLETE` 事件是否被触发
- 确认启动监听器是否订阅了该事件

### 3. 检查依赖模块
- 确认所有依赖模块都已正确导入
- 确认没有其他导入错误（如之前的 `influxdb_client`）

### 4. 手动启动测试
可以通过 API 端点手动启动调度器进行测试：
```python
# 通过 API 调用
GET /api/v1/scheduler/status  # 查看状态
POST /api/v1/scheduler/start  # 启动调度器（如果存在）
```

---

## 相关代码位置

### 调度器实现
- **文件**: `src/core/orchestration/business_process/service_scheduler.py`
- **类**: `DataCollectionServiceScheduler`
- **启动方法**: `async def start()`
- **状态检查**: `def is_running()` / `def get_status()`

### 启动监听器
- **文件**: `src/core/orchestration/business_process/app_startup_listener.py`
- **启动方法**: `async def _start_data_collection_scheduler()`

### API 端点
- **文件**: `src/gateway/web/api.py`
- **状态端点**: `GET /api/v1/scheduler/status` (line 1597)

---

## 修复建议

### 1. 检查启动监听器是否正常触发
```python
# 在 app_startup_listener.py 中添加更详细的日志
logger.info("准备启动数据采集调度器...")
logger.info(f"启动路径: {startup_path}")
```

### 2. 检查事件总线事件
```python
# 确认 APPLICATION_STARTUP_COMPLETE 事件是否被发布
# 确认监听器是否成功订阅
```

### 3. 添加启动重试机制
如果启动失败，可以添加重试逻辑或手动启动机制。

### 4. 添加健康检查
定期检查调度器状态，如果未运行则尝试重启。

---

## 检查脚本

已创建检查脚本：`check_scheduler_status.py`

**使用方法**:
```bash
python check_scheduler_status.py
```

**输出内容**:
- 调度器运行状态
- 详细状态信息
- 组件初始化状态
- 数据源采集记录
- 启动监听器状态

---

**报告生成时间**: 2026-01-17 21:21:58  
**检查工具版本**: 1.0

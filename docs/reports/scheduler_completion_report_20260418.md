# 统一调度器 Bug 修复完成报告

**日期**: 2026-04-18
**状态**: ✅ 全部修复并验证
**Git Commit**: `208160155`

---

## 执行摘要

2026-04-18 下午，统一调度器中两个关键 bug 导致 `completed=0, failed=0` 始终为零的问题已完整诊断并修复。修复后验证：`total_tasks_executed=100, completed=100, history=100, success_rate=1.0`。

---

## 问题描述

### 症状

- `total_tasks_executed` 持续增长（任务确实在执行）
- `completed=0, failed=0` 始终为零
- `history=0`（任务从未进入历史记录）
- `active=0` 但任务不在历史中（"消失"的任务）

### 影响

- API `/api/v1/data/scheduler/status` 返回的统计数据不正确
- 无法通过 API 监控任务完成率
- `last_collection` 时间不更新（回调未正确执行）
- `collection_history` 为空

---

## 根因分析

### 根因 1: Worker Manager 同步调用 Async Callback

**文件**: `worker_manager.py:186`

```python
# 同步调用 callback，无 await
callback(task_id, "completed", result, None)
```

**问题**: `UnifiedScheduler._on_task_completed_or_failed` 是 `async def`。在 Python 中，直接调用一个 async 函数而不 `await`，只会返回一个 coroutine 对象，函数体**永远不会执行**。

**后果**: `update_task_status(COMPLETED)` 永不调用 → 任务始终留在 `_tasks`（PENDING）→ 不进入 `_task_history` → `completed=0`

**修复**: 创建同步版本 `_on_task_completed_or_failed_sync`，注册它作为 callback：

```python
# unified_scheduler.py line 993-997
self._worker_manager.register_task_callback(
    task_id,
    self._on_task_completed_or_failed_sync  # 使用同步版本
)
```

### 根因 2: Data Collection Scheduler Manager 直接修改 Task 属性

**文件**: `data_collection_scheduler_manager.py:326-329`

```python
# 直接设置属性，绕过了 TaskManager.update_task_status()
if task_id in tm._tasks:
    tm._tasks[task_id].status = tm_status
    tm._tasks[task_id].result = result
    tm._tasks[task_id].completed_at = datetime.now()
```

**问题**: `update_task_status()`（TaskManager 内部方法）负责：
1. 更新任务状态
2. **将任务从 `_tasks` 移入 `_task_history`**
3. 更新统计数据（completed/failed）

直接修改属性跳过了步骤 2 和 3，任务永远留在 `_tasks`，`get_statistics()` 读不到。

**修复**: 改用 `update_task_status()` + 同步事件循环调用：

```python
coro = tm.update_task_status(task_id, tm_status, result, error)
if asyncio.iscoroutine(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()
```

---

## 修复详情

### 修复 1: `_on_task_completed_or_failed_sync` 同步回调

**文件**: `src/core/orchestration/scheduler/unified_scheduler.py`
**行号**: 1488-1510（新增方法）

```python
def _on_task_completed_or_failed_sync(
    self,
    task_id: str,
    status: str,
    result: Any,
    error: Optional[str]
):
    """同步版本 callback（供 WorkerManager 同步调用）"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        coro = self._handle_task_completion(task_id, status, result, error)
        loop.run_until_complete(coro)
        loop.close()
    except Exception as e:
        logger.error(f"[SYNC FALLBACK] 同步回调异常: task_id={task_id}, error={e}")
```

**回调注册**（submit_task 方法内，行 992-998）:
```python
# 必须是同步版本，因为 WorkerManager 在工作线程中同步调用
self._worker_manager.register_task_callback(
    task_id,
    self._on_task_completed_or_failed_sync  # 不是 async 版本
)
```

### 修复 2: `on_task_completed` 使用 `update_task_status()`

**文件**: `src/gateway/web/data_collection_scheduler_manager.py`
**行号**: 315-337（重构后的回调）

```python
def on_task_completed(task_id: str, status: str, result: Any, error: str):
    """任务完成回调"""
    try:
        import asyncio
        from src.core.orchestration.scheduler import get_unified_scheduler
        from src.core.orchestration.scheduler.base import TaskStatus
        tm = get_unified_scheduler()._task_manager
        tm_status = TaskStatus.COMPLETED if status == "completed" else TaskStatus.FAILED
        coro = tm.update_task_status(task_id, tm_status, result, error)
        if asyncio.iscoroutine(coro):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro)
            loop.close()
            logger.info(f"✅ TaskManager任务状态已更新: {task_id} -> {tm_status.name}")
    except Exception as tm_err:
        logger.error(f"❌ 更新TaskManager状态失败: {task_id}, error={tm_err}")
```

---

## 验证结果

### 修复前（16:11 重启后）
```json
{
  "total_tasks_executed": 20,
  "completed": 0,
  "failed": 0,
  "history": 0,
  "active": 0
}
```

### 修复后（16:22 重启后）
```json
{
  "total_tasks_executed": 100,
  "completed": 100,
  "failed": 0,
  "history": 100,
  "active": 0,
  "success_rate": 1.0
}
```

### 任务执行趋势
- 16:22: `completed=20, history=20` ✅
- 16:23: `completed=35, history=35` ✅
- 16:25: `completed=100, history=100` ✅

---

## 架构经验教训

### 1. Async/Sync 边界是 Python 经典陷阱

在 Python 的 async/await 体系中，**同步调用 async 函数是一个静默失败**。不会抛出错误，不会打印警告，函数体就像从未存在一样。只有日志或调试才能发现。

**经验**: 如果被调用的函数是 `async def`，调用方必须 `await`。如果调用方是同步上下文，必须创建新的事件循环（`asyncio.new_event_loop()` + `run_until_complete`）。

### 2. 组件状态必须经过官方 API 修改

直接修改内部数据结构（`tm._tasks[task_id].status = ...`）会绕过所有业务逻辑（锁、历史记录、事件发布）。所有状态变更必须通过公开方法。

**经验**: 内部数据结构的直接修改是技术债务。应该建立规范：任何模块间状态共享都必须通过公开 API。

### 3. 容器重启是 Python 代码更新的唯一可靠方式

Python 的模块在导入时缓存到 `sys.modules`，uvicorn workers 在启动时导入一次，之后不会重新加载。即使 `docker cp` 覆盖了文件，已运行的 workers 不会受影响。

**经验**: 任何 `.py` 文件修改后，必须 `docker restart` 才能生效。清理 `__pycache__` 可以避免缓存问题。

### 4. 日志是调试的关键

这次诊断的核心证据来自日志：
- `[DEBUG CALLBACK] Calling callback` ✅（callback 被调用）
- `[DEBUG CALLBACK] Callback returned` ✅（callback 返回了）
- `[CALLBACK ENTER]` ❌（callback 体内未执行——关键线索）

**经验**: 在关键路径上添加日志（函数入口、重要分支、跨模块调用），可以在生产环境中快速定位问题。

---

## 相关文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/core/orchestration/scheduler/unified_scheduler.py` | 修改 | 新增 `_on_task_completed_or_failed_sync` 方法，callback 注册改为同步版本 |
| `src/core/orchestration/scheduler/worker_manager.py` | 无变更 | 保持同步调用方式（这是正确的） |
| `src/gateway/web/data_collection_scheduler_manager.py` | 修改 | `on_task_completed` 改用 `update_task_status()` |

---

## 附录: 调试命令

### 检查调度器状态
```bash
curl -s http://localhost:8000/api/v1/data/scheduler/status | jq '.scheduler_status.tasks'
```

### 检查容器日志
```bash
docker logs rqa2025-app --since 2026-04-18T16:22:00Z --tail 200
```

### 检查回调是否正确触发
```bash
docker logs rqa2025-app --since 2026-04-18T16:22:00Z 2>&1 | findstr "TaskManager浠诲姟鐘舵�佸凡鏇存柊"
```

### 强制重启容器
```bash
docker restart rqa2025-app && sleep 5 && curl -s http://localhost:8000/api/v1/data/scheduler/status
```

# 最终检查报告：数据采集写入 akshare_stock_data 表的实现状态

## 检查日期
2026-02-20

## 检查范围
1. 数据源配置 (data-sources-config)
2. 数据采集调度器 (unified_scheduler)
3. 数据采集工作器 (data_collector worker)
4. 数据写入逻辑 (akshare_stock_data)

## 检查结果

### 1. 数据源配置 ✅ 已实现
**文件**: `data/data_sources_config.json`

**状态**: 已配置 16 个数据源
- AKShare A股数据
- AKShare 港股数据
- AKShare 指数数据
- ...

**结论**: 数据源配置已完成

---

### 2. 数据采集调度器 ✅ 已实现
**文件**: `src/distributed/coordinator/unified_scheduler.py`

**已实现功能**:
- ✅ 任务提交: `submit_task()`
- ✅ 任务队列管理: `_task_queues`
- ✅ 任务分发: `_auto_route_tasks()`
- ✅ 任务状态跟踪: `complete_task()`
- ✅ 任务类型映射: `TaskType.DATA_COLLECTION` → `WorkerType.DATA_COLLECTOR`

**代码证据**:
```python
class TaskType(Enum):
    DATA_COLLECTION = "data_collection"  # 数据采集

TASK_TYPE_TO_WORKER_TYPE = {
    TaskType.DATA_COLLECTION: WorkerType.DATA_COLLECTOR,
}
```

**结论**: 调度器机制完整，可以正常分发数据采集任务

---

### 3. 数据采集工作器 ❌ 未实现
**预期文件**: `src/distributed/worker/data_collector_worker.py`

**实际状态**: 
- ❌ 文件不存在
- ❌ 没有 DataCollectorWorker 类实现
- ❌ 没有 Worker 注册到 UnifiedWorkerRegistry

**代码证据**:
```python
# unified_scheduler.py 第 77 行
data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
logger.info(f"👷 当前数据采集器数量: {len(data_collectors)}")
# 输出: 当前数据采集器数量: 0
```

**结论**: 数据采集工作器未实现，没有 Worker 可以处理数据采集任务

---

### 4. 数据写入逻辑 ❌ 未实现
**预期位置**: DataCollectorWorker 或 DataAdapter

**实际状态**:
- ❌ `src/data/china/china_data_adapter.py`: 空壳实现
- ❌ `src/data/adapters/market_data_adapter.py`: 抽象基类，只有接口
- ❌ 没有找到 `INSERT INTO akshare_stock_data` 的代码
- ❌ 没有找到使用 `akshare` 库采集数据的代码

**代码证据**:
```python
# china_data_adapter.py
class ChinaDataAdapter:
    """空壳中国数据适配器，待实现"""
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
```

**结论**: 数据写入逻辑未实现

---

## 数据流分析

### 完整数据流（设计）
```
用户/API → DataCollectionService → UnifiedScheduler → DataCollectorWorker → DataAdapter → PostgreSQL (akshare_stock_data)
```

### 实际数据流（现状）
```
用户/API → DataCollectionService → UnifiedScheduler → ❌ (任务队列中，无Worker处理)
```

**断点**: 调度器将任务放入队列后，没有 DataCollectorWorker 来获取和处理任务

---

## 缺失组件清单

| 组件 | 状态 | 优先级 | 说明 |
|------|------|--------|------|
| DataCollectorWorker | ❌ 未实现 | 高 | 需要创建 Worker 类，注册到 Registry |
| AKShare 采集逻辑 | ❌ 未实现 | 高 | 需要实现 `akshare.stock_zh_a_hist()` 调用 |
| 数据写入逻辑 | ❌ 未实现 | 高 | 需要实现 `INSERT INTO akshare_stock_data` |
| Worker 注册 | ❌ 未实现 | 高 | 需要在系统启动时注册 DataCollectorWorker |
| 数据库表 | ❓ 待确认 | 中 | 需要确认 `akshare_stock_data` 表是否存在 |

---

## 建议实施计划

### Phase 1: 创建 DataCollectorWorker（2-3小时）
1. 创建 `src/distributed/worker/data_collector_worker.py`
2. 实现 `DataCollectorWorker` 类
3. 实现 `process_task()` 方法
4. 注册 Worker 到 UnifiedWorkerRegistry

### Phase 2: 实现数据采集逻辑（2-3小时）
1. 创建 `src/data/collectors/akshare_collector.py`
2. 实现 `collect_stock_data()` 方法
3. 使用 `akshare.stock_zh_a_hist()` 获取数据
4. 数据清洗和转换

### Phase 3: 实现数据写入逻辑（1-2小时）
1. 实现 `save_to_database()` 方法
2. 实现 `INSERT INTO akshare_stock_data` 逻辑
3. 处理冲突（ON CONFLICT UPDATE）
4. 添加错误处理和日志

### Phase 4: 集成和测试（1-2小时）
1. 集成所有组件
2. 手动触发数据采集
3. 验证数据已写入数据库
4. 测试信号生成功能

**总计**: 6-10 小时

---

## 验证方法

### 1. 检查 Worker 是否注册
```python
from src.distributed.registry import get_unified_worker_registry, WorkerType
registry = get_unified_worker_registry()
workers = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
print(f"数据采集器数量: {len(workers)}")
# 预期: 数据采集器数量: >= 1
```

### 2. 检查数据库是否有数据
```sql
SELECT COUNT(*) FROM akshare_stock_data;
-- 预期: count > 0
```

### 3. 检查信号生成是否使用数据库数据
```python
# 查看日志
# 预期: "成功获取股票数据: 000001, 记录数: XX"
```

---

## 最终结论

**数据采集写入 `akshare_stock_data` 表的逻辑尚未实现。**

系统有完整的任务调度和分发机制（UnifiedScheduler），但缺少实际执行数据采集和存储的 Worker 组件。需要按照上述实施计划完成开发。

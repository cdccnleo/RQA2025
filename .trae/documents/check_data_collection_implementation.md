# 数据采集服务实现状态检查计划

## 检查目标
确认数据采集服务是否已经实现了填充 `akshare_stock_data` 表的功能。

## 当前发现

### 1. 数据采集服务已实现
**文件**: `src/gateway/web/data_collectors.py`

**核心函数**:
- `collect_data_via_data_layer()` - 通过数据层采集数据
- `collect_single_batch()` - 采集单个批次数据
- `submit_data_collection_task()` - 提交数据采集任务到统一调度器

**数据流**:
```
数据采集请求 → DataCollectionService → UnifiedScheduler → DataCollector Worker → DataAdapter → DataLayer
```

### 2. 需要检查的关键点

#### 检查点 1: 数据层适配器是否写入 akshare_stock_data 表
**文件**: `src/adapter/unified_adapter_factory.py` 或相关数据适配器

**检查内容**:
- 数据适配器是否实现了 `collect_data()` 方法
- 是否将数据写入 `akshare_stock_data` 表
- 数据格式是否符合表结构

#### 检查点 2: 数据采集工作器是否启动
**文件**: `src/distributed/worker/data_collector_worker.py`

**检查内容**:
- 数据采集工作器是否已注册到 UnifiedWorkerRegistry
- 是否能正常接收和处理数据采集任务
- 是否调用数据层适配器存储数据

#### 检查点 3: 统一调度器是否正常运行
**文件**: `src/distributed/coordinator/unified_scheduler.py`

**检查内容**:
- 调度器是否已启动
- 是否能正常分发数据采集任务
- 任务状态是否正常更新

#### 检查点 4: 数据库表结构是否正确
**表**: `akshare_stock_data`

**检查内容**:
- 表是否存在
- 列结构是否符合要求
- 是否有数据

## 检查步骤

### 步骤 1: 检查数据层适配器
```python
# 检查数据适配器实现
# 文件: src/adapter/unified_adapter_factory.py
# 或: src/data/adapter/*.py

# 查找 collect_data 方法实现
# 检查是否写入 akshare_stock_data 表
```

### 步骤 2: 检查数据采集工作器
```python
# 检查数据采集工作器
# 文件: src/distributed/worker/data_collector_worker.py

# 检查是否注册到 WorkerRegistry
# 检查是否能处理 DATA_COLLECTION 任务
```

### 步骤 3: 检查统一调度器状态
```python
# 检查调度器状态
# 通过 API: /api/v1/scheduler/status
# 或通过日志
```

### 步骤 4: 检查数据库表
```sql
-- 检查表结构
\d akshare_stock_data

-- 检查是否有数据
SELECT COUNT(*) FROM akshare_stock_data;

-- 检查最近的数据
SELECT * FROM akshare_stock_data ORDER BY date DESC LIMIT 5;
```

## 可能的缺失点

### 情况 1: 数据适配器未实现写入逻辑
**解决方案**: 在数据适配器中添加写入 `akshare_stock_data` 表的逻辑

### 情况 2: 数据采集工作器未启动
**解决方案**: 启动数据采集工作器并注册到 WorkerRegistry

### 情况 3: 统一调度器未运行
**解决方案**: 启动统一调度器

### 情况 4: 数据流断裂
**解决方案**: 检查数据流各环节，修复断裂点

## 下一步行动

根据检查结果，确定需要实施的工作：

1. **如果数据适配器未实现写入**: 实现数据写入逻辑
2. **如果工作器未启动**: 启动并注册工作器
3. **如果调度器未运行**: 启动调度器
4. **如果表结构不正确**: 修复表结构
5. **如果全部正常但未采集**: 手动触发数据采集任务

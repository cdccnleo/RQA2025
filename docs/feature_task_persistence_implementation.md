# 特征提取任务持久化实现总结

## 实现时间
2025年1月7日

## 功能概述

实现了特征工程监控仪表盘中特征提取任务的完整持久化功能，确保任务创建、状态更新和查询都能正确保存和加载。

## 实现内容

### 1. 持久化模块 (`feature_task_persistence.py`)

**核心功能**:

1. **`save_feature_task(task)`** - 保存特征提取任务
   - 保存到文件系统（JSON格式）
   - 同时保存到PostgreSQL（如果可用）
   - 自动处理时间戳转换

2. **`load_feature_task(task_id)`** - 加载单个任务
   - 优先从PostgreSQL加载
   - 如果PostgreSQL不可用，从文件系统加载

3. **`list_feature_tasks(status, limit)`** - 列出任务
   - 支持按状态过滤
   - 支持限制返回数量
   - 优先从PostgreSQL加载，文件系统补充

4. **`update_feature_task(task_id, updates)`** - 更新任务
   - 更新任务状态、进度等信息
   - 自动更新 `updated_at` 时间戳

5. **`delete_feature_task(task_id)`** - 删除任务
   - 从文件系统和PostgreSQL同时删除

**存储策略**:
- **文件系统**: `data/feature_tasks/{task_id}.json`
- **PostgreSQL**: `feature_engineering_tasks` 表
- **双重存储**: 确保数据可靠性，PostgreSQL优先，文件系统作为备份

### 2. 服务层集成 (`feature_engineering_service.py`)

**修改的函数**:

1. **`get_feature_tasks()`**
   - ✅ 优先从持久化存储加载任务
   - ✅ 如果特征引擎有任务，自动保存到持久化存储
   - ✅ 确保任务数据不丢失

2. **`create_feature_task()`**
   - ✅ 创建任务后立即持久化
   - ✅ 同时保存到文件系统和PostgreSQL
   - ✅ 确保任务创建后可以恢复

3. **`stop_feature_task()`**
   - ✅ 停止任务后更新持久化存储
   - ✅ 更新任务状态为"stopped"
   - ✅ 记录结束时间

### 3. PostgreSQL表结构

```sql
CREATE TABLE feature_engineering_tasks (
    task_id VARCHAR(100) PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    progress INTEGER DEFAULT 0,
    feature_count INTEGER DEFAULT 0,
    start_time BIGINT,
    end_time BIGINT,
    config JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_feature_tasks_status ON feature_engineering_tasks(status);
CREATE INDEX idx_feature_tasks_created ON feature_engineering_tasks(created_at DESC);
```

## 数据流

### 任务创建流程

```
用户创建任务
    ↓
create_feature_task()
    ↓
生成task_id和任务信息
    ↓
save_feature_task()
    ↓
┌─────────────────┬─────────────────┐
│   文件系统       │   PostgreSQL     │
│  JSON文件保存   │   数据库保存     │
└─────────────────┴─────────────────┘
    ↓
返回任务信息给用户
```

### 任务查询流程

```
用户查询任务列表
    ↓
get_feature_tasks()
    ↓
list_feature_tasks()
    ↓
┌─────────────────┐
│  PostgreSQL     │ ← 优先
└─────────────────┘
    ↓ (如果PostgreSQL无数据)
┌─────────────────┐
│   文件系统       │ ← 备用
└─────────────────┘
    ↓
返回任务列表
```

### 任务更新流程

```
任务状态变化
    ↓
update_feature_task()
    ↓
加载现有任务
    ↓
更新字段
    ↓
save_feature_task()
    ↓
同时更新文件系统和PostgreSQL
```

## 任务数据结构

```json
{
    "task_id": "task_1704643200",
    "task_type": "技术指标",
    "status": "running",
    "progress": 45,
    "feature_count": 12,
    "start_time": 1704643200,
    "end_time": null,
    "config": {
        "indicators": ["MA", "RSI", "MACD"],
        "symbols": ["000001", "000002"]
    },
    "error_message": null,
    "saved_at": 1704643200.123,
    "updated_at": 1704643250.456
}
```

## 特性

### 1. 双重存储保障
- PostgreSQL作为主存储，提供查询性能
- 文件系统作为备份，确保数据不丢失
- 自动故障转移：PostgreSQL不可用时使用文件系统

### 2. 自动时间戳管理
- `start_time`: 任务开始时间（Unix时间戳）
- `end_time`: 任务结束时间（Unix时间戳）
- `created_at`: 任务创建时间（PostgreSQL自动管理）
- `updated_at`: 任务更新时间（自动更新）
- `saved_at`: 文件系统保存时间

### 3. 状态管理
- `pending`: 待执行
- `running`: 运行中
- `completed`: 已完成
- `stopped`: 已停止
- `failed`: 失败

### 4. 错误处理
- 数据库连接失败时自动降级到文件系统
- 文件系统操作失败时记录错误但不影响主流程
- 所有操作都有异常处理和日志记录

## 使用示例

### 创建任务

```python
from src.gateway.web.feature_engineering_service import create_feature_task

task = create_feature_task(
    task_type="技术指标",
    config={
        "indicators": ["MA", "RSI"],
        "symbols": ["000001"]
    }
)
# 任务已自动持久化
```

### 查询任务

```python
from src.gateway.web.feature_engineering_service import get_feature_tasks

# 获取所有任务
tasks = get_feature_tasks()

# 从持久化存储直接查询
from src.gateway.web.feature_task_persistence import list_feature_tasks

# 获取运行中的任务
running_tasks = list_feature_tasks(status="running", limit=10)
```

### 更新任务

```python
from src.gateway.web.feature_task_persistence import update_feature_task

# 更新任务进度
update_feature_task("task_1704643200", {
    "progress": 75,
    "feature_count": 20
})
```

## 验证方法

### 1. 测试任务创建和持久化

```python
from src.gateway.web.feature_engineering_service import create_feature_task
from src.gateway.web.feature_task_persistence import load_feature_task

# 创建任务
task = create_feature_task("技术指标", {"test": True})
task_id = task["task_id"]

# 验证持久化
loaded_task = load_feature_task(task_id)
assert loaded_task is not None
assert loaded_task["task_id"] == task_id
```

### 2. 测试任务列表查询

```python
from src.gateway.web.feature_task_persistence import list_feature_tasks

# 列出所有任务
all_tasks = list_feature_tasks()

# 列出运行中的任务
running_tasks = list_feature_tasks(status="running")
```

### 3. 检查持久化文件

```bash
# 检查文件系统
ls -la data/feature_tasks/

# 检查PostgreSQL
psql -d rqa2025 -c "SELECT * FROM feature_engineering_tasks LIMIT 10;"
```

## 与仪表盘的集成

### 前端API调用

```javascript
// 创建任务
fetch('/api/v1/features/engineering/tasks', {
    method: 'POST',
    body: JSON.stringify({
        task_type: '技术指标',
        config: {...}
    })
})

// 查询任务列表
fetch('/api/v1/features/engineering/tasks')
    .then(res => res.json())
    .then(data => {
        // 任务已从持久化存储加载
        console.log(data.tasks);
    })
```

### 任务状态更新

当特征引擎更新任务状态时，服务层会自动调用 `update_feature_task()` 更新持久化存储，确保仪表盘显示的是最新状态。

## 优势

1. **数据可靠性**: 双重存储确保数据不丢失
2. **性能优化**: PostgreSQL提供快速查询
3. **容错能力**: 自动故障转移
4. **易于扩展**: 可以轻松添加更多存储后端
5. **符合系统要求**: 使用真实数据，不使用模拟数据

## 注意事项

1. **数据一致性**: 文件系统和PostgreSQL可能短暂不一致，但最终会同步
2. **性能考虑**: 大量任务时，建议使用PostgreSQL查询
3. **清理策略**: 建议定期清理已完成或失败的任务，避免存储空间浪费

## 后续优化建议

1. **任务归档**: 实现任务归档机制，将旧任务移到归档存储
2. **批量操作**: 支持批量更新和查询
3. **任务历史**: 记录任务状态变更历史
4. **监控告警**: 监控持久化存储的健康状态

## 总结

特征提取任务持久化功能已完整实现，确保：
- ✅ 任务创建后立即持久化
- ✅ 任务状态更新时同步持久化
- ✅ 任务查询优先从持久化存储加载
- ✅ 双重存储保障数据可靠性
- ✅ 符合量化交易系统要求：使用真实数据


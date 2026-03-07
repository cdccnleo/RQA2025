# 基础设施层接口文档

## 1. 统一数据库管理器（UnifiedDatabaseManager）

### 1.1 初始化

```python
from src.infrastructure.database.unified_database_manager import UnifiedDatabaseManager

config = {
    "sqlite": {"db_path": "test.db"},
    "postgresql": {"host": "...", "port": 5432, ...},
    "redis": {"host": "...", "port": 6379, ...},
    "influxdb": {"url": "...", "token": "...", ...}
}
manager = UnifiedDatabaseManager(config)
```

### 1.2 获取适配器

```python
sqlite_adapter = manager.get_adapter("sqlite")
postgres_adapter = manager.get_adapter("postgresql")
```

### 1.3 适配器通用接口

- `connect(config: dict) -> bool`
- `disconnect() -> bool`
- `write(...)`
- `query(...)`
- `health_check() -> HealthCheckResult`
- `get_connection_info() -> dict`

### 1.4 健康检查

```python
result = sqlite_adapter.health_check()
print(result['status'])  # 'connected' or 'error'
print(result['response_time'])
```

### 1.5 错误处理

所有适配器方法均支持异常捕获，错误会通过 ErrorHandler 统一处理并记录日志。

---

## 2. 监控与配置模块接口（简要）

### 2.1 PerformanceMonitor

- `collect_metrics()`
- `report_metrics()`
- `get_metrics_dict()`

### 2.2 ConfigManager

- `load_config(path: str)`
- `reload_config()`
- `get(key: str, default=None)`

---

## 3. 典型用法示例

### 3.1 写入与查询

```python
# 写入
sqlite_adapter.write("measurement", {"field1": 1.0}, {"tag1": "A"})

# 查询
rows = sqlite_adapter.query("SELECT * FROM time_series")
```

### 3.2 连接池与事务

```python
pool = manager.get_connection_pool("sqlite")
with pool.connection() as conn:
    # do something
    pass
```

---

## 4. 健康检查结果结构

```python
{
    "status": "connected",
    "response_time": 0.01,
    "error_count": 0,
    "active_connections": 1,
    "total_connections": 1,
    "details": {...}
}
```

---

## 5. 常见异常与处理

- 连接失败：抛出异常并记录日志
- 查询/写入异常：通过 ErrorHandler 处理
- 健康检查异常：返回 status='error'，details 包含异常信息

---

> **如需详细接口参数、返回值、异常说明，请参考各适配器源码或补充详细文档。**
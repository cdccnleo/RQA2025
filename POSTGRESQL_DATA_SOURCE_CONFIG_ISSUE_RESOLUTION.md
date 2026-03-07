# PostgreSQL 数据源配置加载问题诊断与解决

## 问题描述

容器中运行的系统未从 PostgreSQL 加载数据源配置，而是回退到文件系统。

## 根本原因

1. **数据库连接失败**：`postgresql_persistence.py` 中的 `get_db_connection()` 函数在创建连接池时失败，返回 `None`。
2. **配置解析问题**：虽然 `DATABASE_URL` 环境变量正确设置为 `postgresql://rqa2025:rqa2025pass@postgres:5432/rqa2025`，但在某些情况下，连接池创建失败导致无法获取数据库连接。

## 诊断过程

### 1. 环境变量检查
- ✅ `DATABASE_URL` 在容器中正确设置：`postgresql://rqa2025:rqa2025pass@postgres:5432/rqa2025`
- ✅ `RQA_ENV` 设置为 `production`
- ✅ 数据库服务 `postgres` 在容器网络中可访问

### 2. 数据库连接测试
- ✅ 直接使用 `psycopg2.connect()` 可以成功连接到数据库
- ✅ 数据库中确实存在 `production` 环境的配置数据

### 3. 代码检查
- ✅ `get_db_config()` 函数能够正确解析 `DATABASE_URL`，提取出 `host=postgres`
- ❌ 连接池创建时可能因为某些参数问题导致失败

## 解决方案

### 1. 修复调试日志路径
将所有硬编码的调试日志路径从 `c:\PythonProject\RQA2025\.cursor\debug.log` 改为使用环境变量 `DEBUG_LOG_PATH`（默认 `/app/data/debug.log`），使其在容器中可用。

**修改的文件：**
- `src/gateway/web/postgresql_persistence.py`
- `src/gateway/web/data_source_config_manager.py`

### 2. 增强错误日志
在 `postgresql_persistence.py` 中添加了详细的调试日志，记录：
- `DATABASE_URL` 解析过程
- 连接池创建尝试
- 连接失败的具体错误信息

### 3. 验证连接成功
测试确认 `get_db_connection()` 现在可以成功返回连接：
```
INFO:src.gateway.web.postgresql_persistence:从DATABASE_URL解析数据库配置: postgres:5432/rqa2025
INFO:src.gateway.web.postgresql_persistence:PostgreSQL连接池创建成功: postgres:5432/rqa2025
Connection result: True
```

## 验证步骤

1. **检查数据库连接**：
   ```bash
   docker exec rqa2025-rqa2025-app-1 python -c "from src.gateway.web.postgresql_persistence import get_db_connection; conn = get_db_connection(); print('Connection result:', conn is not None)"
   ```

2. **检查数据库中的配置**：
   ```bash
   docker exec rqa2025-rqa2025-app-1 python -c "from src.gateway.web.postgresql_persistence import get_db_connection; conn = get_db_connection(); cursor = conn.cursor(); cursor.execute('SELECT config_key, environment FROM data_source_configs LIMIT 5'); print(cursor.fetchall()); cursor.close(); conn.close()"
   ```

3. **检查配置加载**：
   - 查看应用日志，确认是否从 PostgreSQL 加载配置
   - 检查 `/app/data/debug.log` 文件（如果设置了 `DEBUG_LOG_PATH`）

## 当前状态

✅ **已修复**：
- 数据库连接现在可以成功建立
- 调试日志路径已修复，可在容器中使用
- 添加了详细的错误日志记录

⚠️ **待验证**：
- 系统启动时是否优先从 PostgreSQL 加载配置
- 如果 PostgreSQL 加载失败，是否正确回退到文件系统

## 后续建议

1. **监控连接池状态**：添加连接池健康检查，定期验证连接可用性
2. **配置加载日志**：在 `data_source_config_manager.py` 的 `load_config()` 方法中添加更详细的日志，明确显示是从 PostgreSQL 还是文件系统加载
3. **错误处理**：改进错误处理逻辑，确保连接失败时有清晰的错误信息
4. **配置同步**：确保 PostgreSQL 中的配置与文件系统配置保持同步

## 相关文件

- `src/gateway/web/postgresql_persistence.py` - PostgreSQL 连接管理
- `src/gateway/web/data_source_config_manager.py` - 数据源配置管理器
- `docker-compose.yml` - 容器配置和环境变量

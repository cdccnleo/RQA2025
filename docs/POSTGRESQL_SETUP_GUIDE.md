# PostgreSQL + TimescaleDB 设置指南

## ✅ 当前状态

PostgreSQL服务已成功启动并初始化！

### 数据库信息

- **容器名称**: `rqa2025-postgres-1`
- **数据库**: `rqa2025`
- **用户**: `rqa2025`
- **密码**: `rqa2025pass`
- **端口**: `5432` (已映射到主机)
- **PostgreSQL版本**: 15.15
- **TimescaleDB**: 未安装（可选，不影响功能）

### 表结构

- ✅ `akshare_stock_data` 表已创建
- ✅ 6个索引已创建
- ✅ UNIQUE约束已设置（防止重复数据）
- ✅ 统计视图已创建

## 🔧 配置应用使用PostgreSQL

### 方式1: 环境变量（推荐）

在启动应用前设置环境变量：

**Windows PowerShell**:
```powershell
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="rqa2025"
$env:DB_USER="rqa2025"
$env:DB_PASSWORD="rqa2025pass"
python scripts/start_production.py
```

**Windows CMD**:
```cmd
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=rqa2025
set DB_USER=rqa2025
set DB_PASSWORD=rqa2025pass
python scripts/start_production.py
```

**Linux/Mac**:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=rqa2025
export DB_USER=rqa2025
export DB_PASSWORD=rqa2025pass
python scripts/start_production.py
```

### 方式2: .env文件

创建 `.env` 文件在项目根目录：

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rqa2025
DB_USER=rqa2025
DB_PASSWORD=rqa2025pass
```

然后使用 `python-dotenv` 加载（如果应用支持）。

### 方式3: 使用默认配置

如果不设置环境变量，系统会使用默认配置：
- Host: localhost
- Port: 5432
- Database: rqa2025
- User: rqa_user
- Password: (空)

**注意**: 当前Docker容器使用的是 `rqa2025` 用户，密码是 `rqa2025pass`，所以需要设置环境变量。

## 🚀 验证配置

### 1. 测试数据库连接

```bash
python scripts/test_pg_persistence_direct.py
```

### 2. 测试API持久化

设置环境变量后，重启API服务：

```bash
# 停止当前服务
taskkill /f /im python.exe

# 设置环境变量并启动
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="rqa2025"
$env:DB_USER="rqa2025"
$env:DB_PASSWORD="rqa2025pass"
python scripts/start_production.py
```

然后运行测试：

```bash
python scripts/test_postgresql_persistence.py
```

### 3. 检查数据库数据

```bash
docker exec rqa2025-postgres-1 psql -U rqa2025 -d rqa2025 -c "SELECT COUNT(*) FROM akshare_stock_data;"
```

## 📊 数据库管理命令

### 连接数据库

```bash
docker exec -it rqa2025-postgres-1 psql -U rqa2025 -d rqa2025
```

### 查看表结构

```sql
\d akshare_stock_data
```

### 查看数据

```sql
SELECT * FROM akshare_stock_data LIMIT 10;
```

### 查看统计信息

```sql
SELECT * FROM v_akshare_stock_summary;
SELECT * FROM v_akshare_collection_stats;
```

### 清空测试数据

```sql
TRUNCATE TABLE akshare_stock_data;
```

## 🔍 故障排除

### 问题1: 连接被拒绝

**症状**: `Connection refused`

**解决**:
1. 检查容器是否运行: `docker ps | grep postgres`
2. 检查端口映射: `docker port rqa2025-postgres-1`
3. 重启容器: `docker-compose restart postgres`

### 问题2: 认证失败

**症状**: `password authentication failed`

**解决**:
1. 确认用户名和密码正确
2. 检查环境变量是否设置
3. 使用容器内连接测试: `docker exec rqa2025-postgres-1 psql -U rqa2025 -d rqa2025`

### 问题3: 表不存在

**症状**: `relation "akshare_stock_data" does not exist`

**解决**:
```bash
docker cp scripts/sql/akshare_stock_data_schema.sql rqa2025-postgres-1:/tmp/
docker exec rqa2025-postgres-1 psql -U rqa2025 -d rqa2025 -f /tmp/akshare_stock_data_schema.sql
```

### 问题4: API仍使用文件存储

**症状**: 持久化结果中 `storage_type` 为 `file`

**解决**:
1. 确认环境变量已设置
2. 重启API服务
3. 检查日志中的错误信息
4. 运行 `python scripts/test_pg_persistence_direct.py` 验证连接

## 📝 下一步

1. ✅ PostgreSQL服务已启动
2. ✅ 数据库表已创建
3. ✅ 端口映射已配置
4. ⏭️ 设置环境变量并重启API服务
5. ⏭️ 测试API持久化功能
6. ⏭️ （可选）安装TimescaleDB扩展

## 📚 相关文档

- [持久化文档](AKSHARE_PERSISTENCE.md)
- [迁移指南](AKSHARE_POSTGRESQL_MIGRATION.md)
- [检查报告](POSTGRESQL_TIMESCALEDB_CHECK_REPORT.md)


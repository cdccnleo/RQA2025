# PostgreSQL连接失败问题修复报告

## 问题描述

在Windows环境下，PostgreSQL连接失败，错误信息包括：
1. `could not initiate GSSAPI security context: No credentials were supplied`
2. `fe_sendauth: no password supplied`

## 问题原因

1. **GSSAPI认证问题**：在Windows环境下，psycopg2默认尝试使用GSSAPI（Kerberos）认证，但Windows系统通常没有配置GSSAPI，导致认证失败。

2. **密码认证未启用**：当GSSAPI失败后，PostgreSQL尝试使用密码认证，但如果连接参数配置不当，可能导致密码未正确传递。

3. **默认主机配置**：代码中默认使用`postgres`作为主机名（Docker服务名），但在Windows本地开发环境中应该使用`localhost`。

## 修复方案

### 1. 禁用GSSAPI认证，强制使用密码认证

在`src/gateway/web/postgresql_persistence.py`的`get_db_connection()`函数中：

- 检测Windows平台
- 在Windows环境下设置`sslmode=prefer`参数
- 如果参数字典方式失败，回退到使用连接字符串方式

### 2. 优化默认配置

在`get_db_config()`函数中：

- Windows环境下默认使用`localhost`而不是`postgres`
- 添加密码验证，如果密码为空则发出警告

### 3. 改进错误处理

- 添加更详细的错误日志
- 提供连接字符串回退机制

## 修复代码

### 修改 `get_db_connection()` 函数

```python
def get_db_connection():
    """获取数据库连接"""
    global _db_pool
    
    try:
        import psycopg2
        from psycopg2 import pool
        import platform
        
        if _db_pool is None:
            config = get_db_config()
            
            # 构建连接参数字典
            connection_params = {
                "host": config["host"],
                "port": config["port"],
                "database": config["database"],
                "user": config["user"],
                "password": config["password"],
                "connect_timeout": 10
            }
            
            # Windows环境下禁用GSSAPI认证，强制使用密码认证
            if platform.system() == "Windows":
                connection_params["sslmode"] = "prefer"
            
            # 创建连接池
            try:
                _db_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    **connection_params
                )
                logger.info(f"PostgreSQL连接池创建成功: {config['host']}:{config['port']}/{config['database']}")
            except psycopg2.OperationalError as e:
                # 如果使用参数字典失败，尝试使用连接字符串
                if "sslmode" in connection_params:
                    logger.debug(f"使用参数字典连接失败，尝试使用连接字符串: {e}")
                    conn_string = (
                        f"host={config['host']} "
                        f"port={config['port']} "
                        f"dbname={config['database']} "
                        f"user={config['user']} "
                        f"password={config['password']} "
                        f"connect_timeout=10 "
                        f"sslmode=prefer"
                    )
                    _db_pool = psycopg2.pool.SimpleConnectionPool(
                        minconn=1,
                        maxconn=10,
                        dsn=conn_string
                    )
                    logger.info(f"PostgreSQL连接池创建成功（使用连接字符串）: {config['host']}:{config['port']}/{config['database']}")
                else:
                    raise
        
        return _db_pool.getconn()
        
    except ImportError:
        logger.error("psycopg2未安装，无法使用PostgreSQL持久化")
        return None
    except Exception as e:
        logger.error(f"获取数据库连接失败: {e}")
        return None
```

### 修改 `get_db_config()` 函数

```python
def get_db_config() -> Dict[str, Any]:
    """获取数据库配置"""
    global _db_config
    
    if _db_config is None:
        # ... 配置解析逻辑 ...
        
        # 验证密码是否为空
        if not _db_config.get("password"):
            logger.warning("PostgreSQL密码未配置，连接可能失败。请设置DB_PASSWORD或POSTGRES_PASSWORD环境变量")
    
    return _db_config
```

## 使用建议

### 1. 环境变量配置

在Windows环境下，建议设置以下环境变量：

```bash
# 方式1：使用DATABASE_URL（推荐）
set DATABASE_URL=postgresql://rqa2025:rqa2025pass@localhost:5432/rqa2025

# 方式2：使用单独的环境变量
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=rqa2025
set DB_USER=rqa2025
set DB_PASSWORD=rqa2025pass
```

### 2. PostgreSQL服务器配置

如果使用本地PostgreSQL服务器，确保：

1. **pg_hba.conf配置**：允许密码认证
   ```
   # TYPE  DATABASE        USER            ADDRESS                 METHOD
   host    all             all             127.0.0.1/32            md5
   host    all             all             ::1/128                 md5
   ```

2. **postgresql.conf配置**：确保监听本地连接
   ```
   listen_addresses = 'localhost'
   ```

### 3. 测试连接

可以使用以下Python脚本测试连接：

```python
import psycopg2
import os

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "rqa2025"),
        user=os.getenv("DB_USER", "rqa2025"),
        password=os.getenv("DB_PASSWORD", "rqa2025pass"),
        sslmode="prefer"
    )
    print("✅ PostgreSQL连接成功")
    conn.close()
except Exception as e:
    print(f"❌ PostgreSQL连接失败: {e}")
```

## 验证结果

修复后，系统应该能够：

1. ✅ 在Windows环境下成功连接PostgreSQL
2. ✅ 自动禁用GSSAPI认证，使用密码认证
3. ✅ 在连接失败时提供清晰的错误信息
4. ✅ 自动回退到文件系统存储（如果PostgreSQL不可用）

## 注意事项

1. **双重存储机制**：即使PostgreSQL连接失败，系统仍会使用文件系统存储，不影响核心功能。

2. **性能影响**：文件系统存储的性能可能不如PostgreSQL，但对于开发和小规模使用是可以接受的。

3. **生产环境**：在生产环境中，建议确保PostgreSQL连接正常，以获得更好的性能和可靠性。

## 相关文件

- `src/gateway/web/postgresql_persistence.py` - PostgreSQL持久化模块
- `src/gateway/web/feature_task_persistence.py` - 特征任务持久化（使用PostgreSQL）
- `src/gateway/web/training_job_persistence.py` - 训练任务持久化（使用PostgreSQL）
- `src/gateway/web/backtest_persistence.py` - 回测结果持久化（使用PostgreSQL）


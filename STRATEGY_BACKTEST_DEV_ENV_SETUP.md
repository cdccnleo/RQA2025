# 策略回测历史数据采集系统 - 开发环境搭建指南

## 📋 环境概述

**目标环境**: 本地开发环境 + 容器化部署
**操作系统**: Windows 10/11 + WSL2 (推荐) 或 Linux/macOS
**容器化**: Docker + Docker Compose
**开发工具**: Python 3.9+, VS Code, Git

## 🛠️ 环境准备

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **CPU** | 4核心 | 8核心+ |
| **内存** | 8GB | 16GB+ |
| **磁盘** | 50GB可用空间 | 100GB+ SSD |
| **网络** | 10Mbps带宽 | 50Mbps+带宽 |

### 软件依赖

#### **核心软件**
- **Python**: 3.9.0+
- **Docker**: 20.10.0+
- **Docker Compose**: 2.0.0+
- **Git**: 2.30.0+

#### **开发工具**
- **VS Code**: 1.70.0+
- **Python扩展**: Pylance, Python
- **Docker扩展**: Docker
- **Git扩展**: GitLens

#### **可选工具**
- **PostgreSQL客户端**: pgAdmin 或 DBeaver
- **Redis客户端**: RedisInsight
- **API测试**: Postman 或 Insomnia
- **监控工具**: Prometheus/Grafana (可选)

## 🚀 快速开始

### 步骤1: 克隆项目

```bash
# 克隆项目代码
git clone https://github.com/your-org/strategy-backtest-data-collection.git
cd strategy-backtest-data-collection

# 创建开发分支
git checkout -b feature/dev-env-setup
```

### 步骤2: 环境检查

```bash
# 检查Python版本
python --version
# 应显示: Python 3.9.x 或更高版本

# 检查Docker版本
docker --version
# 应显示: Docker version 20.10.x

# 检查Docker Compose版本
docker-compose --version
# 应显示: Docker Compose version 2.x.x

# 检查磁盘空间
df -h
# 确保有至少50GB可用空间
```

### 步骤3: 安装Python依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 步骤4: 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env.development

# 编辑环境变量文件
# Windows:
notepad .env.development
# Linux/macOS:
nano .env.development
```

**环境变量配置示例**:

```bash
# 应用配置
APP_NAME=strategy-backtest-collector
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# 服务器配置
HOST=0.0.0.0
PORT=8000
WORKERS=2

# 数据库配置 (PostgreSQL + TimescaleDB)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=backtest_db
POSTGRES_USER=backtest_user
POSTGRES_PASSWORD=dev_password_123
POSTGRES_URL=postgresql://backtest_user:dev_password_123@localhost:5432/backtest_db

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_URL=redis://localhost:6379

# MinIO配置 (对象存储)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=backtest_access
MINIO_SECRET_KEY=backtest_secret
MINIO_SECURE=false

# 数据源配置
AKSHARE_ENABLED=true
YAHOO_ENABLED=true
TUSHARE_ENABLED=true

# 安全配置 (开发环境)
JWT_SECRET_KEY=dev_jwt_secret_key_for_development_only
API_KEY=dev_api_key_for_development_only

# 监控配置
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_PORT=9090

# 开发配置
RELOAD=true
AUTO_RELOAD=true
```

### 步骤5: 启动基础服务

```bash
# 启动PostgreSQL + TimescaleDB
docker run -d \
  --name backtest-postgres \
  -e POSTGRES_DB=backtest_db \
  -e POSTGRES_USER=backtest_user \
  -e POSTGRES_PASSWORD=dev_password_123 \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15

# 启动Redis
docker run -d \
  --name backtest-redis \
  -p 6379:6379 \
  redis:7-alpine

# 启动MinIO
docker run -d \
  --name backtest-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ACCESS_KEY=backtest_access" \
  -e "MINIO_SECRET_KEY=backtest_secret" \
  minio/minio server /data --console-address ":9001"
```

### 步骤6: 初始化数据库

```bash
# 等待PostgreSQL启动
sleep 10

# 执行数据库初始化脚本
docker exec -i backtest-postgres psql -U backtest_user -d backtest_db < scripts/init_database.sql

# 或者使用Python脚本初始化
python scripts/init_database.py
```

### 步骤7: 启动应用服务

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 启动FastAPI应用
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 或者使用开发服务器脚本
python scripts/dev_server.py
```

### 步骤8: 验证环境

```bash
# 检查服务状态
curl http://localhost:8000/health

# 应该返回:
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "redis": "connected",
    "minio": "connected"
  }
}

# 测试API接口
curl http://localhost:8000/api/v1/status

# 测试数据采集接口
curl -X POST http://localhost:8000/api/v1/acquisition/start \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["000001.SZ"],
    "start_date": "2023-01-01",
    "end_date": "2023-01-05",
    "data_types": ["price"]
  }'
```

## 🐳 Docker Compose环境

### 使用Docker Compose快速搭建

**创建docker-compose.dev.yml**:

```yaml
version: '3.8'

services:
  # PostgreSQL + TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: backtest-postgres
    environment:
      POSTGRES_DB: backtest_db
      POSTGRES_USER: backtest_user
      POSTGRES_PASSWORD: dev_password_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init_database.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U backtest_user -d backtest_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: backtest-redis
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    container_name: backtest-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ACCESS_KEY: backtest_access
      MINIO_SECRET_KEY: backtest_secret
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_dev_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 应用服务
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: backtest-app
    environment:
      - ENVIRONMENT=development
      - POSTGRES_URL=postgresql://backtest_user:dev_password_123@postgres:5432/backtest_db
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=backtest_access
      - MINIO_SECRET_KEY=backtest_secret
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/venv
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    develop:
      watch:
        - action: sync
          path: .
          target: /app

volumes:
  postgres_dev_data:
  redis_dev_data:
  minio_dev_data:
```

**启动开发环境**:

```bash
# 启动所有服务
docker-compose -f docker-compose.dev.yml up -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml ps

# 查看服务日志
docker-compose -f docker-compose.dev.yml logs -f app
```

## 🔧 手动配置详解

### PostgreSQL + TimescaleDB配置

#### **安装TimescaleDB**
```bash
# 使用Docker安装
docker run -d \
  --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_DB=backtest_db \
  -e POSTGRES_USER=backtest_user \
  -e POSTGRES_PASSWORD=dev_password_123 \
  timescale/timescaledb:latest-pg15
```

#### **创建数据库和表结构**
```sql
-- 连接数据库
psql -h localhost -U backtest_user -d backtest_db

-- 启用TimescaleDB扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 创建股票价格数据表
CREATE TABLE IF NOT EXISTS stock_price_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    amount DECIMAL(20,4),
    adj_close DECIMAL(10,4),

    -- 数据标签
    data_source VARCHAR(50),
    collection_type VARCHAR(20),
    data_quality VARCHAR(20),

    PRIMARY KEY (symbol, time)
);

-- 转换为超表
SELECT create_hypertable('stock_price_data', 'time', partitioning_column => 'symbol', number_partitions => 100);

-- 创建索引
CREATE INDEX idx_stock_price_time_symbol ON stock_price_data (time DESC, symbol);
CREATE INDEX idx_stock_price_symbol_time ON stock_price_data (symbol, time DESC);
CREATE INDEX idx_stock_price_source ON stock_price_data (data_source);
```

### Redis配置

#### **安装和配置Redis**
```bash
# 使用Docker安装
docker run -d \
  --name redis-dev \
  -p 6379:6379 \
  redis:7-alpine redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

#### **Redis数据结构设计**
```python
# 缓存键设计
cache_keys = {
    'stock_price': 'stock:price:{symbol}:{date}',
    'hot_symbols': 'hot:symbols',
    'collection_stats': 'stats:collection:{date}',
    'data_quality': 'quality:{symbol}:{data_type}'
}
```

### MinIO配置

#### **安装MinIO**
```bash
# 使用Docker安装
docker run -d \
  --name minio-dev \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ACCESS_KEY=backtest_access" \
  -e "MINIO_SECRET_KEY=backtest_secret" \
  minio/minio server /data --console-address ":9001"
```

#### **创建存储桶**
```bash
# 安装MinIO客户端
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc

# 配置客户端
./mc alias set local http://localhost:9000 backtest_access backtest_secret

# 创建存储桶
./mc mb local/backtest-data
./mc mb local/backtest-backup
./mc mb local/backtest-temp
```

## 🧪 环境验证

### 自动化验证脚本

**创建环境验证脚本** `scripts/verify_dev_env.py`:

```python
#!/usr/bin/env python3
"""
开发环境验证脚本
"""

import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
import sys
from pathlib import Path

class DevEnvironmentVerifier:
    """开发环境验证器"""

    def __init__(self):
        self.results = {}

    async def verify_all(self):
        """验证所有组件"""
        print("🔍 开始验证开发环境...")

        # 验证数据库连接
        self.results['database'] = await self._verify_database()

        # 验证Redis连接
        self.results['redis'] = await self._verify_redis()

        # 验证MinIO连接
        self.results['minio'] = await self._verify_minio()

        # 验证API服务
        self.results['api'] = await self._verify_api()

        # 输出结果
        self._print_results()

        return all(self.results.values())

    async def _verify_database(self):
        """验证数据库连接"""
        try:
            conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                user='backtest_user',
                password='dev_password_123',
                database='backtest_db'
            )

            # 测试查询
            result = await conn.fetchval("SELECT version()")
            await conn.close()

            print(f"✅ 数据库连接成功: PostgreSQL {result.split()[1]}")
            return True

        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False

    async def _verify_redis(self):
        """验证Redis连接"""
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            await r.ping()

            # 测试读写
            await r.set('test_key', 'test_value')
            value = await r.get('test_key')
            await r.delete('test_key')

            print("✅ Redis连接成功")
            return True

        except Exception as e:
            print(f"❌ Redis连接失败: {e}")
            return False

    async def _verify_minio(self):
        """验证MinIO连接"""
        try:
            # 这里可以添加MinIO连接验证
            print("✅ MinIO连接验证 (暂略)")
            return True

        except Exception as e:
            print(f"❌ MinIO连接失败: {e}")
            return False

    async def _verify_api(self):
        """验证API服务"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'healthy':
                            print("✅ API服务运行正常")
                            return True

            print("❌ API服务响应异常")
            return False

        except Exception as e:
            print(f"❌ API服务连接失败: {e}")
            return False

    def _print_results(self):
        """输出验证结果"""
        print("\n" + "="*50)
        print("🏁 环境验证结果汇总")
        print("="*50)

        total = len(self.results)
        passed = sum(self.results.values())

        for component, status in self.results.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {component}: {'通过' if status else '失败'}")

        print(f"\n📊 总体结果: {passed}/{total} 组件验证通过")

        if passed == total:
            print("🎉 开发环境配置成功！可以开始开发工作。")
        else:
            print("⚠️  部分组件验证失败，请检查配置。")

if __name__ == '__main__':
    verifier = DevEnvironmentVerifier()
    success = asyncio.run(verifier.verify_all())
    sys.exit(0 if success else 1)
```

**运行验证脚本**:
```bash
python scripts/verify_dev_env.py
```

### 手动验证步骤

#### **1. 数据库验证**
```bash
# 连接数据库
psql -h localhost -U backtest_user -d backtest_db

# 测试查询
SELECT version();
\dt
\q
```

#### **2. Redis验证**
```bash
# 连接Redis
redis-cli

# 测试命令
PING
SET test_key "Hello World"
GET test_key
DEL test_key
QUIT
```

#### **3. MinIO验证**
访问 http://localhost:9001 进入MinIO管理界面

#### **4. API服务验证**
```bash
# 健康检查
curl http://localhost:8000/health

# API文档
open http://localhost:8000/docs
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### **端口冲突**
```bash
# 检查端口占用
netstat -ano | findstr :5432
netstat -ano | findstr :6379
netstat -ano | findstr :9000
netstat -ano | findstr :8000

# 杀死进程 (Windows)
taskkill /PID <PID> /F

# 修改端口配置
# 编辑 .env.development 文件，修改端口设置
```

#### **Docker权限问题**
```bash
# Windows: 以管理员身份运行
# Linux: 添加用户到docker组
sudo usermod -aG docker $USER
# 重新登录或运行: newgrp docker
```

#### **内存不足**
```bash
# 检查系统内存
free -h  # Linux
systeminfo | findstr Memory  # Windows

# 调整Docker内存限制
# Docker Desktop -> Settings -> Resources -> Memory
```

#### **网络连接问题**
```bash
# 检查Docker网络
docker network ls
docker network inspect bridge

# 重启Docker服务
sudo systemctl restart docker  # Linux
# Windows: 重启Docker Desktop
```

## 📚 开发资源

### 项目结构
```
strategy-backtest-data-collection/
├── src/
│   ├── core/
│   │   ├── orchestration/
│   │   │   ├── incremental_collection_strategy.py
│   │   │   ├── data_complement_scheduler.py
│   │   │   ├── batch_complement_processor.py
│   │   │   ├── market_adaptive_monitor.py
│   │   │   ├── ai_driven_scheduler.py
│   │   │   └── distributed_scheduler.py
│   │   └── ...
│   ├── gateway/
│   │   └── web/
│   │       ├── data_collectors.py
│   │       └── ...
│   └── infrastructure/
│       ├── logging/
│       ├── monitoring/
│       └── ...
├── scripts/
│   ├── dev_server.py
│   ├── verify_dev_env.py
│   └── ...
├── tests/
│   ├── unit/
│   ├── integration/
│   └── ...
├── config/
├── docs/
└── requirements.txt
```

### 开发规范
- **代码风格**: PEP 8
- **提交规范**: 遵循Conventional Commits
- **分支策略**: Git Flow
- **测试覆盖**: 单元测试覆盖率 ≥ 85%

### 团队培训资源
- [项目架构文档](./STRATEGY_BACKTEST_TECHNICAL_DESIGN.md)
- [API接口文档](./api_documentation.md)
- [测试用例说明](./testing_guide.md)
- [部署运维手册](./deployment_guide.md)

---

**环境搭建版本**: v1.0
**更新日期**: 2026-01-24
**适用环境**: 开发环境
**维护人员**: 开发团队
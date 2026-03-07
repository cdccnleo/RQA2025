# 🚀 RQA2025系统启动指南

## 📋 概述

RQA2025是一款完整的量化交易系统，采用分层架构设计，提供完整的交易功能。本指南将帮助您快速启动和运行系统。

## 🛠️ 系统要求

### 基本要求
- **Python**: 3.8+
- **操作系统**: Windows 10+ / Linux / macOS
- **内存**: 4GB+ 推荐8GB+
- **磁盘空间**: 2GB+

### 推荐配置（生产环境）
- **CPU**: 4核心+
- **内存**: 8GB+
- **数据库**: PostgreSQL 12+
- **缓存**: Redis 6+
- **磁盘**: SSD 20GB+

## 📦 安装依赖

### 1. 克隆项目
```bash
git clone <repository-url>
cd RQA2025
```

### 2. 安装Python依赖
```bash
# 推荐使用conda环境
conda create -n rqa2025 python=3.9
conda activate rqa2025

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装可选依赖（生产环境）
```bash
# PostgreSQL支持
pip install asyncpg psycopg2-binary

# Redis支持
pip install aioredis

# 性能优化
pip install numpy pandas numba
```

## 🗄️ 数据库设置

### 开发环境（SQLite）
系统默认使用SQLite，无需额外配置。

### 生产环境（PostgreSQL）

#### 1. 安装PostgreSQL
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib
sudo postgresql-setup initdb

# macOS
brew install postgresql
```

#### 2. 创建数据库和用户
```sql
-- 连接到PostgreSQL
sudo -u postgres psql

-- 创建数据库和用户
CREATE DATABASE rqa2025_prod;
CREATE USER rqa_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rqa2025_prod TO rqa_user;
```

#### 3. 设置环境变量
```bash
export DATABASE_PASSWORD="your_password"
export JWT_SECRET_KEY="your_jwt_secret"
export REDIS_PASSWORD="your_redis_password"  # 如果需要
```

## 🔧 配置系统

### 1. 环境变量
创建 `.env` 文件：
```bash
# 数据库配置
DATABASE_PASSWORD=your_db_password
JWT_SECRET_KEY=your_jwt_secret_key
REDIS_PASSWORD=your_redis_password

# 应用配置
RQA_ENV=production
RQA_LOG_LEVEL=INFO
```

### 2. 配置文件
复制并修改配置文件：
```bash
cp config/production.yml config/custom.yml
# 编辑 custom.yml 中的配置项
```

## 🚀 启动系统

### 开发环境启动
```bash
# 使用启动脚本
python scripts/start_rqa_system.py

# 或者直接运行
python src/app.py
```

### 生产环境启动
```bash
# 使用配置文件启动
python scripts/start_rqa_system.py --config config/production.yml --env production

# 指定端口和主机
python scripts/start_rqa_system.py --host 0.0.0.0 --port 8000 --env production
```

### Docker启动（推荐）
```bash
# 构建镜像
docker build -t rqa2025 .

# 运行容器
docker run -p 8000:8000 \
  -e DATABASE_PASSWORD=your_password \
  -e JWT_SECRET_KEY=your_secret \
  rqa2025
```

## 📖 访问系统

启动成功后，访问以下地址：

- **主页**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **交易API**: http://localhost:8000/api/v1/trading/docs
- **健康检查**: http://localhost:8000/health
- **系统指标**: http://localhost:8000/metrics

## 🧪 测试系统

### 1. 健康检查
```bash
curl http://localhost:8000/health
```

### 2. 用户注册
```bash
curl -X POST http://localhost:8000/api/v1/trading/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123",
    "initial_balance": 10000
  }'
```

### 3. 用户登录
```bash
curl -X POST http://localhost:8000/api/v1/trading/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "password123"
  }'
```

### 4. 创建订单
```bash
curl -X POST http://localhost:8000/api/v1/trading/orders \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol": "000001.SZ",
    "quantity": 100,
    "price": 10.50,
    "order_type": "limit",
    "side": "buy"
  }'
```

## 🔍 监控和日志

### 日志文件
- 应用日志：`logs/rqa2025.log`
- 错误日志：`logs/error.log`

### 监控指标
- 系统健康：http://localhost:8000/health
- 性能指标：http://localhost:8000/metrics
- Prometheus指标：http://localhost:9090 (如果启用)

## 🛠️ 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 查找占用端口的进程
netstat -tlnp | grep :8000
# 杀死进程或更换端口
python scripts/start_rqa_system.py --port 8001
```

#### 2. 数据库连接失败
```bash
# 检查PostgreSQL服务
sudo systemctl status postgresql

# 检查数据库是否存在
psql -U rqa_user -d rqa2025_prod -c "SELECT 1;"
```

#### 3. 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 4. 内存不足
```bash
# 检查内存使用
free -h

# 增加交换空间或升级内存
# 对于Docker，增加内存限制
docker run --memory=4g rqa2025
```

### 调试模式
```bash
# 启用调试日志
python scripts/start_rqa_system.py --log-level DEBUG

# 检查系统状态
python -c "from src.app import RQAApplication; import asyncio; asyncio.run(RQAApplication().initialize())"
```

## 📞 支持

如果遇到问题，请：

1. 查看日志文件：`logs/rqa2025.log`
2. 检查系统健康：`curl http://localhost:8000/health`
3. 查看API文档：http://localhost:8000/docs

## 🔄 更新和维护

### 系统更新
```bash
# 拉取最新代码
git pull origin main

# 重新安装依赖
pip install -r requirements.txt

# 重启服务
docker-compose restart rqa2025
```

### 数据备份
```bash
# PostgreSQL备份
pg_dump -U rqa_user rqa2025_prod > backup_$(date +%Y%m%d).sql

# Redis备份
redis-cli save
```

---

## 🎯 Phase 4B完成总结

✅ **已完成的工作：**

1. **核心数据库服务** - 实现了PostgreSQL + Redis的完整数据库架构
2. **REST API服务** - 创建了完整的交易API，支持用户管理、订单处理、持仓查询
3. **业务逻辑服务** - 实现了策略执行、组合再平衡、市场分析等核心业务功能
4. **应用框架** - 构建了FastAPI应用框架，支持异步处理和自动文档生成
5. **启动脚本** - 提供了完整的启动脚本和配置文件
6. **部署配置** - 创建了生产环境配置和Docker支持

🚀 **系统现已具备：**
- ✅ 用户注册和认证
- ✅ 实时交易订单处理
- ✅ 投资组合管理
- ✅ 量化策略执行框架
- ✅ 市场数据分析
- ✅ 系统健康监控
- ✅ 完整的API文档

**恭喜！RQA2025系统Phase 4B实现完成，可以开始Phase 4C质量提升专项了！** 🎉



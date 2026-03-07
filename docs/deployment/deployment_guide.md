# RQA2025 系统部署和运维指南

## 📋 目录

1. [概述](#概述)
2. [环境要求](#环境要求)
3. [快速开始](#快速开始)
4. [详细部署步骤](#详细部署步骤)
5. [配置管理](#配置管理)
6. [监控和维护](#监控和维护)
7. [故障排除](#故障排除)
8. [升级指南](#升级指南)
9. [安全配置](#安全配置)

## 📖 概述

RQA2025 是一个企业级的量化交易系统，采用分层架构设计，提供完整的交易策略开发、回测、实盘交易等功能。本指南将帮助您在各种环境中部署和维护系统。

### 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   策略层        │    │   交易层        │    │   数据层        │
│   ├─AI策略       │    │   ├─订单管理    │    │   ├─实时数据    │
│   ├─传统策略     │    │   ├─风险控制    │    │   ├─历史数据    │
│   └─组合优化     │    │   └─执行引擎    │    │   └─数据质量    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   基础设施层    │    │   监控层        │    │   风控层        │
│   ├─消息队列     │    │   ├─性能监控    │    │   ├─实时风控    │
│   ├─缓存系统     │    │   ├─业务监控    │    │   ├─合规检查    │
│   └─存储系统     │    │   └─告警系统    │    │   └─风险评估    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 环境要求

### 系统要求

| 组件 | 最低要求 | 推荐配置 | 生产环境 |
|------|----------|----------|----------|
| **CPU** | 4核 | 8核 | 16核+ |
| **内存** | 8GB | 16GB | 64GB+ |
| **存储** | 100GB SSD | 500GB SSD | 2TB+ NVMe |
| **网络** | 100Mbps | 1Gbps | 10Gbps |
| **操作系统** | Windows 10/Linux | Windows Server/Linux | Linux (RHEL/CentOS/Ubuntu) |

### 软件依赖

```yaml
# Python 环境
python: ">=3.9,<3.11"
pip: ">=21.0"

# 核心依赖包
dependencies:
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - tensorflow>=2.8.0
  - pytorch>=1.9.0
  - fastapi>=0.68.0
  - sqlalchemy>=1.4.0
  - redis>=4.0.0
  - kafka-python>=2.0.0
  - prometheus-client>=0.12.0

# 数据库
databases:
  - PostgreSQL >=13
  - Redis >=6.0
  - ClickHouse >=21.0 (可选，用于高性能分析)

# 消息队列
message_queue:
  - Apache Kafka >=2.8
  - RabbitMQ >=3.8

# 监控
monitoring:
  - Prometheus >=2.30
  - Grafana >=8.0
  - ELK Stack (可选)
```

## 🚀 快速开始

### 使用Docker部署（推荐）

```bash
# 1. 克隆代码库
git clone https://github.com/your-org/rqa2025.git
cd rqa2025

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置数据库、Redis等信息

# 3. 使用Docker Compose启动
docker-compose up -d

# 4. 验证部署
curl http://localhost:8000/health
```

### 手动部署

```bash
# 1. 安装Python环境
conda create -n rqa2025 python=3.9
conda activate rqa2025

# 2. 安装依赖
pip install -r requirements.txt

# 3. 初始化数据库
python -m alembic upgrade head

# 4. 启动服务
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. 启动监控
python scripts/monitoring/start_monitoring.py
```

## 📋 详细部署步骤

### 1. 环境准备

```bash
# 创建项目目录
mkdir -p /opt/rqa2025/{src,logs,data,config}
cd /opt/rqa2025

# 设置权限
chown -R rqa2025:rqa2025 /opt/rqa2025
chmod -R 755 /opt/rqa2025
```

### 2. 数据库初始化

```bash
# PostgreSQL 初始化
sudo -u postgres createdb rqa2025
sudo -u postgres createuser rqa2025_user
sudo -u postgres psql -c "ALTER USER rqa2025_user PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE rqa2025 TO rqa2025_user;"

# Redis 配置
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# 创建数据目录
mkdir -p /var/lib/rqa2025/{market_data,strategy_cache,model_store}
```

### 3. 应用部署

```bash
# 下载最新版本
wget https://github.com/your-org/rqa2025/releases/latest/download/rqa2025.tar.gz
tar -xzf rqa2025.tar.gz

# 安装Python依赖
pip install -r requirements.txt

# 运行数据库迁移
alembic upgrade head

# 配置systemd服务
sudo cp deploy/systemd/rqa2025.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rqa2025
sudo systemctl start rqa2025
```

### 4. 负载均衡配置

```nginx
# Nginx 配置示例
upstream rqa2025_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name trading.your-domain.com;

    location / {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://rqa2025_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## ⚙️ 配置管理

### 环境变量配置

创建 `.env` 文件：

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/rqa2025
REDIS_URL=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here

# 交易配置
TRADING_MODE=paper  # paper/live
BROKER_API_KEY=your-broker-key
BROKER_API_SECRET=your-broker-secret

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=/var/log/rqa2025/app.log
```

### 配置文件结构

```
/opt/rqa2025/config/
├── database.yaml      # 数据库配置
├── trading.yaml       # 交易配置
├── monitoring.yaml    # 监控配置
├── strategies.yaml    # 策略配置
└── security.yaml      # 安全配置
```

## 📊 监控和维护

### 健康检查端点

```bash
# API健康检查
curl http://localhost:8000/health

# 数据库连接检查
curl http://localhost:8000/health/database

# 外部服务检查
curl http://localhost:8000/health/services
```

### 监控指标

系统提供以下监控指标：

- **业务指标**: 交易量、成功率、盈亏情况
- **性能指标**: 响应时间、吞吐量、资源使用率
- **系统指标**: CPU、内存、磁盘、网络
- **错误指标**: 异常数量、错误率、重试次数

### 日志管理

```bash
# 查看应用日志
tail -f /var/log/rqa2025/app.log

# 查看错误日志
grep ERROR /var/log/rqa2025/app.log | tail -20

# 日志轮转配置
/var/log/rqa2025/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### 备份策略

```bash
# 数据库备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump rqa2025 > /backup/rqa2025_db_$DATE.sql

# 策略和模型备份
tar -czf /backup/rqa2025_strategies_$DATE.tar.gz /var/lib/rqa2025/strategies/
tar -czf /backup/rqa2025_models_$DATE.tar.gz /var/lib/rqa2025/models/

# 清理旧备份（保留30天）
find /backup -name "rqa2025_*" -mtime +30 -delete
```

## 🔧 故障排除

### 常见问题

#### 1. 服务启动失败

```bash
# 检查日志
journalctl -u rqa2025 -f

# 检查端口占用
netstat -tulpn | grep :8000

# 检查配置文件
python -c "import yaml; yaml.safe_load(open('config/main.yaml'))"
```

#### 2. 数据库连接问题

```bash
# 测试数据库连接
psql -h localhost -U rqa2025_user -d rqa2025 -c "SELECT 1;"

# 检查连接池
python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/rqa2025')
print('Connection successful' if engine.execute('SELECT 1').fetchone() else 'Connection failed')
"
```

#### 3. 内存不足

```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head -10

# 调整JVM参数（如果使用）
export JAVA_OPTS='-Xmx4g -Xms1g'

# 优化Python内存
export PYTHONOPTIMIZE=1
```

#### 4. 高CPU使用率

```bash
# 分析CPU使用
top -H
perf record -F 99 -p $(pgrep python) -g -- sleep 30
perf report

# 优化策略计算
# 1. 使用并行计算
# 2. 实现缓存机制
# 3. 优化算法复杂度
```

### 性能优化

```bash
# 数据库优化
# 1. 创建适当的索引
# 2. 优化查询语句
# 3. 配置连接池

# 缓存优化
# 1. 使用Redis缓存热点数据
# 2. 实现应用级缓存
# 3. 设置合理的TTL

# 异步处理
# 1. 使用异步任务队列
# 2. 实现流式处理
# 3. 避免阻塞操作
```

## ⬆️ 升级指南

### 版本升级流程

```bash
# 1. 备份当前版本
./backup.sh

# 2. 停止服务
sudo systemctl stop rqa2025

# 3. 下载新版本
wget https://github.com/your-org/rqa2025/releases/download/v2.0.0/rqa2025-v2.0.0.tar.gz

# 4. 备份配置文件
cp -r config config.backup

# 5. 部署新版本
tar -xzf rqa2025-v2.0.0.tar.gz
pip install -r requirements.txt --upgrade

# 6. 运行数据库迁移
alembic upgrade head

# 7. 恢复配置文件
cp config.backup/* config/

# 8. 启动服务
sudo systemctl start rqa2025

# 9. 验证升级
curl http://localhost:8000/health
```

### 回滚计划

```bash
# 快速回滚脚本
#!/bin/bash
echo "Rolling back to previous version..."

# 停止服务
sudo systemctl stop rqa2025

# 恢复备份
cp -r /backup/rqa2025_config_* /opt/rqa2025/config/
cp -r /backup/rqa2025_src_* /opt/rqa2025/src/

# 恢复数据库
psql -d rqa2025 < /backup/rqa2025_db_*.sql

# 重启服务
sudo systemctl start rqa2025

echo "Rollback completed. Please verify system functionality."
```

## 🔒 安全配置

### 网络安全

```bash
# 配置防火墙
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# SSL证书配置
sudo certbot --nginx -d trading.your-domain.com

# SSH强化
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

### 应用安全

```python
# 安全配置示例
from fastapi.security import HTTPBearer, HTTPBasic
from passlib.context import CryptContext

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
SECRET_KEY = "your-256-bit-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# CORS配置
ALLOWED_HOSTS = ["trading.your-domain.com"]
CORS_ORIGINS = ["https://trading.your-domain.com"]

# 速率限制
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # 秒
```

### 数据安全

```bash
# 数据库加密
# 启用SSL连接
ssl = True
ssl_cert = "/path/to/client-cert.pem"
ssl_key = "/path/to/client-key.pem"

# 数据脱敏
# 敏感数据加密存储
# 审计日志记录所有访问

# 备份加密
# 使用GPG加密备份文件
gpg --encrypt --recipient backup@your-domain.com backup.tar.gz
```

## 📞 支持和维护

### 联系信息

- **技术支持**: support@your-domain.com
- **紧急联系**: emergency@your-domain.com
- **文档**: https://docs.your-domain.com/rqa2025

### 维护窗口

- **日常维护**: 每周日 02:00-04:00
- **紧急维护**: 24/7 支持
- **计划维护**: 提前24小时通知

### 监控告警

系统配置了以下告警：

- 服务不可用 > 5分钟
- CPU使用率 > 90%
- 内存使用率 > 85%
- 磁盘空间 < 10%
- 交易失败率 > 5%
- API响应时间 > 2秒

---

## 📝 附录

### 端口使用说明

| 服务 | 端口 | 协议 | 说明 |
|------|------|------|------|
| API服务 | 8000 | HTTP/HTTPS | REST API和WebSocket |
| 监控面板 | 3000 | HTTP | Grafana监控面板 |
| Prometheus | 9090 | HTTP | 指标收集服务 |
| PostgreSQL | 5432 | TCP | 主数据库 |
| Redis | 6379 | TCP | 缓存和会话存储 |
| Kafka | 9092 | TCP | 消息队列 |

### 环境变量参考

```bash
# 必需环境变量
export RQA2025_ENV=production
export DATABASE_URL=postgresql://...
export REDIS_URL=redis://...
export SECRET_KEY=...

# 可选环境变量
export LOG_LEVEL=INFO
export WORKERS=4
export MAX_MEMORY=4GB
export ENABLE_PROFILING=false
```

### 健康检查脚本

```bash
#!/bin/bash
# 完整健康检查脚本

HEALTHY=true

# 检查API服务
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ API服务不可用"
    HEALTHY=false
fi

# 检查数据库
if ! psql -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ 数据库连接失败"
    HEALTHY=false
fi

# 检查Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis连接失败"
    HEALTHY=false
fi

if [ "$HEALTHY" = true ]; then
    echo "✅ 所有服务正常"
    exit 0
else
    echo "❌ 系统存在问题"
    exit 1
fi
```

---

*最后更新: 2025年12月7日 | 版本: v2.0.0*
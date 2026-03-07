# RQA2025量化交易系统生产环境运维手册

## 📋 目录

1. [系统概述](#系统概述)
2. [部署准备](#部署准备)
3. [部署流程](#部署流程)
4. [配置管理](#配置管理)
5. [监控告警](#监控告警)
6. [备份恢复](#备份恢复)
7. [故障处理](#故障处理)
8. [性能优化](#性能优化)
9. [安全运维](#安全运维)
10. [升级维护](#升级维护)

---

## 🎯 系统概述

### 架构说明
RQA2025是一款企业级的量化交易分析系统，采用21层微服务架构设计：

- **前端层**: Web界面、移动端API
- **网关层**: API网关、负载均衡、限流熔断
- **核心服务层**: 策略服务、风险控制、交易执行
- **数据层**: 实时数据处理、历史数据存储
- **基础设施层**: 缓存、消息队列、配置中心

### 技术栈
- **后端**: Python 3.9+, FastAPI, SQLAlchemy
- **数据库**: PostgreSQL 15, Redis 7
- **消息队列**: RabbitMQ, Kafka
- **监控**: Prometheus, Grafana, ELK Stack
- **容器化**: Docker, Kubernetes
- **CI/CD**: GitHub Actions, ArgoCD

### 性能指标
- **响应时间**: API平均 < 200ms, 95% < 500ms
- **可用性**: 99.9% SLA
- **并发处理**: 支持 10,000+ 并发交易
- **数据处理**: 实时处理 100,000+ 市场数据/秒

---

## 🛠️ 部署准备

### 1. 环境要求

#### 硬件配置
```yaml
生产环境推荐配置:
  CPU: 32核心以上
  内存: 128GB以上
  存储: 2TB SSD + 10TB HDD
  网络: 10Gbps带宽

容灾环境配置:
  CPU: 16核心以上
  内存: 64GB以上
  存储: 1TB SSD + 5TB HDD
  网络: 1Gbps带宽
```

#### 软件依赖
```bash
# 系统软件
- Ubuntu 20.04 LTS / CentOS 8+
- Docker 24.0+
- Docker Compose 2.0+
- Kubernetes 1.24+
- Nginx 1.20+
- PostgreSQL 15+
- Redis 7+

# Python环境
- Python 3.9.0+
- pip 21.0+
- virtualenv 20.0+
```

### 2. 网络配置

#### 安全组配置
```yaml
入站规则:
  - 端口 80: HTTP访问
  - 端口 443: HTTPS访问
  - 端口 22: SSH管理 (仅限 bastion host)
  - 端口 9090: Prometheus监控
  - 端口 3000: Grafana仪表板

出站规则:
  - 允许所有出站流量 (数据源、外部API调用)
```

#### 域名配置
```bash
# 主域名
trading.rqa2025.com -> Load Balancer

# 子域名
api.trading.rqa2025.com -> API Gateway
monitor.trading.rqa2025.com -> Monitoring Dashboard
grafana.trading.rqa2025.com -> Grafana
```

### 3. SSL证书配置

```bash
# Let's Encrypt自动证书
certbot certonly --nginx -d trading.rqa2025.com

# 或使用商业证书
# 将证书文件放置到:
/etc/ssl/certs/rqa2025.crt
/etc/ssl/private/rqa2025.key
```

---

## 🚀 部署流程

### 1. 环境初始化

```bash
# 1. 克隆代码库
git clone https://github.com/rqa2025/RQA2025.git
cd RQA2025

# 2. 创建Python环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/production.txt

# 3. 生成生产配置
python scripts/deploy/production_deployment_config.py

# 4. 验证配置
python scripts/deploy/production_deployment_validator.py
```

### 2. 数据库部署

```bash
# 1. 启动PostgreSQL
docker run -d \
  --name rqa2025-postgres \
  -e POSTGRES_DB=rqa2025_prod \
  -e POSTGRES_USER=rqa2025_user \
  -e POSTGRES_PASSWORD=${DB_PASSWORD} \
  -v rqa2025_postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15

# 2. 执行数据库迁移
cd scripts/database
python migrate_production.py

# 3. 创建索引和约束
python setup_indexes.py
```

### 3. Redis集群部署

```bash
# 1. 启动Redis集群
docker run -d \
  --name rqa2025-redis \
  -e REDIS_PASSWORD=${REDIS_PASSWORD} \
  -v rqa2025_redis_data:/data \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes

# 2. 配置Redis集群 (如果需要)
# redis-cli --cluster create 127.0.0.1:7001 127.0.0.1:7002 127.0.0.1:7003
```

### 4. 应用部署

#### Docker Compose部署
```bash
# 1. 构建镜像
docker-compose -f config/production/docker-compose.yml build

# 2. 启动服务
docker-compose -f config/production/docker-compose.yml up -d

# 3. 查看服务状态
docker-compose -f config/production/docker-compose.yml ps
```

#### Kubernetes部署
```bash
# 1. 应用Kubernetes配置
kubectl apply -f config/production/kubernetes.yml

# 2. 检查Pod状态
kubectl get pods -n rqa2025

# 3. 检查服务状态
kubectl get services -n rqa2025

# 4. 检查Ingress状态
kubectl get ingress -n rqa2025
```

### 5. Nginx配置

```bash
# 1. 复制配置文件
cp config/production/nginx.conf /etc/nginx/sites-available/rqa2025

# 2. 启用站点
ln -s /etc/nginx/sites-available/rqa2025 /etc/nginx/sites-enabled/

# 3. 测试配置
nginx -t

# 4. 重载配置
nginx -s reload
```

### 6. 监控系统部署

```bash
# 1. 启动Prometheus
docker run -d \
  --name rqa2025-prometheus \
  -p 9090:9090 \
  -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# 2. 启动Grafana
docker run -d \
  --name rqa2025-grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD} \
  grafana/grafana

# 3. 配置监控仪表板
# 访问 http://localhost:3000 导入仪表板配置
```

---

## ⚙️ 配置管理

### 1. 配置文件结构

```
config/production/
├── database.json          # 数据库配置
├── redis.json            # Redis配置
├── api.json              # API服务配置
├── monitoring.json       # 监控配置
├── logging.json          # 日志配置
├── security.json         # 安全配置
├── nginx.conf            # Nginx配置
├── docker-compose.yml    # Docker Compose配置
└── kubernetes.yml        # Kubernetes配置
```

### 2. 环境变量管理

```bash
# 创建环境变量文件
cat > .env.production << EOF
# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rqa2025_prod
DB_USER=rqa2025_user
DB_PASSWORD=your_secure_password

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# API配置
API_PORT=8080
API_WORKERS=4

# 监控配置
GRAFANA_PASSWORD=your_grafana_password

# 安全配置
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
EOF

# 加载环境变量
source .env.production
```

### 3. 配置热更新

```python
# 配置热更新机制
from config.hot_reload import ConfigReloader

# 初始化配置重载器
config_reloader = ConfigReloader(config_dir="config/production")

# 监听配置变化
config_reloader.watch_config_changes()

# 在应用中响应配置变化
@config_reloader.on_config_change
def handle_config_update(new_config):
    logging.info("配置已更新，重新初始化服务...")
    # 重新初始化数据库连接、缓存等
    reload_services(new_config)
```

---

## 📊 监控告警

### 1. 监控指标

#### 系统级监控
```yaml
# CPU使用率
cpu_usage_percent > 80

# 内存使用率
memory_usage_percent > 85

# 磁盘使用率
disk_usage_percent > 90

# 网络流量
network_traffic_mbps > 1000
```

#### 应用级监控
```yaml
# API响应时间
api_response_time_seconds > 1

# 错误率
api_error_rate_percent > 5

# 数据库连接数
database_connections_active > 100

# 队列积压
queue_backlog_messages > 1000
```

#### 业务级监控
```yaml
# 交易成功率
trade_success_rate_percent < 95

# 数据延迟
market_data_delay_seconds > 5

# 策略执行时间
strategy_execution_time_seconds > 30
```

### 2. 告警配置

#### Prometheus Alertmanager配置
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@rqa2025.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'

receivers:
- name: 'email'
  email_configs:
  - to: 'ops@rqa2025.com'
    send_resolved: true
```

#### Grafana告警规则
```json
{
  "alertRule": {
    "name": "API响应时间过高",
    "condition": "C",
    "data": [
      {
        "refId": "A",
        "queryType": "metrics",
        "relativeTimeRange": {
          "from": 300,
          "to": 0
        },
        "datasourceUid": "prometheus",
        "model": {
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1",
          "intervalMs": 1000,
          "maxDataPoints": 43200
        }
      }
    ],
    "noDataState": "NoData",
    "execErrState": "Alerting",
    "for": "5m"
  }
}
```

### 3. 日志管理

#### 日志收集配置
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/rqa2025/*.log
  fields:
    service: rqa2025-api

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "rqa2025-%{+yyyy.MM.dd}"
```

#### 日志轮转配置
```bash
# /etc/logrotate.d/rqa2025
/var/log/rqa2025/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 rqa2025 rqa2025
    postrotate
        systemctl reload rqa2025-api
    endscript
}
```

---

## 🔄 备份恢复

### 1. 数据库备份

#### 自动备份脚本
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backup/database"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/rqa2025_backup_${TIMESTAMP}.sql"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
pg_dump -h localhost -U rqa2025_user -d rqa2025_prod > $BACKUP_FILE

# 压缩备份文件
gzip $BACKUP_FILE

# 清理7天前的备份
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "数据库备份完成: ${BACKUP_FILE}.gz"
```

#### 备份策略
```yaml
database_backup:
  schedule: "0 2 * * *"  # 每天凌晨2点
  retention_days: 30
  compression: true
  encryption: true
  verification: true
```

### 2. 配置备份

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backup/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 备份配置文件
tar -czf ${BACKUP_DIR}/config_backup_${TIMESTAMP}.tar.gz \
    -C /app config/production/

# 备份环境变量 (注意：敏感信息已加密)
cp .env.production ${BACKUP_DIR}/env_backup_${TIMESTAMP}

# 清理30天前的备份
find $BACKUP_DIR -mtime +30 -delete
```

### 3. 应用镜像备份

```bash
# 备份Docker镜像
docker save rqa2025/rqa2025-api:latest > /backup/images/rqa2025_api_$(date +%Y%m%d).tar

# 备份到镜像仓库
docker tag rqa2025/rqa2025-api:latest rqa2025/rqa2025-api:backup-$(date +%Y%m%d)
docker push rqa2025/rqa2025-api:backup-$(date +%Y%m%d)
```

### 4. 恢复流程

#### 数据库恢复
```bash
# 1. 停止应用服务
docker-compose down

# 2. 创建新数据库 (如果需要)
createdb -U rqa2025_user rqa2025_prod_new

# 3. 恢复数据
gunzip -c /backup/database/rqa2025_backup_20231201.sql.gz | \
    psql -U rqa2025_user -d rqa2025_prod_new

# 4. 重命名数据库
psql -U postgres -c "ALTER DATABASE rqa2025_prod_new RENAME TO rqa2025_prod;"

# 5. 重启服务
docker-compose up -d
```

#### 应用恢复
```bash
# 1. 从备份镜像恢复
docker load < /backup/images/rqa2025_api_20231201.tar

# 2. 重新部署
docker-compose up -d --no-deps rqa2025-api

# 3. 验证服务健康
curl -f http://localhost:8080/health
```

---

## 🚨 故障处理

### 1. 常见故障类型

#### 1.1 应用服务故障
```bash
# 检查服务状态
docker-compose ps

# 查看服务日志
docker-compose logs rqa2025-api

# 重启服务
docker-compose restart rqa2025-api

# 如果重启无效，重新部署
docker-compose up -d --no-deps --build rqa2025-api
```

#### 1.2 数据库连接故障
```bash
# 检查数据库状态
docker exec rqa2025-postgres pg_isready -U rqa2025_user -d rqa2025_prod

# 检查连接数
docker exec rqa2025-postgres psql -U rqa2025_user -d rqa2025_prod \
    -c "SELECT count(*) FROM pg_stat_activity;"

# 重启数据库
docker-compose restart postgres
```

#### 1.3 Redis连接故障
```bash
# 检查Redis状态
docker exec rqa2025-redis redis-cli ping

# 检查内存使用
docker exec rqa2025-redis redis-cli info memory

# 重启Redis
docker-compose restart redis
```

#### 1.4 网络故障
```bash
# 检查网络连接
curl -f http://localhost:8080/health

# 检查Nginx状态
nginx -t
systemctl status nginx

# 检查防火墙规则
ufw status
iptables -L
```

### 2. 故障排查流程

#### 2.1 快速诊断脚本
```bash
#!/bin/bash
# diagnose_system.sh

echo "=== RQA2025系统诊断 ==="

# 1. 检查服务状态
echo "1. 服务状态:"
docker-compose ps

# 2. 检查资源使用
echo "2. 系统资源:"
echo "CPU使用率:"
top -bn1 | grep "Cpu(s)"
echo "内存使用:"
free -h
echo "磁盘使用:"
df -h

# 3. 检查网络连接
echo "3. 网络连接:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health

# 4. 检查数据库连接
echo "4. 数据库连接:"
docker exec rqa2025-postgres pg_isready -U rqa2025_user -d rqa2025_prod

# 5. 检查Redis连接
echo "5. Redis连接:"
docker exec rqa2025-redis redis-cli ping

# 6. 检查日志错误
echo "6. 最近的错误日志:"
docker-compose logs --tail=20 | grep -i error

echo "=== 诊断完成 ==="
```

#### 2.2 性能问题排查
```bash
# 1. 检查慢查询
docker exec rqa2025-postgres psql -U rqa2025_user -d rqa2025_prod \
    -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# 2. 检查缓存命中率
docker exec rqa2025-redis redis-cli info stats | grep keyspace

# 3. 检查API性能
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/v1/market-data

# 4. 分析内存泄漏
docker stats rqa2025-api
```

### 3. 应急响应计划

#### 3.1 故障分级
- **P0**: 系统完全不可用，影响所有用户
- **P1**: 核心功能不可用，影响大部分用户
- **P2**: 非核心功能不可用，影响部分用户
- **P3**: 性能问题，不影响功能使用

#### 3.2 响应时间目标
- **P0**: 15分钟内响应，2小时内恢复
- **P1**: 30分钟内响应，4小时内恢复
- **P2**: 1小时内响应，8小时内恢复
- **P3**: 4小时内响应，24小时内恢复

#### 3.3 升级流程
```yaml
emergency_response:
  detection:
    - 监控告警触发
    - 用户报告
    - 自动化检测

  assessment:
    - 确定影响范围
    - 评估业务影响
    - 确定故障等级

  communication:
    - 通知相关团队
    - 更新状态页面
    - 通知用户 (如需要)

  resolution:
    - 激活应急团队
    - 执行恢复流程
    - 验证系统恢复

  post_mortem:
    - 分析根本原因
    - 制定改进措施
    - 更新文档和流程
```

---

## ⚡ 性能优化

### 1. 应用层优化

#### 1.1 API性能优化
```python
# 使用连接池
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30
)

# 缓存优化
from cachetools import TTLCache

cache = TTLCache(maxsize=1000, ttl=300)

@cache.cache
def get_market_data(symbol: str):
    return fetch_from_database(symbol)
```

#### 1.2 异步处理优化
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 创建线程池
executor = ThreadPoolExecutor(max_workers=10)

async def process_trading_signals(signals):
    """异步处理交易信号"""
    loop = asyncio.get_event_loop()

    # 并行处理信号
    tasks = [
        loop.run_in_executor(executor, validate_signal, signal)
        for signal in signals
    ]

    results = await asyncio.gather(*tasks)
    return results
```

### 2. 数据库优化

#### 2.1 查询优化
```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_trades_symbol_time
ON trades (symbol, created_at DESC);

-- 优化查询
EXPLAIN ANALYZE
SELECT * FROM trades
WHERE symbol = 'AAPL'
  AND created_at >= '2024-01-01'
ORDER BY created_at DESC
LIMIT 100;

-- 分区表
CREATE TABLE trades_y2024 PARTITION OF trades
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

#### 2.2 连接池配置
```yaml
database_pool:
  min_size: 5
  max_size: 20
  max_idle_time: 300
  max_lifetime: 3600
  connection_timeout: 30
```

### 3. 缓存优化

#### 3.1 多级缓存策略
```python
from cachetools import LRUCache, TTLCache

# L1缓存: 应用内缓存
l1_cache = LRUCache(maxsize=10000)

# L2缓存: Redis缓存
import redis
redis_client = redis.Redis(host='localhost', port=6379)

def get_data_with_cache(key):
    """多级缓存获取数据"""
    # 检查L1缓存
    if key in l1_cache:
        return l1_cache[key]

    # 检查L2缓存
    cached_data = redis_client.get(key)
    if cached_data:
        data = json.loads(cached_data)
        l1_cache[key] = data  # 写入L1缓存
        return data

    # 从数据库获取
    data = fetch_from_database(key)

    # 写入缓存
    redis_client.setex(key, 300, json.dumps(data))
    l1_cache[key] = data

    return data
```

### 4. 系统级优化

#### 4.1 Linux内核优化
```bash
# /etc/sysctl.conf
# 网络优化
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535

# 内存优化
vm.swappiness = 10
vm.dirty_ratio = 60
vm.dirty_background_ratio = 2

# 文件系统优化
vm.vfs_cache_pressure = 50

# 应用生效
sysctl -p
```

#### 4.2 Docker优化
```yaml
# docker-compose.yml 优化配置
services:
  rqa2025-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

    # 日志配置
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

---

## 🔒 安全运维

### 1. 访问控制

#### 1.1 用户认证
```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """生成密码哈希"""
    return pwd_context.hash(password)

# JWT令牌
def create_access_token(data: dict, expires_delta=None):
    """创建访问令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

#### 1.2 角色权限
```python
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"

# 权限检查装饰器
def require_role(required_role: UserRole):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取当前用户
            current_user = get_current_user()

            # 检查权限
            if current_user.role not in [required_role, UserRole.ADMIN]:
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 使用示例
@app.get("/admin/dashboard")
@require_role(UserRole.ADMIN)
async def admin_dashboard():
    return {"message": "Admin dashboard"}
```

### 2. 数据加密

#### 2.1 传输加密
```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# HTTPS重定向
app.add_middleware(HTTPSRedirectMiddleware)

# 信任主机
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["trading.rqa2025.com", "*.rqa2025.com"]
)

# SSL配置
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(
    certfile="/etc/ssl/certs/rqa2025.crt",
    keyfile="/etc/ssl/private/rqa2025.key"
)
```

#### 2.2 数据存储加密
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class DataEncryption:
    """数据加密工具"""

    def __init__(self, password: str):
        # 生成密钥
        salt = b'rqa2025_salt'  # 生产环境应该随机生成
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet = Fernet(key)

    def encrypt(self, data: str) -> str:
        """加密数据"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# 使用示例
encryptor = DataEncryption(password=os.getenv("ENCRYPTION_PASSWORD"))

# 加密敏感数据
encrypted_api_key = encryptor.encrypt("sk-1234567890abcdef")
print(f"Encrypted: {encrypted_api_key}")

# 解密数据
decrypted_api_key = encryptor.decrypt(encrypted_api_key)
print(f"Decrypted: {decrypted_api_key}")
```

### 3. 安全监控

#### 3.1 入侵检测
```yaml
# fail2ban配置
[nginx-ddos]
enabled = true
port = http,https
filter = nginx-ddos
logpath = /var/log/nginx/access.log
maxretry = 10
findtime = 600
bantime = 3600

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
findtime = 600
bantime = 3600
```

#### 3.2 日志审计
```python
import logging
from logging.handlers import RotatingFileHandler

# 安全日志记录器
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

# 创建轮转处理器
handler = RotatingFileHandler(
    '/var/log/rqa2025/security.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# 设置格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - IP:%(ip)s - User:%(user)s'
)
handler.setFormatter(formatter)
security_logger.addHandler(handler)

# 安全事件记录
def log_security_event(event_type: str, user: str, ip: str, details: dict = None):
    """记录安全事件"""
    extra = {
        'ip': ip,
        'user': user
    }

    if event_type == 'login_success':
        security_logger.info(f"用户 {user} 登录成功", extra=extra)
    elif event_type == 'login_failure':
        security_logger.warning(f"用户 {user} 登录失败", extra=extra)
    elif event_type == 'unauthorized_access':
        security_logger.error(f"用户 {user} 尝试未授权访问", extra=extra, extra={'details': details})
```

---

## 🔄 升级维护

### 1. 版本管理

#### 1.1 语义化版本
```
版本格式: MAJOR.MINOR.PATCH

- MAJOR: 不兼容的API变更
- MINOR: 向后兼容的功能新增
- PATCH: 向后兼容的bug修复

示例:
- 1.0.0: 初始版本
- 1.1.0: 新增功能
- 1.1.1: bug修复
- 2.0.0: 重大重构
```

#### 1.2 发布流程
```yaml
release_process:
  planning:
    - 功能需求分析
    - 技术方案设计
    - 资源评估

  development:
    - 创建功能分支
    - 实现功能代码
    - 编写单元测试
    - 代码审查

  testing:
    - 单元测试
    - 集成测试
    - 性能测试
    - 安全测试
    - 用户验收测试

  staging:
    - 部署到预发布环境
    - 执行冒烟测试
    - 业务验收测试

  production:
    - 灰度发布
    - 监控系统指标
    - 逐步增加流量
    - 全量发布

  post_release:
    - 验证生产环境
    - 监控告警确认
    - 文档更新
```

### 2. 滚动升级

#### 2.1 零停机部署
```yaml
# Kubernetes滚动更新
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: api
        image: rqa2025/rqa2025-api:v2.1.0
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### 2.2 数据库迁移
```python
# Alembic迁移脚本
from alembic import op
import sqlalchemy as sa

def upgrade():
    """升级数据库schema"""
    # 创建新表
    op.create_table(
        'trading_signals_v2',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('signal_type', sa.String(20), nullable=False),
        sa.Column('strength', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 数据迁移
    op.execute("""
        INSERT INTO trading_signals_v2 (symbol, signal_type, strength, created_at)
        SELECT symbol, 'BUY', confidence, created_at
        FROM trading_signals
        WHERE action = 'BUY' AND confidence > 0.7
    """)

    # 删除旧表
    op.drop_table('trading_signals')

def downgrade():
    """回滚数据库schema"""
    # 重新创建旧表结构
    op.create_table(
        'trading_signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(10), nullable=False),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 数据回滚
    op.execute("""
        INSERT INTO trading_signals (symbol, action, confidence, created_at)
        SELECT symbol, 'BUY', strength, created_at
        FROM trading_signals_v2
    """)

    # 删除新表
    op.drop_table('trading_signals_v2')
```

### 3. 回滚策略

#### 3.1 快速回滚
```bash
# Docker Compose回滚
docker-compose down
docker tag rqa2025/rqa2025-api:v2.0.0 rqa2025/rqa2025-api:latest
docker-compose up -d

# Kubernetes回滚
kubectl rollout undo deployment/rqa2025-api
kubectl rollout status deployment/rqa2025-api
```

#### 3.2 数据库回滚
```bash
# 恢复数据库备份
docker exec -i rqa2025-postgres psql -U rqa2025_user -d rqa2025_prod < backup.sql

# 或者使用时间点恢复
docker exec rqa2025-postgres pg_ctl stop -m fast
docker exec rqa2025-postgres pg_ctl start
```

### 4. 维护窗口

#### 4.1 定期维护计划
```yaml
maintenance_schedule:
  daily:
    - 数据库备份验证
    - 日志轮转检查
    - 磁盘空间清理
    - 监控指标检查

  weekly:
    - 安全补丁更新
    - 依赖包更新
    - 性能优化检查
    - 配置一致性验证

  monthly:
    - 完整系统备份
    - 容量规划评估
    - 安全审计执行
    - 文档更新确认

  quarterly:
    - 主要版本升级
    - 架构优化实施
    - 灾难恢复演练
    - 性能基准重新确立
```

#### 4.2 维护通知流程
```yaml
maintenance_notification:
  advance_notice:
    - 提前1周发布维护计划
    - 通知所有利益相关方
    - 更新状态页面

  during_maintenance:
    - 实时状态更新
    - 预计完成时间通知
    - 问题解决进展报告

  post_maintenance:
    - 维护完成确认
    - 功能验证结果
    - 后续监控计划
    - 问题总结和改进措施
```

---

## 📞 联系支持

### 技术支持
- **邮箱**: support@rqa2025.com
- **电话**: 400-123-4567
- **在线文档**: https://docs.rqa2025.com
- **状态页面**: https://status.rqa2025.com

### 紧急联系人
- **技术负责人**: tech-lead@rqa2025.com
- **运维负责人**: ops-lead@rqa2025.com
- **安全负责人**: security@rqa2025.com
- **业务负责人**: business-lead@rqa2025.com

### 响应时间
- **工作时间**: 9:00-18:00 (北京时间)
- **紧急情况**: 24/7 全天候响应
- **P0故障**: 15分钟内响应
- **一般问题**: 4小时内响应

---

**RQA2025量化交易系统运维手册**

*最后更新: 2025年12月2日*

*版本: v1.0.0*



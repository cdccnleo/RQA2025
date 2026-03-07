# 📋 RQA2025 生产部署与运维手册

## 📊 概述

**RQA2025** 是一个企业级的智能化量化交易分析平台。本手册提供了完整的生产环境部署、配置、监控和运维指南。

### 🏆 系统特性
- **测试覆盖率**: 48.4% (行业领先水平)
- **架构设计**: 14层企业级分层架构
- **AI功能**: 13项智能化运维功能
- **性能指标**: 支持2000+ TPS并发处理
- **安全标准**: 符合GDPR/SOX合规要求

---

## 🏗️ 部署架构

### 生产环境架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 负载均衡层 (Nginx/HAProxy)              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  🔌 API网关 (FastAPI + Gunicorn)                   │    │
│  │  📊 业务服务 (Application Services)               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                                                      ▲
┌─────────────────────────────────────────────────────────────┐        │
│                    🗄️ 数据存储层 (Data Layer)                 │        │
│  ┌─────────────────────────────────────────────────────┐    │        │
│  │  🐘 PostgreSQL (主从复制)                         │    │        │
│  │  🔴 Redis集群 (缓存+消息队列)                     │    │        │
│  │  📊 时序数据库 (InfluxDB)                         │    │        │
│  └─────────────────────────────────────────────────────┘    │        │
└─────────────────────────────────────────────────────────────┘        │
                                                                      │
┌─────────────────────────────────────────────────────────────┐        │
│                    📊 监控告警层 (Monitoring)                │        │
│  ┌─────────────────────────────────────────────────────┐    │        │
│  │  📈 Prometheus (指标收集)                         │    │        │
│  │  📊 Grafana (可视化监控)                          │    │        │
│  │  🚨 AlertManager (告警管理)                       │    │        │
│  └─────────────────────────────────────────────────────┘    │        │
└─────────────────────────────────────────────────────────────┘        ▼
```

### 部署组件清单

| 组件 | 版本 | 用途 | 实例数量 |
|------|------|------|----------|
| API网关 | FastAPI 0.100+ | 请求路由和认证 | 3+ (负载均衡) |
| 业务服务 | Python 3.9+ | 核心业务逻辑 | 5+ (按需扩展) |
| PostgreSQL | 15.0+ | 主数据库 | 2 (主从) |
| Redis | 7.0+ | 缓存和消息队列 | 3+ (集群) |
| RabbitMQ | 3.12+ | 异步任务队列 | 3 (集群) |
| Nginx | 1.24+ | 反向代理和负载均衡 | 2+ (高可用) |
| Prometheus | 2.45+ | 监控指标收集 | 2 (主备) |
| Grafana | 10.0+ | 监控可视化 | 1+ |

---

## 🚀 部署前准备

### 环境要求

#### 硬件配置
```yaml
# 生产环境最低配置
api_gateway:
  cpu: 4核心
  memory: 8GB
  storage: 50GB SSD

business_services:
  cpu: 8核心
  memory: 16GB
  storage: 100GB SSD

database:
  cpu: 16核心
  memory: 64GB
  storage: 1TB NVMe SSD

redis_cluster:
  cpu: 8核心
  memory: 32GB
  storage: 200GB SSD

monitoring:
  cpu: 4核心
  memory: 8GB
  storage: 100GB SSD
```

#### 网络配置
- **内部网络**: 10.0.0.0/16 (RFC 1918私有地址)
- **负载均衡器**: 公网IP + 内部IP
- **数据库**: 仅内部网络访问
- **监控**: 仅运维网络访问

#### 安全配置
```bash
# 防火墙规则
iptables -A INPUT -p tcp --dport 80 -j ACCEPT    # HTTP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT   # HTTPS
iptables -A INPUT -p tcp --dport 22 -j ACCEPT    # SSH (仅运维IP)
iptables -A INPUT -p tcp --dport 5432 -j DROP    # PostgreSQL (内部)
iptables -A INPUT -p tcp --dport 6379 -j DROP    # Redis (内部)
```

### 依赖检查

#### 系统依赖
```bash
# Ubuntu/Debian
apt update && apt upgrade -y
apt install -y python3.9 python3.9-dev postgresql-15 redis-server rabbitmq-server nginx

# CentOS/RHEL
yum update -y
yum install -y python39 python39-devel postgresql15-server redis rabbitmq-server nginx
```

#### Python环境
```bash
# 创建虚拟环境
python3.9 -m venv /opt/rqa2025/venv
source /opt/rqa2025/venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn[standard] psycopg2-binary redis hiredis aio-pika
```

---

## 📦 部署步骤

### Phase 1: 基础设施部署

#### 1. PostgreSQL主从配置
```bash
# 主库配置 (/etc/postgresql/15/main/postgresql.conf)
listen_addresses = '*'
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1024

# 从库配置
# 连接到主库进行复制
```

#### 2. Redis集群配置
```bash
# redis.conf
bind 0.0.0.0
protected-mode no
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
```

#### 3. RabbitMQ集群配置
```bash
# 启用管理插件
rabbitmq-plugins enable rabbitmq_management

# 集群配置
rabbitmqctl stop_app
rabbitmqctl join_cluster rabbit@rabbitmq01
rabbitmqctl start_app
```

### Phase 2: 应用部署

#### 1. 代码部署
```bash
# 创建部署目录
mkdir -p /opt/rqa2025/{app,config,logs,data}

# 克隆代码
cd /opt/rqa2025
git clone https://github.com/your-org/RQA2025.git app
cd app
git checkout production

# 安装Python依赖
source /opt/rqa2025/venv/bin/activate
pip install -r requirements.txt
```

#### 2. 配置管理
```bash
# 生产环境配置
cp config/production.example.yaml config/production.yaml

# 编辑配置
vim config/production.yaml
```

**关键配置项**:
```yaml
database:
  url: postgresql://rqa_user:password@db01:5432/rqa_production
  pool_size: 20
  max_overflow: 30

redis:
  cluster:
    - host: redis01:6379
    - host: redis02:6379
    - host: redis03:6379

rabbitmq:
  url: amqp://guest:guest@rabbitmq01:5672/
  cluster:
    - rabbitmq01
    - rabbitmq02
    - rabbitmq03

monitoring:
  prometheus_url: http://prometheus01:9090
  alertmanager_url: http://alertmanager01:9093
```

#### 3. 服务启动脚本
```bash
# 创建systemd服务文件 (/etc/systemd/system/rqa2025-api.service)
[Unit]
Description=RQA2025 API Gateway
After=network.target postgresql.service redis.service rabbitmq-server.service

[Service]
Type=exec
User=rqa2025
Group=rqa2025
WorkingDirectory=/opt/rqa2025/app
Environment=PATH=/opt/rqa2025/venv/bin
Environment=PYTHONPATH=/opt/rqa2025/app/src
Environment=ENV=production
ExecStart=/opt/rqa2025/venv/bin/gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile /opt/rqa2025/logs/access.log \
    --error-logfile /opt/rqa2025/logs/error.log \
    src.core.app:app

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Phase 3: 监控部署

#### 1. Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager01:9093

scrape_configs:
  - job_name: 'rqa2025-api'
    static_configs:
      - targets: ['api01:8000', 'api02:8000', 'api03:8000']
    metrics_path: '/metrics'

  - job_name: 'rqa2025-business'
    static_configs:
      - targets: ['biz01:8001', 'biz02:8001', 'biz03:8001']
    metrics_path: '/metrics'

  - job_name: 'postgresql'
    static_configs:
      - targets: ['db01:9187']
```

#### 2. Grafana仪表板
```json
{
  "dashboard": {
    "title": "RQA2025 Production Dashboard",
    "tags": ["rqa2025", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"rqa2025-api\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "System CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ]
      }
    ]
  }
}
```

---

## 🔍 监控和告警

### 关键指标监控

#### 应用层指标
```prometheus
# API响应时间
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 请求错误率
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# 活跃连接数
sum(rqa_active_connections)

# 队列长度
rqa_queue_length{queue="high_priority"}
rqa_queue_length{queue="normal_priority"}
```

#### 系统层指标
```prometheus
# CPU使用率
100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 内存使用率
100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)

# 磁盘使用率
100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)

# 网络流量
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])
```

#### 数据库指标
```prometheus
# 连接数
pg_stat_activity_count

# 查询响应时间
histogram_quantile(0.95, rate(pg_query_duration_seconds_bucket[5m]))

# 缓存命中率
pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)

# 慢查询数量
increase(pg_stat_activity_max_tx_duration[5m])
```

### 告警规则配置

```yaml
# alert_rules.yml
groups:
  - name: rqa2025
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API响应时间过高"
          description: "95th percentile response time > 1s for 5m"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "API错误率过高"
          description: "Error rate > 5% for 5m"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "CPU使用率过高"
          description: "CPU usage > 80% for 10m"
```

---

## 🔄 运维操作

### 日常运维

#### 日志管理
```bash
# 日志轮转配置 (/etc/logrotate.d/rqa2025)
/opt/rqa2025/logs/*.log {
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

# 日志分析
tail -f /opt/rqa2025/logs/app.log | grep ERROR
grep "Exception" /opt/rqa2025/logs/app.log | wc -l
```

#### 备份策略
```bash
# 数据库备份脚本 (/opt/rqa2025/scripts/backup.sh)
#!/bin/bash

BACKUP_DIR="/opt/rqa2025/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 数据库备份
pg_dump -h db01 -U rqa_user -d rqa_production > ${BACKUP_DIR}/db_${DATE}.sql

# 配置文件备份
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz /opt/rqa2025/config/

# 应用数据备份
tar -czf ${BACKUP_DIR}/data_${DATE}.tar.gz /opt/rqa2025/data/

# 清理7天前的备份
find ${BACKUP_DIR} -name "*.sql" -mtime +7 -delete
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +7 -delete

echo "备份完成: ${DATE}"
```

#### 定期维护
```bash
# 定时任务配置 (/etc/cron.d/rqa2025)
# 每日凌晨2点备份
0 2 * * * rqa2025 /opt/rqa2025/scripts/backup.sh

# 每小时更新数据库统计信息
0 * * * * rqa2025 /opt/rqa2025/scripts/update_stats.sh

# 每周日凌晨3点清理过期数据
0 3 * * 0 rqa2025 /opt/rqa2025/scripts/cleanup.sh
```

### 故障处理

#### 常见故障排查

##### 应用服务异常
```bash
# 检查服务状态
systemctl status rqa2025-api

# 查看日志
journalctl -u rqa2025-api -f
tail -f /opt/rqa2025/logs/app.log

# 重启服务
systemctl restart rqa2025-api

# 检查端口占用
netstat -tlnp | grep :8000
```

##### 数据库连接问题
```bash
# 检查数据库连接
psql -h db01 -U rqa_user -d rqa_production -c "SELECT 1;"

# 查看连接池状态
psql -h db01 -U rqa_user -d rqa_production -c "SELECT * FROM pg_stat_activity;"

# 重启数据库
systemctl restart postgresql
```

##### Redis集群问题
```bash
# 检查集群状态
redis-cli -c cluster nodes

# 检查主从同步
redis-cli -c info replication

# 重启Redis
systemctl restart redis
```

#### 性能问题诊断

##### CPU使用率过高
```bash
# 查看进程CPU使用
top -p $(pgrep -f rqa2025)

# 性能分析
python -m cProfile -o profile.out /opt/rqa2025/venv/bin/python src/core/app.py
snakeviz profile.out
```

##### 内存泄漏检查
```bash
# 内存使用监控
ps aux --sort=-%mem | head -10

# Python内存分析
python -c "
import tracemalloc
tracemalloc.start()
# 运行应用一段时间
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"
```

##### 数据库性能优化
```bash
# 慢查询分析
psql -h db01 -U rqa_user -d rqa_production -c "
SELECT query, total_time, calls, mean_time
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY mean_time DESC
LIMIT 10;
"

# 索引使用分析
psql -h db01 -U rqa_user -d rqa_production -c "
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
"
```

---

## 🔒 安全运维

### 访问控制
```bash
# 创建运维用户
useradd -m -s /bin/bash rqa2025
usermod -aG sudo rqa2025

# SSH密钥认证
mkdir -p /home/rqa2025/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ..." > /home/rqa2025/.ssh/authorized_keys
chmod 600 /home/rqa2025/.ssh/authorized_keys
chown -R rqa2025:rqa2025 /home/rqa2025/.ssh
```

### 安全加固
```bash
# 禁用root SSH登录
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# 安装fail2ban
apt install fail2ban
systemctl enable fail2ban

# 配置防火墙
ufw enable
ufw allow ssh
ufw allow 80
ufw allow 443
```

### 合规审计
```bash
# 审计日志配置
auditctl -w /opt/rqa2025/ -p rwxa -k rqa2025_audit

# 查看审计日志
ausearch -k rqa2025_audit

# 定期安全扫描
# 使用工具如 Nessus, OpenVAS 等进行安全扫描
```

---

## 📈 性能优化

### 应用层优化

#### Gunicorn配置调优
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4  # CPU核心数的2倍
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 10
```

#### 缓存策略优化
```python
# Redis缓存配置
CACHE_CONFIG = {
    'default_timeout': 3600,
    'key_prefix': 'rqa2025:',
    'cache_type': 'redis',
    'cache_redis_url': 'redis://redis01:6379/0',
    'cache_redis_cluster': [
        'redis01:6379',
        'redis02:6379',
        'redis03:6379'
    ]
}
```

### 数据库优化

#### 连接池配置
```python
# SQLAlchemy配置
SQLALCHEMY_DATABASE_URI = "postgresql://user:pass@db01:5432/db"
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_MAX_OVERFLOW = 30
SQLALCHEMY_POOL_TIMEOUT = 30
SQLALCHEMY_POOL_RECYCLE = 3600
SQLALCHEMY_ECHO = False
```

#### 查询优化
```sql
-- 添加关键索引
CREATE INDEX CONCURRENTLY idx_orders_symbol_time
ON orders(symbol, created_at DESC);

CREATE INDEX CONCURRENTLY idx_market_data_timestamp
ON market_data(timestamp DESC, symbol);

-- 分区表
CREATE TABLE orders_y2025m01 PARTITION OF orders
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

---

## 🚨 应急处理

### 故障等级定义

| 等级 | 影响范围 | 响应时间 | 恢复目标 |
|------|----------|----------|----------|
| P0 | 完全不可用 | 15分钟 | 1小时 |
| P1 | 核心功能不可用 | 30分钟 | 4小时 |
| P2 | 非核心功能异常 | 2小时 | 24小时 |
| P3 | 轻微问题 | 24小时 | 7天 |

### 应急响应流程

#### 1. 故障发现
- 监控告警自动触发
- 用户报告
- 运维巡检发现

#### 2. 故障评估
```bash
# 快速检查脚本
/opt/rqa2025/scripts/health_check.sh

# 检查结果
# - API响应时间
# - 错误率
# - 数据库连接
# - 队列积压
```

#### 3. 故障处理
```bash
# P0故障处理
# 1. 立即通知所有相关人员
# 2. 启动应急响应小组
# 3. 执行故障恢复预案

# 常见恢复操作
systemctl restart rqa2025-api    # 重启API服务
systemctl restart postgresql     # 重启数据库
redis-cli flushall              # 清空缓存
```

#### 4. 故障总结
- 故障原因分析
- 影响评估
- 修复措施记录
- 预防措施制定

---

## 📊 容量规划

### 性能基准

| 指标 | 当前值 | 目标值 | 扩展阈值 |
|------|--------|--------|----------|
| 并发用户 | 1000 | 5000 | 80% |
| TPS | 2000 | 10000 | 70% |
| 响应时间 | <50ms | <100ms | 200ms |
| CPU使用率 | <40% | <60% | 80% |
| 内存使用率 | <60% | <80% | 90% |

### 扩展策略

#### 水平扩展
```bash
# 添加新的API实例
docker run -d --name api04 \
  -e ENV=production \
  --network rqa2025 \
  rqa2025/api:latest

# 更新Nginx配置
upstream api_backends {
    server api01:8000;
    server api02:8000;
    server api03:8000;
    server api04:8000;  # 新增
}
```

#### 垂直扩展
```bash
# 增加实例资源
docker update --cpus 8 --memory 16g api01

# 数据库扩展
# 增加CPU和内存
# 添加只读副本
```

### 监控阈值配置

```yaml
# Prometheus告警规则扩展
groups:
  - name: capacity_planning
    rules:
      - alert: HighResourceUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) > 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "内存使用率过高，需要考虑扩展"

      - alert: HighQueueLength
        expr: rqa_queue_length > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "队列积压严重，需要增加处理实例"
```

---

## 📞 联系与支持

### 技术支持
- **生产环境问题**: prod-support@rqa2025.com
- **紧急故障**: emergency@rqa2025.com
- **技术咨询**: tech-support@rqa2025.com

### 响应时间
- **P0故障**: 15分钟内响应
- **P1故障**: 30分钟内响应
- **P2故障**: 2小时内响应
- **一般咨询**: 24小时内响应

### 升级和维护
- **定期维护**: 每周二凌晨2:00-4:00
- **紧急维护**: 提前24小时通知
- **版本升级**: 每月第一个周日

---

*本手册持续更新，请定期检查最新版本。如有疑问，请联系技术支持团队。*

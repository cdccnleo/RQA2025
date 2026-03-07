# RQA2025 生产环境部署计划

## 📋 部署概述

本计划详细描述了RQA2025项目在生产环境的部署方案，包括Redis集群、负载均衡、监控告警等企业级特性。

## 🏗️ 架构设计

### 1. 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Redis Cluster │    │  Monitoring     │
│   (Nginx/HAProxy│    │   (6 nodes)     │    │  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RQA2025 API    │    │  Async Inference│    │  Alert Manager  │
│  (3 instances)  │    │  Engine Cluster │    │  (Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Database       │    │  Model Storage  │    │  Log Aggregation│
│  (PostgreSQL)   │    │  (MinIO/S3)     │    │  (ELK Stack)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. 服务组件
- **API Gateway**: Nginx/HAProxy负载均衡
- **应用服务**: RQA2025核心服务集群
- **缓存集群**: Redis 6.x集群模式
- **推理引擎**: 异步推理引擎集群
- **监控系统**: Prometheus + Grafana
- **日志系统**: ELK Stack (Elasticsearch + Logstash + Kibana)

## 🚀 部署步骤

### 1. 基础设施准备

#### 1.1 服务器配置
```bash
# 生产环境服务器配置
# API服务器 (3台)
CPU: 8核, 内存: 16GB, 磁盘: 100GB SSD
# Redis服务器 (6台)
CPU: 4核, 内存: 8GB, 磁盘: 50GB SSD
# 监控服务器 (1台)
CPU: 4核, 内存: 8GB, 磁盘: 100GB SSD
# 数据库服务器 (1台)
CPU: 8核, 内存: 16GB, 磁盘: 200GB SSD
```

#### 1.2 网络配置
```bash
# 防火墙规则
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 6379/tcp  # Redis
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9200/tcp  # Elasticsearch
sudo ufw enable
```

### 2. Redis集群部署

#### 2.1 安装Redis
```bash
# 在所有Redis节点上执行
sudo apt update
sudo apt install redis-server

# 配置Redis集群
sudo nano /etc/redis/redis.conf

# 集群配置
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
```

#### 2.2 创建集群
```bash
# 在第一个节点上执行
redis-cli --cluster create \
  192.168.1.10:6379 192.168.1.11:6379 192.168.1.12:6379 \
  192.168.1.13:6379 192.168.1.14:6379 192.168.1.15:6379 \
  --cluster-replicas 1

# 验证集群状态
redis-cli -h 192.168.1.10 -p 6379 cluster info
```

#### 2.3 集群配置更新
```yaml
# config/redis_cluster.yaml
redis_cluster:
  nodes:
    - host: 192.168.1.10
      port: 6379
      role: master
    - host: 192.168.1.11
      port: 6379
      role: master
    - host: 192.168.1.12
      port: 6379
      role: master
    - host: 192.168.1.13
      port: 6379
      role: slave
    - host: 192.168.1.14
      port: 6379
      role: slave
    - host: 192.168.1.15
      port: 6379
      role: slave
  connection_pool:
    max_connections: 50
    retry_on_timeout: true
    health_check_interval: 30
```

### 3. 负载均衡配置

#### 3.1 Nginx配置
```nginx
# /etc/nginx/sites-available/rqa2025
upstream rqa2025_backend {
    least_conn;
    server 192.168.1.20:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.21:8000 max_fails=3 fail_timeout=30s;
    server 192.168.1.22:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream inference_backend {
    least_conn;
    server 192.168.1.30:8001 max_fails=3 fail_timeout=30s;
    server 192.168.1.31:8001 max_fails=3 fail_timeout=30s;
    server 192.168.1.32:8001 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.rqa2025.com;
    
    # 健康检查
    location /health {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # API路由
    location /api/ {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超时设置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # 推理服务路由
    location /inference/ {
        proxy_pass http://inference_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 推理服务超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### 3.2 启用配置
```bash
sudo ln -s /etc/nginx/sites-available/rqa2025 /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. 应用服务部署

#### 4.1 Docker化部署
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "src.main", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.2 Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  rqa2025-api:
    build: .
    image: rqa2025/api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    environment:
      - REDIS_CLUSTER_HOSTS=192.168.1.10:6379,192.168.1.11:6379,192.168.1.12:6379
      - DATABASE_URL=postgresql://user:pass@192.168.1.40:5432/rqa2025
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    networks:
      - rqa2025-network
    restart: unless-stopped

  inference-engine:
    build: .
    image: rqa2025/inference:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      - MAX_WORKERS=4
      - BATCH_SIZE=32
      - ENABLE_CACHE=true
    ports:
      - "8001:8001"
    networks:
      - rqa2025-network
    restart: unless-stopped

networks:
  rqa2025-network:
    driver: bridge
```

### 5. 监控系统部署

#### 5.1 Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'rqa2025-api'
    static_configs:
      - targets: ['192.168.1.20:8000', '192.168.1.21:8000', '192.168.1.22:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'inference-engine'
    static_configs:
      - targets: ['192.168.1.30:8001', '192.168.1.31:8001', '192.168.1.32:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis-cluster'
    static_configs:
      - targets: ['192.168.1.10:6379', '192.168.1.11:6379', '192.168.1.12:6379']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['192.168.1.20:9100', '192.168.1.21:9100', '192.168.1.22:9100']
    scrape_interval: 30s
```

#### 5.2 告警规则
```yaml
# alert_rules.yml
groups:
- name: rqa2025_alerts
  rules:
  - alert: HighCPUUsage
    expr: system_cpu_usage > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}%"

  - alert: HighMemoryUsage
    expr: system_memory_usage > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}%"

  - alert: RedisClusterDown
    expr: redis_up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Redis cluster is down"
      description: "Redis cluster node is not responding"

  - alert: InferenceEngineError
    expr: inference_engine_errors > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Inference engine errors detected"
      description: "{{ $value }} errors in inference engine"

  - alert: APILatencyHigh
    expr: api_request_duration_seconds > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "API response time is {{ $value }}s"
```

#### 5.3 Grafana仪表板
```json
{
  "dashboard": {
    "title": "RQA2025 Production Dashboard",
    "panels": [
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "system_cpu_usage",
            "legendFormat": "CPU Usage"
          },
          {
            "expr": "system_memory_usage",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Redis Cluster Status",
        "type": "stat",
        "targets": [
          {
            "expr": "redis_up",
            "legendFormat": "Redis Nodes"
          }
        ]
      },
      {
        "title": "Inference Engine Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "inference_requests_total",
            "legendFormat": "Total Requests"
          },
          {
            "expr": "inference_duration_seconds",
            "legendFormat": "Average Duration"
          }
        ]
      }
    ]
  }
}
```

### 6. 日志系统部署

#### 6.1 ELK Stack配置
```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "rqa2025-api" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["192.168.1.50:9200"]
    index => "rqa2025-%{+YYYY.MM.dd}"
  }
}
```

#### 6.2 应用日志配置
```python
# src/infrastructure/logging/production_config.py
import logging
import logging.handlers
import os

def setup_production_logging():
    """生产环境日志配置"""
    
    # 创建日志目录
    log_dir = "/var/log/rqa2025"
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            # 控制台输出
            logging.StreamHandler(),
            # 文件输出
            logging.handlers.RotatingFileHandler(
                f"{log_dir}/rqa2025.log",
                maxBytes=100*1024*1024,  # 100MB
                backupCount=10
            ),
            # 错误日志单独文件
            logging.handlers.RotatingFileHandler(
                f"{log_dir}/rqa2025_error.log",
                maxBytes=50*1024*1024,   # 50MB
                backupCount=5,
                level=logging.ERROR
            )
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
```

### 7. 部署脚本

#### 7.1 自动化部署脚本
```bash
#!/bin/bash
# deploy.sh

set -e

echo "开始部署RQA2025生产环境..."

# 1. 检查环境
echo "检查环境依赖..."
python -c "import redis, psycopg2, numpy, pandas" || {
    echo "缺少必要的Python依赖"
    exit 1
}

# 2. 备份现有配置
echo "备份现有配置..."
sudo cp -r /etc/rqa2025 /etc/rqa2025.backup.$(date +%Y%m%d_%H%M%S)

# 3. 部署Redis集群
echo "部署Redis集群..."
./scripts/deploy_redis_cluster.sh

# 4. 部署应用服务
echo "部署应用服务..."
docker-compose -f docker-compose.yml up -d

# 5. 配置负载均衡
echo "配置负载均衡..."
sudo cp config/nginx/rqa2025 /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/rqa2025 /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 6. 部署监控系统
echo "部署监控系统..."
docker-compose -f docker-compose.monitoring.yml up -d

# 7. 健康检查
echo "执行健康检查..."
sleep 30
curl -f http://localhost/health || {
    echo "健康检查失败"
    exit 1
}

echo "部署完成！"
```

#### 7.2 回滚脚本
```bash
#!/bin/bash
# rollback.sh

set -e

echo "开始回滚RQA2025..."

# 1. 停止新服务
echo "停止新服务..."
docker-compose down

# 2. 恢复配置
echo "恢复配置..."
sudo cp -r /etc/rqa2025.backup.* /etc/rqa2025

# 3. 启动旧服务
echo "启动旧服务..."
docker-compose -f docker-compose.previous.yml up -d

# 4. 健康检查
echo "执行健康检查..."
sleep 30
curl -f http://localhost/health || {
    echo "回滚后健康检查失败"
    exit 1
}

echo "回滚完成！"
```

## 📊 性能监控

### 1. 关键指标
- **API响应时间**: < 200ms (95th percentile)
- **推理延迟**: < 500ms (95th percentile)
- **缓存命中率**: > 80%
- **系统资源使用率**: CPU < 80%, 内存 < 85%
- **Redis集群可用性**: > 99.9%

### 2. 告警阈值
- CPU使用率 > 80% (5分钟)
- 内存使用率 > 85% (5分钟)
- API响应时间 > 2秒 (5分钟)
- Redis节点不可用 (1分钟)
- 推理引擎错误率 > 5% (2分钟)

## 🔧 运维操作

### 1. 日常维护
```bash
# 日志清理
find /var/log/rqa2025 -name "*.log" -mtime +30 -delete

# 监控数据清理
curl -X POST http://localhost:9090/api/v1/admin/tsdb/clean_tombstones

# 缓存清理
redis-cli -h 192.168.1.10 FLUSHDB
```

### 2. 故障处理
```bash
# 服务重启
docker-compose restart rqa2025-api

# 节点替换
./scripts/replace_redis_node.sh old_node new_node

# 配置热更新
curl -X POST http://localhost:8000/api/v1/config/reload
```

## 📋 部署检查清单

- [ ] 服务器硬件配置检查
- [ ] 网络连通性测试
- [ ] Redis集群部署和验证
- [ ] 负载均衡器配置
- [ ] 应用服务部署
- [ ] 监控系统部署
- [ ] 日志系统配置
- [ ] 告警规则配置
- [ ] 性能测试
- [ ] 故障恢复测试
- [ ] 文档更新

---

**部署计划版本**: v1.0
**创建时间**: 2025年1月
**负责人**: 运维团队 
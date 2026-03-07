# RQA2025 量化交易系统运维手册

## 📋 手册概述

**版本**：V1.0
**更新日期**：2024年12月
**适用对象**：系统管理员、运维工程师、DevOps工程师
**文档目的**：指导系统的日常运维、监控和维护工作

---

## 🏗️ 系统架构概览

### 1. 技术栈架构

```
RQA2025技术栈架构
==============================================
应用层 (Application Layer)
├── Web界面 (Flask + React)
├── API网关 (Nginx + Lua)
├── 微服务框架 (FastAPI + gRPC)
└── 消息队列 (Kafka + Redis)

服务层 (Service Layer)
├── 交易引擎 (Cython + NumPy)
├── 策略管理 (Python + MLflow)
├── 风险控制 (Python + Pandas)
├── 数据处理 (Python + Dask)
└── 监控告警 (Prometheus + Grafana)

数据层 (Data Layer)
├── 时序数据库 (InfluxDB)
├── 关系数据库 (PostgreSQL)
├── 缓存系统 (Redis Cluster)
├── 消息队列 (Kafka Cluster)
└── 对象存储 (MinIO)

基础设施层 (Infrastructure Layer)
├── 容器化 (Docker + Kubernetes)
├── 服务网格 (Istio)
├── 负载均衡 (Nginx Ingress)
├── 网络安全 (Calico + Security Groups)
└── 存储系统 (Ceph + EBS)
```

### 2. 部署环境

#### 生产环境架构
- **多可用区部署**：跨3个可用区的分布式部署
- **容器化管理**：Kubernetes集群管理
- **服务发现**：自动服务注册和发现
- **配置管理**：集中化配置管理

#### 环境规格
- **计算资源**：100+ vCPU, 256GB+ 内存
- **存储资源**：10TB+ SSD存储
- **网络带宽**：1Gbps+ 专线网络
- **并发处理**：10,000+ TPS

---

## 📊 日常运维任务

### 1. 日常检查清单

#### 每日检查 (Daily Checklist)

```bash
#!/bin/bash
# RQA2025每日运维检查脚本

echo "=== RQA2025每日运维检查 ==="
echo "检查时间: $(date)"

# 1. 系统状态检查
echo "1. 系统资源使用情况:"
kubectl top nodes
kubectl top pods

# 2. 服务健康检查
echo "2. 服务健康状态:"
kubectl get pods --all-namespaces | grep -v Running

# 3. 数据库状态检查
echo "3. 数据库连接状态:"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1;" 2>/dev/null && echo "✓ 数据库连接正常" || echo "✗ 数据库连接异常"

# 4. 队列积压检查
echo "4. 消息队列状态:"
./check-queue-depth.sh

# 5. 业务指标检查
echo "5. 核心业务指标:"
./check-business-metrics.sh

echo "=== 每日检查完成 ==="
```

#### 每周检查 (Weekly Checklist)

```bash
#!/bin/bash
# RQA2025每周运维检查脚本

echo "=== RQA2025每周运维检查 ==="

# 1. 备份完整性检查
echo "1. 备份文件验证:"
./verify-backups.sh --last-week

# 2. 安全补丁检查
echo "2. 系统补丁状态:"
./check-security-patches.sh

# 3. 性能趋势分析
echo "3. 性能指标趋势:"
./analyze-performance-trend.sh --period "7d"

# 4. 日志分析
echo "4. 异常日志分析:"
./analyze-logs.sh --period "7d" --level "ERROR"

# 5. 容量规划检查
echo "5. 容量使用趋势:"
./check-capacity-planning.sh
```

### 2. 性能监控

#### 核心监控指标

| 指标类型 | 具体指标 | 阈值 | 告警级别 |
|---------|---------|------|---------|
| **系统资源** |  |  |  |
| - CPU使用率 | 平均使用率 | 80% | 警告 |
| - 内存使用率 | 平均使用率 | 85% | 警告 |
| - 磁盘使用率 | 使用率 | 85% | 警告 |
| - 网络带宽 | 使用率 | 80% | 警告 |
| **应用性能** |  |  |  |
| - API响应时间 | P95响应时间 | 1秒 | 警告 |
| - 错误率 | 5xx错误率 | 1% | 错误 |
| - TPS | 交易处理能力 | 8,000 | 警告 |
| **业务指标** |  |  |  |
| - 交易成功率 | 成功率 | 99.5% | 错误 |
| - 数据延迟 | 市场数据延迟 | 5秒 | 警告 |
| - 用户活跃度 | 在线用户数 | - | 监控 |

#### 监控工具链

```yaml
# Prometheus监控配置示例
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rqa2025-core'
    static_configs:
      - targets: ['core-service:8080']
    metrics_path: '/metrics'
    params:
      format: ['prometheus']

  - job_name: 'rqa2025-trading'
    static_configs:
      - targets: ['trading-engine:8081']
    metrics_path: '/metrics'

  - job_name: 'kubernetes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - source_labels: [__address__]
        regex: '(.+):10250'
        replacement: '${1}:10255'
        target_label: __address__
```

### 3. 日志管理

#### 日志收集架构

```
日志收集架构
==============================================
应用服务 → 日志文件 → Filebeat → Kafka → Elasticsearch
                    ↓
              结构化日志 → 统一格式 → 索引存储 → 可视化展示
```

#### 日志配置

```python
# 日志配置示例
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': '/var/log/rqa2025/app.log',
            'formatter': 'json',
            'level': 'INFO'
        },
        'elasticsearch': {
            'class': 'elasticsearch.ElasticsearchHandler',
            'hosts': ['elasticsearch:9200'],
            'index_name': 'rqa2025-logs',
            'formatter': 'json',
            'level': 'INFO'
        }
    },
    'root': {
        'handlers': ['console', 'file', 'elasticsearch'],
        'level': 'INFO'
    }
}
```

#### 日志分析命令

```bash
# 实时日志监控
kubectl logs -f deployment/rqa2025-core --tail=100

# 错误日志统计
kubectl logs --all-containers --since=1h | grep ERROR | wc -l

# 性能日志分析
kubectl logs deployment/rqa2025-trading --since=1h | grep "response_time" | sort -k3 -n

# 日志聚合查询
curl -X GET "elasticsearch:9200/rqa2025-logs-*/_search" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": [
          {"match": {"level": "ERROR"}},
          {"range": {"@timestamp": {"gte": "now-1h"}}}
        ]
      }
    }
  }'
```

---

## 🛠️ 维护操作指南

### 1. 系统更新

#### 应用服务更新

```bash
# 1. 创建备份
./create-backup.sh --service rqa2025-core

# 2. 构建新镜像
docker build -t rqa2025-core:v1.2.0 .

# 3. 更新Kubernetes部署
kubectl set image deployment/rqa2025-core core=rqa2025-core:v1.2.0

# 4. 验证更新
kubectl rollout status deployment/rqa2025-core

# 5. 健康检查
./health-check.sh --service rqa2025-core
```

#### 数据库更新

```bash
# 1. 创建数据库备份
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. 停止应用服务
kubectl scale deployment rqa2025-core --replicas=0

# 3. 执行数据库迁移
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f migration_script.sql

# 4. 启动应用服务
kubectl scale deployment rqa2025-core --replicas=3

# 5. 验证数据一致性
./validate-data-consistency.sh
```

### 2. 容量管理

#### 水平扩缩容

```bash
# 自动扩缩容配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rqa2025-core-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rqa2025-core
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 垂直扩容

```bash
# 垂直扩容操作
kubectl patch deployment rqa2025-core -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "core",
          "resources": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"}
          }
        }]
      }
    }
  }
}'
```

### 3. 备份恢复

#### 数据库备份策略

```bash
# 全量备份脚本
#!/bin/bash
BACKUP_DIR="/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建全量备份
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --format=custom \
  --compress=9 \
  --verbose \
  --file=$BACKUP_DIR/full_backup_$TIMESTAMP.dump

# 上传到对象存储
aws s3 cp $BACKUP_DIR/full_backup_$TIMESTAMP.dump s3://rqa2025-backups/database/

# 清理过期备份
find $BACKUP_DIR -name "full_backup_*.dump" -mtime +30 -delete
```

#### 配置文件备份

```bash
# 配置文件备份脚本
#!/bin/bash
CONFIG_BACKUP_DIR="/data/config-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 备份Kubernetes配置
kubectl get all --all-namespaces -o yaml > $CONFIG_BACKUP_DIR/k8s-config-$TIMESTAMP.yaml

# 备份应用配置
cp -r /etc/rqa2025/config $CONFIG_BACKUP_DIR/app-config-$TIMESTAMP/

# 备份Helm Chart值
helm get values rqa2025-core > $CONFIG_BACKUP_DIR/helm-values-$TIMESTAMP.yaml

# 压缩备份
tar -czf $CONFIG_BACKUP_DIR/config-backup-$TIMESTAMP.tar.gz $CONFIG_BACKUP_DIR/*$TIMESTAMP*

# 上传到对象存储
aws s3 cp $CONFIG_BACKUP_DIR/config-backup-$TIMESTAMP.tar.gz s3://rqa2025-backups/config/
```

#### 数据恢复流程

```bash
# 数据恢复脚本
#!/bin/bash
BACKUP_FILE=$1

# 1. 停止应用服务
kubectl scale deployment rqa2025-core --replicas=0

# 2. 恢复数据库
pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --clean \
  --if-exists \
  --verbose \
  $BACKUP_FILE

# 3. 验证数据完整性
./validate-data-integrity.sh

# 4. 重启应用服务
kubectl scale deployment rqa2025-core --replicas=3

# 5. 验证业务功能
./business-function-test.sh
```

---

## 🚨 故障排除指南

### 1. 常见故障

#### 服务启动失败

**现象**：Pod处于CrashLoopBackOff状态

**排查步骤**：
```bash
# 1. 查看Pod状态
kubectl get pods | grep rqa2025-core

# 2. 查看详细状态
kubectl describe pod <pod-name>

# 3. 查看日志
kubectl logs <pod-name> --previous

# 4. 检查资源限制
kubectl get resourcequota
```

**解决方法**：
```bash
# 增加资源限制
kubectl patch deployment rqa2025-core -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "core",
          "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "1000m", "memory": "2Gi"}
          }
        }]
      }
    }
  }
}'
```

#### 数据库连接异常

**现象**：应用无法连接数据库

**排查步骤**：
```bash
# 1. 检查数据库服务状态
kubectl get svc | grep postgres

# 2. 测试数据库连接
kubectl exec -it <pod-name> -- psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1;"

# 3. 检查网络策略
kubectl get networkpolicy

# 4. 查看数据库日志
kubectl logs <postgres-pod-name>
```

**解决方法**：
```bash
# 重启数据库服务
kubectl rollout restart deployment postgres

# 或者更新连接配置
kubectl patch configmap db-config -p '{"data":{"DB_HOST":"new-host"}}'
```

#### 高CPU使用率

**现象**：CPU使用率持续超过80%

**排查步骤**：
```bash
# 1. 查看资源使用
kubectl top pods

# 2. 查看具体进程
kubectl exec -it <pod-name> -- top -c

# 3. 分析性能热点
kubectl exec -it <pod-name> -- python -m cProfile /app/main.py

# 4. 检查内存泄漏
kubectl exec -it <pod-name> -- python -c "import psutil; print(psutil.virtual_memory())"
```

**解决方法**：
```bash
# 增加CPU资源
kubectl set resources deployment rqa2025-core -c=core --limits=cpu=2000m

# 或进行水平扩容
kubectl scale deployment rqa2025-core --replicas=5
```

### 2. 性能调优

#### 数据库调优

```sql
-- 创建索引优化查询
CREATE INDEX CONCURRENTLY idx_trades_symbol_time
ON trades (symbol, trade_time DESC);

-- 优化查询语句
EXPLAIN ANALYZE
SELECT * FROM trades
WHERE symbol = 'AAPL'
  AND trade_time >= '2024-01-01'
  AND trade_time < '2024-12-31';

-- 配置连接池参数
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
```

#### 应用性能调优

```python
# 异步处理优化
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_trades_async(trades):
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single_trade, trade)
            for trade in trades
        ]
        return await asyncio.gather(*tasks)

# 缓存优化
from functools import lru_cache
import redis

@lru_cache(maxsize=1000)
def get_market_data_cached(symbol, date):
    return redis_client.get(f"market:{symbol}:{date}")

# 内存优化
import gc
def memory_optimization():
    gc.collect()  # 主动垃圾回收
    # 清理大对象
    del large_dataframe
    gc.collect()
```

---

## 📊 容量规划

### 1. 当前容量评估

#### 计算资源规划

| 资源类型 | 当前使用 | 峰值使用 | 容量规划 | 建议措施 |
|---------|---------|---------|---------|---------|
| **CPU** | 65% | 85% | 3个月内扩容 | 增加节点或升级实例 |
| **内存** | 55% | 75% | 6个月内扩容 | 优化内存使用 |
| **存储** | 45% | 65% | 12个月内扩容 | 定期清理数据 |
| **网络** | 40% | 60% | 6个月内扩容 | 优化网络配置 |

#### 业务容量规划

| 业务指标 | 当前值 | 目标值 | 增长率 | 建议措施 |
|---------|-------|-------|-------|---------|
| **TPS** | 8,500 | 15,000 | 15%/月 | 3个月内扩容 |
| **并发用户** | 2,000 | 5,000 | 20%/月 | 6个月内扩容 |
| **数据存储** | 2TB | 10TB | 25%/月 | 12个月内扩容 |
| **API调用** | 100K/天 | 500K/天 | 30%/月 | 3个月内扩容 |

### 2. 扩容策略

#### 水平扩容方案

```yaml
# HPA配置 - 基于CPU和内存自动扩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rqa2025-trading-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rqa2025-trading
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Custom
    custom:
      metric:
        name: trading_queue_depth
      target:
        type: AverageValue
        averageValue: "100"
```

#### 垂直扩容方案

```yaml
# 垂直扩容配置
apiVersion: v1
kind: ResourceQuota
metadata:
  name: rqa2025-resource-quota
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 32Gi
    limits.cpu: "20"
    limits.memory: 64Gi
```

---

## 📋 安全运维

### 1. 访问控制

#### RBAC权限配置

```yaml
# Kubernetes RBAC配置
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: rqa2025-admin
rules:
- apiGroups: [""]
  resources: ["pods", "services", "deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["extensions"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: rqa2025-admin-binding
roleRef:
  apiVersion: rbac.authorization.k8s.io/v1
  kind: ClusterRole
  name: rqa2025-admin
subjects:
- kind: User
  name: admin-user
  apiGroup: rbac.authorization.k8s.io
```

#### 应用级访问控制

```python
# Flask应用访问控制
from flask import request, abort
from functools import wraps

def require_role(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.has_role(role):
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/admin/trades')
@login_required
@require_role('admin')
def admin_trades():
    # 只有管理员才能访问
    pass
```

### 2. 安全监控

#### 日志审计配置

```bash
# 审计日志收集
apiVersion: v1
kind: ConfigMap
metadata:
  name: audit-config
data:
  audit-policy.yaml: |
    apiVersion: audit.k8s.io/v1
    kind: Policy
    rules:
    - level: RequestResponse
      verbs: ["create", "update", "delete"]
      resources:
      - group: ""
        resources: ["pods", "services"]
      - group: "apps"
        resources: ["deployments", "replicasets"]
    - level: Metadata
      verbs: ["get", "list", "watch"]
```

#### 安全扫描

```bash
# 容器安全扫描
docker scan rqa2025-core:latest

# 依赖漏洞扫描
safety check --file requirements.txt

# SAST静态应用安全测试
bandit -r src/

# DAST动态应用安全测试
owasp-zap -cmd -quickurl https://rqa2025.example.com
```

---

## 📞 联系与支持

### 内部联系方式
- **运维团队**：ops@rqa2025.com
- **技术支持**：tech-support@rqa2025.com
- **紧急联系**：emergency@rqa2025.com

### 外部支持资源
- **云服务商**：阿里云技术支持
- **开源社区**：Kubernetes、PostgreSQL社区
- **安全厂商**：安全产品技术支持

### 知识库
- **内部Wiki**：https://wiki.rqa2025.com
- **运维文档**：https://docs.rqa2025.com/ops
- **最佳实践**：https://docs.rqa2025.com/best-practices

---

## 🔗 相关文档

- [应急响应手册](EMERGENCY_RESPONSE_MANUAL.md)
- [系统监控手册](SYSTEM_MONITORING_MANUAL.md)
- [故障排除指南](TROUBLESHOOTING_GUIDE.md)
- [备份恢复手册](BACKUP_RECOVERY_MANUAL.md)
- [安全运维指南](SECURITY_OPERATIONS_GUIDE.md)

---

**文档维护人**：RQA2025运维团队
**最后更新**：2024年12月
**版本**：V1.0

*本手册将定期更新，请关注最新版本*
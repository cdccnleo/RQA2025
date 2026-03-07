# RQA2026 部署指南

## 📋 目录

- [环境要求](./environment.md)
- [Docker部署](./docker_deployment.md)
- [Kubernetes部署](./kubernetes_deployment.md)
- [生产环境配置](./production_config.md)
- [监控和运维](./monitoring.md)
- [备份和恢复](./backup_recovery.md)
- [故障排除](./troubleshooting.md)

## 🚀 快速部署

### 单机Docker部署 (推荐用于开发/测试)

```bash
# 克隆项目
git clone https://github.com/rqa2026/RQA2026.git
cd RQA2026

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 生产环境Kubernetes部署

```bash
# 部署到Kubernetes集群
kubectl apply -f k8s/

# 等待服务就绪
kubectl wait --for=condition=ready pod -l app=rqa2026

# 查看服务状态
kubectl get pods,services,ingress
```

## 🏗️ 架构概述

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │ Service Registry│    │  Config Center  │
│   (Traefik)     │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
          ┌─────────┴─────────┐     ┌─────────┴─────────┐
          │  Quantum Engine  │     │   AI Engine      │
          │                  │     │                   │
          └──────────────────┘     └───────────────────┘
                    │
          ┌─────────┴─────────┐
          │   BMI Engine     │
          │                  │
          └──────────────────┘
```

## 📦 部署组件

### 核心服务

| 服务 | 端口 | 描述 | 依赖 |
|------|------|------|------|
| api-gateway | 8000 | API网关 | - |
| quantum-engine | 8001 | 量子计算引擎 | Qiskit |
| ai-engine | 8002 | AI分析引擎 | Transformers, PyTorch |
| bmi-engine | 8003 | 脑机接口引擎 | MNE, NumPy |
| service-registry | 8500 | 服务注册中心 | Redis |
| config-center | 8501 | 配置中心 | etcd/PostgreSQL |
| data-lake | 8502 | 多模态数据湖 | MinIO |

### 基础设施

| 组件 | 版本 | 用途 |
|------|------|------|
| PostgreSQL | 15+ | 主数据库 |
| Redis | 7+ | 缓存和会话存储 |
| MinIO | latest | 对象存储 |
| Prometheus | 2.40+ | 监控指标收集 |
| Grafana | 9+ | 可视化仪表板 |
| Jaeger | 1.40+ | 分布式追踪 |

## ⚙️ 配置管理

### 环境变量

```bash
# 应用配置
export RQA_ENV=production
export RQA_LOG_LEVEL=INFO
export RQA_SECRET_KEY=your-secret-key

# 数据库配置
export DATABASE_URL=postgresql://user:password@db:5432/rqa2026
export REDIS_URL=redis://redis:6379/0

# 外部服务
export QUANTUM_BACKEND=ibm_quantum
export AI_MODEL_PATH=/models
export BMI_DEVICE=/dev/eeg

# 监控配置
export PROMETHEUS_URL=http://prometheus:9090
export JAEGER_URL=http://jaeger:14268/api/traces
```

### 配置文件

```yaml
# config/production.yaml
app:
  name: RQA2026
  version: "1.0.0"
  environment: production

database:
  host: postgresql
  port: 5432
  database: rqa2026
  username: ${DATABASE_USER}
  password: ${DATABASE_PASSWORD}

cache:
  redis:
    host: redis
    port: 6379
    db: 0

storage:
  minio:
    endpoint: minio:9000
    access_key: ${MINIO_ACCESS_KEY}
    secret_key: ${MINIO_SECRET_KEY}
    bucket: rqa2026-data

monitoring:
  prometheus:
    enabled: true
    path: /metrics
  jaeger:
    enabled: true
    service_name: rqa2026

quantum:
  backend: ibm_quantum
  token: ${IBM_QUANTUM_TOKEN}
  hub: ibm-q
  group: open
  project: main

ai:
  model_cache: /models
  transformers_cache: /transformers
  gpu_memory_fraction: 0.8

bmi:
  sampling_rate: 250
  channels: 32
  buffer_size: 1000
  quality_threshold: 0.8
```

## 🐳 Docker部署

### 构建镜像

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash rqa
USER rqa

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "scripts/start_production.py"]
```

### Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  api-gateway:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RQA_ENV=production
    depends_on:
      - quantum-engine
      - ai-engine
      - bmi-engine
    networks:
      - rqa2026

  quantum-engine:
    build: .
    environment:
      - RQA_SERVICE=quantum-engine
      - QUANTUM_BACKEND=aer_simulator
    networks:
      - rqa2026

  ai-engine:
    build: .
    environment:
      - RQA_SERVICE=ai-engine
    volumes:
      - ./models:/models
    networks:
      - rqa2026

  bmi-engine:
    build: .
    environment:
      - RQA_SERVICE=bmi-engine
    devices:
      - /dev/eeg:/dev/eeg
    networks:
      - rqa2026

  database:
    image: postgres:15
    environment:
      - POSTGRES_DB=rqa2026
      - POSTGRES_USER=rqa2026
      - POSTGRES_PASSWORD=${DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rqa2026

  redis:
    image: redis:7-alpine
    networks:
      - rqa2026

  minio:
    image: minio/minio
    environment:
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
    command: server /data
    volumes:
      - minio_data:/data
    networks:
      - rqa2026

networks:
  rqa2026:
    driver: bridge

volumes:
  postgres_data:
  minio_data:
```

## ☸️ Kubernetes部署

### 命名空间

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rqa2026
  labels:
    name: rqa2026
```

### 配置映射

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rqa2026-config
  namespace: rqa2026
data:
  config.yaml: |
    app:
      name: RQA2026
      version: "1.0.0"
      environment: production
    database:
      host: postgresql
      port: 5432
```

### 密钥管理

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rqa2026-secrets
  namespace: rqa2026
type: Opaque
data:
  database-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  ibm-quantum-token: <base64-encoded-token>
```

### 部署配置

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: rqa2026
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: rqa2026/api-gateway:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: RQA_ENV
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 服务配置

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: rqa2026
spec:
  selector:
    app: api-gateway
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 入口配置

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rqa2026-ingress
  namespace: rqa2026
  annotations:
    kubernetes.io/ingress.class: "traefik"
spec:
  rules:
  - host: api.rqa2026.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8000
```

## 📊 监控和运维

### Prometheus配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rqa2026'
    static_configs:
      - targets: ['api-gateway:8000', 'quantum-engine:8001', 'ai-engine:8002', 'bmi-engine:8003']
    metrics_path: '/metrics'
```

### Grafana仪表板

预配置的仪表板包括：

- **系统概览**: CPU、内存、磁盘使用率
- **服务性能**: 响应时间、吞吐量、错误率
- **业务指标**: 投资组合优化成功率、AI分析准确性
- **量子计算**: 电路执行时间、量子优势指标

### 日志聚合

```yaml
# logging/fluent-bit-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [INPUT]
        Name tail
        Path /var/log/containers/*rqa2026*.log
        Parser docker

    [OUTPUT]
        Name es
        Host elasticsearch
        Port 9200
        Index rqa2026-logs
```

## 🔄 备份和恢复

### 数据备份

```bash
# 数据库备份
docker exec rqa2026_database pg_dump -U rqa2026 rqa2026 > backup_$(date +%Y%m%d_%H%M%S).sql

# 对象存储备份
mc mirror rqa2026-data/ backup/$(date +%Y%m%d)/

# 配置备份
kubectl get configmap,secret -n rqa2026 -o yaml > config_backup_$(date +%Y%m%d).yaml
```

### 灾难恢复

```bash
# 1. 恢复数据库
docker exec -i rqa2026_database psql -U rqa2026 rqa2026 < backup_file.sql

# 2. 恢复对象存储
mc mirror backup/$(date +%Y%m%d)/ rqa2026-data/

# 3. 恢复配置
kubectl apply -f config_backup_file.yaml

# 4. 重启服务
kubectl rollout restart deployment -n rqa2026
```

## 🚨 故障排除

### 常见问题

#### 服务启动失败

```bash
# 检查日志
kubectl logs -f deployment/api-gateway -n rqa2026

# 检查资源使用
kubectl top pods -n rqa2026

# 检查配置
kubectl describe configmap rqa2026-config -n rqa2026
```

#### 性能问题

```bash
# 检查资源限制
kubectl get pods -o jsonpath='{.spec.containers[*].resources}' -n rqa2026

# 监控指标
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# 访问 http://localhost:9090
```

#### 网络连接问题

```bash
# 检查服务发现
kubectl exec -it deployment/api-gateway -- nslookup quantum-engine.rqa2026.svc.cluster.local

# 检查网络策略
kubectl get networkpolicy -n rqa2026
```

## 📈 扩展和优化

### 水平扩展

```bash
# 增加副本数
kubectl scale deployment api-gateway --replicas=5 -n rqa2026

# 自动扩缩容
kubectl autoscale deployment api-gateway --cpu-percent=70 --min=3 --max=10 -n rqa2026
```

### 性能优化

1. **启用缓存**: Redis集群缓存热点数据
2. **数据库优化**: 读写分离、索引优化
3. **异步处理**: 使用消息队列处理重任务
4. **CDN加速**: 静态资源和API响应缓存

### 安全加固

```yaml
# k8s/security-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rqa2026-security
  namespace: rqa2026
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

## 📞 支持与维护

### 定期维护任务

- **每日**: 检查服务健康状态和日志
- **每周**: 更新安全补丁和依赖包
- **每月**: 性能评估和容量规划
- **每季度**: 全面安全审计和备份验证

### 联系支持

- **紧急问题**: 24/7热线 +1-800-RQA2026
- **技术支持**: support@rqa2026.com
- **文档更新**: docs@rqa2026.com
- **社区论坛**: https://community.rqa2026.com

---

**🚀 按照此指南，您可以快速部署和维护RQA2026创新系统。如有问题，请参考故障排除章节或联系技术支持。**
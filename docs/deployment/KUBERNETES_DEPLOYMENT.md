# RQA2025 Kubernetes部署指南

## 📋 部署概述

**版本**：V1.0
**更新日期**：2024年12月
**适用环境**：生产环境、测试环境
**部署方式**：Kubernetes + Helm

---

## 🏗️ 架构设计

### 1. Kubernetes集群架构

```
RQA2025 Kubernetes架构
==============================================
┌─────────────────────────────────────────┐
│              负载均衡层                  │
│  ┌─────────────────────────────────┐   │
│  │        Nginx Ingress          │   │
│  │  ┌────────┬────────┬────────┐ │   │
│  │  │ Web UI │ API GW │ Mobile │ │   │
│  │  └────────┴────────┴────────┘ │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              服务层                     │
│  ┌─────────────────────────────────┐   │
│  │         微服务网格             │   │
│  │  ┌─────┬─────┬─────┬─────┐   │   │
│  │  │交易  │策略  │风控  │数据  │   │   │
│  │  │引擎  │管理  │管理  │处理  │   │   │
│  │  └─────┴─────┴─────┴─────┘   │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              数据层                     │
│  ┌─────────────────────────────────┐   │
│  │         存储服务               │   │
│  │  ┌─────┬─────┬─────┬─────┐   │   │
│  │  │Postgre│Redis │Kafka │MinIO│   │   │
│  │  │SQL   │     │     │     │   │   │
│  │  └─────┴─────┴─────┴─────┘   │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              基础设施层                 │
│  ┌─────────────────────────────────┐   │
│  │       Kubernetes集群           │   │
│  │  ┌─────┬─────┬─────┬─────┐   │   │
│  │  │Master│Node1 │Node2 │Node3 │   │   │
│  │  └─────┴─────┴─────┴─────┘   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2. 服务组件

#### 核心应用服务
- **rqa2025-core**: 主应用服务 (交易引擎、策略管理)
- **rqa2025-api**: API网关服务
- **rqa2025-web**: Web界面服务
- **rqa2025-mobile**: 移动端服务

#### 数据服务
- **postgresql**: 主数据库
- **redis**: 缓存和会话存储
- **kafka**: 消息队列
- **minio**: 对象存储
- **influxdb**: 时序数据库

#### 基础设施服务
- **prometheus**: 监控数据收集
- **grafana**: 可视化监控面板
- **elasticsearch**: 日志聚合
- **fluent-bit**: 日志收集器

---

## 📋 部署准备

### 1. 环境要求

#### 硬件要求

| 组件 | CPU | 内存 | 存储 | 网络 |
|-----|-----|------|------|------|
| **Master节点** | 4核 | 8GB | 100GB SSD | 1Gbps |
| **Worker节点** | 8核 | 32GB | 500GB SSD | 10Gbps |
| **数据库节点** | 16核 | 64GB | 2TB NVMe | 10Gbps |
| **缓存节点** | 8核 | 32GB | 200GB SSD | 10Gbps |

#### 软件要求

```bash
# Kubernetes版本
Kubernetes: v1.28+
Container Runtime: containerd v1.7+
CNI Plugin: Calico v3.25+
CSI Driver: AWS EBS CSI Driver v1.20+

# Helm版本
Helm: v3.12+

# 操作系统
Ubuntu 22.04 LTS 或 RHEL 8.8+
Kernel: 5.15+
```

### 2. 网络规划

#### 网络分段

```yaml
# Calico网络策略示例
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rqa2025-network-policy
  namespace: rqa2025
spec:
  podSelector:
    matchLabels:
      app: rqa2025
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
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

#### 域名规划

| 服务 | 内部域名 | 外部域名 | 端口 |
|-----|---------|---------|------|
| Web界面 | web.rqa2025.svc.cluster.local | web.rqa2025.com | 80/443 |
| API网关 | api.rqa2025.svc.cluster.local | api.rqa2025.com | 80/443 |
| 移动端 | mobile.rqa2025.svc.cluster.local | mobile.rqa2025.com | 80/443 |
| 管理后台 | admin.rqa2025.svc.cluster.local | admin.rqa2025.com | 80/443 |

### 3. 存储规划

#### 存储类配置

```yaml
# AWS EBS存储类
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: rqa2025-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
  kmsKeyId: "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

#### PVC配置

```yaml
# PostgreSQL数据存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgresql-data
  namespace: rqa2025
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: rqa2025-ssd
  resources:
    requests:
      storage: 1Ti

# Redis缓存存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-data
  namespace: rqa2025
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: rqa2025-ssd
  resources:
    requests:
      storage: 100Gi
```

---

## 🚀 部署步骤

### 1. 环境初始化

#### 创建命名空间

```bash
# 创建RQA2025命名空间
kubectl create namespace rqa2025

# 设置命名空间默认配置
kubectl label namespace rqa2025 name=rqa2025
kubectl annotate namespace rqa2025 description="RQA2025量化交易系统命名空间"
```

#### 配置服务账户

```yaml
# 创建服务账户
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rqa2025-service-account
  namespace: rqa2025
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/rqa2025-eks-role

---
# 创建角色绑定
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: rqa2025-admin-binding
roleRef:
  apiVersion: rbac.authorization.k8s.io/v1
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: rqa2025-service-account
  namespace: rqa2025
```

### 2. 部署数据服务

#### PostgreSQL部署

```yaml
# PostgreSQL StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: rqa2025
spec:
  serviceName: postgresql
  replicas: 3
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      serviceAccountName: rqa2025-service-account
      containers:
      - name: postgresql
        image: postgres:15.4
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: rqa2025
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgresql-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: 2
            memory: 4Gi
          limits:
            cpu: 4
            memory: 8Gi
  volumeClaimTemplates:
  - metadata:
      name: postgresql-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: rqa2025-ssd
      resources:
        requests:
          storage: 1Ti
```

#### Redis部署

```yaml
# Redis部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: rqa2025
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
        readinessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 15
          periodSeconds: 20
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-data
```

### 3. 部署应用服务

#### 核心应用部署

```yaml
# RQA2025核心服务部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-core
  namespace: rqa2025
  labels:
    app: rqa2025-core
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rqa2025-core
  template:
    metadata:
      labels:
        app: rqa2025-core
        version: v1.0.0
    spec:
      serviceAccountName: rqa2025-service-account
      containers:
      - name: core
        image: rqa2025/rqa2025-core:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: rqa2025-config
              key: db.host
        - name: DB_PORT
          valueFrom:
            configMapKeyRef:
              name: rqa2025-config
              key: db.port
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: rqa2025-config
              key: redis.host
        resources:
          requests:
            cpu: 2
            memory: 4Gi
          limits:
            cpu: 4
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: rqa2025-config
      - name: logs-volume
        emptyDir: {}
```

#### API网关部署

```yaml
# API网关部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-api-gateway
  namespace: rqa2025
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rqa2025-api-gateway
  template:
    metadata:
      labels:
        app: rqa2025-api-gateway
    spec:
      containers:
      - name: api-gateway
        image: rqa2025/rqa2025-api-gateway:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: CORE_SERVICE_URL
          value: "http://rqa2025-core:8080"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1
            memory: 2Gi
```

### 4. 配置网络访问

#### Ingress配置

```yaml
# Nginx Ingress配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rqa2025-ingress
  namespace: rqa2025
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.rqa2025.com
    - web.rqa2025.com
    - mobile.rqa2025.com
    secretName: rqa2025-tls
  rules:
  - host: api.rqa2025.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rqa2025-api-gateway
            port:
              number: 80
  - host: web.rqa2025.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rqa2025-web
            port:
              number: 80
  - host: mobile.rqa2025.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rqa2025-mobile
            port:
              number: 80
```

### 5. 部署监控系统

#### Prometheus部署

```yaml
# Prometheus部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: data
        persistentVolumeClaim:
          claimName: prometheus-data
```

#### Grafana部署

```yaml
# Grafana部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.1.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-password
        volumeMounts:
        - name: data
          mountPath: /var/lib/grafana
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1
            memory: 2Gi
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: grafana-data
```

---

## 🔧 配置管理

### 1. ConfigMap配置

```yaml
# 应用配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: rqa2025-config
  namespace: rqa2025
data:
  # 数据库配置
  db.host: "postgresql.rqa2025.svc.cluster.local"
  db.port: "5432"
  db.name: "rqa2025"
  db.max_connections: "100"

  # Redis配置
  redis.host: "redis.rqa2025.svc.cluster.local"
  redis.port: "6379"
  redis.db: "0"
  redis.max_connections: "50"

  # Kafka配置
  kafka.brokers: "kafka-0.kafka.rqa2025.svc.cluster.local:9092,kafka-1.kafka.rqa2025.svc.cluster.local:9092"
  kafka.topic_trades: "trades"
  kafka.topic_signals: "signals"

  # 应用配置
  app.env: "production"
  app.debug: "false"
  app.log_level: "INFO"
  app.max_workers: "4"
```

### 2. Secret配置

```yaml
# 敏感信息配置
apiVersion: v1
kind: Secret
metadata:
  name: rqa2025-secrets
  namespace: rqa2025
type: Opaque
data:
  # 数据库密码 (base64编码)
  db.password: "c3VwZXJzZWNyZXRwYXNzd29yZA=="

  # API密钥
  api.key: "eW91cl9hcGlfa2V5X2hlcmU="

  # JWT密钥
  jwt.secret: "eW91cl9qd3Rfc2VjcmV0X2hlcmU="

  # 加密密钥
  encryption.key: "eW91cl9lbmNyeXB0aW9uX2tleQ=="
```

---

## 📊 部署验证

### 1. 部署状态检查

```bash
# 检查Pod状态
kubectl get pods -n rqa2025

# 检查服务状态
kubectl get svc -n rqa2025

# 检查Ingress状态
kubectl get ingress -n rqa2025

# 检查PVC状态
kubectl get pvc -n rqa2025
```

### 2. 应用健康检查

```bash
# 检查应用健康状态
curl -f https://api.rqa2025.com/health

# 检查数据库连接
kubectl exec -it deployment/rqa2025-core -n rqa2025 -- python -c "
import psycopg2
conn = psycopg2.connect('host=postgresql dbname=rqa2025 user=rqa2025 password=password')
print('Database connection successful')
"

# 检查Redis连接
kubectl exec -it deployment/rqa2025-core -n rqa2025 -- python -c "
import redis
r = redis.Redis(host='redis', decode_responses=True)
r.set('test', 'success')
print('Redis connection successful')
"
```

### 3. 性能基准测试

```bash
# API性能测试
ab -n 1000 -c 10 https://api.rqa2025.com/api/v1/market-data/

# 数据库性能测试
kubectl exec -it deployment/postgresql-0 -n rqa2025 -- pgbench -c 10 -j 2 -T 60 rqa2025

# Redis性能测试
kubectl exec -it deployment/redis-0 -n rqa2025 -- redis-benchmark -n 10000 -c 10
```

---

## 🛡️ 安全配置

### 1. 网络安全

#### NetworkPolicy配置

```yaml
# 网络安全策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rqa2025-security-policy
  namespace: rqa2025
spec:
  podSelector:
    matchLabels:
      app: rqa2025-core
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: rqa2025-api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: kafka
    ports:
    - protocol: TCP
      port: 9092
  - to: [] # 允许DNS解析
    ports:
    - protocol: UDP
      port: 53
```

### 2. 安全扫描

#### 容器镜像安全扫描

```bash
# Trivy容器安全扫描
trivy image rqa2025/rqa2025-core:v1.0.0

# 漏洞扫描报告
trivy image --format json --output results.json rqa2025/rqa2025-core:v1.0.0

# 安全合规检查
trivy image --severity HIGH,CRITICAL rqa2025/rqa2025-core:v1.0.0
```

#### 集群安全配置

```yaml
# Pod安全标准
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: rqa2025
  labels:
    app: rqa2025-core
spec:
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    runAsNonRoot: true
  containers:
  - name: core
    image: rqa2025/rqa2025-core:v1.0.0
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 1000
      runAsGroup: 1000
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp-volume
      mountPath: /tmp
    - name: cache-volume
      mountPath: /app/cache
  volumes:
  - name: tmp-volume
    emptyDir: {}
  - name: cache-volume
    emptyDir: {}
```

---

## 📋 部署清单

### 部署验证清单

- [ ] Kubernetes集群已创建并可访问
- [ ] 所有命名空间和RBAC已配置
- [ ] 存储类和PVC已创建
- [ ] ConfigMap和Secret已配置
- [ ] PostgreSQL已部署并可访问
- [ ] Redis已部署并可访问
- [ ] Kafka已部署并可访问
- [ ] MinIO已部署并可访问
- [ ] Prometheus监控已部署
- [ ] Grafana可视化已配置
- [ ] RQA2025核心服务已部署
- [ ] API网关已部署
- [ ] Web界面已部署
- [ ] 移动端服务已部署
- [ ] Ingress已配置并可访问
- [ ] SSL证书已配置
- [ ] 网络安全策略已应用
- [ ] 健康检查通过
- [ ] 性能基准测试通过

### 回滚计划

#### 快速回滚脚本

```bash
#!/bin/bash
# RQA2025回滚脚本

# 设置变量
NAMESPACE="rqa2025"
DEPLOYMENT="rqa2025-core"
ROLLBACK_TO="v0.9.0"

echo "开始回滚 $DEPLOYMENT 到 $ROLLBACK_TO"

# 1. 检查当前版本
CURRENT_IMAGE=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "当前镜像: $CURRENT_IMAGE"

# 2. 执行回滚
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# 3. 等待回滚完成
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# 4. 验证回滚结果
NEW_IMAGE=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
echo "回滚后镜像: $NEW_IMAGE"

# 5. 健康检查
kubectl exec -it deployment/$DEPLOYMENT -n $NAMESPACE -- curl -f http://localhost:8080/health || echo "健康检查失败"

echo "回滚完成"
```

---

## 📞 联系支持

### 部署支持
- **技术支持**：devops@rqa2025.com
- **紧急联系**：emergency@rqa2025.com
- **云服务商**：阿里云技术支持

### 相关文档
- [系统架构文档](../../ARCHITECTURE.md)
- [运维手册](../operations/OPERATIONS_MANUAL.md)
- [应急响应手册](../operations/EMERGENCY_RESPONSE_MANUAL.md)
- [监控系统部署](../monitoring/MONITORING_DEPLOYMENT.md)

---

**文档维护人**：RQA2025 DevOps团队
**最后更新**：2024年12月
**版本**：V1.0

*请在部署前仔细阅读本指南，如有问题及时联系技术支持*

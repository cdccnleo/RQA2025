# RQA2025 部署文档

## 1. 部署概述
RQA2025系统采用容器化部署方案...

## 2. 环境要求
### 2.1 硬件要求
#### 生产环境
- CPU: 32 cores (Intel Xeon 或 AMD EPYC)
- 内存: 128 GB DDR4
- 存储: 2TB NVMe SSD
- 网络: 10Gbps Ethernet

#### 测试环境
- CPU: 8 cores
- 内存: 32 GB
- 存储: 500GB SSD
- 网络: 1Gbps Ethernet

### 2.2 软件要求
- Kubernetes 1.28+
- Docker 24.0+
- PostgreSQL 15.0+
- Redis 7.0+
- Nginx Ingress Controller

## 3. 部署前准备
### 3.1 基础设施准备
#### 网络配置
```bash
# 创建网络策略
kubectl apply -f infrastructure/configs/network/network-policy.yaml
```

#### 存储配置
```bash
# 创建存储类
kubectl apply -f infrastructure/configs/storage/storage-class.yaml

# 创建持久卷
kubectl apply -f infrastructure/configs/storage/postgresql-pvc.yaml
kubectl apply -f infrastructure/configs/storage/redis-pvc.yaml
```

### 3.2 证书和密钥
#### TLS证书
```bash
# 创建TLS证书
kubectl create secret tls rqa2025-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n production
```

#### 数据库密码
```bash
# 创建数据库密码
kubectl create secret generic postgresql-secret \
  --from-literal=password='your_password' \
  -n production
```

### 3.3 配置管理
#### ConfigMap创建
```bash
# 创建应用配置
kubectl create configmap rqa2025-config \
  --from-file=config.yaml \
  -n production
```

## 4. 数据库部署
### 4.1 PostgreSQL部署
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: production
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
      containers:
      - name: postgresql
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: rqa2025_db
        - name: POSTGRES_USER
          value: rqa2025
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
        volumeMounts:
        - name: postgresql-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
    name: postgresql-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 500Gi
```

### 4.2 Redis部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: production
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
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
```

## 5. 应用部署
### 5.1 API服务部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rqa2025-api
  template:
    metadata:
      labels:
        app: rqa2025-api
    spec:
      containers:
      - name: api
        image: rqa2025/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          value: redis://redis:6379
        resources:
          requests:
            memory: "2Gi"
            cpu: "2"
          limits:
            memory: "4Gi"
            cpu: "4"
```

### 5.2 Web界面部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-web
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rqa2025-web
  template:
    metadata:
      labels:
        app: rqa2025-web
    spec:
      containers:
      - name: web
        image: rqa2025/web:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1"
```

## 6. 网络配置
### 6.1 Ingress配置
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rqa2025-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.rqa2025.com
    - web.rqa2025.com
    secretName: rqa2025-tls
  rules:
  - host: api.rqa2025.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rqa2025-api
            port:
              number: 8000
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
```

### 6.2 服务发现
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rqa2025-api
  namespace: production
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: rqa2025-api
```

## 7. 监控部署
### 7.1 Prometheus部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: data
        persistentVolumeClaim:
          claimName: prometheus-pvc
```

### 7.2 Grafana部署
```yaml
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
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: password
        volumeMounts:
        - name: data
          mountPath: /var/lib/grafana
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: grafana-pvc
```

## 8. 部署验证
### 8.1 健康检查
```bash
# 检查Pod状态
kubectl get pods -n production

# 检查服务状态
kubectl get services -n production

# 检查Ingress状态
kubectl get ingress -n production
```

### 8.2 功能测试
```bash
# API健康检查
curl https://api.rqa2025.com/health

# 数据库连接测试
kubectl exec -it postgresql-0 -- psql -U rqa2025 -d rqa2025_db -c "SELECT version();"

# Redis连接测试
kubectl exec -it redis-0 -- redis-cli ping
```

### 8.3 性能测试
```bash
# 运行性能测试
kubectl run performance-test --image=performance-test:latest \
  --restart=Never \
  --rm -it \
  -- /bin/bash -c "python performance_test.py"
```

## 9. 回滚策略
### 9.1 应用回滚
```bash
# 查看部署历史
kubectl rollout history deployment/rqa2025-api

# 回滚到上一个版本
kubectl rollout undo deployment/rqa2025-api

# 回滚到指定版本
kubectl rollout undo deployment/rqa2025-api --to-revision=2
```

### 9.2 数据库回滚
```bash
# 执行数据库回滚脚本
kubectl run db-rollback --image=postgres:15 \
  --restart=Never \
  --rm -it \
  -- psql -h postgresql -U rqa2025 -d rqa2025_db -f rollback.sql
```

## 10. 扩展和维护
### 10.1 水平扩展
```bash
# 扩展API服务
kubectl scale deployment rqa2025-api --replicas=5

# 扩展数据库
kubectl scale statefulset postgresql --replicas=5
```

### 10.2 垂直扩展
```bash
# 更新资源限制
kubectl set resources deployment rqa2025-api \
  -c api \
  --requests=cpu=4,memory=8Gi \
  --limits=cpu=8,memory=16Gi
```

### 10.3 更新策略
```bash
# 滚动更新
kubectl set image deployment/rqa2025-api api=rqa2025/api:v2.0.0

# 蓝绿部署
kubectl apply -f blue-green-deployment.yaml
```

## 11. 故障排除
### 11.1 常见问题
#### Pod启动失败
```bash
# 查看Pod详情
kubectl describe pod <pod-name>

# 查看Pod日志
kubectl logs <pod-name>
```

#### 服务无法访问
```bash
# 检查服务状态
kubectl get endpoints <service-name>

# 检查网络策略
kubectl get networkpolicies
```

### 11.2 诊断工具
```bash
# 集群诊断
kubectl cluster-info dump

# 网络诊断
kubectl run network-test --image=busybox --rm -it \
  -- wget -qO- http://<service-name>

# 性能诊断
kubectl top nodes
kubectl top pods
```

## 附录
### 部署清单
- [ ] 基础设施准备完成
- [ ] 证书和密钥配置完成
- [ ] ConfigMap创建完成
- [ ] 数据库部署完成
- [ ] 应用部署完成
- [ ] 网络配置完成
- [ ] 监控部署完成
- [ ] 部署验证完成
- [ ] 文档更新完成

### 版本信息
- Kubernetes: 1.28.0
- Docker: 24.0.0
- PostgreSQL: 15.0
- Redis: 7.0
- Nginx Ingress: 1.8.0

---
*版本：2.0.0 | 更新日期：2025-01-15 | 作者：DevOps团队*

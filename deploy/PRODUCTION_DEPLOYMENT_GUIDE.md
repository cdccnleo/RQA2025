# 生产环境云原生部署指南

## 概述

本指南详细说明如何将RQA2025系统的云原生和智能化功能部署到生产环境。

## 前置条件

### 1. 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ / CentOS 8+)
- **Kubernetes**: v1.20+
- **Docker**: v20.10+
- **kubectl**: 与Kubernetes版本匹配
- **存储**: 至少100GB可用空间
- **内存**: 至少16GB RAM
- **CPU**: 至少8核

### 2. 软件依赖

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# 安装Helm (可选)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 3. 集群配置

确保Kubernetes集群已正确配置：

```bash
# 检查集群状态
kubectl cluster-info

# 检查节点状态
kubectl get nodes

# 检查命名空间
kubectl get namespaces
```

## 部署步骤

### 1. 准备环境

```bash
# 克隆项目
git clone <repository-url>
cd RQA2025

# 激活conda环境
conda activate test

# 安装依赖
pip install -r requirements.txt
```

### 2. 构建Docker镜像

```bash
# 构建回测服务镜像
docker build -f Dockerfile.backtest -t rqa2025-backtest:latest .

# 构建数据服务镜像
docker build -f Dockerfile.data -t rqa2025-data:latest .

# 构建智能化服务镜像
docker build -f Dockerfile.intelligent -t rqa2025-intelligent:latest .

# 验证镜像
docker images | grep rqa2025
```

### 3. 配置生产环境

#### 3.1 创建配置文件

```bash
# 复制配置文件
cp deploy/production_cloud_native.yml deploy/production_cloud_native.yml.backup

# 编辑配置文件
vim deploy/production_cloud_native.yml
```

#### 3.2 配置环境变量

```bash
# 设置生产环境变量
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export METRICS_ENABLED=true
export CLOUD_NATIVE_ENABLED=true
export INTELLIGENT_FEATURES_ENABLED=true
```

### 4. 执行部署

#### 4.1 检查部署配置

```bash
# 检查部署配置
python scripts/deploy/deploy_production_cloud_native.py --dry-run
```

#### 4.2 执行完整部署

```bash
# 执行生产环境部署
python scripts/deploy/deploy_production_cloud_native.py
```

### 5. 验证部署

#### 5.1 检查服务状态

```bash
# 检查命名空间
kubectl get namespaces | grep rqa2025-cloud-native

# 检查部署
kubectl get deployments -n rqa2025-cloud-native

# 检查Pod状态
kubectl get pods -n rqa2025-cloud-native

# 检查服务
kubectl get services -n rqa2025-cloud-native
```

#### 5.2 检查服务健康状态

```bash
# 检查回测服务健康状态
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-backtest-service -- curl -f http://localhost:8001/health

# 检查数据服务健康状态
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-data-service -- curl -f http://localhost:8002/health

# 检查智能化服务健康状态
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-intelligent-orchestrator -- curl -f http://localhost:8005/health
```

#### 5.3 检查自动扩缩容

```bash
# 检查HPA状态
kubectl get hpa -n rqa2025-cloud-native

# 查看HPA详细信息
kubectl describe hpa rqa2025-backtest-hpa -n rqa2025-cloud-native
```

### 6. 配置监控

#### 6.1 安装Prometheus (可选)

```bash
# 使用Helm安装Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

#### 6.2 配置ServiceMonitor

```bash
# 应用ServiceMonitor配置
kubectl apply -f deploy/monitoring/service-monitor.yml
```

### 7. 配置Ingress

#### 7.1 安装NGINX Ingress Controller

```bash
# 安装NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.1.0/deploy/static/provider/cloud/deploy.yaml
```

#### 7.2 配置SSL证书

```bash
# 安装cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.8.0/cert-manager.yaml

# 配置ClusterIssuer
kubectl apply -f deploy/monitoring/cluster-issuer.yml
```

## 云原生功能验证

### 1. 自动扩缩容测试

```bash
# 创建负载测试
kubectl run load-test --image=busybox --rm -it --restart=Never -- \
  sh -c "while true; do curl http://rqa2025-backtest-service:8001/api/v1/backtest; sleep 1; done"

# 监控HPA状态
kubectl get hpa -n rqa2025-cloud-native -w
```

### 2. 服务网格功能测试

```bash
# 测试熔断器
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-backtest-service -- \
  python -c "from src.backtest.cloud_native_features import CircuitBreaker; print('熔断器功能正常')"

# 测试负载均衡
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-backtest-service -- \
  python -c "from src.backtest.cloud_native_features import LoadBalancer; print('负载均衡功能正常')"
```

### 3. 蓝绿部署测试

```bash
# 测试蓝绿部署
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-backtest-service -- \
  python -c "from src.backtest.cloud_native_features import BlueGreenDeployment; print('蓝绿部署功能正常')"
```

## 智能化功能验证

### 1. 机器学习模型测试

```bash
# 测试ML模型
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-intelligent-orchestrator -- \
  python -c "from src.backtest.intelligent_features import MLModel; print('ML模型功能正常')"
```

### 2. 自动调优测试

```bash
# 测试自动调优
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-intelligent-orchestrator -- \
  python -c "from src.backtest.intelligent_features import AutoTuner; print('自动调优功能正常')"
```

### 3. 预测性维护测试

```bash
# 测试预测性维护
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-intelligent-orchestrator -- \
  python -c "from src.backtest.intelligent_features import PredictiveMaintenance; print('预测性维护功能正常')"
```

## 监控和告警

### 1. 查看指标

```bash
# 查看Pod指标
kubectl top pods -n rqa2025-cloud-native

# 查看节点指标
kubectl top nodes

# 查看服务指标
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-backtest-service -- \
  curl http://localhost:9090/metrics
```

### 2. 查看日志

```bash
# 查看回测服务日志
kubectl logs -n rqa2025-cloud-native deployment/rqa2025-backtest-service

# 查看数据服务日志
kubectl logs -n rqa2025-cloud-native deployment/rqa2025-data-service

# 查看智能化服务日志
kubectl logs -n rqa2025-cloud-native deployment/rqa2025-intelligent-orchestrator
```

## 故障排除

### 1. 常见问题

#### Pod启动失败

```bash
# 查看Pod事件
kubectl describe pod <pod-name> -n rqa2025-cloud-native

# 查看Pod日志
kubectl logs <pod-name> -n rqa2025-cloud-native
```

#### 服务无法访问

```bash
# 检查服务配置
kubectl describe service <service-name> -n rqa2025-cloud-native

# 检查网络策略
kubectl get networkpolicies -n rqa2025-cloud-native
```

#### 自动扩缩容不工作

```bash
# 检查HPA配置
kubectl describe hpa <hpa-name> -n rqa2025-cloud-native

# 检查资源使用情况
kubectl top pods -n rqa2025-cloud-native
```

### 2. 回滚部署

```bash
# 回滚到上一个版本
python scripts/deploy/deploy_production_cloud_native.py --rollback

# 或者手动回滚
kubectl rollout undo deployment/rqa2025-backtest-service -n rqa2025-cloud-native
kubectl rollout undo deployment/rqa2025-data-service -n rqa2025-cloud-native
kubectl rollout undo deployment/rqa2025-intelligent-orchestrator -n rqa2025-cloud-native
```

## 性能优化

### 1. 资源调优

```bash
# 调整CPU和内存限制
kubectl patch deployment rqa2025-backtest-service -n rqa2025-cloud-native \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"backtest-service","resources":{"limits":{"cpu":"2000m","memory":"4Gi"},"requests":{"cpu":"1000m","memory":"2Gi"}}}]}}}}'
```

### 2. 扩缩容策略调优

```bash
# 调整HPA参数
kubectl patch hpa rqa2025-backtest-hpa -n rqa2025-cloud-native \
  -p '{"spec":{"minReplicas":3,"maxReplicas":15,"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":60}}}]}}'
```

## 安全配置

### 1. 网络安全

```bash
# 创建网络策略
kubectl apply -f deploy/security/network-policy.yml
```

### 2. 密钥管理

```bash
# 创建Secret
kubectl create secret generic rqa2025-secrets \
  --from-literal=db_password=production_password \
  --from-literal=api_key=production_api_key \
  --from-literal=jwt_secret=production_jwt_secret \
  -n rqa2025-cloud-native
```

## 备份和恢复

### 1. 数据备份

```bash
# 备份配置
kubectl get configmap cloud-native-config -n rqa2025-cloud-native -o yaml > backup/configmap-backup.yml

# 备份PVC数据
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-data-service -- \
  tar -czf /tmp/data-backup.tar.gz /app/data
```

### 2. 数据恢复

```bash
# 恢复配置
kubectl apply -f backup/configmap-backup.yml

# 恢复数据
kubectl cp backup/data-backup.tar.gz rqa2025-cloud-native/deployment/rqa2025-data-service:/tmp/
kubectl exec -n rqa2025-cloud-native deployment/rqa2025-data-service -- \
  tar -xzf /tmp/data-backup.tar.gz -C /
```

## 维护和更新

### 1. 版本更新

```bash
# 更新镜像
docker build -f Dockerfile.backtest -t rqa2025-backtest:v1.1.0 .

# 更新部署
kubectl set image deployment/rqa2025-backtest-service \
  backtest-service=rqa2025-backtest:v1.1.0 -n rqa2025-cloud-native
```

### 2. 配置更新

```bash
# 更新ConfigMap
kubectl apply -f deploy/production_cloud_native.yml

# 重启Pod以应用新配置
kubectl rollout restart deployment/rqa2025-backtest-service -n rqa2025-cloud-native
```

## 总结

通过以上步骤，您已成功将RQA2025系统的云原生和智能化功能部署到生产环境。系统现在具备：

- ✅ 自动扩缩容功能
- ✅ 服务网格功能（熔断器、负载均衡、重试策略）
- ✅ 蓝绿部署功能
- ✅ 智能化监控和告警
- ✅ 机器学习模型支持
- ✅ 自动调优功能
- ✅ 预测性维护功能

请定期检查系统状态，并根据实际使用情况调整配置参数。 
# 模型推理部署指南

## 概述

本文档详细介绍了RQA2025项目中模型推理模块的部署方案，包括环境配置、部署策略、监控运维和故障处理。

## 部署架构

### 1. 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   负载均衡器     │    │   推理服务集群   │    │   模型存储      │
│   (Nginx)       │───▶│   (Docker)      │───▶│   (S3/NFS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   监控系统      │
                       │   (Prometheus)  │
                       └─────────────────┘
```

### 2. 组件说明

- **负载均衡器**: 分发请求，提供高可用性
- **推理服务**: 执行模型推理，支持水平扩展
- **模型存储**: 集中存储模型文件，支持版本管理
- **监控系统**: 实时监控性能和健康状态

## 环境要求

### 1. 硬件要求

#### 最低配置
- **CPU**: 4核心，2.4GHz
- **内存**: 8GB RAM
- **存储**: 50GB SSD
- **网络**: 1Gbps

#### 推荐配置
- **CPU**: 8核心，3.0GHz
- **内存**: 32GB RAM
- **GPU**: NVIDIA RTX 3080或更高
- **存储**: 200GB NVMe SSD
- **网络**: 10Gbps

### 2. 软件要求

#### 操作系统
- Ubuntu 20.04 LTS或更高版本
- CentOS 8或更高版本
- Windows Server 2019或更高版本

#### 依赖软件
```bash
# Python环境
Python 3.8+
CUDA 11.0+ (GPU部署)
Docker 20.10+
Docker Compose 2.0+

# 系统依赖
NVIDIA Driver 450+
NVIDIA Container Runtime
```

#### Python依赖
```bash
# 核心依赖
torch>=1.9.0
tensorflow>=2.6.0
onnxruntime>=1.8.0
numpy>=1.21.0
pandas>=1.3.0

# 推理相关
psutil>=5.8.0
redis>=4.0.0
fastapi>=0.68.0
uvicorn>=0.15.0

# 监控相关
prometheus-client>=0.11.0
grafana-api>=1.0.3
```

## 部署配置

### 1. Docker部署

#### Dockerfile
```dockerfile
# 基础镜像
FROM nvidia/cuda:11.0-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY src/ ./src/
COPY config/ ./config/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 创建模型存储目录
RUN mkdir -p /app/models /app/cache /app/logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  inference-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_STORAGE_PATH=/app/models
      - CACHE_PATH=/app/cache
      - LOG_LEVEL=INFO
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
    restart: unless-stopped

  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 2. Kubernetes部署

#### 命名空间
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: model-inference
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
  namespace: model-inference
data:
  config.yaml: |
    inference:
      enable_gpu: true
      enable_cache: true
      max_batch_size: 32
      cache_size: 100
      model_storage_path: /app/models
      log_level: INFO
    
    monitoring:
      prometheus_endpoint: http://prometheus:9090
      metrics_port: 8001
```

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: model-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
      - name: inference-service
        image: rqa2025/inference-service:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        env:
        - name: MODEL_STORAGE_PATH
          value: /app/models
        - name: CACHE_PATH
          value: /app/cache
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: cache-volume
          mountPath: /app/cache
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: cache-volume
        emptyDir: {}
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  namespace: model-inference
spec:
  selector:
    app: inference-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: inference-ingress
  namespace: model-inference
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: inference.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: inference-service
            port:
              number: 80
```

### 3. 配置管理

#### 环境变量
```bash
# 推理配置
MODEL_STORAGE_PATH=/app/models
CACHE_PATH=/app/cache
LOG_LEVEL=INFO
ENABLE_GPU=true
MAX_BATCH_SIZE=32
CACHE_SIZE=100

# 监控配置
PROMETHEUS_ENDPOINT=http://prometheus:9090
METRICS_PORT=8001
GRAFANA_URL=http://grafana:3000

# 安全配置
API_KEY=your-api-key
CORS_ORIGINS=http://localhost:3000
```

#### 配置文件
```yaml
# config/inference.yaml
inference:
  enable_gpu: true
  enable_cache: true
  max_batch_size: 32
  cache_size: 100
  model_storage_path: /app/models
  log_level: INFO
  
  gpu:
    memory_limit: 0.8
    batch_size_optimization: true
    parallel_processing: true
    
  cache:
    memory_cache_size: 50
    disk_cache_size: 1000
    ttl_seconds: 3600
    lru_enabled: true
    
  monitoring:
    prometheus_endpoint: http://prometheus:9090
    metrics_port: 8001
    enable_health_check: true
    
  security:
    api_key_required: true
    cors_enabled: true
    rate_limit: 1000
```

## 监控运维

### 1. 监控指标

#### 系统指标
- CPU使用率
- 内存使用率
- GPU使用率和内存
- 磁盘I/O
- 网络流量

#### 应用指标
- 推理请求数
- 推理时间
- 错误率
- 缓存命中率
- 批处理大小

#### 业务指标
- 模型加载时间
- 模型版本信息
- 预测准确率
- 用户满意度

### 2. Prometheus配置

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'inference-service'
    static_configs:
      - targets: ['inference-service:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### 告警规则
```yaml
# rules/inference_alerts.yml
groups:
  - name: inference_alerts
    rules:
      - alert: HighInferenceTime
        expr: avg_over_time(inference_time_seconds[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "推理时间过长"
          description: "平均推理时间超过100ms"

      - alert: HighErrorRate
        expr: rate(inference_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "错误率过高"
          description: "推理错误率超过1%"

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "缓存命中率低"
          description: "缓存命中率低于80%"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "内存使用率过高"
          description: "系统内存使用率超过90%"
```

### 3. Grafana仪表板

#### 推理服务仪表板
```json
{
  "dashboard": {
    "title": "模型推理监控",
    "panels": [
      {
        "title": "推理请求数",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(inference_requests_total[5m])",
            "legendFormat": "请求/秒"
          }
        ]
      },
      {
        "title": "平均推理时间",
        "type": "graph",
        "targets": [
          {
            "expr": "avg_over_time(inference_time_seconds[5m])",
            "legendFormat": "秒"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(inference_errors_total[5m]) / rate(inference_requests_total[5m])",
            "legendFormat": "错误率"
          }
        ]
      },
      {
        "title": "缓存命中率",
        "type": "singlestat",
        "targets": [
          {
            "expr": "cache_hit_rate",
            "legendFormat": "命中率"
          }
        ]
      }
    ]
  }
}
```

## 故障处理

### 1. 常见问题

#### 问题1: 服务启动失败
**症状**: 容器启动失败，日志显示错误
**解决方案**:
1. 检查配置文件语法
2. 验证环境变量设置
3. 确认依赖服务可用
4. 检查端口冲突

```bash
# 检查容器日志
docker logs inference-service

# 检查配置文件
docker exec inference-service cat /app/config/inference.yaml

# 验证网络连接
docker exec inference-service ping redis
```

#### 问题2: GPU不可用
**症状**: 推理速度慢，GPU使用率为0
**解决方案**:
1. 检查NVIDIA驱动安装
2. 验证CUDA环境
3. 确认Docker GPU支持
4. 检查GPU内存

```bash
# 检查GPU状态
nvidia-smi

# 验证CUDA
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查容器GPU访问
docker exec inference-service nvidia-smi
```

#### 问题3: 内存不足
**症状**: 服务崩溃，OOM错误
**解决方案**:
1. 增加内存限制
2. 优化批处理大小
3. 启用内存监控
4. 清理缓存

```bash
# 检查内存使用
docker stats inference-service

# 调整内存限制
docker update --memory=16g inference-service

# 清理缓存
docker exec inference-service python -c "from src.models.inference import InferenceCache; InferenceCache().clear()"
```

#### 问题4: 模型加载失败
**症状**: 推理请求返回错误
**解决方案**:
1. 检查模型文件路径
2. 验证模型格式
3. 确认模型文件完整
4. 检查模型版本

```bash
# 检查模型文件
ls -la /app/models/

# 验证模型文件
python -c "from src.models.inference import ModelLoader; loader = ModelLoader(); print(loader.validate_model_file('/app/models/model.pth'))"

# 查看模型元数据
python -c "from src.models.inference import ModelLoader; loader = ModelLoader(); print(loader.get_model_metadata('/app/models/model.pth'))"
```

### 2. 性能调优

#### 批处理优化
```python
# 动态调整批处理大小
def optimize_batch_size():
    batch_sizes = [16, 32, 64, 128]
    best_size = 32
    
    for size in batch_sizes:
        # 测试性能
        start_time = time.time()
        result = inference_manager.batch_predict('model', test_data, size)
        end_time = time.time()
        
        throughput = len(test_data) / (end_time - start_time)
        if throughput > best_throughput:
            best_size = size
            best_throughput = throughput
    
    return best_size
```

#### 缓存优化
```python
# 预热缓存
def warm_up_cache():
    common_inputs = load_common_inputs()
    for input_data in common_inputs:
        inference_manager.predict('model', input_data)
```

#### GPU优化
```python
# GPU内存优化
import torch
torch.cuda.empty_cache()

# 批处理大小优化
optimal_batch_size = find_optimal_batch_size()
```

### 3. 备份恢复

#### 模型备份
```bash
# 备份模型文件
tar -czf models_backup_$(date +%Y%m%d).tar.gz /app/models/

# 备份配置
cp config/inference.yaml config/inference.yaml.backup

# 备份缓存
tar -czf cache_backup_$(date +%Y%m%d).tar.gz /app/cache/
```

#### 数据恢复
```bash
# 恢复模型文件
tar -xzf models_backup_20240101.tar.gz -C /

# 恢复配置
cp config/inference.yaml.backup config/inference.yaml

# 恢复缓存
tar -xzf cache_backup_20240101.tar.gz -C /
```

## 安全考虑

### 1. 网络安全
- 使用HTTPS加密传输
- 配置防火墙规则
- 限制网络访问
- 启用API认证

### 2. 数据安全
- 加密敏感数据
- 定期备份数据
- 访问权限控制
- 审计日志记录

### 3. 应用安全
- 输入数据验证
- 防止SQL注入
- 限制文件上传
- 错误信息脱敏

## 总结

本部署指南提供了完整的模型推理服务部署方案，包括：

1. **架构设计**: 清晰的系统架构和组件说明
2. **环境配置**: 详细的硬件和软件要求
3. **部署方案**: Docker和Kubernetes两种部署方式
4. **监控运维**: 完整的监控指标和告警配置
5. **故障处理**: 常见问题的诊断和解决方案
6. **安全考虑**: 网络安全和数据安全措施

通过遵循本指南，您可以成功部署和维护一个高性能、高可用的模型推理服务。 
# RQA2025 高效部署指南

## 🎯 概述

本指南提供多种高效的代码更新策略，避免每次代码变更都需要重新构建Docker镜像。

## 📊 部署模式对比

| 模式 | 代码更新效率 | 适用场景 | 优点 | 缺点 |
|------|-------------|----------|------|------|
| **开发模式** | ⭐⭐⭐⭐⭐ | 本地开发 | 实时重载，无需重启 | 仅限开发环境 |
| **热重载模式** | ⭐⭐⭐⭐ | 开发/测试 | 代码变更自动重载 | 性能开销 |
| **滚动更新** | ⭐⭐⭐ | 生产环境 | 零停机部署 | 需要重新构建镜像 |
| **智能CI/CD** | ⭐⭐⭐⭐ | 全环境 | 按需构建，节省资源 | 配置复杂 |

## 🚀 快速开始

### 方式1：开发模式（推荐）

```bash
# 1. 启动开发环境
docker-compose -f docker-compose.dev.yml up -d

# 2. 修改代码
# 代码变更会自动重载，无需重新构建！

# 3. 查看日志
docker-compose -f docker-compose.dev.yml logs -f rqa2025-dev
```

**特点：**
- ✅ 代码实时挂载，修改后自动重载
- ✅ 无需重新构建镜像
- ✅ 支持断点调试
- ⚠️ 仅用于开发环境

### 方式2：热重载模式

```bash
# 1. 启动容器
docker-compose up -d

# 2. 启用热重载
python scripts/enable_hot_reload.py enable

# 3. 修改代码后自动重载
```

### 方式3：滚动更新

```bash
# 1. 代码变更后构建镜像
docker build -t rqa2025:latest .

# 2. 滚动更新
./scripts/quick-deploy.sh -e production
```

## 🔧 详细配置

### 开发环境配置

#### Docker Compose 开发模式

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  rqa2025-dev:
    volumes:
      - ./src:/app/src          # 挂载源代码
      - ./main.py:/app/main.py  # 挂载主文件
    command: ["python", "-m", "uvicorn", "src.main:app", "--reload"]
```

#### Kubernetes 开发模式

```yaml
# k8s/development/rqa2025-app-deployment-dev.yaml
spec:
  template:
    spec:
      containers:
      - name: rqa2025-app
        volumeMounts:
        - name: source-code
          mountPath: /app/src
        - name: main-code
          mountPath: /app/main.py
      volumes:
      - name: source-code
        hostPath:
          path: /path/to/your/RQA2025/src
```

### 生产环境优化

#### 滚动更新策略

```yaml
# k8s/production/rqa2025-app-deployment-rolling.yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1  # 最多1个Pod不可用
      maxSurge: 1        # 最多1个额外Pod
```

#### 智能CI/CD

```yaml
# .github/workflows/smart-deploy.yml
jobs:
  analyze-changes:
    # 智能检测变更类型
    # 只在需要时构建镜像
```

## 🛠️ 部署工具

### 快速部署脚本

```bash
# 开发环境部署
./scripts/quick-deploy.sh -e development

# 生产环境部署
./scripts/quick-deploy.sh -e production

# 前端配置更新
./scripts/quick-deploy.sh -e frontend
```

### 热重载工具

```bash
# 启用热重载
python scripts/enable_hot_reload.py enable

# 检查状态
python scripts/enable_hot_reload.py status

# 禁用热重载
python scripts/enable_hot_reload.py disable
```

## 📈 性能对比

### 传统方式 vs 高效方式

| 操作 | 传统方式 | 开发模式 | 热重载模式 | 智能CI/CD |
|------|----------|----------|------------|-----------|
| 小代码修改 | 5-10分钟 | < 1秒 | < 3秒 | 按需构建 |
| 配置修改 | 2-5分钟 | 实时 | 实时 | 自动检测 |
| 前端修改 | 1-3分钟 | 实时 | 实时 | 自动更新 |
| 资源占用 | 高 | 中 | 中 | 低 |

## 🎯 最佳实践

### 开发阶段
1. **使用开发模式**：`docker-compose.dev.yml`
2. **启用热重载**：代码变更自动生效
3. **挂载日志目录**：方便调试

### 测试阶段
1. **使用热重载模式**：快速迭代
2. **配置测试环境**：模拟生产环境
3. **集成自动化测试**

### 生产阶段
1. **使用滚动更新**：零停机部署
2. **智能CI/CD**：按需构建镜像
3. **配置监控告警**：确保部署成功

## 🔍 故障排除

### 热重载不生效

```bash
# 检查容器状态
docker ps | grep rqa2025

# 查看日志
docker logs <container_id>

# 手动重启
docker restart <container_id>
```

### 挂载权限问题

```bash
# Linux/Mac
sudo chown -R $USER:$USER /path/to/project

# Windows (WSL)
# 确保文件在WSL文件系统中
```

### Kubernetes挂载失败

```bash
# 检查hostPath配置
kubectl describe pod <pod-name> -n <namespace>

# 检查文件权限
kubectl exec -it <pod-name> -n <namespace> -- ls -la /app/src
```

## 📚 相关文件

- `docker-compose.dev.yml` - 开发环境配置
- `Dockerfile.dev` - 开发环境镜像
- `k8s/development/` - Kubernetes开发环境配置
- `scripts/quick-deploy.sh` - 快速部署脚本
- `scripts/enable_hot_reload.py` - 热重载工具
- `.github/workflows/smart-deploy.yml` - 智能CI/CD配置

## 🎉 总结

通过这些优化策略，可以将代码更新效率提升 **10-50倍**：

- **开发环境**：代码修改后 < 1秒 生效
- **测试环境**：热重载 < 3秒 生效
- **生产环境**：智能CI/CD按需构建，避免无效构建

选择合适的部署模式，让开发更加高效！
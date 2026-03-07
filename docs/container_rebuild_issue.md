# 容器重建问题诊断

## 问题描述

即使使用 `--no-cache` 重新构建镜像，容器中的启动脚本仍然是旧版本。

### 检查结果

1. **本地文件**（已更新）：
   - 包含 `asyncio.run(server.serve())` ✅
   - MD5: `A0EDFB37D764955F647D8A30448FF5D8`

2. **镜像中的文件**（旧版本）：
   - 仍使用 `uvicorn.run()` ❌
   - MD5: `7062e0c6c52ad15d82652c9ae5e74921`

3. **文件哈希值不同**：
   - 说明 Docker 构建时没有正确复制最新文件

## 可能的原因

1. **Docker 构建缓存问题**：
   - 即使使用 `--no-cache`，某些层可能仍被缓存
   - 需要完全清理所有相关镜像和容器

2. **构建上下文问题**：
   - Dockerfile 的构建上下文可能不包含最新文件
   - 需要检查 `.dockerignore` 是否排除了 scripts 目录

3. **文件时间戳问题**：
   - Docker 可能基于文件时间戳判断是否需要复制
   - 需要确保文件修改时间正确

## 解决方案

### 方案1：完全清理并重建

```powershell
# 1. 停止并删除所有相关容器
docker-compose stop rqa2025-app
docker-compose rm -f rqa2025-app

# 2. 删除镜像（强制）
docker rmi -f rqa2025-app:latest

# 3. 清理构建缓存
docker builder prune -f

# 4. 重新构建（不使用缓存）
docker-compose build --no-cache rqa2025-app

# 5. 启动容器
docker-compose up -d rqa2025-app
```

### 方案2：使用 volume 挂载 scripts 目录（临时方案）

修改 `docker-compose.yml`，临时挂载 scripts 目录：

```yaml
volumes:
  - ./scripts:/app/scripts:ro  # 临时挂载，确保使用最新脚本
```

**注意**：这只是临时方案，不推荐用于生产环境。

### 方案3：验证构建过程

```powershell
# 1. 检查构建日志，确认 COPY scripts 步骤
docker-compose build --no-cache --progress=plain rqa2025-app 2>&1 | Select-String -Pattern "COPY scripts"

# 2. 在构建后立即检查镜像中的文件
docker run --rm rqa2025-app:latest cat /app/scripts/start_api_server.py | Select-String -Pattern "asyncio.run|uvicorn.run"
```

## 验证步骤

重建后，验证修复：

```powershell
# 1. 检查容器中的脚本
docker exec rqa2025-rqa2025-app-1 tail -30 /app/scripts/start_api_server.py | Select-String -Pattern "asyncio.run|server.serve"

# 2. 检查启动日志
docker logs rqa2025-rqa2025-app-1 | Select-String -Pattern "🚀 正在启动|服务器配置完成"

# 3. 验证服务
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

## 当前状态

- ✅ 本地脚本已更新
- ❌ 镜像中的脚本仍是旧版本
- ⚠️ 需要完全清理并重建

## 下一步

执行方案1的完整清理和重建流程，确保镜像包含最新的启动脚本。

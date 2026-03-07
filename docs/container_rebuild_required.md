# 容器重建要求

## 问题确认

**关键发现**：容器中的启动脚本是旧版本！

### 检查结果

1. **本地启动脚本**（已修复）：
   - 使用 `asyncio.run(server.serve())` ✅
   - 包含 `import asyncio` ✅
   - 使用 `uvicorn.Server` + `asyncio.run()` ✅

2. **容器中的启动脚本**（旧版本）：
   - 仍使用 `uvicorn.run()` ❌
   - 没有 `import asyncio` ❌
   - 没有 `asyncio.run(server.serve())` ❌

### 原因分析

1. **Dockerfile 构建时复制脚本**：
   ```dockerfile
   COPY scripts/ ./scripts/
   ```
   - 脚本在构建镜像时被复制到镜像中
   - 容器使用的是构建时的脚本版本

2. **docker-compose.yml 未挂载 scripts**：
   ```yaml
   volumes:
     - ./src:/app/src:ro  # 只挂载了 src，没有挂载 scripts
   ```
   - 本地脚本修改不会同步到容器中
   - 需要重新构建镜像

## 解决方案

### 必须执行：重新构建镜像

```powershell
# 1. 停止容器
docker-compose stop rqa2025-app

# 2. 删除旧容器（可选，但推荐）
docker-compose rm -f rqa2025-app

# 3. 重新构建镜像（关键步骤！）
docker-compose build rqa2025-app

# 4. 启动容器
docker-compose up -d rqa2025-app

# 5. 等待服务启动（45秒）
Start-Sleep -Seconds 45

# 6. 验证服务
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

### 验证修复

重建后，检查容器中的启动脚本：

```bash
# 检查启动脚本是否包含最新修复
docker exec rqa2025-rqa2025-app-1 tail -30 /app/scripts/start_api_server.py | grep -E "asyncio.run|server.serve|服务器配置完成"
# 应该看到这些内容
```

## 自动启动检查结果

### ✅ 自动启动机制已配置

1. **重启策略**：`restart: unless-stopped` ✅
2. **启动命令**：`python scripts/start_api_server.py` ✅
3. **Init 系统**：`init: true` ✅（已添加）

### ⚠️ 需要重建容器

- 启动脚本已修复，但容器中的代码未同步
- **必须重新构建镜像**以应用所有修复

## 自动启动流程（重建后）

当容器重启时：

1. **Docker 自动重启**（`restart: unless-stopped`）
2. **执行启动命令**：`python scripts/start_api_server.py`
3. **启动脚本执行**：
   - 导入应用模块
   - 创建 FastAPI 应用
   - 配置 uvicorn 服务器（使用 `uvicorn.Server`）
   - 启动服务器（使用 `asyncio.run(server.serve())`）
4. **服务运行**：
   - uvicorn 监听 0.0.0.0:8000
   - FastAPI 应用启动完成
   - 数据采集调度器后台任务启动

## 总结

✅ **自动启动已配置**：
- 重启策略、启动命令、Init 系统都已正确配置

⚠️ **必须重建容器**：
- 容器中的启动脚本是旧版本
- 需要重新构建镜像以应用最新修复
- 重建后，容器重启时会自动启动API服务

## 相关文档

- [容器后端最终诊断](./container_backend_final_diagnosis.md)
- [容器自动启动检查](./container_auto_startup_check.md)
- [容器自动启动总结](./container_auto_startup_summary.md)

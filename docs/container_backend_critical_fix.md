# 容器后端服务关键修复

## 问题根本原因

经过深入诊断，发现问题的根本原因：

1. **uvicorn.run() 在容器中可能无法正确启动**
   - 日志显示 "Application startup complete" 但没有 "Uvicorn running"
   - 端口8000未监听，说明服务器启动后立即退出

2. **启动方式问题**
   - `uvicorn.run()` 在某些情况下可能无法在容器中正确运行
   - 需要使用 `uvicorn.Server` + `asyncio.run()` 方式

## 修复方案

### 已修复的启动脚本

修改了 `scripts/start_api_server.py`，使用异步方式启动：

```python
# 配置 uvicorn 服务器
config = uvicorn.Config(
    app,
    host="0.0.0.0",
    port=8000,
    reload=False,
    log_level="info",
    access_log=True,
    loop="asyncio"
)
server = uvicorn.Server(config)

# 使用 asyncio.run() 运行服务器
asyncio.run(server.serve())
```

### 验证修复

1. **重建容器**：
   ```bash
   docker-compose stop rqa2025-app
   docker-compose build rqa2025-app
   docker-compose up -d rqa2025-app
   ```

2. **等待服务启动**（45秒）：
   ```bash
   Start-Sleep -Seconds 45
   ```

3. **验证服务**：
   ```bash
   # 从容器内测试
   docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
   
   # 从宿主机测试
   curl http://localhost:8000/health
   ```

## 如果问题仍然存在

如果修复后问题仍然存在，请检查：

1. **容器日志**：
   ```bash
   docker logs rqa2025-rqa2025-app-1 --tail 100
   ```

2. **端口监听**：
   ```bash
   docker exec rqa2025-rqa2025-app-1 sh -c "lsof -i :8000 || ss -tln | grep 8000"
   ```

3. **手动启动测试**：
   ```bash
   docker exec -it rqa2025-rqa2025-app-1 python scripts/start_api_server.py
   ```

## 相关文件

- `scripts/start_api_server.py` - 已修复的启动脚本
- `Dockerfile` - 容器构建配置
- `docker-compose.yml` - 容器编排配置

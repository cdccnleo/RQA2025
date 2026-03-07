# 容器后端服务无法访问问题诊断和修复总结

## 问题现象

容器 `rqa2025-rqa2025-app-1` 重启后，依然无法连接到后端服务（端口8000）。

## 诊断结果

### 1. 容器状态
- ✅ 容器运行正常（状态：Up 2 minutes (healthy)）
- ✅ 端口映射正确（0.0.0.0:8000->8000/tcp）
- ❌ 容器内端口8000未监听
- ❌ HTTP健康检查端点无响应

### 2. 日志分析

从容器日志中发现：

1. **服务启动信息**：
   ```
   INFO:     Started server process [1]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

2. **导入错误**：
   ```
   ImportError: cannot import name 'WebSocketManager' from 'src.gateway.web.websocket_manager'
   ```

3. **启动脚本降级**：
   - 由于导入错误，启动脚本创建了基本的FastAPI应用
   - 基本应用可能缺少完整的路由和端点

### 3. 根本原因

**问题1：导入错误导致应用创建失败**
- `api.py` 在模块级别导入时，某些地方尝试导入 `WebSocketManager`
- 虽然 `websocket_manager.py` 中定义了 `WebSocketManager = ConnectionManager`，但在某些导入时机可能还未定义
- 这导致应用创建失败，启动脚本降级到基本应用

**问题2：服务可能未正确启动**
- 即使日志显示 "Uvicorn running"，但端口未监听
- 可能是 uvicorn 启动后立即退出，或绑定失败

## 修复方案

### 方案1：修复导入问题（推荐）

确保所有导入 `WebSocketManager` 的地方都正确处理：

1. **检查并修复所有导入语句**：
   - 确保使用 `ConnectionManager` 而不是 `WebSocketManager`
   - 或确保 `WebSocketManager` 别名在导入时已定义

2. **延迟导入**：
   - 将可能出问题的导入移到函数内部
   - 使用 try-except 处理导入错误

### 方案2：重建容器

由于容器中的代码可能未同步最新修复：

```bash
# 1. 停止并删除容器
docker-compose down rqa2025-app

# 2. 重新构建镜像
docker-compose build rqa2025-app

# 3. 启动容器
docker-compose up -d rqa2025-app

# 4. 等待服务启动（40秒）
sleep 40

# 5. 验证服务
curl http://localhost:8000/health
```

### 方案3：检查代码同步

确保容器中的代码与本地代码一致：

```bash
# 检查容器中的文件
docker exec rqa2025-rqa2025-app-1 cat /app/src/gateway/web/websocket_manager.py | tail -5

# 应该看到：
# WebSocketManager = ConnectionManager
# websocket_manager = manager
```

## 立即行动

### 步骤1：检查当前状态

```bash
# 查看容器日志
docker logs rqa2025-rqa2025-app-1 --tail 50

# 检查进程
docker exec rqa2025-rqa2025-app-1 ps aux

# 检查端口
docker exec rqa2025-rqa2025-app-1 netstat -tlnp
```

### 步骤2：重建容器

```bash
# 停止容器
docker-compose stop rqa2025-app

# 重新构建（如果需要）
docker-compose build rqa2025-app

# 启动容器
docker-compose up -d rqa2025-app
```

### 步骤3：验证修复

```bash
# 等待服务启动
sleep 45

# 测试健康检查
curl http://localhost:8000/health

# 或从容器内测试
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

## 预防措施

1. **确保代码同步**：
   - 使用 volume 挂载时，确保本地代码是最新的
   - 或使用 COPY 指令确保容器内代码正确

2. **改进错误处理**：
   - 启动脚本应该检测应用创建是否成功
   - 如果失败，应该退出并记录错误

3. **健康检查改进**：
   - 健康检查应该真正验证HTTP端点
   - 增加启动等待时间（已修复：start_period: 40s）

## 相关文件

- `Dockerfile` - 容器构建配置
- `docker-compose.yml` - 容器编排配置
- `scripts/start_api_server.py` - 启动脚本
- `src/gateway/web/api.py` - 主应用文件
- `src/gateway/web/websocket_manager.py` - WebSocket管理器

## 下一步

1. 重建容器以应用最新代码
2. 验证服务是否正常启动
3. 如果问题持续，检查导入错误的根本原因
4. 考虑使用统一启动脚本 `scripts/start_server.py`

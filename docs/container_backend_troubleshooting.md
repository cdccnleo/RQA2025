# 容器后端服务故障排除指南

## 问题描述

容器中的后端API服务无法访问，前端显示 "无法连接到后端服务，请确保后端服务正在运行（端口8000）"。

## 诊断步骤

### 1. 检查容器状态

```bash
# 查看容器是否运行
docker ps | grep rqa2025-app

# 查看容器日志
docker logs rqa2025-app --tail 100

# 查看容器详细信息
docker inspect rqa2025-app
```

### 2. 检查端口监听

```bash
# 在容器内检查端口
docker exec rqa2025-app netstat -tlnp | grep 8000

# 或使用 ss 命令
docker exec rqa2025-app ss -tlnp | grep 8000
```

### 3. 检查健康检查端点

```bash
# 从容器内测试
docker exec rqa2025-app python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"

# 从宿主机测试
curl http://localhost:8000/health
```

### 4. 运行诊断脚本

```bash
# 在容器内运行诊断脚本
docker exec rqa2025-app python scripts/check_container_backend.py
```

## 常见问题及解决方案

### 问题1：容器启动但服务未运行

**症状**：
- 容器状态为 "Up"，但端口8000未监听
- 日志显示启动脚本执行但服务未启动

**可能原因**：
1. 启动脚本执行失败但容器未退出
2. uvicorn 启动失败
3. 应用导入错误

**解决方案**：

1. **检查启动日志**：
   ```bash
   docker logs rqa2025-app --tail 200
   ```

2. **手动测试启动脚本**：
   ```bash
   docker exec -it rqa2025-app python scripts/start_api_server.py
   ```

3. **检查应用导入**：
   ```bash
   docker exec rqa2025-app python -c "from src.gateway.web.app_factory import create_app; app = create_app(); print(f'路由数: {len(app.routes)}')"
   ```

### 问题2：服务启动但无法从外部访问

**症状**：
- 容器内端口8000正在监听
- 容器内可以访问 http://localhost:8000/health
- 但宿主机无法访问

**可能原因**：
1. uvicorn 绑定到 127.0.0.1 而不是 0.0.0.0
2. 端口映射配置错误
3. 防火墙阻止

**解决方案**：

1. **检查 uvicorn 绑定地址**：
   确保启动脚本中使用 `host="0.0.0.0"`：
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8000, ...)
   ```

2. **检查端口映射**：
   ```bash
   docker port rqa2025-app
   # 应该显示: 8000/tcp -> 0.0.0.0:8000
   ```

3. **检查 docker-compose.yml**：
   ```yaml
   ports:
     - "8000:8000"  # 确保格式正确
   ```

### 问题3：服务启动但健康检查失败

**症状**：
- 服务已启动
- 但健康检查端点无响应或返回错误

**可能原因**：
1. 健康检查端点未正确实现
2. 服务启动但未完全初始化
3. 健康检查命令不正确

**解决方案**：

1. **验证健康检查端点**：
   ```bash
   docker exec rqa2025-app curl http://localhost:8000/health
   ```

2. **检查健康检查配置**：
   - Dockerfile 中的 HEALTHCHECK 命令
   - docker-compose.yml 中的 healthcheck 配置

3. **增加启动等待时间**：
   如果服务需要较长时间初始化，增加 `start_period`：
   ```yaml
   healthcheck:
     start_period: 40s  # 增加到40秒
   ```

### 问题4：lifespan 启动事件问题

**症状**：
- 应用创建成功
- 但 lifespan 中的启动逻辑未执行
- 数据采集调度器未启动

**可能原因**：
1. lifespan 上下文管理器未正确设置
2. 启动验证函数失败导致阻塞

**解决方案**：

1. **检查 lifespan 是否正确设置**：
   ```bash
   docker exec rqa2025-app python -c "from src.gateway.web.app_factory import create_app; app = create_app(); print('lifespan:', hasattr(app, 'router'))"
   ```

2. **查看启动日志**：
   查找 "后端服务启动事件触发" 消息

3. **检查启动验证函数**：
   验证函数使用非阻塞方式，不应阻塞服务启动

## 修复措施

### 已修复的问题

1. **健康检查命令**：
   - 修复前：只检查 Python 是否可用
   - 修复后：检查 HTTP 健康检查端点

2. **启动等待时间**：
   - 修复前：start_period: 10s（太短）
   - 修复后：start_period: 40s（足够服务启动）

3. **启动验证**：
   - 使用非阻塞方式验证服务就绪
   - 验证失败不影响服务启动

### 验证修复

运行以下命令验证修复：

```bash
# 1. 重建容器
docker-compose build rqa2025-app

# 2. 重启服务
docker-compose up -d rqa2025-app

# 3. 等待服务启动（40秒）
sleep 40

# 4. 检查健康状态
docker ps | grep rqa2025-app
# 应该显示 "healthy" 状态

# 5. 测试端点
curl http://localhost:8000/health
```

## 预防措施

1. **使用统一启动脚本**：
   推荐使用 `scripts/start_server.py`，包含完整的验证逻辑

2. **监控容器日志**：
   ```bash
   docker logs -f rqa2025-app
   ```

3. **设置资源限制**：
   确保容器有足够的 CPU 和内存资源

4. **定期健康检查**：
   使用监控系统定期检查服务健康状态

## 相关文档

- [后端服务启动指南](./backend_startup_guide.md)
- [Docker 部署文档](./DOCKER_DEPLOYMENT_README.md)

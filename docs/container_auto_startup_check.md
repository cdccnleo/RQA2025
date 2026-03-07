# 容器自动启动检查报告

## 检查结果

### 1. 重启策略配置

**docker-compose.yml**：
```yaml
rqa2025-app:
  restart: unless-stopped
  init: true  # 使用 init 系统处理信号和回收僵尸进程
```

**当前容器状态**：
- 重启策略：`unless-stopped` ✅
- 启动命令：`["python", "scripts/start_api_server.py"]` ✅
- Init 系统：`true` ✅（已添加，解决僵尸进程问题）

**说明**：
- `unless-stopped` 表示容器会在停止后自动重启（除非手动停止）
- 容器启动时会自动执行 `CMD` 或 `command` 中指定的命令
- `init: true` 使用 Docker 内置的 tini init 系统，处理信号和回收僵尸进程

### 2. 启动命令配置

**Dockerfile**：
```dockerfile
CMD ["python", "scripts/start_api_server.py"]
```

**docker-compose.yml**：
```yaml
command: ["python", "scripts/start_api_server.py"]
```

**说明**：
- docker-compose.yml 中的 `command` 会覆盖 Dockerfile 中的 `CMD`
- 容器启动时会自动执行启动脚本

### 3. 自动启动流程

当容器重启时，Docker 会：

1. **执行启动命令**：
   ```bash
   python scripts/start_api_server.py
   ```

2. **启动脚本执行流程**：
   - 导入应用模块
   - 创建 FastAPI 应用
   - 启动 uvicorn 服务器

3. **服务启动**：
   - uvicorn 服务器监听 0.0.0.0:8000
   - FastAPI 应用启动完成
   - 数据采集调度器后台任务启动

### 4. 验证自动启动

#### 测试容器重启

```bash
# 1. 重启容器
docker restart rqa2025-rqa2025-app-1

# 2. 等待服务启动（45秒）
Start-Sleep -Seconds 45

# 3. 检查容器状态
docker ps | grep rqa2025-app
# 应该显示 "Up" 状态

# 4. 检查启动日志
docker logs rqa2025-rqa2025-app-1 --tail 50
# 应该看到启动相关的日志

# 5. 验证服务
docker exec rqa2025-rqa2025-app-1 python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

#### 测试自动重启

```bash
# 1. 停止容器（模拟崩溃）
docker stop rqa2025-rqa2025-app-1

# 2. 等待几秒
Start-Sleep -Seconds 5

# 3. 检查容器是否自动重启
docker ps -a | grep rqa2025-app
# 应该显示容器已自动重启（状态：Up）

# 4. 验证服务
curl http://localhost:8000/health
```

## 已修复的问题

### 问题1：僵尸进程导致容器无法重启

**症状**：
- 容器重启失败，错误：`container PID is zombie and can not be killed`
- 容器状态异常，无法正常重启

**原因**：
- 容器中没有 init 系统来处理信号和回收僵尸进程
- 子进程变成僵尸进程后无法被正确清理

**解决方案**：
- ✅ 已在 docker-compose.yml 中添加 `init: true`
- ✅ 使用 Docker 内置的 tini init 系统

### 问题2：服务启动但立即退出

**症状**：
- 容器重启后，启动脚本执行
- 但服务启动后立即退出
- 端口8000未监听

**原因**：
- uvicorn 启动方式问题（已修复）
- 需要重建容器以应用修复

**解决方案**：
```bash
# 重建容器以应用最新修复
docker-compose stop rqa2025-app
docker-compose build rqa2025-app
docker-compose up -d rqa2025-app
```

### 问题2：启动脚本执行失败

**症状**：
- 容器重启，但启动脚本报错
- 容器状态为 "Exited"

**检查方法**：
```bash
# 查看容器退出代码
docker inspect rqa2025-rqa2025-app-1 --format='{{.State.ExitCode}}'
# 0 表示正常退出，非0 表示异常退出

# 查看完整日志
docker logs rqa2025-rqa2025-app-1
```

## 最佳实践

### 1. 确保启动脚本健壮

启动脚本应该：
- ✅ 捕获所有异常
- ✅ 记录详细的错误信息
- ✅ 在失败时退出并返回非0代码

### 2. 配置健康检查

已配置健康检查：
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 3. 监控容器状态

```bash
# 查看容器状态
docker ps -a | grep rqa2025-app

# 查看重启次数
docker inspect rqa2025-rqa2025-app-1 --format='{{.RestartCount}}'

# 查看最后启动时间
docker inspect rqa2025-rqa2025-app-1 --format='{{.State.StartedAt}}'
```

## 总结

✅ **自动启动已配置**：
- 重启策略：`unless-stopped`
- 启动命令：`python scripts/start_api_server.py`
- 容器重启时会自动执行启动脚本

⚠️ **需要注意**：
- 如果服务启动后立即退出，需要重建容器以应用最新修复
- 确保启动脚本能正确处理错误
- 定期检查容器日志以确认服务正常运行

## 相关文档

- [容器后端故障排除指南](./container_backend_troubleshooting.md)
- [容器后端最终诊断](./container_backend_final_diagnosis.md)
- [后端服务启动指南](./backend_startup_guide.md)

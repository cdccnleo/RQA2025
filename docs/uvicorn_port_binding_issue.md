# Uvicorn 端口绑定问题诊断

## 问题现象

1. ✅ uvicorn.run() 执行成功
2. ✅ 服务器进程启动：`INFO: Started server process [7]`
3. ✅ 应用启动完成：`INFO: Application startup complete.`
4. ❌ 但没有看到 "Uvicorn running on http://0.0.0.0:8000"
5. ❌ 端口8000未监听
6. ❌ 服务无法访问

## 可能的原因

### 1. Uvicorn 版本问题

当前版本：`uvicorn 0.39.0`

某些版本的 uvicorn 可能不会输出 "Uvicorn running" 消息，或者消息格式不同。

### 2. 日志级别问题

`log_level="info"` 可能不会输出 "Uvicorn running" 消息。可能需要使用 `log_level="debug"`。

### 3. 服务器启动后立即退出

虽然日志显示 "Application startup complete"，但服务器可能因为某种原因立即退出。

### 4. 端口绑定失败

服务器可能无法绑定到端口（权限、端口占用等），但错误被忽略了。

## 诊断步骤

### 步骤1：检查 uvicorn 日志输出

查看是否有 "Uvicorn running" 或类似消息：

```bash
docker logs rqa2025-rqa2025-app-1 | grep -i "uvicorn\|running\|listening"
```

### 步骤2：使用 debug 日志级别

修改启动脚本，使用 `log_level="debug"` 查看更详细的日志。

### 步骤3：检查进程状态

检查 uvicorn 进程是否在运行：

```bash
docker exec rqa2025-rqa2025-app-1 sh -c "ps aux | grep python"
```

### 步骤4：手动测试启动

在容器中手动运行启动脚本，观察完整输出：

```bash
docker exec -it rqa2025-rqa2025-app-1 python scripts/start_api_server.py
```

## 下一步建议

1. **使用 debug 日志级别**：查看更详细的 uvicorn 日志
2. **检查进程状态**：确认 uvicorn 进程是否在运行
3. **手动测试**：在容器中手动运行启动脚本
4. **检查端口绑定**：确认是否有端口绑定错误

## 相关文件

- `scripts/start_api_server.py` - 启动脚本
- `src/gateway/web/api.py` - FastAPI 应用
- `docs/uvicorn_startup_issue.md` - Uvicorn 启动问题文档

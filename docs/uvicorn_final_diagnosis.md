# Uvicorn 启动问题最终诊断

## 问题总结

### 当前状态

1. ✅ **启动脚本执行正常**：可以执行到 uvicorn 启动
2. ✅ **应用创建成功**：路由数 177
3. ✅ **诊断功能正常**：时间戳和日志输出正常
4. ❌ **Uvicorn 服务器未监听端口**：端口8000未监听

### 诊断发现

从日志可以看到：

1. **使用 asyncio.run(server.serve()) 方式**：
   - 执行到 `await server.serve()`
   - 但没有看到 "Uvicorn running on http://0.0.0.0:8000"
   - 端口未监听

2. **使用 uvicorn.run() 方式**：
   - 需要验证是否执行

3. **关键日志**：
   ```
   INFO:     Started server process [7]
   INFO:     Application startup complete.
   ```
   - 说明 uvicorn 启动了，但可能立即退出

## 可能的原因

1. **事件循环问题**：`asyncio.run()` 可能无法正确运行服务器
2. **服务器立即退出**：服务器启动后可能因为某种原因立即退出
3. **端口绑定失败**：服务器可能无法绑定到端口（权限、端口占用等）
4. **容器环境问题**：容器环境中的信号处理或进程管理可能有问题

## 已尝试的修复

1. ✅ 使用 `asyncio.run(server.serve())`
2. ✅ 使用 `await server.serve()` 在异步函数中
3. ✅ 添加详细诊断日志
4. ✅ 切换到 `uvicorn.run()` 方式

## 下一步建议

### 方案1：检查 uvicorn.run() 执行情况

查看最新日志，确认 `uvicorn.run()` 是否执行，以及是否有错误。

### 方案2：使用 uvicorn CLI 方式

直接在 Dockerfile 或启动命令中使用 uvicorn CLI：

```dockerfile
CMD ["uvicorn", "src.gateway.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 方案3：检查应用生命周期

检查 FastAPI 应用的 `lifespan` 是否有阻塞操作导致服务器无法正常启动。

### 方案4：简化启动方式

创建一个最简单的启动脚本，只启动基本的 FastAPI 应用，排除其他干扰因素。

## 相关文件

- `scripts/start_api_server.py` - 当前启动脚本
- `src/gateway/web/api.py` - FastAPI 应用
- `src/gateway/web/app_factory.py` - 应用工厂
- `docs/uvicorn_startup_issue.md` - Uvicorn 启动问题文档

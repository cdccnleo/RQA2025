# Uvicorn 启动问题诊断

## 当前状态

### ✅ 已解决的问题

1. **UnboundLocalError**：已修复
2. **启动脚本执行**：可以正常执行到 `await server.serve()`
3. **应用创建**：成功创建，路由数 177
4. **诊断功能**：时间戳和详细日志正常工作

### ⚠️ 当前问题

**Uvicorn 服务器启动但端口未监听**：
- 日志显示：`🔄 执行: await server.serve()`
- 但没有看到 "Uvicorn running on http://0.0.0.0:8000"
- 端口8000未监听
- 服务无法访问

## 诊断信息

从日志可以看到：

```
[16:52:32] [168.17s] 🔄 执行: asyncio.run(run_server())
[16:52:32] [168.17s] 🔄 执行: await server.serve()
```

**关键发现**：
- 执行到了 `await server.serve()`
- 但之后没有更多日志输出
- 说明 `server.serve()` 可能卡住或没有正确启动

## 可能的原因

1. **事件循环问题**：`asyncio.run()` 可能无法正确运行 `server.serve()`
2. **服务器配置问题**：`uvicorn.Server` 配置可能有问题
3. **阻塞操作**：`server.serve()` 内部可能有阻塞操作
4. **信号处理问题**：容器环境中的信号处理可能有问题

## 已尝试的修复

1. ✅ 使用 `asyncio.run(run_server())` 包装 `await server.serve()`
2. ✅ 添加详细诊断日志
3. ✅ 添加异常捕获和降级方案

## 下一步建议

### 方案1：使用 uvicorn.run() 作为主要方式

直接使用 `uvicorn.run()`，这是最简单可靠的方式：

```python
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
```

### 方案2：检查是否有事件循环冲突

如果 `asyncio.run()` 无法工作，可能需要：
- 检查是否已有运行中的事件循环
- 使用 `nest_asyncio` 处理嵌套事件循环

### 方案3：使用 uvicorn CLI

直接使用 uvicorn 命令行工具：

```python
import subprocess
subprocess.run(["uvicorn", "src.gateway.web.api:app", "--host", "0.0.0.0", "--port", "8000"])
```

## 相关文件

- `scripts/start_api_server.py` - 当前启动脚本
- `docs/diagnosis_guide.md` - 诊断指南
- `docs/diagnosis_fix_summary.md` - 诊断修复总结

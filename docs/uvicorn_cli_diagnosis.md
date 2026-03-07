# Uvicorn CLI 方式诊断结果

## 当前状态

### ✅ 已完成的修改

1. **切换到 uvicorn CLI 方式**：
   - 修改 `docker-compose.yml` 使用 `command: ["uvicorn", "src.gateway.web.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]`
   - 这是官方推荐的启动方式

### ⚠️ 当前问题

1. **服务器启动但端口未监听**：
   - 日志显示：`INFO: Started server process [7]`
   - 日志显示：`INFO: Application startup complete.`
   - 但没有看到 "Uvicorn running on http://0.0.0.0:8000"
   - 端口8000未监听（连接测试返回 111）
   - curl 返回 "Empty reply from server"

2. **关键发现**：
   - 使用 uvicorn CLI 后，curl 返回 "Empty reply from server" 而不是 "Connection refused"
   - 这说明连接建立了但立即关闭
   - 可能是服务器启动后立即退出

## 可能的原因

根据搜索结果，可能的原因包括：

1. **Uvicorn 没有正确绑定到端口**
   - 虽然设置了 `--host 0.0.0.0`，但可能没有生效
   - 需要检查 uvicorn 的完整启动日志

2. **应用启动后立即退出**
   - lifespan 中的某些操作可能导致服务器退出
   - 需要检查是否有异常或退出信号

3. **端口冲突或绑定失败**
   - 端口可能被占用或无法绑定
   - 需要检查端口状态

4. **Docker 网络配置问题**
   - 端口映射可能有问题
   - 需要检查 docker-compose.yml 配置

## 下一步诊断

1. **检查 uvicorn 完整启动日志**：
   - 查看是否有 "Uvicorn running" 消息
   - 查看是否有错误或警告

2. **检查端口绑定**：
   - 使用 `ss` 或 `lsof` 检查端口状态
   - 检查是否有端口冲突

3. **简化测试**：
   - 创建一个最简单的 FastAPI 应用测试
   - 排除应用代码的干扰

4. **检查 lifespan**：
   - 确认 lifespan 中的操作不会导致服务器退出
   - 检查是否有阻塞操作

## 相关文件

- `docker-compose.yml` - 已修改为使用 uvicorn CLI
- `src/gateway/web/api.py` - FastAPI 应用和 lifespan
- `docs/uvicorn_cli_alternative.md` - CLI 备选方案文档

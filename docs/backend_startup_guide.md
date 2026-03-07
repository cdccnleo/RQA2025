# 后端服务启动指南

## 概述

RQA2025 后端服务基于 FastAPI 框架，使用 uvicorn 作为 ASGI 服务器。服务启动时会自动初始化数据采集调度器等后台任务。

## 启动顺序说明

后端服务的启动顺序已确保：

1. **后端服务（FastAPI/uvicorn）启动**
   - FastAPI 应用创建和配置（使用 `lifespan` 上下文管理器）
   - 路由注册
   - CORS 中间件配置
   - uvicorn 服务器启动

2. **后端服务完全就绪**
   - 等待服务器完全启动（1秒延迟）
   - 自动验证服务器健康检查端点
   - 服务器可以接受HTTP请求

3. **数据采集调度器启动**
   - 在后台任务中启动数据采集调度器
   - 按照数据源配置的 rate_limit 进行自动调度

## 启动方式

### 方式1：使用统一启动脚本（推荐）⭐

```bash
python scripts/start_server.py
```

**特点**：
- ✅ 自动检查端口8000是否被占用
- ✅ 启动后自动验证服务是否就绪
- ✅ 提供清晰的错误信息和状态反馈
- ✅ 统一的启动入口，减少混乱

### 方式2：使用标准启动脚本

```bash
python scripts/start_api_server.py
```

**特点**：
- ✅ 包含完整的错误处理
- ✅ 启动后自动验证服务
- ✅ 适合生产环境使用

### 方式3：使用简化启动脚本

```bash
python scripts/start_backend.py
```

**特点**：
- ✅ 简化的启动流程
- ✅ 启动后自动验证服务
- ✅ 适合快速启动

### 方式4：直接使用 uvicorn

```bash
uvicorn src.gateway.web.app_factory:create_app --host 0.0.0.0 --port 8000
```

**注意**：此方式不会自动验证服务，建议使用启动脚本。

## 检查后端服务状态

### 方式1：使用状态检查脚本

```bash
python scripts/check_backend_status.py
```

### 方式2：手动检查

```bash
# 检查端口是否开放
netstat -an | findstr :8000  # Windows
netstat -an | grep :8000      # Linux/Mac

# 检查健康检查端点
curl http://localhost:8000/health
```

## 服务地址

- **后端服务地址**: http://localhost:8000
- **API文档地址**: http://localhost:8000/docs
- **健康检查地址**: http://localhost:8000/health
- **系统状态地址**: http://localhost:8000/api/v1/status

## 故障排除

### 问题1：端口8000已被占用

**症状**：
- 启动时提示 "端口8000已被占用"
- 或启动失败，提示 "Address already in use"

**解决方案**：
1. 检查是否有其他服务在使用端口8000：
   ```bash
   # Windows
   netstat -ano | findstr :8000
   
   # Linux/Mac
   lsof -i :8000
   ```
2. 停止占用端口的进程
3. 或使用其他端口（设置环境变量 `PORT=8001`）

### 问题2：服务启动但无法连接

**症状**：
- 端口8000已开放，但HTTP请求无响应
- 前端显示 "无法连接到后端服务"

**可能原因**：
1. 服务启动但未完全初始化
2. 使用了已弃用的启动事件（`@app.on_event("startup")`）
3. 应用创建失败但未报错

**解决方案**：
1. 检查服务日志，查看是否有错误信息
2. 使用统一启动脚本（`scripts/start_server.py`），它会自动验证服务
3. 手动访问健康检查端点：http://localhost:8000/health
4. 检查 FastAPI 应用是否正确创建（查看日志中的路由数量）

### 问题3：数据采集调度器未启动

**症状**：
- 后端服务正常，但数据采集任务未执行

**解决方案**：
1. 检查日志中是否有 "数据采集调度器后台任务已启动" 消息
2. 检查是否有导入错误（查看日志中的警告信息）
3. 手动验证调度器状态（如果提供了状态API）

### 问题4：启动脚本导入错误

**症状**：
- 提示 "ImportError" 或 "ModuleNotFoundError"

**解决方案**：
1. 确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 检查 Python 路径是否正确
3. 确保在项目根目录运行启动脚本

## 技术细节

### 启动事件管理

后端服务使用 FastAPI 的 `lifespan` 上下文管理器（替代已弃用的 `@app.on_event("startup")`）来管理应用生命周期：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑
    yield  # 应用运行
    # 关闭逻辑
```

### 启动验证

服务启动后会自动验证：
1. 等待服务器完全启动（1秒延迟）
2. 尝试验证健康检查端点（最多10次，每次间隔0.5秒）
3. 验证成功或超时后继续启动流程

### 架构设计

- **网关层**：API路由和请求处理
- **核心服务层**：业务流程编排（数据采集调度器）
- **数据管理层**：数据采集、存储、处理

启动顺序符合分层架构原则：网关层 → 核心服务层 → 业务流程编排

## 注意事项

1. ✅ 确保端口8000未被占用
2. ✅ 数据采集调度器会在后端服务启动后自动启动
3. ✅ 如果启动失败，检查日志输出中的错误信息
4. ✅ 使用统一启动脚本（`scripts/start_server.py`）获得最佳体验
5. ✅ 启动后会自动验证服务，无需手动检查

## 相关文档

- [数据采集调度器重构文档](../refactoring/service_scheduler_refactoring.md)
- [API文档](http://localhost:8000/docs)（服务启动后访问）

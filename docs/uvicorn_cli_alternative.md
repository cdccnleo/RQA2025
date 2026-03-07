# Uvicorn CLI 备选方案

## 问题总结

当前使用 `uvicorn.run()` 方式启动，但服务器启动后端口未监听。

## 备选方案：使用 Uvicorn CLI

### 方案1：修改 Dockerfile CMD

直接使用 uvicorn CLI 命令：

```dockerfile
CMD ["uvicorn", "src.gateway.web.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
```

### 方案2：修改 docker-compose.yml command

在 docker-compose.yml 中直接使用 uvicorn CLI：

```yaml
command: ["uvicorn", "src.gateway.web.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
```

### 方案3：创建简单的启动脚本

创建一个最简单的启动脚本，只启动 uvicorn：

```python
#!/usr/bin/env python3
import uvicorn
from src.gateway.web.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

## 优势

1. **简单直接**：不需要复杂的启动逻辑
2. **可靠性高**：uvicorn CLI 是官方推荐的方式
3. **易于调试**：可以直接看到 uvicorn 的完整输出

## 实施步骤

1. 修改 `docker-compose.yml` 中的 `command`
2. 或者修改 `Dockerfile` 中的 `CMD`
3. 重建容器并测试

## 相关文件

- `Dockerfile` - 容器构建配置
- `docker-compose.yml` - 容器编排配置
- `scripts/start_api_server.py` - 当前启动脚本

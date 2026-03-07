# Docker API 连接问题修复说明

## 问题描述

当应用在 Docker 容器内运行时，持续输出以下错误日志：

```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; 
check if the path is correct and if the daemon is running: 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

## 问题原因

1. **容器内无法访问宿主机 Docker socket**：容器内部默认无法访问宿主机的 Docker daemon
2. **立即初始化问题**：`DockerManager` 类在 `__init__` 时立即调用 `docker.from_env()`，导致在容器启动时就尝试连接 Docker API
3. **持续重试**：如果连接失败，某些代码可能会持续重试，导致日志不断输出

## 解决方案

已修复 `src/strategy/backtest/microservice_architecture.py` 中的 `DockerManager` 类：

### 主要改进

1. **延迟初始化**：Docker 客户端不再在 `__init__` 时立即连接，而是在实际需要使用 Docker 功能时才连接
2. **环境检测**：自动检测是否在容器内运行，以及是否有 Docker socket 挂载
3. **优雅降级**：如果 Docker 不可用，记录一次警告后禁用 Docker 功能，避免持续报错
4. **错误处理**：所有 Docker 操作都先检查客户端是否可用，不可用时优雅跳过

### 修复后的行为

- ✅ 容器启动时不再尝试连接 Docker API
- ✅ 只在需要 Docker 功能时才尝试连接
- ✅ 连接失败时只记录一次警告，不会持续输出错误
- ✅ 如果不需要 Docker 功能，可以安全忽略警告

## 使用说明

### 如果不需要 Docker 功能

如果应用不需要在运行时管理 Docker 容器，可以安全忽略相关警告。修复后的代码会自动禁用 Docker 功能，不会影响应用正常运行。

### 如果需要 Docker 功能

如果需要在容器内使用 Docker 功能（例如 Docker-in-Docker），需要：

1. **挂载 Docker socket**（Linux）：
   ```yaml
   volumes:
     - /var/run/docker.sock:/var/run/docker.sock
   ```

2. **挂载 Docker socket**（Windows Docker Desktop）：
   ```yaml
   volumes:
     - //./pipe/dockerDesktopLinuxEngine://./pipe/dockerDesktopLinuxEngine
   ```

3. **使用 Docker-in-Docker**：
   在 docker-compose.yml 中使用 `docker:dind` 服务

## 相关文件

- `src/strategy/backtest/microservice_architecture.py` - 已修复的 DockerManager 类

## 测试建议

1. 在容器内运行应用，确认不再有持续的错误日志输出
2. 如果应用需要 Docker 功能，测试挂载 Docker socket 后功能是否正常
3. 验证应用的其他功能不受影响

## 注意事项

- 此修复不会影响应用的核心功能
- Docker 功能是可选的，如果不可用会自动禁用
- 所有 Docker 相关操作都有错误处理，不会导致应用崩溃

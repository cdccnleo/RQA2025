# RQA2025 生产部署脚本使用说明

本文档说明如何使用RQA2025生产部署的PowerShell脚本，完成基础设施准备和基础服务部署。

## 📋 脚本概览

### 主要脚本文件

1. **`deploy_phase1_infrastructure.ps1`** - 主协调脚本
   - 统一入口点，协调所有Phase 1部署步骤
   - 支持灵活的参数配置和跳过选项
   - 生成综合部署报告

2. **`prepare_server_environment.ps1`** - 服务器环境准备脚本
   - 检查硬件配置（CPU、内存、磁盘空间）
   - 验证操作系统版本和配置
   - 检查网络配置和安全设置
   - 验证Docker环境

3. **`prepare_docker_environment.ps1`** - Docker环境准备脚本
   - 检查系统要求（虚拟化、WSL2）
   - 安装和配置Docker Desktop
   - 验证Docker和Docker Compose功能

4. **`deploy_basic_services.ps1`** - 基础服务部署脚本
   - 部署Redis主从配置
   - 部署PostgreSQL主从配置
   - 配置HAProxy负载均衡器
   - 部署监控服务（Prometheus、Grafana）

## 🚀 快速开始

### 前置要求

- Windows 10/11 操作系统
- PowerShell 5.1+ 或 PowerShell Core 7+
- 管理员权限
- Docker Desktop 已安装并运行

### 完整部署（推荐）

```powershell
# 以管理员身份运行PowerShell
.\deploy_phase1_infrastructure.ps1
```

这将执行完整的Phase 1部署，包括：
- 服务器环境准备
- 基础服务部署
- 生成综合报告

### 分步部署

#### 步骤1：仅准备服务器环境

```powershell
.\deploy_phase1_infrastructure.ps1 -SkipBasicServices
```

#### 步骤2：仅部署基础服务

```powershell
.\deploy_phase1_infrastructure.ps1 -SkipEnvironmentPrep
```

#### 步骤3：仅生成报告

```powershell
.\deploy_phase1_infrastructure.ps1 -GenerateReportOnly
```

## ⚙️ 参数配置

### 主协调脚本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-Environment` | string | "production" | 部署环境名称 |
| `-NetworkSubnet` | string | "172.20.0.0/16" | Docker网络子网 |
| `-SkipEnvironmentPrep` | switch | false | 跳过服务器环境准备 |
| `-SkipBasicServices` | switch | false | 跳过基础服务部署 |
| `-GenerateReportOnly` | switch | false | 仅生成报告，不执行部署 |

### 环境准备跳过选项

| 参数 | 说明 |
|------|------|
| `-SkipHardware` | 跳过硬件检查 |
| `-SkipOS` | 跳过操作系统检查 |
| `-SkipNetwork` | 跳过网络配置检查 |
| `-SkipSecurity` | 跳过安全配置检查 |
| `-SkipDocker` | 跳过Docker环境检查 |

### 基础服务跳过选项

| 参数 | 说明 |
|------|------|
| `-SkipRedis` | 跳过Redis部署 |
| `-SkipPostgreSQL` | 跳过PostgreSQL部署 |
| `-SkipHAProxy` | 跳过HAProxy部署 |
| `-SkipMonitoring` | 跳过监控服务部署 |
| `-ForceRecreate` | 强制重新创建现有资源 |

## 🔧 使用示例

### 示例1：开发环境部署

```powershell
.\deploy_phase1_infrastructure.ps1 -Environment "development" -SkipMonitoring
```

### 示例2：跳过特定检查

```powershell
.\deploy_phase1_infrastructure.ps1 -SkipHardware -SkipSecurity
```

### 示例3：仅部署数据库服务

```powershell
.\deploy_phase1_infrastructure.ps1 -SkipEnvironmentPrep -SkipHAProxy -SkipMonitoring
```

### 示例4：强制重新创建

```powershell
.\deploy_phase1_infrastructure.ps1 -ForceRecreate
```

## 📊 部署报告

### 报告位置

部署完成后，报告文件保存在：
```
deploy\reports\
```

### 报告类型

1. **服务器环境准备报告**
   - 文件名：`server_environment_report_YYYYMMDD_HHMMSS.json`
   - 内容：硬件、OS、网络、安全、Docker状态

2. **基础服务部署报告**
   - 文件名：`basic_services_deployment_report_YYYYMMDD_HHMMSS.json`
   - 内容：Redis、PostgreSQL、HAProxy、监控服务状态

3. **Phase 1综合报告**
   - 文件名：`phase1_infrastructure_report_YYYYMMDD_HHMMSS.json`
   - 内容：完整的Phase 1部署状态和健康检查

### 报告格式

所有报告都采用JSON格式，包含：
- 时间戳
- 执行摘要
- 检查点状态
- 服务运行状态
- 整体健康状态

## 🐳 部署的服务

### Redis主从配置

- **主节点**：172.20.1.10:6379
- **从节点**：172.20.1.11:6380
- **功能**：缓存、会话存储、任务队列
- **持久化**：AOF模式，支持故障恢复

### PostgreSQL主从配置

- **主节点**：172.20.2.10:5432
- **从节点**：172.20.2.11:5433
- **功能**：主数据库，读写分离
- **复制**：流式复制，支持故障转移

### HAProxy负载均衡器

- **地址**：172.20.3.10
- **端口**：80（服务）、8080（统计）
- **功能**：负载均衡、健康检查、故障转移
- **算法**：轮询（Round Robin）

### 监控服务

- **Prometheus**：172.20.4.10:9090
  - 指标收集、存储、查询
  - 支持告警规则配置
  
- **Grafana**：172.20.4.11:3000
  - 数据可视化、仪表板
  - 支持多种数据源

## 🔍 故障排除

### 常见问题

1. **权限不足**
   ```
   错误：This script requires administrator privileges
   解决：以管理员身份运行PowerShell
   ```

2. **Docker未运行**
   ```
   错误：Docker service is not running
   解决：启动Docker Desktop
   ```

3. **端口冲突**
   ```
   错误：Port is already in use
   解决：检查端口占用，修改配置或停止冲突服务
   ```

4. **网络创建失败**
   ```
   错误：Failed to create network
   解决：检查Docker网络配置，使用-ForceRecreate参数
   ```

### 日志查看

- 脚本执行日志：PowerShell控制台输出
- Docker容器日志：`docker logs <container_name>`
- 服务状态：`docker ps`

### 清理和重置

```powershell
# 停止所有容器
docker stop $(docker ps -q)

# 删除所有容器
docker rm $(docker ps -aq)

# 删除所有卷
docker volume prune -f

# 删除网络
docker network rm rqa2025-network
```

## 📚 下一步

完成Phase 1基础设施准备后，可以继续：

1. **Phase 2：核心服务部署**
   - RQA2025 API服务
   - 推理引擎服务
   - 任务调度服务

2. **Phase 3：应用配置**
   - 环境变量配置
   - 数据库初始化
   - 服务连接配置

3. **Phase 4：测试验证**
   - 功能测试
   - 性能测试
   - 故障恢复测试

## 🤝 技术支持

如遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查部署报告中的错误信息
3. 查看Docker容器日志
4. 联系系统管理员或开发团队

---

**注意**：这些脚本仅用于生产环境部署，请在生产环境中谨慎使用，并确保已备份重要数据。

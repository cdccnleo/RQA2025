# RQA2025 生产环境部署指南

## 概述

本文档提供了RQA2025项目生产环境部署的完整指南，包括部署工具、检查清单和执行步骤。

## 部署架构

RQA2025采用微服务架构，包含以下核心服务：

- **PostgreSQL**: 主数据库服务
- **Redis**: 缓存和会话存储
- **API服务**: 核心业务API
- **推理服务**: 模型推理和预测
- **Prometheus**: 监控指标收集
- **Grafana**: 监控数据可视化
- **Elasticsearch**: 日志和搜索服务
- **Kibana**: 日志分析界面

## 部署工具

### 1. 生产环境部署验证脚本

**文件**: `validate_production_deployment.py`

**用途**: 验证生产环境的各个组件是否正常工作

**功能**:
- 验证PostgreSQL数据库连接和状态
- 验证Redis缓存服务
- 验证API服务健康状态
- 验证监控系统
- 检查系统资源使用情况

**使用方法**:
```bash
cd deploy
python validate_production_deployment.py
```

### 2. 生产环境部署执行脚本

**文件**: `scripts/deploy_production.py`

**用途**: 自动化部署RQA2025到生产环境

**功能**:
- 检查部署前置条件
- 部署Docker服务
- 等待服务就绪
- 运行健康检查
- 生成部署报告

**使用方法**:
```bash
cd deploy
python scripts/deploy_production.py config/deployment_config.json
```

### 3. 部署检查清单执行脚本

**文件**: `scripts/run_deployment_checklist.py`

**用途**: 自动化执行生产环境部署检查清单

**功能**:
- 基础设施检查（CPU、内存、磁盘、网络）
- 数据库服务检查
- 监控系统检查
- API服务检查
- 生成改进建议

**使用方法**:
```bash
cd deploy
python scripts/run_deployment_checklist.py config/deployment_config.json
```

## 配置文件

### 部署配置文件

**文件**: `config/deployment_config.json`

**用途**: 配置生产环境的服务参数

**主要配置项**:
- 数据库连接参数
- Redis连接参数
- API服务配置
- 监控服务配置
- 安全配置
- 备份策略
- 性能参数

**配置示例**:
```json
{
  "services": {
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "rqa2025",
      "user": "postgres",
      "password": "your_secure_password"
    }
  }
}
```

## 部署流程

### 阶段1: 部署前准备

1. **环境检查**
   ```bash
   # 运行部署检查清单
   python scripts/run_deployment_checklist.py config/deployment_config.json
   ```

2. **配置文件验证**
   - 检查 `deployment_config.json` 配置
   - 验证密码和连接参数
   - 确认端口配置

3. **资源准备**
   - 确保有足够的磁盘空间（至少50GB）
   - 检查网络连通性
   - 验证Docker环境

### 阶段2: 执行部署

1. **运行部署脚本**
   ```bash
   # 执行生产环境部署
   python scripts/deploy_production.py config/deployment_config.json
   ```

2. **监控部署进度**
   - 观察服务启动状态
   - 检查端口监听情况
   - 验证服务健康状态

3. **等待服务就绪**
   - 数据库服务启动（约30秒）
   - 缓存服务启动（约10秒）
   - API服务启动（约20秒）
   - 监控服务启动（约15秒）

### 阶段3: 部署后验证

1. **运行验证脚本**
   ```bash
   # 验证部署结果
   python validate_production_deployment.py
   ```

2. **功能测试**
   - 测试API端点
   - 验证数据库操作
   - 检查监控数据

3. **性能验证**
   - 检查响应时间
   - 验证吞吐量
   - 监控资源使用

## 检查清单

### 基础设施检查

- [ ] CPU使用率 < 70%
- [ ] 内存使用率 < 80%
- [ ] 磁盘使用率 < 90%
- [ ] 网络连通性正常
- [ ] 防火墙配置正确

### 数据库检查

- [ ] PostgreSQL服务运行正常
- [ ] 数据库连接成功
- [ ] 表结构完整
- [ ] 连接池配置合理

### 缓存服务检查

- [ ] Redis服务运行正常
- [ ] 基本操作测试通过
- [ ] 内存使用合理
- [ ] 连接数配置正确

### 监控系统检查

- [ ] Prometheus服务正常
- [ ] Grafana服务正常
- [ ] 监控指标收集正常
- [ ] 告警规则配置正确

### API服务检查

- [ ] API服务健康状态正常
- [ ] 核心端点响应正常
- [ ] 响应时间满足要求
- [ ] 错误处理机制正常

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口占用情况
   - 验证配置文件语法
   - 查看Docker日志

2. **数据库连接失败**
   - 检查PostgreSQL服务状态
   - 验证连接参数
   - 检查防火墙设置

3. **监控服务异常**
   - 检查Prometheus配置
   - 验证目标服务可达性
   - 查看监控日志

### 日志文件

- **部署日志**: `production_deployment.log`
- **验证日志**: `deployment_validation.log`
- **检查清单日志**: `deployment_checklist.log`
- **Docker日志**: `docker-compose logs`

### 回滚步骤

1. **停止服务**
   ```bash
   cd deploy
   docker-compose down
   ```

2. **恢复数据**
   - 从备份恢复数据库
   - 恢复配置文件

3. **重新部署**
   ```bash
   python scripts/deploy_production.py config/deployment_config.json
   ```

## 监控和维护

### 日常监控

- **系统资源**: CPU、内存、磁盘使用率
- **服务状态**: 各服务的健康状态
- **性能指标**: 响应时间、吞吐量
- **错误日志**: 异常和错误信息

### 定期维护

- **日志轮转**: 定期清理和归档日志
- **数据备份**: 按计划执行数据备份
- **安全更新**: 及时应用安全补丁
- **性能优化**: 根据监控数据优化配置

### 告警配置

- **资源告警**: CPU > 80%, 内存 > 85%, 磁盘 > 90%
- **服务告警**: 服务不可用、健康检查失败
- **性能告警**: 响应时间过长、错误率过高

## 安全考虑

### 网络安全

- 使用防火墙限制端口访问
- 配置VPN访问生产环境
- 实施网络分段和隔离

### 访问控制

- 使用SSH密钥认证
- 禁用密码认证
- 限制root用户访问

### 数据安全

- 启用SSL/TLS加密
- 实施数据备份策略
- 定期安全审计

## 联系信息

- **技术支持**: support@rqa2025.com
- **运维团队**: ops@rqa2025.com
- **紧急联系**: emergency@rqa2025.com

---

**文档版本**: 1.0  
**最后更新**: 2025-01-27  
**下次评审**: 2025-02-10

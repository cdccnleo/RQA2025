# RQA2025 部署指南

## 概述

RQA2025 量化交易系统部署指南。本文档提供完整的生产环境部署说明。

## 部署前准备

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 或 CentOS 7+)
- **CPU**: 4核心以上
- **内存**: 8GB以上
- **磁盘**: 50GB以上可用空间
- **网络**: 稳定的互联网连接

### 依赖软件

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Nginx (可选，用于反向代理)

### 部署前检查

运行部署前检查脚本：

```bash
./scripts/pre_deploy_check.sh
```

## 生产就绪评估结果

基于自动化评估，系统当前状态：

- **总体评分**: 66.3/100
- **就绪状态**: not_ready
- **风险等级**: high

### 类别评分

| 类别 | 评分 | 状态 |
|------|------|------|
| 功能完整性 | 80.0 | ✅ |
| 性能就绪 | 25.0 | ❌ |
| 稳定性 | 78.0 | ❌ |
| 安全性 | 61.2 | ❌ |
| 可运维性 | 97.1 | ✅ |
| 文档完整性 | 90.0 | ✅ |

## 部署步骤

### 1. 环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python
sudo apt install python3.9 python3.9-venv python3-pip

# 安装PostgreSQL
sudo apt install postgresql postgresql-contrib

# 安装Redis
sudo apt install redis-server

# 安装Nginx (可选)
sudo apt install nginx
```

### 2. 代码部署

```bash
# 克隆代码
git clone <repository-url> rqa2025
cd rqa2025

# 运行部署脚本
./scripts/deploy.sh
```

### 3. 配置设置

复制并修改配置文件：

```bash
cp deployment/templates/config_production.json config/production.json
# 编辑配置文件，设置数据库连接、API密钥等
```

### 4. 环境变量设置

```bash
# 创建环境变量文件
sudo tee /etc/rqa2025.env > /dev/null <<EOF
RQA2025_ENV=production
RQA2025_DATABASE_URL=postgresql://user:pass@host:port/db
RQA2025_REDIS_URL=redis://host:port
RQA2025_JWT_SECRET=your-secret-key
EOF
```

### 5. 服务启动

```bash
# 启动服务
sudo systemctl start rqa2025
sudo systemctl enable rqa2025

# 检查状态
sudo systemctl status rqa2025
```

## 监控和维护

### 健康检查

```bash
# API健康检查
curl http://localhost:8000/health

# 指标监控
curl http://localhost:8000/metrics
```

### 日志查看

```bash
# 应用日志
tail -f /var/log/rqa2025/app.log

# 系统日志
sudo journalctl -u rqa2025 -f
```

### 备份策略

- **数据库备份**: 每日全量备份，每小时增量备份
- **配置文件备份**: 每次部署时自动备份
- **日志备份**: 按月轮转，保留6个月

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查配置文件是否正确
   - 验证数据库连接
   - 查看详细错误日志

2. **性能问题**
   - 检查系统资源使用情况
   - 调整连接池大小
   - 优化查询性能

3. **内存泄漏**
   - 监控内存使用趋势
   - 重启服务释放内存
   - 检查代码中的资源释放

## 安全注意事项

- 定期更新系统补丁
- 使用强密码和API密钥
- 限制网络访问权限
- 启用审计日志
- 定期安全扫描

## 联系支持

如遇部署或运行问题，请联系技术支持团队。

---
*本文档由RQA2025部署准备系统自动生成*

# RQA2025 安全配置指南

## 概述

本文档指导如何安全地配置RQA2025量化交易系统的敏感信息。

## 安全审计结果

**审计日期**: 2025年  
**风险等级**: 严重 - 发现4个Critical级别漏洞  
**主要问题**: 硬编码密码、默认凭据、不安全配置

### 发现的问题

| 风险等级 | 数量 | 说明 |
|---------|------|------|
| 🔴 **严重 (Critical)** | 4 | 硬编码密码、密钥泄露 |
| 🟠 **高危 (High)** | 5 | 默认凭据、不安全配置 |
| 🟡 **中危 (Medium)** | 4 | 安全头部缺失、日志问题 |
| 🟢 **低危 (Low)** | 3 | 最佳实践问题 |

## 快速开始

### 1. 创建环境变量文件

```bash
# 复制模板文件
cp .env.example .env

# 编辑 .env 文件，填写实际值
# 使用强密码生成器生成密码
```

### 2. 生成强密码

```bash
# 使用Python生成强密码
python -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))"
```

### 3. 配置Docker Secrets（生产环境推荐）

```bash
# 创建secrets目录
mkdir -p secrets

# 生成密码并保存到文件
echo "$(openssl rand -base64 32)" > secrets/db_password.txt
echo "$(openssl rand -base64 32)" > secrets/redis_password.txt

# 设置权限
chmod 600 secrets/*.txt
```

## 密码更换清单

### 必须更换的密码

- [ ] 数据库密码 (DB_PASSWORD)
- [ ] Redis密码 (REDIS_PASSWORD)
- [ ] JWT密钥 (JWT_SECRET)
- [ ] API密钥 (API_KEY)
- [ ] Grafana密码 (GRAFANA_PASSWORD)
- [ ] Elasticsearch密码 (ELASTIC_PASSWORD)

### 更换步骤

1. **生成新密码**
   ```bash
   # 生成32位随机密码
   openssl rand -base64 32
   ```

2. **更新环境变量**
   ```bash
   # 编辑 .env 文件
   nano .env
   ```

3. **重启服务**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

4. **验证连接**
   ```bash
   curl http://localhost:8000/health
   ```

## 已修复的安全问题

### Critical级别修复

1. **硬编码数据库密码** ✅
   - 文件: `config/production/database.json`
   - 修复: 使用环境变量 `${DB_PASSWORD}` 替代

2. **硬编码Redis密码** ✅
   - 文件: `config/production/redis.json`
   - 修复: 使用环境变量 `${REDIS_PASSWORD}` 替代

3. **部署脚本硬编码密码** ✅
   - 文件: `deployment/preprod/deploy.py`
   - 修复: 从环境变量读取密码

4. **Python代码硬编码凭据** ✅
   - 文件: `deploy/health_check_api.py`
   - 修复: 使用 `os.environ.get()` 读取环境变量

## 安全最佳实践

### 1. 环境变量管理

- 永远不要在代码中硬编码密码
- 使用 `.env` 文件管理敏感信息
- 将 `.env` 添加到 `.gitignore`
- 定期轮换密码（建议每90天）

### 2. 密码强度要求

- 最少32个字符
- 包含大小写字母、数字、特殊字符
- 不使用字典单词
- 每个服务使用独立密码

### 3. 生产环境额外措施

- 使用Docker Secrets管理密码
- 启用数据库SSL连接
- 配置防火墙限制访问
- 启用审计日志
- 实施网络隔离

## 安全配置模板

### Docker Compose安全配置

```yaml
version: '3.8'

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt

services:
  app:
    build: .
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
    secrets:
      - db_password
      - redis_password
    networks:
      - backend
    user: "1000:1000"  # 非root用户

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    expose:
      - "5432"
    networks:
      - backend
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass $(cat /run/secrets/redis_password)
    secrets:
      - redis_password
    expose:
      - "6379"
    networks:
      - backend

networks:
  backend:
    internal: true
```

## 安全扫描工具

### 1. 密码扫描脚本

```bash
#!/bin/bash
# scan_hardcoded_passwords.sh

echo "扫描硬编码密码..."

grep -r -n "password.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "passwd.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "secret.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "token.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .

echo "扫描完成"
```

### 2. 安全配置检查

```bash
# 检查Docker Compose配置
python scripts/security_check.py
```

## 紧急响应

### 如果发现密码泄露

1. **立即更换密码**
   ```bash
   # 生成新密码
   openssl rand -base64 32
   
   # 更新 .env 文件
   nano .env
   
   # 重启服务
   docker-compose restart
   ```

2. **检查日志**
   ```bash
   # 查看是否有异常访问
   docker-compose logs | grep -i error
   ```

3. **通知团队**
   - 发送安全事件通知
   - 更新安全事件日志
   - 审查访问记录

## 后续优化计划

### Phase 2: 高危修复 (第2周)
- [ ] 移除Docker Compose默认密码
- [ ] 启用Elasticsearch安全认证
- [ ] 限制端口暴露
- [ ] 配置网络隔离

### Phase 3: 中危修复 (第3周)
- [ ] 添加日志脱敏
- [ ] 实施API认证
- [ ] 创建非root用户

### Phase 4: 低危修复 (第4周)
- [ ] 锁定依赖版本
- [ ] 配置安全头部
- [ ] 优化健康检查

## 联系与支持

如有安全问题，请联系:
- **安全团队**: security@rqa2025.com
- **项目负责人**: [项目负责人邮箱]

---

**最后更新**: 2026-03-08  
**文档版本**: 1.0

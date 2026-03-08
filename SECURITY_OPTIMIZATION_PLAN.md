# RQA2025 安全优化计划

## 审计概述

**审计对象**: RQA2025量化交易系统  
**审计日期**: 2025年  
**执行时间**: 2026-03-08  
**风险等级**: 严重 - 发现4个Critical级别漏洞

---

## 发现的安全问题汇总

### 风险等级分布

| 风险等级 | 数量 | 状态 |
|---------|------|------|
| 🔴 **严重 (Critical)** | 4 | 待修复 |
| 🟠 **高危 (High)** | 5 | 待修复 |
| 🟡 **中危 (Medium)** | 4 | 待修复 |
| 🟢 **低危 (Low)** | 3 | 待修复 |
| **总计** | **16** | **待修复** |

---

## Phase 1: 紧急修复 (第1周) - Critical级别

### 🔴 问题1: 硬编码数据库密码
**位置**: `config/production/database.json`
**风险**: 生产环境数据库密码直接硬编码
**修复方案**:
1. 从配置文件中移除硬编码密码
2. 使用环境变量替代
3. 添加配置文件到.gitignore
4. 创建配置模板文件

### 🔴 问题2: 硬编码Redis密码
**位置**: `config/production/redis.json`
**风险**: Redis缓存密码硬编码
**修复方案**:
1. 移除硬编码密码
2. 使用环境变量或Docker Secrets

### 🔴 问题3: 部署脚本硬编码多组密码
**位置**: `deployment/preprod/deploy.py`
**风险**: 部署脚本自动创建包含硬编码密码的.env文件
**修复方案**:
1. 修改脚本从环境变量读取密码
2. 添加密码生成逻辑
3. 移除所有硬编码字符串

### 🔴 问题4: Python代码中硬编码数据库凭据
**位置**: `deploy/health_check_api.py`
**风险**: 应用程序代码中直接包含数据库凭据
**修复方案**:
1. 使用环境变量
2. 添加配置加载模块

---

## Phase 2: 高危修复 (第2周) - High级别

### 🟠 问题5: Docker Compose默认密码
**位置**: `deployment/preprod/docker-compose.yml`
**修复方案**:
- 移除默认值，强制使用环境变量
- 添加启动检查

### 🟠 问题6: Elasticsearch安全认证禁用
**位置**: `deployment/preprod/docker-compose.yml`
**修复方案**:
- 启用xpack.security
- 配置强密码

### 🟠 问题7: 生产环境使用默认/弱密码
**位置**: `deploy/.env.production`
**修复方案**:
- 替换所有默认密码
- 使用强密码生成器

### 🟠 问题8: PostgreSQL使用默认密码
**位置**: `deploy/docker-compose.production.yml`
**修复方案**:
- 移除默认密码
- 强制环境变量配置

### 🟠 问题9: 端口直接暴露
**位置**: 多个docker-compose文件
**修复方案**:
- 使用expose替代ports
- 配置网络隔离

---

## Phase 3: 中危修复 (第3周) - Medium级别

### 🟡 问题10: 日志可能记录敏感信息
**位置**: `config/production/logging.json`
**修复方案**:
- 添加敏感数据过滤器
- 配置日志脱敏

### 🟡 问题11: 缺少API认证配置
**位置**: `config/production/api.json`
**修复方案**:
- 实施JWT认证
- 添加API密钥验证

### 🟡 问题12: Dockerfile以root用户运行
**位置**: `deploy/Dockerfile`
**修复方案**:
- 创建非root用户
- 配置权限

### 🟡 问题13: 网络隔离配置不足
**位置**: docker-compose文件
**修复方案**:
- 划分frontend/backend网络
- 配置internal网络

---

## Phase 4: 低危修复 (第4周) - Low级别

### 🟢 问题14-16: 最佳实践问题
- 日志文件路径固定
- 缺少依赖包版本锁定
- 健康检查端点公开

---

## 修复实施计划

### Week 1: Critical修复

#### Day 1-2: 移除硬编码密码
- [ ] 扫描所有配置文件中的硬编码密码
- [ ] 创建环境变量模板
- [ ] 修改配置文件加载逻辑

#### Day 3-4: 修改部署脚本
- [ ] 更新deploy.py脚本
- [ ] 添加密码生成工具
- [ ] 测试部署流程

#### Day 5: 验证和文档
- [ ] 运行安全扫描
- [ ] 更新部署文档
- [ ] 创建密码更换清单

### Week 2: High修复

#### Day 6-7: Docker配置安全化
- [ ] 更新所有docker-compose文件
- [ ] 配置Docker Secrets
- [ ] 限制端口暴露

#### Day 8-9: 服务安全配置
- [ ] 启用Elasticsearch安全
- [ ] 配置PostgreSQL强密码
- [ ] 更新Redis配置

#### Day 10: 网络隔离
- [ ] 配置网络分段
- [ ] 测试服务通信
- [ ] 更新网络文档

### Week 3: Medium修复

#### Day 11-13: 应用层安全
- [ ] 添加日志脱敏
- [ ] 实施API认证
- [ ] 创建非root用户

#### Day 14-15: 测试和验证
- [ ] 运行完整测试
- [ ] 验证安全配置
- [ ] 生成安全报告

### Week 4: Low修复和文档

#### Day 16-18: 最佳实践
- [ ] 锁定依赖版本
- [ ] 配置安全头部
- [ ] 优化健康检查

#### Day 19-20: 文档和培训
- [ ] 创建安全操作手册
- [ ] 编写安全部署指南
- [ ] 准备团队培训材料

---

## 安全配置模板

### 环境变量模板 (.env.example)

```bash
# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rqa2025
DB_USER=rqa2025
DB_PASSWORD=<YOUR_STRONG_PASSWORD>

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<YOUR_STRONG_PASSWORD>

# JWT配置
JWT_SECRET=<YOUR_JWT_SECRET>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API配置
API_KEY=<YOUR_API_KEY>
API_RATE_LIMIT=1000

# 外部服务
DATA_PROVIDER_API_KEY=<YOUR_DATA_PROVIDER_API_KEY>
TRADING_API_KEY=<YOUR_TRADING_API_KEY>

# 监控
GRAFANA_PASSWORD=<YOUR_GRAFANA_PASSWORD>
ELASTIC_PASSWORD=<YOUR_ELASTIC_PASSWORD>
```

### Docker Compose安全配置

```yaml
version: '3.8'

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt

services:
  app:
    build: .
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    secrets:
      - db_password
      - redis_password
      - jwt_secret
    networks:
      - frontend
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
  frontend:
  backend:
    internal: true
```

### 密码生成工具

```python
#!/usr/bin/env python3
"""安全密码生成工具"""

import secrets
import string

def generate_password(length=32):
    """生成强密码"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password

def generate_jwt_secret():
    """生成JWT密钥"""
    return secrets.token_urlsafe(32)

def generate_api_key():
    """生成API密钥"""
    return secrets.token_hex(32)

if __name__ == "__main__":
    print("数据库密码:", generate_password())
    print("Redis密码:", generate_password())
    print("JWT密钥:", generate_jwt_secret())
    print("API密钥:", generate_api_key())
```

---

## 安全扫描工具

### 1. 密码扫描脚本

```bash
#!/bin/bash
# scan_hardcoded_passwords.sh

echo "扫描硬编码密码..."

# 搜索常见密码模式
grep -r -n "password.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "passwd.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "secret.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "token.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .
grep -r -n "key.*=.*['\"]" --include="*.py" --include="*.json" --include="*.yml" --include="*.yaml" .

echo "扫描完成"
```

### 2. 安全配置检查脚本

```python
#!/usr/bin/env python3
"""安全配置检查工具"""

import yaml
import json
from pathlib import Path

def check_docker_compose_security(file_path):
    """检查Docker Compose安全配置"""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    issues = []
    
    # 检查端口暴露
    for service_name, service_config in config.get('services', {}).items():
        if 'ports' in service_config:
            for port in service_config['ports']:
                if '5432' in str(port) or '6379' in str(port) or '9200' in str(port):
                    issues.append(f"{service_name}: 数据库端口直接暴露: {port}")
        
        # 检查默认密码
        if 'environment' in service_config:
            for env in service_config['environment']:
                if isinstance(env, str) and ('PASSWORD' in env or 'password' in env):
                    if 'default' in env.lower() or 'postgres' in env.lower():
                        issues.append(f"{service_name}: 可能使用默认密码: {env}")
    
    return issues

def main():
    docker_compose_files = Path('.').rglob('docker-compose*.yml')
    
    for file_path in docker_compose_files:
        print(f"\n检查: {file_path}")
        issues = check_docker_compose_security(file_path)
        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✅ 未发现问题")

if __name__ == "__main__":
    main()
```

---

## 验证清单

### 修复前检查
- [ ] 备份所有配置文件
- [ ] 记录当前密码
- [ ] 通知团队成员
- [ ] 准备回滚方案

### 修复后验证
- [ ] 运行安全扫描脚本
- [ ] 验证所有服务正常启动
- [ ] 测试数据库连接
- [ ] 测试API认证
- [ ] 检查日志无敏感信息
- [ ] 验证网络隔离
- [ ] 运行完整测试套件

### 部署前检查
- [ ] 更新所有环境变量
- [ ] 生成新密码
- [ ] 配置密钥管理服务
- [ ] 更新部署文档
- [ ] 培训运维团队

---

## 联系与支持

如有任何安全问题或建议，请联系：
- **安全负责人**: RQA2025 Security Team
- **紧急联系**: security@rqa2025.com

---

**计划创建时间**: 2026-03-08  
**计划版本**: 1.0  
**最后更新**: 2026-03-08

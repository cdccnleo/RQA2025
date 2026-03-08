#!/usr/bin/env python3
"""
安全漏洞修复脚本
修复审计报告中发现的硬编码密码等安全问题
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime


def log(message):
    """打印日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def fix_database_config():
    """修复数据库配置文件中的硬编码密码"""
    config_file = Path("config/production/database.json")
    
    if not config_file.exists():
        log(f"⚠️  文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 保存原始密码用于备份记录
        original_password = config.get('database', {}).get('password', '')
        
        # 替换为环境变量引用
        config['database']['password'] = '${DB_PASSWORD}'
        
        # 写回文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log(f"✅ 已修复: {config_file}")
        log(f"   原密码已移除，使用环境变量 ${'{DB_PASSWORD}'} 替代")
        return True
        
    except Exception as e:
        log(f"❌ 修复失败 {config_file}: {e}")
        return False


def fix_redis_config():
    """修复Redis配置文件中的硬编码密码"""
    config_file = Path("config/production/redis.json")
    
    if not config_file.exists():
        log(f"⚠️  文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 替换为环境变量引用
        config['redis']['password'] = '${REDIS_PASSWORD}'
        
        # 写回文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log(f"✅ 已修复: {config_file}")
        log(f"   原密码已移除，使用环境变量 ${'{REDIS_PASSWORD}'} 替代")
        return True
        
    except Exception as e:
        log(f"❌ 修复失败 {config_file}: {e}")
        return False


def fix_health_check_api():
    """修复健康检查API中的硬编码密码"""
    api_file = Path("deploy/health_check_api.py")
    
    if not api_file.exists():
        log(f"⚠️  文件不存在: {api_file}")
        return False
    
    try:
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换硬编码的数据库配置
        old_db_config = """DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'rqa2025',
    'user': 'postgres',
    'password': 'postgres'
}"""
        
        new_db_config = """# 从环境变量读取数据库配置
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'postgres'),
    'port': int(os.environ.get('DB_PORT', '5432')),
    'database': os.environ.get('DB_NAME', 'rqa2025'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', '')
}"""
        
        content = content.replace(old_db_config, new_db_config)
        
        # 替换硬编码的Redis配置
        old_redis_config = """REDIS_CONFIG = {
    'host': 'redis',
    'port': 6379,
    'password': '',
    'db': 0
}"""
        
        new_redis_config = """# 从环境变量读取Redis配置
REDIS_CONFIG = {
    'host': os.environ.get('REDIS_HOST', 'redis'),
    'port': int(os.environ.get('REDIS_PORT', '6379')),
    'password': os.environ.get('REDIS_PASSWORD', ''),
    'db': int(os.environ.get('REDIS_DB', '0'))
}"""
        
        content = content.replace(old_redis_config, new_redis_config)
        
        # 添加os导入（如果不存在）
        if 'import os' not in content:
            content = content.replace('import logging', 'import os\nimport logging')
        
        # 写回文件
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log(f"✅ 已修复: {api_file}")
        log(f"   硬编码凭据已替换为环境变量读取")
        return True
        
    except Exception as e:
        log(f"❌ 修复失败 {api_file}: {e}")
        return False


def create_env_example():
    """创建环境变量模板文件"""
    env_content = """# RQA2025 生产环境配置模板
# 复制此文件为 .env 并填写实际值
# 注意: .env 文件不应提交到Git仓库

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rqa2025_prod
DB_USER=rqa2025_user
DB_PASSWORD=<YOUR_STRONG_PASSWORD_HERE>

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<YOUR_STRONG_PASSWORD_HERE>
REDIS_DB=0

# JWT配置
JWT_SECRET=<YOUR_JWT_SECRET_KEY>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API配置
API_KEY=<YOUR_API_KEY>

# 监控配置
GRAFANA_PASSWORD=<YOUR_GRAFANA_PASSWORD>
ELASTIC_PASSWORD=<YOUR_ELASTIC_PASSWORD>

# 外部服务API密钥
DATA_PROVIDER_API_KEY=<YOUR_DATA_PROVIDER_API_KEY>
TRADING_API_KEY=<YOUR_TRADING_API_KEY>
"""
    
    env_file = Path(".env.example")
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        log(f"✅ 已创建: {env_file}")
        return True
    except Exception as e:
        log(f"❌ 创建失败 {env_file}: {e}")
        return False


def update_gitignore():
    """更新.gitignore文件，添加敏感文件"""
    gitignore_file = Path(".gitignore")
    
    sensitive_patterns = [
        "# 环境变量文件（包含敏感信息）",
        ".env",
        ".env.production",
        ".env.local",
        ".env.*.local",
        "",
        "# 密钥文件",
        "secrets/",
        "*.pem",
        "*.key",
        "",
        "# 配置文件备份（可能包含密码）",
        "config/production/*.backup",
        "config/production/*.bak",
        "",
        "# 日志文件",
        "*.log",
        "logs/",
    ]
    
    try:
        existing_content = ""
        if gitignore_file.exists():
            with open(gitignore_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # 检查是否已存在这些模式
        new_patterns = []
        for pattern in sensitive_patterns:
            if pattern and pattern not in existing_content:
                new_patterns.append(pattern)
        
        if new_patterns:
            with open(gitignore_file, 'a', encoding='utf-8') as f:
                f.write('\n')
                f.write('\n'.join(new_patterns))
            log(f"✅ 已更新: {gitignore_file}")
            log(f"   添加了 {len(new_patterns)} 个安全相关的忽略模式")
            return True
        else:
            log(f"ℹ️  {gitignore_file} 已包含所有必要的安全模式")
            return True
            
    except Exception as e:
        log(f"❌ 更新失败 {gitignore_file}: {e}")
        return False


def create_security_readme():
    """创建安全操作指南"""
    readme_content = """# RQA2025 安全配置指南

## 概述

本文档指导如何安全地配置RQA2025量化交易系统的敏感信息。

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
python scripts/generate_secure_passwords.py
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
   python scripts/generate_secure_passwords.py
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
   python deploy/health_check_api.py
   ```

## 安全最佳实践

### 1. 环境变量管理

- 永远不要在代码中硬编码密码
- 使用 `.env` 文件管理敏感信息
- 将 `.env` 添加到 `.gitignore`
- 定期轮换密码

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

## 紧急响应

### 如果发现密码泄露

1. **立即更换密码**
   ```bash
   # 生成新密码
   python scripts/generate_secure_passwords.py
   
   # 更新所有服务
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

## 联系方式

如有安全问题，请联系:
- 安全团队: security@rqa2025.com
- 紧急热线: [紧急联系电话]

---

**最后更新**: 2026-03-08
"""
    
    readme_file = Path("SECURITY.md")
    try:
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        log(f"✅ 已创建: {readme_file}")
        return True
    except Exception as e:
        log(f"❌ 创建失败 {readme_file}: {e}")
        return False


def generate_secure_passwords():
    """生成安全密码"""
    import secrets
    import string
    
    def generate_password(length=32):
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    passwords = {
        'DB_PASSWORD': generate_password(),
        'REDIS_PASSWORD': generate_password(),
        'JWT_SECRET': secrets.token_urlsafe(32),
        'API_KEY': secrets.token_hex(32),
        'GRAFANA_PASSWORD': generate_password(),
        'ELASTIC_PASSWORD': generate_password(),
    }
    
    log("\n🔐 生成的安全密码（请保存到.env文件）:")
    log("=" * 60)
    for key, value in passwords.items():
        log(f"{key}={value}")
    log("=" * 60)
    log("⚠️  警告: 这些密码只显示一次，请立即保存！")
    
    # 保存到文件
    passwords_file = Path("generated_passwords.txt")
    with open(passwords_file, 'w', encoding='utf-8') as f:
        f.write("# 生成的安全密码\n")
        f.write("# 请将这些密码复制到.env文件中\n")
        f.write("# 然后删除此文件\n\n")
        for key, value in passwords.items():
            f.write(f"{key}={value}\n")
    
    log(f"\n密码已保存到: {passwords_file}")
    log("请将这些密码复制到.env文件，然后删除 generated_passwords.txt")


def main():
    """主函数"""
    log("=" * 70)
    log("RQA2025 安全漏洞修复工具")
    log("=" * 70)
    log("")
    
    results = {
        'database_config': False,
        'redis_config': False,
        'health_check_api': False,
        'env_example': False,
        'gitignore': False,
        'security_readme': False,
    }
    
    # 修复配置文件
    log("步骤1: 修复配置文件中的硬编码密码...")
    results['database_config'] = fix_database_config()
    results['redis_config'] = fix_redis_config()
    log("")
    
    # 修复Python代码
    log("步骤2: 修复Python代码中的硬编码凭据...")
    results['health_check_api'] = fix_health_check_api()
    log("")
    
    # 创建环境变量模板
    log("步骤3: 创建环境变量模板...")
    results['env_example'] = create_env_example()
    log("")
    
    # 更新.gitignore
    log("步骤4: 更新.gitignore...")
    results['gitignore'] = update_gitignore()
    log("")
    
    # 创建安全文档
    log("步骤5: 创建安全操作指南...")
    results['security_readme'] = create_security_readme()
    log("")
    
    # 生成安全密码
    log("步骤6: 生成安全密码...")
    generate_secure_passwords()
    log("")
    
    # 显示总结
    log("=" * 70)
    log("修复总结")
    log("=" * 70)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for task, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        log(f"{task}: {status}")
    
    log("")
    log(f"总计: {success_count}/{total_count} 项修复成功")
    log("")
    
    if success_count == total_count:
        log("🎉 所有安全修复已完成！")
        log("")
        log("下一步操作:")
        log("1. 将生成的密码复制到 .env 文件")
        log("2. 运行: docker-compose restart")
        log("3. 验证服务正常启动")
        log("4. 提交代码到GitHub")
    else:
        log("⚠️  部分修复失败，请检查错误信息")
    
    log("=" * 70)


if __name__ == "__main__":
    main()

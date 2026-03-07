# 邮件配置安全指南

## 概述

本指南介绍如何安全地配置邮件功能，避免敏感信息泄露。

## 安全风险

### 常见安全问题
1. **明文密码存储**: 在配置文件中直接存储邮箱密码
2. **版本控制泄露**: 敏感配置文件被提交到Git仓库
3. **权限不当**: 配置文件权限设置过于开放
4. **环境变量暴露**: 环境变量中包含敏感信息

### 安全最佳实践

#### 1. 使用应用专用密码
- 不要使用邮箱账户的主密码
- 在邮箱服务商处生成应用专用密码
- 定期更换应用专用密码

#### 2. 加密存储敏感信息
- 使用Fernet对称加密
- 密钥单独存储，不提交到版本控制
- 支持环境变量配置

#### 3. 环境变量配置
```bash
# 设置环境变量
export EMAIL_USERNAME=your_email@163.com
export EMAIL_PASSWORD=your_app_password
export EMAIL_FROM=your_email@163.com
export EMAIL_TO=recipient@example.com
```

#### 4. 文件权限控制
```bash
# 设置适当的文件权限
chmod 600 config/.email_key
chmod 600 config/email_config.encrypted.json
```

## 配置方法

### 方法一：使用安全配置脚本

1. 运行配置脚本：
```bash
python scripts/security/setup_secure_email.py
```

2. 按提示输入配置信息
3. 脚本会自动加密并保存配置

### 方法二：手动环境变量配置

1. 创建环境变量文件：
```bash
cp config/env_example.txt config/.env
```

2. 编辑 `.env` 文件，填入实际值
3. 确保 `.env` 文件不被提交到版本控制

### 方法三：使用加密配置文件

1. 准备明文配置文件
2. 使用加密工具加密：
```python
from src.infrastructure.email.secure_config import SecureEmailConfig

config_manager = SecureEmailConfig()
config = {
    "smtp_server": "smtp.163.com",
    "smtp_port": 25,
    "username": "your_email@163.com",
    "password": "your_app_password",
    "from_email": "your_email@163.com",
    "to_emails": ["recipient@example.com"]
}

config_manager.save_encrypted_config(config)
```

## 代码使用示例

### 加载配置
```python
from src.infrastructure.email.secure_config import get_email_config

try:
    config = get_email_config()
    print(f"发件人: {config['from_email']}")
    print(f"收件人: {config['to_emails']}")
except Exception as e:
    print(f"配置加载失败: {e}")
```

### 发送邮件
```python
import smtplib
from email.mime.text import MIMEText
from src.infrastructure.email.secure_config import get_email_config

def send_email(subject: str, content: str):
    config = get_email_config()
    
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = config['from_email']
    msg['To'] = ', '.join(config['to_emails'])
    
    with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
```

## 安全检查清单

### 开发环境
- [ ] 敏感信息已从配置文件中移除
- [ ] 使用环境变量或加密配置
- [ ] `.env` 文件已添加到 `.gitignore`
- [ ] 加密密钥文件权限设置为600
- [ ] 使用应用专用密码而非账户密码

### 生产环境
- [ ] 使用密钥管理服务（如AWS KMS、Azure Key Vault）
- [ ] 环境变量通过安全渠道设置
- [ ] 定期轮换密码和密钥
- [ ] 监控异常登录活动
- [ ] 启用邮箱安全功能（如2FA）

### 代码审查
- [ ] 检查是否有硬编码的敏感信息
- [ ] 确认配置文件权限设置正确
- [ ] 验证加密配置正常工作
- [ ] 测试环境变量配置

## 故障排除

### 常见问题

#### 1. 配置加载失败
**症状**: `FileNotFoundError` 或 `ValueError`
**解决**: 检查配置文件路径和格式

#### 2. 加密密钥问题
**症状**: `InvalidToken` 异常
**解决**: 重新生成加密密钥或检查密钥文件

#### 3. 环境变量未设置
**症状**: 配置值为空
**解决**: 检查环境变量是否正确设置

#### 4. 邮件发送失败
**症状**: SMTP认证失败
**解决**: 检查用户名密码和服务器设置

### 调试命令
```bash
# 检查环境变量
echo $EMAIL_USERNAME

# 测试配置加载
python -c "from src.infrastructure.email.secure_config import get_email_config; print(get_email_config())"

# 检查文件权限
ls -la config/.email_key
ls -la config/email_config.encrypted.json
```

## 安全更新

### 定期维护
1. **密码轮换**: 每3-6个月更换应用专用密码
2. **密钥轮换**: 每年更换加密密钥
3. **权限检查**: 定期检查文件权限设置
4. **安全审计**: 定期审查配置和代码

### 应急响应
1. **密码泄露**: 立即更换所有相关密码
2. **密钥泄露**: 重新生成加密密钥并重新加密配置
3. **账户异常**: 检查邮箱登录记录，必要时锁定账户

## 相关文档
- [邮件配置API文档](../api/email_api.md)
- [安全最佳实践](../security/BEST_PRACTICES.md)
- [部署安全指南](../deployment/SECURITY.md) 
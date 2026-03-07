# 邮件告警系统配置指南

## 📧 邮件系统配置

### 1. 配置文件位置
邮件系统配置文件位于：`config/email_config.json`

### 2. 配置项说明

```json
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "your-email@gmail.com",
  "password": "your-app-password",
  "from_email": "your-email@gmail.com",
  "to_emails": ["admin@example.com", "tech@example.com"],
  "alert_levels": {
    "critical": {"threshold": 90, "cooldown": 300},
    "warning": {"threshold": 70, "cooldown": 600},
    "info": {"threshold": 50, "cooldown": 1800}
  }
}
```

### 3. 收件人配置

#### 单个收件人
```json
"to_emails": ["admin@example.com"]
```

#### 多个收件人
```json
"to_emails": [
  "admin@example.com",
  "tech@example.com", 
  "manager@example.com"
]
```

#### 按告警级别配置不同收件人
```json
{
  "to_emails": {
    "critical": ["admin@example.com", "emergency@example.com"],
    "warning": ["tech@example.com", "manager@example.com"],
    "info": ["monitor@example.com"]
  }
}
```

### 4. 邮件服务器配置

#### Gmail配置
```json
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "your-gmail@gmail.com",
  "password": "your-app-password"
}
```

#### QQ邮箱配置
```json
{
  "smtp_server": "smtp.qq.com", 
  "smtp_port": 587,
  "username": "your-qq@qq.com",
  "password": "your-authorization-code"
}
```

#### 163邮箱配置
```json
{
  "smtp_server": "smtp.163.com",
  "smtp_port": 25,
  "username": "your-163@163.com", 
  "password": "your-authorization-code"
}
```

### 5. 告警级别配置

#### 告警级别说明
- **critical**: 严重告警，立即通知
- **warning**: 警告告警，延迟通知
- **info**: 信息告警，定期通知

#### 冷却时间配置
```json
"alert_levels": {
  "critical": {"threshold": 90, "cooldown": 300},   // 5分钟冷却
  "warning": {"threshold": 70, "cooldown": 600},    // 10分钟冷却
  "info": {"threshold": 50, "cooldown": 1800}       // 30分钟冷却
}
```

### 6. 配置步骤

#### 步骤1: 创建配置文件
```bash
mkdir -p config
```

#### 步骤2: 编辑配置文件
```bash
# 创建配置文件
cat > config/email_config.json << EOF
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "username": "your-email@gmail.com",
  "password": "your-app-password",
  "from_email": "your-email@gmail.com",
  "to_emails": ["admin@example.com"],
  "alert_levels": {
    "critical": {"threshold": 90, "cooldown": 300},
    "warning": {"threshold": 70, "cooldown": 600},
    "info": {"threshold": 50, "cooldown": 1800}
  }
}
EOF
```

#### 步骤3: 测试配置
```bash
python scripts/monitoring/email_alert_system.py
```

### 7. 常见问题解决

#### 问题1: 认证失败
**错误信息**: `Username and Password not accepted`

**解决方案**:
1. 确保使用应用专用密码
2. 开启两步验证
3. 检查用户名和密码是否正确

#### 问题2: 连接超时
**错误信息**: `Connection timeout`

**解决方案**:
1. 检查网络连接
2. 确认SMTP服务器地址和端口
3. 检查防火墙设置

#### 问题3: 邮件发送失败
**错误信息**: `Failed to send email`

**解决方案**:
1. 检查收件人邮箱地址格式
2. 确认发件人邮箱已配置
3. 检查邮件服务器设置

### 8. 安全建议

#### 密码安全
- 使用应用专用密码，不要使用登录密码
- 定期更换密码
- 不要在代码中硬编码密码

#### 邮箱安全
- 开启两步验证
- 使用强密码
- 定期检查登录记录

### 9. 测试邮件发送

#### 测试脚本
```python
from scripts.monitoring.email_alert_system import EmailAlertSystem

# 创建邮件系统实例
email_system = EmailAlertSystem()

# 发送测试邮件
success = email_system.send_alert(
    level="info",
    subject="系统测试",
    message="这是一条测试邮件，用于验证邮件配置是否正确。"
)

if success:
    print("✅ 测试邮件发送成功")
else:
    print("❌ 测试邮件发送失败")
```

### 10. 监控和日志

#### 日志文件位置
- 邮件告警日志: `logs/email_alerts.log`
- 系统日志: `logs/system.log`

#### 日志内容
```
2025-07-28 15:30:00 - INFO - 邮件服务器连接测试成功
2025-07-28 15:30:01 - INFO - 告警邮件发送成功: warning - 系统性能告警
2025-07-28 15:30:02 - ERROR - 发送告警邮件失败: 连接超时
```

### 11. 高级配置

#### 邮件模板自定义
```python
# 自定义邮件模板
def custom_email_template(level, subject, message, metrics):
    return f"""
    <html>
    <body>
        <h1>🚨 {level.upper()} 告警</h1>
        <h2>{subject}</h2>
        <p>{message}</p>
        <h3>监控指标:</h3>
        <ul>
            {''.join([f'<li>{k}: {v}</li>' for k, v in metrics.items()])}
        </ul>
    </body>
    </html>
    """
```

#### 告警规则自定义
```python
# 自定义告警规则
def custom_alert_rule(metrics):
    if metrics['cpu_usage'] > 90:
        return 'critical'
    elif metrics['memory_usage'] > 80:
        return 'warning'
    else:
        return 'info'
```

## 📋 配置检查清单

- [ ] SMTP服务器配置正确
- [ ] 用户名和密码正确
- [ ] 收件人邮箱地址正确
- [ ] 告警级别配置合理
- [ ] 冷却时间设置适当
- [ ] 测试邮件发送成功
- [ ] 日志记录正常
- [ ] 安全设置完成

## 🎯 最佳实践

1. **使用应用专用密码**
2. **配置多个收件人**
3. **设置合理的冷却时间**
4. **定期测试邮件功能**
5. **监控邮件发送状态**
6. **备份配置文件**
7. **记录配置变更** 
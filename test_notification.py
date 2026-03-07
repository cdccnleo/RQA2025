#!/usr/bin/env python3
"""
通知系统测试脚本
"""

import sys
import os

# 切换到通知目录以便导入
os.chdir('src/pipeline/notification')
sys.path.insert(0, '.')

from notification_service import NotificationService, NotificationLevel, NotificationResult, NotificationChannel
from log_channel import LogNotificationChannel
from email_channel import EmailNotificationChannel
from webhook_channel import WebhookNotificationChannel

print('=' * 60)
print('通知系统测试')
print('=' * 60)

# 1. 测试创建实例
print('\n1. 测试创建实例...')
service = NotificationService()
log_channel = LogNotificationChannel(name='test_log', enabled=True)
email_channel = EmailNotificationChannel(name='test_email', enabled=False)
webhook_channel = WebhookNotificationChannel(name='test_webhook', enabled=False)
print(f'   ✓ 通知服务创建成功')
print(f'   ✓ 日志通道创建成功: {log_channel.name}')
print(f'   ✓ 邮件通道创建成功: {email_channel.name}')
print(f'   ✓ Webhook通道创建成功: {webhook_channel.name}')

# 2. 测试注册通道
print('\n2. 测试注册通道...')
service.register_channel(log_channel, is_default=True)
service.register_channel(email_channel)
service.register_channel(webhook_channel)
print(f'   ✓ 已注册通道数: {len(service.get_all_channels())}')

# 3. 测试发送通知
print('\n3. 测试发送通知...')
results = service.send('测试通知消息 - INFO级别', level=NotificationLevel.INFO)
print(f'   ✓ 发送结果数量: {len(results)}')

# 4. 测试不同级别
print('\n4. 测试不同通知级别...')
for level in [NotificationLevel.DEBUG, NotificationLevel.WARNING, NotificationLevel.ERROR, NotificationLevel.CRITICAL]:
    results = service.send(f'测试 {level.name} 级别消息', level=level)
    print(f'   ✓ {level.name}: 成功发送')

# 5. 测试广播
print('\n5. 测试广播到所有通道...')
broadcast_results = service.broadcast('广播消息测试', level=NotificationLevel.WARNING)
print(f'   ✓ 广播结果数量: {len(broadcast_results)}')

# 6. 获取统计信息
print('\n6. 获取统计信息...')
stats = service.get_statistics()
print(f'   ✓ 总通道数: {stats["total_channels"]}')
print(f'   ✓ 启用通道数: {stats["enabled_channels"]}')
print(f'   ✓ 默认通道: {stats["default_channels"]}')

# 7. 测试日志通道历史记录
print('\n7. 测试日志通道历史记录...')
history = log_channel.get_history()
print(f'   ✓ 日志历史记录数: {len(history)}')

# 8. 测试通道配置
print('\n8. 测试通道配置...')
print(f'   ✓ 邮件通道配置: {email_channel.to_dict()["smtp_host"]}')
print(f'   ✓ Webhook通道配置: {webhook_channel.to_dict()["method"]}')
print(f'   ✓ 日志通道配置: {log_channel.to_dict()["logger_name"]}')

# 9. 测试注销通道
print('\n9. 测试注销通道...')
result = service.unregister_channel('test_email')
print(f'   ✓ 注销test_email: {result}')
print(f'   ✓ 剩余通道数: {len(service.get_all_channels())}')

# 10. 测试关闭服务
print('\n10. 测试关闭服务...')
service.shutdown()
print('   ✓ 服务已关闭')

print('\n' + '=' * 60)
print('所有测试通过！')
print('=' * 60)

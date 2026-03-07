#!/usr/bin/env python3
"""
配置调试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from src.infrastructure.email.secure_config import SecureEmailConfig

    print("=== 配置调试 ===")

    # 检查环境变量
    print("环境变量:")
    print(f"EMAIL_SMTP_SERVER: {os.getenv('EMAIL_SMTP_SERVER')}")
    print(f"EMAIL_SMTP_PORT: {os.getenv('EMAIL_SMTP_PORT')}")
    print(f"EMAIL_SENDER_USER: {os.getenv('EMAIL_SENDER_USER')}")
    print(
        f"EMAIL_SENDER_PASS: {'*' * len(os.getenv('EMAIL_SENDER_PASS', '')) if os.getenv('EMAIL_SENDER_PASS') else 'None'}")
    print(f"EMAIL_SENDER_ADDRESS: {os.getenv('EMAIL_SENDER_ADDRESS')}")
    print(f"EMAIL_RECEIVER_LIST: {os.getenv('EMAIL_RECEIVER_LIST')}")

    # 检查配置文件
    print("\n配置文件内容:")
    config_path = Path("config/email_config.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            import json
            config = json.load(f)
            for key, value in config.items():
                print(f"  {key}: {value}")
    else:
        print("配置文件不存在")

    # 测试配置加载
    print("\n配置加载测试:")
    config_manager = SecureEmailConfig()
    try:
        config = config_manager.load_config()
        print("✅ 配置加载成功:")
        for key, value in config.items():
            if key == 'password':
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")

except Exception as e:
    print(f"❌ 调试过程出错: {e}")

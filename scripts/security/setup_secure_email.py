#!/usr/bin/env python3
"""
邮件配置安全初始化脚本
帮助用户安全地设置邮件配置，避免敏感信息泄露
"""

from src.infrastructure.email.secure_config import SecureEmailConfig
import sys
import getpass
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))


def get_user_input(prompt: str, is_password: bool = False) -> str:
    """安全获取用户输入"""
    if is_password:
        return getpass.getpass(prompt)
    else:
        return input(prompt).strip()


def validate_email(email: str) -> bool:
    """简单验证邮箱格式"""
    return '@' in email and '.' in email


def setup_email_config():
    """设置邮件配置"""
    print("=== 邮件配置安全设置 ===")
    print("此脚本将帮助您安全地设置邮件配置")
    print("敏感信息将被加密存储\n")

    # 获取邮件服务器信息
    print("1. 邮件服务器配置")
    smtp_server = get_user_input("SMTP服务器地址 (默认: smtp.163.com): ") or "smtp.163.com"
    smtp_port = get_user_input("SMTP端口 (默认: 25): ") or "25"

    try:
        smtp_port = int(smtp_port)
    except ValueError:
        print("端口必须是数字，使用默认值25")
        smtp_port = 25

    # 获取账户信息
    print("\n2. 邮件账户信息")
    username = get_user_input("邮箱用户名: ")
    while not validate_email(username):
        print("请输入有效的邮箱地址")
        username = get_user_input("邮箱用户名: ")

    password = get_user_input("邮箱密码 (将使用应用专用密码): ", is_password=True)
    from_email = get_user_input(f"发件人邮箱 (默认: {username}): ") or username

    # 获取收件人信息
    print("\n3. 收件人配置")
    to_emails = []
    while True:
        recipient = get_user_input("收件人邮箱 (留空结束): ")
        if not recipient:
            break
        if validate_email(recipient):
            to_emails.append(recipient)
        else:
            print("请输入有效的邮箱地址")

    if not to_emails:
        print("至少需要配置一个收件人")
        return False

    # 构建配置
    config = {
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
        "username": username,
        "password": password,
        "from_email": from_email,
        "to_emails": to_emails
    }

    # 验证配置
    config_manager = SecureEmailConfig()
    if not config_manager.validate_config(config):
        print("配置验证失败")
        return False

    # 保存加密配置
    try:
        config_manager.save_encrypted_config(config)
        print("\n✅ 邮件配置已安全保存")
        print("📁 加密配置文件: config/email_config.encrypted.json")
        print("🔑 加密密钥文件: config/.email_key")
        print("\n⚠️  安全提醒:")
        print("   - 请确保 .email_key 文件不被提交到版本控制系统")
        print("   - 定期更换邮箱密码")
        print("   - 建议使用应用专用密码而非账户密码")
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False


def setup_environment_variables():
    """设置环境变量配置"""
    print("\n=== 环境变量配置 ===")
    print("您也可以使用环境变量来配置邮件设置")
    print("创建 .env 文件并添加以下内容:\n")

    env_content = """# 邮件配置环境变量
EMAIL_SMTP_SERVER=smtp.163.com
EMAIL_SMTP_PORT=25
EMAIL_SENDER_USER=your_email@163.com
EMAIL_SENDER_PASS=your_app_password
EMAIL_SENDER_ADDRESS=your_email@163.com
EMAIL_RECEIVER_LIST=recipient1@example.com,recipient2@example.com

# 加密密钥 (可选)
EMAIL_ENCRYPTION_KEY=your_base64_encoded_key
"""

    print(env_content)

    # 创建 .env 文件
    env_file = Path("config/.env")
    if not env_file.exists():
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"✅ 环境变量模板已创建: {env_file}")
        except Exception as e:
            print(f"创建环境变量文件失败: {e}")


def main():
    """主函数"""
    print("邮件配置安全设置工具")
    print("=" * 50)

    # 检查是否已存在配置
    config_manager = SecureEmailConfig()
    encrypted_config_path = Path("config/email_config.encrypted.json")

    if encrypted_config_path.exists():
        choice = get_user_input("检测到现有加密配置，是否重新设置? (y/N): ").lower()
        if choice != 'y':
            print("保持现有配置")
            return

    # 设置配置
    if setup_email_config():
        setup_environment_variables()

        # 测试配置
        print("\n=== 配置测试 ===")
        try:
            config = config_manager.load_encrypted_config()
            print("✅ 配置加载成功")
            print(f"📧 发件人: {config['from_email']}")
            print(f"📬 收件人: {', '.join(config['to_emails'])}")
            print(f"🔗 服务器: {config['smtp_server']}:{config['smtp_port']}")
        except Exception as e:
            print(f"❌ 配置测试失败: {e}")
    else:
        print("配置设置失败")


if __name__ == "__main__":
    main()

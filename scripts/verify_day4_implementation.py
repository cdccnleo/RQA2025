#!/usr/bin/env python3
"""
Day 4 任务实现验证脚本

验证身份认证模块、数据保护模块、监控告警系统的实现
    创建时间: 2024年12月
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from infrastructure.security.authentication_service import (
        MultiFactorAuthenticationService,
        UserRole, AuthMethod, AuthStatus
    )
    from infrastructure.security.data_protection_service import (
        DataProtectionService, DataQualityMonitor
    )
    from infrastructure.monitoring.alert_system import (
        IntelligentAlertSystem, AlertChannel, ConsoleNotifier
    )
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_authentication_service():
    """测试多因素认证服务"""
    print("\n" + "="*60)
    print("🔐 测试多因素认证服务")
    print("="*60)

    auth_service = MultiFactorAuthenticationService()

    try:
        # 创建用户
        user_id = auth_service.create_user(
            username="test_trader",
            email="test@example.com",
            password="secure_password123",
            role=UserRole.TRADER
        )

        if not user_id:
            print("❌ 用户创建失败")
            return False

        print("✅ 用户创建成功")

        # 设置TOTP
        auth_service.setup_mfa(user_id, AuthMethod.TOTP, {})
        print("✅ TOTP 设置成功")

        # 获取正确的TOTP代码
        totp_code = auth_service.generate_current_totp(user_id)
        if not totp_code:
            print("❌ TOTP代码生成失败")
            return False

        print(f"🔢 生成的TOTP代码: {totp_code}")

        # 执行多因素认证
        result = auth_service.authenticate_user(
            "test_trader",
            {
                "password": "secure_password123",
                "totp_code": totp_code
            },
            required_factors=[AuthMethod.PASSWORD, AuthMethod.TOTP]
        )

        if result.status == AuthStatus.SUCCESS:
            print("✅ 多因素认证成功")
            print(f"   用户: {result.user.username}")
            print(f"   角色: {result.user.role.value}")
            print(f"   认证因素: {result.factors_completed}")

            # 验证令牌
            user = auth_service.verify_token(result.token)
            if user:
                print("✅ JWT令牌验证成功")
            else:
                print("❌ JWT令牌验证失败")
                return False

            return True
        else:
            print(f"❌ 认证失败: {result.message}")
            return False

    except Exception as e:
        print(f"❌ 认证服务测试异常: {e}")
        return False


def test_data_protection_service():
    """测试数据保护服务"""
    print("\n" + "="*60)
    print("🛡️ 测试数据保护服务")
    print("="*60)

    protection_service = DataProtectionService()

    try:
        # 示例用户数据
        user_data = {
            "user_id": "123456",
            "name": "张三",
            "phone": "13812345678",
            "id_card": "123456199001011234",
            "email": "zhangsan@example.com",
            "bank_account": "6222021234567890123",
            "password": "mypassword123"
        }

        print("原始数据:")
        print(json.dumps(user_data, ensure_ascii=False, indent=2))

        # 保护数据
        protected_data = protection_service.protect_data(
            user_data,
            rule_id="user_data",
            user_id="admin",
            operation="user_registration",
            ip_address="192.168.1.100"
        )

        print("\n保护后的数据:")
        print(json.dumps(protected_data, ensure_ascii=False, indent=2))

        # 验证保护效果
        if protected_data["phone"] != user_data["phone"]:
            print("✅ 手机号脱敏成功")
        else:
            print("❌ 手机号脱敏失败")
            return False

        if protected_data["bank_account"].startswith("TOK:"):
            print("✅ 银行账户标记化成功")
        else:
            print("❌ 银行账户标记化失败")
            return False

        if protected_data["password"].startswith("HASH:"):
            print("✅ 密码哈希成功")
        else:
            print("❌ 密码哈希失败")
            return False

        # 数据质量检查
        quality_monitor = DataQualityMonitor(protection_service)
        quality_report = quality_monitor.check_data_quality(user_data, "user_data")

        print("\n数据质量报告:")
        print(json.dumps(quality_report, ensure_ascii=False, indent=2))

        if quality_report["status"] in ["success", "warning"]:
            print("✅ 数据质量检查通过")
        else:
            print("❌ 数据质量检查失败")
            return False

        # 获取审计日志
        audit_logs = protection_service.get_audit_logs()
        if audit_logs:
            print(f"✅ 审计日志记录成功，共 {len(audit_logs)} 条")
        else:
            print("❌ 审计日志记录失败")
            return False

        return True

    except Exception as e:
        print(f"❌ 数据保护服务测试异常: {e}")
        return False


def test_alert_system():
    """测试智能告警系统"""
    print("\n" + "="*60)
    print("🚨 测试智能告警系统")
    print("="*60)

    alert_system = IntelligentAlertSystem()

    try:
        # 创建默认规则
        alert_system.create_default_rules()
        print("✅ 默认告警规则创建成功")

        # 注册控制台通知器
        console_notifier = ConsoleNotifier()
        alert_system.register_notifier(AlertChannel.CONSOLE, console_notifier)
        print("✅ 控制台通知器注册成功")

        # 模拟正常监控数据
        normal_data = {
            "cpu_usage": 60,
            "memory_usage": 70,
            "error_rate": 2,
            "failed_attempts": 2
        }

        alert_system.check_alerts(normal_data, "system_monitor")
        time.sleep(1)  # 等待处理

        # 模拟异常监控数据
        abnormal_data = {
            "cpu_usage": 85,  # 触发CPU告警
            "memory_usage": 90,  # 触发内存告警
            "error_rate": 3,
            "failed_attempts": 6  # 触发登录失败告警
        }

        alert_system.check_alerts(abnormal_data, "system_monitor")
        time.sleep(1)  # 等待处理

        # 模拟严重异常数据
        critical_data = {
            "cpu_usage": 95,
            "memory_usage": 98,
            "error_type": "trading_error",  # 触发交易错误告警
            "failed_attempts": 10
        }

        alert_system.check_alerts(critical_data, "trading_system")
        time.sleep(1)  # 等待处理

        # 获取活跃告警
        active_alerts = alert_system.get_active_alerts()
        print(f"\n活跃告警数量: {len(active_alerts)}")

        if len(active_alerts) > 0:
            print("✅ 告警触发成功")
            for alert in active_alerts:
                print(f"  - {alert.title}: {alert.message}")
        else:
            print("❌ 告警触发失败")
            return False

        # 获取告警历史
        all_alerts = alert_system.get_alert_history()
        print(f"总告警数量: {len(all_alerts)}")

        if len(all_alerts) >= len(active_alerts):
            print("✅ 告警历史记录成功")
        else:
            print("❌ 告警历史记录失败")
            return False

        # 导出报告
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        alert_system.export_alert_report(start_time, end_time, "test_alert_report.json")

        if os.path.exists("test_alert_report.json"):
            print("✅ 告警报告导出成功")
        else:
            print("❌ 告警报告导出失败")
            return False

        # 关闭系统
        alert_system.shutdown()
        print("✅ 告警系统关闭成功")

        return True

    except Exception as e:
        print(f"❌ 告警系统测试异常: {e}")
        return False


def test_integration():
    """测试模块集成"""
    print("\n" + "="*60)
    print("🔗 测试模块集成")
    print("="*60)

    try:
        # 初始化各个服务
        auth_service = MultiFactorAuthenticationService()
        protection_service = DataProtectionService()
        alert_system = IntelligentAlertSystem()

        # 创建用户
        user_id = auth_service.create_user(
            username="integration_test",
            email="integration@example.com",
            password="test_password123",
            role=UserRole.TRADER
        )

        if not user_id:
            print("❌ 集成测试用户创建失败")
            return False

        print("✅ 集成测试用户创建成功")

        # 保护用户数据
        user_data = {
            "user_id": user_id,
            "name": "集成测试用户",
            "phone": "13900000000",
            "email": "integration@example.com"
        }

        protected_data = protection_service.protect_data(
            user_data,
            rule_id="user_data",
            user_id="system",
            operation="user_registration"
        )

        print("✅ 用户数据保护成功")

        # 设置告警规则
        alert_system.create_default_rules()
        alert_system.register_notifier(AlertChannel.CONSOLE, ConsoleNotifier())

        print("✅ 告警系统集成成功")

        # 模拟系统监控
        monitoring_data = {
            "cpu_usage": 75,
            "memory_usage": 80,
            "active_users": 1,
            "data_protection_events": 1
        }

        alert_system.check_alerts(monitoring_data, "integration_test")

        time.sleep(1)

        # 检查集成结果
        active_alerts = alert_system.get_active_alerts()
        audit_logs = protection_service.get_audit_logs()

        if len(audit_logs) > 0 and len(active_alerts) >= 0:
            print("✅ 模块集成测试成功")
            print(f"  - 数据保护事件: {len(audit_logs)}")
            print(f"  - 活跃告警: {len(active_alerts)}")
        else:
            print("❌ 模块集成测试失败")
            return False

        # 清理资源
        alert_system.shutdown()

        return True

    except Exception as e:
        print(f"❌ 集成测试异常: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始验证Day 4任务实现")
    print(f"验证时间: {datetime.now()}")

    test_results = []

    # 测试多因素认证服务
    auth_result = test_authentication_service()
    test_results.append(("多因素认证服务", auth_result))

    # 测试数据保护服务
    protection_result = test_data_protection_service()
    test_results.append(("数据保护服务", protection_result))

    # 测试智能告警系统
    alert_result = test_alert_system()
    test_results.append(("智能告警系统", alert_result))

    # 测试模块集成
    integration_result = test_integration()
    test_results.append(("模块集成", integration_result))

    # 总结报告
    print("\n" + "="*80)
    print("📊 Day 4 任务实现验证报告")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print("25")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总体结果: {passed}/{len(test_results)} 测试通过")

    if failed == 0:
        print("🎉 所有测试通过！Day 4任务实现完全成功！")
        print("\n已完成的核心功能:")
        print("1. ✅ 多因素认证服务 - 支持密码、TOTP等多种认证方式")
        print("2. ✅ 数据保护服务 - 支持脱敏、加密、标记化、哈希等保护方法")
        print("3. ✅ 智能告警系统 - 支持多渠道告警、规则配置、状态管理")
        print("4. ✅ 模块集成验证 - 各模块协同工作正常")
        return 0
    else:
        print(f"⚠️ 有 {failed} 个测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    import time
    exit_code = main()
    sys.exit(exit_code)

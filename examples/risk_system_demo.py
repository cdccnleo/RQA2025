#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风控合规层演示脚本

展示实时风险监控、合规检查和智能预警系统的功能
"""

from src.risk import (
    RiskManager,
    RiskManagerConfig
)
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def demo_risk_monitoring():
    """演示实时风险监控"""
    print("🔍 演示实时风险监控...")

    # 创建风控管理器
    config = RiskManagerConfig(
        monitoring_interval=2,  # 2秒监控间隔
        max_risk_score=0.8,
        enable_auto_intervention=True,
        enable_compliance_check=True,
        enable_alert_system=True
    )

    risk_manager = RiskManager(config)

    # 启动风控管理器
    print("启动风控管理器...")
    risk_manager.start()

    try:
        # 运行一段时间
        for i in range(10):
            print(f"\n📊 第 {i+1} 次监控检查:")

            # 获取风险摘要
            summary = risk_manager.get_risk_summary()
            print(f"  - 系统状态: {summary.status.value}")
            print(f"  - 风险分数: {summary.risk_score:.3f}")
            print(f"  - 活跃告警: {summary.active_alerts}")
            print(f"  - 合规检查: {summary.compliance_checks}")
            print(f"  - 系统健康: {summary.system_health}")

            # 获取系统健康状态
            health = risk_manager.get_system_health()
            print(f"  - 组件状态: {health['components']}")

            time.sleep(2)

        # 演示订单风险检查
        print("\n📋 演示订单风险检查...")
        test_orders = [
            {
                "order_id": "ORDER001",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 150.0,
                "side": "buy"
            },
            {
                "order_id": "ORDER002",
                "symbol": "GOOGL",
                "quantity": 10000,  # 大额订单
                "price": 2800.0,
                "side": "buy"
            }
        ]

        for order in test_orders:
            print(f"\n检查订单: {order['order_id']}")
            result, reasons, risk_score = risk_manager.check_order_risk(order)

            if result:
                print(f"  ✅ 订单通过风险检查 (风险分数: {risk_score:.3f})")
            else:
                print(f"  ❌ 订单被拒绝 (风险分数: {risk_score:.3f})")
                for reason in reasons:
                    print(f"    - {reason}")

        # 演示预警系统
        print("\n🚨 演示预警系统...")

        # 模拟高风险数据
        high_risk_data = {
            'risk_score': 0.9,
            'position_ratio': 0.9,
            'volatility': 0.4,
            'liquidity_ratio': 0.05,
            'error_rate': 0.1,
            'response_time': 10.0
        }

        # 检查预警
        alerts = risk_manager.alert_system.check_alerts(high_risk_data)
        if alerts:
            print(f"触发 {len(alerts)} 个预警:")
            for alert in alerts:
                print(f"  - [{alert.alert_level.value.upper()}] {alert.title}: {alert.message}")
        else:
            print("  ✅ 无预警触发")

        # 获取预警摘要
        alert_summary = risk_manager.alert_system.get_alert_summary()
        print(f"\n预警摘要:")
        print(f"  - 活跃预警: {alert_summary['total_active_alerts']}")
        print(f"  - 按级别统计: {dict(alert_summary['alerts_by_level'])}")
        print(f"  - 按类型统计: {dict(alert_summary['alerts_by_type'])}")

        # 演示合规检查
        print("\n📋 演示合规检查...")

        # 获取合规检查记录
        compliance_checks = risk_manager.get_compliance_checks(limit=5)
        if compliance_checks:
            print(f"最近 {len(compliance_checks)} 次合规检查:")
            for check in compliance_checks:
                status_icon = "✅" if check.status.value == "passed" else "❌"
                print(f"  {status_icon} {check.rule_id}: {check.message}")
        else:
            print("  📝 暂无合规检查记录")

        # 演示风险指标
        print("\n📊 演示风险指标...")

        risk_metrics = risk_manager.get_risk_metrics(limit=5)
        if risk_metrics:
            print(f"最近 {len(risk_metrics)} 个风险指标:")
            for metric in risk_metrics:
                level_icon = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴"
                }.get(metric.risk_level.value, "⚪")
                print(f"  {level_icon} {metric.metric_name}: {metric.value:.3f} ({metric.risk_level.value})")
        else:
            print("  📝 暂无风险指标记录")

    finally:
        # 停止风控管理器
        print("\n停止风控管理器...")
        risk_manager.stop()
        print("✅ 演示完成")


def demo_alert_system():
    """演示预警系统"""
    print("\n🚨 演示预警系统...")

    from src.risk import AlertSystem, AlertRule, AlertType, AlertLevel

    alert_system = AlertSystem()

    # 添加自定义预警规则
    custom_rule = AlertRule(
        rule_id="demo_rule_001",
        rule_name="演示预警规则",
        alert_type=AlertType.RISK_THRESHOLD,
        alert_level=AlertLevel.WARNING,
        conditions={"demo_value": 0.5},
        actions=["email", "log"],
        cooldown_minutes=1
    )

    alert_system.add_alert_rule(custom_rule)
    print("✅ 添加自定义预警规则")

    # 测试预警触发
    test_data = {"demo_value": 0.8}
    alerts = alert_system.check_alerts(test_data)

    if alerts:
        print(f"✅ 触发 {len(alerts)} 个预警:")
        for alert in alerts:
            print(f"  - [{alert.alert_level.value.upper()}] {alert.title}: {alert.message}")
    else:
        print("  📝 无预警触发")

    # 获取预警摘要
    summary = alert_system.get_alert_summary()
    print(f"\n预警摘要: {summary['total_active_alerts']} 个活跃预警")


def demo_compliance_checker():
    """演示合规检查"""
    print("\n📋 演示合规检查...")

    from src.risk import ComplianceChecker, ComplianceRule, ComplianceRuleType

    checker = ComplianceChecker()

    # 添加自定义合规规则
    custom_rule = ComplianceRule(
        rule_id="demo_compliance_001",
        rule_name="演示合规规则",
        rule_type=ComplianceRuleType.CUSTOM_RULE,
        description="演示用的合规规则",
        parameters={"max_amount": 10000},
        enabled=True,
        priority=1
    )

    checker.add_rule(custom_rule)
    print("✅ 添加自定义合规规则")

    # 测试合规检查
    test_order = {
        "order_id": "DEMO001",
        "symbol": "DEMO",
        "quantity": 100,
        "price": 100.0
    }

    checks = checker.check_order(test_order)
    print(f"✅ 执行合规检查，共 {len(checks)} 项检查")

    for check in checks:
        status_icon = "✅" if check.status.value == "passed" else "❌"
        print(f"  {status_icon} {check.rule_id}: {check.message}")


def main():
    """主函数"""
    print("🎯 RQA2025 风控合规层演示")
    print("=" * 50)

    try:
        # 演示实时风险监控
        demo_risk_monitoring()

        # 演示预警系统
        demo_alert_system()

        # 演示合规检查
        demo_compliance_checker()

        print("\n🎉 所有演示完成！")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

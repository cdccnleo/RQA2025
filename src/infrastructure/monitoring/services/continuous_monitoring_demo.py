"""命令行演示入口，展示连续监控系统的典型用法。"""

import time

from .continuous_monitoring_service import (
    ContinuousMonitoringSystem,
    TestAutomationOptimizer,
)


def main() -> None:
    """运行演示流程。"""
    print("🚀 RQA2025 基础设施层连续监控和优化系统")
    print("=" * 60)

    monitoring_system = ContinuousMonitoringSystem()
    test_optimizer = TestAutomationOptimizer()

    try:
        _initialize_monitoring_system(monitoring_system)
        _display_monitoring_report(monitoring_system)
        _run_test_optimization(monitoring_system, test_optimizer)
        _show_system_capabilities()
        _run_demo_monitoring_loop(monitoring_system)
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号，正在停止监控系统...")
    except Exception as exc:  # pragma: no cover - 演示场景异常
        print(f"\n❌ 系统异常: {exc}")
    finally:
        monitoring_system.stop_monitoring()
        print("\n✅ 系统已停止，监控数据已写入 monitoring_data.json")


def _initialize_monitoring_system(monitoring_system: ContinuousMonitoringSystem) -> None:
    print("\n📊 执行完整监控周期...")
    monitoring_system.start_monitoring()
    monitoring_system._perform_monitoring_cycle()  # pragma: no cover - 演示辅助调用


def _display_monitoring_report(monitoring_system: ContinuousMonitoringSystem) -> None:
    report = monitoring_system.get_monitoring_report()
    print("\n📈 监控报告摘要:")
    print(f"• 监控状态: {'运行中' if report['monitoring_active'] else '已停止'}")
    print(f"• 收集的指标数: {report['total_metrics_collected']}")
    print(f"• 生成的告警数: {report['total_alerts_generated']}")
    print(f"• 优化建议数: {report['total_suggestions_generated']}")


def _run_test_optimization(
    monitoring_system: ContinuousMonitoringSystem,
    test_optimizer: TestAutomationOptimizer,
) -> None:
    print("\n🔧 执行测试优化...")
    optimizations = test_optimizer.optimize_test_execution()

    print("✅ 测试优化完成:")
    for opt_type, opt_config in optimizations.items():
        print(f"• {opt_type}: {opt_config.get('strategy', 'N / A')}")

    print("📄 导出监控报告...")
    monitoring_system.export_monitoring_report()


def _show_system_capabilities() -> None:
    print("\n🎉 Phase 7: 连续监控和优化系统部署完成!")
    print("=" * 60)
    print("📋 系统功能:")
    print("• 实时监控测试覆盖率、性能指标、资源使用")
    print("• 智能告警系统，及时发现问题")
    print("• 自动化优化建议生成")
    print("• 测试执行策略优化")
    print("• 持续的质量保障")
    print("=" * 60)
    print("\n⏳ 监控系统将继续运行，演示持续监控功能...")
    print("按 Ctrl+C 停止监控系统")


def _run_demo_monitoring_loop(monitoring_system: ContinuousMonitoringSystem) -> None:
    while True:  # pragma: no cover - 演示时手动中断
        time.sleep(60)
        if monitoring_system.metrics_history:
            latest = monitoring_system.metrics_history[-1]
            coverage = latest['data']['coverage'].get('coverage_percent', 0)
            print(
                f"📊 最新监控数据 - 覆盖率: {coverage:.1f}%, 时间: {latest['timestamp']}"
            )


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    main()

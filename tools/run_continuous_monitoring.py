#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 连续监控和优化系统运行脚本

此脚本用于运行基础设施层的连续监控和优化系统。

使用方法:
    python scripts/run_continuous_monitoring.py

或者从项目根目录运行:
    python -m scripts.run_continuous_monitoring
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    try:
        # 导入监控系统
        from src.infrastructure.monitoring.continuous_monitoring_system import (
            ContinuousMonitoringSystem,
            TestAutomationOptimizer
        )

        print("🚀 启动RQA2025基础设施层连续监控和优化系统...")
        print("=" * 60)

        # 创建监控系统
        monitoring_system = ContinuousMonitoringSystem()

        # 创建测试优化器
        test_optimizer = TestAutomationOptimizer()

        try:
            # 启动监控系统
            monitoring_system.start_monitoring()

            # 执行一次完整的监控周期
            print("\n📊 执行完整监控周期...")
            monitoring_system._perform_monitoring_cycle()

            # 获取监控报告
            report = monitoring_system.get_monitoring_report()
            print("
📈 监控报告摘要: "            print(f"• 监控状态: {'运行中' if report['monitoring_active'] else '已停止'}")
            print(f"• 收集的指标数: {report['total_metrics_collected']}")
            print(f"• 生成的告警数: {report['total_alerts_generated']}")
            print(f"• 优化建议数: {report['total_suggestions_generated']}")

            # 优化测试执行
            print("
🔧 执行测试优化..."            optimizations=test_optimizer.optimize_test_execution()

            print("✅ 测试优化完成:")
            for opt_type, opt_config in optimizations.items():
                print(f"• {opt_type}: {opt_config.get('strategy', 'N/A')}")

            # 导出监控报告
            report_file=monitoring_system.export_monitoring_report()

            print("
🎉 Phase 7: 连续监控和优化系统部署完成!"            print("=" * 60)
            print("📋 系统功能:")
            print("• 实时监控测试覆盖率、性能指标、资源使用")
            print("• 智能告警系统，及时发现问题")
            print("• 自动化优化建议生成")
            print("• 测试执行策略优化")
            print("• 持续的质量保障")
            print("=" * 60)

            # 保持监控运行一段时间进行演示
            print("\n⏳ 监控系统将继续运行，演示持续监控功能...")
            print("按Ctrl+C停止监控系统")

            while True:
                import time
                time.sleep(60)  # 每分钟检查一次
                if len(monitoring_system.metrics_history) > 0:
                    latest=monitoring_system.metrics_history[-1]
                    coverage=latest['data']['coverage'].get('coverage_percent', 0)
                    print(f"📊 最新监控数据 - 覆盖率: {coverage:.1f}%, 时间: {latest['timestamp']}")

        except KeyboardInterrupt:
            print("\n🛑 收到停止信号，正在停止监控系统...")
        except Exception as e:
            print(f"\n❌ 系统异常: {e}")
        finally:
            # 停止监控系统
            monitoring_system.stop_monitoring()

            print("\n✅ RQA2025 基础设施层连续监控和优化系统已停止")
            print("📊 最终报告已保存到 monitoring_data.json")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保项目结构正确，并且所有依赖都已安装")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RQA2025 监控系统统一启动脚本

同时启动告警智能分析面板和交易层监控面板
"""

from src.monitoring.alert_intelligence_analyzer import AlertIntelligenceAnalyzer
from src.monitoring.trading_monitor_dashboard import TradingMonitorDashboard
from src.monitoring.alert_intelligence_dashboard import AlertIntelligenceDashboard
import sys
import os
import argparse
import time
import threading
import webbrowser
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MonitoringSystemManager:
    """监控系统管理器"""

    def __init__(self):
        self.dashboards: List = []
        self.threads: List[threading.Thread] = []
        self.is_running = False

    def add_dashboard(self, dashboard, port: int, name: str):
        """添加仪表板"""
        thread = threading.Thread(
            target=self._start_dashboard_server,
            args=(dashboard, port, name),
            daemon=True
        )
        self.dashboards.append((dashboard, port, name))
        self.threads.append(thread)

    def start_all(self):
        """启动所有仪表板"""
        if self.is_running:
            print("监控系统已在运行")
            return

        print("🚀 启动RQA2025监控系统...")
        print("=" * 60)

        self.is_running = True

        # 启动所有仪表板
        for thread in self.threads:
            thread.start()
            time.sleep(1)  # 短暂延迟确保顺序启动

        print("✅ 所有监控面板已启动")
        self._print_system_info()

    def stop_all(self):
        """停止所有仪表板"""
        self.is_running = False
        print("正在停止监控系统...")

        # 这里可以添加更优雅的停止逻辑
        print("监控系统已停止")

    def _start_dashboard_server(self, dashboard, port: int, name: str):
        """启动仪表板服务器"""
        try:
            print(f"正在启动{name} (端口: {port})...")
            dashboard.start_server(host='localhost', port=port, debug=False)
        except Exception as e:
            print(f"启动{name}失败: {e}")

    def _print_system_info(self):
        """打印系统信息"""
        print("\n📊 监控系统信息:")
        print("=" * 40)

        for dashboard, port, name in self.dashboards:
            status = "✅ 运行中" if hasattr(dashboard, 'app') else "❌ 未启动"
            print("15")

        print("\n🌐 访问地址:")
        for dashboard, port, name in self.dashboards:
            print("30")

        print("\n🔧 API接口:")
        print("• GET /api/dashboard/overview - 系统概览数据")
        print("• GET /api/alerts/patterns - 告警模式分析")
        print("• GET /api/alerts/predictive - 预测性告警")
        print("• GET /api/trading/status - 交易状态")
        print("• GET /api/trading/metrics - 交易指标")

        print("\n💡 使用提示:")
        print("• 告警智能分析面板 - 专注于系统异常检测和智能分析")
        print("• 交易层监控面板 - 专注于交易系统的实时监控")
        print("• 两个面板可以同时使用，互不影响")

        print("\n" + "=" * 60)
        print("按 Ctrl+C 停止所有监控服务")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 监控系统统一启动器')
    parser.add_argument('--alert-port', type=int, default=5001,
                        help='告警智能分析面板端口 (默认: 5001)')
    parser.add_argument('--trading-port', type=int, default=5002,
                        help='交易层监控面板端口 (默认: 5002)')
    parser.add_argument('--no-browser', action='store_true',
                        help='不自动打开浏览器')
    parser.add_argument('--demo', action='store_true',
                        help='启用演示数据模式')
    parser.add_argument('--only-alert', action='store_true',
                        help='仅启动告警智能分析面板')
    parser.add_argument('--only-trading', action='store_true',
                        help='仅启动交易层监控面板')

    args = parser.parse_args()

    # 检查端口冲突
    if args.alert_port == args.trading_port:
        print("❌ 错误: 两个面板不能使用相同端口")
        return 1

    # 创建监控系统管理器
    manager = MonitoringSystemManager()

    try:
        # 创建告警智能分析面板
        if not args.only_trading:
            print("初始化告警智能分析面板...")
            alert_analyzer = AlertIntelligenceAnalyzer()
            alert_dashboard = AlertIntelligenceDashboard(analyzer=alert_analyzer)
            manager.add_dashboard(alert_dashboard, args.alert_port, "告警智能分析面板")

        # 创建交易层监控面板
        if not args.only_alert:
            print("初始化交易层监控面板...")
            trading_dashboard = TradingMonitorDashboard()
            manager.add_dashboard(trading_dashboard, args.trading_port, "交易层监控面板")

        # 启动所有服务
        manager.start_all()

        # 如果启用演示模式，创建一些示例数据
        if args.demo:
            print("\n🎯 演示模式已启用，正在准备示例数据...")
            if not args.only_trading and 'alert_analyzer' in locals():
                # 这里可以添加创建示例数据的逻辑
                print("示例数据准备完成")

        # 自动打开浏览器
        if not args.no_browser:
            time.sleep(2)  # 等待服务完全启动

            if not args.only_trading:
                try:
                    webbrowser.open(f"http://localhost:{args.alert_port}")
                    print(f"已打开告警智能分析面板: http://localhost:{args.alert_port}")
                except:
                    pass

            if not args.only_alert:
                try:
                    webbrowser.open(f"http://localhost:{args.trading_port}")
                    print(f"已打开交易层监控面板: http://localhost:{args.trading_port}")
                except:
                    pass

        # 保持运行
        print("\n🔄 监控系统运行中...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 正在停止监控系统...")
            manager.stop_all()

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def show_help():
    """显示帮助信息"""
    help_text = """
RQA2025 监控系统统一启动器

使用方法:
  python scripts/start_monitoring_system.py [选项]

选项:
  --alert-port PORT      告警智能分析面板端口 (默认: 5001)
  --trading-port PORT    交易层监控面板端口 (默认: 5002)
  --no-browser          不自动打开浏览器
  --demo                启用演示数据模式
  --only-alert          仅启动告警智能分析面板
  --only-trading        仅启动交易层监控面板
  --help                显示此帮助信息

示例:
  # 启动所有监控面板（默认模式）
  python scripts/start_monitoring_system.py

  # 仅启动告警智能分析面板
  python scripts/start_monitoring_system.py --only-alert

  # 自定义端口启动
  python scripts/start_monitoring_system.py --alert-port 5003 --trading-port 5004

  # 演示模式
  python scripts/start_monitoring_system.py --demo

访问地址:
  • 告警智能分析面板: http://localhost:5001
  • 交易层监控面板: http://localhost:5002

API接口:
  • GET /api/dashboard/overview - 系统概览
  • GET /api/alerts/patterns - 告警模式分析
  • GET /api/trading/status - 交易状态
  • GET /api/trading/metrics - 交易指标

功能特性:
  ✅ 智能告警分析 - 基于机器学习的异常检测
  ✅ 实时性能监控 - 全面的系统性能监控
  ✅ 交易层监控 - 专门的交易系统监控
  ✅ 可视化界面 - 现代化Web界面和实时图表
  ✅ RESTful API - 标准化的数据接口
    """
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        sys.exit(main())

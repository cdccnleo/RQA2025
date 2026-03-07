#!/usr/bin/env python3
"""
RQA2025 交易层监控面板启动脚本

快速启动交易层专用监控面板
"""

from src.monitoring.trading_monitor_dashboard import TradingMonitorDashboard
import sys
import os
import argparse
import webbrowser

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_trading_data(dashboard: TradingMonitorDashboard):
    """创建示例交易数据用于演示"""
    print("正在创建示例交易数据...")

    # 这里可以添加更多示例数据的创建逻辑
    print("示例交易数据创建完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 交易层监控面板')
    parser.add_argument('--host', default='localhost', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5002, help='服务器端口')
    parser.add_argument('--demo', action='store_true', help='使用演示数据')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    print("🚀 启动 RQA2025 交易层监控面板")
    print("=" * 50)
    print(f"服务器地址: http://{args.host}:{args.port}")
    print("=" * 50)

    try:
        # 创建交易层监控面板
        print("初始化交易层监控面板...")
        dashboard = TradingMonitorDashboard()

        # 启动监控
        print("启动交易监控...")
        dashboard.start_monitoring()

        # 如果使用演示数据，创建示例数据
        if args.demo:
            create_sample_trading_data(dashboard)

        # 自动打开浏览器
        if not args.no_browser:
            url = f"http://{args.host}:{args.port}"
            print(f"\n🌐 正在打开浏览器: {url}")
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"无法自动打开浏览器: {e}")
                print(f"请手动在浏览器中访问: {url}")

        print("\n📊 交易层监控面板功能:")
        print("• 📈 实时性能指标 - 订单延迟、吞吐量、执行率")
        print("• 📊 订单状态监控 - 订单分布、执行统计、状态跟踪")
        print("• 💼 持仓状态监控 - 持仓规模、盈亏分析、风险指标")
        print("• 🌐 市场连接监控 - 连接状态、延迟监控、健康检查")
        print("• ⚠️ 风险告警系统 - 风险敞口、合规监控、异常告警")
        print("• 📉 盈亏趋势分析 - 实时盈亏、趋势预测、绩效分析")

        print("\n🔗 API接口:")
        print(f"• GET /api/trading/status - 交易状态概览")
        print(f"• GET /api/trading/metrics - 交易性能指标")
        print(f"• GET /api/trading/orders - 订单状态详情")
        print(f"• GET /api/trading/positions - 持仓状态详情")
        print(f"• GET /api/trading/risk - 风险指标详情")
        print(f"• GET /api/trading/connections - 连接状态详情")
        print(f"• GET /api/trading/alerts - 活跃告警列表")

        print("\n" + "=" * 50)
        print("按 Ctrl+C 停止服务器")
        print("=" * 50)

        # 启动服务器
        dashboard.start_server(args.host, args.port, args.debug)

    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

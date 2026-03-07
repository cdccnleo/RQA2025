#!/usr/bin/env python3
"""
RQA2025 告警智能分析可视化仪表板启动脚本

快速启动告警智能分析可视化仪表板
"""

from src.monitoring.alert_intelligence_dashboard import AlertIntelligenceDashboard
from src.monitoring.alert_intelligence_analyzer import AlertIntelligenceAnalyzer
import sys
import os
import argparse
import webbrowser

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_data(analyzer: AlertIntelligenceAnalyzer):
    """创建示例数据用于演示"""
    print("正在创建示例数据...")

    import random
    from datetime import datetime, timedelta

    # 创建一些示例指标数据
    metrics = {
        "system.cpu.usage": {"base": 50, "range": 20},
        "system.memory.usage": {"base": 60, "range": 25},
        "app.response_time": {"base": 150, "range": 100},
        "system.disk.usage": {"base": 45, "range": 15},
        "network.latency": {"base": 20, "range": 30}
    }

    base_time = datetime.now() - timedelta(hours=24)

    for metric_name, config in metrics.items():
        print(f"生成 {metric_name} 数据...")

        for i in range(1000):  # 1000个数据点
            timestamp = base_time + timedelta(minutes=i * 1.44)  # 约24小时

            # 生成基础值
            base_value = config["base"]
            noise = random.uniform(-config["range"], config["range"])
            trend = (i / 1000) * 5  # 轻微趋势

            value = base_value + noise + trend

            # 添加一些异常
            if random.random() < 0.02:  # 2%的概率出现异常
                if metric_name == "app.response_time":
                    value *= 5  # 响应时间异常
                else:
                    value = min(100, value * 1.5)  # 其他指标异常

            # 确保值在合理范围内
            if "usage" in metric_name or "disk" in metric_name:
                value = max(0, min(100, value))
            elif "response_time" in metric_name or "latency" in metric_name:
                value = max(1, value)

            # 添加到分析器
            analyzer.process_metric(
                metric_name=metric_name,
                value=value,
                timestamp=timestamp,
                labels={
                    "source": "demo",
                    "component": metric_name.split(".")[1],
                    "server": "demo-server"
                }
            )

    print("示例数据创建完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 告警智能分析可视化仪表板')
    parser.add_argument('--host', default='localhost', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5001, help='服务器端口')
    parser.add_argument('--demo', action='store_true', help='使用演示数据')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    print("🚀 启动 RQA2025 告警智能分析可视化仪表板")
    print("=" * 50)
    print(f"服务器地址: http://{args.host}:{args.port}")
    print("=" * 50)

    try:
        # 创建分析器
        print("初始化告警智能分析器...")
        analyzer = AlertIntelligenceAnalyzer()

        # 如果使用演示数据，创建示例数据
        if args.demo:
            create_sample_data(analyzer)

        # 创建仪表板
        print("初始化可视化仪表板...")
        dashboard = AlertIntelligenceDashboard(
            analyzer=analyzer,
            host=args.host,
            port=args.port
        )

        # 如果没有演示数据，显示使用提示
        if not args.demo:
            print("\n💡 提示:")
            print("当前没有示例数据，您可以:")
            print("1. 使用 --demo 参数加载示例数据")
            print("2. 通过API接口添加真实数据")
            print("3. 查看静态报告功能")

        # 自动打开浏览器
        if not args.no_browser:
            url = f"http://{args.host}:{args.port}"
            print(f"\n🌐 正在打开浏览器: {url}")
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"无法自动打开浏览器: {e}")
                print(f"请手动在浏览器中访问: {url}")

        print("\n📊 仪表板功能:")
        print("• 系统概览 - 实时健康状态和关键指标")
        print("• 告警模式分析 - 智能识别异常模式")
        print("• 预测性告警 - 提前预测潜在问题")
        print("• 根因分析 - 深度分析问题根源")
        print("• 告警关联 - 发现告警之间的关联")
        print("• 实时图表 - 动态数据可视化")

        print("\n⚡ API接口:")
        print(f"• GET /api/dashboard/overview - 仪表板概览数据")
        print(f"• GET /api/alerts/patterns - 告警模式分析")
        print(f"• GET /api/alerts/predictive - 预测性告警")
        print(f"• GET /api/alerts/correlation - 告警关联分析")

        print("\n" + "=" * 50)
        print("按 Ctrl+C 停止服务器")
        print("=" * 50)

        # 启动服务器
        dashboard.start_server(debug=args.debug)

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

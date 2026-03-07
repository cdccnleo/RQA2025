"""
特征层监控集成演示

演示如何将监控体系集成到特征层组件中。
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.feature_engineer import FeatureEngineer
from src.features.technical.technical_processor import TechnicalProcessor
from src.features.monitoring import (
    MonitoringIntegrationManager,
    get_monitor,
    get_persistence_manager,
    get_dashboard
)


def create_sample_data():
    """创建示例数据"""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    stock_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(150, 250, 50),
        'low': np.random.uniform(50, 150, 50),
        'close': np.random.uniform(100, 200, 50),
        'volume': np.random.uniform(1000, 10000, 50)
    }, index=dates)
    return stock_data


def demo_monitoring_integration():
    """演示监控集成"""
    print("🚀 特征层监控集成演示")

    # 1. 创建监控集成管理器
    integration_manager = MonitoringIntegrationManager()

    # 2. 创建特征层组件
    feature_engineer = FeatureEngineer()
    technical_processor = TechnicalProcessor()

    # 3. 集成组件到监控体系
    integration_manager.integrate_component(feature_engineer, 'FeatureEngineer')
    integration_manager.integrate_component(technical_processor, 'TechnicalProcessor')

    # 4. 启动监控
    monitor = get_monitor()
    monitor.start_monitoring()

    # 5. 执行特征工程操作
    stock_data = create_sample_data()

    print("执行特征工程操作...")
    technical_features = feature_engineer.generate_technical_features(stock_data)

    print("执行技术指标计算...")
    rsi_result = technical_processor.calculate_rsi(stock_data)

    # 6. 等待监控数据收集
    time.sleep(3)

    # 7. 获取监控状态
    integration_status = integration_manager.get_integration_status()
    print(f"集成组件数: {integration_status['integrated_components']}")
    print(f"监控状态: {'运行中' if integration_status['monitor_status'] else '已停止'}")

    # 8. 导出集成报告
    report_path = "reports/monitoring_integration_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    integration_manager.export_integration_report(report_path)
    print(f"集成报告已导出到: {report_path}")

    # 9. 停止监控
    monitor.stop_monitoring()

    print("✅ 监控集成演示完成！")


def demo_persistence_and_dashboard():
    """演示持久化和面板"""
    print("\n📊 指标持久化和监控面板演示")

    # 1. 创建持久化管理器
    persistence_manager = get_persistence_manager({
        'backend': 'sqlite',
        'path': './monitoring_data'
    })

    # 2. 存储示例指标
    from src.features.monitoring import MetricType
    persistence_manager.store_metric(
        'FeatureEngineer_123', 'feature_generation_time', 2.5, MetricType.HISTOGRAM)
    persistence_manager.store_metric('TechnicalProcessor_456',
                                     'indicator_calculation_time', 1.2, MetricType.HISTOGRAM)

    time.sleep(2)

    # 3. 查询指标数据
    df = persistence_manager.query_metrics(limit=5)
    print(f"查询到 {len(df)} 条指标记录")

    # 4. 创建监控面板
    dashboard = get_dashboard({
        'title': '特征层监控面板',
        'refresh_interval': 5.0
    })

    # 5. 启动面板
    dashboard.start_dashboard(auto_open=False)
    print("监控面板已启动: ./monitoring_dashboard/dashboard.html")

    time.sleep(5)
    dashboard.stop_dashboard()

    print("✅ 持久化和面板演示完成！")


def main():
    """主函数"""
    try:
        demo_monitoring_integration()
        demo_persistence_and_dashboard()

        print("\n🎉 演示完成！")
        print("生成的文件:")
        print("- reports/monitoring_integration_report.json")
        print("- ./monitoring_dashboard/dashboard.html")
        print("- ./monitoring_data/metrics.db")

    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()

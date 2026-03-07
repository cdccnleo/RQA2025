#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQA2025 监控系统演示脚本

演示监控系统的各项功能：
- 实时指标收集
- 告警触发和通知
- Web界面访问
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.core.real_time_monitor import get_monitor, update_business_metric
from src.monitoring.alert.alert_notifier import NotificationConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def simulate_business_operations():
    """模拟业务操作，生成各种指标"""
    logger.info("Starting business operations simulation...")

    monitor = get_monitor()

    # 模拟交易请求
    for i in range(100):
        # 模拟请求
        update_business_metric('request', 1)

        # 模拟响应时间 (10-100ms随机)
        response_time = 10 + (i % 90)
        update_business_metric('response_time', response_time)

        # 偶尔模拟错误
        if i % 15 == 0:
            update_business_metric('error', 1)

        time.sleep(0.1)  # 每100ms一个请求

        if i % 20 == 0:
            logger.info(f"Processed {i+1} requests")

    logger.info("Business operations simulation completed")


def simulate_system_load():
    """模拟系统负载，触发告警"""
    logger.info("Starting system load simulation...")

    # 创建一些CPU密集型操作来触发告警
    def cpu_intensive_task():
        result = 0
        for i in range(10000000):  # CPU密集计算
            result += i ** 2
        return result

    # 在多个线程中运行CPU密集任务
    threads = []
    for i in range(4):  # 4个并发线程
        thread = threading.Thread(target=cpu_intensive_task)
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    logger.info("System load simulation completed")


def display_monitoring_status():
    """显示监控状态"""
    monitor = get_monitor()

    logger.info("=== 监控系统状态 ===")

    # 系统状态
    status = monitor.get_system_status()
    logger.info(f"系统健康状态: {status['system_health']}")
    logger.info(f"活跃告警数量: {status['active_alerts']}")
    logger.info(f"收集的指标数量: {status['metrics_count']}")

    # 当前指标
    metrics = monitor.get_current_metrics()
    logger.info("\n=== 当前关键指标 ===")
    key_metrics = ['cpu_percent', 'memory_percent', 'requests_total', 'avg_response_time_ms', 'errors_total']
    for metric_name in key_metrics:
        if metric_name in metrics:
            value = metrics[metric_name].value
            logger.info(f"{metric_name}: {value}")

    # 告警摘要
    alerts = monitor.get_alerts_summary()
    logger.info(f"\n=== 告警状态 ===")
    logger.info(f"活跃告警: {alerts['active_count']}")
    logger.info(f"最近告警: {alerts['recent_count']}")

    if alerts['active_alerts']:
        logger.info("活跃告警详情:")
        for alert in alerts['active_alerts']:
            logger.info(f"  - {alert['rule_name']}: {alert['message']}")


def main():
    """主演示函数"""
    logger.info("🚀 RQA2025 监控系统演示开始")
    logger.info("=" * 50)

    # 启动监控系统
    monitor = get_monitor()
    monitor.start_monitoring()

    logger.info("✅ 监控系统已启动")
    logger.info("📊 Web界面访问地址: http://localhost:5000")
    logger.info("⏱️  等待系统初始化...")

    time.sleep(5)  # 等待系统初始化

    try:
        # 显示初始状态
        display_monitoring_status()

        # 阶段1: 正常业务操作
        logger.info("\n" + "="*50)
        logger.info("📈 阶段1: 模拟正常业务操作")
        logger.info("="*50)

        simulate_business_operations()
        time.sleep(2)

        display_monitoring_status()

        # 阶段2: 高负载测试
        logger.info("\n" + "="*50)
        logger.info("⚡ 阶段2: 模拟高负载情况（触发告警）")
        logger.info("="*50)

        simulate_system_load()
        time.sleep(5)  # 等待告警检测

        display_monitoring_status()

        # 阶段3: 告警恢复观察
        logger.info("\n" + "="*50)
        logger.info("🔄 阶段3: 观察告警恢复")
        logger.info("="*50)

        logger.info("等待告警自动恢复...")
        for i in range(30):  # 等待30秒
            time.sleep(1)
            if i % 10 == 0:
                status = monitor.get_system_status()
                logger.info(f"当前活跃告警: {status['active_alerts']}")

        display_monitoring_status()

        logger.info("\n" + "="*50)
        logger.info("🎉 监控系统演示完成！")
        logger.info("请访问 http://localhost:5000 查看Web监控面板")
        logger.info("按 Ctrl+C 停止监控系统")
        logger.info("="*50)

        # 保持运行，让用户可以访问Web界面
        while True:
            time.sleep(10)
            # 每10秒显示一次状态摘要
            status = monitor.get_system_status()
            logger.info(f"系统状态: {status['system_health']}, 活跃告警: {status['active_alerts']}")

    except KeyboardInterrupt:
        logger.info("\n🛑 收到中断信号，正在停止监控系统...")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
    finally:
        # 停止监控系统
        monitor.stop_monitoring()
        logger.info("✅ 监控系统已停止")


if __name__ == "__main__":
    main()

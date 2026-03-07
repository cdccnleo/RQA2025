#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云日志集成使用示例

演示如何使用RQA2025的云日志集成功能。
"""

from infrastructure.logging.cloud.cloud_aggregator import CloudProvider
from infrastructure.logging.cloud import (
    CloudConfigManager, CloudLogAggregator, MultiCloudLogger,
    CloudMetricsCollector, CloudPerformanceMonitor,
    CloudSecurityManager, CloudAuditLogger
)
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def setup_cloud_credentials(config_manager: CloudConfigManager):
    """设置云服务凭据"""
    print("=== 设置云服务凭据 ===")

    # AWS凭据（示例）
    aws_creds = config_manager.get_credentials('aws', 'us-east-1')
    if not aws_creds:
        print("添加AWS凭据...")
        # 注意：在实际使用中，请使用真实的凭据
        from infrastructure.logging.cloud.cloud_config import CloudCredentials
        aws_credentials = CloudCredentials(
            provider='aws',
            region='us-east-1',
            # access_key_id='YOUR_ACCESS_KEY',  # 实际使用时取消注释
            # secret_access_key='YOUR_SECRET_KEY'  # 实际使用时取消注释
        )
        config_manager.add_credentials(aws_credentials)

    # Azure凭据（示例）
    azure_creds = config_manager.get_credentials('azure')
    if not azure_creds:
        print("添加Azure凭据...")
        from infrastructure.logging.cloud.cloud_config import CloudCredentials
        azure_credentials = CloudCredentials(
            provider='azure',
            # instrumentation_key='YOUR_INSTRUMENTATION_KEY'  # 实际使用时取消注释
        )
        config_manager.add_credentials(azure_credentials)

    # GCP凭据（示例）
    gcp_creds = config_manager.get_credentials('gcp')
    if not gcp_creds:
        print("添加GCP凭据...")
        from infrastructure.logging.cloud.cloud_config import CloudCredentials
        gcp_credentials = CloudCredentials(
            provider='gcp',
            project_id='your-gcp-project',  # 替换为实际项目ID
            # credentials_file='/path/to/credentials.json'  # 实际使用时取消注释
        )
        config_manager.add_credentials(gcp_credentials)

    print(f"凭据配置完成，共 {len(config_manager.credentials_cache)} 个凭据")


def setup_cloud_loggers(config_manager: CloudConfigManager):
    """设置云日志记录器"""
    print("\n=== 设置云日志记录器 ===")

    # AWS CloudWatch配置
    aws_config = config_manager.get_logger_config('aws_logger')
    if not aws_config:
        print("创建AWS CloudWatch配置...")
        from infrastructure.logging.cloud.cloud_config import CloudLoggerConfig
        aws_config = CloudLoggerConfig(
            name='aws_logger',
            provider='aws',
            enabled=True,
            log_group='rqa2025-application',
            log_stream='main-stream',
            level='INFO',
            batch_size=10,
            flush_interval=30.0
        )
        config_manager.add_logger_config(aws_config)

    # Azure Monitor配置
    azure_config = config_manager.get_logger_config('azure_logger')
    if not azure_config:
        print("创建Azure Monitor配置...")
        from infrastructure.logging.cloud.cloud_config import CloudLoggerConfig
        azure_config = CloudLoggerConfig(
            name='azure_logger',
            provider='azure',
            enabled=True,
            log_group='application',
            log_stream='main',
            level='INFO'
        )
        config_manager.add_logger_config(azure_config)

    # GCP Logging配置
    gcp_config = config_manager.get_logger_config('gcp_logger')
    if not gcp_config:
        print("创建GCP Logging配置...")
        from infrastructure.logging.cloud.cloud_config import CloudLoggerConfig
        gcp_config = CloudLoggerConfig(
            name='gcp_logger',
            provider='gcp',
            enabled=True,
            log_group='application',
            log_stream='main',
            level='INFO'
        )
        config_manager.add_logger_config(gcp_config)

    print(f"日志记录器配置完成，共 {len(config_manager.logger_configs)} 个配置")


def create_multi_cloud_setup(config_manager: CloudConfigManager):
    """创建多云设置"""
    print("\n=== 创建多云日志设置 ===")

    # 创建聚合器
    aggregator = CloudLogAggregator(
        buffer_size=1000,
        flush_interval=60.0,  # 1分钟刷新一次
        max_batch_size=50
    )

    # 创建多云日志记录器
    multi_logger = MultiCloudLogger(aggregator)

    # 为每个配置的日志记录器创建实例
    for config_name in config_manager.logger_configs.keys():
        try:
            logger_instance = config_manager.create_logger_from_config(config_name)
            if logger_instance:
                provider = CloudProvider(
                    config_manager.logger_configs[config_name].provider.lower())
                aggregator.add_cloud_logger(provider, logger_instance)
                print(f"✓ 添加 {config_name} 到聚合器")
            else:
                print(f"⚠ 无法创建 {config_name} 的日志记录器（可能缺少凭据）")
        except Exception as e:
            print(f"✗ 创建 {config_name} 失败: {e}")

    # 启动聚合器
    aggregator.start()

    print(f"多云设置完成，活跃日志记录器: {len(aggregator.cloud_loggers)} 个")

    return aggregator, multi_logger


def setup_monitoring(aggregator: CloudLogAggregator):
    """设置监控"""
    print("\n=== 设置监控系统 ===")

    # 创建指标收集器
    metrics_collector = CloudMetricsCollector(collection_interval=30.0)

    # 添加云客户端到指标收集器
    for provider, logger_instance in aggregator.cloud_loggers.items():
        metrics_collector.add_cloud_client(provider.value, logger_instance)

    # 创建默认指标收集器
    metrics_collector.create_default_collectors()

    # 启动指标收集
    metrics_collector.start_collection()

    # 创建性能监控器
    performance_monitor = CloudPerformanceMonitor(metrics_collector)

    # 添加告警回调
    def alert_callback(alert_type, alert_data):
        print(f"🚨 云性能告警: {alert_type} - {alert_data['message']}")

    performance_monitor.add_alert_callback(alert_callback)

    print("监控系统设置完成")

    return metrics_collector, performance_monitor


def setup_security():
    """设置安全管理"""
    print("\n=== 设置安全管理 ===")

    # 创建安全管理器
    security_manager = CloudSecurityManager()

    # 创建审计日志记录器
    audit_logger = CloudAuditLogger(security_manager)

    # 添加安全事件回调
    def security_callback(event):
        print(f"🔒 安全事件: {event.event_type} - {event.description}")

    security_manager.add_security_callback(security_callback)

    print("安全管理系统设置完成")

    return security_manager, audit_logger


def simulate_cloud_logging(multi_logger: MultiCloudLogger,
                           security_manager: CloudSecurityManager,
                           audit_logger: CloudAuditLogger,
                           duration_minutes: int = 1):
    """模拟云日志记录"""
    print(f"\n=== 模拟云日志记录 ({duration_minutes}分钟) ===")

    # 模拟的日志消息
    log_messages = [
        ("INFO", "用户登录系统", "user_auth", "login"),
        ("WARNING", "检测到异常流量", "security", "alert"),
        ("ERROR", "数据库连接失败", "database", "error"),
        ("INFO", "订单处理完成", "business", "order"),
        ("DEBUG", "缓存命中率统计", "performance", "cache"),
        ("INFO", "API请求处理", "api", "request"),
        ("WARNING", "内存使用率过高", "system", "memory"),
        ("INFO", "数据同步完成", "data", "sync"),
    ]

    start_time = time.time()
    log_count = 0

    try:
        while time.time() - start_time < duration_minutes * 60:
            # 随机选择一条日志消息
            level, message, logger_name, action = log_messages[log_count % len(log_messages)]

            # 模拟用户会话
            session_result = security_manager.authenticate_request({
                'user_id': f'user_{log_count % 5}',
                'password': 'test_password'
            })

            if session_result['authenticated']:
                session_id = session_result['session_id']

                # 授权检查
                if security_manager.authorize_action(session_id, action, logger_name):
                    # 数据访问审计
                    if audit_logger.security_manager.audit_data_access(session_id, 'logs', 'write', 1):
                        # 记录日志到云服务
                        multi_logger.log_to_cloud(
                            logger_name=logger_name,
                            level=level,
                            message=f"{message} (#{log_count + 1})",
                            log_group="rqa2025-demo",
                            log_stream="simulation",
                            metadata={
                                'session_id': session_id,
                                'timestamp': time.time(),
                                'simulation': True
                            }
                        )

                        log_count += 1

                        # 每10条日志打印一次进度
                        if log_count % 10 == 0:
                            print(f"已记录 {log_count} 条日志...")

            time.sleep(0.1)  # 短暂延迟

    except KeyboardInterrupt:
        print("日志记录模拟被用户中断")

    print(f"云日志记录模拟完成，共记录 {log_count} 条日志")


def monitor_performance(metrics_collector: CloudMetricsCollector,
                        performance_monitor: CloudPerformanceMonitor):
    """监控性能"""
    print("\n=== 性能监控演示 ===")

    # 等待一些指标数据
    print("等待指标数据收集...")
    time.sleep(10)

    # 获取指标历史
    recent_metrics = metrics_collector.get_metrics_history(limit=20)
    print(f"收集到 {len(recent_metrics)} 个指标数据点")

    # 执行性能检查
    check_results = performance_monitor.check_performance()

    print("性能检查结果:")
    print(f"  检查项目: {len(check_results.get('checks', {}))}")
    print(f"  触发告警: {len(check_results.get('alerts', []))}")

    for alert in check_results.get('alerts', []):
        print(f"    🚨 {alert['type']}: {alert['message']}")

    print(f"  优化建议: {len(check_results.get('recommendations', []))}")
    for rec in check_results.get('recommendations', []):
        print(f"    💡 {rec}")

    # 获取统计信息
    collector_stats = metrics_collector.get_collection_stats()
    monitor_stats = performance_monitor.get_monitor_stats()

    print(f"\n收集器统计: {collector_stats['total_metrics_collected']} 个指标")
    print(f"监控器统计: {monitor_stats['performance_checks']} 次检查")


def demonstrate_security_audit(security_manager: CloudSecurityManager,
                               audit_logger: CloudAuditLogger):
    """演示安全审计"""
    print("\n=== 安全审计演示 ===")

    # 获取安全事件
    recent_events = security_manager.get_security_events(limit=10)
    print(f"最近安全事件: {len(recent_events)} 个")

    for event in recent_events[-3:]:  # 显示最近3个
        print(f"  {event.event_type}: {event.description} ({event.severity})")

    # 获取审计追踪
    audit_trail = audit_logger.get_audit_trail(limit=10)
    print(f"\n审计追踪记录: {len(audit_trail)} 个")

    for entry in audit_trail[-3:]:  # 显示最近3个
        print(f"  {entry['action']} on {entry['resource']} by {entry['user']} - {entry['result']}")

    # 生成审计报告
    report = audit_logger.generate_audit_report()
    print(f"\n审计报告摘要:")
    print(
        f"  报告周期: {report['report_period']['start_time']} 至 {report['report_period']['end_time']}")
    print(f"  总事件数: {report['total_events']}")
    print(f"  成功率: {report['summary'].get('success_rate', 'N/A')}")
    print(f"  最活跃用户: {report['summary'].get('most_active_user', 'N/A')}")


def main():
    """主函数"""
    print("RQA2025 云日志集成演示")
    print("=" * 50)

    # 创建配置管理器
    config_manager = CloudConfigManager()

    try:
        # 设置云服务凭据和配置
        setup_cloud_credentials(config_manager)
        setup_cloud_loggers(config_manager)

        # 创建多云日志设置
        aggregator, multi_logger = create_multi_cloud_setup(config_manager)

        # 设置监控和安全
        metrics_collector, performance_monitor = setup_monitoring(aggregator)
        security_manager, audit_logger = setup_security()

        # 运行演示
        simulate_cloud_logging(multi_logger, security_manager, audit_logger, duration_minutes=1)
        monitor_performance(metrics_collector, performance_monitor)
        demonstrate_security_audit(security_manager, audit_logger)

        print("\n=== 云日志集成演示完成 ===")
        print("RQA2025的云日志集成系统展示了以下特性:")
        print("• 多云平台日志聚合 (AWS/Azure/GCP)")
        print("• 统一配置管理和凭据管理")
        print("• 实时性能监控和智能告警")
        print("• 企业级安全管理和审计追踪")
        print("• 批量处理和异步发送优化")
        print("• 容错设计和降级处理")

    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            if 'aggregator' in locals():
                aggregator.stop()
            if 'metrics_collector' in locals():
                metrics_collector.stop_collection()
        except Exception as e:
            print(f"清理资源时出错: {e}")

    print("\n云日志集成演示完成")


if __name__ == "__main__":
    main()

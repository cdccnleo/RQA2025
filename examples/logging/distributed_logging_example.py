#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式Logger使用示例

演示分布式Logger系统的完整功能。
"""

from infrastructure.logging.distributed import (
    DistributedLogCoordinator,
    ConsulServiceDiscovery,
    DistributedConfigSync,
    AdaptiveLoadBalancer
)
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def create_distributed_logger(node_id: str, host: str = "localhost", port: int = 8080):
    """创建分布式Logger实例"""
    print(f"创建分布式Logger节点: {node_id}")

    # 创建服务发现
    discovery = ConsulServiceDiscovery(
        consul_host="localhost",
        consul_port=8500,
        service_name="distributed-logger"
    )

    # 创建配置同步器
    config_sync = DistributedConfigSync(
        node_id=node_id,
        sync_interval=30.0
    )

    # 创建自适应负载均衡器
    load_balancer = AdaptiveLoadBalancer(
        nodes=[node_id],  # 初始只包含自己
        adaptation_interval=60.0
    )

    # 创建分布式协调器
    coordinator = DistributedLogCoordinator(
        node_id=node_id,
        host=host,
        port=port,
        service_discovery=discovery,
        config_sync=config_sync,
        load_balancer=load_balancer
    )

    return coordinator


def simulate_distributed_logging(coordinator: DistributedLogCoordinator):
    """模拟分布式日志记录"""
    print(f"\n开始分布式日志记录模拟 (节点: {coordinator.node_id})")

    # 模拟不同类型的日志
    log_entries = [
        ("INFO", "系统启动", "SYSTEM", {"version": "1.0.0"}),
        ("INFO", "用户登录", "BUSINESS", {"user_id": "user123", "ip": "192.168.1.1"}),
        ("WARNING", "高频交易检测", "TRADING", {"symbol": "AAPL", "volume": 10000}),
        ("ERROR", "数据库连接失败", "DATABASE", {"db_host": "db.example.com", "error": "timeout"}),
        ("INFO", "风险评估完成", "RISK", {"risk_score": 0.85, "action": "monitor"}),
        ("DEBUG", "性能指标收集", "PERFORMANCE", {"cpu": 45.2, "memory": 67.8}),
    ]

    for level, message, category, metadata in log_entries:
        success = coordinator.submit_log_entry(
            level=level,
            message=message,
            category=category,
            metadata=metadata,
            trace_id=f"trace-{int(time.time())}",
            span_id=f"span-{coordinator.node_id}"
        )

        status = "✓" if success else "✗"
        print(f"{status} {level}: {message} ({category})")

        time.sleep(0.1)  # 短暂延迟

    print("分布式日志记录模拟完成")


def monitor_system_status(coordinator: DistributedLogCoordinator):
    """监控系统状态"""
    print(f"\n系统状态监控 (节点: {coordinator.node_id})")

    status = coordinator.get_system_status()

    print("=== 总体状态 ===")
    print(f"运行状态: {'运行中' if status['running'] else '已停止'}")
    print(f"协调统计: {status['coordination_stats']['total_coordinations']} 次协调")

    print("\n=== 组件状态 ===")

    if 'aggregator' in status['components']:
        agg_stats = status['components']['aggregator']['stats']
        print(f"日志聚合器: {agg_stats['total_entries']} 条日志")

    if 'cluster' in status['components']:
        cluster_stats = status['components']['cluster']['stats']
        print(f"集群: {cluster_stats['total_nodes']} 节点, {cluster_stats['active_nodes']} 活跃")

    if 'discovery' in status['components']:
        disc_stats = status['components']['discovery']['stats']
        print(f"服务发现: {disc_stats['total_instances']} 实例")

    print("\n=== 集群状态 ===")
    if hasattr(coordinator, 'get_distributed_status'):
        dist_status = coordinator.get_distributed_status()
        dist_config = dist_status.get('distributed_config', {})
        print(f"高可用模式: {'启用' if dist_config.get('auto_failover', False) else '禁用'}")
        print(f"复制因子: {dist_config.get('replication_factor', 1)}")


def demonstrate_config_sync(coordinator: DistributedLogCoordinator):
    """演示配置同步"""
    print(f"\n配置同步演示 (节点: {coordinator.node_id})")

    # 更新本地配置
    coordinator.broadcast_config_update("logger.level", "DEBUG")
    coordinator.broadcast_config_update("pool.max_size", 200)

    print("配置更新已广播到集群")

    # 等待同步
    time.sleep(2)

    # 检查同步状态
    if hasattr(coordinator, 'config_sync') and coordinator.config_sync:
        sync_stats = coordinator.config_sync.get_sync_stats()
        print(f"同步统计: {sync_stats['total_syncs']} 次同步, {sync_stats['successful_syncs']} 次成功")


def demonstrate_load_balancing(coordinator: DistributedLogCoordinator):
    """演示负载均衡"""
    print(f"\n负载均衡演示 (节点: {coordinator.node_id})")

    # 模拟多个请求
    for i in range(10):
        selected_node = coordinator.cluster_manager.select_node({
            'request_id': f'req-{i}',
            'type': 'log_entry'
        })

        print(f"请求 {i+1} 路由到节点: {selected_node}")

        # 记录请求
        if coordinator.load_balancer:
            coordinator.load_balancer.record_request(
                selected_node, success=True, response_time=0.01)

        time.sleep(0.05)

    # 显示负载均衡统计
    if coordinator.load_balancer:
        stats = coordinator.load_balancer.get_all_stats()
        print("\n负载均衡统计:")
        for node_id, node_stats in stats.items():
            print(
                f"  节点 {node_id}: {node_stats.active_connections} 活跃连接, {node_stats.success_rate:.2%} 成功率")


def main():
    """主函数"""
    print("RQA2025 分布式Logger系统演示")
    print("=" * 50)

    # 创建分布式Logger节点
    node_id = f"node-{int(time.time())}"
    coordinator = create_distributed_logger(node_id)

    try:
        # 启动协调器
        print("启动分布式协调器...")
        coordinator.start()

        # 等待系统初始化
        time.sleep(2)

        # 执行各种演示
        simulate_distributed_logging(coordinator)
        monitor_system_status(coordinator)
        demonstrate_config_sync(coordinator)
        demonstrate_load_balancing(coordinator)

        # 最终状态检查
        print(f"\n最终系统状态:")
        final_status = coordinator.get_system_status()
        print(f"总协调次数: {final_status['coordination_stats']['total_coordinations']}")

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 优雅关闭
        print("关闭分布式Logger系统...")
        if hasattr(coordinator, 'perform_graceful_shutdown'):
            coordinator.perform_graceful_shutdown()
        else:
            coordinator.stop()

        print("演示完成")


if __name__ == "__main__":
    main()

"""
MicroService演示脚本
展示微服务基础框架的核心功能
"""

from src.core import EventBus, ServiceContainer
from src.services import MicroService, ServiceType, ServiceInfo


def main():
    """主演示函数"""
    print("🚀 MicroService演示开始")
    print("=" * 50)

    # 创建基础组件
    event_bus = EventBus()
    container = ServiceContainer()

    # 创建微服务框架
    micro_service = MicroService(event_bus, container, "DemoMicroService")

    print("1. 服务初始化")
    print(f"   - 服务名称: {micro_service.name}")
    print(f"   - 服务状态: {micro_service.get_status()}")
    print(f"   - 注册服务数量: {len(micro_service.service_registry)}")
    print(f"   - 发现配置数量: {len(micro_service.service_discovery)}")
    print()

    # 启动服务
    print("2. 启动服务")
    micro_service.start()
    print(f"   - 服务状态: {micro_service.get_status()}")
    print()

    # 注册新服务
    print("3. 注册新服务")
    new_service = ServiceInfo(
        service_id="demo_service_001",
        service_name="demo_service",
        service_type=ServiceType.API,
        host="localhost",
        port=9000,
        version="1.0.0",
        metadata={"environment": "demo", "region": "us-east-1"}
    )

    success = micro_service.register_service(new_service)
    print(f"   - 注册结果: {success}")
    print(f"   - 注册服务数量: {len(micro_service.service_registry)}")
    print()

    # 发现服务
    print("4. 发现服务")
    discovered = micro_service.discover_service("demo_service")
    print(f"   - 发现服务数量: {len(discovered)}")
    if discovered:
        service = discovered[0]
        print(f"   - 服务ID: {service.service_id}")
        print(f"   - 服务名称: {service.service_name}")
        print(f"   - 服务类型: {service.service_type.value}")
        print(f"   - 主机: {service.host}:{service.port}")
        print(f"   - 版本: {service.version}")
        print(f"   - 元数据: {service.metadata}")
    print()

    # 按类型发现服务
    print("5. 按类型发现服务")
    api_services = micro_service.discover_service("api_service", ServiceType.API)
    print(f"   - API服务数量: {len(api_services)}")

    business_services = micro_service.discover_service("business_service", ServiceType.BUSINESS)
    print(f"   - 业务服务数量: {len(business_services)}")
    print()

    # 获取服务信息
    print("6. 获取服务信息")
    service_info = micro_service.get_service_info("demo_service_001")
    if service_info:
        print(f"   - 服务名称: {service_info.service_name}")
        print(f"   - 注册时间: {service_info.registered_at}")
        print(f"   - 最后心跳: {service_info.last_heartbeat}")
    print()

    # 列出所有服务
    print("7. 列出所有服务")
    all_services = micro_service.list_services()
    print(f"   - 总服务数量: {len(all_services)}")

    # 按类型列出服务
    for service_type in ServiceType:
        services = micro_service.list_services(service_type)
        print(f"   - {service_type.value}服务: {len(services)}个")
    print()

    # 设置和获取配置
    print("8. 配置管理")
    micro_service.set_config("demo_key", "demo_value")
    micro_service.set_config("app_config", {"timeout": 30, "retries": 3})

    value = micro_service.get_config("demo_key")
    config = micro_service.get_config("app_config")
    default_value = micro_service.get_config("non_existent", "default")

    print(f"   - demo_key: {value}")
    print(f"   - app_config: {config}")
    print(f"   - non_existent: {default_value}")
    print()

    # 健康检查
    print("9. 健康检查")
    health = micro_service._health_check()
    print(f"   - 总服务数: {health['total_services']}")
    print(f"   - 健康服务数: {health['healthy_services']}")
    print(f"   - 不健康服务数: {health['unhealthy_services']}")
    print(f"   - 服务类型数: {health['service_types']}")
    print(f"   - 活跃发现数: {health['active_discoveries']}")
    print()

    # 检查特定服务健康状态
    print("10. 检查特定服务健康状态")
    status = micro_service.check_service_health("demo_service_001")
    print(f"   - 服务状态: {status.value}")

    # 检查不存在的服务
    unknown_status = micro_service.check_service_health("non_existent_service")
    print(f"   - 不存在服务状态: {unknown_status.value}")
    print()

    # 获取服务统计
    print("11. 获取服务统计")
    stats = micro_service.get_service_stats()
    print(f"   - 总服务数: {stats['total_services']}")
    print(f"   - 配置数量: {stats['config_count']}")
    print(f"   - 发现配置数: {stats['discovery_count']}")

    print("   - 服务类型分布:")
    for service_type, count in stats['service_types'].items():
        print(f"     * {service_type}: {count}个")

    print("   - 健康状态分布:")
    for status_type, count in stats['health_status'].items():
        print(f"     * {status_type}: {count}个")
    print()

    # 注销服务
    print("12. 注销服务")
    unregister_success = micro_service.unregister_service("demo_service_001")
    print(f"   - 注销结果: {unregister_success}")
    print(f"   - 剩余服务数: {len(micro_service.service_registry)}")

    # 尝试注销不存在的服务
    unregister_fail = micro_service.unregister_service("non_existent_service")
    print(f"   - 注销不存在服务: {unregister_fail}")
    print()

    # 服务发现配置
    print("13. 服务发现配置")
    print("   - 默认发现配置:")
    for service_name, discovery in micro_service.service_discovery.items():
        print(f"     * {service_name}: {discovery.service_type.value} ({discovery.load_balancer})")
    print()

    # 健康检查器
    print("14. 健康检查器")
    print("   - 默认健康检查器:")
    for service_type, checker in micro_service.health_checkers.items():
        print(f"     * {service_type}: {checker.__name__}")
    print()

    # 负载均衡器
    print("15. 负载均衡器")
    print("   - 默认负载均衡器:")
    for service_type, backends in micro_service.load_balancers.items():
        print(f"     * {service_type}: {len(backends)}个后端")
    print()

    # 停止服务
    print("16. 停止服务")
    micro_service.stop()
    print(f"   - 服务状态: {micro_service.get_status()}")
    print()

    print("✅ MicroService演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()

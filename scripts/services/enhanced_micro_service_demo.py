#!/usr/bin/env python3
"""
增强版MicroService演示脚本
展示第三阶段新增功能
"""

import logging
from src.core import EventBus, ServiceContainer
from src.services.micro_service import MicroService, ServiceType, ServiceInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_third_stage_features():
    """演示第三阶段功能"""
    print("🚀 增强版MicroService第三阶段功能演示")

    # 初始化
    event_bus = EventBus()
    container = ServiceContainer()
    micro_service = MicroService(event_bus, container, "EnhancedDemo")
    micro_service.start()

    try:
        # 1. 缓存机制演示
        print("\n🔍 缓存机制演示")
        micro_service.set_cached("test_key", "test_value", ttl=60)
        cached_value = micro_service.get_cached("test_key")
        print(f"缓存值: {cached_value}")

        # 2. 熔断器演示
        print("\n⚡ 熔断器模式演示")
        service_id = "demo_service"
        breaker = micro_service.get_circuit_breaker(service_id)
        print(f"熔断器状态: {breaker.state.value}")

        # 3. 重试机制演示
        print("\n🔄 重试机制演示")

        def test_operation():
            return "success"

        result = micro_service.retry_call("test_op", test_operation)
        print(f"操作结果: {result}")

        # 4. 增强服务发现演示
        print("\n🔍 增强服务发现演示")
        service_info = ServiceInfo(
            service_id="demo_001",
            service_name="demo_service",
            service_type=ServiceType.API,
            host="localhost",
            port=8000,
            version="1.0.0"
        )
        micro_service.register_service(service_info)

        discovered = micro_service.discover_service_cached("demo_service")
        print(f"发现服务数: {len(discovered)}")

        # 5. 综合统计演示
        print("\n📊 综合统计演示")
        stats = micro_service.get_comprehensive_stats()
        print(f"服务统计: {stats['service_stats']['total_services']} 个服务")
        print(f"缓存统计: {stats['cache_stats']['total_entries']} 个缓存条目")
        print(f"熔断器统计: {stats['circuit_breaker_stats']['total_breakers']} 个熔断器")

        print("\n✅ 第三阶段功能演示完成！")

    finally:
        micro_service.stop()


if __name__ == "__main__":
    demo_third_stage_features()

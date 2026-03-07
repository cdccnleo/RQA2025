#!/usr/bin/env python3
"""
分布式服务发现与注册组件

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- service_registry.py: 服务注册中心、服务实例、健康检查
- discovery_client.py: 服务发现客户端、负载均衡器

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import uuid
from typing import Dict, List, Optional
from pathlib import Path
import sys

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# 导入所有组件
from .service_registry import (
    ServiceStatus,
    ServiceInstance,
    HealthCheckResult,
    ServiceRegistry
)
from .discovery_client import (
    LoadBalanceStrategy,
    LoadBalancer,
    ServiceDiscoveryClient
)

logger = logging.getLogger(__name__)


# 全局服务注册中心实例
_global_registry: Optional[ServiceRegistry] = None


def get_service_registry(config: Optional[Dict] = None) -> ServiceRegistry:
    """获取全局服务注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry(config)
    return _global_registry


def create_service_instance(service_name: str, host: str, port: int,
                            metadata: Optional[Dict] = None,
                            tags: Optional[List[str]] = None,
                            health_check_url: Optional[str] = None) -> ServiceInstance:
    """创建服务实例的便捷方法"""
    service_id = f"{service_name}-{host}-{port}-{uuid.uuid4().hex[:8]}"

    return ServiceInstance(
        service_id=service_id,
        service_name=service_name,
        host=host,
        port=port,
        metadata=metadata or {},
        tags=tags or [],
        health_check_url=health_check_url
    )


# ServiceDiscovery别名（向后兼容）
ServiceDiscovery = ServiceRegistry

# 导出所有组件
__all__ = [
    # 枚举
    'ServiceStatus',
    'LoadBalanceStrategy',
    # 数据类
    'ServiceInstance',
    'HealthCheckResult',
    # 核心类
    'ServiceRegistry',
    'ServiceDiscovery',  # 别名
    'ServiceDiscoveryClient',
    'LoadBalancer',
    # 工具函数
    'get_service_registry',
    'create_service_instance'
]


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 创建服务注册中心
    registry = get_service_registry()
    
    # 创建服务实例
    service1 = create_service_instance("api-service", "localhost", 8080)
    service2 = create_service_instance("api-service", "localhost", 8081)
    
    # 注册服务
    registry.register(service1)
    registry.register(service2)
    
    # 创建服务发现客户端
    client = ServiceDiscoveryClient(registry)
    
    # 发现服务
    service = client.discover("api-service")
    if service:
        logger.info(f"发现服务: {service.host}:{service.port}")
    
    # 发现所有实例
    services = client.discover_all("api-service")
    logger.info(f"发现 {len(services)} 个服务实例")

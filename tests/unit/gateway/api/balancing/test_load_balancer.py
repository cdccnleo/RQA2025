#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负载均衡器测试

测试目标：提升load_balancer.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

# 确保Python路径正确配置（必须在所有导入之前）
# 文件路径: tests/unit/gateway/api/balancing/test_load_balancer.py
# 需要向上6级到达项目根目录
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

# 确保路径在sys.path的最前面
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
if src_path_str in sys.path:
    sys.path.remove(src_path_str)

sys.path.insert(0, project_root_str)
sys.path.insert(0, src_path_str)


def _import_gateway_modules():
    """动态导入网关模块"""
    # 确保路径配置（每次调用时重新配置）
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    src_path_str = str(project_root / "src")
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
    
    load_balancer_module = importlib.import_module('src.gateway.api.balancing.load_balancer')
    gateway_types_module = importlib.import_module('src.gateway.api.gateway_types')
    return load_balancer_module.LoadBalancer, gateway_types_module.ServiceEndpoint, gateway_types_module.ServiceStatus, gateway_types_module.HttpMethod


class TestLoadBalancer:
    """测试负载均衡器"""
    
    @pytest.fixture
    def gateway_modules(self):
        """网关模块fixture"""
        return _import_gateway_modules()
    
    @pytest.fixture
    def load_balancer(self, gateway_modules):
        """负载均衡器fixture"""
        LoadBalancer, _, _, _ = gateway_modules
        return LoadBalancer(algorithm="round_robin")
    
    @pytest.fixture
    def endpoints(self, gateway_modules):
        """服务端点fixture"""
        _, ServiceEndpoint, ServiceStatus, _ = gateway_modules
        return [
            ServiceEndpoint(
                service_name="service1",
                upstream_url="http://localhost:8001",
                weight=1,
                status=ServiceStatus.HEALTHY
            ),
            ServiceEndpoint(
                service_name="service2",
                upstream_url="http://localhost:8002",
                weight=2,
                status=ServiceStatus.HEALTHY
            ),
            ServiceEndpoint(
                service_name="service3",
                upstream_url="http://localhost:8003",
                weight=1,
                status=ServiceStatus.UNHEALTHY
            )
        ]
    
    def test_load_balancer_init_round_robin(self, gateway_modules):
        """测试负载均衡器初始化 - 轮询算法"""
        LoadBalancer, _, _, _ = gateway_modules
        lb = LoadBalancer(algorithm="round_robin")
        assert lb.algorithm == "round_robin"
        assert lb.endpoints == []
        assert lb.current_index == 0
    
    def test_load_balancer_init_weighted(self, gateway_modules):
        """测试负载均衡器初始化 - 加权算法"""
        LoadBalancer, _, _, _ = gateway_modules
        lb = LoadBalancer(algorithm="weighted")
        assert lb.algorithm == "weighted"
    
    def test_load_balancer_init_random(self, gateway_modules):
        """测试负载均衡器初始化 - 随机算法"""
        LoadBalancer, _, _, _ = gateway_modules
        lb = LoadBalancer(algorithm="random")
        assert lb.algorithm == "random"
    
    def test_add_endpoint(self, load_balancer, endpoints):
        """测试添加服务端点"""
        load_balancer.add_endpoint(endpoints[0])
        assert len(load_balancer.endpoints) == 1
        assert load_balancer.endpoints[0].service_name == "service1"
    
    def test_get_endpoint_round_robin(self, load_balancer, endpoints):
        """测试获取端点 - 轮询算法"""
        load_balancer.add_endpoint(endpoints[0])
        load_balancer.add_endpoint(endpoints[1])
        
        endpoint1 = load_balancer.get_endpoint()
        assert endpoint1 is not None
        assert endpoint1.service_name == "service1"
        
        endpoint2 = load_balancer.get_endpoint()
        assert endpoint2 is not None
        assert endpoint2.service_name == "service2"
        
        endpoint3 = load_balancer.get_endpoint()
        assert endpoint3.service_name == "service1"  # 轮询回到第一个
    
    def test_get_endpoint_no_endpoints(self, load_balancer):
        """测试获取端点 - 无端点"""
        endpoint = load_balancer.get_endpoint()
        assert endpoint is None
    
    def test_get_endpoint_only_unhealthy(self, load_balancer, endpoints):
        """测试获取端点 - 只有不健康端点"""
        load_balancer.add_endpoint(endpoints[2])  # UNHEALTHY
        endpoint = load_balancer.get_endpoint()
        assert endpoint is None
    
    def test_get_endpoint_weighted(self, gateway_modules, endpoints):
        """测试获取端点 - 加权算法"""
        LoadBalancer, _, _, _ = gateway_modules
        lb = LoadBalancer(algorithm="weighted")
        lb.add_endpoint(endpoints[0])  # weight=1
        lb.add_endpoint(endpoints[1])  # weight=2
        
        endpoint = lb.get_endpoint()
        assert endpoint is not None
        assert endpoint.service_name in ["service1", "service2"]
    
    def test_get_endpoint_random(self, gateway_modules, endpoints):
        """测试获取端点 - 随机算法"""
        LoadBalancer, _, _, _ = gateway_modules
        lb = LoadBalancer(algorithm="random")
        lb.add_endpoint(endpoints[0])
        lb.add_endpoint(endpoints[1])
        
        endpoint = lb.get_endpoint()
        assert endpoint is not None
        assert endpoint.service_name in ["service1", "service2"]
    
    def test_select_endpoint_alias(self, load_balancer, endpoints):
        """测试select_endpoint别名方法"""
        load_balancer.add_endpoint(endpoints[0])
        endpoint = load_balancer.select_endpoint()
        assert endpoint is not None
        assert endpoint.service_name == "service1"
    
    def test_get_endpoint_skips_unhealthy(self, gateway_modules, load_balancer, endpoints):
        """测试获取端点跳过不健康端点"""
        _, _, ServiceStatus, _ = gateway_modules
        load_balancer.add_endpoint(endpoints[0])  # HEALTHY
        load_balancer.add_endpoint(endpoints[2])  # UNHEALTHY
        load_balancer.add_endpoint(endpoints[1])  # HEALTHY
        
        endpoint = load_balancer.get_endpoint()
        assert endpoint is not None
        assert endpoint.status == ServiceStatus.HEALTHY


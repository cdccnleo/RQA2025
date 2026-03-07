"""
load_balancer 模块

提供 load_balancer 相关功能和接口。
"""

import logging
import random
import secrets

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
"""
负载均衡器
提供多种负载均衡策略，支持健康检查和故障转移
"""

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


@dataclass
class Endpoint:
    url: str
    weight: int = 1
    is_healthy: bool = True
    connections: int = 0


class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.endpoints: List[Endpoint] = []
        self.current_index = 0

    def add_endpoint(self, url: str, weight: int = 1) -> None:
        """添加端点"""
        endpoint = Endpoint(url=url, weight=weight)
        self.endpoints.append(endpoint)
        logger.info(f"Added endpoint: {url}")

    def remove_endpoint(self, url: str) -> bool:
        """移除端点"""
        for i, endpoint in enumerate(self.endpoints):
            if endpoint.url == url:
                self.endpoints.pop(i)
                logger.info(f"Removed endpoint: {url}")
                return True
        return False

    def get_endpoint(self, client_ip: Optional[str] = None) -> Optional[Endpoint]:
        """获取端点"""
        if not self.endpoints:
            return None

        # 过滤健康的端点
        healthy_endpoints = [ep for ep in self.endpoints if ep.is_healthy]
        if not healthy_endpoints:
            return None

        # 根据策略选择端点
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
            self.current_index += 1
            return endpoint
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_endpoints, key=lambda ep: ep.connections)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            total_weight = sum(ep.weight for ep in healthy_endpoints)
            rand_val = random.uniform(0, total_weight)
            current_weight = 0
            for endpoint in healthy_endpoints:
                current_weight += endpoint.weight
                if rand_val <= current_weight:
                    return endpoint
            return healthy_endpoints[0]  # fallback
        else:
            return secrets.choice(healthy_endpoints)

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始负载均衡器模块健康检查")

        health_checks = {
            "load_balancer_class": check_load_balancer_class(),
            "strategy_implementation": check_strategy_implementation(),
            "endpoint_management": check_endpoint_management()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "load_balancer",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("负载均衡器模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"负载均衡器模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"负载均衡器模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "load_balancer",
            "error": str(e)
        }


def check_load_balancer_class() -> Dict[str, Any]:
    """检查负载均衡器类定义

    Returns:
        Dict[str, Any]: 负载均衡器类检查结果
    """
    try:
        # 检查LoadBalancer类存在
        load_balancer_exists = 'LoadBalancer' in globals()

        if not load_balancer_exists:
            return {"healthy": False, "error": "LoadBalancer class not found"}

        # 检查必需的方法
        required_methods = ['__init__', 'add_endpoint', 'remove_endpoint', 'get_endpoint']
        existing_methods = [method for method in dir(LoadBalancer) if not method.startswith('_')]

        methods_complete = all(method in existing_methods for method in required_methods)

        # 测试类实例化
        instantiation_works = False
        try:
            lb = LoadBalancer()
            instantiation_works = lb is not None
        except Exception:
            instantiation_works = False

        return {
            "healthy": load_balancer_exists and methods_complete and instantiation_works,
            "load_balancer_exists": load_balancer_exists,
            "methods_complete": methods_complete,
            "instantiation_works": instantiation_works,
            "existing_methods": existing_methods
        }
    except Exception as e:
        logger.error(f"负载均衡器类检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_strategy_implementation() -> Dict[str, Any]:
    """检查策略实现

    Returns:
        Dict[str, Any]: 策略实现检查结果
    """
    try:
        # 检查枚举定义
        strategy_enum_exists = 'LoadBalancingStrategy' in globals()

        if not strategy_enum_exists:
            return {"healthy": False, "error": "LoadBalancingStrategy enum not found"}

        # 检查所有策略值
        expected_strategies = ["round_robin", "least_connections", "weighted"]
        actual_strategies = [strategy.value for strategy in LoadBalancingStrategy]

        strategies_complete = set(actual_strategies) == set(expected_strategies)

        # 测试不同策略
        strategy_tests = {}
        for strategy in LoadBalancingStrategy:
            try:
                lb = LoadBalancer(strategy)
                # 添加测试端点
                lb.add_endpoint("http://test1.com")
                lb.add_endpoint("http://test2.com")
                # 测试获取端点
                endpoint = lb.get_endpoint()
                strategy_tests[strategy.value] = {
                    "success": endpoint is not None,
                    "endpoint_url": endpoint.url if endpoint else None
                }
            except Exception as e:
                strategy_tests[strategy.value] = {"success": False, "error": str(e)}

        all_strategies_work = all(test["success"] for test in strategy_tests.values())

        return {
            "healthy": strategies_complete and all_strategies_work,
            "strategies_complete": strategies_complete,
            "all_strategies_work": all_strategies_work,
            "strategy_tests": strategy_tests,
            "expected_strategies": expected_strategies,
            "actual_strategies": actual_strategies
        }
    except Exception as e:
        logger.error(f"策略实现检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_endpoint_management() -> Dict[str, Any]:
    """检查端点管理功能

    Returns:
        Dict[str, Any]: 端点管理检查结果
    """
    try:
        # 检查Endpoint数据类
        endpoint_class_exists = 'Endpoint' in globals()

        if not endpoint_class_exists:
            return {"healthy": False, "error": "Endpoint dataclass not found"}

        # 测试端点管理功能
        lb = LoadBalancer()
        endpoint_management_works = True
        management_tests = {}

        try:
            # 测试添加端点
            lb.add_endpoint("http://test1.com", weight=2)
            lb.add_endpoint("http://test2.com", weight=1)
            management_tests["add_endpoints"] = {
                "success": len(lb.endpoints) == 2,
                "endpoint_count": len(lb.endpoints)
            }
        except Exception as e:
            endpoint_management_works = False
            management_tests["add_endpoints"] = {"success": False, "error": str(e)}

        try:
            # 测试移除端点
            removed = lb.remove_endpoint("http://test1.com")
            management_tests["remove_endpoint"] = {
                "success": removed and len(lb.endpoints) == 1,
                "remaining_count": len(lb.endpoints)
            }
        except Exception as e:
            endpoint_management_works = False
            management_tests["remove_endpoint"] = {"success": False, "error": str(e)}

        try:
            # 测试获取端点
            endpoint = lb.get_endpoint()
            management_tests["get_endpoint"] = {
                "success": endpoint is not None,
                "endpoint_url": endpoint.url if endpoint else None
            }
        except Exception as e:
            endpoint_management_works = False
            management_tests["get_endpoint"] = {"success": False, "error": str(e)}

        return {
            "healthy": endpoint_class_exists and endpoint_management_works,
            "endpoint_class_exists": endpoint_class_exists,
            "endpoint_management_works": endpoint_management_works,
            "management_tests": management_tests
        }
    except Exception as e:
        logger.error(f"端点管理检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要

    Returns:
        Dict[str, Any]: 健康状态摘要
    """
    try:
        health_check = check_health()

        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "load_balancer",
            "health_check": health_check,
            "strategies_count": len(LoadBalancingStrategy) if 'LoadBalancingStrategy' in globals() else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康状态摘要失败: {str(e)}")
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告

    Returns:
        Dict[str, Any]: 健康摘要报告
    """
    try:
        health_check = check_health()

        # 统计负载均衡器信息
        strategies_available = len(
            LoadBalancingStrategy) if 'LoadBalancingStrategy' in globals() else 0

        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "load_balancer_module_info": {
                "service_name": "load_balancer",
                "purpose": "负载均衡和故障转移",
                "operational": health_check["healthy"]
            },
            "balancing_capabilities": {
                "strategies_available": strategies_available,
                "endpoint_management_working": health_check["checks"]["endpoint_management"]["healthy"],
                "strategy_implementation_complete": health_check["checks"]["strategy_implementation"]["healthy"]
            },
            "functionality_status": {
                "load_balancer_class_working": health_check["checks"]["load_balancer_class"]["healthy"],
                "all_strategies_functional": health_check["checks"]["strategy_implementation"]["all_strategies_work"],
                "endpoint_operations_working": health_check["checks"]["endpoint_management"]["endpoint_management_works"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_load_balancer_module() -> Dict[str, Any]:
    """监控负载均衡器模块状态

    Returns:
        Dict[str, Any]: 模块监控结果
    """
    try:
        health_check = check_health()

        # 计算模块效率指标
        module_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "module_metrics": {
                "service_name": "load_balancer",
                "module_efficiency": module_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "balancing_metrics": {
                "strategies_implemented": len(LoadBalancingStrategy) if 'LoadBalancingStrategy' in globals() else 0,
                "endpoint_management_working": health_check["checks"]["endpoint_management"]["healthy"],
                "strategy_selection_working": health_check["checks"]["strategy_implementation"]["healthy"]
            }
        }
    except Exception as e:
        logger.error(f"负载均衡器模块监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_load_balancer_config() -> Dict[str, Any]:
    """验证负载均衡器配置

    Returns:
        Dict[str, Any]: 配置验证结果
    """
    try:
        validation_results = {
            "class_validation": _validate_load_balancer_classes(),
            "enum_validation": _validate_strategy_enum(),
            "functionality_validation": _validate_balancing_functionality()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"负载均衡器配置验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_load_balancer_classes() -> Dict[str, Any]:
    """验证负载均衡器类"""
    try:
        # 检查必需的类
        required_classes = ['LoadBalancer', 'Endpoint', 'LoadBalancingStrategy']
        classes_exist = all(cls in globals() for cls in required_classes)

        # 检查类是否可以实例化
        instantiation_tests = {}
        if classes_exist:
            try:
                lb = LoadBalancer()
                instantiation_tests["LoadBalancer"] = {"success": True}
            except Exception as e:
                instantiation_tests["LoadBalancer"] = {"success": False, "error": str(e)}

            try:
                endpoint = Endpoint(url="http://test.com")
                instantiation_tests["Endpoint"] = {"success": True}
            except Exception as e:
                instantiation_tests["Endpoint"] = {"success": False, "error": str(e)}

        all_instantiable = all(
            test["success"] for test in instantiation_tests.values()) if instantiation_tests else False

        return {
            "valid": classes_exist and all_instantiable,
            "classes_exist": classes_exist,
            "all_instantiable": all_instantiable,
            "instantiation_tests": instantiation_tests,
            "required_classes": required_classes
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_strategy_enum() -> Dict[str, Any]:
    """验证策略枚举"""
    try:
        # 检查枚举存在
        strategy_enum_exists = 'LoadBalancingStrategy' in globals()

        if not strategy_enum_exists:
            return {"valid": False, "error": "LoadBalancingStrategy not found"}

        # 检查枚举值
        expected_values = ["round_robin", "least_connections", "weighted"]
        actual_values = [strategy.value for strategy in LoadBalancingStrategy]

        values_match = set(actual_values) == set(expected_values)

        # 检查枚举方法
        has_from_string = hasattr(LoadBalancingStrategy, '__members__')

        return {
            "valid": strategy_enum_exists and values_match and has_from_string,
            "strategy_enum_exists": strategy_enum_exists,
            "values_match": values_match,
            "has_members": has_from_string,
            "expected_values": expected_values,
            "actual_values": actual_values
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_balancing_functionality() -> Dict[str, Any]:
    """验证负载均衡功能"""
    try:
        # 创建测试负载均衡器
        lb = LoadBalancer()

        # 测试端点管理
        endpoint_tests = {}
        try:
            lb.add_endpoint("http://test1.com", weight=2)
            lb.add_endpoint("http://test2.com", weight=1)
            endpoint_tests["add_endpoints"] = {"success": len(lb.endpoints) == 2}
        except Exception as e:
            endpoint_tests["add_endpoints"] = {"success": False, "error": str(e)}

        # 测试负载均衡策略
        strategy_tests = {}
        for strategy in LoadBalancingStrategy:
            try:
                test_lb = LoadBalancer(strategy)
                test_lb.add_endpoint("http://test.com")
                endpoint = test_lb.get_endpoint()
                strategy_tests[strategy.value] = {"success": endpoint is not None}
            except Exception as e:
                strategy_tests[strategy.value] = {"success": False, "error": str(e)}

        all_strategies_work = all(test["success"] for test in strategy_tests.values())
        endpoint_management_works = endpoint_tests.get("add_endpoints", {}).get("success", False)

        return {
            "valid": endpoint_management_works and all_strategies_work,
            "endpoint_management_works": endpoint_management_works,
            "all_strategies_work": all_strategies_work,
            "endpoint_tests": endpoint_tests,
            "strategy_tests": strategy_tests
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

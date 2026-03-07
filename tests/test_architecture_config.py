#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试架构配置文件
定义分层测试执行策略、测试依赖管理和性能基准测试配置
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class TestLayer(Enum):
    """测试层级枚举"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"


class TestCategory(Enum):
    """测试类别枚举"""
    BUSINESS_LOGIC = "business_logic"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


@dataclass
class TestExecutionConfig:
    """测试执行配置"""
    layer: TestLayer
    category: TestCategory
    timeout: int = 300
    max_workers: int = 4
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    required_services: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestDependency:
    """测试依赖配置"""
    test_module: str
    dependencies: List[str]
    resource_requirements: Dict[str, Any]
    setup_commands: List[str] = field(default_factory=list)
    teardown_commands: List[str] = field(default_factory=list)


@dataclass
class PerformanceBenchmark:
    """性能基准配置"""
    test_name: str
    metric_name: str
    baseline_value: float
    tolerance_percent: float
    comparison_operator: str  # "lt", "le", "gt", "ge", "eq"
    sample_size: int = 10


class TestArchitectureConfig:
    """测试架构配置管理器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_configs = self._load_test_configs()
        self.dependency_configs = self._load_dependency_configs()
        self.performance_baselines = self._load_performance_baselines()

    def _load_test_configs(self) -> Dict[str, TestExecutionConfig]:
        """加载测试执行配置"""
        return {
            # 单元测试配置
            "unit.*": TestExecutionConfig(
                layer=TestLayer.UNIT,
                category=TestCategory.BUSINESS_LOGIC,
                timeout=60,
                max_workers=8,
                dependencies=[]
            ),

            # 基础设施集成测试
            "integration.infrastructure.*": TestExecutionConfig(
                layer=TestLayer.INTEGRATION,
                category=TestCategory.INFRASTRUCTURE,
                timeout=180,
                max_workers=2,
                dependencies=["infrastructure_services"],
                required_services=["redis", "postgresql"],
                environment_variables={
                    "TEST_DB_HOST": "localhost",
                    "TEST_REDIS_HOST": "localhost"
                }
            ),

            # 业务逻辑集成测试
            "integration.*_layer_integration": TestExecutionConfig(
                layer=TestLayer.INTEGRATION,
                category=TestCategory.BUSINESS_LOGIC,
                timeout=300,
                max_workers=4,
                dependencies=["core_services"],
                required_services=["mock_services"]
            ),

            # 端到端测试配置
            "e2e.*": TestExecutionConfig(
                layer=TestLayer.E2E,
                category=TestCategory.BUSINESS_LOGIC,
                timeout=600,
                max_workers=2,
                dependencies=["full_system"],
                required_services=["full_stack"],
                environment_variables={
                    "TEST_ENVIRONMENT": "staging",
                    "ENABLE_E2E_LOGGING": "true"
                }
            ),

            # 性能测试配置
            "performance.*": TestExecutionConfig(
                layer=TestLayer.PERFORMANCE,
                category=TestCategory.PERFORMANCE,
                timeout=900,
                max_workers=1,
                dependencies=["performance_infrastructure"],
                required_services=["monitoring", "load_generator"]
            ),

            # 安全测试配置
            "security.*": TestExecutionConfig(
                layer=TestLayer.SECURITY,
                category=TestCategory.SECURITY,
                timeout=300,
                max_workers=2,
                dependencies=["security_services"],
                required_services=["security_tools"]
            )
        }

    def _load_dependency_configs(self) -> Dict[str, TestDependency]:
        """加载测试依赖配置"""
        return {
            "infrastructure_services": TestDependency(
                test_module="infrastructure",
                dependencies=["database", "cache", "message_queue"],
                resource_requirements={
                    "cpu_cores": 2,
                    "memory_gb": 4,
                    "disk_gb": 10
                },
                setup_commands=[
                    "docker-compose -f docker-compose.test.yml up -d",
                    "sleep 30"  # 等待服务启动
                ],
                teardown_commands=[
                    "docker-compose -f docker-compose.test.yml down"
                ]
            ),

            "core_services": TestDependency(
                test_module="core",
                dependencies=["business_orchestrator", "event_bus", "config_manager"],
                resource_requirements={
                    "cpu_cores": 1,
                    "memory_gb": 2,
                    "disk_gb": 5
                }
            ),

            "full_system": TestDependency(
                test_module="full_system",
                dependencies=[
                    "infrastructure_services",
                    "core_services",
                    "all_business_layers",
                    "monitoring_system"
                ],
                resource_requirements={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "disk_gb": 20
                },
                setup_commands=[
                    "./scripts/setup_test_environment.sh",
                    "sleep 60"  # 等待完整系统启动
                ],
                teardown_commands=[
                    "./scripts/cleanup_test_environment.sh"
                ]
            ),

            "performance_infrastructure": TestDependency(
                test_module="performance",
                dependencies=["load_generator", "monitoring", "data_generator"],
                resource_requirements={
                    "cpu_cores": 8,
                    "memory_gb": 16,
                    "disk_gb": 50
                }
            )
        }

    def _load_performance_baselines(self) -> Dict[str, PerformanceBenchmark]:
        """加载性能基准配置"""
        return {
            "strategy_backtest_time": PerformanceBenchmark(
                test_name="test_strategy_backtest_performance",
                metric_name="backtest_execution_time",
                baseline_value=120.0,  # 2分钟
                tolerance_percent=20.0,
                comparison_operator="le"
            ),

            "api_response_time": PerformanceBenchmark(
                test_name="test_api_response_performance",
                metric_name="average_response_time",
                baseline_value=100.0,  # 100ms
                tolerance_percent=15.0,
                comparison_operator="le"
            ),

            "memory_usage": PerformanceBenchmark(
                test_name="test_memory_usage",
                metric_name="peak_memory_mb",
                baseline_value=512.0,  # 512MB
                tolerance_percent=25.0,
                comparison_operator="le"
            ),

            "cpu_usage": PerformanceBenchmark(
                test_name="test_cpu_usage",
                metric_name="average_cpu_percent",
                baseline_value=70.0,  # 70%
                tolerance_percent=10.0,
                comparison_operator="le"
            ),

            "database_query_time": PerformanceBenchmark(
                test_name="test_database_performance",
                metric_name="query_execution_time",
                baseline_value=50.0,  # 50ms
                tolerance_percent=30.0,
                comparison_operator="le"
            ),

            "concurrent_users": PerformanceBenchmark(
                test_name="test_concurrency_performance",
                metric_name="max_concurrent_users",
                baseline_value=1000,
                tolerance_percent=10.0,
                comparison_operator="ge"
            ),

            "error_rate": PerformanceBenchmark(
                test_name="test_error_rate",
                metric_name="error_rate_percent",
                baseline_value=0.1,  # 0.1%
                tolerance_percent=50.0,
                comparison_operator="le"
            ),

            "throughput": PerformanceBenchmark(
                test_name="test_system_throughput",
                metric_name="requests_per_second",
                baseline_value=500,
                tolerance_percent=15.0,
                comparison_operator="ge"
            )
        }

    def get_test_config(self, test_pattern: str) -> TestExecutionConfig:
        """根据测试模式获取配置"""
        for pattern, config in self.test_configs.items():
            if self._matches_pattern(test_pattern, pattern):
                return config

        # 返回默认配置
        return TestExecutionConfig(
            layer=TestLayer.UNIT,
            category=TestCategory.BUSINESS_LOGIC,
            timeout=300,
            max_workers=4
        )

    def get_dependency_config(self, dependency_name: str) -> TestDependency:
        """获取依赖配置"""
        return self.dependency_configs.get(dependency_name)

    def get_performance_baseline(self, benchmark_name: str) -> PerformanceBenchmark:
        """获取性能基准"""
        return self.performance_baselines.get(benchmark_name)

    def get_execution_strategy(self, test_layer: TestLayer) -> Dict[str, Any]:
        """获取测试执行策略"""
        strategies = {
            TestLayer.UNIT: {
                "parallel_execution": True,
                "fail_fast": False,
                "report_coverage": True,
                "generate_reports": False
            },
            TestLayer.INTEGRATION: {
                "parallel_execution": True,
                "fail_fast": True,
                "report_coverage": True,
                "generate_reports": True,
                "require_dependencies": True
            },
            TestLayer.E2E: {
                "parallel_execution": False,
                "fail_fast": True,
                "report_coverage": False,
                "generate_reports": True,
                "require_dependencies": True,
                "full_environment": True
            },
            TestLayer.PERFORMANCE: {
                "parallel_execution": False,
                "fail_fast": False,
                "report_coverage": False,
                "generate_reports": True,
                "performance_monitoring": True
            },
            TestLayer.SECURITY: {
                "parallel_execution": False,
                "fail_fast": True,
                "report_coverage": False,
                "generate_reports": True,
                "security_audit": True
            }
        }

        return strategies.get(test_layer, strategies[TestLayer.UNIT])

    def get_layer_hierarchy(self) -> List[TestLayer]:
        """获取测试层级执行顺序"""
        return [
            TestLayer.UNIT,
            TestLayer.INTEGRATION,
            TestLayer.E2E,
            TestLayer.PERFORMANCE,
            TestLayer.SECURITY
        ]

    def get_resource_requirements(self, test_pattern: str) -> Dict[str, Any]:
        """获取测试资源需求"""
        config = self.get_test_config(test_pattern)
        dependency_config = self.get_dependency_config(config.dependencies[0]) if config.dependencies else None

        requirements = {
            "cpu_cores": 1,
            "memory_gb": 2,
            "disk_gb": 5,
            "network_bandwidth": "10Mbps"
        }

        if dependency_config:
            requirements.update(dependency_config.resource_requirements)

        # 根据测试层级调整资源
        if config.layer == TestLayer.E2E:
            requirements["cpu_cores"] = max(requirements["cpu_cores"], 2)
            requirements["memory_gb"] = max(requirements["memory_gb"], 4)
        elif config.layer == TestLayer.PERFORMANCE:
            requirements["cpu_cores"] = max(requirements["cpu_cores"], 4)
            requirements["memory_gb"] = max(requirements["memory_gb"], 8)

        return requirements

    def _matches_pattern(self, test_name: str, pattern: str) -> bool:
        """检查测试名称是否匹配模式"""
        import fnmatch
        return fnmatch.fnmatch(test_name, pattern)


# 全局配置实例
test_architecture_config = TestArchitectureConfig()


def get_test_config(test_pattern: str) -> TestExecutionConfig:
    """获取测试配置的便捷函数"""
    return test_architecture_config.get_test_config(test_pattern)


def get_execution_strategy(test_layer: TestLayer) -> Dict[str, Any]:
    """获取执行策略的便捷函数"""
    return test_architecture_config.get_execution_strategy(test_layer)


def get_performance_baseline(benchmark_name: str) -> PerformanceBenchmark:
    """获取性能基准的便捷函数"""
    return test_architecture_config.get_performance_baseline(benchmark_name)


def get_resource_requirements(test_pattern: str) -> Dict[str, Any]:
    """获取资源需求的便捷函数"""
    return test_architecture_config.get_resource_requirements(test_pattern)

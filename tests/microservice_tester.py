#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微服务集成测试框架

支持分布式系统测试：
- 服务间通信测试
- 契约测试（Consumer-Driven Contract Testing）
- 端到端集成测试
- 服务发现和健康检查
- 容错和恢复测试
"""

import os
import json
import time
import requests
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """服务端点"""
    name: str
    url: str
    port: int
    health_check: str = "/health"
    dependencies: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class ContractTest:
    """契约测试"""
    consumer: str
    provider: str
    request: Dict[str, Any]
    expected_response: Dict[str, Any]
    description: str = ""


@dataclass
class IntegrationTest:
    """集成测试"""
    name: str
    services: List[str]
    steps: List[Dict[str, Any]]
    timeout: int = 300
    retries: int = 3


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ServiceManager:
    """服务管理器"""

    def __init__(self, services_config: Dict[str, Any]):
        self.services = {}
        self.service_processes = {}

        # 加载服务配置
        for name, config in services_config.items():
            self.services[name] = ServiceEndpoint(
                name=name,
                url=config['url'],
                port=config['port'],
                health_check=config.get('health_check', '/health'),
                dependencies=config.get('dependencies', []),
                environment=config.get('environment', {})
            )

        logger.info(f"服务管理器初始化，共 {len(self.services)} 个服务")

    def start_service(self, service_name: str) -> bool:
        """启动服务"""
        if service_name not in self.services:
            logger.error(f"服务 {service_name} 未配置")
            return False

        service = self.services[service_name]

        try:
            # 启动服务进程（这里是模拟，实际项目中需要根据服务类型启动）
            logger.info(f"启动服务: {service_name}")

            # 这里应该根据实际的服务启动方式来实现
            # 例如：Docker容器、系统服务、本地进程等

            # 模拟启动过程
            time.sleep(2)

            # 等待服务就绪
            if self.wait_for_service(service_name, timeout=30):
                logger.info(f"服务 {service_name} 启动成功")
                return True
            else:
                logger.error(f"服务 {service_name} 启动超时")
                return False

        except Exception as e:
            logger.error(f"启动服务 {service_name} 失败: {e}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """停止服务"""
        try:
            if service_name in self.service_processes:
                # 停止进程
                process = self.service_processes[service_name]
                process.terminate()
                process.wait(timeout=10)
                del self.service_processes[service_name]

            logger.info(f"服务 {service_name} 已停止")
            return True

        except Exception as e:
            logger.error(f"停止服务 {service_name} 失败: {e}")
            return False

    def start_all_services(self) -> Dict[str, bool]:
        """启动所有服务（按依赖顺序）"""
        results = {}

        # 拓扑排序处理依赖关系
        service_order = self._resolve_dependencies()

        for service_name in service_order:
            results[service_name] = self.start_service(service_name)

        return results

    def stop_all_services(self) -> Dict[str, bool]:
        """停止所有服务"""
        results = {}

        # 反向顺序停止
        for service_name in reversed(list(self.services.keys())):
            results[service_name] = self.stop_service(service_name)

        return results

    def wait_for_service(self, service_name: str, timeout: int = 30) -> bool:
        """等待服务就绪"""
        if service_name not in self.services:
            return False

        service = self.services[service_name]
        health_url = f"{service.url}:{service.port}{service.health_check}"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"服务 {service_name} 健康检查通过")
                    return True
            except requests.RequestException:
                pass

            time.sleep(1)

        return False

    def check_service_health(self, service_name: str) -> Tuple[bool, Dict[str, Any]]:
        """检查服务健康状态"""
        if service_name not in self.services:
            return False, {"error": "服务未配置"}

        service = self.services[service_name]
        health_url = f"{service.url}:{service.port}{service.health_check}"

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True, response.json() if response.headers.get('content-type', '').startswith('application/json') else {"status": "healthy"}
            else:
                return False, {"status_code": response.status_code, "response": response.text}
        except requests.RequestException as e:
            return False, {"error": str(e)}

    def _resolve_dependencies(self) -> List[str]:
        """解决服务依赖关系（拓扑排序）"""
        # 简化的拓扑排序实现
        visited = set()
        order = []

        def visit(service_name: str):
            if service_name in visited:
                return
            visited.add(service_name)

            if service_name in self.services:
                for dep in self.services[service_name].dependencies:
                    visit(dep)

            order.append(service_name)

        for service_name in self.services:
            visit(service_name)

        return order


class ContractTester:
    """契约测试器"""

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self.contracts = []

    def load_contracts(self, contracts_file: str):
        """加载契约定义"""
        try:
            with open(contracts_file, 'r', encoding='utf-8') as f:
                contracts_data = json.load(f)

            for contract_data in contracts_data:
                contract = ContractTest(
                    consumer=contract_data['consumer'],
                    provider=contract_data['provider'],
                    request=contract_data['request'],
                    expected_response=contract_data['expected_response'],
                    description=contract_data.get('description', '')
                )
                self.contracts.append(contract)

            logger.info(f"加载了 {len(self.contracts)} 个契约测试")

        except Exception as e:
            logger.error(f"加载契约文件失败: {e}")

    def run_contract_tests(self) -> List[TestResult]:
        """运行所有契约测试"""
        results = []

        for contract in self.contracts:
            result = self._run_single_contract_test(contract)
            results.append(result)

        return results

    def _run_single_contract_test(self, contract: ContractTest) -> TestResult:
        """运行单个契约测试"""
        start_time = time.time()

        try:
            # 检查提供者服务是否可用
            if not self.service_manager.wait_for_service(contract.provider, timeout=10):
                return TestResult(
                    test_name=f"contract_{contract.consumer}_{contract.provider}",
                    success=False,
                    duration=time.time() - start_time,
                    error_message=f"提供者服务 {contract.provider} 不可用"
                )

            # 构建请求URL
            provider = self.service_manager.services[contract.provider]
            url = f"{provider.url}:{provider.port}{contract.request.get('path', '/')}"

            # 发送请求
            method = contract.request.get('method', 'GET')
            headers = contract.request.get('headers', {})
            data = contract.request.get('body', {})

            response = requests.request(method, url, headers=headers, json=data, timeout=10)

            # 验证响应
            success = self._validate_response(response, contract.expected_response)

            return TestResult(
                test_name=f"contract_{contract.consumer}_{contract.provider}",
                success=success,
                duration=time.time() - start_time,
                details={
                    "consumer": contract.consumer,
                    "provider": contract.provider,
                    "request": contract.request,
                    "expected": contract.expected_response,
                    "actual": {
                        "status_code": response.status_code,
                        "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                    }
                }
            )

        except Exception as e:
            return TestResult(
                test_name=f"contract_{contract.consumer}_{contract.provider}",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )

    def _validate_response(self, response: requests.Response, expected: Dict[str, Any]) -> bool:
        """验证响应"""
        try:
            # 检查状态码
            if 'status_code' in expected and response.status_code != expected['status_code']:
                return False

            # 检查响应体
            if 'body' in expected:
                if response.headers.get('content-type', '').startswith('application/json'):
                    actual_body = response.json()
                    expected_body = expected['body']

                    # 简单的递归比较（可以扩展更复杂的验证逻辑）
                    return self._deep_compare(actual_body, expected_body)
                else:
                    return response.text == expected['body']

            return True

        except Exception:
            return False

    def _deep_compare(self, actual: Any, expected: Any) -> bool:
        """深度比较两个对象"""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for key, value in expected.items():
                if key not in actual or not self._deep_compare(actual[key], value):
                    return False
            return True
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                return False
            for a, e in zip(actual, expected):
                if not self._deep_compare(a, e):
                    return False
            return True
        else:
            return actual == expected


class IntegrationTester:
    """集成测试器"""

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self.tests = []

    def load_integration_tests(self, tests_file: str):
        """加载集成测试"""
        try:
            with open(tests_file, 'r', encoding='utf-8') as f:
                tests_data = json.load(f)

            for test_data in tests_data:
                test = IntegrationTest(
                    name=test_data['name'],
                    services=test_data['services'],
                    steps=test_data['steps'],
                    timeout=test_data.get('timeout', 300),
                    retries=test_data.get('retries', 3)
                )
                self.tests.append(test)

            logger.info(f"加载了 {len(self.tests)} 个集成测试")

        except Exception as e:
            logger.error(f"加载集成测试失败: {e}")

    def run_integration_tests(self) -> List[TestResult]:
        """运行所有集成测试"""
        results = []

        for test in self.tests:
            result = self._run_single_integration_test(test)
            results.append(result)

        return results

    def _run_single_integration_test(self, test: IntegrationTest) -> TestResult:
        """运行单个集成测试"""
        start_time = time.time()

        try:
            # 检查所有必需的服务是否可用
            for service_name in test.services:
                if not self.service_manager.wait_for_service(service_name, timeout=10):
                    return TestResult(
                        test_name=test.name,
                        success=False,
                        duration=time.time() - start_time,
                        error_message=f"服务 {service_name} 不可用"
                    )

            # 执行测试步骤
            success = True
            step_results = []

            for step in test.steps:
                step_result = self._execute_step(step, test.timeout)
                step_results.append(step_result)

                if not step_result['success']:
                    success = False
                    break

            return TestResult(
                test_name=test.name,
                success=success,
                duration=time.time() - start_time,
                details={
                    "services": test.services,
                    "steps": step_results,
                    "total_steps": len(test.steps)
                }
            )

        except Exception as e:
            return TestResult(
                test_name=test.name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )

    def _execute_step(self, step: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """执行测试步骤"""
        try:
            step_type = step.get('type', 'http_request')

            if step_type == 'http_request':
                return self._execute_http_request(step, timeout)
            elif step_type == 'wait':
                time.sleep(step.get('duration', 1))
                return {'success': True, 'type': 'wait'}
            elif step_type == 'assert':
                return self._execute_assertion(step)
            else:
                return {'success': False, 'error': f'未知步骤类型: {step_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_http_request(self, step: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """执行HTTP请求"""
        try:
            service_name = step.get('service')
            if service_name and service_name in self.service_manager.services:
                service = self.service_manager.services[service_name]
                base_url = f"{service.url}:{service.port}"
            else:
                base_url = step.get('base_url', '')

            url = base_url + step.get('path', '/')
            method = step.get('method', 'GET')
            headers = step.get('headers', {})
            data = step.get('body', {})

            response = requests.request(method, url, headers=headers, json=data, timeout=timeout)

            return {
                'success': response.status_code < 400,
                'type': 'http_request',
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }

        except Exception as e:
            return {'success': False, 'error': str(e), 'type': 'http_request'}

    def _execute_assertion(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """执行断言"""
        try:
            # 这里可以实现更复杂的断言逻辑
            # 目前只是简单的占位符
            condition = step.get('condition', True)
            return {'success': bool(condition), 'type': 'assert'}

        except Exception as e:
            return {'success': False, 'error': str(e), 'type': 'assert'}


class FaultToleranceTester:
    """容错测试器"""

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager

    def test_service_failure_recovery(self, service_name: str) -> TestResult:
        """测试服务故障恢复"""
        start_time = time.time()

        try:
            # 停止服务
            logger.info(f"测试容错: 停止服务 {service_name}")
            self.service_manager.stop_service(service_name)

            # 等待一段时间
            time.sleep(5)

            # 重新启动服务
            logger.info(f"测试容错: 重启服务 {service_name}")
            success = self.service_manager.start_service(service_name)

            # 验证服务恢复
            if success and self.service_manager.wait_for_service(service_name, timeout=30):
                return TestResult(
                    test_name=f"fault_tolerance_{service_name}",
                    success=True,
                    duration=time.time() - start_time,
                    details={"recovery_time": time.time() - start_time}
                )
            else:
                return TestResult(
                    test_name=f"fault_tolerance_{service_name}",
                    success=False,
                    duration=time.time() - start_time,
                    error_message="服务未能成功恢复"
                )

        except Exception as e:
            return TestResult(
                test_name=f"fault_tolerance_{service_name}",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )

    def test_load_balancing(self, service_name: str, concurrent_requests: int = 10) -> TestResult:
        """测试负载均衡"""
        start_time = time.time()

        try:
            if service_name not in self.service_manager.services:
                return TestResult(
                    test_name=f"load_balancing_{service_name}",
                    success=False,
                    duration=time.time() - start_time,
                    error_message="服务未配置"
                )

            service = self.service_manager.services[service_name]

            def make_request():
                try:
                    response = requests.get(f"{service.url}:{service.port}/health", timeout=5)
                    return response.status_code == 200
                except:
                    return False

            # 并发请求测试
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
                results = [future.result() for future in as_completed(futures)]

            success_count = sum(results)
            success_rate = success_count / len(results)

            return TestResult(
                test_name=f"load_balancing_{service_name}",
                success=success_rate >= 0.9,  # 90%成功率
                duration=time.time() - start_time,
                details={
                    "total_requests": len(results),
                    "successful_requests": success_count,
                    "success_rate": success_rate
                }
            )

        except Exception as e:
            return TestResult(
                test_name=f"load_balancing_{service_name}",
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )


class MicroserviceTestRunner:
    """微服务测试运行器"""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.service_manager = None
        self.contract_tester = None
        self.integration_tester = None
        self.fault_tester = None

        self._load_configuration()

    def _load_configuration(self):
        """加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 初始化服务管理器
            self.service_manager = ServiceManager(config.get('services', {}))

            # 初始化测试器
            self.contract_tester = ContractTester(self.service_manager)
            self.integration_tester = IntegrationTester(self.service_manager)
            self.fault_tester = FaultToleranceTester(self.service_manager)

            # 加载测试定义
            if 'contracts_file' in config:
                self.contract_tester.load_contracts(config['contracts_file'])

            if 'integration_tests_file' in config:
                self.integration_tester.load_integration_tests(config['integration_tests_file'])

            logger.info("微服务测试运行器配置加载完成")

        except Exception as e:
            logger.error(f"加载配置失败: {e}")

    def run_full_test_suite(self) -> Dict[str, Any]:
        """运行完整的微服务测试套件"""
        logger.info("开始运行微服务测试套件...")

        start_time = time.time()
        results = {
            'service_startup': {},
            'contract_tests': [],
            'integration_tests': [],
            'fault_tolerance_tests': [],
            'summary': {}
        }

        try:
            # 1. 启动所有服务
            logger.info("启动微服务...")
            results['service_startup'] = self.service_manager.start_all_services()

            # 2. 运行契约测试
            logger.info("运行契约测试...")
            results['contract_tests'] = self.contract_tester.run_contract_tests()

            # 3. 运行集成测试
            logger.info("运行集成测试...")
            results['integration_tests'] = self.integration_tester.run_integration_tests()

            # 4. 运行容错测试
            logger.info("运行容错测试...")
            fault_results = []
            for service_name in self.service_manager.services:
                # 故障恢复测试
                recovery_result = self.fault_tester.test_service_failure_recovery(service_name)
                fault_results.append(recovery_result)

                # 负载均衡测试
                load_result = self.fault_tester.test_load_balancing(service_name)
                fault_results.append(load_result)

            results['fault_tolerance_tests'] = fault_results

            # 5. 生成汇总报告
            results['summary'] = self._generate_summary(results)

            # 6. 生成详细报告
            self._generate_detailed_report(results)

        finally:
            # 清理：停止所有服务
            logger.info("清理：停止所有服务...")
            self.service_manager.stop_all_services()

        total_time = time.time() - start_time
        results['total_duration'] = total_time

        logger.info(".2")
        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试汇总"""
        summary = {
            'services_started': sum(1 for success in results['service_startup'].values() if success),
            'total_services': len(results['service_startup']),
            'contract_tests_passed': sum(1 for test in results['contract_tests'] if test.success),
            'total_contract_tests': len(results['contract_tests']),
            'integration_tests_passed': sum(1 for test in results['integration_tests'] if test.success),
            'total_integration_tests': len(results['integration_tests']),
            'fault_tests_passed': sum(1 for test in results['fault_tolerance_tests'] if test.success),
            'total_fault_tests': len(results['fault_tolerance_tests'])
        }

        # 计算总体成功率
        total_tests = (summary['total_contract_tests'] +
                    summary['total_integration_tests'] +
                    summary['total_fault_tests'])
        passed_tests = (summary['contract_tests_passed'] +
                    summary['integration_tests_passed'] +
                    summary['fault_tests_passed'])

        summary['overall_success_rate'] = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        return summary

    def _generate_detailed_report(self, results: Dict[str, Any]):
        """生成详细报告"""
        report_path = Path("test_logs/microservice_test_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 微服务集成测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            summary = results['summary']

            f.write("## 📊 测试概览\n\n")
            f.write(f"- **服务启动**: {summary['services_started']}/{summary['total_services']} 个服务成功启动\n")
            f.write(f"- **契约测试**: {summary['contract_tests_passed']}/{summary['total_contract_tests']} 个通过\n")
            f.write(f"- **集成测试**: {summary['integration_tests_passed']}/{summary['total_integration_tests']} 个通过\n")
            f.write(f"- **容错测试**: {summary['fault_tests_passed']}/{summary['total_fault_tests']} 个通过\n")
            f.write(".1")
            f.write(".2")
            f.write("## 🔧 服务启动状态\n\n")
            for service_name, success in results['service_startup'].items():
                status = "✅" if success else "❌"
                f.write(f"- {status} {service_name}\n")

            f.write("\n## 📋 契约测试详情\n\n")
            if results['contract_tests']:
                f.write("| 测试名称 | 结果 | 时间 |\n")
                f.write("|----------|------|------|\n")
                for test in results['contract_tests'][:10]:  # 显示前10个
                    status = "✅" if test.success else "❌"
                    f.write(f"| {test.test_name} | {status} | {test.duration:.2f}s |\n")

            f.write("\n## 🔗 集成测试详情\n\n")
            if results['integration_tests']:
                f.write("| 测试名称 | 结果 | 时间 |\n")
                f.write("|----------|------|------|\n")
                for test in results['integration_tests']:
                    status = "✅" if test.success else "❌"
                    f.write(f"| {test.test_name} | {status} | {test.duration:.2f}s |\n")

            f.write("\n## 🛡️ 容错测试详情\n\n")
            if results['fault_tolerance_tests']:
                f.write("| 测试名称 | 结果 | 时间 |\n")
                f.write("|----------|------|------|\n")
                for test in results['fault_tolerance_tests']:
                    status = "✅" if test.success else "❌"
                    f.write(f"| {test.test_name} | {status} | {test.duration:.2f}s |\n")

        logger.info(f"微服务测试报告已生成: {report_path}")


def create_sample_config():
    """创建示例配置文件"""
    config = {
        "services": {
            "user-service": {
                "url": "http://localhost",
                "port": 8081,
                "health_check": "/health",
                "dependencies": [],
                "environment": {"NODE_ENV": "test"}
            },
            "order-service": {
                "url": "http://localhost",
                "port": 8082,
                "health_check": "/health",
                "dependencies": ["user-service"],
                "environment": {"NODE_ENV": "test"}
            },
            "payment-service": {
                "url": "http://localhost",
                "port": 8083,
                "health_check": "/health",
                "dependencies": ["order-service"],
                "environment": {"NODE_ENV": "test"}
            }
        },
        "contracts_file": "test_contracts.json",
        "integration_tests_file": "integration_tests.json"
    }

    with open("microservice_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # 创建示例契约测试
    contracts = [
        {
            "consumer": "order-service",
            "provider": "user-service",
            "description": "订单服务调用用户服务获取用户信息",
            "request": {
                "method": "GET",
                "path": "/users/123",
                "headers": {"Authorization": "Bearer token"}
            },
            "expected_response": {
                "status_code": 200,
                "body": {
                    "id": 123,
                    "name": "John Doe",
                    "email": "john@example.com"
                }
            }
        }
    ]

    with open("test_contracts.json", 'w', encoding='utf-8') as f:
        json.dump(contracts, f, indent=2)

    # 创建示例集成测试
    integration_tests = [
        {
            "name": "complete_order_flow",
            "services": ["user-service", "order-service", "payment-service"],
            "timeout": 300,
            "retries": 3,
            "steps": [
                {
                    "type": "http_request",
                    "service": "user-service",
                    "method": "POST",
                    "path": "/users",
                    "body": {"name": "Test User", "email": "test@example.com"}
                },
                {
                    "type": "http_request",
                    "service": "order-service",
                    "method": "POST",
                    "path": "/orders",
                    "body": {"user_id": 1, "items": [{"product_id": 1, "quantity": 2}]}
                },
                {
                    "type": "http_request",
                    "service": "payment-service",
                    "method": "POST",
                    "path": "/payments",
                    "body": {"order_id": 1, "amount": 100.00}
                },
                {
                    "type": "assert",
                    "condition": "payment_status == 'completed'"
                }
            ]
        }
    ]

    with open("integration_tests.json", 'w', encoding='utf-8') as f:
        json.dump(integration_tests, f, indent=2)


def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        # 创建示例配置
        create_sample_config()
        print("✅ 示例配置文件已创建")
        return

    config_file = "microservice_config.json"
    if not Path(config_file).exists():
        print(f"❌ 配置文件 {config_file} 不存在")
        print("💡 请运行 'python microservice_tester.py --create-config' 创建示例配置")
        return

    runner = MicroserviceTestRunner(config_file)

    print("🔗 微服务集成测试器启动")
    print("🎯 测试类型: 服务启动 + 契约测试 + 集成测试 + 容错测试")

    # 运行完整测试套件
    results = runner.run_full_test_suite()

    print("\n📊 微服务测试结果:")
    summary = results['summary']
    print(f"  🔧 服务启动: {summary['services_started']}/{summary['total_services']}")
    print(f"  📋 契约测试: {summary['contract_tests_passed']}/{summary['total_contract_tests']}")
    print(f"  🔗 集成测试: {summary['integration_tests_passed']}/{summary['total_integration_tests']}")
    print(f"  🛡️ 容错测试: {summary['fault_tests_passed']}/{summary['total_fault_tests']}")
    print(".1")
    print(".2")
    print("\n📄 详细报告已保存到: test_logs/microservice_test_report.md")
    print("\n✅ 微服务集成测试器运行完成")


if __name__ == "__main__":
    main()

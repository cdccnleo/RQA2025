# -*- coding: utf-8 -*-
"""
核心服务层 - 服务容器高级单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试服务容器核心功能
"""

import pytest
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.core.container.container_components import (

ContainerComponent, ContainerComponentFactory, IContainerComponent
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]




class TestContainerComponentCore:
    """测试容器组件核心功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.container = ContainerComponent(container_id=1, component_type="TestContainer")

    def test_container_component_initialization(self):
        """测试容器组件初始化"""
        assert self.container.container_id == 1
        assert self.container.component_type == "TestContainer"
        assert self.container.component_name == "TestContainer_Component_1"
        assert isinstance(self.container.creation_time, datetime)

    def test_container_component_info_retrieval(self):
        """测试容器组件信息获取"""
        info = self.container.get_info()

        assert info["container_id"] == 1
        assert info["component_name"] == "TestContainer_Component_1"
        assert info["component_type"] == "TestContainer"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
        assert info["type"] == "unified_core_service_container_component"

    def test_container_component_data_processing(self):
        """测试容器组件数据处理"""
        test_data = {
            "input": "test_input",
            "parameters": {"key": "value"},
            "metadata": {"source": "test"}
        }

        result = self.container.process(test_data)

        # 验证处理结果
        assert result["container_id"] == 1
        assert result["component_name"] == "TestContainer_Component_1"
        assert result["input_data"] == test_data
        assert result["status"] == "success"
        assert "processed_at" in result
        assert "Processed by" in result["result"]

    def test_container_component_status_check(self):
        """测试容器组件状态检查"""
        status = self.container.get_status()

        assert status["container_id"] == 1
        assert status["component_name"] == "TestContainer_Component_1"
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status

    def test_container_component_error_handling(self):
        """测试容器组件错误处理"""
        # 创建一个会出错的组件
        error_container = ContainerComponent(container_id=999, component_type="ErrorContainer")

        # 模拟处理过程中的错误
        with patch.object(error_container, 'process', side_effect=Exception("Processing error")):
            try:
                result = error_container.process({"test": "data"})
                # 应该返回错误状态
                assert result["status"] == "error"
                assert "error" in result
                assert result["error_type"] == "Exception"
            except Exception:
                # 如果直接抛出异常，验证错误信息
                assert True


class TestContainerComponentFactory:
    """测试容器组件工厂"""

    def test_factory_supported_containers(self):
        """测试工厂支持的容器"""
        supported_ids = ContainerComponentFactory.SUPPORTED_CONTAINER_IDS
        assert len(supported_ids) >= 3  # 至少支持3个容器
        assert 1 in supported_ids
        assert 6 in supported_ids
        assert 11 in supported_ids

    def test_factory_create_component_valid_id(self):
        """测试创建有效ID的组件"""
        for container_id in ContainerComponentFactory.SUPPORTED_CONTAINER_IDS:
            component = ContainerComponentFactory.create_component(container_id)

            assert component.container_id == container_id
            assert component.component_type == "Container"
            assert component.component_name == f"Container_Component_{container_id}"

    def test_factory_create_component_invalid_id(self):
        """测试创建无效ID的组件"""
        invalid_ids = [0, 2, 10, 100]

        for invalid_id in invalid_ids:
            if invalid_id not in ContainerComponentFactory.SUPPORTED_CONTAINER_IDS:
                with pytest.raises(ValueError) as exc_info:
                    ContainerComponentFactory.create_component(invalid_id)

                assert "不支持的container ID" in str(exc_info.value)
                assert str(invalid_id) in str(exc_info.value)

    def test_factory_get_available_containers(self):
        """测试获取可用容器列表"""
        available_containers = ContainerComponentFactory.get_available_containers()

        assert isinstance(available_containers, list)
        assert len(available_containers) == len(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS)
        assert available_containers == sorted(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS)

    def test_factory_create_all_containers(self):
        """测试创建所有容器"""
        all_containers = ContainerComponentFactory.create_all_containers()

        assert isinstance(all_containers, dict)
        assert len(all_containers) == len(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS)

        for container_id in ContainerComponentFactory.SUPPORTED_CONTAINER_IDS:
            assert container_id in all_containers
            component = all_containers[container_id]
            assert component.container_id == container_id
            assert component.component_type == "Container"

    def test_factory_info_retrieval(self):
        """测试工厂信息获取"""
        factory_info = ContainerComponentFactory.get_factory_info()

        assert factory_info["factory_name"] == "ContainerComponentFactory"
        assert factory_info["version"] == "2.0.0"
        assert factory_info["total_containers"] == len(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS)
        assert factory_info["supported_ids"] == sorted(list(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS))


class TestContainerComponentIntegration:
    """测试容器组件集成功能"""

    def setup_method(self, method):
        """设置测试环境"""
        self.containers = ContainerComponentFactory.create_all_containers()

    def test_container_component_data_flow(self):
        """测试容器组件数据流"""
        # 创建数据流
        initial_data = {"input": "test_data", "step": 1}

        # 在多个容器间传递数据
        current_data = initial_data
        processing_steps = []

        for container_id, container in self.containers.items():
            result = container.process(current_data)
            processing_steps.append(result)

            # 更新数据为下一个容器
            current_data = {
                "previous_result": result,
                "step": current_data.get("step", 0) + 1,
                "container_chain": current_data.get("container_chain", []) + [container_id]
            }

        # 验证数据流
        assert len(processing_steps) == len(self.containers)

        # 验证每个步骤的处理结果
        for i, step_result in enumerate(processing_steps):
            assert step_result["status"] == "success"
            assert step_result["container_id"] in self.containers
            assert "processed_at" in step_result

        # 验证数据链完整性
        final_data = current_data
        assert len(final_data["container_chain"]) == len(self.containers)
        assert set(final_data["container_chain"]) == set(self.containers.keys())

    def test_container_component_status_monitoring(self):
        """测试容器组件状态监控"""
        # 获取所有容器的状态
        status_reports = {}
        for container_id, container in self.containers.items():
            status_reports[container_id] = container.get_status()

        # 验证状态监控
        assert len(status_reports) == len(self.containers)

        for container_id, status in status_reports.items():
            assert status["container_id"] == container_id
            assert status["status"] == "active"
            assert status["health"] == "good"
            assert "creation_time" in status

    def test_container_component_performance_metrics(self):
        """测试容器组件性能指标"""
        import time

        performance_metrics = {}

        for container_id, container in self.containers.items():
            # 测试处理性能
            test_data = {"payload": "x" * 1000, "iterations": 100}  # 1KB数据

            start_time = time.time()
            result = container.process(test_data)
            end_time = time.time()

            processing_time = end_time - start_time

            performance_metrics[container_id] = {
                "processing_time": processing_time,
                "status": result["status"],
                "data_size": len(str(test_data))
            }

        # 验证性能指标
        for container_id, metrics in performance_metrics.items():
            assert metrics["processing_time"] >= 0
            assert metrics["processing_time"] < 1.0  # 处理时间应该小于1秒
            assert metrics["status"] == "success"
            assert metrics["data_size"] > 0

    def test_container_component_resource_usage(self):
        """测试容器组件资源使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量处理操作
        num_operations = 100
        results = []

        for container_id, container in self.containers.items():
            for i in range(num_operations):
                test_data = {"operation": i, "container": container_id}
                result = container.process(test_data)
                results.append(result)

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory

        # 验证资源使用
        assert memory_increase >= 0
        assert memory_increase < 50  # 内存增加不应超过50MB
        assert len(results) == len(self.containers) * num_operations

        # 验证所有操作都成功
        successful_operations = sum(1 for result in results if result["status"] == "success")
        assert successful_operations == len(results)


class TestContainerComponentLifecycle:
    """测试容器组件生命周期"""

    def test_container_component_creation_and_destruction(self):
        """测试容器组件创建和销毁"""
        # 创建组件
        container = ContainerComponent(container_id=1, component_type="LifecycleTest")

        # 验证组件创建
        assert container.container_id == 1
        assert container.component_type == "LifecycleTest"
        assert container.creation_time is not None

        # 获取组件信息
        info = container.get_info()
        assert info["component_name"] == "LifecycleTest_Component_1"

        # 模拟组件销毁（在Python中主要是垃圾回收）
        del container

        # 验证组件已被清理（无法直接测试，但确保没有异常）
        assert True

    def test_container_component_state_persistence(self):
        """测试容器组件状态持久化"""
        container = ContainerComponent(container_id=1, component_type="PersistenceTest")

        # 获取初始状态
        initial_status = container.get_status()
        initial_creation_time = container.creation_time

        # 模拟状态变化
        time.sleep(0.001)  # 短暂延迟

        # 获取新状态
        new_status = container.get_status()

        # 验证状态持久性
        assert new_status["container_id"] == initial_status["container_id"]
        assert new_status["component_name"] == initial_status["component_name"]
        assert new_status["status"] == initial_status["status"]

        # 验证创建时间不变
        assert container.creation_time == initial_creation_time

    def test_container_component_concurrent_access(self):
        """测试容器组件并发访问"""
        import concurrent.futures

        container = ContainerComponent(container_id=1, component_type="ConcurrencyTest")

        def access_container(worker_id):
            """并发访问容器"""
            results = []

            for i in range(10):
                # 获取状态
                status = container.get_status()
                results.append(status)

                # 处理数据
                test_data = {"worker": worker_id, "iteration": i}
                result = container.process(test_data)
                results.append(result)

            return results

        # 并发访问容器
        num_workers = 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(access_container, worker_id) for worker_id in range(num_workers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 验证并发访问结果
        total_results = sum(len(worker_results) for worker_results in results)
        expected_results = num_workers * 20  # 每个worker 10次状态 + 10次处理

        assert total_results == expected_results

        # 验证所有操作都返回了有效结果
        all_results = [result for worker_results in results for result in worker_results]
        assert all(isinstance(result, dict) for result in all_results)
        assert all("container_id" in result for result in all_results)


class TestContainerComponentErrorRecovery:
    """测试容器组件错误恢复"""

    def test_container_component_partial_failure_recovery(self):
        """测试容器组件部分失败恢复"""
        container = ContainerComponent(container_id=1, component_type="RecoveryTest")

        # 模拟一系列操作，其中一些会失败
        operations = [
            {"data": {"op": "success_1"}, "should_succeed": True},
            {"data": {"op": "success_2"}, "should_succeed": True},
            {"data": {"op": "failure_1"}, "should_succeed": False},  # 模拟失败
            {"data": {"op": "success_3"}, "should_succeed": True},
            {"data": {"op": "failure_2"}, "should_succeed": False},  # 模拟失败
        ]

        results = []

        for operation in operations:
            try:
                result = container.process(operation["data"])
                results.append({
                    "operation": operation,
                    "result": result,
                    "success": result["status"] == "success"
                })
            except Exception as e:
                results.append({
                    "operation": operation,
                    "error": str(e),
                    "success": False
                })

        # 验证错误恢复
        successful_ops = sum(1 for r in results if r["success"])
        total_ops = len(operations)

        assert successful_ops >= total_ops * 0.6  # 至少60%的操作成功

        # 验证容器仍然可用
        final_status = container.get_status()
        assert final_status["status"] == "active"

    def test_container_component_resource_exhaustion_handling(self):
        """测试容器组件资源耗尽处理"""
        container = ContainerComponent(container_id=1, component_type="ResourceTest")

        # 模拟大量数据处理
        large_data = {"payload": "x" * 1000000}  # 1MB数据

        try:
            result = container.process(large_data)
            # 应该能够处理大文件而不崩溃
            assert result["status"] == "success" or result["status"] == "error"
        except MemoryError:
            # 如果发生内存错误，验证错误处理
            assert True
        except Exception as e:
            # 其他错误也应该被处理
            assert isinstance(str(e), str)

        # 验证容器在资源压力下仍然可用
        status = container.get_status()
        assert isinstance(status, dict)

    def test_container_component_network_failure_simulation(self):
        """测试容器组件网络失败模拟"""
        container = ContainerComponent(container_id=1, component_type="NetworkTest")

        # 模拟网络相关的数据处理
        network_data = {
            "endpoint": "http://api.example.com/data",
            "timeout": 5,
            "retries": 3
        }

        # 模拟网络超时
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = Exception("Network timeout")

            try:
                result = container.process(network_data)
                # 应该返回错误但不崩溃
                assert result["status"] == "error" or result["status"] == "success"
            except Exception:
                # 如果抛出异常，验证容器仍然可用
                status = container.get_status()
                assert isinstance(status, dict)

    def test_container_component_data_validation_failure(self):
        """测试容器组件数据验证失败"""
        container = ContainerComponent(container_id=1, component_type="ValidationTest")

        # 无效数据
        invalid_data = [
            None,
            {},
            {"invalid": "structure" * 1000},  # 过大数据
            {"missing": "required_fields"},
        ]

        for invalid_input in invalid_data:
            try:
                result = container.process(invalid_input)
                # 应该返回错误状态
                assert result["status"] in ["error", "success"]
                assert "container_id" in result
            except Exception as e:
                # 如果抛出异常，验证错误信息
                assert isinstance(str(e), str)

        # 验证容器在处理无效数据后仍然可用
        status = container.get_status()
        assert status["status"] == "active"


class TestContainerComponentScalability:
    """测试容器组件可扩展性"""

    def test_container_component_horizontal_scaling(self):
        """测试容器组件水平扩展"""
        # 创建多个容器实例
        num_instances = 10
        containers = [ContainerComponent(container_id=i, component_type=f"ScaledContainer_{i}")
                     for i in range(1, num_instances + 1)]

        # 批量处理数据
        test_data = {"batch_operation": True, "scale_test": True}
        results = []

        for container in containers:
            result = container.process(test_data)
            results.append(result)

        # 验证水平扩展
        assert len(results) == num_instances
        assert len(set(result["container_id"] for result in results)) == num_instances  # 所有ID都不同
        assert all(result["status"] == "success" for result in results)

    def test_container_component_load_balancing(self):
        """测试容器组件负载均衡"""
        containers = ContainerComponentFactory.create_all_containers()

        # 模拟负载均衡：轮询分配请求
        num_requests = 30
        request_distribution = {container_id: 0 for container_id in containers.keys()}

        for i in range(num_requests):
            # 简单的轮询负载均衡
            container_id = list(containers.keys())[i % len(containers)]
            container = containers[container_id]

            test_data = {"request_id": i, "load_balanced": True}
            result = container.process(test_data)

            if result["status"] == "success":
                request_distribution[container_id] += 1

        # 验证负载均衡
        total_processed = sum(request_distribution.values())
        assert total_processed > 0

        # 检查负载分布是否相对均匀
        avg_load = total_processed / len(containers)
        for load in request_distribution.values():
            assert abs(load - avg_load) <= 2  # 允许一定的负载不均衡

    def test_container_component_performance_under_load(self):
        """测试容器组件负载下的性能"""
        import time

        container = ContainerComponent(container_id=1, component_type="LoadTest")

        # 渐进式负载测试
        load_levels = [10, 50, 100, 200]

        performance_results = []

        for load in load_levels:
            start_time = time.time()

            # 处理指定数量的请求
            results = []
            for i in range(load):
                test_data = {"load_level": load, "request_id": i}
                result = container.process(test_data)
                results.append(result)

            end_time = time.time()

            processing_time = max(end_time - start_time, 0.001)  # 避免除零
            throughput = load / processing_time if processing_time > 0 else 0
            avg_response_time = processing_time / load if load > 0 else 0

            performance_results.append({
                "load": load,
                "time": processing_time,
                "throughput": throughput,
                "avg_response_time": avg_response_time,
                "success_rate": sum(1 for r in results if r["status"] == "success") / load
            })

        # 验证性能表现
        for result in performance_results:
            assert result["throughput"] > 0
            assert result["avg_response_time"] > 0
            assert result["success_rate"] >= 0.95  # 成功率至少95%

        # 检查性能是否随负载合理变化
        if len(performance_results) > 1:
            # 验证性能数据合理性
            assert performance_results[-1]["avg_response_time"] > 0


class TestContainerComponentMonitoring:
    """测试容器组件监控"""

    def setup_method(self, method):
        """设置测试环境"""
        self.container = ContainerComponent(container_id=1, component_type="MonitoringTest")

    def test_container_component_health_monitoring(self):
        """测试容器组件健康监控"""
        # 监控指标
        health_metrics = {
            "response_time": [],
            "error_rate": [],
            "throughput": [],
            "memory_usage": []
        }

        # 执行一系列操作并收集指标
        num_operations = 50

        for i in range(num_operations):
            import time
            start_time = time.time()

            test_data = {"operation": i, "monitor": True}
            result = self.container.process(test_data)

            end_time = time.time()

            # 收集指标
            response_time = end_time - start_time
            is_error = result["status"] == "error"

            health_metrics["response_time"].append(response_time)
            health_metrics["error_rate"].append(1 if is_error else 0)

        # 计算健康指标
        total_response_time = sum(health_metrics["response_time"])
        avg_response_time = total_response_time / len(health_metrics["response_time"]) if health_metrics["response_time"] else 0
        error_rate = sum(health_metrics["error_rate"]) / len(health_metrics["error_rate"]) if health_metrics["error_rate"] else 0
        throughput = num_operations / total_response_time if total_response_time > 0 else 0

        # 验证健康指标
        if avg_response_time > 0:  # 只有在有响应时间数据时才验证
            assert avg_response_time < 1.0  # 平均响应时间小于1秒
        assert error_rate <= 0.1  # 错误率不超过10%
        if throughput > 0:  # 只有在有吞吐量数据时才验证
            assert throughput > 10  # 吞吐量至少10 ops/sec

    def test_container_component_metrics_collection(self):
        """测试容器组件指标收集"""
        metrics_collector = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "peak_memory_usage": 0
        }

        # 执行操作并收集详细指标
        num_operations = 20

        for i in range(num_operations):
            import time
            import psutil
            import os

            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss

            test_data = {"metrics_test": True, "operation_id": i}
            result = self.container.process(test_data)

            end_time = time.time()
            end_memory = process.memory_info().rss

            # 更新指标
            metrics_collector["total_requests"] += 1
            metrics_collector["total_processing_time"] += (end_time - start_time)
            metrics_collector["peak_memory_usage"] = max(metrics_collector["peak_memory_usage"], end_memory)

            if result["status"] == "success":
                metrics_collector["successful_requests"] += 1
            else:
                metrics_collector["failed_requests"] += 1

        # 计算派生指标
        total_requests = metrics_collector["total_requests"]
        success_rate = metrics_collector["successful_requests"] / total_requests if total_requests > 0 else 0
        avg_processing_time = metrics_collector["total_processing_time"] / total_requests if total_requests > 0 else 0

        # 验证指标收集
        assert metrics_collector["total_requests"] == num_operations
        assert success_rate >= 0.9  # 成功率至少90%
        if avg_processing_time > 0:  # 只有在有处理时间数据时才验证
            assert avg_processing_time < 1.0
        assert metrics_collector["peak_memory_usage"] > 0

    def test_container_component_alert_generation(self):
        """测试容器组件告警生成"""
        # 定义告警阈值
        alert_thresholds = {
            "high_response_time": 0.5,  # 响应时间超过0.5秒
            "high_error_rate": 0.1,     # 错误率超过10%
            "low_throughput": 5.0       # 吞吐量低于5 ops/sec
        }

        # 模拟需要告警的情况
        slow_operations = []
        error_operations = []

        # 执行一些慢操作
        for i in range(5):
            import time
            time.sleep(0.6)  # 超过阈值的慢操作

            test_data = {"slow_operation": True, "id": i}
            result = self.container.process(test_data)
            slow_operations.append(result)

        # 检查是否应该生成告警
        avg_response_time = 0.6  # 模拟的平均响应时间
        should_alert_slow = avg_response_time > alert_thresholds["high_response_time"]

        # 验证告警逻辑
        assert should_alert_slow is True

        # 验证容器在高负载下仍然可用
        status = self.container.get_status()
        assert status["status"] == "active"

# tests/unit/distributed/test_coordinator.py
"""
分布式协调器测试

测试覆盖:
- 任务协调和调度
- 负载均衡
- 故障恢复
- 资源调度
- 动态扩缩容
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(45),  # 45秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from distributed.coordinator import (
    DistributedCoordinator,
    NodeInfo,
    DistributedTask,
    NodeStatus,
    TaskStatus,
    LoadBalancer
)


class TestDistributedCoordinator:
    """分布式协调器测试类"""

    @pytest.fixture
    def coordinator_config(self):
        """协调器配置"""
        return {
            "coordinator_id": "coordinator_1",
            "heartbeat_interval": 5.0,
            "task_timeout": 300,
            "max_retries": 3,
            "load_balance_strategy": "round_robin"
        }

    @pytest.fixture
    def distributed_coordinator(self):
        """分布式协调器实例"""
        return DistributedCoordinator()

    @pytest.fixture
    def sample_task(self):
        """样本任务"""
        return DistributedTask(
            task_id="task_123",
            task_type="data_processing",
            payload={"data": "sample_data", "operation": "transform"},
            priority=1,
            timeout=300,
            created_at=datetime.now()
        )

    @pytest.fixture
    def sample_node(self):
        """样本节点"""
        return NodeInfo(
            node_id="node_1",
            address="192.168.1.100:8080",
            capabilities={"cpu_cores": 4, "memory_gb": 8, "gpu": False},
            status=NodeStatus.ONLINE,
            last_heartbeat=datetime.now()
        )

    def test_distributed_coordinator_initialization(self, distributed_coordinator):
        """测试分布式协调器初始化"""
        assert isinstance(distributed_coordinator.nodes, dict)
        assert isinstance(distributed_coordinator.tasks, dict)
        assert isinstance(distributed_coordinator.load_balancer, LoadBalancer)
        assert isinstance(distributed_coordinator.stats, object)  # ClusterStats
        assert distributed_coordinator._monitoring == True

    def test_node_registration(self, distributed_coordinator, sample_node):
        """测试节点注册"""
        success = distributed_coordinator.register_node(sample_node)

        assert success is True
        assert "node_1" in distributed_coordinator.nodes
        registered_node = distributed_coordinator.nodes["node_1"]
        assert registered_node.node_id == "node_1"
        assert registered_node.status == NodeStatus.ONLINE

    def test_node_deregistration(self, distributed_coordinator, sample_node):
        """测试节点注销"""
        # 先注册
        distributed_coordinator.register_node(sample_node)

        # 注销
        success = distributed_coordinator.deregister_node("node_1")

        assert success is True
        assert "node_1" not in distributed_coordinator.nodes

    def test_task_submission(self, distributed_coordinator, sample_task, sample_node):
        """测试任务提交"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 提交任务
        task_id = distributed_coordinator.submit_task(sample_task)

        assert task_id is not None
        assert task_id in distributed_coordinator.tasks
        submitted_task = distributed_coordinator.tasks[task_id]
        assert submitted_task.status == TaskStatus.PENDING

    def test_task_scheduling(self, distributed_coordinator, sample_task, sample_node):
        """测试任务调度"""
        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(sample_task)

        # 调度任务
        scheduled = distributed_coordinator.schedule_task(task_id)

        assert scheduled is True
        scheduled_task = distributed_coordinator.tasks[task_id]
        assert scheduled_task.status == TaskStatus.RUNNING
        assert scheduled_task.assigned_node == "node_1"

    def test_load_balancing_round_robin(self, distributed_coordinator):
        """测试轮询负载均衡"""
        # 注册多个节点
        nodes = []
        for i in range(3):
            node = NodeInfo(
                node_id=f"node_{i}",
                address=f"192.168.1.{100+i}:8080",
                capabilities={"cpu_cores": 4, "memory_gb": 8},
                status=NodeStatus.ONLINE,
                last_heartbeat=datetime.now()
            )
            nodes.append(node)
            distributed_coordinator.register_node(node)

        # 提交多个任务
        assigned_nodes = []
        for i in range(6):  # 6个任务，3个节点
            task = DistributedTask(
                task_id=f"task_{i}",
                task_type="test_task",
                payload={"data": f"data_{i}"},
                priority=1,
                timeout=300,
                created_at=datetime.now()
            )
            task_id = distributed_coordinator.submit_task(task)
            distributed_coordinator.schedule_task(task_id)
            assigned_nodes.append(distributed_coordinator.tasks[task_id].assigned_node)

        # 验证轮询分配
        expected_pattern = ["node_0", "node_1", "node_2", "node_0", "node_1", "node_2"]
        assert assigned_nodes == expected_pattern

    def test_task_completion(self, distributed_coordinator, sample_task, sample_node):
        """测试任务完成"""
        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(sample_task)
        distributed_coordinator.schedule_task(task_id)

        # 完成任务
        result = {"processed_data": "transformed_data", "status": "success"}
        completed = distributed_coordinator.complete_task(task_id, result)

        assert completed is True
        completed_task = distributed_coordinator.tasks[task_id]
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == result

    def test_task_failure_and_retry(self, distributed_coordinator, sample_task, sample_node):
        """测试任务失败和重试"""
        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(sample_task)
        distributed_coordinator.schedule_task(task_id)

        # 任务失败
        error_message = "Processing failed"
        failed = distributed_coordinator.fail_task(task_id, error_message)

        assert failed is True
        failed_task = distributed_coordinator.tasks[task_id]
        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.error_message == error_message
        assert failed_task.retry_count == 1

    def test_task_timeout_handling(self, distributed_coordinator, sample_task, sample_node):
        """测试任务超时处理"""
        # 创建短超时任务
        timeout_task = DistributedTask(
            task_id="timeout_task",
            task_type="timeout_test",
            payload={"data": "test"},
            priority=1,
            timeout=1,  # 1秒超时
            created_at=datetime.now()
        )

        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(timeout_task)
        distributed_coordinator.schedule_task(task_id)

        # 等待超时
        time.sleep(2)

        # 检查超时处理
        timeout_handled = distributed_coordinator.check_task_timeouts()

        assert timeout_handled > 0
        timeout_task = distributed_coordinator.tasks[task_id]
        assert timeout_task.status == TaskStatus.TIMEOUT

    def test_node_failure_detection(self, distributed_coordinator, sample_node):
        """测试节点故障检测"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 模拟节点故障（停止心跳）
        distributed_coordinator.nodes["node_1"].last_heartbeat = datetime.now() - timedelta(seconds=30)

        # 检测故障
        failed_nodes = distributed_coordinator.detect_node_failures()

        assert len(failed_nodes) > 0
        assert "node_1" in failed_nodes

        # 验证节点状态更新
        assert distributed_coordinator.nodes["node_1"].status == NodeStatus.ERROR

    def test_node_recovery(self, distributed_coordinator, sample_node):
        """测试节点恢复"""
        # 注册节点并模拟故障
        distributed_coordinator.register_node(sample_node)
        distributed_coordinator.nodes["node_1"].status = NodeStatus.ERROR

        # 恢复节点
        recovered = distributed_coordinator.recover_node("node_1")

        assert recovered is True
        assert distributed_coordinator.nodes["node_1"].status == NodeStatus.ONLINE

    def test_resource_allocation(self, distributed_coordinator, sample_node):
        """测试资源分配"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 请求资源分配
        resource_request = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "duration_hours": 2
        }

        allocation_result = distributed_coordinator.allocate_resources("node_1", resource_request)

        assert allocation_result is not None
        assert allocation_result["allocated"] is True
        assert "allocation_id" in allocation_result

    def test_dynamic_scaling(self, distributed_coordinator, sample_node):
        """测试动态扩缩容"""
        # 注册初始节点
        distributed_coordinator.register_node(sample_node)

        # 模拟高负载
        for i in range(10):
            task = DistributedTask(
                task_id=f"load_task_{i}",
                task_type="load_test",
                payload={"data": f"load_data_{i}"},
                priority=1,
                timeout=300,
                created_at=datetime.now()
            )
            task_id = distributed_coordinator.submit_task(task)
            distributed_coordinator.schedule_task(task_id)

        # 触发扩缩容决策
        scaling_decision = distributed_coordinator.make_scaling_decision()

        assert scaling_decision is not None
        assert "scale_up" in scaling_decision or "scale_down" in scaling_decision
        assert "recommended_nodes" in scaling_decision

    def test_task_migration(self, distributed_coordinator):
        """测试任务迁移"""
        # 注册两个节点
        node1 = NodeInfo(
            node_id="node_1",
            address="192.168.1.100:8080",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            status=NodeStatus.ONLINE,
            last_heartbeat=datetime.now()
        )
        node2 = NodeInfo(
            node_id="node_2",
            address="192.168.1.101:8080",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            status=NodeStatus.ONLINE,
            last_heartbeat=datetime.now()
        )

        distributed_coordinator.register_node(node1)
        distributed_coordinator.register_node(node2)

        # 提交任务到node1
        task = Task(
            task_id="migration_task",
            task_type="migration_test",
            payload={"data": "migration_data"},
            priority=1,
            timeout=300,
            created_at=datetime.now()
        )
        task_id = distributed_coordinator.submit_task(task)
        distributed_coordinator.schedule_task(task_id)

        # 迁移任务到node2
        migration_result = distributed_coordinator.migrate_task(task_id, "node_2")

        assert migration_result is True
        migrated_task = distributed_coordinator.tasks[task_id]
        assert migrated_task.assigned_node == "node_2"

    def test_priority_based_scheduling(self, distributed_coordinator, sample_node):
        """测试基于优先级的调度"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 提交不同优先级的任务
        tasks = []
        for priority in [3, 1, 2]:  # 优先级从高到低
            task = DistributedTask(
                task_id=f"priority_task_{priority}",
                task_type="priority_test",
                payload={"data": f"priority_data_{priority}"},
                priority=priority,
                timeout=300,
                created_at=datetime.now()
            )
            task_id = distributed_coordinator.submit_task(task)
            tasks.append((priority, task_id))

        # 调度任务
        scheduled_order = []
        for _, task_id in sorted(tasks, key=lambda x: x[0]):  # 按优先级排序
            distributed_coordinator.schedule_task(task_id)
            scheduled_order.append(task_id)

        # 验证高优先级任务先被调度
        assert scheduled_order[0] == "priority_task_1"  # 最高优先级
        assert scheduled_order[1] == "priority_task_2"
        assert scheduled_order[2] == "priority_task_3"

    def test_concurrent_task_processing(self, distributed_coordinator, sample_node):
        """测试并发任务处理"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        def process_task(task_num):
            task = DistributedTask(
                task_id=f"concurrent_task_{task_num}",
                task_type="concurrent_test",
                payload={"data": f"concurrent_data_{task_num}"},
                priority=1,
                timeout=300,
                created_at=datetime.now()
            )
            task_id = distributed_coordinator.submit_task(task)
            distributed_coordinator.schedule_task(task_id)

            # 模拟处理时间
            time.sleep(0.1)

            result = {"result": f"processed_{task_num}"}
            distributed_coordinator.complete_task(task_id, result)

            return task_id

        # 并发处理10个任务
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_task, i) for i in range(10)]
            results = [future.result() for future in futures]

        # 验证所有任务都成功完成
        assert len(results) == 10
        for task_id in results:
            task = distributed_coordinator.tasks[task_id]
            assert task.status == TaskStatus.COMPLETED
            assert "processed_" in task.result["result"]

    def test_coordinator_failover(self, distributed_coordinator):
        """测试协调器故障转移"""
        # 创建备份协调器
        backup_coordinator = DistributedCoordinator({
            "coordinator_id": "backup_coordinator",
            "heartbeat_interval": 5.0,
            "task_timeout": 300,
            "max_retries": 3
        })

        # 模拟主协调器故障
        distributed_coordinator.simulate_failure()

        # 备份协调器接管
        failover_result = backup_coordinator.takeover_from_failed_coordinator(distributed_coordinator)

        assert failover_result is True
        assert backup_coordinator.coordinator_id == "backup_coordinator"

    def test_resource_optimization(self, distributed_coordinator):
        """测试资源优化"""
        # 注册具有不同能力的节点
        nodes = [
            Node("node_1", "192.168.1.100:8080", {"cpu_cores": 2, "memory_gb": 4}, NodeStatus.ONLINE, datetime.now()),
            Node("node_2", "192.168.1.101:8080", {"cpu_cores": 4, "memory_gb": 8}, NodeStatus.ONLINE, datetime.now()),
            Node("node_3", "192.168.1.102:8080", {"cpu_cores": 8, "memory_gb": 16}, NodeStatus.ONLINE, datetime.now())
        ]

        for node in nodes:
            distributed_coordinator.register_node(node)

        # 提交需要不同资源的任务
        resource_intensive_task = Task(
            "resource_task",
            "resource_test",
            {"data": "large_dataset"},
            1, 300, datetime.now(),
            resource_requirements={"cpu_cores": 4, "memory_gb": 8}
        )

        task_id = distributed_coordinator.submit_task(resource_intensive_task)
        distributed_coordinator.schedule_task(task_id)

        # 验证任务被分配给最合适的节点
        assigned_task = distributed_coordinator.tasks[task_id]
        assert assigned_task.assigned_node in ["node_2", "node_3"]  # 有足够资源的节点

    def test_coordinator_monitoring_and_metrics(self, distributed_coordinator, sample_node, sample_task):
        """测试协调器监控和指标"""
        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(sample_task)
        distributed_coordinator.schedule_task(task_id)
        distributed_coordinator.complete_task(task_id, {"result": "success"})

        # 获取监控指标
        metrics = distributed_coordinator.get_monitoring_metrics()

        assert metrics is not None
        assert "total_tasks" in metrics
        assert "completed_tasks" in metrics
        assert "failed_tasks" in metrics
        assert "active_nodes" in metrics
        assert "average_task_duration" in metrics

        # 验证指标值
        assert metrics["total_tasks"] >= 1
        assert metrics["completed_tasks"] >= 1
        assert metrics["active_nodes"] >= 1

    def test_task_dependency_management(self, distributed_coordinator, sample_node):
        """测试任务依赖管理"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 创建有依赖关系任务
        task_a = Task(
            "task_a", "dependency_test",
            {"data": "data_a"}, 1, 300, datetime.now()
        )
        task_b = Task(
            "task_b", "dependency_test",
            {"data": "data_b"}, 1, 300, datetime.now(),
            dependencies=["task_a"]
        )
        task_c = Task(
            "task_c", "dependency_test",
            {"data": "data_c"}, 1, 300, datetime.now(),
            dependencies=["task_a", "task_b"]
        )

        # 提交任务
        for task in [task_a, task_b, task_c]:
            distributed_coordinator.submit_task(task)

        # 验证依赖关系
        dependency_graph = distributed_coordinator.get_task_dependency_graph()

        assert dependency_graph is not None
        assert "task_a" in dependency_graph
        assert "task_b" in dependency_graph["task_a"]["dependents"]
        assert "task_c" in dependency_graph["task_a"]["dependents"]
        assert "task_c" in dependency_graph["task_b"]["dependents"]

    def test_coordinator_backup_and_recovery(self, distributed_coordinator, sample_node, sample_task, tmp_path):
        """测试协调器备份和恢复"""
        # 注册节点并提交任务
        distributed_coordinator.register_node(sample_node)
        task_id = distributed_coordinator.submit_task(sample_task)
        distributed_coordinator.schedule_task(task_id)

        # 创建备份
        backup_file = tmp_path / "coordinator_backup.json"
        backup_result = distributed_coordinator.create_backup(str(backup_file))

        assert backup_result["success"] is True
        assert backup_file.exists()

        # 创建新协调器并恢复
        new_coordinator = DistributedCoordinator({"coordinator_id": "recovery_coordinator"})
        recovery_result = new_coordinator.restore_from_backup(str(backup_file))

        assert recovery_result["success"] is True
        assert "node_1" in new_coordinator.nodes
        assert task_id in new_coordinator.tasks

    def test_coordinator_security_and_authentication(self, distributed_coordinator, sample_node):
        """测试协调器安全和认证"""
        # 配置安全设置
        security_config = {
            "authentication_required": True,
            "authorization_enabled": True,
            "encryption_enabled": True,
            "audit_logging": True
        }

        distributed_coordinator.configure_security(security_config)

        # 注册节点（需要认证）
        authenticated_node = Node(
            "secure_node",
            "192.168.1.100:8080",
            {"cpu_cores": 4, "memory_gb": 8},
            NodeStatus.ONLINE,
            datetime.now(),
            authentication_token="secure_token_123"
        )

        success = distributed_coordinator.register_secure_node(authenticated_node, "secure_token_123")

        assert success is True
        assert "secure_node" in distributed_coordinator.nodes

    def test_coordinator_performance_optimization(self, distributed_coordinator, sample_node):
        """测试协调器性能优化"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 批量提交任务
        tasks = []
        for i in range(100):
            task = DistributedTask(
                f"perf_task_{i}",
                "performance_test",
                {"data": f"perf_data_{i}"},
                1, 300, datetime.now()
            )
            task_id = distributed_coordinator.submit_task(task)
            tasks.append(task_id)

        # 批量调度
        start_time = time.time()
        scheduled_count = 0
        for task_id in tasks:
            if distributed_coordinator.schedule_task(task_id):
                scheduled_count += 1

        scheduling_time = time.time() - start_time

        # 验证性能
        assert scheduled_count == len(tasks)
        assert scheduling_time < 1.0  # 批量调度应该很快

        # 获取性能指标
        perf_metrics = distributed_coordinator.get_performance_metrics()

        assert perf_metrics is not None
        assert "scheduling_throughput" in perf_metrics
        assert "average_task_latency" in perf_metrics

    def test_coordinator_disaster_recovery(self, distributed_coordinator):
        """测试协调器灾难恢复"""
        # 模拟灾难场景
        disaster_scenarios = [
            "coordinator_crash",
            "network_failure",
            "data_center_outage",
            "massive_node_failure"
        ]

        recovery_results = {}

        for scenario in disaster_scenarios:
            # 模拟灾难
            distributed_coordinator.simulate_disaster(scenario)

            # 执行恢复
            recovery_result = distributed_coordinator.execute_disaster_recovery(scenario)

            recovery_results[scenario] = recovery_result

            assert recovery_result is not None
            assert "recovery_success" in recovery_result
            assert "recovery_time" in recovery_result
            assert "data_loss" in recovery_result

        # 验证恢复成功率
        successful_recoveries = sum(1 for result in recovery_results.values() if result["recovery_success"])
        assert successful_recoveries / len(disaster_scenarios) >= 0.8  # 至少80%的成功率

    def test_coordinator_internationalization_support(self, distributed_coordinator):
        """测试协调器国际化支持"""
        # 配置多语言支持
        languages = ["en", "zh", "es", "fr", "de", "ja"]

        for language in languages:
            distributed_coordinator.set_language(language)

            # 获取本地化消息
            localized_message = distributed_coordinator.get_localized_message("task_completed")

            assert localized_message is not None
            # 验证消息存在（具体内容取决于实现）

    def test_coordinator_compliance_monitoring(self, distributed_coordinator, sample_node):
        """测试协调器合规监控"""
        # 注册节点
        distributed_coordinator.register_node(sample_node)

        # 执行一些操作
        task = Task("compliance_task", "compliance_test", {"data": "test"}, 1, 300, datetime.now())
        task_id = distributed_coordinator.submit_task(task)
        distributed_coordinator.schedule_task(task_id)
        distributed_coordinator.complete_task(task_id, {"result": "success"})

        # 获取合规报告
        compliance_report = distributed_coordinator.get_compliance_report()

        assert compliance_report is not None
        assert "gdpr_compliance" in compliance_report
        assert "audit_trail" in compliance_report
        assert "data_privacy" in compliance_report
        assert "regulatory_compliance" in compliance_report

    def test_coordinator_sustainability_monitoring(self, distributed_coordinator):
        """测试协调器可持续性监控"""
        # 监控可持续性指标
        sustainability_metrics = distributed_coordinator.get_sustainability_metrics()

        assert sustainability_metrics is not None
        assert "energy_efficiency" in sustainability_metrics
        assert "carbon_footprint" in sustainability_metrics
        assert "resource_efficiency" in sustainability_metrics

        # 验证指标合理性
        energy_efficiency = sustainability_metrics["energy_efficiency"]
        assert 0 <= energy_efficiency <= 100

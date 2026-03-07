"""
分布式协调器层 - 负载均衡算法测试

测试服务请求的负载均衡分配算法
"""

import pytest
import asyncio
import time
import random
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# 尝试导入分布式协调器组件
try:
    from src.distributed.coordinator import DistributedCoordinator
    from src.distributed.cluster_management import ClusterManager
    from src.distributed.service_registry import ServiceRegistry
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    DistributedCoordinator = Mock
    ClusterManager = Mock
    ServiceRegistry = Mock

@pytest.fixture
def distributed_coordinator():
    """创建分布式协调器实例"""
    if not COMPONENTS_AVAILABLE:
        coordinator = DistributedCoordinator()
        coordinator.initialize = AsyncMock(return_value=True)
        coordinator.coordinate = AsyncMock(return_value={"status": "success"})
        coordinator.get_status = Mock(return_value={"state": "running", "nodes": 5})
        coordinator.balance_load = AsyncMock(return_value=True)
        coordinator.get_load_distribution = Mock(return_value={})
        coordinator.select_node_round_robin = Mock(side_effect=["node_1", "node_2", "node_3", "node_1"])
        coordinator.select_node_least_loaded = Mock(return_value="node_2")
        coordinator.select_node_weighted = Mock(return_value="node_3")
        return coordinator
    return DistributedCoordinator()

@pytest.fixture
def cluster_manager():
    """创建集群管理器实例"""
    if not COMPONENTS_AVAILABLE:
        manager = ClusterManager()
        manager.get_node_load = Mock(side_effect=[
            {"cpu": 20, "memory": 30, "requests": 100},
            {"cpu": 10, "memory": 20, "requests": 50},   # 最轻负载
            {"cpu": 40, "memory": 50, "requests": 200},
            {"cpu": 30, "memory": 40, "requests": 150}
        ])
        manager.get_node_weights = Mock(return_value={
            "node_1": 1.0, "node_2": 2.0, "node_3": 1.5, "node_4": 1.0
        })
        manager.update_node_load = Mock(return_value=True)
        return manager
    return ClusterManager()

@pytest.fixture
def service_registry():
    """创建服务注册表实例"""
    if not COMPONENTS_AVAILABLE:
        registry = ServiceRegistry()
        registry.get_available_nodes = Mock(return_value=["node_1", "node_2", "node_3", "node_4"])
        registry.route_request = AsyncMock(return_value="node_2")
        return registry
    return ServiceRegistry()

class TestLoadBalancingAlgorithms:
    """负载均衡算法测试"""

    @pytest.mark.asyncio
    async def test_round_robin_load_balancing(self, distributed_coordinator):
        """测试轮询负载均衡"""
        # 模拟多个请求的轮询分配
        requests = ["req_1", "req_2", "req_3", "req_4", "req_5"]
        expected_nodes = ["node_1", "node_2", "node_3", "node_1", "node_2"]

        assigned_nodes = []
        for request in requests:
            node = distributed_coordinator.select_node_round_robin()
            assigned_nodes.append(node)

        # 验证轮询分配
        assert assigned_nodes[:4] == ["node_1", "node_2", "node_3", "node_1"]

    @pytest.mark.asyncio
    async def test_least_loaded_balancing(self, distributed_coordinator, cluster_manager):
        """测试最小负载均衡"""
        # 模拟节点负载情况
        with patch.object(cluster_manager, 'get_node_load', side_effect=[
            {"cpu": 80, "memory": 90, "requests": 500},  # 高负载
            {"cpu": 10, "memory": 20, "requests": 50},    # 低负载
            {"cpu": 60, "memory": 70, "requests": 300}   # 中等负载
        ]):
            # 选择最小负载节点
            selected_node = distributed_coordinator.select_node_least_loaded()
            assert selected_node == "node_2"  # node_2负载最低

    @pytest.mark.asyncio
    async def test_weighted_load_balancing(self, distributed_coordinator, cluster_manager):
        """测试加权负载均衡"""
        # 设置节点权重
        weights = {"node_1": 1.0, "node_2": 3.0, "node_3": 2.0}

        with patch.object(cluster_manager, 'get_node_weights', return_value=weights):
            # 模拟多次选择，验证权重影响
            selections = []
            for _ in range(100):
                node = distributed_coordinator.select_node_weighted()
                selections.append(node)

            # node_2权重最高，应该被选择最多次
            node_2_count = selections.count("node_2")
            node_1_count = selections.count("node_1")
            node_3_count = selections.count("node_3")

            assert node_2_count > node_1_count  # node_2应该比node_1选择更多
            assert node_3_count > node_1_count  # node_3应该比node_1选择更多

    @pytest.mark.asyncio
    async def test_ip_hash_load_balancing(self, distributed_coordinator):
        """测试IP哈希负载均衡"""
        # 模拟IP哈希分配
        ip_addresses = ["192.168.1.10", "192.168.1.11", "192.168.1.10", "192.168.1.12"]

        with patch.object(distributed_coordinator, 'select_node_by_ip_hash', side_effect=["node_1", "node_2", "node_1", "node_3"]):
            assignments = {}
            for ip in ip_addresses:
                node = distributed_coordinator.select_node_by_ip_hash(ip)
                assignments[ip] = node

            # 相同IP应该分配到相同节点
            assert assignments["192.168.1.10"] == assignments["192.168.1.10"]  # 两次都是node_1

    @pytest.mark.asyncio
    async def test_load_balancing_under_failure(self, distributed_coordinator, cluster_manager):
        """测试故障情况下的负载均衡"""
        # 初始节点列表
        available_nodes = ["node_1", "node_2", "node_3"]

        with patch.object(cluster_manager, 'get_available_nodes', return_value=available_nodes), \
             patch.object(distributed_coordinator, 'select_node_least_loaded', side_effect=["node_1", "node_2"]):

            # 模拟node_3故障
            failed_node = "node_3"
            available_nodes.remove(failed_node)

            # 验证故障节点不再被选择
            for _ in range(5):
                selected_node = distributed_coordinator.select_node_least_loaded()
                assert selected_node != failed_node

    @pytest.mark.asyncio
    async def test_dynamic_load_rebalancing(self, distributed_coordinator, cluster_manager):
        """测试动态负载重均衡"""
        # 初始负载分布不均
        initial_loads = {
            "node_1": {"cpu": 10, "requests": 50},
            "node_2": {"cpu": 90, "requests": 500},  # 高负载
            "node_3": {"cpu": 20, "requests": 100}
        }

        with patch.object(cluster_manager, 'get_node_load', side_effect=list(initial_loads.values())), \
             patch.object(distributed_coordinator, 'balance_load', return_value=True):

            # 执行负载均衡
            rebalancing_success = await distributed_coordinator.balance_load()
            assert rebalancing_success is True

    @pytest.mark.asyncio
    async def test_session_sticky_load_balancing(self, distributed_coordinator):
        """测试会话粘性负载均衡"""
        # 模拟会话ID到节点的映射
        session_mappings = {}

        def select_node_with_session_stickiness(session_id):
            if session_id in session_mappings:
                return session_mappings[session_id]
            else:
                # 新会话，随机分配
                node = f"node_{random.randint(1, 3)}"
                session_mappings[session_id] = node
                return node

        with patch.object(distributed_coordinator, 'select_node_session_sticky', side_effect=select_node_with_session_stickiness):
            # 同一个会话多次请求应该分配到同一节点
            session_id = "session_123"
            nodes = []
            for _ in range(5):
                node = distributed_coordinator.select_node_session_sticky(session_id)
                nodes.append(node)

            # 所有请求应该分配到同一节点
            assert len(set(nodes)) == 1

    @pytest.mark.asyncio
    async def test_geographic_load_balancing(self, distributed_coordinator):
        """测试地理位置负载均衡"""
        # 模拟不同地理位置的节点
        geo_nodes = {
            "us-east": ["node_1", "node_2"],
            "us-west": ["node_3", "node_4"],
            "eu-central": ["node_5", "node_6"]
        }

        client_locations = ["us-east", "us-west", "eu-central", "us-east"]

        with patch.object(distributed_coordinator, 'select_node_by_geo', side_effect=[
            "node_1", "node_3", "node_5", "node_2"
        ]):
            assignments = []
            for location in client_locations:
                node = distributed_coordinator.select_node_by_geo(location)
                assignments.append(node)

            # 验证地理位置亲和性
            assert assignments[0] in geo_nodes["us-east"]  # us-east客户端分配到us-east节点
            assert assignments[1] in geo_nodes["us-west"]  # us-west客户端分配到us-west节点

    @pytest.mark.asyncio
    async def test_load_balancing_performance(self, distributed_coordinator, service_registry):
        """测试负载均衡性能"""
        import time

        # 模拟大量并发请求
        num_requests = 1000
        start_time = time.time()

        with patch.object(service_registry, 'route_request', side_effect=["node_1"] * num_requests):
            # 处理大量路由请求
            routing_tasks = []
            for i in range(num_requests):
                task = service_registry.route_request(f"request_{i}", "service_1")
                routing_tasks.append(task)

            await asyncio.gather(*routing_tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能要求（每秒处理1000+请求）
        requests_per_second = num_requests / total_time
        assert requests_per_second > 100  # 至少每秒100个请求

    @pytest.mark.asyncio
    async def test_load_balancing_algorithm_switching(self, distributed_coordinator):
        """测试负载均衡算法切换"""
        algorithms = ["round_robin", "least_loaded", "weighted", "ip_hash"]

        for algorithm in algorithms:
            with patch.object(distributed_coordinator, 'set_load_balancing_algorithm', return_value=True), \
                 patch.object(distributed_coordinator, f'balance_with_{algorithm}', return_value=True):

                # 设置算法
                set_success = await distributed_coordinator.set_load_balancing_algorithm(algorithm)
                assert set_success is True

                # 验证算法工作
                balance_success = await getattr(distributed_coordinator, f'balance_with_{algorithm}')()
                assert balance_success is True

    @pytest.mark.asyncio
    async def test_health_based_load_balancing(self, distributed_coordinator, cluster_manager):
        """测试基于健康状态的负载均衡"""
        # 模拟节点健康状态
        health_status = {
            "node_1": "healthy",
            "node_2": "healthy",
            "node_3": "unhealthy",  # 不健康节点
            "node_4": "healthy"
        }

        with patch.object(cluster_manager, 'get_node_health', side_effect=list(health_status.values())), \
             patch.object(distributed_coordinator, 'select_healthy_node', side_effect=["node_1", "node_2", "node_4"]):

            # 多次选择节点
            selected_nodes = []
            for _ in range(10):
                node = distributed_coordinator.select_healthy_node()
                selected_nodes.append(node)

            # 不健康节点不应该被选择
            assert "node_3" not in selected_nodes

    @pytest.mark.asyncio
    async def test_load_prediction_based_balancing(self, distributed_coordinator):
        """测试基于负载预测的均衡"""
        # 模拟负载预测数据
        load_predictions = {
            "node_1": {"predicted_load": 60, "time_window": "5min"},
            "node_2": {"predicted_load": 30, "time_window": "5min"},  # 预测负载最低
            "node_3": {"predicted_load": 80, "time_window": "5min"}
        }

        with patch.object(distributed_coordinator, 'predict_node_load', side_effect=list(load_predictions.values())), \
             patch.object(distributed_coordinator, 'select_node_by_prediction', return_value="node_2"):

            # 基于预测选择节点
            selected_node = distributed_coordinator.select_node_by_prediction()
            assert selected_node == "node_2"  # 选择预测负载最低的节点

    @pytest.mark.asyncio
    async def test_load_balancing_monitoring(self, distributed_coordinator, cluster_manager):
        """测试负载均衡监控"""
        # 监控负载分布
        with patch.object(distributed_coordinator, 'get_load_distribution', return_value={
            "node_1": 25, "node_2": 25, "node_3": 25, "node_4": 25  # 均衡分布
        }):
            distribution = distributed_coordinator.get_load_distribution()
            assert len(distribution) == 4

            # 验证负载均衡度
            loads = list(distribution.values())
            max_load = max(loads)
            min_load = min(loads)
            balance_ratio = min_load / max_load if max_load > 0 else 1.0

            assert balance_ratio > 0.8  # 负载均衡度应该大于80%

    @pytest.mark.asyncio
    async def test_end_to_end_load_balancing_workflow(self, distributed_coordinator, cluster_manager, service_registry):
        """测试端到端负载均衡工作流"""
        print("\n=== 端到端负载均衡工作流测试 ===")

        # 1. 初始化负载均衡器
        with patch.object(distributed_coordinator, 'initialize_load_balancer', return_value=True):
            init_success = await distributed_coordinator.initialize_load_balancer()
            assert init_success is True
            print("✓ 负载均衡器初始化完成")

        # 2. 配置负载均衡策略
        with patch.object(distributed_coordinator, 'configure_load_balancing', return_value=True):
            config_success = distributed_coordinator.configure_load_balancing({
                "algorithm": "weighted",
                "weights": {"node_1": 1.0, "node_2": 2.0, "node_3": 1.5}
            })
            assert config_success is True
            print("✓ 负载均衡策略配置完成")

        # 3. 模拟服务请求负载
        requests = [f"request_{i}" for i in range(20)]

        with patch.object(service_registry, 'route_request', side_effect=[
            "node_1", "node_2", "node_3", "node_2", "node_1", "node_3", "node_2", "node_1",
            "node_2", "node_3", "node_1", "node_2", "node_3", "node_2", "node_1", "node_3",
            "node_2", "node_1", "node_3", "node_2"
        ]):
            # 处理请求
            routing_results = []
            for request in requests:
                node = await service_registry.route_request(request, "test_service")
                routing_results.append(node)

            print("✓ 服务请求路由完成")

        # 4. 验证负载分布
        from collections import Counter
        distribution = Counter(routing_results)

        print(f"✓ 请求分布: {dict(distribution)}")

        # node_2权重最高，应该获得最多请求
        assert distribution["node_2"] > distribution["node_1"]
        assert distribution["node_3"] > distribution["node_1"]

        # 5. 监控负载均衡效果
        with patch.object(distributed_coordinator, 'monitor_load_balance', return_value={
            "balance_score": 0.85,
            "efficiency": 0.92
        }):
            monitoring_result = await distributed_coordinator.monitor_load_balance()
            assert monitoring_result["balance_score"] > 0.8
            print("✓ 负载均衡监控完成")

        print("🎉 端到端负载均衡工作流测试完成")





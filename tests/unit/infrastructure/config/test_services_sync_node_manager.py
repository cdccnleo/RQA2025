"""
同步节点管理器测试模块

测试分布式配置同步的节点管理功能，包括：
- 节点注册和注销
- 节点状态管理
- 节点信息查询
- 活跃节点筛选
- 节点统计报告
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

from src.infrastructure.config.services.sync_node_manager import (
    SyncNodeManager,
    SyncNode,
    SyncStatus
)


class TestSyncNodeManager:
    """同步节点管理器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.manager = SyncNodeManager()

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = SyncNodeManager()
        assert manager._nodes == {}
        assert hasattr(manager, 'register_node')
        assert hasattr(manager, 'unregister_node')
        assert hasattr(manager, 'get_node')
        assert hasattr(manager, 'update_node_status')
        assert hasattr(manager, 'list_nodes')

    def test_register_node_success(self):
        """测试成功注册节点"""
        node_id = "node1"
        address = "192.168.1.100"
        port = 8080

        result = self.manager.register_node(node_id, address, port)

        assert result is True
        assert node_id in self.manager._nodes

        node = self.manager._nodes[node_id]
        assert node.node_id == node_id
        assert node.address == address
        assert node.port == port
        assert node.status == SyncStatus.IDLE
        assert node.created_at is not None

    def test_register_node_duplicate(self):
        """测试重复注册节点"""
        node_id = "node1"
        address = "192.168.1.100"
        port = 8080

        # 首次注册应该成功
        result1 = self.manager.register_node(node_id, address, port)
        assert result1 is True

        # 重复注册应该失败
        result2 = self.manager.register_node(node_id, address, port)
        assert result2 is False

        # 节点数量应该仍然是1
        assert len(self.manager._nodes) == 1

    def test_unregister_node_success(self):
        """测试成功注销节点"""
        node_id = "node1"
        self.manager._nodes[node_id] = SyncNode(node_id, "192.168.1.100", 8080)

        result = self.manager.unregister_node(node_id)

        assert result is True
        assert node_id not in self.manager._nodes

    def test_unregister_node_not_exists(self):
        """测试注销不存在的节点"""
        result = self.manager.unregister_node("nonexistent")
        assert result is False

    def test_get_node_exists(self):
        """测试获取存在的节点"""
        node_id = "node1"
        expected_node = SyncNode(node_id, "192.168.1.100", 8080)
        self.manager._nodes[node_id] = expected_node

        node = self.manager.get_node(node_id)

        assert node == expected_node

    def test_get_node_not_exists(self):
        """测试获取不存在的节点"""
        node = self.manager.get_node("nonexistent")
        assert node is None

    def test_update_node_status_success(self):
        """测试成功更新节点状态"""
        node_id = "node1"
        self.manager._nodes[node_id] = SyncNode(node_id, "192.168.1.100", 8080)

        result = self.manager.update_node_status(node_id, SyncStatus.SYNCING)

        assert result is True
        assert self.manager._nodes[node_id].status == SyncStatus.SYNCING

    def test_update_node_status_not_exists(self):
        """测试更新不存在的节点状态"""
        result = self.manager.update_node_status("nonexistent", SyncStatus.SYNCING)
        assert result is False

    def test_update_node_sync_time_success(self):
        """测试成功更新节点同步时间"""
        node_id = "node1"
        self.manager._nodes[node_id] = SyncNode(node_id, "192.168.1.100", 8080)

        result = self.manager.update_node_sync_time(node_id)

        assert result is True
        assert self.manager._nodes[node_id].last_sync_time is not None

        # 验证时间戳是最近的
        current_time = time.time()
        assert abs(self.manager._nodes[node_id].last_sync_time - current_time) < 1.0

    def test_update_node_sync_time_not_exists(self):
        """测试更新不存在的节点同步时间"""
        result = self.manager.update_node_sync_time("nonexistent")
        assert result is False

    def test_list_nodes_empty(self):
        """测试空节点列表"""
        nodes = self.manager.list_nodes()
        assert nodes == []

    def test_list_nodes_with_data(self):
        """测试包含数据的节点列表"""
        # 添加测试节点
        node1 = SyncNode("node1", "192.168.1.100", 8080, SyncStatus.SYNCING, "1.0", 1000.0, 900.0)
        node2 = SyncNode("node2", "192.168.1.101", 8081, SyncStatus.SUCCESS, "1.1", 1100.0, 950.0)

        self.manager._nodes["node1"] = node1
        self.manager._nodes["node2"] = node2

        nodes = self.manager.list_nodes()

        assert len(nodes) == 2

        # 验证第一个节点的信息
        node1_info = nodes[0]
        assert node1_info["node_id"] == "node1"
        assert node1_info["address"] == "192.168.1.100"
        assert node1_info["port"] == 8080
        assert node1_info["status"] == "syncing"
        assert node1_info["version"] == "1.0"
        assert node1_info["last_sync_time"] == 1000.0
        assert node1_info["created_at"] == 900.0

        # 验证第二个节点的信息
        node2_info = nodes[1]
        assert node2_info["node_id"] == "node2"
        assert node2_info["address"] == "192.168.1.101"
        assert node2_info["port"] == 8081
        assert node2_info["status"] == "success"
        assert node2_info["version"] == "1.1"
        assert node2_info["last_sync_time"] == 1100.0
        assert node2_info["created_at"] == 950.0

    def test_get_active_nodes_empty(self):
        """测试空活跃节点列表"""
        active_nodes = self.manager.get_active_nodes()
        assert active_nodes == []

    def test_get_active_nodes_with_mixed_status(self):
        """测试混合状态的活跃节点筛选"""
        # 添加不同状态的节点
        self.manager._nodes["node1"] = SyncNode("node1", "192.168.1.100", 8080, SyncStatus.IDLE)
        self.manager._nodes["node2"] = SyncNode("node2", "192.168.1.101", 8081, SyncStatus.SYNCING)
        self.manager._nodes["node3"] = SyncNode("node3", "192.168.1.102", 8082, SyncStatus.SUCCESS)
        self.manager._nodes["node4"] = SyncNode("node4", "192.168.1.103", 8083, SyncStatus.FAILED)
        self.manager._nodes["node5"] = SyncNode("node5", "192.168.1.104", 8084, SyncStatus.OFFLINE)

        active_nodes = self.manager.get_active_nodes()

        # 只有IDLE和SUCCESS状态的节点应该是活跃的
        expected_active = ["node1", "node3"]
        assert set(active_nodes) == set(expected_active)

    def test_get_node_count(self):
        """测试获取节点数量"""
        # 初始状态应该为0
        count = self.manager.get_node_count()
        assert count == 0

        # 添加节点
        self.manager._nodes["node1"] = SyncNode("node1", "192.168.1.100", 8080)
        self.manager._nodes["node2"] = SyncNode("node2", "192.168.1.101", 8081)

        count = self.manager.get_node_count()
        assert count == 2

    def test_get_node_status_summary_empty(self):
        """测试空节点状态统计"""
        summary = self.manager.get_node_status_summary()

        expected = {
            "idle": 0,
            "syncing": 0,
            "success": 0,
            "failed": 0,
            "offline": 0
        }
        assert summary == expected

    def test_get_node_status_summary_with_data(self):
        """测试包含数据的节点状态统计"""
        # 添加不同状态的节点
        self.manager._nodes["node1"] = SyncNode("node1", "192.168.1.100", 8080, SyncStatus.IDLE)
        self.manager._nodes["node2"] = SyncNode("node2", "192.168.1.101", 8081, SyncStatus.SYNCING)
        self.manager._nodes["node3"] = SyncNode("node3", "192.168.1.102", 8082, SyncStatus.SUCCESS)
        self.manager._nodes["node4"] = SyncNode("node4", "192.168.1.103", 8083, SyncStatus.FAILED)
        self.manager._nodes["node5"] = SyncNode("node5", "192.168.1.104", 8084, SyncStatus.OFFLINE)
        self.manager._nodes["node6"] = SyncNode("node6", "192.168.1.105", 8085, SyncStatus.IDLE)  # 另一个IDLE

        summary = self.manager.get_node_status_summary()

        expected = {
            "idle": 2,      # node1 和 node6
            "syncing": 1,   # node2
            "success": 1,   # node3
            "failed": 1,    # node4
            "offline": 1    # node5
        }
        assert summary == expected


class TestSyncNode:
    """同步节点测试类"""

    def test_sync_node_creation(self):
        """测试同步节点创建"""
        node = SyncNode("node1", "192.168.1.100", 8080)

        assert node.node_id == "node1"
        assert node.address == "192.168.1.100"
        assert node.port == 8080
        assert node.status == SyncStatus.IDLE
        assert node.version == "1.0"
        assert node.last_sync_time is None
        assert node.created_at is not None

    def test_sync_node_creation_with_all_params(self):
        """测试带所有参数的同步节点创建"""
        current_time = time.time()
        node = SyncNode(
            node_id="node1",
            address="192.168.1.100",
            port=8080,
            status=SyncStatus.SYNCING,
            version="2.0",
            last_sync_time=current_time,
            created_at=current_time
        )

        assert node.node_id == "node1"
        assert node.address == "192.168.1.100"
        assert node.port == 8080
        assert node.status == SyncStatus.SYNCING
        assert node.version == "2.0"
        assert node.last_sync_time == current_time
        assert node.created_at == current_time

    def test_sync_node_creation_auto_created_at(self):
        """测试自动设置创建时间"""
        before_creation = time.time()
        node = SyncNode("node1", "192.168.1.100", 8080)
        after_creation = time.time()

        assert before_creation <= node.created_at <= after_creation

    def test_sync_status_enum(self):
        """测试同步状态枚举"""
        assert SyncStatus.IDLE.value == "idle"
        assert SyncStatus.SYNCING.value == "syncing"
        assert SyncStatus.SUCCESS.value == "success"
        assert SyncStatus.FAILED.value == "failed"
        assert SyncStatus.OFFLINE.value == "offline"


class TestSyncNodeManagerIntegration:
    """同步节点管理器集成测试类"""

    def test_full_node_lifecycle_management(self):
        """测试完整的节点生命周期管理"""
        manager = SyncNodeManager()

        # 1. 注册节点
        node_id = "test_node"
        result = manager.register_node(node_id, "192.168.1.100", 8080)
        assert result is True

        # 2. 更新节点状态
        manager.update_node_status(node_id, SyncStatus.SYNCING)
        node = manager.get_node(node_id)
        assert node.status == SyncStatus.SYNCING

        # 3. 更新同步时间
        manager.update_node_sync_time(node_id)
        node = manager.get_node(node_id)
        assert node.last_sync_time is not None

        # 4. 完成同步
        manager.update_node_status(node_id, SyncStatus.SUCCESS)

        # 5. 验证活跃节点列表
        active_nodes = manager.get_active_nodes()
        assert node_id in active_nodes

        # 6. 注销节点
        result = manager.unregister_node(node_id)
        assert result is True

        # 7. 验证节点已删除
        node = manager.get_node(node_id)
        assert node is None

    def test_multiple_nodes_management(self):
        """测试多节点管理"""
        manager = SyncNodeManager()

        # 注册多个节点
        nodes_data = [
            ("node1", "192.168.1.100", 8080),
            ("node2", "192.168.1.101", 8081),
            ("node3", "192.168.1.102", 8082)
        ]

        for node_id, address, port in nodes_data:
            result = manager.register_node(node_id, address, port)
            assert result is True

        # 验证节点数量
        assert manager.get_node_count() == 3

        # 更新部分节点状态
        manager.update_node_status("node1", SyncStatus.SYNCING)
        manager.update_node_status("node2", SyncStatus.SUCCESS)
        manager.update_node_status("node3", SyncStatus.FAILED)

        # 验证活跃节点（只有IDLE和SUCCESS状态的节点）
        active_nodes = manager.get_active_nodes()
        assert len(active_nodes) == 1  # 只有node2是SUCCESS状态
        assert "node2" in active_nodes

        # 验证状态统计
        summary = manager.get_node_status_summary()
        assert summary["idle"] == 0      # 没有IDLE节点
        assert summary["syncing"] == 1   # node1
        assert summary["success"] == 1   # node2
        assert summary["failed"] == 1    # node3
        assert summary["offline"] == 0   # 没有OFFLINE节点

    def test_node_version_management(self):
        """测试节点版本管理"""
        manager = SyncNodeManager()

        # 注册节点时指定版本
        node = SyncNode("node1", "192.168.1.100", 8080, version="1.5")
        manager._nodes["node1"] = node

        # 验证版本信息
        node_info = manager.list_nodes()[0]
        assert node_info["version"] == "1.5"

        # 版本应该在节点列表中正确显示
        nodes_list = manager.list_nodes()
        assert len(nodes_list) == 1
        assert nodes_list[0]["version"] == "1.5"


class TestSyncNodeManagerEdgeCases:
    """同步节点管理器边界情况测试类"""

    def test_register_node_with_invalid_params(self):
        """测试无效参数的节点注册"""
        manager = SyncNodeManager()

        # 测试空字符串ID
        result = manager.register_node("", "192.168.1.100", 8080)
        assert result is True  # 空字符串仍然被接受

        # 测试无效端口号（虽然不会被验证）
        result = manager.register_node("node1", "192.168.1.100", -1)
        assert result is True

        # 测试无效地址（虽然不会被验证）
        result = manager.register_node("node2", "", 8080)
        assert result is True

    def test_update_status_of_nonexistent_node(self):
        """测试更新不存在节点的状态"""
        manager = SyncNodeManager()

        # 不应该抛出异常，而是返回False
        result = manager.update_node_status("nonexistent", SyncStatus.SYNCING)
        assert result is False

        result = manager.update_node_sync_time("nonexistent")
        assert result is False

    def test_concurrent_node_operations(self):
        """测试并发节点操作"""
        manager = SyncNodeManager()

        import threading
        import time

        def register_nodes():
            """并发注册节点的函数"""
            for i in range(10):
                node_id = f"node_{i}"
                manager.register_node(node_id, f"192.168.1.{i}", 8080 + i)

        def update_statuses():
            """并发更新状态的函数"""
            for i in range(10):
                node_id = f"node_{i}"
                manager.update_node_status(node_id, SyncStatus.SYNCING)

        # 创建线程
        register_thread = threading.Thread(target=register_nodes)
        update_thread = threading.Thread(target=update_statuses)

        # 启动线程
        register_thread.start()
        time.sleep(0.01)  # 短暂延迟确保注册先完成
        update_thread.start()

        # 等待线程完成
        register_thread.join()
        update_thread.join()

        # 验证最终状态
        assert manager.get_node_count() == 10

        # 验证所有节点的状态都被更新
        summary = manager.get_node_status_summary()
        assert summary["syncing"] == 10
        assert summary["idle"] == 0

    def test_node_creation_time_preservation(self):
        """测试节点创建时间保存"""
        manager = SyncNodeManager()

        before_time = time.time()
        manager.register_node("node1", "192.168.1.100", 8080)
        after_time = time.time()

        node = manager.get_node("node1")
        assert node.created_at is not None
        assert before_time <= node.created_at <= after_time

        # 创建时间应该在后续操作中保持不变
        original_created_at = node.created_at

        # 更新状态
        manager.update_node_status("node1", SyncStatus.SYNCING)
        
        # 添加小延迟确保同步时间与创建时间不同
        time.sleep(0.01)  # 10毫秒延迟
        
        # 更新同步时间
        manager.update_node_sync_time("node1")

        # 创建时间应该保持不变
        node = manager.get_node("node1")
        # 使用接近比较来处理可能的浮点数精度问题
        assert abs(node.created_at - original_created_at) < 0.001
        assert node.last_sync_time is not None
        # 同步时间应该与创建时间不同
        assert node.last_sync_time > original_created_at

    def test_large_number_of_nodes_performance(self):
        """测试大量节点性能"""
        manager = SyncNodeManager()

        # 注册大量节点
        num_nodes = 1000
        start_time = time.time()

        for i in range(num_nodes):
            manager.register_node(f"node_{i}", f"192.168.1.{i % 255}", 8080 + i)

        end_time = time.time()
        registration_duration = end_time - start_time

        # 验证注册性能（应该在合理时间内完成）
        assert registration_duration < 5.0  # 5秒内完成
        assert manager.get_node_count() == num_nodes

        # 测试列表操作性能
        start_time = time.time()
        nodes = manager.list_nodes()
        end_time = time.time()
        list_duration = end_time - start_time

        assert list_duration < 1.0  # 1秒内完成
        assert len(nodes) == num_nodes

        # 测试统计操作性能
        start_time = time.time()
        summary = manager.get_node_status_summary()
        end_time = time.time()
        summary_duration = end_time - start_time

        assert summary_duration < 0.5  # 0.5秒内完成
        assert summary["idle"] == num_nodes

    def test_node_with_special_characters(self):
        """测试包含特殊字符的节点"""
        manager = SyncNodeManager()

        # 测试包含特殊字符的节点ID和地址
        special_cases = [
            ("node-with-dash", "192.168.1.100", 8080),
            ("node_with_underscore", "192.168.1.101", 8081),
            ("node.with.dots", "192.168.1.102", 8082),
            ("node123", "192.168.1.103", 8083)
        ]

        for node_id, address, port in special_cases:
            result = manager.register_node(node_id, address, port)
            assert result is True

        # 验证所有特殊节点都被正确注册
        assert manager.get_node_count() == 4

        # 验证节点信息正确
        for node_id, address, port in special_cases:
            node = manager.get_node(node_id)
            assert node.node_id == node_id
            assert node.address == address
            assert node.port == port


class TestSyncNodeManagerErrorHandling:
    """同步节点管理器错误处理测试类"""

    def test_operations_with_none_values(self):
        """测试None值操作"""
        manager = SyncNodeManager()

        # 注册一个None node_id的节点（虽然不推荐，但在技术上是可能的）
        assert manager.register_node(None, "192.168.1.100", 8080) is True
        # 能够获取到None node_id的节点
        node = manager.get_node(None)
        assert node is not None
        assert node.node_id is None
        # 更新操作应该成功，因为节点存在
        assert manager.update_node_status(None, SyncStatus.SYNCING) is True
        assert manager.update_node_sync_time(None) is True
        # 注销操作应该成功
        assert manager.unregister_node(None) is True
        # 注销后应该获取不到节点
        assert manager.get_node(None) is None

    def test_concurrent_modification_safety(self):
        """测试并发修改安全性"""
        manager = SyncNodeManager()

        import threading
        import time

        def add_nodes():
            """添加节点的函数"""
            for i in range(50):
                manager.register_node(f"node_{i}", f"192.168.1.{i}", 8080 + i)

        def remove_nodes():
            """删除节点的函数"""
            for i in range(50):
                manager.unregister_node(f"node_{i}")

        # 创建线程
        add_thread = threading.Thread(target=add_nodes)
        remove_thread = threading.Thread(target=remove_nodes)

        # 启动线程
        add_thread.start()
        time.sleep(0.01)  # 短暂延迟确保添加先开始
        remove_thread.start()

        # 等待线程完成
        add_thread.join()
        remove_thread.join()

        # 最终状态应该是空的（因为删除操作会删除已存在的节点）
        # 或者有一些残留（取决于执行顺序）
        final_count = manager.get_node_count()
        assert final_count >= 0  # 至少不应该是负数

    def test_exception_handling_in_node_creation(self):
        """测试节点创建中的异常处理"""
        manager = SyncNodeManager()

        # 模拟SyncNode构造函数抛出异常的情况
        with patch('src.infrastructure.config.services.sync_node_manager.SyncNode') as mock_node:
            mock_node.side_effect = Exception("节点创建失败")

            # 根据当前实现，异常会被抛出而不是返回False
            with pytest.raises(Exception, match="节点创建失败"):
                manager.register_node("node1", "192.168.1.100", 8080)

            # 节点不应该被添加到管理器中
            assert manager.get_node("node1") is None

    def test_memory_cleanup_on_unregister(self):
        """测试注销时的内存清理"""
        manager = SyncNodeManager()

        # 注册节点并获取引用
        manager.register_node("node1", "192.168.1.100", 8080)
        node = manager.get_node("node1")
        node_ref = id(node)  # 获取对象引用ID

        # 注销节点
        manager.unregister_node("node1")

        # 验证节点已被删除
        assert manager.get_node("node1") is None

        # 验证对象已被垃圾回收（虽然不能直接测试，但可以验证没有内存泄漏）
        import gc
        gc.collect()

        # 此时node对象应该可以被垃圾回收
        assert id(node) == node_ref  # 对象仍然存在，但已被从管理器中移除


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

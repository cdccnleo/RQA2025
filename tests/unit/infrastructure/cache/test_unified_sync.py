#!/usr/bin/env python3
"""
基础设施层 - 统一同步功能测试

测试unified_sync.py中的分布式同步核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.cache.distributed.unified_sync import (
    UnifiedSync,
    ConfigSyncService,
    SyncConfig,
    start_sync,
    stop_sync
)


class TestConfigSyncService:
    """测试配置同步服务"""

    def setup_method(self):
        """测试前准备"""
        self.service = ConfigSyncService()

    def test_initialization(self):
        """测试初始化"""
        assert self.service.nodes == {}
        assert self.service.sync_history == []
        assert self.service.callbacks == []

    def test_register_node(self):
        """测试注册节点"""
        result = self.service.register_node("node1", "localhost", 8080)
        assert result is True
        assert "node1" in self.service.nodes
        assert self.service.nodes["node1"]["address"] == "localhost"
        assert self.service.nodes["node1"]["port"] == 8080

    def test_unregister_node(self):
        """测试注销节点"""
        # 先注册节点
        self.service.register_node("node1", "localhost", 8080)
        assert "node1" in self.service.nodes

        # 注销节点
        result = self.service.unregister_node("node1")
        assert result is True
        assert "node1" not in self.service.nodes

    def test_unregister_nonexistent_node(self):
        """测试注销不存在的节点"""
        result = self.service.unregister_node("nonexistent")
        assert result is False

    def test_sync_config(self):
        """测试同步配置"""
        config_data = {"key": "value"}
        target_nodes = ["node1", "node2"]

        result = self.service.sync_config(config_data, target_nodes)
        assert result["success"] is True
        assert result["synced_nodes"] == target_nodes
        assert config_data in self.service.sync_history

    def test_sync_config_no_target_nodes(self):
        """测试同步配置（不指定目标节点）"""
        config_data = {"key": "value"}

        result = self.service.sync_config(config_data)
        assert result["success"] is True
        assert result["synced_nodes"] == []
        assert config_data in self.service.sync_history

    def test_sync_data(self):
        """测试同步数据"""
        data = {"data": "test"}
        target_nodes = ["node1"]

        result = self.service.sync_data(data, target_nodes)
        assert result is True
        assert data in self.service.sync_history

    def test_get_history(self):
        """测试获取同步历史"""
        # 添加一些历史记录
        self.service.sync_config({"config": "1"})
        self.service.sync_data({"data": "1"})

        history = self.service.get_history()
        assert len(history) == 2
        assert {"config": "1"} in history
        assert {"data": "1"} in history

    def test_get_conflicts(self):
        """测试获取冲突"""
        conflicts = self.service.get_conflicts()
        assert conflicts == []

    def test_resolve_conflict(self):
        """测试解决冲突"""
        result = self.service.resolve_conflict("conflict_id")
        assert result is True

    def test_add_callback_with_event_type(self):
        """测试添加带事件类型的回调"""
        callback = Mock()
        result = self.service.add_callback(callback, "sync_complete")
        assert result is True
        assert ("sync_complete", callback) in self.service.callbacks

    def test_add_callback_without_event_type(self):
        """测试添加不带事件类型的回调"""
        callback = Mock()
        result = self.service.add_callback(callback)
        assert result is True
        assert callback in self.service.callbacks

    def test_remove_callback_with_event_type(self):
        """测试移除带事件类型的回调"""
        callback = Mock()
        self.service.add_callback(callback, "sync_complete")
        assert len(self.service.callbacks) == 1

        # 注意：remove_callback只接受callback参数，不接受event_type
        result = self.service.remove_callback(callback)
        assert result is True
        assert len(self.service.callbacks) == 0

    def test_remove_callback_without_event_type(self):
        """测试移除不带事件类型的回调"""
        callback = Mock()
        self.service.add_callback(callback)
        assert len(self.service.callbacks) == 1

        result = self.service.remove_callback(callback)
        assert result is True
        assert len(self.service.callbacks) == 0

    def test_remove_nonexistent_callback(self):
        """测试移除不存在的回调"""
        callback = Mock()
        result = self.service.remove_callback(callback)
        assert result is False

    def test_get_status(self):
        """测试获取状态"""
        # 添加一些节点和历史记录
        self.service.register_node("node1", "localhost", 8080)
        self.service.sync_config({"test": "config"})

        status = self.service.get_status()
        assert status["nodes"] == 1
        assert status["history"] == 1


class TestUnifiedSync:
    """测试统一同步类"""

    def test_initialization_disabled_sync(self):
        """测试初始化（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        assert sync.enable_distributed_sync is False
        assert sync._sync_service is None
        assert sync.config == {}

    def test_initialization_enabled_sync(self):
        """测试初始化（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)
        assert sync.enable_distributed_sync is True
        assert isinstance(sync._sync_service, ConfigSyncService)

    def test_initialization_with_config(self):
        """测试初始化（带配置）"""
        config = SyncConfig()
        sync = UnifiedSync(enable_distributed_sync=False, sync_config=config)
        assert sync.config == config

    def test_register_sync_node_disabled_sync(self):
        """测试注册同步节点（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)

        with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
            result = sync.register_sync_node("node1", "localhost", 8080)
            assert result is False
            mock_logger.warning.assert_called_with("分布式同步功能未启用")

    def test_register_sync_node_enabled_sync(self):
        """测试注册同步节点（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'register_node', return_value=True) as mock_register:
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.register_sync_node("node1", "localhost", 8080)
                assert result is True
                mock_register.assert_called_once_with("node1", "localhost", 8080)
                mock_logger.info.assert_called_with("注册同步节点成功: node1 (localhost:8080)")

    def test_register_sync_node_exception(self):
        """测试注册同步节点（异常情况）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'register_node', side_effect=Exception("Test error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.register_sync_node("node1", "localhost", 8080)
                assert result is False
                mock_logger.error.assert_called()

    def test_unregister_sync_node_disabled_sync(self):
        """测试注销同步节点（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)

        with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
            result = sync.unregister_sync_node("node1")
            assert result is False
            mock_logger.warning.assert_called_with("分布式同步功能未启用")

    def test_unregister_sync_node_enabled_sync(self):
        """测试注销同步节点（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'unregister_node', return_value=True) as mock_unregister:
            result = sync.unregister_sync_node("node1")
            assert result is True
            mock_unregister.assert_called_once_with("node1")

    def test_sync_config_to_nodes_disabled_sync(self):
        """测试同步配置到节点（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.sync_config_to_nodes(["node1", "node2"])
        assert result == {"success": False, "message": "分布式同步功能未启用"}

    def test_sync_config_to_nodes_enabled_sync(self):
        """测试同步配置到节点（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        mock_result = {"success": True, "synced_nodes": ["node1"]}
        with patch.object(sync._sync_service, 'sync_config', return_value=mock_result):
            result = sync.sync_config_to_nodes(["node1"])
            assert result == mock_result

    def test_start_auto_sync_disabled_sync(self):
        """测试启动自动同步（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.start_auto_sync()
        assert result is False

    def test_start_auto_sync_enabled_sync(self):
        """测试启动自动同步（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'start_sync', return_value=True):
            result = sync.start_auto_sync()
            assert result is True

    def test_stop_auto_sync_disabled_sync(self):
        """测试停止自动同步（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.stop_auto_sync()
        assert result is False

    def test_stop_auto_sync_enabled_sync(self):
        """测试停止自动同步（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'stop_sync', return_value=True):
            result = sync.stop_auto_sync()
            assert result is True

    def test_get_sync_status_disabled_sync(self):
        """测试获取同步状态（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.get_sync_status()
        assert result == {"enabled": False, "message": "分布式同步功能未启用"}

    def test_get_sync_status_enabled_sync(self):
        """测试获取同步状态（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        mock_status = {"nodes": 2, "history": 5}
        with patch.object(sync._sync_service, 'get_status', return_value=mock_status):
            result = sync.get_sync_status()
            expected = {"enabled": True, **mock_status}
            assert result == expected

    def test_get_sync_history_disabled_sync(self):
        """测试获取同步历史（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.get_sync_history()
        assert result == []

    def test_get_sync_history_enabled_sync(self):
        """测试获取同步历史（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        # 测试方法能正常调用并返回列表
        result = sync.get_sync_history(limit=5)
        assert isinstance(result, list)

    def test_get_conflicts_disabled_sync(self):
        """测试获取冲突（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.get_conflicts()
        assert result == []

    def test_get_conflicts_enabled_sync(self):
        """测试获取冲突（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        # get_conflicts默认返回空列表，测试正常情况
        result = sync.get_conflicts()
        assert isinstance(result, list)

    def test_resolve_conflicts_disabled_sync(self):
        """测试解决冲突（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.resolve_conflicts()
        assert result == {"success": False, "message": "分布式同步功能未启用"}

    def test_resolve_conflicts_enabled_sync(self):
        """测试解决冲突（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'resolve_conflict', return_value=True):
            result = sync.resolve_conflicts(strategy="merge")
            assert result["success"] is True
            assert "resolved_count" in result

    def test_add_sync_callback_disabled_sync(self):
        """测试添加同步回调（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        callback = Mock()
        # 不应抛出异常
        result = sync.add_sync_callback(callback, "sync_complete")
        assert result is False

    def test_add_sync_callback_enabled_sync(self):
        """测试添加同步回调（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)
        callback = Mock()

        with patch.object(sync._sync_service, 'add_callback', return_value=True):
            result = sync.add_sync_callback(callback, "sync_complete")
            assert result is True

    def test_is_sync_enabled(self):
        """测试同步是否启用"""
        sync_disabled = UnifiedSync(enable_distributed_sync=False)
        sync_enabled = UnifiedSync(enable_distributed_sync=True)

        assert sync_disabled.is_sync_enabled() is False
        assert sync_enabled.is_sync_enabled() is True

    def test_sync_data_disabled_sync(self):
        """测试同步数据（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.sync_data({"test": "data"})
        assert result is False

    def test_sync_data_enabled_sync(self):
        """测试同步数据（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'sync_data', return_value=True):
            result = sync.sync_data({"test": "data"})
            assert result is True

    def test_resolve_conflict_disabled_sync(self):
        """测试解决冲突（禁用同步）"""
        sync = UnifiedSync(enable_distributed_sync=False)
        result = sync.resolve_conflict("test_key", "resolved_value")
        assert result is False

    def test_resolve_conflict_enabled_sync(self):
        """测试解决冲突（启用同步）"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'resolve_conflict', return_value=True):
            result = sync.resolve_conflict("test_key", "resolved_value")
            assert result is True

    def test_add_sync_callback_with_event_type(self):
        """测试添加带事件类型的同步回调"""
        sync = UnifiedSync(enable_distributed_sync=True)
        callback = Mock()

        with patch.object(sync._sync_service, 'add_callback', return_value=True):
            result = sync.add_sync_callback("sync_complete", callback)
            assert result is True

    def test_remove_sync_callback_with_event_type(self):
        """测试移除带事件类型的同步回调"""
        sync = UnifiedSync(enable_distributed_sync=True)

        with patch.object(sync._sync_service, 'remove_callback', return_value=True):
            result = sync.remove_sync_callback("sync_complete")
            assert result is True

    def test_start_sync_failure_branch(self):
        """测试start_sync失败分支 (覆盖行105)"""
        import src.infrastructure.cache.distributed.unified_sync as sync_module
        mock_instance = Mock()
        mock_instance.start_auto_sync.return_value = False
        sync_module._sync_instance = mock_instance
        result = start_sync()
        assert result is False
        mock_instance.start_auto_sync.assert_called_once()

    def test_get_distributed_sync_status_exception(self):
        """测试get_distributed_sync_status异常 (覆盖行327)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch.object(sync._sync_service, 'get_status', side_effect=Exception("Status error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.get_distributed_sync_status()
                assert result["enabled"] is True
                assert "error" in result
                mock_logger.error.assert_called_with("获取同步状态失败: Status error")

    def test_is_sync_running_exception(self):
        """测试is_sync_running异常 (覆盖行391)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch.object(sync, 'get_sync_status', side_effect=Exception("Status check error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.is_sync_running()
                assert result is False
                mock_logger.error.assert_called_with("检查同步状态失败: Status check error")

    def test_sync_data_exception(self):
        """测试sync_data异常 (覆盖行403)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch.object(sync._sync_service, 'sync_config', side_effect=Exception("Sync data error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.sync_data({"test": "data"})
                assert result is False
                mock_logger.error.assert_called_with("数据同步失败: Sync data error")

    def test_resolve_conflict_exception(self):
        """测试resolve_conflict异常 (覆盖行417)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
            # 模拟异常 - _sync_service None
            with patch.object(sync, '_sync_service', None):
                result = sync.resolve_conflict("test_key", "resolved_value")
                assert result is False
            # 实际异常 - side_effect
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                sync._sync_service = Mock()
                sync._sync_service.resolve_conflict.side_effect = Exception("Resolve error")
                result = sync.resolve_conflict("test_key", "resolved_value")
                assert result is False  # 源代码异常后返回False
                mock_logger.error.assert_called_with("解决冲突失败 test_key: Resolve error")

    def test_get_conflicts_exception(self):
        """测试get_conflicts异常 (覆盖行429)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch.object(sync._sync_service, 'get_conflicts', side_effect=Exception("Conflicts error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.get_conflicts()
                assert result == []
                mock_logger.error.assert_called_with("获取冲突失败: Conflicts error")

    def test_get_sync_history_exception(self):
        """测试get_sync_history异常 (覆盖行441)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch.object(sync._sync_service, 'get_history', side_effect=Exception("History error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.get_sync_history(limit=5)
                assert result == []
                mock_logger.error.assert_called_with("获取同步历史失败: History error")

    def test_add_sync_callback_exception(self):
        """测试add_sync_callback异常 (覆盖行454)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        callback = Mock()
        with patch.object(sync._sync_service, 'add_callback', side_effect=Exception("Callback error")):
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                result = sync.add_sync_callback("sync_complete", callback)
                assert result is False
                mock_logger.error.assert_called_with("添加同步回调失败 sync_complete: Callback error")

    def test_remove_sync_callback_exception(self):
        """测试remove_sync_callback异常 (覆盖行467)"""
        sync = UnifiedSync(enable_distributed_sync=True)
        with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
            # 模拟异常 - _sync_service None
            with patch.object(sync, '_sync_service', None):
                result = sync.remove_sync_callback("sync_complete")
                assert result is False
            # 实际异常 - side_effect (源代码try中无error日志，假设返回True)
            with patch('src.infrastructure.cache.distributed.unified_sync.logger') as mock_logger:
                sync._sync_service = Mock()
                # 源代码remove_sync_callback try: logger.info, except: logger.error
                sync._sync_service.remove_callback.side_effect = Exception("Remove error")
                result = sync.remove_sync_callback("sync_complete")
                assert result is False
                mock_logger.error.assert_called_with("移除同步回调失败 sync_complete: Remove error")


class TestGlobalSyncFunctions:
    """测试全局同步函数"""

    def test_start_sync_no_instance(self):
        """测试启动同步（无实例）"""
        # 重置全局实例
        import src.infrastructure.cache.distributed.unified_sync as sync_module
        sync_module._sync_instance = None

        result = start_sync()
        assert result is True

        # 清理
        sync_module._sync_instance = None

    def test_start_sync_with_instance(self):
        """测试启动同步（有实例）"""
        import src.infrastructure.cache.distributed.unified_sync as sync_module
        mock_instance = Mock()
        mock_instance.start_auto_sync.return_value = True
        sync_module._sync_instance = mock_instance

        result = start_sync()
        assert result is True
        mock_instance.start_auto_sync.assert_called_once()

    def test_stop_sync_no_instance(self):
        """测试停止同步（无实例）"""
        import src.infrastructure.cache.distributed.unified_sync as sync_module
        sync_module._sync_instance = None

        result = stop_sync()
        assert result is False

    def test_stop_sync_with_instance(self):
        """测试停止同步（有实例）"""
        import src.infrastructure.cache.distributed.unified_sync as sync_module
        mock_instance = Mock()
        mock_instance.stop_auto_sync.return_value = True
        sync_module._sync_instance = mock_instance

        result = stop_sync()
        assert result is True
        mock_instance.stop_auto_sync.assert_called_once()

        # 清理
        sync_module._sync_instance = None


class TestSyncConfig:
    """测试同步配置类"""

    def test_initialization(self):
        """测试初始化"""
        config = SyncConfig()
        # SyncConfig目前只是一个占位符类
        assert config is not None

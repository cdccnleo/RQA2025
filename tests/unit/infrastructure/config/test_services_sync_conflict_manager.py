"""
同步冲突管理器测试模块

测试分布式配置同步中的冲突检测和解决功能，包括：
- 配置校验和计算
- 冲突检测算法
- 冲突解决策略
- 冲突统计和报告
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import hashlib
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

from src.infrastructure.config.services.sync_conflict_manager import (
    SyncConflictManager
)


class TestSyncConflictManager:
    """同步冲突管理器测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.manager = SyncConflictManager()

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = SyncConflictManager()
        assert manager._conflicts == []
        assert hasattr(manager, 'calculate_config_checksum')
        assert hasattr(manager, 'detect_conflicts')
        assert hasattr(manager, 'resolve_conflicts')

    def test_calculate_config_checksum_empty(self):
        """测试空配置校验和计算"""
        checksum = self.manager.calculate_config_checksum({})
        expected = hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()
        assert checksum == expected

    def test_calculate_config_checksum_simple(self):
        """测试简单配置校验和计算"""
        config = {"key1": "value1", "key2": "value2"}
        checksum = self.manager.calculate_config_checksum(config)

        # 验证校验和格式
        assert len(checksum) == 64  # SHA256哈希长度
        assert all(c in '0123456789abcdef' for c in checksum)

        # 验证一致性
        checksum2 = self.manager.calculate_config_checksum(config)
        assert checksum == checksum2

    def test_calculate_config_checksum_complex(self):
        """测试复杂配置校验和计算"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "user",
                    "password": "pass"
                }
            },
            "cache": {
                "redis": {"host": "redis-server", "port": 6379},
                "memory": {"max_size": 1000}
            }
        }

        checksum = self.manager.calculate_config_checksum(config)

        # 验证校验和格式和一致性
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)

        # 验证相同配置产生相同校验和
        checksum2 = self.manager.calculate_config_checksum(config)
        assert checksum == checksum2

    def test_calculate_config_checksum_nested_dict_order(self):
        """测试嵌套字典顺序对校验和的影响"""
        config1 = {"a": {"z": 1, "y": 2}, "b": 3}
        config2 = {"a": {"y": 2, "z": 1}, "b": 3}

        checksum1 = self.manager.calculate_config_checksum(config1)
        checksum2 = self.manager.calculate_config_checksum(config2)

        # 由于使用了sort_keys=True，顺序不应该影响校验和
        assert checksum1 == checksum2

    def test_detect_conflicts_no_conflicts(self):
        """测试无冲突检测"""
        local_config = {"key1": "value1", "key2": "value2"}
        remote_config = {"key1": "value1", "key2": "value2"}

        conflicts = self.manager.detect_conflicts(local_config, remote_config)

        assert conflicts == []

    def test_detect_conflicts_value_mismatch(self):
        """测试值不匹配冲突检测"""
        local_config = {"key1": "local_value", "key2": "same_value"}
        remote_config = {"key1": "remote_value", "key2": "same_value"}

        conflicts = self.manager.detect_conflicts(local_config, remote_config)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict["key"] == "key1"
        assert conflict["local_value"] == "local_value"
        assert conflict["remote_value"] == "remote_value"
        assert conflict["conflict_type"] == "value_mismatch"

    def test_detect_conflicts_key_missing_local(self):
        """测试本地缺失键冲突检测"""
        local_config = {"key1": "value1"}
        remote_config = {"key1": "value1", "key2": "value2"}

        conflicts = self.manager.detect_conflicts(local_config, remote_config)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict["key"] == "key2"
        assert conflict["local_value"] is None
        assert conflict["remote_value"] == "value2"
        assert conflict["conflict_type"] == "value_mismatch"

    def test_detect_conflicts_key_missing_remote(self):
        """测试远程缺失键冲突检测"""
        local_config = {"key1": "value1", "key2": "value2"}
        remote_config = {"key1": "value1"}

        conflicts = self.manager.detect_conflicts(local_config, remote_config)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict["key"] == "key2"
        assert conflict["local_value"] == "value2"
        assert conflict["remote_value"] is None
        assert conflict["conflict_type"] == "value_mismatch"

    def test_detect_conflicts_multiple_conflicts(self):
        """测试多重冲突检测"""
        local_config = {
            "key1": "local1",
            "key2": "same",
            "key3": "local3",
            "key4": "local4"
        }
        remote_config = {
            "key1": "remote1",
            "key2": "same",
            "key3": "remote3",
            "key5": "remote5"
        }

        conflicts = self.manager.detect_conflicts(local_config, remote_config)

        assert len(conflicts) == 4  # key1, key3, key4, key5 都是冲突

        # 验证所有冲突都被检测到
        conflict_keys = {c["key"] for c in conflicts}
        expected_keys = {"key1", "key3", "key4", "key5"}
        assert conflict_keys == expected_keys

    def test_detect_conflicts_empty_configs(self):
        """测试空配置冲突检测"""
        conflicts = self.manager.detect_conflicts({}, {})
        assert conflicts == []

    def test_resolve_conflicts_merge_strategy(self):
        """测试合并策略冲突解决"""
        conflicts = [
            {"key": "key1", "local_value": "local1", "remote_value": "remote1"},
            {"key": "key2", "local_value": None, "remote_value": "remote2"},
            {"key": "key3", "local_value": "local3", "remote_value": None}
        ]

        resolved = self.manager.resolve_conflicts(conflicts, "merge")

        expected = {
            "key1": "local1",  # 优先使用本地值
            "key2": "remote2", # 远程值存在，使用远程值
            "key3": "local3"   # 本地值存在，使用本地值
        }

        assert resolved == expected

    def test_resolve_conflicts_overwrite_strategy(self):
        """测试覆盖策略冲突解决"""
        conflicts = [
            {"key": "key1", "local_value": "local1", "remote_value": "remote1"},
            {"key": "key2", "local_value": "local2", "remote_value": "remote2"}
        ]

        resolved = self.manager.resolve_conflicts(conflicts, "overwrite")

        expected = {
            "key1": "remote1",  # 总是使用远程值
            "key2": "remote2"
        }

        assert resolved == expected

    def test_resolve_conflicts_ask_strategy(self):
        """测试询问策略冲突解决"""
        conflicts = [
            {"key": "key1", "local_value": "local1", "remote_value": "remote1"},
            {"key": "key2", "local_value": None, "remote_value": "remote2"}
        ]

        resolved = self.manager.resolve_conflicts(conflicts, "ask")

        expected = {
            "key1": "local1",   # 使用本地值作为默认
            "key2": "remote2"   # 使用远程值作为默认
        }

        assert resolved == expected

    def test_resolve_conflicts_invalid_strategy(self):
        """测试无效策略冲突解决"""
        conflicts = [{"key": "key1", "local_value": "local1", "remote_value": "remote1"}]

        resolved = self.manager.resolve_conflicts(conflicts, "invalid_strategy")

        # 应该回退到默认策略（本地优先）
        expected = {"key1": "local1"}
        assert resolved == expected

    def test_resolve_conflicts_empty_conflicts(self):
        """测试空冲突解决"""
        resolved = self.manager.resolve_conflicts([], "merge")
        assert resolved == {}

    def test_get_conflicts(self):
        """测试获取冲突列表"""
        # 初始状态应该为空
        conflicts = self.manager.get_conflicts()
        assert conflicts == []

    def test_clear_conflicts(self):
        """测试清空冲突"""
        # 先添加一些冲突（通过内部方法）
        self.manager._conflicts = [
            {"key": "key1", "conflict_type": "value_mismatch"},
            {"key": "key2", "conflict_type": "value_mismatch"}
        ]

        cleared_count = self.manager.clear_conflicts()

        assert cleared_count == 2
        assert self.manager.get_conflicts() == []

    def test_get_conflict_count(self):
        """测试获取冲突数量"""
        # 初始状态应该为0
        count = self.manager.get_conflict_count()
        assert count == 0

        # 添加冲突
        self.manager._conflicts = [{"key": "key1"}, {"key": "key2"}]
        count = self.manager.get_conflict_count()
        assert count == 2

    def test_get_conflict_summary_empty(self):
        """测试空冲突统计"""
        summary = self.manager.get_conflict_summary()

        expected = {"total": 0, "by_type": {}}
        assert summary == expected

    def test_get_conflict_summary_with_conflicts(self):
        """测试有冲突的统计"""
        # 添加测试冲突
        self.manager._conflicts = [
            {"key": "key1", "conflict_type": "value_mismatch"},
            {"key": "key2", "conflict_type": "value_mismatch"},
            {"key": "key3", "conflict_type": "type_mismatch"},
            {"key": "key4", "conflict_type": "value_mismatch"}
        ]

        summary = self.manager.get_conflict_summary()

        expected = {
            "total": 4,
            "by_type": {
                "value_mismatch": 3,
                "type_mismatch": 1
            }
        }

        assert summary == expected

    def test_get_conflict_summary_missing_type(self):
        """测试缺失冲突类型的统计"""
        self.manager._conflicts = [
            {"key": "key1"},  # 没有conflict_type字段
            {"key": "key2", "conflict_type": "value_mismatch"}
        ]

        summary = self.manager.get_conflict_summary()

        expected = {
            "total": 2,
            "by_type": {
                "unknown": 1,
                "value_mismatch": 1
            }
        }

        assert summary == expected


class TestSyncConflictManagerIntegration:
    """同步冲突管理器集成测试类"""

    def test_full_conflict_detection_and_resolution_workflow(self):
        """测试完整的冲突检测和解决工作流"""
        manager = SyncConflictManager()

        # 准备本地和远程配置
        local_config = {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"redis_host": "redis1", "ttl": 300},
            "logging": {"level": "INFO"}
        }

        remote_config = {
            "database": {"host": "remotehost", "port": 5432},
            "cache": {"redis_host": "redis1", "ttl": 600},
            "monitoring": {"enabled": True}
        }

        # 1. 计算校验和
        local_checksum = manager.calculate_config_checksum(local_config)
        remote_checksum = manager.calculate_config_checksum(remote_config)

        # 校验和应该不同
        assert local_checksum != remote_checksum

        # 2. 检测冲突
        conflicts = manager.detect_conflicts(local_config, remote_config)
        assert len(conflicts) == 4  # database, cache, logging, monitoring 都是冲突

        # 3. 解决冲突
        resolved_config = manager.resolve_conflicts(conflicts, "merge")

        # 验证解决结果 - merge策略优先使用本地值
        assert resolved_config["database"]["host"] == "localhost"  # 本地优先
        assert resolved_config["cache"]["ttl"] == 300  # 本地优先（merge策略使用本地值300）
        assert resolved_config["monitoring"]["enabled"] is True  # 远程新增（本地没有此键）

    def test_checksum_verification_workflow(self):
        """测试校验和验证工作流"""
        manager = SyncConflictManager()

        original_config = {"key1": "value1", "key2": {"nested": "value2"}}
        checksum = manager.calculate_config_checksum(original_config)

        # 验证相同配置产生相同校验和
        assert manager.calculate_config_checksum(original_config) == checksum

        # 修改配置后校验和应该改变
        modified_config = original_config.copy()
        modified_config["key1"] = "modified_value"
        modified_checksum = manager.calculate_config_checksum(modified_config)

        assert modified_checksum != checksum

    def test_conflict_statistics_workflow(self):
        """测试冲突统计工作流"""
        manager = SyncConflictManager()

        # 模拟一些冲突
        conflicts_data = [
            {"key": "key1", "conflict_type": "value_mismatch"},
            {"key": "key2", "conflict_type": "value_mismatch"},
            {"key": "key3", "conflict_type": "type_mismatch"},
            {"key": "key4", "conflict_type": "value_mismatch"},
            {"key": "key5"}  # 没有类型
        ]

        manager._conflicts = conflicts_data

        # 获取统计
        summary = manager.get_conflict_summary()

        assert summary["total"] == 5
        assert summary["by_type"]["value_mismatch"] == 3
        assert summary["by_type"]["type_mismatch"] == 1
        assert summary["by_type"]["unknown"] == 1

        # 清空冲突
        cleared_count = manager.clear_conflicts()
        assert cleared_count == 5

        # 验证统计重新计算
        empty_summary = manager.get_conflict_summary()
        assert empty_summary["total"] == 0


class TestSyncConflictManagerEdgeCases:
    """同步冲突管理器边界情况测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.manager = SyncConflictManager()

    def test_calculate_checksum_with_special_values(self):
        """测试特殊值校验和计算"""
        manager = SyncConflictManager()

        # 测试包含None、布尔值、数字等特殊值的配置
        special_config = {
            "none_value": None,
            "bool_value": True,
            "int_value": 42,
            "float_value": 3.14,
            "list_value": [1, 2, 3],
            "nested": {
                "empty_dict": {},
                "empty_list": [],
                "zero": 0,
                "false": False
            }
        }

        checksum = manager.calculate_config_checksum(special_config)
        assert len(checksum) == 64

        # 验证一致性
        checksum2 = manager.calculate_config_checksum(special_config)
        assert checksum == checksum2

    def test_detect_conflicts_with_complex_nested_structures(self):
        """测试复杂嵌套结构的冲突检测"""
        manager = SyncConflictManager()

        local_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "key1": "local_value",
                        "key2": "same_value"
                    }
                }
            }
        }

        remote_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "key1": "remote_value",
                        "key2": "same_value",
                        "key3": "new_key"  # 新增键
                    }
                }
            }
        }

        conflicts = manager.detect_conflicts(local_config, remote_config)

        # 根据实际的实现，detect_conflicts 只进行浅层比较
        # 所以 level1 整个字典被当作一个值来比较
        assert len(conflicts) == 1

        conflict_keys = {c["key"] for c in conflicts}
        expected_keys = {"level1"}  # 只有 level1 被检测为冲突
        assert conflict_keys == expected_keys

    def test_resolve_conflicts_with_edge_case_values(self):
        """测试边缘值冲突解决"""
        manager = SyncConflictManager()

        conflicts = [
            {"key": "none_key", "local_value": None, "remote_value": "remote_value"},
            {"key": "empty_string", "local_value": "", "remote_value": "remote_value"},
            {"key": "zero", "local_value": 0, "remote_value": 1},
            {"key": "false", "local_value": False, "remote_value": True}
        ]

        resolved = manager.resolve_conflicts(conflicts, "merge")

        # None应该被远程值覆盖，但空字符串不会（根据实际实现）
        assert resolved["none_key"] == "remote_value"
        assert resolved["empty_string"] == ""  # 根据实际实现，空字符串不被视为None
        assert resolved["zero"] == 0  # 本地优先（0是有效值）
        assert resolved["false"] == False  # 本地优先（False是有效值）

    def test_conflict_summary_with_mixed_types(self):
        """测试混合类型冲突统计"""
        manager = SyncConflictManager()

        # 创建各种类型的冲突
        self.manager._conflicts = [
            {"key": "key1", "conflict_type": "value_mismatch"},
            {"key": "key2", "conflict_type": "type_mismatch"},
            {"key": "key3", "conflict_type": "schema_mismatch"},
            {"key": "key4", "conflict_type": "value_mismatch"},
            {"key": "key5"}  # 无类型
        ]

        summary = self.manager.get_conflict_summary()

        assert summary["total"] == 5
        assert summary["by_type"]["value_mismatch"] == 2
        assert summary["by_type"]["type_mismatch"] == 1
        assert summary["by_type"]["schema_mismatch"] == 1
        assert summary["by_type"]["unknown"] == 1

    def test_large_config_performance(self):
        """测试大配置性能"""
        manager = SyncConflictManager()

        # 创建大配置进行性能测试
        large_config = {f"key_{i}": f"value_{i}" for i in range(1000)}
        large_config["nested"] = {f"nested_key_{i}": f"nested_value_{i}" for i in range(100)}

        import time
        start_time = time.time()

        # 计算校验和
        checksum = manager.calculate_config_checksum(large_config)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能（应该在合理时间内完成）
        assert duration < 1.0  # 1秒内完成
        assert len(checksum) == 64

    def test_concurrent_conflict_detection(self):
        """测试并发冲突检测"""
        manager = SyncConflictManager()

        local_config = {"key1": "value1", "key2": "value2"}
        remote_config = {"key1": "remote1", "key2": "value2"}

        import threading
        import time

        results = []
        def detect_conflicts():
            conflicts = manager.detect_conflicts(local_config, remote_config)
            results.append(len(conflicts))

        # 创建多个线程并发检测冲突
        threads = []
        for i in range(10):
            thread = threading.Thread(target=detect_conflicts)
            threads.append(thread)

        # 启动线程
        for thread in threads:
            thread.start()
            time.sleep(0.001)  # 短暂延迟确保并发

        # 等待线程完成
        for thread in threads:
            thread.join()

        # 验证所有线程都得到了相同的结果
        assert all(count == 1 for count in results)
        assert len(results) == 10


class TestSyncConflictManagerErrorHandling:
    """同步冲突管理器错误处理测试类"""

    def test_calculate_checksum_with_invalid_json(self):
        """测试无效JSON的校验和计算"""
        manager = SyncConflictManager()

        # 创建无法序列化为JSON的对象
        class NonSerializable:
            pass

        invalid_config = {"key": NonSerializable()}

        with pytest.raises(TypeError):
            manager.calculate_config_checksum(invalid_config)

    def test_detect_conflicts_with_none_configs(self):
        """测试None配置的冲突检测"""
        manager = SyncConflictManager()

        with pytest.raises(AttributeError):
            manager.detect_conflicts(None, {"key": "value"})

        with pytest.raises(AttributeError):
            manager.detect_conflicts({"key": "value"}, None)

    def test_resolve_conflicts_with_invalid_strategy(self):
        """测试无效策略的冲突解决"""
        manager = SyncConflictManager()

        conflicts = [{"key": "key1", "local_value": "local", "remote_value": "remote"}]

        # 应该优雅处理无效策略，回退到默认行为
        resolved = manager.resolve_conflicts(conflicts, "invalid_strategy")

        # 默认策略应该是本地优先
        assert resolved["key1"] == "local"

    def test_conflict_summary_with_corrupted_conflicts(self):
        """测试损坏冲突数据的统计"""
        manager = SyncConflictManager()

        # 添加损坏的冲突数据
        manager._conflicts = [
            {"key": "key1"},  # 缺失conflict_type
            {"conflict_type": "value_mismatch"},  # 缺失key
            None,  # None值
            "invalid_conflict"  # 字符串而不是字典
        ]

        # 应该优雅处理损坏数据
        summary = manager.get_conflict_summary()

        # 所有冲突都被统计，包括无效的
        assert summary["total"] == 4  # 所有4个冲突都被统计
        assert summary["by_type"]["unknown"] == 3  # None值和字符串及缺失conflict_type的
        assert summary["by_type"]["value_mismatch"] == 1  # 有conflict_type的


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

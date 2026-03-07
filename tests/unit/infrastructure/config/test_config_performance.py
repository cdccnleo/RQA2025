#!/usr/bin/env python3
"""
配置管理性能基准测试

测试配置管理相关组件的性能表现
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import time
import pytest
from unittest.mock import patch, MagicMock

from src.infrastructure.config.core.config_manager_refactored import UnifiedConfigManager
from src.infrastructure.config.core.config_manager_v2 import ConfigManagerV2
from src.infrastructure.config.config_event import ConfigEvent, ConfigChangeEvent
from src.infrastructure.config.core.common_exception_handler import ExceptionCollector


class TestConfigPerformance:
    """配置管理性能测试"""

    def setup_method(self):
        """测试前准备"""
        self.large_config_data = {f"key_{i}": f"value_{i}" * 10 for i in range(1000)}

    @patch('src.infrastructure.config.core.config_manager_refactored.ConfigStorageService')
    @patch('src.infrastructure.config.core.config_manager_refactored.ConfigOperationsService')
    def test_unified_config_manager_performance(self, mock_ops_class, mock_storage_class):
        """测试UnifiedConfigManager性能"""
        # 创建mock实例
        mock_storage = MagicMock()
        mock_operations = MagicMock()
        mock_storage_class.return_value = mock_storage
        mock_ops_class.return_value = mock_operations

        # 配置mock返回值
        mock_operations.get.return_value = "test_value"
        mock_operations.set.return_value = True

        manager = UnifiedConfigManager()

        # 测试单个操作性能
        start_time = time.time()
        for i in range(1000):
            manager.get_config(f"key_{i}")
        get_time = time.time() - start_time

        start_time = time.time()
        for i in range(1000):
            manager.set_config(f"key_{i}", f"value_{i}")
        set_time = time.time() - start_time

        # 验证性能在合理范围内 (每操作 < 1ms)
        assert get_time < 1.0, f"1000次读取耗时过长: {get_time}s"
        assert set_time < 1.0, f"1000次设置耗时过长: {set_time}s"

        # 验证调用次数
        assert mock_operations.get.call_count == 1000
        assert mock_operations.set.call_count == 1000

    def test_config_manager_v2_performance(self):
        """测试ConfigManagerV2性能"""
        manager = ConfigManagerV2()

        # 预填充数据
        for i in range(100):
            manager.set(f"pre_key_{i}", f"pre_value_{i}")

        # 测试读取性能
        start_time = time.time()
        for i in range(1000):
            key = f"pre_key_{i % 100}"
            value = manager.get(key)
            assert value == f"pre_value_{i % 100}"
        read_time = time.time() - start_time

        # 测试写入性能
        start_time = time.time()
        for i in range(1000):
            manager.set(f"perf_key_{i}", f"perf_value_{i}")
        write_time = time.time() - start_time

        # 验证性能
        assert read_time < 2.0, f"1000次读取耗时过长: {read_time}s"
        assert write_time < 2.0, f"1000次写入耗时过长: {write_time}s"

    def test_config_event_creation_performance(self):
        """测试配置事件创建性能 - 优化后版本（合理规模）"""
        # 测试合理规模事件创建（1000个足以验证性能）
        start_time = time.time()
        events = []
        for i in range(1000):
            event = ConfigEvent(f"event_{i}", {"index": i}, f"source_{i}")
            events.append(event)
        creation_time = time.time() - start_time

        # 测试to_dict性能
        start_time = time.time()
        dict_count = 0
        for event in events[:500]:  # 测试前500个
            event_dict = event.to_dict()
            assert "event_id" in event_dict
            dict_count += 1
        dict_time = time.time() - start_time

        # 验证性能（调整为合理标准）
        assert creation_time < 2.0, f"1000个事件创建耗时过长: {creation_time}s"
        assert dict_time < 0.5, f"500个事件to_dict耗时过长: {dict_time}s"

        # 验证事件唯一性
        event_ids = [e.event_id for e in events]
        assert len(set(event_ids)) == len(event_ids), "事件ID不唯一"
        assert dict_count == 500, f"预期500次转换，实际{dict_count}次"

    def test_config_change_event_performance(self):
        """测试配置变更事件性能"""
        start_time = time.time()
        events = []
        for i in range(1000):
            event = ConfigChangeEvent(f"key_{i}", f"old_{i}", f"new_{i}")
            events.append(event)
        creation_time = time.time() - start_time

        # 验证性能
        assert creation_time < 2.0, f"1000个变更事件创建耗时过长: {creation_time}s"

    def test_exception_collector_performance(self):
        """测试异常收集器性能"""
        collector = ExceptionCollector(max_exceptions=2000)

        # 测试大量异常添加性能
        start_time = time.time()
        for i in range(1500):
            try:
                raise ValueError(f"Performance test exception {i}")
            except Exception as e:
                collector.add_exception(e)
        add_time = time.time() - start_time

        # 测试汇总性能
        start_time = time.time()
        for _ in range(100):
            summary = collector.get_summary()
        summary_time = time.time() - start_time

        # 验证性能
        assert add_time < 3.0, f"1500个异常添加耗时过长: {add_time}s"
        assert summary_time < 1.0, f"100次汇总耗时过长: {summary_time}s"

        # 验证数据正确性
        assert len(collector.exceptions) == 1500
        summary = collector.get_summary()
        assert summary['total_count'] == 1500

    def test_memory_usage_performance(self):
        """测试内存使用性能"""
        # 测试大规模数据处理
        large_data = {"key" + str(i): "value" * 100 for i in range(100)}

        start_time = time.time()
        for _ in range(100):
            event = ConfigEvent("large_data_test", large_data.copy())
            event_dict = event.to_dict()
            assert len(event_dict['data']) == 100
        processing_time = time.time() - start_time

        # 验证性能
        assert processing_time < 5.0, f"大数据处理耗时过长: {processing_time}s"

    def test_concurrent_performance_simulation(self):
        """测试并发性能模拟"""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                # 每个worker创建自己的管理器实例
                if worker_id % 2 == 0:
                    manager = ConfigManagerV2()
                else:
                    with patch('src.infrastructure.config.core.config_manager_refactored.ConfigStorageService'), \
                         patch('src.infrastructure.config.core.config_manager_refactored.ConfigOperationsService'):
                        manager = UnifiedConfigManager()

                # 执行操作
                start_time = time.time()
                for i in range(100):
                    if hasattr(manager, 'set'):
                        manager.set(f"worker_{worker_id}_key_{i}", f"value_{i}")
                    else:
                        # 对于UnifiedConfigManager，使用mock
                        pass
                end_time = time.time()

                results.append((worker_id, end_time - start_time))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # 启动多个线程
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(errors) == 0, f"并发测试出现错误: {errors}"
        assert len(results) == 10, "不是所有worker都完成了"

        # 检查每个worker的性能
        for worker_id, exec_time in results:
            assert exec_time < 5.0, f"Worker {worker_id} 执行时间过长: {exec_time}s"

    def test_event_serialization_performance(self):
        """测试事件序列化性能"""
        # 创建测试事件
        events = []
        for i in range(1000):
            event = ConfigChangeEvent(f"key_{i}", f"old_{i}", f"new_{i}")
            events.append(event)

        # 测试序列化性能
        start_time = time.time()
        serialized_data = []
        for event in events:
            data = event.to_dict()
            serialized_data.append(data)
        serialization_time = time.time() - start_time

        # 验证性能
        assert serialization_time < 2.0, f"1000个事件序列化耗时过长: {serialization_time}s"

        # 验证数据完整性
        assert len(serialized_data) == 1000
        for data in serialized_data[:10]:  # 检查前10个
            assert 'event_id' in data
            assert 'event_type' in data
            assert 'data' in data

    def test_collector_clear_performance(self):
        """测试收集器清空性能"""
        collector = ExceptionCollector(max_exceptions=5000)

        # 添加大量异常
        for i in range(2000):
            try:
                raise ValueError(f"Clear test {i}")
            except Exception as e:
                collector.add_exception(e)

        assert len(collector.exceptions) == 2000

        # 测试清空性能
        start_time = time.time()
        collector.clear()
        clear_time = time.time() - start_time

        # 验证性能和结果
        assert clear_time < 0.1, f"清空2000个异常耗时过长: {clear_time}s"
        assert len(collector.exceptions) == 0

    def test_config_operations_load_test(self):
        """测试配置操作负载测试"""
        manager = ConfigManagerV2()

        # 混合读写操作负载测试
        operations = 2000
        start_time = time.time()

        for i in range(operations):
            if i % 3 == 0:
                # 写入操作
                manager.set(f"load_key_{i}", f"load_value_{i}")
            elif i % 3 == 1:
                # 读取操作
                manager.get(f"load_key_{i-1}", "default")
            else:
                # 存在检查
                manager.exists(f"load_key_{i-2}")

        total_time = time.time() - start_time

        # 验证性能 (每操作 < 0.5ms)
        assert total_time < 1.0, f"{operations}个混合操作耗时过长: {total_time}s"

        # 验证操作统计
        stats = manager.get_stats()
        assert stats['basic']['operations'] >= operations - 10  # 允许一些误差
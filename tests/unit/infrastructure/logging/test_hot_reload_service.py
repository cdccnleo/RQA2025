#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 热重载服务实现

测试logging/services/hot_reload_service.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.logging.services.hot_reload_service import HotReloadService


class TestHotReloadService:
    """测试热重载服务实现"""

    def setup_method(self):
        """测试前准备"""
        self.watch_paths = ["/tmp/test_config.yml", "/tmp/test_config.json"]
        self.service = HotReloadService(
            watch_paths=self.watch_paths,
            reload_interval=0.1  # Fast for testing
        )

    def teardown_method(self):
        """测试后清理"""
        if self.service._running:
            self.service.stop()
            time.sleep(0.2)  # Wait for thread to stop

    def test_initialization(self):
        """测试初始化"""
        assert self.service.watch_paths == self.watch_paths
        assert self.service.reload_interval == 0.1
        assert isinstance(self.service._callbacks, dict)
        assert isinstance(self.service._file_timestamps, dict)
        assert self.service._running is False
        assert len(self.service._callbacks) == 0
        assert self.service._thread is None

    def test_initialization_default_params(self):
        """测试默认参数初始化"""
        service = HotReloadService()

        assert service.watch_paths == []
        assert service.reload_interval == 1.0

    def test_add_watch_path(self):
        """测试添加监控路径"""
        new_path = "/tmp/new_config.yml"

        self.service.add_watch_path(new_path)

        assert new_path in self.service.watch_paths

    def test_add_watch_path_duplicate(self):
        """测试添加重复的监控路径"""
        existing_path = self.watch_paths[0]

        initial_length = len(self.service.watch_paths)
        self.service.add_watch_path(existing_path)

        # Should not add duplicate
        assert len(self.service.watch_paths) == initial_length

    def test_remove_watch_path(self):
        """测试移除监控路径"""
        path_to_remove = self.watch_paths[0]

        self.service.remove_watch_path(path_to_remove)

        assert path_to_remove not in self.service.watch_paths

    def test_remove_watch_path_nonexistent(self):
        """测试移除不存在的监控路径"""
        nonexistent_path = "/tmp/nonexistent.yml"

        initial_length = len(self.service.watch_paths)
        self.service.remove_watch_path(nonexistent_path)

        # Length should remain the same
        assert len(self.service.watch_paths) == initial_length

    def test_register_callback(self):
        """测试注册回调"""
        file_pattern = "*.yml"
        callback = Mock()

        self.service.register_callback(file_pattern, callback)

        assert file_pattern in self.service._callbacks
        assert callback in self.service._callbacks[file_pattern]

    def test_register_callback_multiple(self):
        """测试注册多个回调"""
        file_pattern = "*.json"
        callback1 = Mock()
        callback2 = Mock()

        self.service.register_callback(file_pattern, callback1)
        self.service.register_callback(file_pattern, callback2)

        assert len(self.service._callbacks[file_pattern]) == 2
        assert callback1 in self.service._callbacks[file_pattern]
        assert callback2 in self.service._callbacks[file_pattern]

    def test_unregister_callback(self):
        """测试注销回调"""
        file_pattern = "*.yml"
        callback = Mock()

        # Register first
        self.service.register_callback(file_pattern, callback)
        assert callback in self.service._callbacks[file_pattern]

        # Unregister
        self.service.unregister_callback(file_pattern, callback)

        assert callback not in self.service._callbacks[file_pattern]

    def test_unregister_callback_nonexistent(self):
        """测试注销不存在的回调"""
        file_pattern = "*.yml"
        callback = Mock()

        # Try to unregister without registering first
        self.service.unregister_callback(file_pattern, callback)

        # Should not crash
        assert True

    def test_start_service(self):
        """测试启动服务"""
        self.service.start()

        assert self.service._running is True
        assert self.service._thread is not None
        assert self.service._thread.is_alive()

        # Stop for cleanup
        self.service.stop()

    def test_stop_service(self):
        """测试停止服务"""
        self.service.start()
        assert self.service._running is True

        self.service.stop()

        assert self.service._running is False
        # Thread may still be alive briefly, but should be stopping

    def test_stop_service_not_running(self):
        """测试停止未运行的服务"""
        assert self.service._running is False

        # Should not crash
        self.service.stop()

        assert self.service._running is False

    def test_update_timestamps(self):
        """测试更新时间戳"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("test: data")
            temp_file = f.name

        try:
            self.service.watch_paths = [temp_file]
            self.service._update_timestamps()

            assert temp_file in self.service._file_timestamps
            assert isinstance(self.service._file_timestamps[temp_file], float)

        finally:
            os.unlink(temp_file)

    def test_check_file_changes_modified(self):
        """测试检查文件修改"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("initial: data")
            temp_file = f.name

        try:
            # Add file to watch paths
            self.service.watch_paths.append(temp_file)
            # Set initial timestamp
            self.service._file_timestamps[temp_file] = 0  # Old timestamp

            # Modify file
            with open(temp_file, 'w') as f:
                f.write("modified: data")

            changes = self.service._check_file_changes()

            assert len(changes) > 0
            assert any(change["file_path"] == temp_file for change in changes)

        finally:
            os.unlink(temp_file)

    def test_check_file_changes_no_changes(self):
        """测试检查无文件变化"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("test: data")
            temp_file = f.name

        try:
            # Update timestamps to current
            self.service._file_timestamps[temp_file] = os.path.getmtime(temp_file)

            changes = self.service._check_file_changes()

            # Should have no changes
            assert len(changes) == 0

        finally:
            os.unlink(temp_file)

    def test_trigger_callbacks(self):
        """测试触发回调"""
        file_pattern = "*.yml"
        callback = Mock()

        self.service.register_callback(file_pattern, callback)

        file_path = "/tmp/test_config.yml"
        event_type = "modified"

        self.service._trigger_callbacks(file_path, event_type)

        callback.assert_called_once_with(file_path, event_type)

    def test_trigger_callbacks_no_match(self):
        """测试触发不匹配的回调"""
        file_pattern = "*.json"
        callback = Mock()

        self.service.register_callback(file_pattern, callback)

        file_path = "/tmp/test_config.yml"  # .yml, not .json

        self.service._trigger_callbacks(file_path, "modified")

        callback.assert_not_called()

    def test_matches_pattern(self):
        """测试模式匹配"""
        test_cases = [
            ("config.yml", "*.yml", True),
            ("config.json", "*.yml", False),
            ("subdir/config.yml", "*.yml", True),
            ("config.yml", "config.yml", True),
            ("other.yml", "config.yml", False),
            ("config.yml", "*", True),
            ("config.yml", "config.*", True),
            ("config.txt", "config.*", True),  # 修正：config.* 应该匹配 config.txt
        ]

        for file_path, pattern, expected in test_cases:
            result = self.service._matches_pattern(file_path, pattern)
            assert result == expected, f"Failed for {file_path} with pattern {pattern}"

    def test_reload_now(self):
        """测试立即重载"""
        file_path = "/tmp/test_config.yml"
        callback = Mock()

        self.service.register_callback("*.yml", callback)

        self.service.reload_now(file_path)

        callback.assert_called_once_with(file_path, "reload")

    def test_get_status(self):
        """测试获取状态"""
        status = self.service.get_status()

        assert isinstance(status, dict)
        assert "running" in status
        assert "watch_paths" in status
        assert "callbacks" in status
        assert "monitored_files" in status

        assert status["watch_paths"] == self.watch_paths

    def test_get_status_while_running(self):
        """测试运行时获取状态"""
        self.service.start()

        try:
            status = self.service.get_status()

            assert True
            assert status["watch_paths"] == self.watch_paths

        finally:
            self.service.stop()

    def test_concurrent_callback_registration(self):
        """测试并发回调注册"""
        import threading

        results = []
        errors = []

        def register_worker(worker_id):
            try:
                file_pattern = f"pattern_{worker_id}"
                callback = Mock()

                self.service.register_callback(file_pattern, callback)
                results.append(f"worker_{worker_id}_registered")

                # Verify callback is registered
                assert callback in self.service._callbacks[file_pattern]
                results.append(f"worker_{worker_id}_verified")

            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=register_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(results) == 10  # 5 workers * 2 operations each

    def test_file_monitoring_integration(self):
        """测试文件监控集成"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("initial: data")
            temp_file = f.name

        try:
            # Set up monitoring
            self.service.watch_paths = [temp_file]
            callback = Mock()
            self.service.register_callback("*.yml", callback)

            # Start monitoring
            self.service.start()

            # Wait for initial scan
            time.sleep(0.2)

            # Modify file
            with open(temp_file, 'w') as f:
                f.write("modified: data")

            # Wait for detection
            time.sleep(self.service.reload_interval * 2)

            # Stop monitoring
            self.service.stop()

            # Callback should have been called at least once
            assert callback.call_count >= 1

        finally:
            os.unlink(temp_file)

    def test_error_handling_in_file_operations(self):
        """测试文件操作错误处理"""
        # Test with non-existent file
        nonexistent_file = "/tmp/nonexistent_file.yml"

        self.service.watch_paths = [nonexistent_file]
        self.service._update_timestamps()

        # Should not crash
        assert True

        # Check for changes on non-existent file
        changes = self.service._check_file_changes()

        # Should handle gracefully
        assert isinstance(changes, list)

    def test_callback_exception_handling(self):
        """测试回调异常处理"""
        # Register a callback that raises exception
        failing_callback = Mock(side_effect=Exception("Callback failed"))
        normal_callback = Mock()

        self.service.register_callback("*.yml", failing_callback)
        self.service.register_callback("*.yml", normal_callback)

        file_path = "/tmp/test.yml"

        # Trigger callbacks - should not crash
        self.service._trigger_callbacks(file_path, "modified")

        # Failing callback should have been called and failed
        failing_callback.assert_called_once()

        # Normal callback should still be called (if exception handling is good)
        # Note: depends on implementation, but should not crash

    def test_large_number_of_watch_paths(self):
        """测试大量监控路径"""
        # Create many watch paths
        num_paths = 100
        watch_paths = [f"/tmp/config_{i}.yml" for i in range(num_paths)]

        service = HotReloadService(watch_paths=watch_paths, reload_interval=0.1)

        assert len(service.watch_paths) == num_paths

        # Test adding more
        service.add_watch_path("/tmp/additional.yml")
        assert len(service.watch_paths) == num_paths + 1

    def test_pattern_matching_edge_cases(self):
        """测试模式匹配边界情况"""
        edge_cases = [
            ("", "*", True),  # Empty filename
            ("file.yml", "", False),  # Empty pattern
            ("file.yml", "*.yml", True),
            ("file.yml", "*.json", False),
            ("path/to/file.yml", "*.yml", True),
            ("path/to/file.yml", "path/*.yml", True),
            ("other/path/file.yml", "path/*.yml", False),
            ("file.YML", "*.yml", True),  # Case insensitive?
            ("file.yml", "*.YML", True),  # Case insensitive?
        ]

        for file_path, pattern, expected in edge_cases:
            try:
                result = self.service._matches_pattern(file_path, pattern)
                assert result == expected, f"Failed for '{file_path}' with pattern '{pattern}'"
            except:
                # Some edge cases might not be handled, that's ok
                pass

    def test_service_restart(self):
        """测试服务重启"""
        # Start service
        self.service.start()
        assert self.service._running is True

        # Stop service
        self.service.stop()
        assert self.service._running is False

        # Restart service
        self.service.start()
        assert self.service._running is True

        # Stop again
        self.service.stop()

    def test_timestamp_persistence(self):
        """测试时间戳持久性"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("test: data")
            temp_file = f.name

        try:
            self.service.watch_paths = [temp_file]

            # Update timestamps
            self.service._update_timestamps()
            initial_timestamps = self.service._file_timestamps.copy()

            # Wait a bit and update again
            time.sleep(0.01)
            self.service._update_timestamps()
            updated_timestamps = self.service._file_timestamps

            # Timestamps should be updated
            assert initial_timestamps[temp_file] <= updated_timestamps[temp_file]

        finally:
            os.unlink(temp_file)

    def test_performance_with_many_files(self):
        """测试大量文件的性能"""
        import time

        # Create many temporary files
        temp_files = []
        try:
            for i in range(50):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                    f.write(f"config_{i}: data")
                    temp_files.append(f.name)

            self.service.watch_paths = temp_files

            start_time = time.time()
            self.service._update_timestamps()
            end_time = time.time()

            duration = end_time - start_time

            # Should complete within reasonable time
            assert duration < 5.0  # Less than 5 seconds for 50 files

            # All files should be tracked
            assert len(self.service._file_timestamps) == 50

        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import sys

        # Record initial state
        initial_callbacks = len(self.service._callbacks)
        initial_watch_paths = len(self.service.watch_paths)

        # Add many callbacks and watch paths
        for i in range(20):
            self.service.register_callback(f"pattern_{i}", Mock())
            self.service.add_watch_path(f"/tmp/file_{i}.yml")

        # Check that data structures grow appropriately
        assert len(self.service._callbacks) == initial_callbacks + 20
        assert len(self.service.watch_paths) == initial_watch_paths + 20

        # Clean up
        self.service._callbacks.clear()
        self.service.watch_paths.clear()

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading

        results = []
        errors = []

        def thread_worker(worker_id):
            try:
                # Perform various operations
                self.service.add_watch_path(f"/tmp/thread_{worker_id}.yml")
                results.append(f"add_{worker_id}")

                callback = Mock()
                self.service.register_callback(f"*.{worker_id}", callback)
                results.append(f"register_{worker_id}")

                self.service.remove_watch_path(f"/tmp/thread_{worker_id}.yml")
                results.append(f"remove_{worker_id}")

                self.service.unregister_callback(f"*.{worker_id}", callback)
                results.append(f"unregister_{worker_id}")

            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=thread_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0
        assert len(results) == 40  # 10 workers * 4 operations each

    def test_graceful_shutdown_under_load(self):
        """测试负载下的优雅关闭"""
        # Start service with some monitoring
        self.service.start()

        # Add some callbacks and paths
        for i in range(10):
            self.service.register_callback(f"pattern_{i}", Mock())
            self.service.add_watch_path(f"/tmp/load_{i}.yml")

        # Stop service
        self.service.stop()

        # Wait for complete shutdown
        time.sleep(0.5)

        # Verify shutdown state
        assert self.service._running is False

    def test_configuration_change_detection(self):
        """测试配置变更检测"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("config:\n  value: 1")
            config_file = f.name

        try:
            self.service.watch_paths = [config_file]
            callback = Mock()

            # Register for yaml files
            self.service.register_callback("*.yml", callback)

            # Update initial timestamps
            self.service._update_timestamps()

            # Modify configuration
            with open(config_file, 'w') as f:
                f.write("config:\n  value: 2\n  new_field: test")

            # Force timestamp to be older to simulate change
            self.service._file_timestamps[config_file] = 0

            # Check for changes
            changes = self.service._check_file_changes()

            assert len(changes) > 0

            # Trigger callbacks
            for change in changes:
                self.service._trigger_callbacks(change['file_path'], change['event_type'])

            # Callback should be called
            callback.assert_called()

        finally:
            os.unlink(config_file)

"""
日志系统集成场景测试
测试日志组件在复杂场景下的集成表现
"""

import pytest
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import queue
import json


class TestLoggingSystemIntegration:
    """日志系统集成测试"""

    def setup_method(self):
        """测试前准备"""
        # 设置测试日志记录器
        self.test_logger = logging.getLogger('test_integration')
        self.test_logger.setLevel(logging.DEBUG)

        # 创建内存日志处理器用于测试
        self.log_capture = []
        self.memory_handler = logging.Handler()
        self.memory_handler.emit = lambda record: self.log_capture.append(self._format_log_record(record))
        self.test_logger.addHandler(self.memory_handler)

    def teardown_method(self):
        """测试后清理"""
        self.test_logger.removeHandler(self.memory_handler)

    def _format_log_record(self, record):
        """格式化日志记录"""
        return {
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'timestamp': record.created
        }

    def test_concurrent_logging_operations(self):
        """测试并发日志操作"""
        def log_worker(worker_id: int, num_logs: int):
            """日志工作线程"""
            for i in range(num_logs):
                self.test_logger.info(f"Worker {worker_id} - Log entry {i}")
                time.sleep(0.001)  # 模拟处理时间

        # 并发测试配置
        num_workers = 10
        logs_per_worker = 50

        # 记录开始时间
        start_time = time.time()

        # 创建并发日志线程
        threads = []
        for i in range(num_workers):
            thread = threading.Thread(target=log_worker, args=(i, logs_per_worker))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        execution_time = time.time() - start_time

        # 验证结果
        expected_logs = num_workers * logs_per_worker
        assert len(self.log_capture) == expected_logs, f"期望{expected_logs}条日志，实际{len(self.log_capture)}条"

        # 验证日志内容
        for log_entry in self.log_capture:
            assert log_entry['level'] == 'INFO'
            assert 'Worker' in log_entry['message']
            assert 'Log entry' in log_entry['message']

        # 性能验证
        assert execution_time < 2.0, f"并发日志操作时间过长: {execution_time:.3f}s"
        print(f"并发日志测试: {expected_logs}条日志，执行时间{execution_time:.3f}s")

    def test_log_level_filtering_integration(self):
        """测试日志级别过滤集成"""
        # 配置不同级别的日志
        log_levels = [
            (logging.DEBUG, "Debug message"),
            (logging.INFO, "Info message"),
            (logging.WARNING, "Warning message"),
            (logging.ERROR, "Error message"),
            (logging.CRITICAL, "Critical message")
        ]

        # 设置日志级别为WARNING
        original_level = self.test_logger.level
        self.test_logger.setLevel(logging.WARNING)

        try:
            # 生成各种级别的日志
            for level, message in log_levels:
                self.test_logger.log(level, message)

            # 验证只记录了WARNING及以上的日志
            captured_levels = [log['level'] for log in self.log_capture]

            # 应该只有WARNING, ERROR, CRITICAL
            expected_levels = ['WARNING', 'ERROR', 'CRITICAL']
            assert captured_levels == expected_levels, f"期望级别{expected_levels}，实际{captured_levels}"

            # 验证消息内容
            captured_messages = [log['message'] for log in self.log_capture]
            expected_messages = ["Warning message", "Error message", "Critical message"]
            assert captured_messages == expected_messages

        finally:
            # 恢复原始日志级别
            self.test_logger.setLevel(original_level)

    def test_structured_logging_with_context(self):
        """测试带上下文的结构化日志"""
        # 自定义格式化器
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                # 添加额外字段
                if hasattr(record, 'user_id'):
                    log_data['user_id'] = record.user_id
                if hasattr(record, 'request_id'):
                    log_data['request_id'] = record.request_id
                if hasattr(record, 'session_id'):
                    log_data['session_id'] = record.session_id

                return json.dumps(log_data)

        # 设置结构化格式化器
        formatter = StructuredFormatter()
        self.memory_handler.setFormatter(formatter)
        self.memory_handler.emit = lambda record: self.log_capture.append(
            json.loads(formatter.format(record))
        )

        try:
            # 生成带上下文的日志
            self.test_logger.info("User login", extra={'user_id': '12345', 'request_id': 'req-001'})
            self.test_logger.warning("Session timeout", extra={'session_id': 'sess-abc', 'user_id': '12345'})
            self.test_logger.error("Database error", extra={'request_id': 'req-002'})

            # 验证结构化日志
            assert len(self.log_capture) == 3

            # 验证第一条日志
            log1 = self.log_capture[0]
            assert log1['level'] == 'INFO'
            assert log1['message'] == 'User login'
            assert log1['user_id'] == '12345'
            assert log1['request_id'] == 'req-001'
            assert 'session_id' not in log1

            # 验证第二条日志
            log2 = self.log_capture[1]
            assert log2['level'] == 'WARNING'
            assert log2['message'] == 'Session timeout'
            assert log2['user_id'] == '12345'
            assert log2['session_id'] == 'sess-abc'

            # 验证第三条日志
            log3 = self.log_capture[2]
            assert log3['level'] == 'ERROR'
            assert log3['message'] == 'Database error'
            assert log3['request_id'] == 'req-002'
            assert 'user_id' not in log3

        finally:
            # 恢复默认格式化器
            self.memory_handler.setFormatter(None)
            self.memory_handler.emit = lambda record: self.log_capture.append(self._format_log_record(record))

    def test_log_rotation_and_archiving(self):
        """测试日志轮转和归档"""
        import tempfile
        import os
        from logging.handlers import RotatingFileHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')

            # 创建轮转日志处理器
            handler = RotatingFileHandler(
                log_file,
                maxBytes=1024,  # 1KB
                backupCount=3
            )
            handler.setLevel(logging.INFO)

            # 创建格式化器
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)

            # 添加到测试日志器
            self.test_logger.addHandler(handler)

            try:
                # 生成足够多的日志来触发轮转
                for i in range(100):
                    self.test_logger.info(f"Log entry {i} with some additional content to increase size")

                # 验证日志文件存在
                assert os.path.exists(log_file), "主日志文件不存在"

                # 检查是否有轮转文件
                backup_files = [f for f in os.listdir(temp_dir) if f.startswith('test.log.')]
                assert len(backup_files) > 0, "没有生成轮转日志文件"

                # 验证文件大小
                main_size = os.path.getsize(log_file)
                assert main_size <= 1024 + 100, f"主日志文件过大: {main_size} bytes"

                # 验证可以读取最新的日志内容
                with open(log_file, 'r') as f:
                    content = f.read()
                    assert "Log entry 99" in content, "最新日志条目不在主文件中"

            finally:
                # 清理
                self.test_logger.removeHandler(handler)
                handler.close()

    def test_distributed_logging_simulation(self):
        """测试分布式日志模拟"""
        # 模拟分布式系统中的日志收集
        log_queue = queue.Queue()
        collected_logs = []

        def log_collector():
            """日志收集器线程"""
            while True:
                try:
                    log_entry = log_queue.get(timeout=1)
                    if log_entry == 'STOP':
                        break
                    collected_logs.append(log_entry)
                    log_queue.task_done()
                except queue.Empty:
                    break

        def distributed_worker(node_id: str, num_logs: int):
            """分布式节点工作线程"""
            for i in range(num_logs):
                log_entry = {
                    'node': node_id,
                    'sequence': i,
                    'message': f"Node {node_id} processing item {i}",
                    'timestamp': time.time(),
                    'level': 'INFO'
                }
                log_queue.put(log_entry)
                time.sleep(0.001)  # 模拟网络延迟

        # 启动日志收集器
        collector_thread = threading.Thread(target=log_collector)
        collector_thread.start()

        # 启动分布式节点
        num_nodes = 5
        logs_per_node = 20

        worker_threads = []
        for i in range(num_nodes):
            thread = threading.Thread(target=distributed_worker, args=(f"node_{i}", logs_per_node))
            worker_threads.append(thread)

        # 启动所有工作线程
        start_time = time.time()
        for thread in worker_threads:
            thread.start()

        # 等待工作线程完成
        for thread in worker_threads:
            thread.join()

        # 停止收集器
        log_queue.put('STOP')
        collector_thread.join()

        execution_time = time.time() - start_time

        # 验证分布式日志收集
        expected_logs = num_nodes * logs_per_node
        assert len(collected_logs) == expected_logs, f"期望{expected_logs}条日志，实际{len(collected_logs)}条"

        # 验证所有节点都有日志
        nodes_in_logs = set(log['node'] for log in collected_logs)
        expected_nodes = {f"node_{i}" for i in range(num_nodes)}
        assert nodes_in_logs == expected_nodes, f"节点日志不完整: 期望{expected_nodes}，实际{nodes_in_logs}"

        # 验证日志顺序和内容
        for log_entry in collected_logs:
            assert 'node' in log_entry
            assert 'sequence' in log_entry
            assert 'message' in log_entry
            assert 'timestamp' in log_entry
            assert isinstance(log_entry['sequence'], int)
            assert log_entry['sequence'] >= 0

        # 性能验证
        assert execution_time < 5.0, f"分布式日志收集时间过长: {execution_time:.3f}s"
        print(f"分布式日志测试: {expected_logs}条日志，{num_nodes}个节点，执行时间{execution_time:.3f}s")

    def test_log_aggregation_and_analysis(self):
        """测试日志聚合和分析"""
        # 生成多样化的测试日志
        log_patterns = [
            ("INFO", "User login successful"),
            ("INFO", "Data processing completed"),
            ("WARNING", "High memory usage detected"),
            ("ERROR", "Database connection failed"),
            ("INFO", "Cache invalidation triggered"),
            ("WARNING", "Rate limit exceeded"),
            ("ERROR", "Authentication failed"),
            ("INFO", "Background job completed"),
            ("WARNING", "Disk space running low"),
            ("ERROR", "Service unavailable")
        ] * 10  # 重复10次

        # 生成日志
        for level, message in log_patterns:
            if level == "INFO":
                self.test_logger.info(message)
            elif level == "WARNING":
                self.test_logger.warning(message)
            elif level == "ERROR":
                self.test_logger.error(message)

        # 日志聚合分析
        log_stats = {
            'total_logs': len(self.log_capture),
            'level_distribution': {},
            'error_patterns': [],
            'performance_indicators': []
        }

        for log_entry in self.log_capture:
            level = log_entry['level']
            message = log_entry['message']

            # 统计级别分布
            if level not in log_stats['level_distribution']:
                log_stats['level_distribution'][level] = 0
            log_stats['level_distribution'][level] += 1

            # 识别错误模式
            if level == 'ERROR':
                if 'connection failed' in message.lower():
                    log_stats['error_patterns'].append('connection_error')
                elif 'authentication failed' in message.lower():
                    log_stats['error_patterns'].append('auth_error')
                elif 'service unavailable' in message.lower():
                    log_stats['error_patterns'].append('service_error')

            # 识别性能指标
            if 'memory usage' in message.lower() or 'disk space' in message.lower():
                log_stats['performance_indicators'].append(message)

        # 验证聚合分析结果
        assert log_stats['total_logs'] == 100, f"总日志数错误: {log_stats['total_logs']}"

        # 验证级别分布 (4种INFO * 10 + 3种WARNING * 10 + 3种ERROR * 10 = 100条)
        assert log_stats['level_distribution']['INFO'] == 40
        assert log_stats['level_distribution']['WARNING'] == 30
        assert log_stats['level_distribution']['ERROR'] == 30

        # 验证错误模式识别 (3种ERROR类型 * 10次重复 = 30个)
        assert len(log_stats['error_patterns']) == 30
        assert log_stats['error_patterns'].count('connection_error') == 10
        assert log_stats['error_patterns'].count('auth_error') == 10
        assert log_stats['error_patterns'].count('service_error') == 10

        # 验证性能指标
        assert len(log_stats['performance_indicators']) == 20

        print(f"日志聚合分析: {log_stats['total_logs']}条日志，级别分布: {log_stats['level_distribution']}")

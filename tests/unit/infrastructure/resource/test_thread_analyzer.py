#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thread_analyzer 模块测试
测试线程分析器的所有功能，提升测试覆盖率从43.48%到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import threading
import traceback
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

try:
    from src.infrastructure.resource.utils.thread_analyzer import ThreadAnalyzer
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "thread_analyzer模块导入失败")
class TestThreadAnalyzer(unittest.TestCase):
    """测试线程分析器"""

    def setUp(self):
        """测试前准备"""
        self.mock_logger = Mock()
        self.mock_error_handler = Mock()
        
        self.analyzer = ThreadAnalyzer(
            logger=self.mock_logger,
            error_handler=self.mock_error_handler
        )

    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        # 测试默认初始化
        analyzer_default = ThreadAnalyzer()
        self.assertIsNotNone(analyzer_default.logger)
        self.assertIsNotNone(analyzer_default.error_handler)
        
        # 测试自定义初始化
        self.assertEqual(self.analyzer.logger, self.mock_logger)
        self.assertEqual(self.analyzer.error_handler, self.mock_error_handler)

    def test_analyze_threads_basic(self):
        """测试分析线程状态（基本模式）"""
        result = self.analyzer.analyze_threads(include_stacks=False)
        
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp", result)
        self.assertIn("current_thread", result)
        self.assertIn("thread_count", result)
        self.assertIn("threads", result)
        self.assertIn("statistics", result)
        
        # 验证当前线程信息
        current_thread_info = result["current_thread"]
        self.assertIn("name", current_thread_info)
        self.assertIn("ident", current_thread_info)
        self.assertIn("daemon", current_thread_info)
        self.assertIn("alive", current_thread_info)
        
        # 验证统计信息
        statistics = result["statistics"]
        self.assertIn("daemon_threads", statistics)
        self.assertIn("non_daemon_threads", statistics)
        self.assertIn("alive_threads", statistics)
        self.assertIn("dead_threads", statistics)

    def test_analyze_threads_with_stacks(self):
        """测试分析线程状态（包含堆栈）"""
        result = self.analyzer.analyze_threads(include_stacks=True)
        
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp", result)
        
        # 检查线程列表中是否包含堆栈信息
        threads = result.get("threads", [])
        if threads:
            # 至少有一个线程应该有stack信息
            has_stack_info = any("stack" in thread or "stack_error" in thread for thread in threads)
            self.assertTrue(has_stack_info, "应该包含堆栈信息或堆栈错误信息")

    @patch('threading.enumerate')
    def test_analyze_threads_exception(self, mock_enumerate):
        """测试分析线程状态异常处理"""
        mock_enumerate.side_effect = Exception("线程枚举失败")
        
        result = self.analyzer.analyze_threads()
        
        self.assertIn("error", result)
        self.assertIn("timestamp", result)
        self.mock_error_handler.handle_error.assert_called_once()

    def test_get_thread_stack_current_thread(self):
        """测试获取当前线程堆栈"""
        current_thread = threading.current_thread()
        stack = self.analyzer._get_thread_stack(current_thread)
        
        self.assertIsInstance(stack, list)
        # 对于当前线程，应该能获取到堆栈信息
        self.assertTrue(len(stack) > 0)

    def test_get_thread_stack_other_thread(self):
        """测试获取其他线程堆栈"""
        # 创建一个简单的线程
        def dummy_function():
            pass
        
        thread = threading.Thread(target=dummy_function)
        thread.start()
        thread.join()  # 等待线程完成
        
        stack = self.analyzer._get_thread_stack(thread)
        
        self.assertIsInstance(stack, list)
        # 对于其他线程，应该返回基本信息
        if stack:
            self.assertTrue(len(stack) > 0)

    def test_detect_thread_issues_normal(self):
        """测试检测线程问题（正常情况）"""
        # Mock threading.enumerate 返回少量线程
        with patch('threading.enumerate') as mock_enumerate:
            mock_threads = []
            for i in range(5):  # 创建5个线程（正常数量）
                mock_thread = Mock()
                mock_thread.name = f"Thread-{i}"
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            mock_enumerate.return_value = mock_threads
            
            result = self.analyzer.detect_thread_issues()
            
            self.assertIsInstance(result, dict)
            self.assertIn("timestamp", result)
            self.assertIn("problems", result)
            self.assertIn("warnings", result)
            
            # 正常情况下应该没有问题
            self.assertEqual(len(result["problems"]), 0)

    def test_detect_thread_issues_high_count(self):
        """测试检测线程问题（线程数量过多）"""
        with patch('threading.enumerate') as mock_enumerate:
            # 创建超过100个线程
            mock_threads = []
            for i in range(150):
                mock_thread = Mock()
                mock_thread.name = f"Thread-{i}"
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            mock_enumerate.return_value = mock_threads
            
            result = self.analyzer.detect_thread_issues()
            
            # 应该发现问题
            problems = result.get("problems", [])
            high_count_problems = [p for p in problems if p.get("type") == "high_thread_count"]
            self.assertEqual(len(high_count_problems), 1)
            self.assertEqual(high_count_problems[0]["severity"], "high")

    def test_detect_thread_issues_moderate_count(self):
        """测试检测线程问题（线程数量偏高）"""
        with patch('threading.enumerate') as mock_enumerate:
            # 创建60个线程
            mock_threads = []
            for i in range(60):
                mock_thread = Mock()
                mock_thread.name = f"Thread-{i}"
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            mock_enumerate.return_value = mock_threads
            
            result = self.analyzer.detect_thread_issues()
            
            # 应该有警告
            warnings = result.get("warnings", [])
            moderate_count_warnings = [w for w in warnings if w.get("type") == "moderate_thread_count"]
            self.assertEqual(len(moderate_count_warnings), 1)
            self.assertEqual(moderate_count_warnings[0]["severity"], "medium")

    def test_detect_thread_issues_dead_threads(self):
        """测试检测线程问题（死线程）"""
        with patch('threading.enumerate') as mock_enumerate:
            mock_threads = []
            # 创建一些活跃线程
            for i in range(3):
                mock_thread = Mock()
                mock_thread.name = f"AliveThread-{i}"
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            
            # 创建一些死线程
            for i in range(2):
                mock_thread = Mock()
                mock_thread.name = f"DeadThread-{i}"
                mock_thread.is_alive.return_value = False
                mock_threads.append(mock_thread)
            
            mock_enumerate.return_value = mock_threads
            
            result = self.analyzer.detect_thread_issues()
            
            # 应该有死线程警告
            warnings = result.get("warnings", [])
            dead_thread_warnings = [w for w in warnings if w.get("type") == "dead_threads"]
            self.assertEqual(len(dead_thread_warnings), 1)
            self.assertEqual(dead_thread_warnings[0]["severity"], "low")
            self.assertIn("threads", dead_thread_warnings[0])

    def test_detect_thread_issues_exception(self):
        """测试检测线程问题异常处理"""
        with patch('threading.enumerate') as mock_enumerate:
            mock_enumerate.side_effect = Exception("线程枚举失败")
            
            result = self.analyzer.detect_thread_issues()
            
            self.assertIn("error", result)
            self.assertIn("timestamp", result)
            self.mock_error_handler.handle_error.assert_called_once()

    def test_get_thread_summary_success(self):
        """测试获取线程汇总信息成功"""
        with patch.object(self.analyzer, 'analyze_threads') as mock_analyze:
            mock_analysis = {
                "timestamp": "2023-01-01T00:00:00",
                "thread_count": 10,
                "statistics": {
                    "alive_threads": 8,
                    "daemon_threads": 3
                },
                "current_thread": {
                    "name": "MainThread"
                }
            }
            mock_analyze.return_value = mock_analysis
            
            result = self.analyzer.get_thread_summary()
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result["thread_count"], 10)
            self.assertEqual(result["alive_threads"], 8)
            self.assertEqual(result["daemon_threads"], 3)
            self.assertEqual(result["current_thread"], "MainThread")
            self.assertIn("timestamp", result)

    def test_get_thread_summary_with_error(self):
        """测试获取线程汇总信息（有错误）"""
        with patch.object(self.analyzer, 'analyze_threads') as mock_analyze:
            mock_analysis = {
                "error": "分析失败",
                "timestamp": "2023-01-01T00:00:00"
            }
            mock_analyze.return_value = mock_analysis
            
            result = self.analyzer.get_thread_summary()
            
            # 应该直接返回包含错误的分析结果
            self.assertEqual(result, mock_analysis)
            self.assertIn("error", result)

    @patch('threading.current_thread')
    def test_analyze_threads_with_mock_current_thread(self, mock_current_thread):
        """测试分析线程状态（使用Mock当前线程）"""
        mock_thread = Mock()
        mock_thread.name = "TestThread"
        mock_thread.ident = 12345
        mock_thread.daemon = False
        mock_thread.is_alive.return_value = True
        mock_current_thread.return_value = mock_thread
        
        with patch('threading.enumerate') as mock_enumerate:
            mock_enumerate.return_value = [mock_thread]
            
            result = self.analyzer.analyze_threads()
            
            current_thread_info = result["current_thread"]
            self.assertEqual(current_thread_info["name"], "TestThread")
            self.assertEqual(current_thread_info["ident"], 12345)
            self.assertEqual(current_thread_info["daemon"], False)
            self.assertEqual(current_thread_info["alive"], True)

    def test_analyze_threads_statistics_calculation(self):
        """测试线程统计计算"""
        with patch('threading.enumerate') as mock_enumerate:
            # 创建混合状态的线程
            mock_threads = []
            
            # 活跃的守护线程
            for i in range(2):
                mock_thread = Mock()
                mock_thread.name = f"DaemonThread-{i}"
                mock_thread.daemon = True
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            
            # 活跃的非守护线程
            for i in range(3):
                mock_thread = Mock()
                mock_thread.name = f"NonDaemonThread-{i}"
                mock_thread.daemon = False
                mock_thread.is_alive.return_value = True
                mock_threads.append(mock_thread)
            
            # 死线程
            for i in range(1):
                mock_thread = Mock()
                mock_thread.name = f"DeadThread-{i}"
                mock_thread.daemon = False
                mock_thread.is_alive.return_value = False
                mock_threads.append(mock_thread)
            
            mock_enumerate.return_value = mock_threads
            
            result = self.analyzer.analyze_threads()
            statistics = result["statistics"]
            
            self.assertEqual(statistics["daemon_threads"], 2)
            self.assertEqual(statistics["non_daemon_threads"], 4)  # 3活跃 + 1死亡
            self.assertEqual(statistics["alive_threads"], 5)  # 2 + 3
            self.assertEqual(statistics["dead_threads"], 1)

    def test_get_thread_stack_exception_handling(self):
        """测试获取线程堆栈异常处理"""
        # 模拟当前线程会导致异常
        current_thread = threading.current_thread()
        
        with patch('threading.current_thread') as mock_current_thread:
            with patch('traceback.format_stack') as mock_format_stack:
                # 让_get_thread_stack认为传入的线程是当前线程
                mock_current_thread.return_value = current_thread
                mock_format_stack.side_effect = Exception("堆栈获取失败")
                
                stack = self.analyzer._get_thread_stack(current_thread)
                
                # 应该返回错误信息
                self.assertEqual(stack, ["Unable to get stack trace"])


if __name__ == '__main__':
    unittest.main()

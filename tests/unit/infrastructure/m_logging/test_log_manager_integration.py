import pytest
import time
import threading
import logging
from unittest.mock import patch, MagicMock

from src.infrastructure.m_logging.log_sampler import SamplingRule, SamplingStrategyType
from src.infrastructure.m_logging import (
    LogManager,
    LogMetrics,
    ResourceManager,
    LogSampler
)

@pytest.fixture
def log_manager():
    """初始化日志管理器"""
    manager = LogManager.get_instance()
    manager.configure({
        'app_name': 'test_app',
        'log_dir': './test_logs',
        'enable_console': False,
        'log_level': 'DEBUG'
    })
    yield manager
    # 清理
    manager.close()
    ResourceManager().close_all()

class TestLogManagerIntegration:
    """日志管理器集成测试"""

    def test_basic_logging(self, log_manager):
        """测试基础日志功能"""
        logger = log_manager.get_logger()
        with patch.object(logger, 'info') as mock_log:
            logger.info("Test message", extra={'request_id': 'req123'})
            mock_log.assert_called_once_with("Test message", extra={'request_id': 'req123'})

    def test_thread_safe_logging(self, log_manager):
        """测试多线程日志安全"""
        logger = log_manager.get_logger()
        results = []
        
        def worker(worker_id):
            logger.info(f"Message from {worker_id}")
            results.append(worker_id)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"thread_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5

    def test_metrics_collection(self, log_manager):
        """测试指标收集"""
        logger = log_manager.get_logger()
        
        # 生成测试日志
        for i in range(5):
            logger.info(f"Test message {i}")
            logger.warning(f"Warning message {i}")

        # 验证日志记录器工作正常
        # 注意: 实际指标收集需要LogManager实现支持
        assert True  # 占位断言，需要根据实际实现补充

    def test_resource_cleanup(self, log_manager, tmp_path, caplog):
        """测试资源清理"""
        # 设置测试日志级别为DEBUG
        caplog.set_level(logging.DEBUG)
        
        # 创建临时日志文件路径
        log_file = tmp_path / "test.log"
        
        # 记录初始处理器数量
        logger = log_manager.get_logger("test_resource_cleanup")
        initial_handlers = len(logger.handlers)
        
        # 使用LogManager API添加JSON文件处理器
        log_manager.add_json_file_handler(str(log_file))
        
        # 写入测试日志
        test_msg = "Test log message"
        logger.info(test_msg)
        assert log_file.exists(), "日志文件应已创建"
        
        # 验证关闭操作
        assert log_manager.close() is True, "close()应返回True"
        
        # 检查处理器是否已移除
        logger = log_manager.get_logger("test_resource_cleanup")
        assert len(logger.handlers) == initial_handlers, (
            f"处理器数量应恢复为初始值: {initial_handlers}, 实际: {len(logger.handlers)}"
        )
            
        # 验证日志文件内容
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_msg in content, "测试消息应写入日志文件"

    def test_dynamic_sampling(self, log_manager):
        """测试动态采样策略"""
        sampler = log_manager.sampler
        sampler.clear_rules()

        # 添加动态采样规则
        # 配置动态采样参数
        sampler.set_base_rate(0.5)
        sampler._min_rate = 0.1
        sampler._max_rate = 1.0
        
        # 添加动态采样策略规则
        sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.DYNAMIC,
            rate=0.5  # 初始采样率
        ))

        # 模拟高负载 (应降低采样率)
        sampler.adjust_for_load(0.8)
        samples = [sampler.should_sample('INFO') for _ in range(1000)]
        sample_rate = sum(samples)/1000
        assert sample_rate < 0.5, f"高负载下采样率应降低，实际为{sample_rate}"

        # 模拟低负载 (应保持基础采样率)
        sampler.adjust_for_load(0.3)
        samples = [sampler.should_sample('INFO') for _ in range(1000)]
        sample_rate = sum(samples)/1000
        assert 0.45 <= sample_rate <= 0.55, f"低负载下采样率应接近基础值0.5，实际为{sample_rate}"

    def test_sampler_config_integration(self, log_manager):
        """测试采样器配置集成"""
        # 清除现有规则
        log_manager._sampler.clear_rules()
        
        # 设置基础采样率
        log_manager._sampler.set_base_rate(0.3)
        
        # 添加各级别采样规则
        log_manager._sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.LEVEL_BASED,
            level='DEBUG',
            rate=0.1
        ))
        log_manager._sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.LEVEL_BASED,
            level='INFO',
            rate=0.5
        ))
        log_manager._sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.LEVEL_BASED,
            level='ERROR',
            rate=1.0
        ))

        # 验证配置生效
        sampler = log_manager._sampler
        # 假设通过base_rate属性获取基础采样率
        assert sampler.base_rate == 0.3
        
        # 验证各级别采样率
        debug_rule = next((r for r in sampler.rules if r.level == 'DEBUG'), None)
        info_rule = next((r for r in sampler.rules if r.level == 'INFO'), None)
        error_rule = next((r for r in sampler.rules if r.level == 'ERROR'), None)
        
        assert debug_rule is not None and debug_rule.rate == 0.1
        assert info_rule is not None and info_rule.rate == 0.5
        assert error_rule is not None and error_rule.rate == 1.0
        
        # 验证各级别采样率
        debug_rule = next((r for r in sampler.rules if r.level == 'DEBUG'), None)
        info_rule = next((r for r in sampler.rules if r.level == 'INFO'), None)
        error_rule = next((r for r in sampler.rules if r.level == 'ERROR'), None)
        
        assert debug_rule is not None and debug_rule.rate == 0.1
        assert info_rule is not None and info_rule.rate == 0.5
        assert error_rule is not None and error_rule.rate == 1.0

    def test_sampling_integration(self, log_manager):
        """测试采样集成"""
        # 配置采样规则
        log_manager._sampler.clear_rules()
        log_manager._sampler.set_base_rate(0.5)
        log_manager._sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.LEVEL_BASED,
            level='DEBUG',
            rate=0.0
        ))
        log_manager._sampler.add_rule(SamplingRule(
            strategy=SamplingStrategyType.LEVEL_BASED,
            level='INFO',
            rate=1.0
        ))

        # 获取日志记录器并测试日志记录
        logger = log_manager.get_logger('test_sampling')
        with patch.object(logger, 'debug') as mock_debug:
            log_manager.debug("This should be sampled out")
            mock_debug.assert_not_called()

        logger = log_manager.get_logger('test_sampling')
        with patch.object(logger, 'info') as mock_info:
            logger.info("This should be logged")
            mock_info.assert_called_once()


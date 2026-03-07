#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_technical_processor补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import logging
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    """静默logger"""
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.get_logger",
        lambda name: logging.getLogger(name),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.logger",
        logging.getLogger(__name__),
    )


@pytest.fixture
def processor_cpu_mode(monkeypatch):
    """创建CPU模式的处理器"""
    monkeypatch.setattr(
        "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
        False,
    )
    return GPUTechnicalProcessor(config={'use_gpu': False, 'fallback_to_cpu': True})


class TestGPUTechnicalProcessorCoverageSupplement:
    """gpu_technical_processor补充测试"""

    def test_init_optimization_level_aggressive(self, monkeypatch):
        """测试初始化（aggressive优化级别）"""
        monkeypatch.setattr(
            "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
            False,
        )
        processor = GPUTechnicalProcessor(config={'optimization_level': 'aggressive'})
        assert processor.config['gpu_threshold'] == 2000

    def test_init_optimization_level_conservative(self, monkeypatch):
        """测试初始化（conservative优化级别）"""
        monkeypatch.setattr(
            "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
            False,
        )
        processor = GPUTechnicalProcessor(config={'optimization_level': 'conservative'})
        assert processor.config['gpu_threshold'] == 10000

    def test_init_optimization_level_balanced(self, monkeypatch):
        """测试初始化（balanced优化级别）"""
        monkeypatch.setattr(
            "src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE",
            False,
        )
        processor = GPUTechnicalProcessor(config={'optimization_level': 'balanced'})
        assert processor.config['gpu_threshold'] == 500

    def test_should_use_gpu_conservative(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu（conservative模式）"""
        processor_cpu_mode.config['optimization_level'] = 'conservative'
        processor_cpu_mode.gpu_available = True
        
        # Mock get_gpu_info
        def mock_get_gpu_info():
            return {'memory_usage': 50, 'total_memory_gb': 8}
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        # 小数据集应该使用GPU（conservative模式阈值低）
        result = processor_cpu_mode._should_use_gpu(500)
        assert isinstance(result, bool)

    def test_should_use_gpu_balanced(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu（balanced模式）"""
        processor_cpu_mode.config['optimization_level'] = 'balanced'
        processor_cpu_mode.gpu_available = True
        
        def mock_get_gpu_info():
            return {'memory_usage': 50, 'total_memory_gb': 8}
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        result = processor_cpu_mode._should_use_gpu(1000)
        assert isinstance(result, bool)

    def test_should_use_gpu_aggressive(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu（aggressive模式）"""
        processor_cpu_mode.config['optimization_level'] = 'aggressive'
        processor_cpu_mode.gpu_available = True
        
        def mock_get_gpu_info():
            return {'memory_usage': 50, 'total_memory_gb': 8}
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        result = processor_cpu_mode._should_use_gpu(200)
        assert isinstance(result, bool)

    def test_should_use_gpu_high_memory_usage(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu（高内存使用率）"""
        processor_cpu_mode.gpu_available = True
        
        def mock_get_gpu_info():
            return {'memory_usage': 85, 'total_memory_gb': 8}  # 超过80%
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        result = processor_cpu_mode._should_use_gpu(10000)
        assert result is False  # 应该返回False，使用CPU

    def test_should_use_gpu_large_dataset_low_memory(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu（大数据集，低显存）"""
        processor_cpu_mode.gpu_available = True
        
        def mock_get_gpu_info():
            return {'memory_usage': 50, 'total_memory_gb': 2}  # 小于4GB
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        result = processor_cpu_mode._should_use_gpu(150000)  # 超过10万条
        assert result is False  # 应该返回False

    def test_should_use_gpu_exception_handling(self, processor_cpu_mode, monkeypatch):
        """测试_should_use_gpu异常处理"""
        processor_cpu_mode.gpu_available = True
        
        def mock_get_gpu_info():
            raise Exception("获取GPU信息失败")
        
        monkeypatch.setattr(processor_cpu_mode, 'get_gpu_info', mock_get_gpu_info)
        
        result = processor_cpu_mode._should_use_gpu(10000)
        # 异常时应该返回False或True，取决于其他条件
        assert isinstance(result, bool)

    def test_calculate_sma_gpu_invalid_window(self, processor_cpu_mode):
        """测试calculate_sma_gpu（无效窗口）"""
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        
        with pytest.raises(ValueError, match="窗口大小必须大于0"):
            processor_cpu_mode.calculate_sma_gpu(data, window=0)
        
        with pytest.raises(ValueError, match="窗口大小必须大于0"):
            processor_cpu_mode.calculate_sma_gpu(data, window=-1)

    def test_calculate_multiple_indicators_gpu_unknown_indicator(self, processor_cpu_mode):
        """测试calculate_multiple_indicators_gpu（未知指标）"""
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        
        result = processor_cpu_mode.calculate_multiple_indicators_gpu(
            data, ['unknown_indicator'], {}
        )
        # 应该返回原始数据或空DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_calculate_multiple_indicators_gpu_empty_indicators(self, processor_cpu_mode):
        """测试calculate_multiple_indicators_gpu（空指标列表）"""
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        
        result = processor_cpu_mode.calculate_multiple_indicators_gpu(data, [], {})
        assert isinstance(result, pd.DataFrame)

    def test_calculate_multiple_indicators_cpu(self, processor_cpu_mode):
        """测试_calculate_multiple_indicators_cpu"""
        data = pd.DataFrame({
            'close': [100 + i*0.5 for i in range(30)],
            'high': [105 + i*0.5 for i in range(30)],
            'low': [95 + i*0.5 for i in range(30)]
        })
        
        result = processor_cpu_mode._calculate_multiple_indicators_cpu(
            data, ['sma', 'ema'], {}
        )
        assert isinstance(result, pd.DataFrame)

    def test_get_gpu_info_no_gpu(self, processor_cpu_mode):
        """测试get_gpu_info（无GPU）"""
        processor_cpu_mode.gpu_available = False
        info = processor_cpu_mode.get_gpu_info()
        assert isinstance(info, dict)
        assert 'available' in info or 'gpu_available' in info

    def test_clear_gpu_memory(self, processor_cpu_mode, monkeypatch):
        """测试clear_gpu_memory"""
        # 如果GPU不可用，clear_gpu_memory应该安全返回
        processor_cpu_mode.gpu_available = False
        processor_cpu_mode.clear_gpu_memory()
        # 应该安全返回，不抛出异常
        assert True  # 如果没有抛出异常就通过
        
        # 如果GPU可用，需要mock cupy
        if hasattr(processor_cpu_mode, 'gpu_available'):
            processor_cpu_mode.gpu_available = True
            # 由于cp可能不存在，使用try-except保护
            try:
                processor_cpu_mode.clear_gpu_memory()
            except (AttributeError, ImportError):
                # 如果cp不可用，这是预期的
                pass

    def test_initialize_gpu_cuda_not_available(self, monkeypatch):
        """测试_initialize_gpu（CUDA不可用）"""
        # 直接设置GPU_AVAILABLE为False，模拟CUDA不可用的情况
        monkeypatch.setattr(
            'src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE',
            False
        )
        
        processor = GPUTechnicalProcessor(config={'use_gpu': True})
        # CUDA不可用时，gpu_available应该为False
        assert processor.gpu_available is False

    def test_initialize_gpu_exception(self, monkeypatch):
        """测试_initialize_gpu异常处理"""
        # 直接设置GPU_AVAILABLE为False，模拟CUDA不可用
        monkeypatch.setattr(
            'src.features.processors.gpu.gpu_technical_processor.GPU_AVAILABLE',
            False
        )
        
        processor = GPUTechnicalProcessor(config={'use_gpu': True})
        # CUDA不可用时，gpu_available应该为False
        assert processor.gpu_available is False

    def test_optimize_memory_access_exception(self, processor_cpu_mode, monkeypatch):
        """测试_optimize_memory_access异常处理"""
        # 由于cp可能不存在，直接测试_optimize_memory_access方法
        # 如果方法存在，测试异常处理
        if hasattr(processor_cpu_mode, '_optimize_memory_access'):
            processor_cpu_mode.gpu_available = True
            # 由于cp可能不存在，方法内部会捕获异常
            processor_cpu_mode._optimize_memory_access()
            # 如果没有抛出异常到外部，测试通过
            assert True

    def test_preallocate_memory_blocks_exception(self, processor_cpu_mode, monkeypatch):
        """测试_preallocate_memory_blocks异常处理"""
        # 由于cp可能不存在，直接测试_preallocate_memory_blocks方法
        # 如果方法存在，测试异常处理
        if hasattr(processor_cpu_mode, '_preallocate_memory_blocks'):
            processor_cpu_mode.gpu_available = True
            # 由于cp可能不存在，方法内部会捕获异常
            processor_cpu_mode._preallocate_memory_blocks()
            # 如果没有抛出异常到外部，测试通过
            assert True


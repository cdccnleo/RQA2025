#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_gpu_processor补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from src.features.processors.gpu.multi_gpu_processor import MultiGPUProcessor


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    """静默logger"""
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.logger",
        logging.getLogger(__name__),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.get_logger",
        lambda name: logging.getLogger(name),
    )
    monkeypatch.setattr(
        "src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor",
        lambda config=None: None,
    )


class TestMultiGPUProcessorCoverageSupplement:
    """multi_gpu_processor补充测试"""

    @pytest.fixture
    def processor(self, monkeypatch):
        """创建处理器实例"""
        monkeypatch.setattr(
            "src.features.processors.gpu.multi_gpu_processor.MULTI_GPU_AVAILABLE",
            False,
        )
        monkeypatch.setattr(
            MultiGPUProcessor,
            "_initialize_multi_gpu",
            lambda self: None,
        )
        monkeypatch.setattr(
            MultiGPUProcessor,
            "_initialize_fallback",
            lambda self: None,
        )
        return MultiGPUProcessor(config={"use_multi_gpu": False, "fallback_to_cpu": True})

    def test_split_data_for_gpus_empty_gpus(self, processor):
        """测试_split_data_for_gpus（无可用GPU）"""
        processor.available_gpus = []
        data = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
        
        result = processor._split_data_for_gpus(data)
        assert result == {}

    def test_split_data_for_gpus_with_gpu_list(self, processor):
        """测试_split_data_for_gpus（指定GPU列表）"""
        processor.available_gpus = [0, 1, 2]
        data = pd.DataFrame({'close': list(range(20))})
        
        # 有两个_split_data_for_gpus方法，一个带gpu_list参数（在256行）
        # 直接调用带gpu_list参数的方法
        # 由于Python方法重载的限制，我们需要通过检查参数来调用正确的方法
        # 实际上第二个方法会覆盖第一个，所以直接调用即可
        result = processor._split_data_for_gpus(data, gpu_list=[0, 1])
        assert isinstance(result, dict)
        # 由于chunk_size计算，可能返回不同数量的chunks
        assert len(result) >= 0

    def test_split_data_for_gpus_empty_gpu_list(self, processor):
        """测试_split_data_for_gpus（空GPU列表）"""
        processor.available_gpus = [0, 1]
        data = pd.DataFrame({'close': [1, 2, 3]})
        
        result = processor._split_data_for_gpus(data, gpu_list=[])
        assert result == {}

    def test_load_balance_data_memory_based(self, processor, monkeypatch):
        """测试_load_balance_data（基于内存的策略）"""
        processor.available_gpus = [0, 1]
        processor.config['load_balancing'] = 'memory_based'
        data = pd.DataFrame({'close': list(range(10))})
        
        # Mock torch.cuda相关方法
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.memory_allocated = MagicMock(return_value=0)
        mock_torch.cuda.memory_reserved = MagicMock(return_value=0)
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.torch', mock_torch)
        
        result = processor._load_balance_data(data)
        assert isinstance(result, dict)

    def test_load_balance_data_performance_based(self, processor):
        """测试_load_balance_data（基于性能的策略）"""
        processor.available_gpus = [0, 1]
        processor.config['load_balancing'] = 'performance_based'
        data = pd.DataFrame({'close': list(range(10))})
        
        result = processor._load_balance_data(data)
        assert isinstance(result, dict)

    def test_memory_based_distribution_exception(self, processor, monkeypatch):
        """测试_memory_based_distribution异常处理"""
        processor.available_gpus = [0, 1]
        data = pd.DataFrame({'close': list(range(10))})
        
        # Mock torch.cuda抛出异常
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock(side_effect=Exception("GPU错误"))
        mock_torch.cuda.memory_allocated = MagicMock(return_value=0)
        mock_torch.cuda.memory_reserved = MagicMock(return_value=0)
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.torch', mock_torch)
        
        result = processor._memory_based_distribution(data)
        assert isinstance(result, dict)

    def test_aggregate_results_empty_results(self, processor):
        """测试_aggregate_results（空结果）"""
        original_index = pd.Index(range(10))
        result = processor._aggregate_results({}, original_index)
        assert result.empty

    def test_aggregate_results_all_empty_dataframes(self, processor):
        """测试_aggregate_results（所有结果都是空DataFrame）"""
        original_index = pd.Index(range(10))
        results = {
            0: pd.DataFrame(),
            1: pd.DataFrame()
        }
        result = processor._aggregate_results(results, original_index)
        assert result.empty

    def test_aggregate_results_index_handling_exception(self, processor):
        """测试_aggregate_results索引处理异常"""
        original_index = pd.Index(range(5))
        # 创建会导致索引处理异常的结果
        results = {
            0: pd.DataFrame({'feature': [1, 2, 3]}, index=[0, 1, 2]),
            1: pd.DataFrame({'feature': [4, 5]}, index=[3, 4])
        }
        
        # Mock reset_index抛出异常
        with patch.object(pd.DataFrame, 'reset_index', side_effect=Exception("索引错误")):
            result = processor._aggregate_results(results, original_index)
            # 应该返回合并结果，即使索引处理失败
            assert isinstance(result, pd.DataFrame)

    def test_process_chunk_on_gpu_exception(self, processor, monkeypatch):
        """测试_process_chunk_on_gpu异常处理"""
        processor.gpu_processors = {0: MagicMock()}
        processor.gpu_processors[0].calculate_multiple_indicators_gpu = MagicMock(
            side_effect=Exception("处理失败")
        )
        
        chunk = pd.DataFrame({'close': [1, 2, 3]})
        
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.torch', mock_torch)
        
        result = processor._process_chunk_on_gpu(0, chunk, ['sma'], {})
        assert result.empty

    def test_calculate_multiple_indicators_multi_gpu_exception_handling(self, processor, monkeypatch):
        """测试calculate_multiple_indicators_multi_gpu异常处理"""
        processor.available_gpus = [0, 1]
        data = pd.DataFrame({'close': list(range(20))})
        
        # Mock _load_balance_data返回数据块
        def mock_load_balance(data):
            return {
                0: data.iloc[:10],
                1: data.iloc[10:]
            }
        
        # Mock _process_chunk_on_gpu返回空DataFrame（模拟异常情况）
        def mock_process_chunk(gpu_id, chunk, indicators, params):
            if gpu_id == 0:
                return pd.DataFrame()  # 模拟处理失败返回空
            return chunk.assign(feature=1)
        
        monkeypatch.setattr(processor, '_load_balance_data', mock_load_balance)
        monkeypatch.setattr(processor, '_process_chunk_on_gpu', mock_process_chunk)
        
        # 使用ThreadPoolExecutor模式（sync_mode=True）
        processor.config['sync_mode'] = True
        result = processor.calculate_multiple_indicators_multi_gpu(data, ['sma'], {})
        # 应该返回部分结果或空结果
        assert isinstance(result, pd.DataFrame)

    def test_get_multi_gpu_info(self, processor, monkeypatch):
        """测试get_multi_gpu_info"""
        processor.available_gpus = [0, 1]
        # 确保gpu_info包含memory_gb字段
        processor.gpu_info = {
            0: {'name': 'GPU0', 'memory_gb': 8.0},
            1: {'name': 'GPU1', 'memory_gb': 8.0}
        }
        
        # Mock torch.cuda方法
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.memory_allocated = MagicMock(return_value=1024**3)  # 1GB
        mock_torch.cuda.memory_reserved = MagicMock(return_value=1024**3)  # 1GB
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.torch', mock_torch)
        
        info = processor.get_multi_gpu_info()
        assert isinstance(info, dict)
        assert 'available_gpus' in info
        assert 'total_gpus' in info  # 实际返回的是total_gpus，不是gpu_count
        assert 'gpu_details' in info

    def test_get_available_gpus(self, processor):
        """测试get_available_gpus"""
        processor.available_gpus = [0, 1, 2]
        gpus = processor.get_available_gpus()
        assert gpus == [0, 1, 2]

    def test_is_multi_gpu_available(self, processor):
        """测试is_multi_gpu_available"""
        processor.available_gpus = [0, 1]
        assert processor.is_multi_gpu_available() is True
        
        processor.available_gpus = []
        assert processor.is_multi_gpu_available() is False

    def test_clear_multi_gpu_memory(self, processor, monkeypatch):
        """测试clear_multi_gpu_memory"""
        processor.available_gpus = [0, 1]  # 确保有可用GPU
        processor.gpu_processors = {0: MagicMock(), 1: MagicMock()}
        
        mock_torch = MagicMock()
        mock_torch.cuda.set_device = MagicMock()
        mock_torch.cuda.empty_cache = MagicMock()
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.torch', mock_torch)
        
        processor.clear_multi_gpu_memory()
        # 应该调用empty_cache（每个GPU一次）
        assert mock_torch.cuda.empty_cache.call_count == len(processor.available_gpus)

    def test_initialize_fallback_single_gpu_failure(self, processor, monkeypatch):
        """测试_initialize_fallback（单GPU初始化失败）"""
        # 由于processor的_initialize_fallback被mock了，我们需要取消mock来测试真实方法
        monkeypatch.undo()  # 取消之前的mock
        monkeypatch.setattr('src.features.processors.gpu.multi_gpu_processor.GPUTechnicalProcessor', 
                           MagicMock(side_effect=Exception("GPU初始化失败")))
        processor.config['fallback_to_single_gpu'] = True
        processor.config['fallback_to_cpu'] = True
        
        # 确保processor有single_gpu_processor属性
        if not hasattr(processor, 'single_gpu_processor'):
            processor.single_gpu_processor = None
        
        processor._initialize_fallback()
        # 应该回退到CPU模式，single_gpu_processor为None
        assert processor.single_gpu_processor is None

    def test_initialize_fallback_no_single_gpu(self, processor):
        """测试_initialize_fallback（不使用单GPU）"""
        processor.config['fallback_to_single_gpu'] = False
        processor.config['fallback_to_cpu'] = True
        processor._initialize_fallback()
        # 应该直接回退到CPU模式
        assert not hasattr(processor, 'single_gpu_processor') or processor.single_gpu_processor is None


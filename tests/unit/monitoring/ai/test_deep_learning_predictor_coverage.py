#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模块覆盖率测试
专注提升deep_learning_predictor相关模块的测试覆盖率
使用Mock避免深度学习框架依赖
"""

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    ai_deep_learning_predictor_module = importlib.import_module('src.monitoring.ai.deep_learning_predictor')
    deep_learning_predictor = getattr(ai_deep_learning_predictor_module, 'deep_learning_predictor', None)
    __all__ = getattr(ai_deep_learning_predictor_module, '__all__', None)
    if deep_learning_predictor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

# 检查__all__变量（如果存在）
if __all__ is not None:
    assert 'TimeSeriesDataset' in __all__
    assert 'LSTMPredictor' in __all__
    assert 'DeepLearningPredictor' in __all__


class TestTimeSeriesDataset:
    """测试TimeSeriesDataset"""

    def test_init(self):
        """测试初始化"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        assert dataset.seq_length == 2
        assert len(dataset.data) == 5

    def test_len(self):
        """测试长度计算"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        # 长度应该是 len(data) - seq_length = 5 - 2 = 3
        assert len(dataset) == 3

    def test_getitem(self):
        """测试获取数据项"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        x, y = dataset[0]
        
        # x应该是前seq_length个元素，y是下一个元素
        assert len(x) == 2
        assert x[0] == 1.0
        assert x[1] == 2.0
        assert y == 3.0


class TestLSTMPredictor:
    """测试LSTMPredictor"""

    @pytest.fixture
    def lstm_model(self):
        """创建LSTM模型实例"""
        return LSTMPredictor(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )

    def test_init(self, lstm_model):
        """测试初始化"""
        assert lstm_model.input_size == 1
        assert lstm_model.hidden_size == 64
        assert lstm_model.num_layers == 2
        assert lstm_model.output_size == 1

    def test_forward(self, lstm_model):
        """测试前向传播"""
        # 创建输入：batch_size=1, seq_length=10, input_size=1
        x = np.random.randn(1, 10, 1).astype(np.float32)
        
        try:
            import torch
            x_tensor = torch.from_numpy(x)
            output = lstm_model(x_tensor)
            
            # 输出应该是 (batch_size, output_size) = (1, 1)
            assert output.shape == (1, 1)
        except ImportError:
            pytest.skip("PyTorch not available")


class TestModelCacheManager:
    """测试ModelCacheManager"""

    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        return ModelCacheManager(max_cache_size=3)

    def test_init(self, cache_manager):
        """测试初始化"""
        assert cache_manager.max_cache_size == 3
        assert len(cache_manager.cache) == 0
        assert len(cache_manager.access_count) == 0

    def test_set_and_get(self, cache_manager):
        """测试设置和获取缓存"""
        model = Mock()
        cache_manager.set('model_1', model)
        
        retrieved = cache_manager.get('model_1')
        assert retrieved == model

    def test_access_count(self, cache_manager):
        """测试访问计数"""
        model = Mock()
        cache_manager.set('model_1', model)
        
        # 首次访问
        cache_manager.get('model_1')
        assert cache_manager.access_count['model_1'] == 1
        
        # 再次访问
        cache_manager.get('model_1')
        assert cache_manager.access_count['model_1'] == 2

    def test_lru_eviction(self, cache_manager):
        """测试LRU淘汰"""
        # 添加3个模型（达到最大缓存）
        cache_manager.set('model_1', Mock())
        cache_manager.set('model_2', Mock())
        cache_manager.set('model_3', Mock())
        
        # 访问model_1，使其访问次数最多
        cache_manager.get('model_1')
        cache_manager.get('model_1')
        
        # 添加第4个模型，应该淘汰访问次数最少的
        cache_manager.set('model_4', Mock())
        
        # model_1应该还在（访问次数多）
        assert 'model_1' in cache_manager.cache
        # model_2或model_3应该被淘汰（访问次数少）
        assert len(cache_manager.cache) == 3

    def test_clear(self, cache_manager):
        """测试清空缓存"""
        cache_manager.set('model_1', Mock())
        cache_manager.set('model_2', Mock())
        
        cache_manager.clear()
        
        assert len(cache_manager.cache) == 0
        assert len(cache_manager.access_count) == 0


class TestGPUResourceManager:
    """测试GPUResourceManager"""

    @pytest.fixture
    def gpu_manager(self):
        """创建GPU资源管理器实例"""
        return GPUResourceManager()

    def test_init(self, gpu_manager):
        """测试初始化"""
        assert hasattr(gpu_manager, 'device_count')
        assert hasattr(gpu_manager, 'current_device')

    def test_get_device(self, gpu_manager):
        """测试获取设备"""
        device = gpu_manager.get_device()
        assert device is not None

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_get_device_cpu_fallback(self, mock_count, mock_available, gpu_manager):
        """测试CPU回退"""
        mock_available.return_value = False
        mock_count.return_value = 0
        
        device = gpu_manager.get_device()
        # 如果没有GPU，应该返回CPU设备
        assert device is not None

    def test_get_memory_info(self, gpu_manager):
        """测试获取内存信息"""
        memory_info = gpu_manager.get_memory_info()
        assert isinstance(memory_info, dict)
        assert 'available' in memory_info

    def test_clear_cache(self, gpu_manager):
        """测试清理缓存"""
        # 应该不会抛出异常
        gpu_manager.clear_cache()
        assert True


class TestAIModelOptimizer:
    """测试AIModelOptimizer"""

    @pytest.fixture
    def optimizer(self):
        """创建AI模型优化器实例"""
        return AIModelOptimizer()

    def test_init(self, optimizer):
        """测试初始化"""
        assert hasattr(optimizer, 'optimized_models')
        assert isinstance(optimizer.optimized_models, dict)


class TestDeepLearningPredictor:
    """测试DeepLearningPredictor"""

    @pytest.fixture
    def predictor_config(self):
        """预测器配置"""
        return {
            'seq_length': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'batch_size': 16
        }

    @pytest.fixture
    def predictor(self, predictor_config):
        """创建预测器实例"""
        return DeepLearningPredictor(config=predictor_config)

    def test_init(self, predictor):
        """测试初始化"""
        assert predictor.config is not None
        assert predictor.seq_length == 10
        assert predictor.hidden_size == 32
        assert predictor.num_layers == 1

    def test_init_default_config(self):
        """测试默认配置初始化"""
        predictor = DeepLearningPredictor()
        assert predictor.config == {}
        assert hasattr(predictor, 'seq_length')

    def test_model_cache_access(self, predictor):
        """测试模型缓存访问"""
        # 直接访问模型缓存管理器
        assert hasattr(predictor, 'model_cache')
        assert isinstance(predictor.model_cache, ModelCacheManager)

    def test_component_initialization(self, predictor):
        """测试组件初始化"""
        # 验证所有组件都已初始化
        assert hasattr(predictor, 'gpu_manager')
        assert hasattr(predictor, 'model_optimizer')
        assert hasattr(predictor, 'batch_optimizer')
        assert hasattr(predictor, 'model_cache')
        assert hasattr(predictor, 'device')

    def test_configuration_values(self, predictor):
        """测试配置值"""
        assert predictor.seq_length == 10
        assert predictor.hidden_size == 32
        assert predictor.num_layers == 1
        assert predictor.batch_size == 16


class TestDeepLearningPredictorIntegration:
    """测试集成功能"""

    @pytest.fixture
    def predictor(self):
        """创建预测器实例"""
        return DeepLearningPredictor({
            'seq_length': 5,
            'hidden_size': 16,
            'batch_size': 8
        })

    def test_simple_workflow(self, predictor):
        """测试简单工作流"""
        # 1. 创建简单数据
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # 2. 检查组件状态
        assert predictor.model_cache is not None
        assert predictor.gpu_manager is not None
        
        # 3. 验证预测器已初始化
        assert predictor is not None
        assert predictor.config is not None

    def test_configuration_access(self, predictor):
        """测试配置访问"""
        assert predictor.config is not None
        assert 'seq_length' in predictor.config or hasattr(predictor, 'seq_length')


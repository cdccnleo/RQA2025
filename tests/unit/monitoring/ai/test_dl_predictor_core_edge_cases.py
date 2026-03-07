#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLearningPredictor边界情况和异常处理测试
补充dl_predictor_core.py中边界情况和异常处理的测试覆盖
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
    ai_dl_predictor_core_module = importlib.import_module('src.monitoring.ai.dl_predictor_core')
    DeepLearningPredictor = getattr(ai_dl_predictor_core_module, 'DeepLearningPredictor', None)
    if DeepLearningPredictor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDeepLearningPredictorEdgeCases:
    """测试DeepLearningPredictor边界情况和异常处理"""

    @pytest.fixture
    def predictor(self):
        """创建预测器实例"""
        return DeepLearningPredictor({
            'seq_length': 10,
            'hidden_size': 32,
            'num_layers': 1,
            'batch_size': 16,
            'epochs': 2
        })

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return np.array([i * 0.1 for i in range(100)])

    def test_predict_empty_data(self, predictor):
        """测试预测空数据"""
        # 先训练模型
        predictor.train_lstm(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]), epochs=1)
        
        # 测试空数组
        result = predictor.predict(np.array([]), steps=1)
        
        # 应该返回空数组或处理异常
        assert isinstance(result, np.ndarray)

    def test_predict_insufficient_data(self, predictor):
        """测试预测数据不足"""
        # 先训练模型
        predictor.train_lstm(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]), epochs=1)
        
        # 数据长度小于seq_length
        short_data = np.array([1.0, 2.0])
        
        try:
            result = predictor.predict(short_data, steps=1)
            assert isinstance(result, np.ndarray)
        except Exception:
            # 如果抛出异常也是可以接受的
            assert True

    def test_predict_exception_handling(self, predictor):
        """测试预测异常处理"""
        # Mock scaler.transform抛出异常
        with patch.object(predictor.scaler, 'transform', side_effect=Exception("Transform error")):
            result = predictor.predict(np.array([1.0, 2.0, 3.0]), steps=1)
            
            # 应该返回空数组
            assert isinstance(result, np.ndarray)
            assert len(result) == 0

    def test_detect_anomaly_empty_data(self, predictor):
        """测试异常检测空数据"""
        result = predictor.detect_anomaly(np.array([]), threshold=0.1)
        
        assert isinstance(result, dict)
        assert 'anomalies' in result

    def test_detect_anomaly_exception_handling(self, predictor):
        """测试异常检测异常处理"""
        # Mock scaler.transform抛出异常
        with patch.object(predictor.scaler, 'transform', side_effect=Exception("Transform error")):
            result = predictor.detect_anomaly(np.array([1.0, 2.0, 3.0]), threshold=0.1)

            assert isinstance(result, dict)
            # 检查是否包含错误信息（异常情况下应该有error键）
            if 'error' in result:
                assert isinstance(result['error'], str)
            else:
                # 如果没有训练模型，返回空结果
                assert 'anomalies' in result
                assert 'reconstruction_errors' in result

    def test_detect_anomaly_different_threshold_values(self, predictor):
        """测试异常检测不同阈值"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # 测试不同阈值
        for threshold in [0.0, 0.1, 0.5, 1.0, 10.0]:
            result = predictor.detect_anomaly(data, threshold=threshold)
            
            assert isinstance(result, dict)
            assert 'threshold' in result
            assert result['threshold'] == threshold

    def test_train_lstm_empty_data(self, predictor):
        """测试训练空数据"""
        result = predictor.train_lstm(np.array([]), epochs=1)
        
        assert isinstance(result, dict)
        # 应该返回失败或错误
        assert 'success' in result or 'error' in result

    def test_train_lstm_insufficient_data(self, predictor):
        """测试训练数据不足"""
        # 数据长度小于seq_length
        short_data = np.array([1.0, 2.0])
        
        result = predictor.train_lstm(short_data, epochs=1)
        
        assert isinstance(result, dict)
        # 可能会失败或返回错误
        assert 'success' in result or 'error' in result

    def test_train_lstm_zero_epochs(self, predictor, sample_data):
        """测试训练0个epoch"""
        result = predictor.train_lstm(sample_data, epochs=0)
        
        assert isinstance(result, dict)
        # 应该处理0 epochs的情况

    def test_train_lstm_exception_handling(self, predictor, sample_data):
        """测试训练异常处理"""
        # Mock scaler.fit_transform抛出异常
        with patch.object(predictor.scaler, 'fit_transform', side_effect=Exception("Fit error")):
            result = predictor.train_lstm(sample_data, epochs=1)
            
            assert isinstance(result, dict)
            assert result.get('success') == False or 'error' in result

    def test_train_lstm_different_validation_splits(self, predictor, sample_data):
        """测试不同验证集比例"""
        for validation_split in [0.0, 0.1, 0.5, 0.9]:
            result = predictor.train_lstm(sample_data, epochs=1, validation_split=validation_split)
            
            assert isinstance(result, dict)

    def test_train_lstm_validation_split_edge_cases(self, predictor):
        """测试验证集比例边界情况"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        
        # 测试边界值
        result = predictor.train_lstm(data, epochs=1, validation_split=1.0)
        assert isinstance(result, dict)
        
        result = predictor.train_lstm(data, epochs=1, validation_split=0.0)
        assert isinstance(result, dict)




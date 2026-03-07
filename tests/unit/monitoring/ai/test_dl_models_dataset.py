#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLearning模型和数据集测试
补充TimeSeriesDataset的测试
"""

import pytest
import numpy as np

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np

try:
    import torch
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
    ai_dl_models_module = importlib.import_module('src.monitoring.ai.dl_models')
    TimeSeriesDataset = getattr(ai_dl_models_module, 'TimeSeriesDataset', None)
    if TimeSeriesDataset is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

except ImportError:
    pytestmark = pytest.mark.skip("AI modules not available")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTimeSeriesDataset:
    """测试TimeSeriesDataset"""

    def test_init(self):
        """测试初始化"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        assert dataset.data is not None
        assert dataset.seq_length == 2

    def test_len(self):
        """测试长度"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        # 长度应该是 len(data) - seq_length
        assert len(dataset) == 3  # 5 - 2 = 3

    def test_len_edge_case(self):
        """测试边界情况长度"""
        data = np.array([1.0, 2.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        # 如果seq_length >= len(data)，长度应该是0
        assert len(dataset) == 0

    def test_getitem(self):
        """测试获取项目"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        x, y = dataset[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (2,)  # seq_length
        assert y.shape == ()  # scalar

    def test_getitem_values(self):
        """测试获取的数值正确性"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = TimeSeriesDataset(data, seq_length=2)
        
        x, y = dataset[0]
        
        # 第一个样本：x=[1,2], y=3
        assert x.tolist() == [1.0, 2.0]
        assert y.item() == 3.0

    def test_getitem_multiple(self):
        """测试获取多个项目"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dataset = TimeSeriesDataset(data, seq_length=3)
        
        x1, y1 = dataset[0]  # x=[1,2,3], y=4
        x2, y2 = dataset[1]  # x=[2,3,4], y=5
        
        assert x1.tolist() == [1.0, 2.0, 3.0]
        assert y1.item() == 4.0
        assert x2.tolist() == [2.0, 3.0, 4.0]
        assert y2.item() == 5.0

    def test_getitem_dtype(self):
        """测试数据类型"""
        data = np.array([1.0, 2.0, 3.0])
        dataset = TimeSeriesDataset(data, seq_length=1)
        
        x, y = dataset[0]
        
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


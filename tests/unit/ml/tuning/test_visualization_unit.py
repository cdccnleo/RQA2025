"""
测试调参可视化工具
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.ml.tuning.utils.visualization import TuningVisualizer


class TestTuningVisualizer:
    """测试调参可视化工具"""

    def test_plot_optimization_history(self):
        """测试绘制优化历史曲线"""
        # 创建模拟trials数据
        trials_data = {
            'number': list(range(10)),
            'value': np.random.randn(10).tolist()
        }
        trials = pd.DataFrame(trials_data)

        # 测试不保存
        with patch('matplotlib.pyplot.figure'), \
             patch('seaborn.lineplot'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.savefig', side_effect=Exception("Should not be called")):
            result = TuningVisualizer.plot_optimization_history(trials)
            assert result is None

    def test_plot_optimization_history_with_save(self):
        """测试绘制优化历史曲线并保存"""
        trials_data = {
            'number': list(range(5)),
            'value': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        trials = pd.DataFrame(trials_data)

        with patch('matplotlib.pyplot.figure'), \
             patch('seaborn.lineplot'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('pathlib.Path.mkdir'), \
             patch('matplotlib.pyplot.gcf') as mock_gcf:
            mock_gcf.return_value = MagicMock()
            result = TuningVisualizer.plot_optimization_history(trials, save_path="test_path/history.png")
            mock_savefig.assert_called_once()
            assert result is not None

    def test_plot_param_importance(self):
        """测试绘制参数重要性"""
        importance = {
            'param1': 0.8,
            'param2': 0.6,
            'param3': 0.4
        }

        with patch('matplotlib.pyplot.figure'), \
             patch('seaborn.barplot'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.savefig', side_effect=Exception("Should not be called")):
            result = TuningVisualizer.plot_param_importance(importance)
            assert result is not None

    def test_plot_param_importance_with_save(self):
        """测试绘制参数重要性并保存"""
        importance = {
            'lr': 0.7,
            'batch_size': 0.3
        }

        with patch('matplotlib.pyplot.figure'), \
             patch('seaborn.barplot'), \
             patch('matplotlib.pyplot.title'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('pathlib.Path.mkdir'), \
             patch('matplotlib.pyplot.gcf') as mock_gcf:
            mock_gcf.return_value = MagicMock()
            result = TuningVisualizer.plot_param_importance(importance, save_path="test_path/importance.png")
            mock_savefig.assert_called_once()
            assert result is not None

    def test_plot_parallel_coordinate(self):
        """测试绘制平行坐标图"""
        # 由于optuna的平行坐标图需要真实的study对象，这里跳过实际测试
        pytest.skip("Parallel coordinate plot requires real optuna study object")

    def test_plot_parallel_coordinate_with_save(self):
        """测试绘制平行坐标图并保存"""
        # 由于optuna的平行坐标图需要真实的study对象，这里跳过实际测试
        pytest.skip("Parallel coordinate plot with save requires real optuna study object")

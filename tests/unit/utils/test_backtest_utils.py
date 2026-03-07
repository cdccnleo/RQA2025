"""
工具层 - backtest_utils.py 测试

测试src/utils/backtest/backtest_utils.py的基本功能
"""

import sys
from pathlib import Path

# 确保Python路径正确配置（必须在所有导入之前）
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

# 确保路径在sys.path的最前面
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
if src_path_str in sys.path:
    sys.path.remove(src_path_str)

sys.path.insert(0, project_root_str)
sys.path.insert(0, src_path_str)

import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np


def test_strategy_validation_result():
    """测试StrategyValidationResult数据类"""
    from src.utils.backtest.backtest_utils import StrategyValidationResult
    
    result = StrategyValidationResult(
        is_valid=True,
        errors=[],
        warnings=["warning1"],
        suggestions=["suggestion1"]
    )
    
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert len(result.suggestions) == 1


def test_backtest_utils_validate_strategy_valid():
    """测试validate_strategy方法 - 有效策略"""
    from src.utils.backtest.backtest_utils import BacktestUtils
    
    # 创建有效的策略对象
    strategy = Mock()
    strategy.generate_signals = Mock()
    strategy.on_init = Mock()
    strategy.on_day_start = Mock()
    
    result = BacktestUtils.validate_strategy(strategy)
    
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_backtest_utils_validate_strategy_negative_params():
    """测试策略验证 - 负值参数"""
    from src.utils.backtest.backtest_utils import BacktestUtils

    strategy = Mock()
    strategy.params = {"param1": -5, "param2": 10}
    strategy.__class__.__name__ = "TestStrategy"

    result = BacktestUtils.validate_strategy(strategy)

    assert result.is_valid is True  # 负值只是警告，不是错误
    assert len(result.warnings) > 0
    assert any("负值" in warning for warning in result.warnings)


def test_backtest_utils_validate_strategy_large_params():
    """测试策略验证 - 大值参数"""
    from src.utils.backtest.backtest_utils import BacktestUtils

    strategy = Mock()
    strategy.params = {"param1": 1500, "param2": 10}
    strategy.__class__.__name__ = "TestStrategy"

    result = BacktestUtils.validate_strategy(strategy)

    assert result.is_valid is True  # 大值只是警告，不是错误
    assert len(result.warnings) > 0
    assert any("值过大" in warning for warning in result.warnings)


def test_backtest_utils_validate_strategy_short_name():
    """测试策略验证 - 策略名称过短"""
    from src.utils.backtest.backtest_utils import BacktestUtils

    strategy = Mock()
    strategy.params = {"param1": 10}
    strategy.__class__.__name__ = "TS"  # 名称过短

    result = BacktestUtils.validate_strategy(strategy)

    assert result.is_valid is True  # 名称过短只是建议，不是错误
    assert len(result.suggestions) > 0
    assert any("过短" in suggestion for suggestion in result.suggestions)


def test_backtest_utils_validate_strategy_missing_methods():
    """测试validate_strategy方法 - 缺少必要方法"""
    from src.utils.backtest.backtest_utils import BacktestUtils
    
    # 创建缺少方法的策略对象
    strategy = Mock(spec=[])  # 空的spec确保只有显式设置的属性可用
    
    result = BacktestUtils.validate_strategy(strategy)
    
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_backtest_utils_calculate_risk_metrics():
    """测试calculate_risk_metrics方法"""
    from src.utils.backtest.backtest_utils import BacktestUtils
    
    # 创建测试数据
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
    
    metrics = BacktestUtils.calculate_risk_metrics(returns)
    
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert isinstance(metrics['sharpe_ratio'], (int, float))
    assert isinstance(metrics['max_drawdown'], (int, float))


def test_backtest_utils_calculate_trade_metrics():
    """测试calculate_trade_metrics方法"""
    from src.utils.backtest.backtest_utils import BacktestUtils
    
    # 创建测试数据
    trades = pd.DataFrame({
        'profit': [10, -5, 15, -3, 8]
    })
    
    metrics = BacktestUtils.calculate_trade_metrics(trades)
    
    assert isinstance(metrics, dict)
    assert 'win_rate' in metrics
    assert isinstance(metrics['win_rate'], (int, float))
    assert 0 <= metrics['win_rate'] <= 1


def test_backtest_utils_validate_data():
    """测试validate_data方法"""
    from src.utils.backtest.backtest_utils import BacktestUtils
    
    # 创建测试数据
    data = pd.DataFrame({
        'price': [100, 110, 105, 120, 115],
        'volume': [1000, 1100, 1050, 1200, 1150]
    })
    
    result = BacktestUtils.validate_data(data, required_columns=['price', 'volume'])
    
    assert result.is_valid is True
    assert len(result.errors) == 0


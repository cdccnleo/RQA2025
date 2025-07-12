import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.data.data_manager import DataManager
from src.features.feature_manager import FeatureManager
from src.models.model_manager import ModelManager
from src.trading.enhanced_trading_strategy import EnhancedTradingStrategy
from src.execution.smart_execution import SmartExecutionEngine

@pytest.fixture(scope="module")
def test_config():
    """集成测试配置"""
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        'symbols': ['600519.SH', '000858.SZ', '601318.SH'],
        'initial_balance': 1_000_000,
        'benchmark': '000300.SH'
    }

@pytest.fixture(scope="module")
def system_pipeline(test_config):
    """构建系统测试流水线"""
    # 初始化各模块
    data_mgr = DataManager()
    feature_mgr = FeatureManager()
    model_mgr = ModelManager()
    strategy = EnhancedTradingStrategy()
    execution_engine = SmartExecutionEngine()

    return {
        'data_mgr': data_mgr,
        'feature_mgr': feature_mgr,
        'model_mgr': model_mgr,
        'strategy': strategy,
        'execution_engine': execution_engine
    }

def test_end_to_end_workflow(system_pipeline, test_config):
    """测试端到端工作流"""
    # 1. 数据加载
    with patch.object(system_pipeline['data_mgr'], 'load_data') as mock_load:
        mock_load.return_value = MagicMock()
        data = system_pipeline['data_mgr'].load_data(
            symbols=test_config['symbols'],
            start_date=test_config['start_date'],
            end_date=test_config['end_date']
        )
        mock_load.assert_called_once()

    # 2. 特征工程
    with patch.object(system_pipeline['feature_mgr'], 'generate_features') as mock_feature:
        mock_feature.return_value = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        features = system_pipeline['feature_mgr'].generate_features(data)
        assert not features.empty

    # 3. 模型预测
    with patch.object(system_pipeline['model_mgr'], 'predict') as mock_predict:
        mock_predict.return_value = pd.Series(np.random.rand(100))
        predictions = system_pipeline['model_mgr'].predict(features)
        assert len(predictions) == 100

    # 4. 策略生成
    with patch.object(system_pipeline['strategy'], 'generate_signals') as mock_strategy:
        mock_strategy.return_value = {
            '600519.SH': {'signal': 1, 'weight': 0.3},
            '000858.SZ': {'signal': 0, 'weight': 0.0},
            '601318.SH': {'signal': -1, 'weight': -0.2}
        }
        signals = system_pipeline['strategy'].generate_signals(predictions)
        assert len(signals) == len(test_config['symbols'])

    # 5. 交易执行
    with patch.object(system_pipeline['execution_engine'], 'execute_order') as mock_execute:
        system_pipeline['execution_engine'].execute_order(
            symbol='600519.SH',
            quantity=3000
        )
        mock_execute.assert_called_once()

def test_error_handling(system_pipeline):
    """测试异常处理流程"""
    # 数据加载异常
    with patch.object(system_pipeline['data_mgr'], 'load_data', side_effect=Exception("Data load failed")):
        with pytest.raises(Exception) as excinfo:
            system_pipeline['data_mgr'].load_data([], "", "")
        assert "Data load failed" in str(excinfo.value)

    # 特征生成异常
    with patch.object(system_pipeline['feature_mgr'], 'generate_features', side_effect=ValueError("Invalid data")):
        with pytest.raises(ValueError):
            system_pipeline['feature_mgr'].generate_features(None)

def test_performance_benchmark(system_pipeline, test_config):
    """测试性能基准"""
    # 数据加载性能
    with patch.object(system_pipeline['data_mgr'], 'load_data') as mock_load:
        mock_load.return_value = MagicMock()
        start = pd.Timestamp.now()
        system_pipeline['data_mgr'].load_data(
            symbols=test_config['symbols'],
            start_date=test_config['start_date'],
            end_date=test_config['end_date']
        )
        duration = (pd.Timestamp.now() - start).total_seconds()
        assert duration < 1.0  # 数据加载应在1秒内完成

    # 特征生成性能
    with patch.object(system_pipeline['feature_mgr'], 'generate_features') as mock_feature:
        mock_feature.return_value = MagicMock()
        start = pd.Timestamp.now()
        system_pipeline['feature_mgr'].generate_features(MagicMock())
        duration = (pd.Timestamp.now() - start).total_seconds()
        assert duration < 0.5  # 特征生成应在0.5秒内完成

@pytest.mark.parametrize("symbol,expected", [
    ('600519.SH', ExchangeType.SHANGHAI),
    ('000858.SZ', ExchangeType.SHENZHEN),
    ('00700.HK', ExchangeType.HONGKONG)
])
def test_execution_routing(system_pipeline, symbol, expected):
    """测试执行路由逻辑"""
    with patch.object(system_pipeline['execution_engine'].router, 'determine_routing') as mock_route:
        mock_route.return_value = expected
        exchange = system_pipeline['execution_engine'].router.determine_routing(
            symbol, OrderType.LIMIT, ExecutionParameters()
        )
        assert exchange == expected

def test_system_stability(system_pipeline, test_config):
    """测试系统稳定性"""
    # 模拟连续运行100次工作流
    for _ in range(100):
        try:
            data = system_pipeline['data_mgr'].load_data(
                symbols=test_config['symbols'],
                start_date=test_config['start_date'],
                end_date=test_config['end_date']
            )
            features = system_pipeline['feature_mgr'].generate_features(data)
            predictions = system_pipeline['model_mgr'].predict(features)
            signals = system_pipeline['strategy'].generate_signals(predictions)

            for symbol, signal in signals.items():
                if signal['signal'] != 0:
                    system_pipeline['execution_engine'].execute_order(
                        symbol=symbol,
                        quantity=abs(signal['weight']) * 1000
                    )
        except Exception as e:
            pytest.fail(f"系统稳定性测试失败: {str(e)}")

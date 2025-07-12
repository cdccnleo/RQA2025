import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from models.model_monitor import (
    KSTestDetector,
    ModelMonitor,
    AdaptiveModelManager,
    OnlineLearner,
    DriftType,
    ModelPerformance,
    DriftAlert
)

@pytest.fixture
def sample_data():
    """生成测试数据"""
    dates = pd.date_range('2023-01-01', periods=100)
    baseline = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.uniform(0, 1, 100)
    }, index=dates)

    # 当前数据有轻微漂移
    current = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 100),
        'feature2': np.random.uniform(0.2, 1.2, 100)
    }, index=dates + pd.Timedelta(days=100))

    return baseline, current

@pytest.fixture
def mock_model():
    """模拟模型"""
    model = MagicMock()
    model.predict.return_value = pd.Series(np.random.randint(0, 2, 100))
    return model

def test_ks_detector(sample_data):
    """测试KS漂移检测"""
    baseline, current = sample_data
    detector = KSTestDetector(threshold=0.05)

    alerts = detector.detect(baseline, current)
    assert isinstance(alerts, list)
    assert len(alerts) == 2  # 两个特征都应检测到漂移
    assert all(alert.severity > 0 for alert in alerts)
    assert all(alert.p_value < 0.05 for alert in alerts)

def test_ks_detector_insufficient_samples():
    """测试样本不足情况"""
    detector = KSTestDetector()
    with pytest.warns(UserWarning):
        result = detector.detect(
            pd.DataFrame({'f1': [1,2,3]}),
            pd.DataFrame({'f1': [1,2]})
        )
        assert result is None

def test_model_monitor_logging(sample_data):
    """测试模型表现记录"""
    monitor = ModelMonitor(detectors={})
    performance = ModelPerformance(
        accuracy=0.9,
        precision=0.85,
        recall=0.8,
        f1=0.82
    )

    monitor.log_performance(
        model_name='test_model',
        performance=performance,
        timestamp=pd.Timestamp.now()
    )

    assert len(monitor.performance_history) == 1
    assert monitor.performance_history[0]['model'] == 'test_model'

def test_model_monitor_drift_check(sample_data):
    """测试漂移检查"""
    baseline, current = sample_data
    monitor = ModelMonitor(detectors={
        'ks_test': KSTestDetector()
    })

    alerts = monitor.check_drift(
        model_name='test_model',
        baseline_data=baseline,
        current_data=current
    )

    assert len(alerts) > 0
    assert len(monitor.drift_alerts) == len(alerts)
    assert all(alert['model'] == 'test_model' for alert in monitor.drift_alerts)

def test_adaptive_model_switch(mock_model):
    """测试模型切换"""
    models = {
        'model1': mock_model,
        'model2': mock_model
    }
    monitor = ModelMonitor(detectors={})
    manager = AdaptiveModelManager(
        models=models,
        monitor=monitor
    )

    assert manager.switch_model('model1')
    assert manager.active_model == 'model1'

    # 切换到不存在的模型
    assert not manager.switch_model('invalid_model')

def test_adaptive_model_evaluation(mock_model):
    """测试模型评估"""
    models = {
        'model1': mock_model,
        'model2': mock_model
    }
    monitor = ModelMonitor(detectors={})
    manager = AdaptiveModelManager(
        models=models,
        monitor=monitor
    )

    X = pd.DataFrame(np.random.randn(100, 3))
    y = pd.Series(np.random.randint(0, 2, 100))

    results = manager.evaluate_candidates(X, y)
    assert len(results) == 2
    assert all(isinstance(perf, ModelPerformance) for perf in results.values())
    assert len(monitor.performance_history) == 2

def test_adaptive_update(mock_model):
    """测试自适应更新"""
    models = {
        'model1': mock_model,
        'model2': mock_model
    }
    monitor = ModelMonitor(detectors={})
    manager = AdaptiveModelManager(
        models=models,
        monitor=monitor,
        fallback_model='model1'
    )

    # 设置model2表现更好
    with patch.object(manager, '_calculate_improvement', return_value=0.2):
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.randint(0, 2, 100))

        manager.candidate_models = ['model2']
        updated = manager.adaptive_update(X, y, threshold=0.1)

        assert updated
        assert manager.active_model == 'model2'

def test_online_learner(mock_model):
    """测试在线学习"""
    learner = OnlineLearner(
        base_model=mock_model,
        batch_size=50
    )

    X = pd.DataFrame(np.random.randn(60, 3))
    y = pd.Series(np.random.randint(0, 2, 60))

    # 分批学习
    learner.partial_fit(X[:20], y[:20])
    learner.partial_fit(X[20:40], y[20:40])
    assert not mock_model.partial_fit.called  # 未达到batch_size

    learner.partial_fit(X[40:], y[40:])
    assert mock_model.partial_fit.called  # 应触发学习

def test_performance_trend():
    """测试表现趋势获取"""
    monitor = ModelMonitor(detectors={})

    # 添加历史记录
    for i in range(5):
        monitor.log_performance(
            model_name='model1',
            performance=ModelPerformance(
                accuracy=0.8 + i*0.05,
                precision=0.7,
                recall=0.75,
                f1=0.72
            ),
            timestamp=pd.Timestamp('2023-01-0{}'.format(i+1))
        )

    trend = monitor.get_performance_trend('model1', 'accuracy')
    assert len(trend) == 5
    assert trend['accuracy'].iloc[-1] == 1.0  # 最后一条记录应为1.0

def test_recent_alerts():
    """测试近期警报获取"""
    monitor = ModelMonitor(detectors={})

    # 添加不同时间的警报
    for i in range(10):
        monitor.drift_alerts.append({
            'model': 'model1',
            'alert': DriftAlert(
                drift_type=DriftType.COVARIATE,
                severity=0.5,
                test_statistic=0.6,
                p_value=0.01,
                baseline_period='P1',
                current_period='P2'
            ),
            'timestamp': pd.Timestamp('2023-01-0{}'.format(i+1))
        })

    # 获取最近3天的警报
    recent = monitor.get_recent_alerts(days=3)
    assert len(recent) == 3
    assert all(alert['timestamp'] >= pd.Timestamp('2023-01-08') for alert in recent)

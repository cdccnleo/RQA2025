import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from models.monitoring import (
    ModelMonitor,
    ModelPerformance,
    DriftDetectionResult,
    DriftType,
    AlertLevel
)

@pytest.fixture
def sample_reference_data():
    """生成测试参考数据"""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.binomial(1, 0.3, 1000)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_current_data():
    """生成测试当前数据"""
    np.random.seed(24)
    data = {
        'feature1': np.random.normal(0.1, 1.1, 1000),
        'feature2': np.random.normal(5.5, 2.2, 1000),
        'feature3': np.random.binomial(1, 0.4, 1000)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_y_true():
    """生成测试真实标签"""
    np.random.seed(42)
    return np.random.binomial(1, 0.5, 1000)

@pytest.fixture
def sample_y_pred():
    """生成测试预测标签"""
    np.random.seed(24)
    return np.random.binomial(1, 0.5, 1000)

@pytest.fixture
def sample_y_prob():
    """生成测试预测概率"""
    np.random.seed(42)
    return np.random.uniform(0, 1, 1000)

def test_model_monitor_performance(sample_y_true, sample_y_pred, sample_y_prob):
    """测试模型性能监控"""
    monitor = ModelMonitor(pd.DataFrame())

    # 测试性能计算
    perf = monitor.check_performance(sample_y_true, sample_y_pred, sample_y_prob)
    assert isinstance(perf, ModelPerformance)
    assert 0 <= perf.accuracy <= 1
    assert 0 <= perf.precision <= 1
    assert 0 <= perf.recall <= 1
    assert 0 <= perf.f1_score <= 1
    assert 0 <= perf.roc_auc <= 1
    assert perf.log_loss >= 0

    # 测试无概率输入
    perf = monitor.check_performance(sample_y_true, sample_y_pred)
    assert perf.roc_auc == 0
    assert perf.log_loss == 0

def test_drift_detection(sample_reference_data, sample_current_data):
    """测试漂移检测"""
    monitor = ModelMonitor(sample_reference_data)

    # 测试漂移检测
    features = ['feature1', 'feature2', 'feature3']
    result = monitor.detect_drift(sample_current_data, features)
    assert isinstance(result, DriftDetectionResult)
    assert result.drift_score > 0
    assert 0 <= result.p_value <= 1
    assert result.timestamp <= datetime.now()

    # 测试无漂移情况
    no_drift_data = sample_reference_data.copy()
    result = monitor.detect_drift(no_drift_data, features)
    assert result.drift_score < 0.1

def test_stability_check(sample_y_true, sample_y_pred):
    """测试稳定性检查"""
    monitor = ModelMonitor(pd.DataFrame())

    # 添加多个性能记录
    for _ in range(50):
        monitor.check_performance(sample_y_true, sample_y_pred)

    # 测试稳定性计算
    stability = monitor.check_stability(window_size=30)
    assert stability
    assert 'accuracy_mean' in stability
    assert 'accuracy_std' in stability
    assert 'accuracy_cv' in stability

    # 测试窗口大小不足
    empty = monitor.check_stability(window_size=100)
    assert not empty

def test_alert_generation(sample_reference_data):
    """测试预警生成"""
    monitor = ModelMonitor(sample_reference_data)

    # 测试临界漂移预警
    critical_drift = DriftDetectionResult(
        drift_type=DriftType.CONCEPT_DRIFT,
        drift_score=0.4,
        p_value=0.001,
        is_drifted=True,
        timestamp=datetime.now()
    )
    perf = ModelPerformance(0.9, 0.8, 0.85, 0.82, 0.95, 0.2)
    alert = monitor.generate_alert(critical_drift, perf)
    assert alert == AlertLevel.CRITICAL

    # 测试性能预警
    warning_drift = DriftDetectionResult(
        drift_type=DriftType.DATA_DRIFT,
        drift_score=0.2,
        p_value=0.01,
        is_drifted=True,
        timestamp=datetime.now()
    )
    bad_perf = ModelPerformance(0.4, 0.3, 0.5, 0.4, 0.6, 1.0)
    alert = monitor.generate_alert(warning_drift, bad_perf)
    assert alert == AlertLevel.WARNING

    # 测试无预警情况
    no_drift = DriftDetectionResult(
        drift_type=None,
        drift_score=0.1,
        p_value=0.5,
        is_drifted=False,
        timestamp=datetime.now()
    )
    good_perf = ModelPerformance(0.9, 0.8, 0.85, 0.82, 0.95, 0.2)
    alert = monitor.generate_alert(no_drift, good_perf)
    assert alert is None

def test_visualizations(sample_reference_data, sample_y_true, sample_y_pred):
    """测试可视化组件"""
    monitor = ModelMonitor(sample_reference_data)

    # 添加测试数据
    for _ in range(20):
        monitor.check_performance(sample_y_true, sample_y_pred)
        monitor.detect_drift(sample_reference_data, ['feature1'])

    # 测试性能趋势图
    fig = monitor.plot_performance_trend()
    assert fig is not None

    # 测试漂移历史图
    fig = monitor.plot_drift_history()
    assert fig is not None

def test_edge_cases():
    """测试边界情况"""
    # 测试空数据
    monitor = ModelMonitor(pd.DataFrame())
    empty_result = monitor.detect_drift(pd.DataFrame(), [])
    assert empty_result.drift_score == 0

    # 测试单样本性能
    perf = ModelMonitor(pd.DataFrame()).check_performance(
        np.array([1]), np.array([1]), np.array([0.9])
    )
    assert perf.accuracy == 1.0

"""
数据质量监控模块测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
from unittest.mock import Mock, patch

from src.data.monitoring.quality_monitor import DataQualityMonitor, QualityMetrics
from src.data.models import DataModel
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def test_data():
    """测试数据fixture"""
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return df


@pytest.fixture
def test_report_dir(tmp_path):
    """测试报告目录fixture"""
    report_dir = tmp_path / "test_reports"
    report_dir.mkdir()
    yield report_dir
    # 清理测试目录
    shutil.rmtree(report_dir)


@pytest.fixture
def quality_monitor(test_report_dir):
    """质量监控器fixture"""
    return DataQualityMonitor(test_report_dir)


@pytest.fixture
def sample_data_model(test_data):
    """样本数据模型fixture"""
    model = DataModel(test_data)
    model.set_metadata({
        'source': 'test',
        'frequency': '1d',
        'symbol': '000001.SZ',
        'created_at': datetime.now().isoformat()
    })
    return model


def test_monitor_init(test_report_dir):
    """测试监控器初始化"""
    monitor = DataQualityMonitor(test_report_dir)

    # 验证目录创建
    assert test_report_dir.exists()
    assert test_report_dir.is_dir()

    # 验证默认配置
    assert monitor.thresholds['completeness'] == 0.95
    assert monitor.alert_config['enabled'] is True


def test_evaluate_quality(quality_monitor, sample_data_model):
    """测试质量评估"""
    # 评估数据质量
    metrics = quality_monitor.evaluate_quality(sample_data_model)

    # 验证指标
    assert isinstance(metrics, QualityMetrics)
    assert 0 <= metrics.completeness <= 1
    assert 0 <= metrics.timeliness <= 1
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.consistency <= 1
    assert 0 <= metrics.total_score <= 1

    # 验证历史记录
    history_file = quality_monitor.report_dir / 'quality_history.json'
    assert history_file.exists()

    with open(history_file, 'r') as f:
        history = json.load(f)
    assert 'test' in history
    assert len(history['test']) > 0


@pytest.mark.parametrize("missing_ratio", [0.0, 0.1, 0.5, 0.9])
def test_completeness_calculation(quality_monitor, sample_data_model, missing_ratio):
    """测试完整性计算"""
    # 创建带有缺失值的数据
    df = sample_data_model.data.copy()
    mask = np.random.random(df.shape) < missing_ratio
    df = df.mask(mask)

    model = DataModel(df)
    model.set_metadata(sample_data_model.get_metadata())

    # 评估质量
    metrics = quality_monitor.evaluate_quality(model)

    # 验证完整性分数
    expected_completeness = 1 - missing_ratio
    # 允许一定的误差，因为计算方法可能有所不同
    assert abs(metrics.completeness - expected_completeness) < 0.2


def test_timeliness_calculation(quality_monitor, sample_data_model):
    """测试时效性计算"""
    # 测试不同的数据创建时间
    for days_ago in [0, 1, 5, 10]:
        # 修改创建时间
        metadata = sample_data_model.get_metadata()
        metadata['created_at'] = (datetime.now() - timedelta(days=days_ago)).isoformat()
        sample_data_model.set_metadata(metadata)

        # 评估质量
        metrics = quality_monitor.evaluate_quality(sample_data_model)

        # 验证时效性分数
        if days_ago == 0:
            assert metrics.timeliness > 0.9  # 当天数据应该有很高的时效性
        elif days_ago > 5:
            assert metrics.timeliness < 0.5  # 较旧的数据应该有较低的时效性


@pytest.mark.parametrize("anomaly_ratio", [0.0, 0.05, 0.2])
def test_accuracy_calculation(quality_monitor, sample_data_model, anomaly_ratio):
    """测试准确性计算"""
    # 创建带有异常值的数据
    df = sample_data_model.data.copy()
    rows = len(df)
    anomaly_count = int(rows * anomaly_ratio)

    if anomaly_count > 0:
        # 添加异常值
        anomaly_indices = np.random.choice(df.index, anomaly_count, replace=False)
        df.loc[anomaly_indices, 'close'] = df['close'].mean() + df['close'].std() * 10  # 明显的异常值

    model = DataModel(df)
    model.set_metadata(sample_data_model.get_metadata())

    # 评估质量
    metrics = quality_monitor.evaluate_quality(model)

    # 验证准确性分数
    if anomaly_ratio == 0:
        assert metrics.accuracy > 0.9  # 无异常值应该有很高的准确性
    elif anomaly_ratio > 0.1:
        assert metrics.accuracy < 0.9  # 较多异常值应该降低准确性


def test_consistency_calculation(quality_monitor):
    """测试一致性计算"""
    # 创建不同一致性的数据

    # 1. 高一致性数据：规律的时间序列
    dates1 = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    df1 = pd.DataFrame({
        'close': np.linspace(100, 110, len(dates1)),  # 线性增长
        'volume': np.ones(len(dates1)) * 5000  # 恒定值
    }, index=dates1)
    model1 = DataModel(df1)
    model1.set_metadata({'source': 'consistent'})

    # 2. 低一致性数据：不规律的时间序列和波动的数据
    dates2 = pd.DatetimeIndex([
        '2023-01-01', '2023-01-02', '2023-01-05', '2023-01-08', '2023-01-10'
    ])
    df2 = pd.DataFrame({
        'close': [100, 105, 90, 120, 95],  # 波动的数据
        'volume': [5000, 6000, 4000, 7000, 3000]  # 波动的数据
    }, index=dates2)
    model2 = DataModel(df2)
    model2.set_metadata({'source': 'inconsistent'})

    # 评估质量
    metrics1 = quality_monitor.evaluate_quality(model1)
    metrics2 = quality_monitor.evaluate_quality(model2)

    # 验证一致性分数
    assert metrics1.consistency > metrics2.consistency


def test_quality_report(quality_monitor, sample_data_model):
    """测试质量报告生成"""
    # 评估多个数据源的质量
    model1 = sample_data_model
    model1.set_metadata({'source': 'source1'})
    quality_monitor.evaluate_quality(model1)

    model2 = DataModel(sample_data_model.data.copy())
    model2.set_metadata({'source': 'source2'})
    quality_monitor.evaluate_quality(model2)

    # 生成报告
    report = quality_monitor.generate_quality_report()

    # 验证报告
    assert 'timestamp' in report
    assert 'sources' in report
    assert 'source1' in report['sources']
    assert 'source2' in report['sources']

    # 验证报告文件
    report_files = list(quality_monitor.report_dir.glob('quality_report_*.json'))
    assert len(report_files) > 0


def test_alerts(quality_monitor):
    """测试告警功能"""
    # 创建低质量数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)) + 100,
    }, index=dates)
    # 添加大量缺失值
    df.loc[df.index[2:8], 'close'] = np.nan

    model = DataModel(df)
    model.set_metadata({'source': 'low_quality'})

    # 设置较高的阈值
    quality_monitor.set_thresholds({
        'completeness': 0.99  # 设置一个很高的完整性阈值
    })

    # 评估质量（应该触发告警）
    quality_monitor.evaluate_quality(model)

    # 获取告警
    alerts = quality_monitor.get_alerts(days=1, source_type='low_quality')

    # 验证告警
    assert len(alerts) > 0
    assert any('completeness' in alert['message'] for alert in alerts)


def test_quality_trend(quality_monitor, sample_data_model):
    """测试质量趋势"""
    # 模拟多天的质量记录
    for i in range(10):
        # 每次稍微修改数据
        df = sample_data_model.data.copy()
        if i > 5:
            # 后半段添加一些缺失值，降低质量
            mask = np.random.random(df.shape) < 0.2
            df = df.mask(mask)

        model = DataModel(df)
        model.set_metadata({
            'source': 'trend_test',
            'created_at': (datetime.now() - timedelta(days=10-i)).isoformat()
        })

        # 评估质量
        quality_monitor.evaluate_quality(model)

    # 获取趋势
    trend = quality_monitor.get_quality_trend('trend_test', 'completeness')

    # 验证趋势数据
    assert 'data' in trend
    assert 'timestamps' in trend['data']
    assert 'values' in trend['data']
    assert len(trend['data']['values']) > 0
    assert 'trend' in trend['statistics']


def test_set_thresholds_and_config(quality_monitor):
    """测试设置阈值和配置"""
    # 设置新的阈值
    new_thresholds = {
        'completeness': 0.9,
        'accuracy': 0.85
    }
    quality_monitor.set_thresholds(new_thresholds)

    # 验证阈值更新
    assert quality_monitor.thresholds['completeness'] == 0.9
    assert quality_monitor.thresholds['accuracy'] == 0.85

    # 设置新的告警配置
    new_config = {
        'enabled': False,
        'min_score_alert': 0.7
    }
    quality_monitor.set_alert_config(new_config)

    # 验证配置更新
    assert quality_monitor.alert_config['enabled'] is False
    assert quality_monitor.alert_config['min_score_alert'] == 0.7


@patch('json.dump')
def test_save_failure(mock_json_dump, quality_monitor, sample_data_model):
    """测试保存失败情况"""
    # 模拟保存失败
    mock_json_dump.side_effect = Exception("Failed to save")

    # 评估质量（不应该抛出异常）
    metrics = quality_monitor.evaluate_quality(sample_data_model)

    # 验证指标仍然计算正确
    assert isinstance(metrics, QualityMetrics)
    assert 0 <= metrics.completeness <= 1


def test_empty_data(quality_monitor):
    """测试空数据"""
    # 创建空数据
    empty_df = pd.DataFrame()
    model = DataModel(empty_df)
    model.set_metadata({'source': 'empty'})

    # 评估质量
    metrics = quality_monitor.evaluate_quality(model)

    # 验证指标
    assert metrics.completeness == 0.0
    assert metrics.accuracy == 0.0


def test_quality_summary(quality_monitor, sample_data_model):
    """测试质量摘要"""
    # 评估多个数据源的质量
    sources = ['summary1', 'summary2', 'summary3']
    for source in sources:
        model = DataModel(sample_data_model.data.copy())
        model.set_metadata({'source': source})
        quality_monitor.evaluate_quality(model)

    # 获取摘要
    summary = quality_monitor.get_quality_summary()

    # 验证摘要
    assert 'timestamp' in summary
    assert 'sources' in summary
    assert 'overall' in summary
    assert summary['overall']['total_sources'] >= len(sources)
    for source in sources:
        assert source in summary['sources']


@pytest.mark.parametrize("data_type", ['numeric', 'mixed', 'categorical'])
def test_different_data_types(quality_monitor, data_type):
    """测试不同数据类型"""
    # 创建不同类型的数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')

    if data_type == 'numeric':
        df = pd.DataFrame({
            'col1': np.random.randn(len(dates)) + 100,
            'col2': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    elif data_type == 'mixed':
        df = pd.DataFrame({
            'col1': np.random.randn(len(dates)) + 100,
            'col2': ['A', 'B', 'C'] * (len(dates) // 3 + 1),
            'col3': np.random.randint(0, 2, len(dates))
        }, index=dates)
    else:  # categorical
        df = pd.DataFrame({
            'col1': ['A', 'B', 'C'] * (len(dates) // 3 + 1),
            'col2': ['X', 'Y', 'Z'] * (len(dates) // 3 + 1)
        }, index=dates)

    model = DataModel(df)
    model.set_metadata({'source': f'{data_type}_test'})

    # 评估质量
    metrics = quality_monitor.evaluate_quality(model)

    # 验证指标
    assert isinstance(metrics, QualityMetrics)
    assert 0 <= metrics.completeness <= 1
    assert 0 <= metrics.accuracy <= 1
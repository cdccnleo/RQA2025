"""
监控示例

展示如何使用模型性能监控、告警和自动回滚功能
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import numpy as np

from ..monitoring.performance_monitor import (
    ModelPerformanceMonitor,
    TechnicalMetricsCollector,
    BusinessMetricsCollector,
    ResourceMetricsCollector
)
from ..monitoring.alert_manager import AlertManager, AlertRule, AlertSeverity, create_default_alert_rules
from ..monitoring.drift_detector import DriftDetector, KSDriftDetector, PSI_DriftDetector, ConceptDriftDetector
from ..monitoring.rollback_manager import RollbackManager, RollbackTrigger


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_mock_model():
    """创建模拟模型"""
    from sklearn.ensemble import RandomForestClassifier
    
    # 创建并训练一个简单模型
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


def create_mock_data_source():
    """创建模拟数据源"""
    def data_source():
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1min')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'feature_0': np.random.randn(1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000),
            'feature_4': np.random.randn(1000),
            'feature_5': np.random.randn(1000),
            'feature_6': np.random.randn(1000),
            'feature_7': np.random.randn(1000),
            'feature_8': np.random.randn(1000),
            'feature_9': np.random.randn(1000),
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 100000, 1000)
        })
        
        # 创建目标变量
        data['target'] = (data['feature_0'] + data['feature_1'] > 0).astype(int)
        
        return data
    
    return data_source


def create_mock_signal_source():
    """创建模拟信号源"""
    def signal_source():
        data = create_mock_data_source()()
        data['returns'] = np.random.randn(len(data)) * 0.01
        data['signal'] = np.random.choice([-1, 0, 1], len(data))
        return data
    
    return signal_source


def example_performance_monitoring():
    """
    性能监控示例
    
    展示如何设置和使用性能监控器
    """
    setup_logging()
    logger = logging.getLogger("monitoring_example")
    
    logger.info("=" * 60)
    logger.info("性能监控示例")
    logger.info("=" * 60)
    
    # 1. 创建模拟模型和数据源
    model = create_mock_model()
    data_source = create_mock_data_source()
    signal_source = create_mock_signal_source()
    
    # 2. 创建性能监控器
    monitor = ModelPerformanceMonitor(
        model_id="example_model",
        monitoring_interval=10  # 10秒间隔
    )
    
    # 3. 注册指标收集器
    monitor.register_collector(TechnicalMetricsCollector(model, data_source))
    monitor.register_collector(BusinessMetricsCollector(signal_source))
    monitor.register_collector(ResourceMetricsCollector(model, data_source()))
    
    # 4. 启动监控
    logger.info("启动性能监控...")
    monitor.start_monitoring()
    
    # 5. 模拟运行一段时间
    logger.info("监控运行中（模拟30秒）...")
    time.sleep(5)
    
    # 6. 手动收集指标
    logger.info("\n手动收集指标...")
    snapshot = monitor.collect_metrics()
    
    logger.info(f"收集到 {len(snapshot.metrics)} 个指标:")
    for metric_name, metric in snapshot.metrics.items():
        logger.info(f"  - {metric_name}: {metric.value:.4f} [{metric.status}]")
    
    # 7. 获取统计信息
    stats = monitor.get_statistics()
    logger.info(f"\n监控统计:")
    logger.info(f"  - 快照数: {stats.get('total_snapshots', 0)}")
    logger.info(f"  - 监控时长: {stats.get('monitoring_duration', 0):.2f}秒")
    
    # 8. 停止监控
    monitor.stop_monitoring()
    logger.info("\n监控已停止")
    
    return monitor


def example_alert_management():
    """
    告警管理示例
    
    展示如何设置告警规则和接收告警
    """
    setup_logging()
    logger = logging.getLogger("alert_example")
    
    logger.info("\n" + "=" * 60)
    logger.info("告警管理示例")
    logger.info("=" * 60)
    
    # 1. 创建告警管理器
    alert_manager = AlertManager()
    
    # 2. 注册默认告警规则
    logger.info("注册默认告警规则...")
    for rule in create_default_alert_rules():
        alert_manager.register_rule(rule)
        logger.info(f"  - {rule.name}: {rule.metric_name} {rule.operator} {rule.threshold}")
    
    # 3. 设置基线
    alert_manager.set_baseline("accuracy", 0.75)
    
    # 4. 模拟指标评估
    logger.info("\n模拟指标评估...")
    
    # 正常指标
    normal_metrics = {
        "accuracy": 0.78,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "error_rate": 0.02,
        "p95_latency_ms": 150
    }
    
    alerts = alert_manager.evaluate_metrics(normal_metrics)
    logger.info(f"正常指标触发 {len(alerts)} 个告警")
    
    # 异常指标
    abnormal_metrics = {
        "accuracy": 0.60,  # 低于阈值
        "sharpe_ratio": 0.3,  # 低于阈值
        "max_drawdown": 0.20,  # 超过阈值
        "error_rate": 0.08,  # 超过阈值
        "p95_latency_ms": 250  # 超过阈值
    }
    
    alerts = alert_manager.evaluate_metrics(abnormal_metrics)
    logger.info(f"异常指标触发 {len(alerts)} 个告警:")
    
    for alert in alerts:
        logger.warning(f"  - [{alert.severity.value.upper()}] {alert.title}: {alert.message}")
        
        # 确认告警
        alert_manager.acknowledge_alert(alert.alert_id, "admin")
        logger.info(f"    已确认 by admin")
    
    # 5. 获取活跃告警
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"\n当前活跃告警: {len(active_alerts)}")
    
    # 6. 解决告警
    for alert in active_alerts:
        alert_manager.resolve_alert(alert.alert_id)
        logger.info(f"已解决告警: {alert.alert_id}")
    
    # 7. 获取统计
    stats = alert_manager.get_statistics()
    logger.info(f"\n告警统计:")
    logger.info(f"  - 总规则数: {stats['total_rules']}")
    logger.info(f"  - 活跃告警: {stats['active_alerts']}")
    logger.info(f"  - 历史告警: {stats['total_alerts_history']}")
    
    return alert_manager


def example_drift_detection():
    """
    漂移检测示例
    
    展示如何检测数据漂移和概念漂移
    """
    setup_logging()
    logger = logging.getLogger("drift_example")
    
    logger.info("\n" + "=" * 60)
    logger.info("漂移检测示例")
    logger.info("=" * 60)
    
    # 1. 创建参考数据
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.exponential(1, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # 2. 创建漂移检测器
    drift_detector = DriftDetector()
    drift_detector.set_reference_data(reference_data)
    
    # 3. 注册检测器
    drift_detector.register_detector(KSDriftDetector(threshold=0.05))
    drift_detector.register_detector(PSI_DriftDetector(threshold=0.25))
    
    # 4. 测试无漂移数据
    logger.info("测试无漂移数据...")
    current_data_normal = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.exponential(1, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    reports = drift_detector.detect(current_data_normal)
    for report in reports:
        logger.info(f"  - {report.drift_type.value}: {report.severity.value} (分数: {report.drift_score:.4f})")
    
    # 5. 测试有漂移数据
    logger.info("\n测试有漂移数据...")
    current_data_drifted = pd.DataFrame({
        'feature_1': np.random.normal(2, 1.5, 1000),  # 均值漂移
        'feature_2': np.random.normal(8, 3, 1000),    # 均值和标准差漂移
        'feature_3': np.random.exponential(2, 1000),  # 参数漂移
        'target': np.random.choice([0, 1], 1000)
    })
    
    reports = drift_detector.detect(current_data_drifted)
    for report in reports:
        logger.warning(f"  - {report.drift_type.value}: {report.severity.value} (分数: {report.drift_score:.4f})")
        if report.affected_features:
            logger.warning(f"    受影响特征: {', '.join(report.affected_features[:3])}")
        if report.recommendations:
            logger.info(f"    建议: {report.recommendations[0]}")
    
    # 6. 获取汇总
    summary = drift_detector.get_drift_summary()
    logger.info(f"\n漂移汇总: {summary}")
    
    # 7. 判断是否触发重新训练
    should_retrain = drift_detector.should_trigger_retraining()
    logger.info(f"建议重新训练: {should_retrain}")
    
    return drift_detector


def example_rollback():
    """
    自动回滚示例
    
    展示如何评估和执行模型回滚
    """
    setup_logging()
    logger = logging.getLogger("rollback_example")
    
    logger.info("\n" + "=" * 60)
    logger.info("自动回滚示例")
    logger.info("=" * 60)
    
    # 1. 创建回滚管理器
    rollback_manager = RollbackManager(
        model_id="example_model",
        model_path="models/model_v2.joblib",
        backup_path="models/model_v1.joblib"
    )
    
    # 2. 设置基线指标（上一版本的性能）
    rollback_manager.set_baseline_metrics({
        "accuracy": 0.75,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.10,
        "error_rate": 0.02
    })
    
    # 3. 设置阈值
    rollback_manager.set_thresholds({
        "accuracy_drop": 0.10,
        "max_drawdown": 0.15,
        "error_rate": 0.05
    })
    
    # 4. 模拟性能下降场景
    logger.info("模拟性能下降场景...")
    
    # 创建模拟监控器
    from unittest.mock import Mock, MagicMock
    
    mock_monitor = Mock()
    mock_snapshot = Mock()
    mock_snapshot.metrics = {
        "accuracy": Mock(value=0.60, to_dict=lambda: {"value": 0.60, "status": "warning"}),
        "max_drawdown": Mock(value=0.20, to_dict=lambda: {"value": 0.20, "status": "critical"}),
        "error_rate": Mock(value=0.08, to_dict=lambda: {"value": 0.08, "status": "warning"})
    }
    mock_monitor.get_latest_metrics.return_value = mock_snapshot
    
    rollback_manager._performance_monitor = mock_monitor
    
    # 5. 评估回滚需求
    logger.info("评估回滚需求...")
    decision = rollback_manager.evaluate_rollback_need()
    
    logger.info(f"回滚决策:")
    logger.info(f"  - 建议回滚: {decision.should_rollback}")
    logger.info(f"  - 触发条件: {decision.trigger.value if decision.trigger else 'N/A'}")
    logger.info(f"  - 置信度: {decision.confidence:.2f}")
    logger.info(f"  - 原因: {decision.reasons}")
    logger.info(f"  - 建议操作: {decision.recommended_action}")
    
    # 6. 获取状态
    status = rollback_manager.get_status()
    logger.info(f"\n回滚管理器状态:")
    logger.info(f"  - 模型ID: {status['model_id']}")
    logger.info(f"  - 阈值: {status['thresholds']}")
    logger.info(f"  - 回滚历史: {status['rollback_count']} 次")
    
    return rollback_manager


def run_monitoring_example():
    """
    运行所有监控示例
    """
    print("\n" + "=" * 60)
    print("模型性能监控和自动回滚示例")
    print("=" * 60)
    
    # 1. 性能监控示例
    monitor = example_performance_monitoring()
    
    # 2. 告警管理示例
    alert_manager = example_alert_management()
    
    # 3. 漂移检测示例
    drift_detector = example_drift_detection()
    
    # 4. 自动回滚示例
    rollback_manager = example_rollback()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成")
    print("=" * 60)
    
    return {
        "monitor": monitor,
        "alert_manager": alert_manager,
        "drift_detector": drift_detector,
        "rollback_manager": rollback_manager
    }


if __name__ == "__main__":
    run_monitoring_example()

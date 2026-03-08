"""
完整管道示例

展示如何使用8阶段自动化训练管道
"""

import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import numpy as np

from ..core.pipeline_controller import MLPipelineController, PipelineExecutionResult
from ..core.pipeline_config import create_default_config, PipelineConfig, StageConfig
from ..stages.data_preparation import DataPreparationStage
from ..stages.feature_engineering import FeatureEngineeringStage
from ..stages.model_training import ModelTrainingStage
from ..stages.model_evaluation import ModelEvaluationStage
from ..stages.model_validation import ModelValidationStage
from ..stages.canary_deployment import CanaryDeploymentStage
from ..stages.full_deployment import FullDeploymentStage
from ..stages.monitoring import MonitoringStage
from ..notification.notification_service import NotificationService
from ..notification.email_channel import EmailChannel
from ..notification.webhook_channel import WebhookChannel


def setup_logging() -> None:
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_full_pipeline_config() -> PipelineConfig:
    """
    创建完整管道配置
    
    Returns:
        管道配置
    """
    return PipelineConfig(
        name="quant_trading_ml_pipeline",
        version="1.0.0",
        stages=[
            StageConfig(
                name="data_preparation",
                enabled=True,
                config={
                    "data_sources": ["market_data"],
                    "date_range": "last_90_days",
                    "quality_checks": True,
                    "max_missing_threshold": 10.0,
                    "handle_outliers": True
                }
            ),
            StageConfig(
                name="feature_engineering",
                enabled=True,
                config={
                    "feature_selection": "variance",
                    "standardization": "zscore",
                    "store_features": True
                },
                dependencies=["data_preparation"]
            ),
            StageConfig(
                name="model_training",
                enabled=True,
                config={
                    "model_type": "xgboost",
                    "target_col": "target",
                    "hyperparameter_search": True
                },
                dependencies=["feature_engineering"]
            ),
            StageConfig(
                name="model_evaluation",
                enabled=True,
                config={
                    "metrics": ["accuracy", "f1", "roc_auc", "sharpe_ratio", "max_drawdown"],
                    "backtest": True,
                    "min_accuracy": 0.55,
                    "min_sharpe": 0.5,
                    "max_drawdown": 0.2
                },
                dependencies=["model_training"]
            ),
            StageConfig(
                name="model_validation",
                enabled=True,
                config={
                    "ab_test": True,
                    "shadow_mode": True,
                    "validate_business_rules": True,
                    "ab_test_days": 7
                },
                dependencies=["model_evaluation"]
            ),
            StageConfig(
                name="canary_deployment",
                enabled=True,
                config={
                    "traffic_percentage": 5,
                    "duration_minutes": 30,
                    "max_error_rate": 0.05,
                    "max_latency_ms": 200,
                    "min_accuracy": 0.55
                },
                dependencies=["model_validation"]
            ),
            StageConfig(
                name="full_deployment",
                enabled=True,
                config={
                    "strategy": "blue_green"
                },
                dependencies=["canary_deployment"]
            ),
            StageConfig(
                name="monitoring",
                enabled=True,
                config={
                    "metrics_interval_seconds": 60,
                    "drift_detection": True,
                    "monitoring_duration_minutes": 5
                },
                dependencies=["full_deployment"]
            )
        ],
        rollback={
            "enabled": True,
            "strategy": "immediate",
            "auto_rollback": True,
            "triggers": [
                {
                    "metric": "accuracy",
                    "threshold": 0.1,
                    "operator": "decrease",
                    "duration_minutes": 5
                },
                {
                    "metric": "max_drawdown",
                    "threshold": 0.15,
                    "operator": "greater_than",
                    "duration_minutes": 1
                },
                {
                    "metric": "drift_score",
                    "threshold": 0.5,
                    "operator": "greater_than",
                    "duration_minutes": 10
                }
            ]
        },
        monitoring={
            "enabled": True,
            "metrics_interval_seconds": 60,
            "drift_detection": True,
            "alert_thresholds": {
                "accuracy": 0.7,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.15,
                "error_rate": 0.05,
                "latency_p95": 200
            }
        }
    )


def setup_notification_service() -> NotificationService:
    """
    设置通知服务
    
    Returns:
        通知服务实例
    """
    service = NotificationService()
    
    # 注册日志通道（始终启用）
    from ..notification.log_channel import LogChannel
    service.register_channel(LogChannel(), is_default=True)
    
    # 注册邮件通道（可选）
    # service.register_channel(EmailChannel(
    #     smtp_host="smtp.example.com",
    #     smtp_port=587,
    #     username="user@example.com",
    #     password="password"
    # ))
    
    # 注册Webhook通道（可选）
    # service.register_channel(WebhookChannel(
    #     webhook_url="https://hooks.example.com/notify"
    # ))
    
    return service


def run_full_pipeline() -> PipelineExecutionResult:
    """
    运行完整8阶段管道
    
    Returns:
        管道执行结果
    """
    setup_logging()
    logger = logging.getLogger("pipeline_example")
    
    logger.info("=" * 60)
    logger.info("启动完整ML训练管道")
    logger.info("=" * 60)
    
    # 1. 创建配置
    config = create_full_pipeline_config()
    logger.info(f"管道名称: {config.name}")
    logger.info(f"管道版本: {config.version}")
    logger.info(f"阶段数量: {len(config.stages)}")
    
    # 2. 创建控制器
    controller = MLPipelineController(config)
    
    # 3. 注册所有阶段
    stages = [
        DataPreparationStage(),
        FeatureEngineeringStage(),
        ModelTrainingStage(),
        ModelEvaluationStage(),
        ModelValidationStage(),
        CanaryDeploymentStage(),
        FullDeploymentStage(),
        MonitoringStage()
    ]
    
    for stage in stages:
        controller.register_stage(stage)
        logger.info(f"注册阶段: {stage.name}")
    
    # 4. 设置通知服务
    notification_service = setup_notification_service()
    
    # 5. 准备初始上下文
    initial_context = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 12, 31),
        "model_dir": "models",
        "alert_thresholds": {
            "error_rate": 0.05,
            "latency_p95": 200,
            "accuracy": 0.55,
            "drift_score": 0.5
        }
    }
    
    # 6. 执行管道
    logger.info("\n开始执行管道...")
    result = controller.execute(
        initial_context=initial_context,
        pipeline_id=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 7. 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("管道执行完成")
    logger.info("=" * 60)
    logger.info(f"执行状态: {result.status.name}")
    logger.info(f"是否成功: {result.is_success}")
    logger.info(f"执行时长: {result.duration_seconds:.2f}秒" if result.duration_seconds else "N/A")
    
    if result.error:
        logger.error(f"错误信息: {result.error}")
    
    if result.summary:
        logger.info(f"\n执行摘要:")
        logger.info(f"  - 完成阶段数: {result.summary.get('stages_completed', 0)}/{result.summary.get('stages_total', 0)}")
        logger.info(f"  - 总耗时: {result.summary.get('total_duration_seconds', 0):.2f}秒")
        
        stage_summaries = result.summary.get('stage_summaries', {})
        logger.info(f"\n各阶段详情:")
        for stage_name, stage_info in stage_summaries.items():
            logger.info(f"  - {stage_name}: {stage_info.get('status')} ({stage_info.get('duration_seconds', 0):.2f}秒)")
    
    # 8. 发送通知
    notification_service.send(
        message=f"管道执行完成 - 状态: {result.status.name}",
        level="INFO" if result.is_success else "ERROR"
    )
    
    return result


def run_pipeline_with_monitoring() -> None:
    """
    运行带监控的管道示例
    
    展示如何集成性能监控和自动回滚
    """
    from ..monitoring.performance_monitor import (
        ModelPerformanceMonitor,
        TechnicalMetricsCollector,
        BusinessMetricsCollector,
        ResourceMetricsCollector
    )
    from ..monitoring.alert_manager import AlertManager, create_default_alert_rules
    from ..monitoring.drift_detector import DriftDetector, KSDriftDetector, PSI_DriftDetector
    from ..monitoring.rollback_manager import RollbackManager
    
    setup_logging()
    logger = logging.getLogger("pipeline_with_monitoring")
    
    logger.info("=" * 60)
    logger.info("启动带监控的ML训练管道")
    logger.info("=" * 60)
    
    # 1. 运行管道
    result = run_full_pipeline()
    
    if not result.is_success:
        logger.error("管道执行失败，跳过监控设置")
        return
    
    # 2. 获取部署的模型信息
    state = result.state
    deployment_info = state.context.get('deployment_info', {})
    model_path = deployment_info.get('model_path', 'models/model_latest.joblib')
    
    logger.info(f"\n设置模型监控: {model_path}")
    
    # 3. 创建性能监控器
    monitor = ModelPerformanceMonitor(
        model_id="quant_model_v1",
        monitoring_interval=60
    )
    
    # 4. 加载模型
    import joblib
    try:
        model = joblib.load(model_path)
    except:
        logger.warning("无法加载模型，使用模拟数据")
        model = None
    
    # 5. 创建模拟数据源
    def mock_data_source() -> pd.DataFrame:
        """模拟数据源"""
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1min')
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 100000, 1000)
        })
    
    def mock_signal_source() -> pd.DataFrame:
        """模拟信号源"""
        data = mock_data_source()
        data['returns'] = np.random.randn(len(data)) * 0.01
        return data
    
    # 6. 注册指标收集器
    if model:
        monitor.register_collector(
            TechnicalMetricsCollector(model, mock_data_source)
        )
        monitor.register_collector(
            ResourceMetricsCollector(model, mock_data_source())
        )
    
    monitor.register_collector(
        BusinessMetricsCollector(mock_signal_source)
    )
    
    # 7. 创建告警管理器
    alert_manager = AlertManager()
    for rule in create_default_alert_rules():
        alert_manager.register_rule(rule)
    
    # 8. 创建漂移检测器
    drift_detector = DriftDetector()
    drift_detector.register_detector(KSDriftDetector())
    drift_detector.register_detector(PSI_DriftDetector())
    drift_detector.set_reference_data(mock_data_source())
    
    # 9. 创建回滚管理器
    rollback_manager = RollbackManager(
        model_id="quant_model_v1",
        model_path=model_path,
        backup_path=f"{model_path}.backup",
        performance_monitor=monitor,
        alert_manager=alert_manager,
        drift_detector=drift_detector
    )
    
    # 10. 启动监控
    logger.info("\n启动性能监控...")
    monitor.start_monitoring()
    
    # 11. 模拟监控周期
    import time
    logger.info("监控运行中（模拟30秒）...")
    time.sleep(5)
    
    # 12. 收集指标
    snapshot = monitor.collect_metrics()
    logger.info(f"\n收集到 {len(snapshot.metrics)} 个指标")
    
    for metric_name, metric in snapshot.metrics.items():
        logger.info(f"  - {metric_name}: {metric.value:.4f} ({metric.status})")
    
    # 13. 评估告警
    metrics_dict = {name: metric.value for name, metric in snapshot.metrics.items()}
    alerts = alert_manager.evaluate_metrics(metrics_dict)
    
    if alerts:
        logger.warning(f"\n触发 {len(alerts)} 个告警:")
        for alert in alerts:
            logger.warning(f"  - {alert.title}: {alert.message}")
    else:
        logger.info("\n未触发告警")
    
    # 14. 评估回滚需求
    decision = rollback_manager.evaluate_rollback_need()
    logger.info(f"\n回滚评估:")
    logger.info(f"  - 建议回滚: {decision.should_rollback}")
    logger.info(f"  - 置信度: {decision.confidence:.2f}")
    logger.info(f"  - 原因: {decision.reasons}")
    
    # 15. 停止监控
    monitor.stop_monitoring()
    logger.info("\n监控已停止")
    
    # 16. 输出统计
    stats = monitor.get_statistics()
    logger.info(f"\n监控统计:")
    logger.info(f"  - 总快照数: {stats.get('total_snapshots', 0)}")
    logger.info(f"  - 监控时长: {stats.get('monitoring_duration', 0):.2f}秒")


if __name__ == "__main__":
    # 运行完整管道
    result = run_full_pipeline()
    
    # 如果成功，运行监控示例
    if result.is_success:
        print("\n" + "=" * 60)
        print("运行监控示例...")
        print("=" * 60)
        run_pipeline_with_monitoring()

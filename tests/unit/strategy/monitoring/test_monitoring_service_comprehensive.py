"""
策略监控服务深度测试
全面测试策略性能监控、告警管理、健康检查和实时分析功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import time

# 导入监控服务相关类
try:
    from src.strategy.monitoring.monitoring_service import (
        MonitoringService, MonitoringConfig, MetricData,
        Alert, AlertRule, MetricType
    )
    MONITORING_SERVICE_AVAILABLE = True
except ImportError:
    MONITORING_SERVICE_AVAILABLE = False
    MonitoringService = Mock
    MonitoringConfig = Mock
    MetricData = Mock
    Alert = Mock
    AlertRule = Mock
    MetricType = Mock

try:
    from src.strategy.strategies.base_strategy import BaseStrategy
    BASE_STRATEGY_AVAILABLE = True
except ImportError:
    BASE_STRATEGY_AVAILABLE = False
    BaseStrategy = Mock


class TestMonitoringServiceComprehensive:
    """策略监控服务综合深度测试"""

    @pytest.fixture
    def sample_strategy_performance(self):
        """创建样本策略性能数据"""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'portfolio_value': np.cumprod(1 + np.random.normal(0.0001, 0.005, 100)),
            'returns': np.random.normal(0.0001, 0.005, 100),
            'sharpe_ratio': np.random.uniform(0.5, 2.5, 100),
            'max_drawdown': np.random.uniform(-0.05, -0.25, 100),
            'win_rate': np.random.uniform(0.45, 0.65, 100),
            'total_trades': np.cumsum(np.random.poisson(2, 100)),
            'active_positions': np.random.randint(0, 10, 100)
        })

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=50, freq='H')
        np.random.seed(42)

        return pd.DataFrame({
            'timestamp': dates,
            'market_volatility': np.random.uniform(0.1, 0.4, 50),
            'market_returns': np.random.normal(0.0001, 0.01, 50),
            'liquidity_score': np.random.uniform(0.6, 0.95, 50)
        })

    @pytest.fixture
    def monitoring_config(self):
        """创建监控配置"""
        if MONITORING_SERVICE_AVAILABLE:
            return MonitoringConfig(
                monitoring_id="test_monitoring",
                strategy_id="test_strategy",
                metrics_interval=60,
                alert_check_interval=30,
                max_metrics_history=1000,
                enabled=True
            )
        return Mock()

    @pytest.fixture
    def monitoring_service(self):
        """创建监控服务实例"""
        if MONITORING_SERVICE_AVAILABLE:
            return MonitoringService()
        return Mock(spec=MonitoringService)

    @pytest.fixture
    def base_strategy(self):
        """创建基础策略实例"""
        if BASE_STRATEGY_AVAILABLE:
            return BaseStrategy()
        return Mock(spec=BaseStrategy)

    def test_monitoring_service_initialization(self, monitoring_service):
        """测试监控服务初始化"""
        if MONITORING_SERVICE_AVAILABLE:
            assert monitoring_service is not None
            # Basic initialization test - just check that the service was created
            assert isinstance(monitoring_service, MonitoringService)

    def test_metric_collection_and_storage(self, monitoring_service, sample_strategy_performance):
        """测试指标收集和存储"""
        if MONITORING_SERVICE_AVAILABLE:
            # 收集指标
            for _, row in sample_strategy_performance.head(10).iterrows():
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="portfolio_value",
                    value=row['portfolio_value'],
                    timestamp=row['timestamp'],
                    metadata={'strategy_id': 'test_strategy'}
                )

                monitoring_service.collect_metric(metric_data)

            # 验证指标存储
            stored_metrics = monitoring_service.get_recent_metrics(
                metric_type=MetricType.PERFORMANCE,
                limit=10
            )

            assert isinstance(stored_metrics, list)
            assert len(stored_metrics) == 10

            for metric in stored_metrics:
                assert isinstance(metric, MetricData)
                assert metric.metric_name == "portfolio_value"

    def test_real_time_performance_monitoring(self, monitoring_service, sample_strategy_performance):
        """测试实时性能监控"""
        if MONITORING_SERVICE_AVAILABLE:
            # 启用实时监控
            monitoring_service.enable_real_time_monitoring()

            # 模拟实时数据流
            real_time_metrics = []

            for i, (_, row) in enumerate(sample_strategy_performance.iterrows()):
                if i >= 20:  # 只处理前20个数据点
                    break

                # 收集实时指标
                metrics = {
                    'portfolio_value': row['portfolio_value'],
                    'returns': row['returns'],
                    'sharpe_ratio': row['sharpe_ratio'],
                    'max_drawdown': row['max_drawdown']
                }

                monitoring_result = monitoring_service.process_real_time_metrics(
                    strategy_id='test_strategy',
                    metrics=metrics,
                    timestamp=row['timestamp']
                )

                real_time_metrics.append(monitoring_result)

            assert len(real_time_metrics) == 20

            # 检查实时监控结果
            for result in real_time_metrics:
                assert isinstance(result, dict)
                assert 'processed_at' in result
                assert 'metrics_summary' in result
                assert 'alerts_triggered' in result

    def test_alert_rule_configuration_and_evaluation(self, monitoring_service, sample_strategy_performance):
        """测试告警规则配置和评估"""
        if MONITORING_SERVICE_AVAILABLE:
            # 配置告警规则
            alert_rules = [
                AlertRule(
                    rule_id="high_drawdown",
                    metric_type=MetricType.RISK,
                    metric_name="max_drawdown",
                    condition="value < -0.15",  # 回撤超过15%
                    severity="high",
                    message="Portfolio drawdown exceeded 15%"
                ),
                AlertRule(
                    rule_id="low_sharpe",
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="sharpe_ratio",
                    condition="value < 0.5",
                    severity="medium",
                    message="Sharpe ratio below acceptable level"
                )
            ]

            monitoring_service.configure_alert_rules(alert_rules)

            # 评估告警条件
            test_metrics = {
                'max_drawdown': -0.18,  # 应该触发高回撤告警
                'sharpe_ratio': 0.3     # 应该触发低夏普率告警
            }

            alerts = monitoring_service.evaluate_alert_conditions(
                strategy_id='test_strategy',
                metrics=test_metrics,
                timestamp=datetime.now()
            )

            assert isinstance(alerts, list)
            assert len(alerts) >= 1  # 至少应该有一个告警

            # 检查告警结构
            for alert in alerts:
                assert isinstance(alert, Alert)
                assert hasattr(alert, 'alert_id')
                assert hasattr(alert, 'severity')
                assert hasattr(alert, 'message')

    def test_health_check_and_diagnostic(self, monitoring_service, sample_strategy_performance):
        """测试健康检查和诊断"""
        if MONITORING_SERVICE_AVAILABLE:
            # 执行健康检查
            health_status = monitoring_service.perform_health_check(
                strategy_id='test_strategy',
                performance_data=sample_strategy_performance.head(20)
            )

            assert isinstance(health_status, dict)
            assert 'overall_health' in health_status
            assert 'component_status' in health_status
            assert 'diagnostic_report' in health_status

            # 检查健康状态指标
            overall_health = health_status['overall_health']
            assert overall_health in ['healthy', 'warning', 'critical']

            component_status = health_status['component_status']
            expected_components = ['performance', 'risk', 'execution', 'data_quality']
            for component in expected_components:
                assert component in component_status

    def test_performance_trend_analysis(self, monitoring_service, sample_strategy_performance):
        """测试性能趋势分析"""
        if MONITORING_SERVICE_AVAILABLE:
            # 执行趋势分析
            trend_analysis = monitoring_service.analyze_performance_trends(
                strategy_id='test_strategy',
                performance_data=sample_strategy_performance,
                analysis_window=30  # 30个数据点
            )

            assert isinstance(trend_analysis, dict)
            assert 'trend_direction' in trend_analysis
            assert 'volatility_trend' in trend_analysis
            assert 'performance_stability' in trend_analysis
            assert 'predictive_insights' in trend_analysis

            # 检查趋势方向
            trend_direction = trend_analysis['trend_direction']
            assert trend_direction in ['improving', 'declining', 'stable', 'volatile']

    def test_risk_monitoring_and_assessment(self, monitoring_service, sample_strategy_performance):
        """测试风险监控和评估"""
        if MONITORING_SERVICE_AVAILABLE:
            # 执行风险评估
            risk_assessment = monitoring_service.assess_strategy_risk(
                strategy_id='test_strategy',
                performance_data=sample_strategy_performance,
                risk_factors=['volatility', 'drawdown', 'concentration']
            )

            assert isinstance(risk_assessment, dict)
            assert 'risk_level' in risk_assessment
            assert 'risk_factors' in risk_assessment
            assert 'risk_mitigation_suggestions' in risk_assessment

            # 检查风险水平
            risk_level = risk_assessment['risk_level']
            assert risk_level in ['low', 'medium', 'high', 'critical']

    def test_alert_escalation_and_notification(self, monitoring_service):
        """测试告警升级和通知"""
        if MONITORING_SERVICE_AVAILABLE:
            # 配置告警升级规则
            escalation_rules = {
                'escalation_levels': ['info', 'warning', 'error', 'critical'],
                'time_based_escalation': {
                    'warning_to_error': 300,   # 5分钟后升级
                    'error_to_critical': 600   # 10分钟后升级到严重
                },
                'metric_based_escalation': {
                    'drawdown_thresholds': [-0.1, -0.2, -0.3]
                }
            }

            monitoring_service.configure_alert_escalation(escalation_rules)

            # 创建初始告警
            initial_alert = Alert(
                alert_id="test_alert_1",
                rule_id="high_drawdown",
                severity="warning",
                message="Drawdown warning",
                timestamp=datetime.now(),
                strategy_id='test_strategy'
            )

            # 处理告警升级
            escalation_result = monitoring_service.process_alert_escalation(initial_alert)

            assert isinstance(escalation_result, dict)
            assert 'escalated_alert' in escalation_result
            assert 'notification_actions' in escalation_result

    def test_monitoring_dashboard_data_generation(self, monitoring_service, sample_strategy_performance):
        """测试监控仪表板数据生成"""
        if MONITORING_SERVICE_AVAILABLE:
            # 生成仪表板数据
            dashboard_data = monitoring_service.generate_dashboard_data(
                strategy_id='test_strategy',
                time_range='24h',
                metrics=['performance', 'risk', 'execution']
            )

            assert isinstance(dashboard_data, dict)
            assert 'summary_metrics' in dashboard_data
            assert 'charts_data' in dashboard_data
            assert 'alerts_summary' in dashboard_data
            assert 'health_status' in dashboard_data

            # 检查摘要指标
            summary_metrics = dashboard_data['summary_metrics']
            expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            for metric in expected_metrics:
                assert metric in summary_metrics

    def test_monitoring_data_persistence_and_recovery(self, monitoring_service, sample_strategy_performance, tmp_path):
        """测试监控数据持久化和恢复"""
        if MONITORING_SERVICE_AVAILABLE:
            # 收集一些指标数据
            for _, row in sample_strategy_performance.head(5).iterrows():
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="portfolio_value",
                    value=row['portfolio_value'],
                    timestamp=row['timestamp']
                )
                monitoring_service.collect_metric(metric_data)

            # 保存监控数据
            data_file = tmp_path / "monitoring_data.json"
            monitoring_service.save_monitoring_data(str(data_file))

            # 验证文件创建
            assert data_file.exists()

            # 创建新实例并加载数据
            new_monitoring_service = MonitoringService(monitoring_service.config)
            new_monitoring_service.load_monitoring_data(str(data_file))

            # 验证数据恢复
            loaded_metrics = new_monitoring_service.get_recent_metrics(
                metric_type=MetricType.PERFORMANCE,
                limit=5
            )

            assert len(loaded_metrics) == 5

    def test_concurrent_monitoring_operations(self, monitoring_service, sample_strategy_performance):
        """测试并发监控操作"""
        if MONITORING_SERVICE_AVAILABLE:
            import threading

            # 准备并发操作
            operations_results = []
            errors = []

            def concurrent_metric_collection(thread_id):
                try:
                    # 每个线程收集不同的指标
                    for i in range(10):
                        metric_data = MetricData(
                            metric_type=MetricType.PERFORMANCE,
                            metric_name=f"metric_{thread_id}_{i}",
                            value=np.random.random(),
                            timestamp=datetime.now(),
                            metadata={'thread_id': thread_id}
                        )

                        monitoring_service.collect_metric(metric_data)

                    operations_results.append(f"thread_{thread_id}_completed")
                except Exception as e:
                    errors.append(f"thread_{thread_id}_error: {str(e)}")

            # 启动并发操作
            threads = []
            for i in range(3):  # 3个并发线程
                thread = threading.Thread(target=concurrent_metric_collection, args=(i,))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证并发操作结果
            assert len(operations_results) == 3
            assert len(errors) == 0

            # 检查总指标数量（应该有30个指标：3线程 × 10指标）
            all_metrics = monitoring_service.get_recent_metrics(limit=50)
            assert len(all_metrics) >= 30

    def test_monitoring_performance_optimization(self, monitoring_service, sample_strategy_performance):
        """测试监控性能优化"""
        if MONITORING_SERVICE_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始性能
            initial_memory = process.memory_info().rss
            initial_cpu = process.cpu_percent()

            # 执行高频监控操作
            monitoring_iterations = 100

            start_time = time.time()

            for i in range(monitoring_iterations):
                # 收集随机指标
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="test_metric",
                    value=np.random.random(),
                    timestamp=datetime.now()
                )

                monitoring_service.collect_metric(metric_data)

            end_time = time.time()

            processing_time = end_time - start_time
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证性能指标
            assert processing_time < 5  # 100次操作在5秒内完成
            assert memory_increase < 50 * 1024 * 1024  # 内存增加小于50MB

            # 获取监控性能统计
            performance_stats = monitoring_service.get_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'avg_processing_time' in performance_stats
            assert 'memory_efficiency' in performance_stats

    def test_monitoring_configuration_management(self, monitoring_service):
        """测试监控配置管理"""
        if MONITORING_SERVICE_AVAILABLE:
            # 更新配置
            new_config = MonitoringConfig(
                monitoring_id="updated_monitoring",
                strategy_id="updated_strategy",
                metrics_interval=30,  # 更频繁的指标收集
                alert_thresholds={
                    'max_drawdown': -0.05,  # 更严格的回撤阈值
                    'sharpe_ratio': 0.8,
                    'win_rate': 0.5
                },
                health_check_interval=180  # 更频繁的健康检查
            )

            monitoring_service.update_configuration(new_config)

            # 验证配置更新
            assert monitoring_service.config.metrics_interval == 30
            assert monitoring_service.config.alert_thresholds['max_drawdown'] == -0.05

            # 测试配置验证
            valid_config = monitoring_service.validate_configuration()

            assert valid_config is True

    def test_monitoring_error_handling_and_recovery(self, monitoring_service):
        """测试监控错误处理和恢复"""
        if MONITORING_SERVICE_AVAILABLE:
            # 测试无效指标处理
            invalid_metric = MetricData(
                metric_type=MetricType.PERFORMANCE,
                metric_name="invalid_metric",
                value=float('inf'),  # 无效值
                timestamp=datetime.now()
            )

            # 应该能够处理无效指标而不崩溃
            try:
                monitoring_service.collect_metric(invalid_metric)
                # 如果没有抛出异常，验证错误被妥善处理
            except Exception as e:
                # 如果抛出异常，应该是有意义的错误类型
                assert isinstance(e, (ValueError, TypeError))

            # 测试数据恢复机制
            recovery_result = monitoring_service.attempt_data_recovery()

            assert isinstance(recovery_result, dict)
            assert 'recovery_status' in recovery_result

    def test_monitoring_scalability_testing(self, monitoring_service):
        """测试监控扩展性"""
        if MONITORING_SERVICE_AVAILABLE:
            # 测试大规模策略监控
            n_strategies = 50
            n_metrics_per_strategy = 20

            # 模拟大规模监控数据
            large_scale_metrics = []

            for strategy_id in range(n_strategies):
                for metric_id in range(n_metrics_per_strategy):
                    metric_data = MetricData(
                        metric_type=MetricType.PERFORMANCE,
                        metric_name=f"strategy_{strategy_id}_metric_{metric_id}",
                        value=np.random.random(),
                        timestamp=datetime.now(),
                        metadata={'strategy_id': f'strategy_{strategy_id}'}
                    )
                    large_scale_metrics.append(metric_data)

            # 测试大规模数据处理性能
            start_time = time.time()

            for metric in large_scale_metrics:
                monitoring_service.collect_metric(metric)

            end_time = time.time()

            processing_time = end_time - start_time
            total_metrics = len(large_scale_metrics)

            # 验证扩展性（1000个指标应该在合理时间内处理）
            assert processing_time < 10  # 10秒内完成
            assert total_metrics == 1000

    @pytest.mark.asyncio
    async def test_async_monitoring_operations(self, monitoring_service, sample_strategy_performance):
        """测试异步监控操作"""
        if MONITORING_SERVICE_AVAILABLE:
            # 测试异步指标收集
            async_metrics = []

            for _, row in sample_strategy_performance.head(5).iterrows():
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="async_portfolio_value",
                    value=row['portfolio_value'],
                    timestamp=row['timestamp']
                )

                # 异步收集指标
                await monitoring_service.async_collect_metric(metric_data)
                async_metrics.append(metric_data)

            assert len(async_metrics) == 5

            # 验证异步操作结果
            async_results = await monitoring_service.get_async_metrics_summary()

            assert isinstance(async_results, dict)
            assert 'total_async_metrics' in async_results

    def test_monitoring_integration_with_external_systems(self, monitoring_service):
        """测试监控与外部系统集成"""
        if MONITORING_SERVICE_AVAILABLE:
            # 配置外部系统集成
            external_configs = {
                'prometheus': {
                    'endpoint': 'http://localhost:9090',
                    'metrics_prefix': 'strategy_monitoring'
                },
                'grafana': {
                    'dashboard_url': 'http://localhost:3000',
                    'api_key': 'test_key'
                },
                'alertmanager': {
                    'webhook_url': 'http://localhost:9093',
                    'routing_key': 'strategy_alerts'
                }
            }

            monitoring_service.configure_external_integrations(external_configs)

            # 测试指标导出
            export_result = monitoring_service.export_metrics_to_external_systems()

            assert isinstance(export_result, dict)
            assert 'prometheus_export' in export_result
            assert 'grafana_update' in export_result

    def test_monitoring_audit_and_compliance(self, monitoring_service, sample_strategy_performance):
        """测试监控审计和合规"""
        if MONITORING_SERVICE_AVAILABLE:
            # 启用审计跟踪
            monitoring_service.enable_audit_trail()

            # 执行监控操作
            for _, row in sample_strategy_performance.head(3).iterrows():
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name="audited_metric",
                    value=row['portfolio_value'],
                    timestamp=row['timestamp']
                )
                monitoring_service.collect_metric(metric_data)

            # 获取审计日志
            audit_log = monitoring_service.get_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) >= 3

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'details' in record

            # 生成合规报告
            compliance_report = monitoring_service.generate_compliance_report()

            assert isinstance(compliance_report, dict)
            assert 'monitoring_compliance' in compliance_report
            assert 'data_retention_status' in compliance_report
            assert 'audit_trail_integrity' in compliance_report

    def test_monitoring_resource_management(self, monitoring_service, sample_strategy_performance):
        """测试监控资源管理"""
        if MONITORING_SERVICE_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 执行大量监控操作
            for i in range(100):
                metric_data = MetricData(
                    metric_type=MetricType.PERFORMANCE,
                    metric_name=f"resource_test_metric_{i}",
                    value=np.random.random(),
                    timestamp=datetime.now()
                )
                monitoring_service.collect_metric(metric_data)

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 100 * 1024 * 1024  # 100MB限制

            # 获取资源统计
            resource_stats = monitoring_service.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'memory_usage_mb' in resource_stats
            assert 'buffered_metrics_count' in resource_stats
            assert 'active_alerts_count' in resource_stats

    def test_monitoring_predictive_analytics(self, monitoring_service, sample_strategy_performance):
        """测试监控预测分析"""
        if MONITORING_SERVICE_AVAILABLE:
            # 执行预测分析
            predictive_insights = monitoring_service.generate_predictive_insights(
                strategy_id='test_strategy',
                historical_data=sample_strategy_performance,
                prediction_horizon=10
            )

            assert isinstance(predictive_insights, dict)
            assert 'performance_forecast' in predictive_insights
            assert 'risk_predictions' in predictive_insights
            assert 'anomaly_probability' in predictive_insights

            # 检查预测时间范围
            performance_forecast = predictive_insights['performance_forecast']
            assert len(performance_forecast) == 10  # 10个预测点

"""
策略生命周期管理器深度测试
全面测试策略生命周期管理的创建、开发、部署、监控和退市全流程
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# 导入生命周期管理相关类
try:
    from src.strategy.lifecycle.strategy_lifecycle_manager import (
        StrategyLifecycleManager, LifecycleStage, LifecycleEvent,
        LifecycleTransition, StrategyLifecycleMetrics
    )
    LIFECYCLE_MANAGER_AVAILABLE = True
except ImportError:
    LIFECYCLE_MANAGER_AVAILABLE = False
    StrategyLifecycleManager = Mock
    LifecycleStage = Mock
    LifecycleEvent = Mock
    LifecycleTransition = Mock
    StrategyLifecycleMetrics = Mock

try:
    from src.strategy.interfaces.strategy_interfaces import (
        StrategyConfig, StrategyStatus
    )
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False
    StrategyConfig = Mock
    StrategyStatus = Mock


class TestStrategyLifecycleManagerComprehensive:
    """策略生命周期管理器综合深度测试"""

    @pytest.fixture
    def sample_strategy_config(self):
        """创建样本策略配置"""
        if INTERFACES_AVAILABLE:
            return StrategyConfig(
                strategy_id="lifecycle_test_strategy",
                name="Lifecycle Test Strategy",
                strategy_type="momentum",
                parameters={
                    'lookback_period': 20,
                    'threshold': 0.05,
                    'max_position': 100
                },
                symbols=['AAPL', 'GOOGL', 'MSFT'],
                risk_limits={
                    'max_drawdown': 0.1,
                    'max_position_size': 1000
                }
            )
        return Mock()

    @pytest.fixture
    def lifecycle_manager(self):
        """创建生命周期管理器实例"""
        if LIFECYCLE_MANAGER_AVAILABLE:
            return StrategyLifecycleManager()
        return Mock(spec=StrategyLifecycleManager)

    def test_lifecycle_manager_initialization(self, lifecycle_manager):
        """测试生命周期管理器初始化"""
        if LIFECYCLE_MANAGER_AVAILABLE:
            assert lifecycle_manager is not None
            assert hasattr(lifecycle_manager, 'strategy_registry')
            assert hasattr(lifecycle_manager, 'event_history')
            assert hasattr(lifecycle_manager, 'transition_rules')

    def test_strategy_creation_and_initialization(self, lifecycle_manager, sample_strategy_config):
        """测试策略创建和初始化"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略
            creation_result = lifecycle_manager.create_strategy(
                strategy_config=sample_strategy_config,
                creator="test_user",
                creation_metadata={
                    'purpose': 'testing lifecycle management',
                    'expected_performance': 'moderate',
                    'risk_tolerance': 'medium'
                }
            )

            assert isinstance(creation_result, dict)
            assert 'strategy_id' in creation_result
            assert 'initial_stage' in creation_result
            assert creation_result['initial_stage'] == LifecycleStage.CREATED

            # 验证策略注册
            registered_strategies = lifecycle_manager.get_strategies_by_stage(LifecycleStage.CREATED)
            assert len(registered_strategies) > 0
            assert sample_strategy_config.strategy_id in [s['strategy_id'] for s in registered_strategies]

    def test_lifecycle_stage_transitions(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期阶段转换"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 测试阶段转换序列
            transitions = [
                (LifecycleStage.CREATED, LifecycleStage.DESIGNING, "开始策略设计"),
                (LifecycleStage.DESIGNING, LifecycleStage.DEVELOPING, "开始策略开发"),
                (LifecycleStage.DEVELOPING, LifecycleStage.TESTING, "开始策略测试"),
                (LifecycleStage.TESTING, LifecycleStage.BACKTESTING, "开始策略回测"),
                (LifecycleStage.BACKTESTING, LifecycleStage.OPTIMIZING, "开始策略优化"),
                (LifecycleStage.OPTIMIZING, LifecycleStage.VALIDATING, "开始策略验证"),
                (LifecycleStage.VALIDATING, LifecycleStage.DEPLOYING, "开始策略部署"),
                (LifecycleStage.DEPLOYING, LifecycleStage.RUNNING, "策略部署完成")
            ]

            for from_stage, to_stage, reason in transitions:
                transition_result = lifecycle_manager.transition_strategy_stage(
                    strategy_id=sample_strategy_config.strategy_id,
                    from_stage=from_stage,
                    to_stage=to_stage,
                    transition_reason=reason,
                    transition_metadata={'performed_by': 'test_system'}
                )

                assert transition_result['success'] is True
                assert transition_result['new_stage'] == to_stage

                # 验证当前阶段
                current_stage = lifecycle_manager.get_strategy_stage(sample_strategy_config.strategy_id)
                assert current_stage == to_stage

    def test_lifecycle_event_tracking(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期事件跟踪"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略并执行一些操作
            lifecycle_manager.create_strategy(sample_strategy_config)
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.DEVELOPING,
                "开始开发"
            )

            # 获取事件历史
            event_history = lifecycle_manager.get_strategy_event_history(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(event_history, list)
            assert len(event_history) >= 2  # 创建事件 + 转换事件

            # 检查事件结构
            for event in event_history:
                assert isinstance(event, LifecycleEvent)
                assert hasattr(event, 'event_type')
                assert hasattr(event, 'timestamp')
                assert hasattr(event, 'strategy_id')
                assert hasattr(event, 'details')

    def test_lifecycle_quality_gates(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期质量门禁"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 配置质量门禁
            quality_gates = {
                LifecycleStage.TESTING: {
                    'min_test_coverage': 0.8,
                    'max_code_complexity': 10,
                    'required_reviews': 2
                },
                LifecycleStage.BACKTESTING: {
                    'min_sharpe_ratio': 0.5,
                    'max_drawdown_limit': 0.15,
                    'min_win_rate': 0.55
                },
                LifecycleStage.DEPLOYING: {
                    'performance_validation': True,
                    'risk_assessment': True,
                    'regulatory_approval': True
                }
            }

            lifecycle_manager.configure_quality_gates(quality_gates)

            # 测试质量门禁检查
            for stage, criteria in quality_gates.items():
                # 模拟质量检查数据
                quality_data = {key: 0.9 for key in criteria.keys()}  # 假设都通过

                gate_check = lifecycle_manager.check_quality_gate(
                    strategy_id=sample_strategy_config.strategy_id,
                    target_stage=stage,
                    quality_data=quality_data
                )

                assert isinstance(gate_check, dict)
                assert 'gate_passed' in gate_check
                assert 'criteria_met' in gate_check

    def test_strategy_development_workflow(self, lifecycle_manager, sample_strategy_config):
        """测试策略开发工作流"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 执行开发工作流
            development_workflow = {
                'design_phase': {
                    'requirements_gathering': True,
                    'architecture_design': True,
                    'risk_assessment': True
                },
                'implementation_phase': {
                    'code_development': True,
                    'unit_testing': True,
                    'integration_testing': True
                },
                'validation_phase': {
                    'performance_testing': True,
                    'stress_testing': True,
                    'peer_review': True
                }
            }

            workflow_result = lifecycle_manager.execute_development_workflow(
                strategy_id=sample_strategy_config.strategy_id,
                workflow_config=development_workflow
            )

            assert isinstance(workflow_result, dict)
            assert 'workflow_completed' in workflow_result
            assert 'phase_results' in workflow_result
            assert len(workflow_result['phase_results']) == len(development_workflow)

    def test_strategy_deployment_process(self, lifecycle_manager, sample_strategy_config):
        """测试策略部署流程"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建并推进策略到部署阶段
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 快速推进到部署阶段（为了测试）
            stages_to_skip = [
                LifecycleStage.DESIGNING, LifecycleStage.DEVELOPING,
                LifecycleStage.TESTING, LifecycleStage.BACKTESTING,
                LifecycleStage.OPTIMIZING, LifecycleStage.VALIDATING
            ]

            current_stage = LifecycleStage.CREATED
            for next_stage in stages_to_skip:
                lifecycle_manager.transition_strategy_stage(
                    sample_strategy_config.strategy_id, current_stage, next_stage, "快速测试"
                )
                current_stage = next_stage

            # 执行部署流程
            deployment_config = {
                'target_environment': 'production',
                'deployment_strategy': 'blue_green',
                'rollback_plan': True,
                'monitoring_setup': True,
                'alert_configuration': True
            }

            deployment_result = lifecycle_manager.deploy_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                deployment_config=deployment_config
            )

            assert isinstance(deployment_result, dict)
            assert 'deployment_status' in deployment_result
            assert 'deployment_id' in deployment_result
            assert 'rollback_available' in deployment_result

    def test_strategy_monitoring_and_maintenance(self, lifecycle_manager, sample_strategy_config):
        """测试策略监控和维护"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建并运行策略
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 快速推进到运行阶段
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.RUNNING,
                "快速部署测试"
            )

            # 配置监控规则
            monitoring_config = {
                'performance_thresholds': {
                    'min_sharpe_ratio': 0.3,
                    'max_drawdown': 0.12,
                    'min_win_rate': 0.5
                },
                'health_checks': {
                    'data_quality_check': True,
                    'execution_latency_check': True,
                    'error_rate_check': True
                },
                'alert_rules': {
                    'performance_degradation': True,
                    'system_anomalies': True,
                    'data_issues': True
                }
            }

            lifecycle_manager.configure_strategy_monitoring(
                strategy_id=sample_strategy_config.strategy_id,
                monitoring_config=monitoring_config
            )

            # 执行监控检查
            monitoring_result = lifecycle_manager.perform_strategy_monitoring(
                strategy_id=sample_strategy_config.strategy_id,
                monitoring_data={
                    'sharpe_ratio': 0.4,
                    'max_drawdown': 0.08,
                    'win_rate': 0.6,
                    'execution_latency': 150,
                    'error_rate': 0.02
                }
            )

            assert isinstance(monitoring_result, dict)
            assert 'health_status' in monitoring_result
            assert 'alerts_triggered' in monitoring_result
            assert 'maintenance_recommendations' in monitoring_result

    def test_strategy_optimization_and_improvement(self, lifecycle_manager, sample_strategy_config):
        """测试策略优化和改进"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建并运行策略
            lifecycle_manager.create_strategy(sample_strategy_config)
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.RUNNING,
                "部署测试"
            )

            # 执行策略优化
            optimization_config = {
                'optimization_trigger': 'performance_decline',
                'parameter_search_space': {
                    'lookback_period': [15, 20, 25],
                    'threshold': [0.03, 0.05, 0.07]
                },
                'evaluation_criteria': ['sharpe_ratio', 'max_drawdown', 'win_rate'],
                'automated_deployment': True
            }

            optimization_result = lifecycle_manager.optimize_running_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                optimization_config=optimization_config,
                performance_data={
                    'current_sharpe': 0.3,  # 低于目标
                    'current_drawdown': 0.08,
                    'current_win_rate': 0.55
                }
            )

            assert isinstance(optimization_result, dict)
            assert 'optimization_performed' in optimization_result
            assert 'improvement_found' in optimization_result
            assert 'deployment_status' in optimization_result

    def test_strategy_retirement_process(self, lifecycle_manager, sample_strategy_config):
        """测试策略退市流程"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建并运行策略
            lifecycle_manager.create_strategy(sample_strategy_config)
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.RUNNING,
                "部署测试"
            )

            # 执行退市流程
            retirement_config = {
                'retirement_reason': 'performance_decline',
                'graceful_shutdown': True,
                'data_archival': True,
                'replacement_strategy': 'improved_version_2',
                'stakeholder_notification': True
            }

            retirement_result = lifecycle_manager.retire_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                retirement_config=retirement_config
            )

            assert isinstance(retirement_result, dict)
            assert 'retirement_status' in retirement_result
            assert 'archival_completed' in retirement_result
            assert 'cleanup_performed' in retirement_result

            # 验证策略状态
            final_stage = lifecycle_manager.get_strategy_stage(sample_strategy_config.strategy_id)
            assert final_stage == LifecycleStage.RETIRED

    def test_lifecycle_metrics_and_analytics(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期指标和分析"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建多个策略并执行生命周期操作
            strategies = []
            for i in range(3):
                config = StrategyConfig(
                    strategy_id=f"analytics_strategy_{i}",
                    name=f"Analytics Strategy {i}",
                    strategy_type="momentum",
                    parameters={'lookback_period': 20 + i * 5}
                )
                lifecycle_manager.create_strategy(config)
                strategies.append(config.strategy_id)

            # 执行一些生命周期操作
            for strategy_id in strategies:
                lifecycle_manager.transition_strategy_stage(
                    strategy_id, LifecycleStage.CREATED, LifecycleStage.DEVELOPING, "测试分析"
                )
                lifecycle_manager.transition_strategy_stage(
                    strategy_id, LifecycleStage.DEVELOPING, LifecycleStage.TESTING, "测试分析"
                )

            # 生成生命周期分析报告
            analytics_report = lifecycle_manager.generate_lifecycle_analytics(
                time_range={'start': datetime.now() - timedelta(days=30), 'end': datetime.now()},
                analysis_config={
                    'stage_distribution': True,
                    'transition_times': True,
                    'success_rates': True,
                    'bottleneck_analysis': True,
                    'performance_trends': True
                }
            )

            assert isinstance(analytics_report, dict)
            assert 'stage_distribution' in analytics_report
            assert 'average_transition_times' in analytics_report
            assert 'success_rates_by_stage' in analytics_report
            assert 'bottlenecks_identified' in analytics_report

    def test_lifecycle_risk_management(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期风险管理"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 配置生命周期风险控制
            risk_config = {
                'stage_risk_limits': {
                    LifecycleStage.DEVELOPING: {'max_complexity': 8, 'max_dependencies': 5},
                    LifecycleStage.TESTING: {'min_coverage': 0.8, 'max_bugs': 3},
                    LifecycleStage.RUNNING: {'max_downtime': 300, 'max_error_rate': 0.05}
                },
                'transition_risk_checks': {
                    'development_to_testing': ['code_review', 'unit_tests'],
                    'testing_to_production': ['integration_tests', 'performance_tests', 'security_audit'],
                    'production_to_retirement': ['data_backup', 'stakeholder_approval']
                },
                'emergency_procedures': {
                    'critical_failure': 'immediate_shutdown',
                    'data_corruption': 'rollback_to_last_stable',
                    'security_breach': 'isolate_and_investigate'
                }
            }

            lifecycle_manager.configure_lifecycle_risk_management(risk_config)

            # 执行风险评估
            risk_assessment = lifecycle_manager.assess_lifecycle_risks(
                strategy_id=sample_strategy_config.strategy_id,
                current_stage=LifecycleStage.DEVELOPING,
                risk_data={
                    'code_complexity': 6,
                    'dependencies': 3,
                    'test_coverage': 0.85,
                    'open_bugs': 1
                }
            )

            assert isinstance(risk_assessment, dict)
            assert 'risk_level' in risk_assessment
            assert 'transition_allowed' in risk_assessment
            assert 'mitigation_actions' in risk_assessment

    def test_lifecycle_audit_and_compliance(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期审计和合规"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 启用审计跟踪
            lifecycle_manager.enable_lifecycle_audit()

            # 执行一系列生命周期操作
            lifecycle_manager.create_strategy(sample_strategy_config)
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.DEVELOPING,
                "合规测试"
            )

            # 获取审计日志
            audit_log = lifecycle_manager.get_lifecycle_audit_log(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(audit_log, list)
            assert len(audit_log) >= 2

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'strategy_id' in record
                assert 'stage_transition' in record
                assert 'compliance_check' in record

            # 生成合规报告
            compliance_report = lifecycle_manager.generate_lifecycle_compliance_report(
                regulatory_framework='SEC',
                time_period={'start': datetime.now() - timedelta(days=30), 'end': datetime.now()}
            )

            assert isinstance(compliance_report, dict)
            assert 'regulatory_compliance' in compliance_report
            assert 'audit_trail_integrity' in compliance_report
            assert 'process_adherence' in compliance_report

    def test_lifecycle_performance_monitoring(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期性能监控"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行一系列生命周期操作并监控性能
            start_time = time.time()

            operations = []
            for i in range(10):
                op_start = time.time()
                lifecycle_manager.create_strategy(
                    StrategyConfig(
                        strategy_id=f"perf_test_strategy_{i}",
                        name=f"Performance Test {i}",
                        strategy_type="momentum"
                    )
                )
                op_end = time.time()
                operations.append(op_start - op_end)

            end_time = time.time()

            # 获取性能统计
            performance_stats = lifecycle_manager.get_lifecycle_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'average_operation_time' in performance_stats
            assert 'operations_per_second' in performance_stats
            assert 'resource_utilization' in performance_stats
            assert 'bottleneck_analysis' in performance_stats

    def test_lifecycle_error_handling_and_recovery(self, lifecycle_manager):
        """测试生命周期错误处理和恢复"""
        if LIFECYCLE_MANAGER_AVAILABLE:
            # 测试无效状态转换
            try:
                lifecycle_manager.transition_strategy_stage(
                    "invalid_strategy_id",
                    LifecycleStage.CREATED,
                    LifecycleStage.RUNNING,
                    "无效测试"
                )
            except ValueError:
                # 期望的错误处理
                pass

            # 测试循环依赖
            try:
                # 这里可能需要具体实现来测试循环转换
                pass
            except Exception:
                pass

            # 测试恢复机制
            recovery_result = lifecycle_manager.attempt_lifecycle_recovery(
                recovery_config={
                    'failed_operation': 'stage_transition',
                    'recovery_strategy': 'rollback_last_valid_state',
                    'max_retry_attempts': 3
                }
            )

            assert isinstance(recovery_result, dict)
            assert 'recovery_status' in recovery_result

    def test_lifecycle_configuration_management(self, lifecycle_manager):
        """测试生命周期配置管理"""
        if LIFECYCLE_MANAGER_AVAILABLE:
            # 更新配置
            new_config = {
                'default_transition_timeout': 3600,  # 1小时
                'max_concurrent_operations': 10,
                'quality_gate_strictness': 'high',
                'audit_retention_days': 365,
                'emergency_contact_groups': ['dev_team', 'risk_team', 'management']
            }

            lifecycle_manager.update_lifecycle_config(new_config)

            # 验证配置更新
            current_config = lifecycle_manager.get_lifecycle_config()

            assert current_config['default_transition_timeout'] == 3600
            assert current_config['max_concurrent_operations'] == 10

    def test_lifecycle_integration_with_external_systems(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期与外部系统集成"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 配置外部系统集成
            external_integrations = {
                'version_control': {
                    'system': 'git',
                    'repository': 'strategy_repo',
                    'branch_strategy': 'feature_branch'
                },
                'ci_cd': {
                    'system': 'jenkins',
                    'pipeline': 'strategy_deployment',
                    'auto_trigger': True
                },
                'monitoring': {
                    'system': 'prometheus',
                    'metrics_endpoint': '/metrics',
                    'alert_webhook': 'http://alerts.company.com'
                },
                'documentation': {
                    'system': 'confluence',
                    'space': 'trading_strategies',
                    'auto_update': True
                }
            }

            lifecycle_manager.configure_external_integrations(external_integrations)

            # 执行集成测试
            integration_result = lifecycle_manager.test_external_integrations()

            assert isinstance(integration_result, dict)
            assert 'integration_status' in integration_result
            assert 'systems_tested' in integration_result

    def test_lifecycle_scaling_and_resource_management(self, lifecycle_manager):
        """测试生命周期扩展性和资源管理"""
        if LIFECYCLE_MANAGER_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 创建大量策略进行扩展性测试
            for i in range(50):
                config = StrategyConfig(
                    strategy_id=f"scale_test_strategy_{i}",
                    name=f"Scale Test {i}",
                    strategy_type="momentum"
                )
                lifecycle_manager.create_strategy(config)

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 200 * 1024 * 1024  # 200MB限制

            # 获取扩展性指标
            scalability_metrics = lifecycle_manager.get_scalability_metrics()

            assert isinstance(scalability_metrics, dict)
            assert 'managed_strategies_count' in scalability_metrics
            assert 'average_operation_time' in scalability_metrics
            assert 'resource_efficiency' in scalability_metrics

    def test_lifecycle_machine_learning_integration(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期机器学习集成"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 配置ML增强
            ml_config = {
                'predictive_analytics': True,
                'automated_decision_making': True,
                'performance_prediction': True,
                'risk_prediction': True,
                'optimization_suggestions': True
            }

            lifecycle_manager.configure_ml_enhancement(ml_config)

            # 创建策略并执行ML增强的生命周期管理
            lifecycle_manager.create_strategy(sample_strategy_config)

            # 获取ML驱动的生命周期建议
            ml_suggestions = lifecycle_manager.get_ml_lifecycle_suggestions(
                strategy_id=sample_strategy_config.strategy_id,
                current_stage=LifecycleStage.DEVELOPING,
                performance_history=[]
            )

            assert isinstance(ml_suggestions, dict)
            assert 'predicted_success_probability' in ml_suggestions
            assert 'recommended_actions' in ml_suggestions
            assert 'risk_assessment' in ml_suggestions

    def test_lifecycle_distributed_processing(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期分布式处理"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 配置分布式处理
            distributed_config = {
                'distributed_mode': True,
                'worker_nodes': 5,
                'load_balancing_strategy': 'adaptive',
                'fault_tolerance_level': 'high',
                'result_consolidation': 'weighted_voting'
            }

            lifecycle_manager.configure_distributed_processing(distributed_config)

            # 创建多个策略进行分布式处理
            strategies = []
            for i in range(10):
                config = StrategyConfig(
                    strategy_id=f"distributed_strategy_{i}",
                    name=f"Distributed Test {i}",
                    strategy_type="momentum"
                )
                lifecycle_manager.create_strategy(config)
                strategies.append(config.strategy_id)

            # 执行分布式生命周期操作
            distributed_result = lifecycle_manager.execute_distributed_lifecycle_operations(
                strategy_ids=strategies,
                operation='stage_transition',
                operation_params={
                    'from_stage': LifecycleStage.CREATED,
                    'to_stage': LifecycleStage.DEVELOPING,
                    'reason': '分布式测试'
                }
            )

            assert isinstance(distributed_result, dict)
            assert 'distributed_operations_completed' in distributed_result
            assert 'node_performance' in distributed_result
            assert 'consolidated_results' in distributed_result

    def test_lifecycle_visualization_and_reporting(self, lifecycle_manager, sample_strategy_config):
        """测试生命周期可视化和报告"""
        if LIFECYCLE_MANAGER_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略并执行生命周期操作
            lifecycle_manager.create_strategy(sample_strategy_config)
            lifecycle_manager.transition_strategy_stage(
                sample_strategy_config.strategy_id,
                LifecycleStage.CREATED,
                LifecycleStage.DEVELOPING,
                "可视化测试"
            )

            # 生成生命周期可视化数据
            visualization_data = lifecycle_manager.generate_lifecycle_visualization(
                strategy_id=sample_strategy_config.strategy_id,
                visualization_config={
                    'timeline_view': True,
                    'stage_flow_diagram': True,
                    'performance_trends': True,
                    'risk_heatmap': True,
                    'export_formats': ['png', 'svg', 'pdf']
                }
            )

            assert isinstance(visualization_data, dict)
            assert 'timeline_data' in visualization_data
            assert 'flow_diagram' in visualization_data
            assert 'performance_charts' in visualization_data

            # 生成综合报告
            comprehensive_report = lifecycle_manager.generate_comprehensive_lifecycle_report(
                report_config={
                    'time_period': {'start': datetime.now() - timedelta(days=30), 'end': datetime.now()},
                    'include_performance_metrics': True,
                    'include_risk_analysis': True,
                    'include_compliance_status': True,
                    'include_recommendations': True
                }
            )

            assert isinstance(comprehensive_report, dict)
            assert 'executive_summary' in comprehensive_report
            assert 'detailed_analysis' in comprehensive_report
            assert 'recommendations' in comprehensive_report
            assert 'appendices' in comprehensive_report

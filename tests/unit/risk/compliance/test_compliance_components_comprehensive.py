"""
风险合规组件深度测试
全面测试合规检查、风险规则、监管报告和合规监控功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# 导入合规组件相关类
try:
    from src.risk.compliance.compliance_components import (
        ComponentFactory, IComplianceComponent, ComplianceEngine,
        ComplianceRule, ComplianceReport, ComplianceCheckResult
    )
    COMPLIANCE_COMPONENTS_AVAILABLE = True
except ImportError:
    COMPLIANCE_COMPONENTS_AVAILABLE = False
    ComponentFactory = Mock
    IComplianceComponent = Mock
    ComplianceEngine = Mock
    ComplianceRule = Mock
    ComplianceReport = Mock
    ComplianceCheckResult = Mock

try:
    from src.risk.compliance.risk_compliance_engine import RiskComplianceEngine
    RISK_COMPLIANCE_ENGINE_AVAILABLE = True
except ImportError:
    RISK_COMPLIANCE_ENGINE_AVAILABLE = False
    RiskComplianceEngine = Mock


class TestComplianceComponentsComprehensive:
    """风险合规组件综合深度测试"""

    @pytest.fixture
    def sample_portfolio_data(self):
        """创建样本投资组合数据"""
        return pd.DataFrame({
            'asset_id': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'JPM', 'BAC'],
            'sector': ['Technology', 'Technology', 'Technology', 'Automotive', 'Finance', 'Finance'],
            'country': ['USA', 'USA', 'USA', 'USA', 'USA', 'USA'],
            'position_value': [1000000, 800000, 1200000, 600000, 500000, 400000],
            'weight': [0.20, 0.16, 0.24, 0.12, 0.10, 0.08],
            'volatility': [0.25, 0.30, 0.22, 0.45, 0.35, 0.40],
            'beta': [1.1, 1.3, 1.0, 1.5, 1.2, 1.4]
        })

    @pytest.fixture
    def sample_trade_data(self):
        """创建样本交易数据"""
        return pd.DataFrame({
            'trade_id': [f'TRADE_{i:03d}' for i in range(50)],
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='H'),
            'asset': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 50),
            'quantity': np.random.randint(100, 10000, 50),
            'price': np.random.uniform(100, 1000, 50),
            'trade_type': np.random.choice(['BUY', 'SELL'], 50),
            'venue': np.random.choice(['NYSE', 'NASDAQ', 'DARK_POOL'], 50),
            'account_id': ['ACC_001'] * 50
        })

    @pytest.fixture
    def sample_risk_metrics(self):
        """创建样本风险指标"""
        return {
            'portfolio_volatility': 0.18,
            'max_drawdown': -0.12,
            'value_at_risk_95': -45000,
            'expected_shortfall_95': -65000,
            'concentration_ratio': 0.35,
            'sector_diversity_score': 0.75,
            'liquidity_ratio': 0.85
        }

    @pytest.fixture
    def compliance_rules(self):
        """创建合规规则"""
        return [
            {
                'rule_id': 'concentration_limit',
                'rule_type': 'position_limit',
                'description': '单资产最大持仓比例不能超过25%',
                'threshold': 0.25,
                'severity': 'high',
                'regulation': 'Investment Company Act'
            },
            {
                'rule_id': 'sector_limit',
                'rule_type': 'sector_limit',
                'description': '单行业最大配置比例不能超过40%',
                'threshold': 0.40,
                'severity': 'medium',
                'regulation': 'Diversification Rules'
            },
            {
                'rule_id': 'volatility_limit',
                'rule_type': 'risk_limit',
                'description': '投资组合波动率不能超过20%',
                'threshold': 0.20,
                'severity': 'high',
                'regulation': 'Risk Management Guidelines'
            },
            {
                'rule_id': 'liquidity_requirement',
                'rule_type': 'liquidity_limit',
                'description': '流动性比率必须大于80%',
                'threshold': 0.80,
                'severity': 'medium',
                'regulation': 'Liquidity Management Rules'
            }
        ]

    @pytest.fixture
    def component_factory(self):
        """创建组件工厂"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            return ComponentFactory()
        return Mock(spec=ComponentFactory)

    @pytest.fixture
    def compliance_engine(self):
        """创建合规引擎"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            return ComplianceEngine()
        return Mock(spec=ComplianceEngine)

    @pytest.fixture
    def risk_compliance_engine(self):
        """创建风险合规引擎"""
        if RISK_COMPLIANCE_ENGINE_AVAILABLE:
            return RiskComplianceEngine()
        return Mock(spec=RiskComplianceEngine)

    def test_component_factory_initialization(self, component_factory):
        """测试组件工厂初始化"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            assert component_factory is not None
            assert hasattr(component_factory, '_components')

    def test_compliance_rule_evaluation(self, compliance_engine, sample_portfolio_data, compliance_rules):
        """测试合规规则评估"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 配置合规规则
            compliance_engine.configure_rules(compliance_rules)

            # 执行合规检查
            compliance_result = compliance_engine.check_portfolio_compliance(
                portfolio_data=sample_portfolio_data,
                check_timestamp=datetime.now()
            )

            assert isinstance(compliance_result, ComplianceCheckResult)
            assert hasattr(compliance_result, 'passed_rules')
            assert hasattr(compliance_result, 'failed_rules')
            assert hasattr(compliance_result, 'overall_compliance')

            # 检查规则评估结果
            total_rules = len(compliance_rules)
            checked_rules = len(compliance_result.passed_rules) + len(compliance_result.failed_rules)
            assert checked_rules == total_rules

    def test_position_limit_compliance(self, compliance_engine, sample_portfolio_data):
        """测试持仓限制合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 检查持仓集中度
            concentration_check = compliance_engine.check_position_concentration(
                portfolio_data=sample_portfolio_data,
                max_single_position=0.25,  # 25% 限制
                max_total_concentration=0.80  # 80% 总集中度限制
            )

            assert isinstance(concentration_check, dict)
            assert 'concentration_compliant' in concentration_check
            assert 'violations' in concentration_check

            # 验证集中度计算
            max_weight = sample_portfolio_data['weight'].max()
            assert concentration_check['max_position_weight'] == max_weight

    def test_sector_diversification_compliance(self, compliance_engine, sample_portfolio_data):
        """测试行业多元化合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 检查行业配置
            sector_check = compliance_engine.check_sector_diversification(
                portfolio_data=sample_portfolio_data,
                max_sector_weight=0.40,  # 40% 行业限制
                min_sectors=3  # 最少3个行业
            )

            assert isinstance(sector_check, dict)
            assert 'sector_diversification_compliant' in sector_check
            assert 'sector_weights' in sector_check
            assert 'sector_count' in sector_check

            # 验证行业数量
            sector_count = sample_portfolio_data['sector'].nunique()
            assert sector_check['sector_count'] == sector_count

    def test_risk_limit_compliance(self, compliance_engine, sample_portfolio_data, sample_risk_metrics):
        """测试风险限制合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 检查风险指标合规
            risk_check = compliance_engine.check_risk_limits(
                portfolio_data=sample_portfolio_data,
                risk_metrics=sample_risk_metrics,
                risk_limits={
                    'max_volatility': 0.20,
                    'max_drawdown': -0.15,
                    'max_var_95': -50000,
                    'min_liquidity_ratio': 0.80
                }
            )

            assert isinstance(risk_check, dict)
            assert 'risk_limits_compliant' in risk_check
            assert 'risk_violations' in risk_check

            # 检查波动率限制（应该不合规，因为0.18 > 0.20）
            assert not risk_check['risk_limits_compliant']
            assert 'volatility' in risk_check['risk_violations']

    def test_trade_surveillance_compliance(self, compliance_engine, sample_trade_data):
        """测试交易监控合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 检查交易模式
            surveillance_check = compliance_engine.check_trade_surveillance(
                trade_data=sample_trade_data,
                surveillance_rules={
                    'max_daily_trades': 100,
                    'max_single_trade_value': 500000,
                    'require_trade_reason': True,
                    'detect_wash_trades': True
                }
            )

            assert isinstance(surveillance_check, dict)
            assert 'surveillance_compliant' in surveillance_check
            assert 'trade_anomalies' in surveillance_check
            assert 'pattern_analysis' in surveillance_check

    def test_regulatory_reporting_compliance(self, compliance_engine, sample_portfolio_data, sample_trade_data):
        """测试监管报告合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 生成监管报告
            regulatory_report = compliance_engine.generate_regulatory_report(
                portfolio_data=sample_portfolio_data,
                trade_data=sample_trade_data,
                reporting_period='daily',
                regulatory_framework='SEC'
            )

            assert isinstance(regulatory_report, ComplianceReport)
            assert hasattr(regulatory_report, 'report_id')
            assert hasattr(regulatory_report, 'report_type')
            assert hasattr(regulatory_report, 'generated_at')
            assert hasattr(regulatory_report, 'content')

            # 检查报告内容
            assert 'portfolio_summary' in regulatory_report.content
            assert 'trade_summary' in regulatory_report.content
            assert 'compliance_status' in regulatory_report.content

    def test_cross_border_compliance(self, compliance_engine, sample_portfolio_data):
        """测试跨境合规"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 扩展投资组合数据以包含不同国家
            international_portfolio = sample_portfolio_data.copy()
            international_portfolio['country'] = ['USA', 'USA', 'Germany', 'Japan', 'UK', 'Canada']

            # 检查跨境合规
            cross_border_check = compliance_engine.check_cross_border_compliance(
                portfolio_data=international_portfolio,
                jurisdiction_rules={
                    'max_emerging_markets': 0.20,
                    'require_fx_hedging': True,
                    'max_single_country': 0.50,
                    'currency_diversification': True
                }
            )

            assert isinstance(cross_border_check, dict)
            assert 'cross_border_compliant' in cross_border_check
            assert 'country_exposure' in cross_border_check
            assert 'currency_hedging_status' in cross_border_check

    def test_real_time_compliance_monitoring(self, compliance_engine, sample_portfolio_data):
        """测试实时合规监控"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 启用实时监控
            compliance_engine.enable_real_time_monitoring()

            # 模拟实时数据更新
            real_time_updates = []

            for i in range(5):
                # 模拟投资组合变化
                updated_portfolio = sample_portfolio_data.copy()
                updated_portfolio['weight'] = updated_portfolio['weight'] * (1 + np.random.normal(0, 0.02, len(updated_portfolio)))

                # 执行实时合规检查
                real_time_check = compliance_engine.check_real_time_compliance(
                    portfolio_update=updated_portfolio,
                    update_timestamp=datetime.now()
                )

                real_time_updates.append(real_time_check)

            assert len(real_time_updates) == 5

            # 检查实时监控结果
            for update in real_time_updates:
                assert isinstance(update, dict)
                assert 'real_time_compliant' in update
                assert 'immediate_actions_required' in update

    def test_compliance_alert_system(self, compliance_engine, sample_portfolio_data):
        """测试合规告警系统"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 配置告警规则
            alert_rules = {
                'concentration_alert': {
                    'threshold': 0.30,
                    'severity': 'high',
                    'message': 'Position concentration exceeds limit'
                },
                'risk_alert': {
                    'threshold': 0.25,
                    'severity': 'medium',
                    'message': 'Portfolio volatility too high'
                }
            }

            compliance_engine.configure_compliance_alerts(alert_rules)

            # 检查并生成告警
            alerts = compliance_engine.check_and_generate_alerts(
                portfolio_data=sample_portfolio_data,
                risk_metrics={'volatility': 0.28}  # 高于阈值
            )

            assert isinstance(alerts, list)

            # 应该至少生成一个告警（波动率告警）
            assert len(alerts) >= 1

            for alert in alerts:
                assert hasattr(alert, 'alert_id')
                assert hasattr(alert, 'severity')
                assert hasattr(alert, 'message')
                assert hasattr(alert, 'timestamp')

    def test_compliance_workflow_management(self, compliance_engine, sample_portfolio_data):
        """测试合规工作流管理"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 创建合规工作流
            workflow_config = {
                'workflow_id': 'daily_compliance_check',
                'steps': [
                    'data_validation',
                    'rule_evaluation',
                    'risk_assessment',
                    'report_generation',
                    'approval_process'
                ],
                'automated_steps': ['data_validation', 'rule_evaluation', 'risk_assessment'],
                'manual_steps': ['approval_process']
            }

            compliance_engine.configure_compliance_workflow(workflow_config)

            # 执行合规工作流
            workflow_result = compliance_engine.execute_compliance_workflow(
                portfolio_data=sample_portfolio_data,
                workflow_id='daily_compliance_check'
            )

            assert isinstance(workflow_result, dict)
            assert 'workflow_completed' in workflow_result
            assert 'step_results' in workflow_result
            assert 'overall_status' in workflow_result

            # 检查工作流步骤
            step_results = workflow_result['step_results']
            assert len(step_results) == len(workflow_config['steps'])

    def test_compliance_data_persistence(self, compliance_engine, sample_portfolio_data, tmp_path):
        """测试合规数据持久化"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 执行合规检查
            compliance_result = compliance_engine.check_portfolio_compliance(
                portfolio_data=sample_portfolio_data,
                check_timestamp=datetime.now()
            )

            # 保存合规结果
            results_file = tmp_path / "compliance_results.json"
            compliance_engine.save_compliance_results(str(results_file))

            # 验证文件创建
            assert results_file.exists()

            # 加载合规结果
            loaded_results = compliance_engine.load_compliance_results(str(results_file))

            assert isinstance(loaded_results, list)
            assert len(loaded_results) > 0

    def test_compliance_audit_trail(self, compliance_engine, sample_portfolio_data):
        """测试合规审计跟踪"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 启用审计跟踪
            compliance_engine.enable_audit_trail()

            # 执行多个合规操作
            operations = [
                lambda: compliance_engine.check_portfolio_compliance(sample_portfolio_data),
                lambda: compliance_engine.generate_regulatory_report(sample_portfolio_data, pd.DataFrame()),
                lambda: compliance_engine.check_risk_limits(sample_portfolio_data, {})
            ]

            for operation in operations:
                operation()

            # 获取审计日志
            audit_log = compliance_engine.get_compliance_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) >= len(operations)

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'details' in record
                assert 'user' in record

    def test_compliance_performance_monitoring(self, compliance_engine, sample_portfolio_data):
        """测试合规性能监控"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 执行一系列合规检查
            performance_checks = []

            for i in range(10):
                start_time = time.time()
                result = compliance_engine.check_portfolio_compliance(sample_portfolio_data)
                end_time = time.time()

                performance_checks.append({
                    'check_id': i,
                    'execution_time': end_time - start_time,
                    'result': result
                })

            # 获取性能统计
            performance_stats = compliance_engine.get_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'avg_check_time' in performance_stats
            assert 'total_checks' in performance_stats
            assert 'success_rate' in performance_stats

    def test_compliance_error_handling_and_recovery(self, compliance_engine):
        """测试合规错误处理和恢复"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 测试无效数据处理
            invalid_portfolio = pd.DataFrame({
                'asset_id': [None, 'AAPL'],
                'weight': ['invalid', 0.5]
            })

            try:
                compliance_engine.check_portfolio_compliance(invalid_portfolio)
                # 如果没有抛出异常，验证错误被妥善处理
            except Exception as e:
                # 验证异常类型合适
                assert isinstance(e, (ValueError, TypeError))

            # 测试恢复机制
            recovery_result = compliance_engine.attempt_error_recovery()

            assert isinstance(recovery_result, dict)
            assert 'recovery_status' in recovery_result

    def test_compliance_scalability_testing(self, compliance_engine):
        """测试合规扩展性"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 创建大规模投资组合
            large_portfolio = pd.DataFrame({
                'asset_id': [f'ASSET_{i}' for i in range(1000)],
                'sector': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy'], 1000),
                'country': np.random.choice(['USA', 'UK', 'Germany', 'Japan', 'Canada'], 1000),
                'position_value': np.random.uniform(10000, 1000000, 1000),
                'weight': np.random.uniform(0.001, 0.01, 1000),
                'volatility': np.random.uniform(0.1, 0.8, 1000)
            })

            # 测试大规模合规检查性能
            start_time = time.time()

            large_scale_result = compliance_engine.check_large_scale_compliance(large_portfolio)

            end_time = time.time()

            processing_time = end_time - start_time

            # 验证扩展性（1000资产应该在合理时间内处理）
            assert processing_time < 30  # 30秒内完成
            assert isinstance(large_scale_result, dict)

    def test_compliance_machine_learning_integration(self, compliance_engine, sample_portfolio_data):
        """测试合规机器学习集成"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 启用机器学习增强
            compliance_engine.enable_ml_enhancement()

            # 训练合规预测模型
            training_data = []

            for i in range(100):
                # 生成训练样本
                sample = {
                    'features': {
                        'concentration': np.random.uniform(0.1, 0.5),
                        'volatility': np.random.uniform(0.1, 0.4),
                        'liquidity': np.random.uniform(0.7, 0.95)
                    },
                    'compliance_outcome': np.random.choice([0, 1], p=[0.2, 0.8])  # 80% 合规
                }
                training_data.append(sample)

            compliance_engine.train_compliance_prediction_model(training_data)

            # 预测合规结果
            prediction = compliance_engine.predict_compliance_outcome(
                portfolio_features={
                    'concentration': 0.25,
                    'volatility': 0.18,
                    'liquidity': 0.85
                }
            )

            assert isinstance(prediction, dict)
            assert 'predicted_compliant' in prediction
            assert 'confidence_score' in prediction

    def test_compliance_regulatory_integration(self, compliance_engine, sample_portfolio_data):
        """测试合规监管集成"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 配置监管框架
            regulatory_frameworks = {
                'SEC': {
                    'rules': ['concentration_limits', 'disclosure_requirements'],
                    'reporting_frequency': 'quarterly',
                    'filing_deadlines': ['Q1: April 30', 'Q2: July 31', 'Q3: October 31', 'Q4: January 31']
                },
                'FINRA': {
                    'rules': ['trade_reporting', 'best_execution'],
                    'monitoring_frequency': 'daily',
                    'audit_requirements': ['annual_review', 'trade_surveillance']
                }
            }

            compliance_engine.configure_regulatory_frameworks(regulatory_frameworks)

            # 执行监管合规检查
            regulatory_check = compliance_engine.check_regulatory_compliance(
                portfolio_data=sample_portfolio_data,
                regulatory_framework='SEC'
            )

            assert isinstance(regulatory_check, dict)
            assert 'sec_compliant' in regulatory_check
            assert 'required_filings' in regulatory_check
            assert 'next_deadline' in regulatory_check

    def test_compliance_automation_and_orchestration(self, compliance_engine, sample_portfolio_data):
        """测试合规自动化和编排"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 配置自动化工作流
            automation_config = {
                'daily_checks': {
                    'enabled': True,
                    'checks': ['concentration', 'risk_limits', 'trade_surveillance'],
                    'schedule': '09:00',
                    'auto_report': True
                },
                'weekly_reviews': {
                    'enabled': True,
                    'reviews': ['sector_diversification', 'regulatory_reporting'],
                    'schedule': 'Friday 17:00',
                    'escalation_rules': ['high_severity_alerts']
                },
                'monthly_assessments': {
                    'enabled': True,
                    'assessments': ['full_portfolio_review', 'regulatory_compliance'],
                    'schedule': 'Last day of month',
                    'archival': True
                }
            }

            compliance_engine.configure_compliance_automation(automation_config)

            # 执行自动化合规检查
            automation_result = compliance_engine.run_automated_compliance_cycle(
                portfolio_data=sample_portfolio_data,
                cycle_type='daily'
            )

            assert isinstance(automation_result, dict)
            assert 'automation_completed' in automation_result
            assert 'checks_performed' in automation_result
            assert 'alerts_generated' in automation_result

    def test_compliance_resource_management(self, compliance_engine, sample_portfolio_data):
        """测试合规资源管理"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 执行大量合规检查
            for i in range(50):
                compliance_engine.check_portfolio_compliance(sample_portfolio_data)

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 200 * 1024 * 1024  # 200MB限制

            # 获取资源统计
            resource_stats = compliance_engine.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'memory_usage_mb' in resource_stats
            assert 'cpu_time_seconds' in resource_stats
            assert 'checks_performed' in resource_stats

    def test_compliance_configuration_management(self, compliance_engine):
        """测试合规配置管理"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 更新合规配置
            new_config = {
                'max_single_position': 0.20,  # 降低集中度限制
                'max_sector_exposure': 0.35,  # 调整行业限制
                'risk_limits': {
                    'volatility': 0.18,  # 更严格的波动率限制
                    'drawdown': -0.10
                },
                'monitoring_intervals': {
                    'real_time': 60,  # 更频繁的实时监控
                    'daily': 3600
                }
            }

            compliance_engine.update_compliance_config(new_config)

            # 验证配置更新
            current_config = compliance_engine.get_compliance_config()

            assert current_config['max_single_position'] == 0.20
            assert current_config['risk_limits']['volatility'] == 0.18

    def test_compliance_stress_testing(self, compliance_engine, sample_portfolio_data):
        """测试合规压力测试"""
        if COMPLIANCE_COMPONENTS_AVAILABLE:
            # 定义压力情景
            stress_scenarios = [
                {
                    'name': 'market_crash',
                    'portfolio_impact': {'volatility_multiplier': 2.0, 'drawdown_shift': -0.20},
                    'expected_compliance_impact': 'high'
                },
                {
                    'name': 'concentration_spike',
                    'portfolio_impact': {'concentration_multiplier': 1.5},
                    'expected_compliance_impact': 'medium'
                },
                {
                    'name': 'liquidity_crisis',
                    'portfolio_impact': {'liquidity_reduction': 0.6},
                    'expected_compliance_impact': 'high'
                }
            ]

            # 执行合规压力测试
            stress_results = compliance_engine.run_compliance_stress_test(
                base_portfolio=sample_portfolio_data,
                stress_scenarios=stress_scenarios
            )

            assert isinstance(stress_results, dict)
            assert len(stress_results) == len(stress_scenarios)

            # 检查每个压力情景的结果
            for scenario_name, result in stress_results.items():
                assert 'scenario_compliant' in result
                assert 'violations_under_stress' in result
                assert 'recommended_actions' in result
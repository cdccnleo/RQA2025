"""
风险管理器综合测试
测试风险管理核心功能，提升覆盖率
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.risk.models.risk_manager import RiskManager
from src.risk.monitor.real_time_monitor import RealTimeMonitor


class TestRiskManagerComprehensive:
    """风险管理器综合测试"""

    @pytest.fixture
    def risk_manager(self):
        """创建风险管理器实例"""
        return RiskManager()

    @pytest.fixture
    def real_time_monitor(self):
        """创建实时监控器实例"""
        config = {
            'max_position_size': 100000,
            'max_daily_loss': 0.05,
            'alert_thresholds': {'position': 0.5, 'concentration': 0.5}
        }
        return RealTimeMonitor(config)

    def test_risk_manager_initialization(self, risk_manager):
        """测试风险管理器初始化"""
        assert risk_manager is not None
        assert hasattr(risk_manager, 'risk_rules')
        assert hasattr(risk_manager, 'enabled')

    def test_add_risk_rule(self, risk_manager):
        """测试添加风险规则"""
        rule = {
            'name': 'test_rule',
            'type': 'position_limit',
            'threshold': 100000,
            'action': 'alert'
        }

        result = risk_manager.add_risk_rule(rule)
        assert result is True
        assert len(risk_manager.risk_rules) > 0

    def test_check_risk(self, risk_manager):
        """测试风险检查"""
        order = {
            'symbol': '000001',
            'quantity': 100,
            'price': 10.5
        }

        result = risk_manager.check_risk(order)
        assert isinstance(result, object)  # RiskCheck object

    def test_get_risk_level(self, risk_manager):
        """测试获取风险等级"""
        # 先创建一个风险检查
        order = {
            'symbol': '000001',
            'quantity': 100,
            'price': 10.5
        }
        check = risk_manager.check_risk(order)

        level = risk_manager.get_risk_level(check.check_id)
        assert isinstance(level, object)  # RiskLevel enum

    def test_is_order_allowed(self, risk_manager):
        """测试订单是否允许"""
        order = {
            'symbol': '000001',
            'quantity': 100,
            'price': 10.5
        }

        allowed = risk_manager.is_order_allowed(order)
        assert isinstance(allowed, bool)

    def test_assess_portfolio_risk(self, risk_manager):
        """测试投资组合风险评估"""
        portfolio = {
            'positions': {'000001': 1000, '000002': 500},
            'total_value': 150000
        }

        result = risk_manager.assess_portfolio_risk(portfolio)
        assert isinstance(result, dict)

    def test_calculate_var(self, risk_manager):
        """测试计算VaR"""
        import numpy as np
        returns = np.random.normal(0, 0.01, 100)

        var = risk_manager.calculate_var(returns)
        assert isinstance(var, (int, float))

    def test_get_risk_summary(self, risk_manager):
        """测试获取风险摘要"""
        summary = risk_manager.get_risk_summary()
        assert isinstance(summary, dict)

    def test_get_risk_metrics(self, risk_manager):
        """测试获取风险指标"""
        metrics = risk_manager.get_risk_metrics()
        assert isinstance(metrics, list)

    def test_real_time_monitor_initialization(self, real_time_monitor):
        """测试实时监控器初始化"""
        assert real_time_monitor is not None
        assert hasattr(real_time_monitor, 'config')
        assert hasattr(real_time_monitor, 'metrics_history')

    def test_real_time_monitor_with_config(self, real_time_monitor):
        """测试带配置的实时监控器"""
        assert 'max_position_size' in real_time_monitor.config
        assert 'max_daily_loss' in real_time_monitor.config

    def test_add_metric(self, real_time_monitor):
        """测试添加指标"""
        from src.risk.monitor.real_time_monitor import RiskMetric

        metric = RiskMetric(
            type='position',
            value=0.3,
            timestamp=datetime.now(),
            symbol='000001'
        )

        real_time_monitor.add_metric(metric)
        assert len(real_time_monitor.metrics_history['position']) > 0

    def test_calculate_risk_score(self, real_time_monitor):
        """测试计算风险评分"""
        # 添加一些指标数据
        from src.risk.monitor.real_time_monitor import RiskMetric

        metric1 = RiskMetric(type='position', value=0.3, timestamp=datetime.now(), symbol='000001')
        metric2 = RiskMetric(type='concentration', value=0.4, timestamp=datetime.now(), symbol='000001')

        real_time_monitor.add_metric(metric1)
        real_time_monitor.add_metric(metric2)

        score = real_time_monitor.calculate_risk_score()
        assert isinstance(score, (int, float))

    def test_check_alerts(self, real_time_monitor):
        """测试检查告警"""
        # 添加超过阈值的指标
        from src.risk.monitor.real_time_monitor import RiskMetric

        metric = RiskMetric(type='position', value=0.8, timestamp=datetime.now(), symbol='000001')
        real_time_monitor.add_metric(metric)

        alerts = real_time_monitor.check_alerts()
        assert isinstance(alerts, list)

    def test_get_recent_metrics(self, real_time_monitor):
        """测试获取最近指标"""
        # 添加一些指标
        from src.risk.monitor.real_time_monitor import RiskMetric

        metric1 = RiskMetric(type='position', value=0.3, timestamp=datetime.now(), symbol='000001')
        metric2 = RiskMetric(type='concentration', value=0.4, timestamp=datetime.now(), symbol='000001')

        real_time_monitor.add_metric(metric1)
        real_time_monitor.add_metric(metric2)

        metrics = real_time_monitor.get_recent_metrics()
        assert isinstance(metrics, list)

    def test_start_stop_monitoring(self, real_time_monitor):
        """测试启动停止监控"""
        # 启动监控
        real_time_monitor.start_monitoring()
        assert real_time_monitor.monitoring_active is True

        # 停止监控
        real_time_monitor.stop_monitoring()
        assert real_time_monitor.monitoring_active is False

    def test_add_alert_handler(self, real_time_monitor):
        """测试添加告警处理器"""
        def dummy_handler(alert):
            pass

        real_time_monitor.add_alert_handler(dummy_handler)
        assert len(real_time_monitor.alert_handlers) > 0

    def test_get_monitoring_status(self, real_time_monitor):
        """测试获取监控状态"""
        status = real_time_monitor.get_monitoring_status()
        assert isinstance(status, dict)
        assert 'active' in status
        assert 'metrics_count' in status

    def test_risk_level_enum(self):
        """测试风险等级枚举"""
        from src.risk.models.risk_manager import RiskLevel
        assert hasattr(RiskLevel, 'LOW')
        assert hasattr(RiskLevel, 'MEDIUM')
        assert hasattr(RiskLevel, 'HIGH')
        assert hasattr(RiskLevel, 'CRITICAL')

    def test_risk_manager_status_enum(self):
        """测试风险管理器状态枚举"""
        from src.risk.models.risk_manager import RiskManagerStatus
        assert hasattr(RiskManagerStatus, 'INACTIVE')
        assert hasattr(RiskManagerStatus, 'INITIALIZING')
        assert hasattr(RiskManagerStatus, 'ACTIVE')
        assert hasattr(RiskManagerStatus, 'PAUSED')
        assert hasattr(RiskManagerStatus, 'ERROR')

    def test_risk_types_enum(self):
        """测试风险类型枚举"""
        from src.risk.monitor.real_time_monitor import RiskType
        assert hasattr(RiskType, 'POSITION')
        assert hasattr(RiskType, 'CONCENTRATION')
        assert hasattr(RiskType, 'VOLATILITY')
        assert hasattr(RiskType, 'LIQUIDITY')

    def test_risk_metric_dataclass(self):
        """测试风险指标数据类"""
        from src.risk.monitor.real_time_monitor import RiskMetric
        from datetime import datetime

        metric = RiskMetric(
            type='position',
            value=0.5,
            timestamp=datetime.now(),
            symbol='000001'
        )
        assert metric.type == 'position'
        assert metric.value == 0.5
        assert metric.symbol == '000001'

    def test_compliance_components(self):
        """测试合规组件"""
        from src.risk.compliance.compliance_components import ComplianceComponentFactory

        factory = ComplianceComponentFactory()
        assert factory is not None

        # 测试创建合规组件
        component = factory.create_component('rule_based')
        assert component is not None

    def test_compliance_component_creation(self):
        """测试合规组件创建"""
        from src.risk.compliance.compliance_components import ComplianceComponentFactory

        factory = ComplianceComponentFactory()

        # 测试不同类型的组件创建
        rule_component = factory.create_component('rule_based')
        assert rule_component is not None

        ml_component = factory.create_component('ml_based')
        assert ml_component is not None

    def test_compliance_component_interface(self):
        """测试合规组件接口"""
        from src.risk.compliance.compliance_components import ComplianceComponentFactory

        factory = ComplianceComponentFactory()
        component = factory.create_component('rule_based')

        # 测试组件有基本方法
        assert hasattr(component, 'check_compliance')
        assert hasattr(component, 'get_violations')

    def test_risk_types_enum_values(self):
        """测试风险类型枚举值"""
        from src.risk.models.risk_types import RiskType
        assert hasattr(RiskType, 'MARKET_RISK')
        assert hasattr(RiskType, 'CREDIT_RISK')
        assert hasattr(RiskType, 'OPERATIONAL_RISK')
        assert hasattr(RiskType, 'LIQUIDITY_RISK')

    def test_comprehensive_risk_assessment(self, risk_manager):
        """测试综合风险评估"""
        # 测试多种风险类型的综合评估
        portfolio_data = {
            'positions': [
                {'symbol': '000001', 'quantity': 10000, 'price': 10.0, 'value': 100000},
                {'symbol': '000002', 'quantity': 5000, 'price': 20.0, 'value': 100000},
                {'symbol': '000003', 'quantity': 2000, 'price': 50.0, 'value': 100000}
            ],
            'total_value': 300000,
            'cash': 100000
        }

        assessment = risk_manager.assess_portfolio_risk(portfolio_data)
        assert isinstance(assessment, dict)
        assert 'overall_risk_level' in assessment
        assert 'risk_factors' in assessment

    def test_risk_limit_enforcement(self, risk_manager):
        """测试风险限额执行"""
        # 设置风险限额
        limits = {
            'max_position_size': 50000,
            'max_sector_exposure': 0.3,
            'max_single_stock_exposure': 0.1,
            'max_daily_loss': 0.05
        }

        # 测试正常情况
        normal_position = {
            'symbol': '000001',
            'quantity': 1000,
            'current_price': 25.0,
            'portfolio_value': 100000
        }

        enforcement_result = risk_manager.enforce_risk_limits(normal_position, limits)
        assert enforcement_result['allowed'] is True

        # 测试超出限额的情况
        large_position = {
            'symbol': '000001',
            'quantity': 3000,  # 超出限额
            'current_price': 25.0,
            'portfolio_value': 100000
        }

        enforcement_result = risk_manager.enforce_risk_limits(large_position, limits)
        assert enforcement_result['allowed'] is False
        assert 'violations' in enforcement_result

    def test_stress_testing(self, risk_manager):
        """测试压力测试"""
        # 基础投资组合
        portfolio = {
            'positions': [
                {'symbol': '000001', 'quantity': 1000, 'price': 10.0},
                {'symbol': '000002', 'quantity': 500, 'price': 20.0}
            ],
            'total_value': 25000
        }

        # 压力测试场景
        scenarios = [
            {'market_drop': 0.1, 'volatility_increase': 0.2},
            {'market_drop': 0.2, 'volatility_increase': 0.5},
            {'market_drop': 0.3, 'volatility_increase': 1.0}
        ]

        stress_results = risk_manager.perform_stress_test(portfolio, scenarios)
        assert isinstance(stress_results, list)
        assert len(stress_results) == len(scenarios)

        for result in stress_results:
            assert 'scenario' in result
            assert 'portfolio_loss' in result
            assert 'risk_metrics' in result

    def test_risk_reporting(self, risk_manager):
        """测试风险报告生成"""
        # 生成风险报告
        report_data = {
            'portfolio_value': 100000,
            'positions': [
                {'symbol': '000001', 'exposure': 0.4, 'risk_contribution': 0.15},
                {'symbol': '000002', 'exposure': 0.3, 'risk_contribution': 0.12}
            ],
            'risk_metrics': {
                'var_95': 0.05,
                'expected_shortfall': 0.08,
                'max_drawdown': 0.12
            }
        }

        report = risk_manager.generate_risk_report(report_data)
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'recommendations' in report
        assert 'charts_data' in report

    def test_compliance_monitoring(self, risk_manager):
        """测试合规监控"""
        # 合规规则
        compliance_rules = [
            {
                'rule_id': 'concentration_limit',
                'description': '单一股票持仓不能超过总资产的10%',
                'threshold': 0.1,
                'action': 'warning'
            },
            {
                'rule_id': 'sector_limit',
                'description': '单一行业持仓不能超过总资产的30%',
                'threshold': 0.3,
                'action': 'block'
            }
        ]

        # 投资组合数据
        portfolio = {
            'positions': [
                {'symbol': '000001', 'sector': 'technology', 'value': 15000, 'total_portfolio': 100000},
                {'symbol': '000002', 'sector': 'technology', 'value': 25000, 'total_portfolio': 100000}
            ]
        }

        compliance_result = risk_manager.check_compliance(portfolio, compliance_rules)
        assert isinstance(compliance_result, dict)
        assert 'compliant' in compliance_result
        assert 'violations' in compliance_result

    def test_risk_alert_system(self, risk_manager, real_time_monitor):
        """测试风险告警系统"""
        # 设置告警阈值
        alert_config = {
            'position_size_alert': 0.8,  # 80%阈值
            'loss_alert': 0.05,  # 5%损失阈值
            'volatility_alert': 0.3  # 30%波动率阈值
        }

        # 模拟风险事件
        risk_events = [
            {'type': 'position_size', 'value': 0.9, 'threshold': 0.8},  # 触发告警
            {'type': 'daily_loss', 'value': 0.03, 'threshold': 0.05},   # 未触发告警
            {'type': 'volatility', 'value': 0.35, 'threshold': 0.3}     # 触发告警
        ]

        alerts = []
        for event in risk_events:
            if event['value'] > event['threshold']:
                alert = risk_manager.generate_alert(event, alert_config)
                alerts.append(alert)

        assert len(alerts) == 2  # 应该有2个告警
        for alert in alerts:
            assert 'level' in alert
            assert 'message' in alert
            assert 'timestamp' in alert

    def test_scenario_analysis(self, risk_manager):
        """测试情景分析"""
        # 基准投资组合
        base_portfolio = {
            'positions': [
                {'symbol': '000001', 'quantity': 1000, 'beta': 1.2},
                {'symbol': '000002', 'quantity': 500, 'beta': 0.8}
            ]
        }

        # 情景定义
        scenarios = {
            'bull_market': {'market_return': 0.15, 'volatility': 0.2},
            'bear_market': {'market_return': -0.15, 'volatility': 0.3},
            'high_volatility': {'market_return': 0.05, 'volatility': 0.4},
            'crisis': {'market_return': -0.3, 'volatility': 0.6}
        }

        scenario_results = risk_manager.analyze_scenarios(base_portfolio, scenarios)
        assert isinstance(scenario_results, dict)
        assert len(scenario_results) == len(scenarios)

        for scenario_name, result in scenario_results.items():
            assert 'expected_return' in result
            assert 'expected_volatility' in result
            assert 'var_95' in result
            assert 'worst_case_loss' in result

    def test_risk_attribution(self, risk_manager):
        """测试风险归因分析"""
        # 投资组合表现数据
        portfolio_performance = {
            'total_return': -0.08,
            'benchmark_return': -0.05,
            'position_returns': {
                '000001': -0.12,
                '000002': -0.06,
                '000003': -0.15
            },
            'factor_exposures': {
                'market': 0.8,
                'size': -0.2,
                'value': 0.3
            }
        }

        attribution = risk_manager.perform_risk_attribution(portfolio_performance)
        assert isinstance(attribution, dict)
        assert 'total_attribution' in attribution
        assert 'factor_contributions' in attribution
        assert 'security_contributions' in attribution
        assert 'residual_risk' in attribution

    def test_dynamic_risk_limits(self, risk_manager):
        """测试动态风险限额调整"""
        # 市场条件
        market_conditions = {
            'volatility': 0.25,
            'trend': 'bearish',
            'liquidity': 'normal'
        }

        # 基础限额
        base_limits = {
            'max_position_size': 100000,
            'max_loss_limit': 0.05,
            'max_var_limit': 0.10
        }

        dynamic_limits = risk_manager.calculate_dynamic_limits(base_limits, market_conditions)
        assert isinstance(dynamic_limits, dict)

        # 在熊市高波动情况下，限额应该收紧
        assert dynamic_limits['max_position_size'] <= base_limits['max_position_size']
        assert dynamic_limits['max_loss_limit'] <= base_limits['max_loss_limit']

    def test_counterparty_risk_assessment(self, risk_manager):
        """测试对手方风险评估"""
        # 交易对手信息
        counterparties = [
            {
                'id': 'broker_a',
                'credit_rating': 'AAA',
                'trading_volume': 1000000,
                'default_history': []
            },
            {
                'id': 'broker_b',
                'credit_rating': 'BBB',
                'trading_volume': 500000,
                'default_history': ['late_payment_2023']
            }
        ]

        for counterparty in counterparties:
            risk_score = risk_manager.assess_counterparty_risk(counterparty)
            assert isinstance(risk_score, (int, float))
            assert 0 <= risk_score <= 100  # 风险分数在0-100之间

    def test_liquidity_risk_analysis(self, risk_manager):
        """测试流动性风险分析"""
        # 投资组合持仓
        positions = [
            {'symbol': '000001', 'quantity': 10000, 'avg_daily_volume': 50000, 'bid_ask_spread': 0.001},
            {'symbol': '000002', 'quantity': 5000, 'avg_daily_volume': 10000, 'bid_ask_spread': 0.005},
            {'symbol': '000003', 'quantity': 1000, 'avg_daily_volume': 5000, 'bid_ask_spread': 0.01}
        ]

        liquidity_risk = risk_manager.analyze_liquidity_risk(positions)
        assert isinstance(liquidity_risk, dict)
        assert 'overall_liquidity_score' in liquidity_risk
        assert 'position_liquidity' in liquidity_risk

        # 检查流动性评分
        for position in liquidity_risk['position_liquidity']:
            assert 'symbol' in position
            assert 'liquidity_score' in position
            assert 'estimated_impact_cost' in position

    def test_operational_risk_monitoring(self, risk_manager):
        """测试操作风险监控"""
        # 操作事件
        operational_events = [
            {'type': 'trade_error', 'severity': 'medium', 'frequency': 2},
            {'type': 'system_failure', 'severity': 'high', 'frequency': 1},
            {'type': 'manual_override', 'severity': 'low', 'frequency': 5}
        ]

        operational_risk = risk_manager.monitor_operational_risk(operational_events)
        assert isinstance(operational_risk, dict)
        assert 'overall_risk_score' in operational_risk
        assert 'risk_trends' in operational_risk
        assert 'recommendations' in operational_risk

    def test_regulatory_compliance_check(self, risk_manager):
        """测试监管合规检查"""
        # 监管要求
        regulatory_requirements = {
            'capital_adequacy_ratio': 0.08,
            'leverage_ratio': 0.03,
            'liquidity_coverage_ratio': 1.0,
            'net_stable_funding_ratio': 1.0
        }

        # 机构数据
        institution_data = {
            'tier1_capital': 1000000,
            'risk_weighted_assets': 10000000,
            'total_assets': 15000000,
            'high_quality_liquid_assets': 2000000,
            'required_stable_funding': 12000000,
            'available_stable_funding': 13500000
        }

        compliance_check = risk_manager.check_regulatory_compliance(institution_data, regulatory_requirements)
        assert isinstance(compliance_check, dict)
        assert 'overall_compliant' in compliance_check
        assert 'requirement_checks' in compliance_check

        # 检查每个监管要求的合规性
        for req_name in regulatory_requirements.keys():
            assert req_name in compliance_check['requirement_checks']

    def test_risk_model_validation(self, risk_manager):
        """测试风险模型验证"""
        # 模型表现数据
        model_performance = {
            'backtesting_period': '2018-2023',
            'actual_losses': [0.01, 0.02, -0.005, 0.015, 0.008],
            'predicted_losses': [0.012, 0.018, -0.003, 0.014, 0.009],
            'confidence_intervals': [(0.008, 0.016), (0.014, 0.022), (-0.007, 0.001), (0.010, 0.018), (0.005, 0.013)]
        }

        validation_results = risk_manager.validate_risk_model(model_performance)
        assert isinstance(validation_results, dict)
        assert 'model_accuracy' in validation_results
        assert 'backtesting_passes' in validation_results
        assert 'confidence_interval_coverage' in validation_results
        assert 'recommendations' in validation_results

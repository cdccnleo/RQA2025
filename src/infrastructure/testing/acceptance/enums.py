"""
用户验收测试枚举定义

Enums for user acceptance testing module.

Extracted from user_acceptance_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from enum import Enum


class UserRole(Enum):
    """用户角色枚举"""
    TRADER = "trader"                    # 交易员
    RISK_MANAGER = "risk_manager"        # 风险经理
    PORTFOLIO_MANAGER = "portfolio_manager"  # 投资组合经理
    QUANTITATIVE_ANALYST = "quantitative_analyst"  # 量化分析师
    SYSTEM_ADMINISTRATOR = "system_administrator"  # 系统管理员
    COMPLIANCE_OFFICER = "compliance_officer"  # 合规官
    BUSINESS_ANALYST = "business_analyst"  # 业务分析师


class AcceptanceTestType(Enum):
    """验收测试类型枚举"""
    BUSINESS_FUNCTIONALITY = "business_functionality"    # 业务功能测试
    USER_INTERFACE = "user_interface"                    # 用户界面测试
    PERFORMANCE_REQUIREMENTS = "performance_requirements"  # 性能需求测试
    SECURITY_REQUIREMENTS = "security_requirements"        # 安全需求测试
    COMPLIANCE_REQUIREMENTS = "compliance_requirements"    # 合规需求测试
    INTEGRATION_SCENARIOS = "integration_scenarios"        # 集成场景测试
    BUSINESS_WORKFLOW = "business_workflow"                # 业务流程测试
    DATA_ACCURACY = "data_accuracy"                        # 数据准确性测试


class TestScenario(Enum):
    """测试场景枚举"""
    MARKET_DATA_PROCESSING = "market_data_processing"      # 市场数据处理
    MODEL_TRAINING_DEPLOYMENT = "model_training_deployment"  # 模型训练部署
    RISK_MONITORING_ALERT = "risk_monitoring_alert"        # 风险监控告警
    AUTOMATED_TRADING_EXECUTION = "automated_trading_execution"  # 自动化交易执行
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"        # 投资组合再平衡
    REPORT_GENERATION = "report_generation"                # 报告生成
    SYSTEM_BACKUP_RECOVERY = "system_backup_recovery"      # 系统备份恢复
    COMPLIANCE_CHECKING = "compliance_checking"            # 合规检查


__all__ = [
    'UserRole',
    'AcceptanceTestType',
    'TestScenario'
]

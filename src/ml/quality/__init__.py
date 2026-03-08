"""
RQA2025 AI质量保障体系

全球领先的量化交易系统AI质量保障平台，提供从传统质量管理到AI智能化保障的完整解决方案。

核心功能:
- 异常预测与告警 (AnomalyPredictionEngine)
- 自动化测试生成 (AutomatedTestGenerator)
- 性能优化建议 (PerformanceAnalyzer)
- 质量趋势分析 (QualityTrendAnalyzer)
- 强化学习优化器 (ReinforcementLearningOptimizer)
- 跨系统相关性分析器 (CrossSystemCorrelationAnalyzer)
- 预测性维护引擎 (MaintenancePredictionEngine)
- 质量AI决策支持系统 (QualityAIDecisionSupportSystem)
- 生产环境集成 (ProductionIntegrationManager)
- 数据管理与基础设施 (DataPipelineManager)
- 模型运维与监控 (ModelOperationsManager)
- 用户接口与工具 (UserInterfaceManager)
- 连续学习与优化 (ContinuousLearningManager)

版本: 1.0.0
作者: RQA2025 AI质量保障团队
"""

__version__ = "1.0.0"
__author__ = "RQA2025 AI质量保障团队"

# 导出主要组件
from .anomaly_prediction import AnomalyPredictionEngine
from .quality_trend_analysis import QualityTrendAnalyzer
from .performance_optimization import PerformanceAnalyzer
from .decision_support_system import QualityAIDecisionSupportSystem
from .production_integration import ProductionIntegrationManager
from .data_management import DataPipelineManager
from .model_operations import ModelOperationsManager
from .user_interfaces import QualityDashboard as UserInterfaceManager
from .continuous_learning import LearningLoopManager as ContinuousLearningManager

__all__ = [
    'AnomalyPredictionEngine',
    'QualityTrendAnalyzer',
    'PerformanceAnalyzer',
    'QualityAIDecisionSupportSystem',
    'ProductionIntegrationManager',
    'DataPipelineManager',
    'ModelOperationsManager',
    'UserInterfaceManager',
    'ContinuousLearningManager'
]

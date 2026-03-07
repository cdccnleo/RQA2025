"""
风险控制层分析模块

提供市场冲击分析和多资产风险管理。
"""

from .market_impact_analyzer import MarketImpactAnalyzer

# 为测试兼容性提供别名
RiskAnalysis = MarketImpactAnalyzer

__all__ = ['MarketImpactAnalyzer', 'RiskAnalysis']
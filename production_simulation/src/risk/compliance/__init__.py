#!/usr/bin/env python3
"""
合规检误模块

合规检查的组件实现
"""

__version__ = "1.0.0"
__author__ = "RQA2025 Team"
__description__ = "合规检查组件"

try:
    from .cross_border_compliance_manager import (
        CrossBorderComplianceManager,
        ComplianceRule,
        ComplianceType,
        Country,
        Currency,
        ComplianceCheckResult,
        CrossBorderTransaction
    )
except ImportError:
    CrossBorderComplianceManager = None
    ComplianceRule = None
    ComplianceType = None
    Country = None
    Currency = None
    ComplianceCheckResult = None
    CrossBorderTransaction = None

__all__ = [
    'CrossBorderComplianceManager',
    'ComplianceRule',
    'ComplianceType',
    'Country',
    'Currency',
    'ComplianceCheckResult',
    'CrossBorderTransaction'
]

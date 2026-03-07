"""
RQA2025 基础设施层质量检查工具

自动化质量检查工具包，提供：
- 代码重复检测
- 接口一致性检查
- 代码复杂度监控
- CI/CD集成支持

作者: 专项修复小组
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "专项修复小组"

from .core.quality_checker import QualityChecker
from .core.check_result import CheckResult, Issue, IssueSeverity
from .checkers.duplicate_checker import DuplicateCodeChecker
from .checkers.interface_checker import InterfaceConsistencyChecker
from .checkers.complexity_checker import ComplexityChecker
from .reporters.console_reporter import ConsoleReporter
from .reporters.json_reporter import JsonReporter
from .reporters.html_reporter import HtmlReporter

__all__ = [
    # 核心组件
    'QualityChecker',
    'CheckResult',
    'Issue',
    'IssueSeverity',

    # 检查器
    'DuplicateCodeChecker',
    'InterfaceConsistencyChecker',
    'ComplexityChecker',

    # 报告器
    'ConsoleReporter',
    'JsonReporter',
    'HtmlReporter'
]

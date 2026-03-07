"""
质量检查结果数据结构

定义检查结果、问题和严重程度的数据模型。
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime


class IssueSeverity(Enum):
    """问题严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Issue:
    """
    检查发现的问题

    封装单个质量问题的详细信息。
    """

    def __init__(self,
                 file_path: str,
                 line_number: Optional[int],
                 message: str,
                 severity: IssueSeverity,
                 rule_id: str,
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化问题

        Args:
            file_path: 问题所在文件路径
            line_number: 问题所在行号
            message: 问题描述
            severity: 问题严重程度
            rule_id: 规则ID
            details: 额外详细信息
        """
        self.file_path = file_path
        self.line_number = line_number
        self.message = message
        self.severity = severity
        self.rule_id = rule_id
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'message': self.message,
            'severity': self.severity.value,
            'rule_id': self.rule_id,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """字符串表示"""
        line_info = f":{self.line_number}" if self.line_number else ""
        return f"[{self.severity.value.upper()}] {self.file_path}{line_info} - {self.message}"


class CheckResult:
    """
    检查结果

    汇总一次质量检查的所有结果。
    """

    def __init__(self, checker_name: str):
        """
        初始化检查结果

        Args:
            checker_name: 检查器名称
        """
        self.checker_name = checker_name
        self.issues: List[Issue] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def add_issue(self, issue: Issue) -> None:
        """添加问题"""
        self.issues.append(issue)

    def set_end_time(self) -> None:
        """设置结束时间"""
        self.end_time = datetime.now()

    def get_duration(self) -> float:
        """获取检查耗时（秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def get_issue_count(self, severity: Optional[IssueSeverity] = None) -> int:
        """获取问题数量"""
        if severity:
            return len([i for i in self.issues if i.severity == severity])
        return len(self.issues)

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总信息"""
        return {
            'checker_name': self.checker_name,
            'total_issues': len(self.issues),
            'issues_by_severity': {
                'info': self.get_issue_count(IssueSeverity.INFO),
                'warning': self.get_issue_count(IssueSeverity.WARNING),
                'error': self.get_issue_count(IssueSeverity.ERROR),
                'critical': self.get_issue_count(IssueSeverity.CRITICAL)
            },
            'duration_seconds': self.get_duration(),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata
        }

    def has_critical_issues(self) -> bool:
        """是否有严重问题"""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    def has_errors(self) -> bool:
        """是否有错误"""
        return any(i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL] for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'summary': self.get_summary(),
            'issues': [issue.to_dict() for issue in self.issues]
        }

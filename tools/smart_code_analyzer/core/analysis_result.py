#!/usr/bin/env python3
"""
分析结果数据结构

定义代码分析结果、质量指标和重构建议的数据模型。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class CodeMetrics:
    """代码度量指标"""

    # 基础指标
    lines_of_code: int = 0
    lines_of_comments: int = 0
    lines_of_blank: int = 0

    # 复杂度指标
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    nesting_depth: int = 0

    # 质量指标
    maintainability_index: float = 0.0
    halstead_volume: float = 0.0
    duplication_percentage: float = 0.0
    test_coverage: float = 0.0

    # 代码异味计数
    long_method_count: int = 0
    long_class_count: int = 0
    large_file_count: int = 0
    duplicate_code_blocks: int = 0
    unused_imports: int = 0
    unused_variables: int = 0
    magic_numbers: int = 0
    nested_conditionals: int = 0
    empty_catch_blocks: int = 0
    long_parameter_lists: int = 0

    # 结构指标
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0
    import_count: int = 0

    # 依赖指标
    afferent_coupling: int = 0  # 传入耦合
    efferent_coupling: int = 0  # 传出耦合
    instability: float = 0.0     # 不稳定性

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'lines_of_code': self.lines_of_code,
            'lines_of_comments': self.lines_of_comments,
            'lines_of_blank': self.lines_of_blank,
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'nesting_depth': self.nesting_depth,
            'maintainability_index': self.maintainability_index,
            'halstead_volume': self.halstead_volume,
            'duplication_percentage': self.duplication_percentage,
            'test_coverage': self.test_coverage,
            'long_method_count': self.long_method_count,
            'long_class_count': self.long_class_count,
            'large_file_count': self.large_file_count,
            'duplicate_code_blocks': self.duplicate_code_blocks,
            'unused_imports': self.unused_imports,
            'unused_variables': self.unused_variables,
            'magic_numbers': self.magic_numbers,
            'nested_conditionals': self.nested_conditionals,
            'empty_catch_blocks': self.empty_catch_blocks,
            'long_parameter_lists': self.long_parameter_lists,
            'class_count': self.class_count,
            'function_count': self.function_count,
            'method_count': self.method_count,
            'import_count': self.import_count,
            'afferent_coupling': self.afferent_coupling,
            'efferent_coupling': self.efferent_coupling,
            'instability': self.instability
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总信息"""
        total_smells = (
            self.long_method_count + self.long_class_count + self.large_file_count +
            self.duplicate_code_blocks + self.unused_imports + self.unused_variables +
            self.magic_numbers + self.nested_conditionals + self.empty_catch_blocks +
            self.long_parameter_lists
        )

        return {
            'total_lines': self.lines_of_code,
            'total_complexity': self.cyclomatic_complexity,
            'total_code_smells': total_smells,
            'maintainability_score': self.maintainability_index,
            'duplication_rate': self.duplication_percentage,
            'test_coverage': self.test_coverage,
            'quality_score': self.calculate_quality_score()
        }

    def calculate_quality_score(self) -> float:
        """计算质量评分"""
        score = 100.0

        # 复杂度惩罚
        if self.cyclomatic_complexity > 50:
            score -= 20
        elif self.cyclomatic_complexity > 20:
            score -= 10

        # 代码行数惩罚
        if self.lines_of_code > 1000:
            score -= 15
        elif self.lines_of_code > 500:
            score -= 8

        # 重复代码惩罚
        score -= self.duplication_percentage * 0.5

        # 可维护性奖励
        if self.maintainability_index > 80:
            score += 5
        elif self.maintainability_index < 50:
            score -= 10

        # 测试覆盖率奖励
        if self.test_coverage > 80:
            score += 5
        elif self.test_coverage < 50:
            score -= 10

        # 代码异味惩罚
        code_smell_penalty = (
            self.long_method_count * 2 +
            self.long_class_count * 5 +
            self.large_file_count * 10 +
            self.duplicate_code_blocks * 3 +
            self.unused_imports * 1 +
            self.unused_variables * 1 +
            self.magic_numbers * 0.5 +
            self.nested_conditionals * 2 +
            self.empty_catch_blocks * 3 +
            self.long_parameter_lists * 1
        )
        score -= min(code_smell_penalty, 30)  # 最多扣30分

        return max(0.0, min(100.0, score))


@dataclass
class RefactoringSuggestion:
    """重构建议"""

    file_path: str
    line_number: int
    suggestion_type: str  # 重构类型
    title: str            # 建议标题
    description: str      # 详细描述
    severity: str         # 'critical', 'high', 'medium', 'low', 'info'
    confidence: float     # 置信度 0.0-1.0
    estimated_effort: str  # 'low', 'medium', 'high'

    # 额外信息
    impact_score: float = 0.0  # 影响分数
    risk_level: str = 'low'     # 风险等级
    category: str = 'general'   # 分类

    # 建议内容
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    rationale: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)

    # 元数据
    analyzer: str = 'unknown'
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'suggestion_type': self.suggestion_type,
            'title': self.title,
            'description': self.description,
            'severity': self.severity,
            'confidence': self.confidence,
            'estimated_effort': self.estimated_effort,
            'impact_score': self.impact_score,
            'risk_level': self.risk_level,
            'category': self.category,
            'before_code': self.before_code,
            'after_code': self.after_code,
            'rationale': self.rationale,
            'alternatives': self.alternatives,
            'analyzer': self.analyzer,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringSuggestion':
        """从字典创建"""
        data_copy = data.copy()
        if 'timestamp' in data_copy:
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)

    def get_priority_score(self) -> float:
        """计算优先级分数"""
        severity_weights = {
            'critical': 100,
            'high': 75,
            'medium': 50,
            'low': 25,
            'info': 10
        }

        effort_weights = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4
        }

        severity_score = severity_weights.get(self.severity, 10)
        effort_score = effort_weights.get(self.estimated_effort, 0.7)
        confidence_score = self.confidence
        impact_score = self.impact_score

        return (severity_score * 0.4 + confidence_score * 100 * 0.3 +
                impact_score * 0.2 + effort_score * 25 * 0.1)


@dataclass
class AnalysisResult:
    """分析结果"""

    file_path: str
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    suggestions: List[RefactoringSuggestion] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # 分析元数据
    analysis_time: float = 0.0
    analyzer_version: str = '1.0.0'
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    # 质量评估
    quality_score: float = 0.0
    quality_grade: str = 'unknown'  # 'A', 'B', 'C', 'D', 'F'

    # 趋势数据
    trend_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        self.quality_score = self.metrics.calculate_quality_score()
        self.quality_grade = self._calculate_grade()

    def _calculate_grade(self) -> str:
        """计算质量等级"""
        score = self.quality_score
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def add_suggestion(self, suggestion: RefactoringSuggestion):
        """添加重构建议"""
        self.suggestions.append(suggestion)

    def add_issue(self, issue_type: str, description: str,
                  line_number: Optional[int] = None, severity: str = 'medium'):
        """添加问题"""
        self.issues.append({
            'type': issue_type,
            'description': description,
            'line_number': line_number,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })

    def add_dependency(self, dependency: str):
        """添加依赖"""
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_path': self.file_path,
            'metrics': self.metrics.to_dict(),
            'suggestions': [s.to_dict() for s in self.suggestions],
            'issues': self.issues,
            'dependencies': self.dependencies,
            'analysis_time': self.analysis_time,
            'analyzer_version': self.analyzer_version,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'quality_score': self.quality_score,
            'quality_grade': self.quality_grade,
            'trend_data': self.trend_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """从字典创建"""
        data_copy = data.copy()

        # 转换嵌套对象
        if 'metrics' in data_copy:
            data_copy['metrics'] = CodeMetrics(**data_copy['metrics'])

        if 'suggestions' in data_copy:
            data_copy['suggestions'] = [
                RefactoringSuggestion.from_dict(s) for s in data_copy['suggestions']
            ]

        if 'analysis_timestamp' in data_copy:
            data_copy['analysis_timestamp'] = datetime.fromisoformat(
                data_copy['analysis_timestamp']
            )

        return cls(**data_copy)

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总信息"""
        return {
            'file_path': self.file_path,
            'quality_score': self.quality_score,
            'quality_grade': self.quality_grade,
            'total_suggestions': len(self.suggestions),
            'total_issues': len(self.issues),
            'total_dependencies': len(self.dependencies),
            'metrics_summary': self.metrics.get_summary(),
            'suggestions_by_severity': self._count_suggestions_by_severity(),
            'issues_by_severity': self._count_issues_by_severity()
        }

    def _count_suggestions_by_severity(self) -> Dict[str, int]:
        """按严重程度统计建议"""
        counts = {}
        for suggestion in self.suggestions:
            counts[suggestion.severity] = counts.get(suggestion.severity, 0) + 1
        return counts

    def _count_issues_by_severity(self) -> Dict[str, int]:
        """按严重程度统计问题"""
        counts = {}
        for issue in self.issues:
            severity = issue.get('severity', 'medium')
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def get_top_suggestions(self, limit: int = 10) -> List[RefactoringSuggestion]:
        """获取优先级最高的建议"""
        sorted_suggestions = sorted(
            self.suggestions,
            key=lambda s: s.get_priority_score(),
            reverse=True
        )
        return sorted_suggestions[:limit]

    def filter_suggestions(self, min_severity: str = 'low',
                           min_confidence: float = 0.0) -> List[RefactoringSuggestion]:
        """过滤建议"""
        severity_levels = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
        min_severity_level = severity_levels.get(min_severity, 0)

        filtered = []
        for suggestion in self.suggestions:
            severity_level = severity_levels.get(suggestion.severity, 0)
            if (severity_level >= min_severity_level and
                    suggestion.confidence >= min_confidence):
                filtered.append(suggestion)

        return filtered

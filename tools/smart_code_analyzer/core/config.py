#!/usr/bin/env python3
"""
智能代码分析器配置管理

提供灵活的配置选项，支持多种预设配置和自定义设置。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class AnalysisThresholds:
    """分析阈值配置"""

    # 质量评分阈值
    quality_score_min: float = 80.0
    quality_score_good: float = 85.0
    quality_score_excellent: float = 90.0

    # 复杂度阈值
    max_cyclomatic_complexity: int = 10
    max_cognitive_complexity: int = 15
    max_nesting_depth: int = 3

    # 大小阈值
    max_lines_per_function: int = 50
    max_lines_per_class: int = 300
    max_lines_per_file: int = 1000

    # 重复度阈值
    max_duplication_percentage: float = 5.0
    min_clone_similarity: float = 0.8

    # 其他阈值
    max_parameters_per_function: int = 5
    min_test_coverage: float = 80.0
    max_coupling_degree: int = 10


@dataclass
class AnalysisOptions:
    """分析选项配置"""

    # 基本选项
    include_test_files: bool = False
    include_generated_files: bool = False
    follow_symlinks: bool = False

    # 深度分析
    enable_deep_analysis: bool = True
    enable_ast_analysis: bool = True
    enable_dependency_analysis: bool = True
    enable_pattern_recognition: bool = True

    # 性能选项
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    incremental_analysis: bool = True

    # 详细程度
    verbose_output: bool = False
    include_raw_data: bool = False
    debug_mode: bool = False


@dataclass
class RefactoringOptions:
    """重构选项配置"""

    # 建议生成
    max_suggestions: int = 50
    min_confidence: float = 0.7
    risk_tolerance: str = 'medium'  # 'low', 'medium', 'high'

    # 优先级排序
    prioritize_by_impact: bool = True
    prioritize_by_effort: bool = False
    include_quick_wins: bool = True

    # 自动修复
    enable_auto_fix: bool = False
    auto_fix_safe_only: bool = True
    backup_before_fix: bool = True


@dataclass
class ReportOptions:
    """报告选项配置"""

    # 输出格式
    format: str = 'html'  # 'html', 'json', 'xml', 'markdown'
    include_charts: bool = True
    include_trends: bool = True
    include_raw_data: bool = False

    # 报告内容
    include_summary: bool = True
    include_details: bool = True
    include_recommendations: bool = True
    include_metrics: bool = True

    # 输出路径
    output_dir: Optional[str] = None
    output_file: Optional[str] = None
    overwrite_existing: bool = True


@dataclass
class SmartAnalysisConfig:
    """智能代码分析器主配置"""

    # 基础配置
    name: str = "Smart Code Analyzer"
    version: str = "1.0.0"

    # 子配置
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
    analysis: AnalysisOptions = field(default_factory=AnalysisOptions)
    refactoring: RefactoringOptions = field(default_factory=RefactoringOptions)
    reporting: ReportOptions = field(default_factory=ReportOptions)

    # 高级配置
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '__pycache__', '*.pyc', '.git', '.svn', 'node_modules',
        '*.egg-info', '.tox', '.coverage', 'htmlcov'
    ])
    include_patterns: List[str] = field(default_factory=lambda: ['*.py'])

    # 集成配置
    integrate_with_quality_check: bool = True
    integrate_with_duplicate_detector: bool = True
    integrate_with_test_coverage: bool = True

    def __post_init__(self):
        """后初始化处理"""
        # 确保配置的一致性
        self._validate_config()

    def _validate_config(self):
        """验证配置有效性"""
        # 验证阈值范围
        assert 0 <= self.thresholds.quality_score_min <= 100
        assert 0 <= self.thresholds.quality_score_good <= 100
        assert 0 <= self.thresholds.quality_score_excellent <= 100

        # 验证复杂度阈值
        assert self.thresholds.max_cyclomatic_complexity > 0
        assert self.thresholds.max_cognitive_complexity > 0

        # 验证文件大小阈值
        assert self.thresholds.max_lines_per_function > 0
        assert self.thresholds.max_lines_per_class > 0
        assert self.thresholds.max_lines_per_file > 0

        # 验证重复度阈值
        assert 0 <= self.thresholds.max_duplication_percentage <= 100
        assert 0 <= self.thresholds.min_clone_similarity <= 1

        # 验证其他阈值
        assert self.thresholds.max_parameters_per_function > 0
        assert 0 <= self.thresholds.min_test_coverage <= 100

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SmartAnalysisConfig':
        """从字典创建配置"""
        # 递归创建子配置对象
        thresholds = AnalysisThresholds(**config_dict.get('thresholds', {}))
        analysis = AnalysisOptions(**config_dict.get('analysis', {}))
        refactoring = RefactoringOptions(**config_dict.get('refactoring', {}))
        reporting = ReportOptions(**config_dict.get('reporting', {}))

        return cls(
            name=config_dict.get('name', cls.name),
            version=config_dict.get('version', cls.version),
            thresholds=thresholds,
            analysis=analysis,
            refactoring=refactoring,
            reporting=reporting,
            custom_rules=config_dict.get('custom_rules', {}),
            exclude_patterns=config_dict.get('exclude_patterns', cls.exclude_patterns),
            include_patterns=config_dict.get('include_patterns', cls.include_patterns),
            integrate_with_quality_check=config_dict.get('integrate_with_quality_check', True),
            integrate_with_duplicate_detector=config_dict.get(
                'integrate_with_duplicate_detector', True),
            integrate_with_test_coverage=config_dict.get('integrate_with_test_coverage', True)
        )

    @classmethod
    def from_file(cls, config_file: str) -> 'SmartAnalysisConfig':
        """从文件加载配置"""
        import json

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'thresholds': {
                'quality_score_min': self.thresholds.quality_score_min,
                'quality_score_good': self.thresholds.quality_score_good,
                'quality_score_excellent': self.thresholds.quality_score_excellent,
                'max_cyclomatic_complexity': self.thresholds.max_cyclomatic_complexity,
                'max_cognitive_complexity': self.thresholds.max_cognitive_complexity,
                'max_nesting_depth': self.thresholds.max_nesting_depth,
                'max_lines_per_function': self.thresholds.max_lines_per_function,
                'max_lines_per_class': self.thresholds.max_lines_per_class,
                'max_lines_per_file': self.thresholds.max_lines_per_file,
                'max_duplication_percentage': self.thresholds.max_duplication_percentage,
                'min_clone_similarity': self.thresholds.min_clone_similarity,
                'max_parameters_per_function': self.thresholds.max_parameters_per_function,
                'min_test_coverage': self.thresholds.min_test_coverage,
                'max_coupling_degree': self.thresholds.max_coupling_degree
            },
            'analysis': {
                'include_test_files': self.analysis.include_test_files,
                'include_generated_files': self.analysis.include_generated_files,
                'follow_symlinks': self.analysis.follow_symlinks,
                'enable_deep_analysis': self.analysis.enable_deep_analysis,
                'enable_ast_analysis': self.analysis.enable_ast_analysis,
                'enable_dependency_analysis': self.analysis.enable_dependency_analysis,
                'enable_pattern_recognition': self.analysis.enable_pattern_recognition,
                'parallel_processing': self.analysis.parallel_processing,
                'max_workers': self.analysis.max_workers,
                'cache_results': self.analysis.cache_results,
                'incremental_analysis': self.analysis.incremental_analysis,
                'verbose_output': self.analysis.verbose_output,
                'include_raw_data': self.analysis.include_raw_data,
                'debug_mode': self.analysis.debug_mode
            },
            'refactoring': {
                'max_suggestions': self.refactoring.max_suggestions,
                'min_confidence': self.refactoring.min_confidence,
                'risk_tolerance': self.refactoring.risk_tolerance,
                'prioritize_by_impact': self.refactoring.prioritize_by_impact,
                'prioritize_by_effort': self.refactoring.prioritize_by_effort,
                'include_quick_wins': self.refactoring.include_quick_wins,
                'enable_auto_fix': self.refactoring.enable_auto_fix,
                'auto_fix_safe_only': self.refactoring.auto_fix_safe_only,
                'backup_before_fix': self.refactoring.backup_before_fix
            },
            'reporting': {
                'format': self.reporting.format,
                'include_charts': self.reporting.include_charts,
                'include_trends': self.reporting.include_trends,
                'include_raw_data': self.reporting.include_raw_data,
                'include_summary': self.reporting.include_summary,
                'include_details': self.reporting.include_details,
                'include_recommendations': self.reporting.include_recommendations,
                'include_metrics': self.reporting.include_metrics,
                'output_dir': self.reporting.output_dir,
                'output_file': self.reporting.output_file,
                'overwrite_existing': self.reporting.overwrite_existing
            },
            'custom_rules': self.custom_rules,
            'exclude_patterns': self.exclude_patterns,
            'include_patterns': self.include_patterns,
            'integrate_with_quality_check': self.integrate_with_quality_check,
            'integrate_with_duplicate_detector': self.integrate_with_duplicate_detector,
            'integrate_with_test_coverage': self.integrate_with_test_coverage
        }

    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        import json

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def get_preset_config(self, preset: str) -> 'SmartAnalysisConfig':
        """获取预设配置"""

        if preset == 'strict':
            # 严格模式：最高质量标准
            self.thresholds.quality_score_min = 90.0
            self.thresholds.max_cyclomatic_complexity = 8
            self.thresholds.max_lines_per_function = 30
            self.thresholds.max_duplication_percentage = 2.0
            self.analysis.enable_deep_analysis = True
            self.refactoring.risk_tolerance = 'low'

        elif preset == 'normal':
            # 正常模式：平衡质量和效率
            self.thresholds.quality_score_min = 80.0
            self.thresholds.max_cyclomatic_complexity = 10
            self.thresholds.max_lines_per_function = 50
            self.thresholds.max_duplication_percentage = 5.0
            self.analysis.enable_deep_analysis = True
            self.refactoring.risk_tolerance = 'medium'

        elif preset == 'relaxed':
            # 宽松模式：降低标准，提高覆盖率
            self.thresholds.quality_score_min = 70.0
            self.thresholds.max_cyclomatic_complexity = 15
            self.thresholds.max_lines_per_function = 70
            self.thresholds.max_duplication_percentage = 10.0
            self.analysis.enable_deep_analysis = False
            self.refactoring.risk_tolerance = 'high'

        elif preset == 'performance':
            # 性能模式：快速分析，降低准确性
            self.analysis.parallel_processing = True
            self.analysis.max_workers = 8
            self.analysis.enable_deep_analysis = False
            self.analysis.enable_dependency_analysis = False
            self.analysis.enable_pattern_recognition = False
            self.refactoring.max_suggestions = 20

        elif preset == 'ci':
            # CI模式：适合持续集成
            self.analysis.parallel_processing = True
            self.analysis.cache_results = True
            self.analysis.incremental_analysis = True
            self.reporting.format = 'json'
            self.reporting.include_raw_data = False

        else:
            raise ValueError(f"未知的预设配置: {preset}")

        return self

    def merge_config(self, other: 'SmartAnalysisConfig'):
        """合并配置"""
        # 深度合并逻辑，这里简化为基本合并
        # 实际实现应该递归合并所有字段

    def validate_project_path(self, project_path: str) -> bool:
        """验证项目路径"""
        path = Path(project_path)
        return path.exists() and path.is_dir()

    def get_analysis_scope(self, project_path: str) -> List[Path]:
        """获取分析范围内的文件"""
        project_path = Path(project_path)

        all_files = []
        for pattern in self.include_patterns:
            all_files.extend(project_path.rglob(pattern))

        # 排除不需要的文件
        filtered_files = []
        for file_path in all_files:
            if not any(ex_pattern in str(file_path) for ex_pattern in self.exclude_patterns):
                filtered_files.append(file_path)

        return filtered_files

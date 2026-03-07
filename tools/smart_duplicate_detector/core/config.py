"""
智能重复代码检测配置

提供灵活的配置选项，支持不同场景的检测需求。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class SimilarityThresholds:
    """相似度阈值配置"""

    exact_clone: float = 0.95      # 完全克隆阈值
    similar_clone: float = 0.8     # 相似克隆阈值
    semantic_clone: float = 0.6    # 语义克隆阈值
    weak_clone: float = 0.4        # 弱克隆阈值

    # 各维度相似度权重
    text_weight: float = 0.15
    token_weight: float = 0.25      # 新增token相似度权重
    ast_weight: float = 0.3
    semantic_weight: float = 0.25
    normalized_weight: float = 0.05


@dataclass
class AnalysisConfig:
    """分析配置"""

    min_fragment_size: int = 5      # 最小的代码片段行数
    max_fragment_size: int = 100    # 最大的代码片段行数

    # 代码片段提取选项
    extract_functions: bool = True
    extract_classes: bool = True
    extract_methods: bool = True
    extract_blocks: bool = False
    extract_statements: bool = False

    # 分析选项
    ignore_imports: bool = True
    ignore_comments: bool = True
    ignore_docstrings: bool = True
    normalize_variables: bool = True
    analyze_ast: bool = True
    complexity_analysis: bool = False  # 是否启用复杂度分析


@dataclass
class PerformanceConfig:
    """性能配置"""

    max_files_to_analyze: int = 1000
    max_fragments_per_file: int = 100
    similarity_cache_size: int = 10000
    parallel_processing: bool = True
    max_workers: int = 4

    # 超时设置
    analysis_timeout: int = 300  # 秒
    similarity_timeout: int = 60  # 秒


@dataclass
class RefactoringConfig:
    """重构配置"""

    enable_suggestions: bool = True
    min_group_size_for_refactor: int = 2
    suggest_method_extraction: bool = True
    suggest_class_extraction: bool = True
    suggest_utility_creation: bool = True
    suggest_config_extraction: bool = True

    # 自动修复选项（未来扩展）
    enable_auto_fix: bool = False
    auto_fix_min_confidence: float = 0.9


@dataclass
class SmartDuplicateConfig:
    """
    智能重复代码检测配置

    统一配置类，包含所有检测选项。
    """

    # 子配置
    similarity: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    refactoring: RefactoringConfig = field(default_factory=RefactoringConfig)

    # 输出配置
    output_format: str = "json"  # json, xml, html
    output_file: Optional[str] = None
    include_details: bool = True
    include_source_code: bool = False

    # 过滤配置
    file_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    exclude_dirs: List[str] = field(default_factory=lambda: ["__pycache__", ".git", "node_modules"])

    def __post_init__(self):
        """验证配置"""
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置参数"""
        # 验证相似度权重
        total_weight = (self.similarity.text_weight +
                        self.similarity.token_weight +
                        self.similarity.ast_weight +
                        self.similarity.semantic_weight +
                        self.similarity.normalized_weight)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"相似度权重总和必须为1.0，当前为{total_weight}")

        # 验证阈值范围
        thresholds = [
            self.similarity.exact_clone,
            self.similarity.similar_clone,
            self.similarity.semantic_clone,
            self.similarity.weak_clone
        ]
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"相似度阈值必须在0.0-1.0之间，当前为{threshold}")

        # 验证大小限制
        if self.analysis.min_fragment_size >= self.analysis.max_fragment_size:
            raise ValueError("最小片段大小不能大于等于最大片段大小")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SmartDuplicateConfig':
        """从字典创建配置"""
        # 递归创建子配置对象
        def create_subconfig(sub_dict: Dict[str, Any], sub_class):
            if not sub_dict:
                return sub_class()
            return sub_class(**sub_dict)

        similarity = create_subconfig(config_dict.get('similarity', {}), SimilarityThresholds)
        analysis = create_subconfig(config_dict.get('analysis', {}), AnalysisConfig)
        performance = create_subconfig(config_dict.get('performance', {}), PerformanceConfig)
        refactoring = create_subconfig(config_dict.get('refactoring', {}), RefactoringConfig)

        # 创建主配置
        config = cls(
            similarity=similarity,
            analysis=analysis,
            performance=performance,
            refactoring=refactoring
        )

        # 设置其他属性
        for key, value in config_dict.items():
            if key not in ['similarity', 'analysis', 'performance', 'refactoring']:
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'similarity': {
                'exact_clone': self.similarity.exact_clone,
                'similar_clone': self.similarity.similar_clone,
                'semantic_clone': self.similarity.semantic_clone,
                'weak_clone': self.similarity.weak_clone,
                'text_weight': self.similarity.text_weight,
                'ast_weight': self.similarity.ast_weight,
                'semantic_weight': self.similarity.semantic_weight,
                'normalized_weight': self.similarity.normalized_weight,
            },
            'analysis': {
                'min_fragment_size': self.analysis.min_fragment_size,
                'max_fragment_size': self.analysis.max_fragment_size,
                'extract_functions': self.analysis.extract_functions,
                'extract_classes': self.analysis.extract_classes,
                'extract_methods': self.analysis.extract_methods,
                'extract_blocks': self.analysis.extract_blocks,
                'extract_statements': self.analysis.extract_statements,
                'ignore_imports': self.analysis.ignore_imports,
                'ignore_comments': self.analysis.ignore_comments,
                'ignore_docstrings': self.analysis.ignore_docstrings,
                'normalize_variables': self.analysis.normalize_variables,
                'analyze_ast': self.analysis.analyze_ast,
            },
            'performance': {
                'max_files_to_analyze': self.performance.max_files_to_analyze,
                'max_fragments_per_file': self.performance.max_fragments_per_file,
                'similarity_cache_size': self.performance.similarity_cache_size,
                'parallel_processing': self.performance.parallel_processing,
                'max_workers': self.performance.max_workers,
                'analysis_timeout': self.performance.analysis_timeout,
                'similarity_timeout': self.performance.similarity_timeout,
            },
            'refactoring': {
                'enable_suggestions': self.refactoring.enable_suggestions,
                'min_group_size_for_refactor': self.refactoring.min_group_size_for_refactor,
                'suggest_method_extraction': self.refactoring.suggest_method_extraction,
                'suggest_class_extraction': self.refactoring.suggest_class_extraction,
                'suggest_utility_creation': self.refactoring.suggest_utility_creation,
                'suggest_config_extraction': self.refactoring.suggest_config_extraction,
                'enable_auto_fix': self.refactoring.enable_auto_fix,
                'auto_fix_min_confidence': self.refactoring.auto_fix_min_confidence,
            },
            'output_format': self.output_format,
            'output_file': self.output_file,
            'include_details': self.include_details,
            'include_source_code': self.include_source_code,
            'file_patterns': self.file_patterns,
            'exclude_patterns': self.exclude_patterns,
            'exclude_dirs': self.exclude_dirs,
        }

    def get_preset_config(self, preset: str) -> 'SmartDuplicateConfig':
        """
        获取预设配置

        Args:
            preset: 预设名称 ('strict', 'normal', 'relaxed', 'performance')

        Returns:
            SmartDuplicateConfig: 预设配置
        """
        if preset == 'strict':
            # 严格模式：只检测高度相似的代码
            self.similarity.exact_clone = 0.98
            self.similarity.similar_clone = 0.9
            self.analysis.min_fragment_size = 10

        elif preset == 'normal':
            # 正常模式：平衡检测精度和性能
            pass  # 使用默认值

        elif preset == 'relaxed':
            # 宽松模式：检测更多潜在重复
            self.similarity.exact_clone = 0.9
            self.similarity.similar_clone = 0.7
            self.similarity.semantic_clone = 0.5
            self.analysis.min_fragment_size = 3

        elif preset == 'performance':
            # 性能模式：快速检测，牺牲一些精度
            self.analysis.min_fragment_size = 8
            self.performance.parallel_processing = True

        elif preset == 'quality':
            # 质量模式：全面的质量检查，包括复杂度分析
            self.similarity.exact_clone = 0.95
            self.similarity.similar_clone = 0.8
            self.similarity.semantic_clone = 0.6
            self.analysis.min_fragment_size = 5
            self.analysis.analyze_ast = True
            self.analysis.normalize_variables = True
            # 启用复杂度分析
            self.analysis.complexity_analysis = True  # 新增配置项
            self.performance.max_workers = 8
            self.similarity.similarity_cache_size = 50000

        else:
            raise ValueError(f"未知的预设配置: {preset}")

        return self

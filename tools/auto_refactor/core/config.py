#!/usr/bin/env python3
"""
自动化重构工具配置管理

定义配置选项和安全级别。
"""

from enum import Enum
from typing import Optional, List
from pathlib import Path


class SafetyLevel(Enum):
    """安全级别"""

    LOW = "low"         # 低安全：快速执行，最少检查
    MEDIUM = "medium"   # 中安全：平衡安全和效率
    HIGH = "high"       # 高安全：最大安全保障


class RefactorConfig:
    """重构配置"""

    def __init__(self):
        # 安全配置
        self.safety_level: SafetyLevel = SafetyLevel.HIGH
        self.backup_enabled: bool = True
        self.validation_enabled: bool = True
        self.rollback_on_failure: bool = True
        self.fail_fast: bool = False

        # 备份配置
        self.backup_dir: Optional[str] = None  # None表示使用临时目录

        # 验证配置
        self.syntax_validation: bool = True
        self.import_validation: bool = True
        self.semantic_validation: bool = True
        self.type_checking: bool = False  # 需要类型注解时启用

        # 执行配置
        self.parallel_processing: bool = True
        self.max_workers: int = 4
        self.batch_size: int = 10

        # 重构配置
        self.max_suggestions_per_file: int = 20
        self.min_confidence_threshold: float = 0.7
        self.risk_tolerance: str = 'medium'

        # 输出配置
        self.verbose_output: bool = False
        self.generate_report: bool = True
        self.report_format: str = 'json'  # 'json', 'html', 'markdown'

        # 文件过滤
        self.include_patterns: List[str] = ['*.py']
        self.exclude_patterns: List[str] = [
            '__pycache__', '*.pyc', '.git', '.svn',
            'node_modules', '*.egg-info', '.tox'
        ]

        # 高级配置
        self.dry_run: bool = False  # 试运行模式
        self.force_execution: bool = False  # 强制执行（跳过某些检查）
        self.interactive_mode: bool = False  # 交互式模式

    @classmethod
    def from_preset(cls, preset: str) -> 'RefactorConfig':
        """从预设创建配置"""

        config = cls()

        if preset == 'safe':
            # 安全模式：最大安全保障
            config.safety_level = SafetyLevel.HIGH
            config.backup_enabled = True
            config.validation_enabled = True
            config.rollback_on_failure = True
            config.parallel_processing = False  # 安全起见不并行
            config.verbose_output = True

        elif preset == 'fast':
            # 快速模式：牺牲一些安全换取速度
            config.safety_level = SafetyLevel.LOW
            config.backup_enabled = False
            config.validation_enabled = False
            config.parallel_processing = True
            config.max_workers = 8

        elif preset == 'balanced':
            # 平衡模式：安全和效率的平衡
            config.safety_level = SafetyLevel.MEDIUM
            config.backup_enabled = True
            config.validation_enabled = True
            config.parallel_processing = True
            config.max_workers = 4

        elif preset == 'ci':
            # CI模式：适合持续集成环境
            config.safety_level = SafetyLevel.HIGH
            config.backup_enabled = True
            config.validation_enabled = True
            config.report_format = 'json'
            config.generate_report = True
            config.fail_fast = True

        elif preset == 'experimental':
            # 实验模式：用于测试新功能
            config.safety_level = SafetyLevel.LOW
            config.backup_enabled = True
            config.validation_enabled = False
            config.verbose_output = True
            config.dry_run = True  # 默认试运行

        return config

    def validate_config(self) -> List[str]:
        """验证配置有效性"""

        errors = []

        # 验证安全级别
        if not isinstance(self.safety_level, SafetyLevel):
            errors.append("Invalid safety_level")

        # 验证数值范围
        if self.max_workers < 1:
            errors.append("max_workers must be >= 1")

        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            errors.append("min_confidence_threshold must be between 0.0 and 1.0")

        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")

        # 验证路径
        if self.backup_dir:
            backup_path = Path(self.backup_dir)
            if not backup_path.parent.exists():
                errors.append(f"Backup directory parent does not exist: {backup_path.parent}")

        # 验证模式组合
        if self.dry_run and self.force_execution:
            errors.append("Cannot combine dry_run and force_execution")

        return errors

    def is_safe_mode(self) -> bool:
        """是否为安全模式"""
        return (self.safety_level == SafetyLevel.HIGH and
                self.backup_enabled and
                self.validation_enabled and
                self.rollback_on_failure)

    def get_risk_score(self) -> float:
        """计算风险评分 (0.0-1.0, 越高风险越大)"""

        risk_score = 0.0

        # 安全级别风险
        if self.safety_level == SafetyLevel.LOW:
            risk_score += 0.4
        elif self.safety_level == SafetyLevel.MEDIUM:
            risk_score += 0.2

        # 备份缺失风险
        if not self.backup_enabled:
            risk_score += 0.3

        # 验证缺失风险
        if not self.validation_enabled:
            risk_score += 0.2

        # 回滚缺失风险
        if not self.rollback_on_failure:
            risk_score += 0.2

        # 并行处理风险
        if self.parallel_processing:
            risk_score += 0.1

        # 强制执行风险
        if self.force_execution:
            risk_score += 0.3

        return min(1.0, risk_score)

    def should_skip_validation(self) -> bool:
        """是否应该跳过验证"""
        return (self.safety_level == SafetyLevel.LOW or
                self.dry_run or
                not self.validation_enabled)

    def get_execution_strategy(self) -> str:
        """获取执行策略"""
        if self.dry_run:
            return 'dry_run'
        elif self.interactive_mode:
            return 'interactive'
        elif self.parallel_processing:
            return 'parallel'
        else:
            return 'sequential'

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'safety_level': self.safety_level.value,
            'backup_enabled': self.backup_enabled,
            'validation_enabled': self.validation_enabled,
            'rollback_on_failure': self.rollback_on_failure,
            'fail_fast': self.fail_fast,
            'backup_dir': self.backup_dir,
            'syntax_validation': self.syntax_validation,
            'import_validation': self.import_validation,
            'semantic_validation': self.semantic_validation,
            'type_checking': self.type_checking,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'max_suggestions_per_file': self.max_suggestions_per_file,
            'min_confidence_threshold': self.min_confidence_threshold,
            'risk_tolerance': self.risk_tolerance,
            'verbose_output': self.verbose_output,
            'generate_report': self.generate_report,
            'report_format': self.report_format,
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns,
            'dry_run': self.dry_run,
            'force_execution': self.force_execution,
            'interactive_mode': self.interactive_mode
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RefactorConfig':
        """从字典创建配置"""
        config = cls()

        # 基本配置
        config.safety_level = SafetyLevel(config_dict.get('safety_level', 'high'))
        config.backup_enabled = config_dict.get('backup_enabled', True)
        config.validation_enabled = config_dict.get('validation_enabled', True)
        config.rollback_on_failure = config_dict.get('rollback_on_failure', True)
        config.fail_fast = config_dict.get('fail_fast', False)

        # 备份和验证配置
        config.backup_dir = config_dict.get('backup_dir')
        config.syntax_validation = config_dict.get('syntax_validation', True)
        config.import_validation = config_dict.get('import_validation', True)
        config.semantic_validation = config_dict.get('semantic_validation', True)
        config.type_checking = config_dict.get('type_checking', False)

        # 执行配置
        config.parallel_processing = config_dict.get('parallel_processing', True)
        config.max_workers = config_dict.get('max_workers', 4)
        config.batch_size = config_dict.get('batch_size', 10)

        # 重构配置
        config.max_suggestions_per_file = config_dict.get('max_suggestions_per_file', 20)
        config.min_confidence_threshold = config_dict.get('min_confidence_threshold', 0.7)
        config.risk_tolerance = config_dict.get('risk_tolerance', 'medium')

        # 输出配置
        config.verbose_output = config_dict.get('verbose_output', False)
        config.generate_report = config_dict.get('generate_report', True)
        config.report_format = config_dict.get('report_format', 'json')

        # 文件过滤
        config.include_patterns = config_dict.get('include_patterns', ['*.py'])
        config.exclude_patterns = config_dict.get('exclude_patterns', [
            '__pycache__', '*.pyc', '.git', '.svn',
            'node_modules', '*.egg-info', '.tox'
        ])

        # 高级配置
        config.dry_run = config_dict.get('dry_run', False)
        config.force_execution = config_dict.get('force_execution', False)
        config.interactive_mode = config_dict.get('interactive_mode', False)

        return config

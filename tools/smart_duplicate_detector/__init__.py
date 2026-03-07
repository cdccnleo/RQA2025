"""
智能重复代码检测工具

高级代码克隆检测、重构建议生成和自动修复功能。
"""

from .core import SmartDuplicateConfig, DetectionResult, CloneGroup
from .analyzers import CloneDetector

__version__ = "1.0.0"


def detect_clones(target_path: str, config: SmartDuplicateConfig = None) -> DetectionResult:
    """
    检测代码克隆

    Args:
        target_path: 检测目标路径
        config: 检测配置，如果为None则使用默认配置

    Returns:
        DetectionResult: 检测结果
    """
    if config is None:
        config = SmartDuplicateConfig()

    detector = CloneDetector(config)
    return detector.analyze(target_path)


def analyze_file_pair(file1: str, file2: str, config: SmartDuplicateConfig = None) -> dict:
    """
    分析两个文件之间的克隆关系

    Args:
        file1: 文件1路径
        file2: 文件2路径
        config: 检测配置

    Returns:
        dict: 分析结果
    """
    if config is None:
        config = SmartDuplicateConfig()

    detector = CloneDetector(config)
    return detector.analyze_file_pair(file1, file2)

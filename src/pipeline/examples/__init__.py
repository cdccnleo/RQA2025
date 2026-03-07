"""
管道示例模块

提供自动化训练管道的使用示例
"""

from .simple_pipeline import run_simple_pipeline
from .full_pipeline import run_full_pipeline
from .monitoring_example import run_monitoring_example

__all__ = [
    'run_simple_pipeline',
    'run_full_pipeline',
    'run_monitoring_example'
]

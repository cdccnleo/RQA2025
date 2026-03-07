"""
信号生成器模块 - 别名文件

为保持向后兼容性，从signal_signal_generator导入所有内容。
"""

from .signal_signal_generator import *

__all__ = ['SignalGenerator', 'SignalType', 'SignalStrength', 'SignalConfig']

#!/usr/bin/env python3
"""
业务流程模型别名文件

提供对models模块中流程相关类的别名导入
"""

# 导入models模块中的流程相关类
from ..models.models import ProcessInstance, ProcessConfig

__all__ = ['ProcessInstance', 'ProcessConfig']

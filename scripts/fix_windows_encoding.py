#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows编码问题修复脚本
"""

import os
import sys


def fix_windows_encoding():
    """修复Windows编码问题"""
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

    # 强制设置标准输出编码
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("Windows编码环境已修复")


if __name__ == "__main__":
    fix_windows_encoding()

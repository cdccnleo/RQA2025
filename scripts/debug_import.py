#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试导入问题
"""

import traceback

try:
    print("✅ ConnectionPool 导入成功")
except Exception as e:
    print("❌ ConnectionPool 导入失败")
    traceback.print_exc()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查最终覆盖率"""

import json
from pathlib import Path

coverage_file = Path("test_logs/coverage_monitoring_final.json")
if coverage_file.exists():
    with open(coverage_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = data['totals']
    print("=" * 80)
    print("监控模块最终覆盖率报告")
    print("=" * 80)
    print(f"\n整体覆盖率: {total['percent_covered']:.2f}%")
    print(f"总行数: {total['num_statements']}")
    print(f"已覆盖行数: {total['covered_lines']}")
    print(f"未覆盖行数: {total['missing_lines']}")
    print(f"\n目标覆盖率: 80%")
    print(f"达成情况: {'✅ 超额完成' if total['percent_covered'] >= 80 else '⚠️ 未达标'}")
else:
    print(f"覆盖率文件不存在: {coverage_file}")


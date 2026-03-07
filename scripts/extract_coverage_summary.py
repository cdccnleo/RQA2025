#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从覆盖率JSON文件中提取汇总数据
"""

import json
from pathlib import Path

def extract_coverage(json_file: Path) -> dict:
    """提取覆盖率数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            totals = data.get('totals', {})
            return {
                "file": json_file.name,
                "percent_covered": totals.get('percent_covered', 0),
                "num_statements": totals.get('num_statements', 0),
                "covered_lines": totals.get('covered_lines', 0),
                "missing_lines": totals.get('missing_lines', 0),
            }
    except Exception as e:
        return {"file": json_file.name, "error": str(e)}

def main():
    """主函数"""
    test_logs = Path(__file__).parent.parent / "test_logs"
    
    # 查找最新的覆盖率文件
    latest_files = {
        "config": None,
        "cache": None,
        "logging": None,
        "security": None,
        "monitoring": None,
        "resource": None,
    }
    
    for module in latest_files.keys():
        # 查找该模块最新的覆盖率文件
        pattern = f"coverage_{module}_2025110*.json"
        files = list(test_logs.glob(pattern))
        if files:
            # 按时间排序，取最新的
            latest = max(files, key=lambda p: p.stat().st_mtime)
            latest_files[module] = latest
    
    print("="*80)
    print("📊 大型模块覆盖率数据提取")
    print("="*80)
    
    results = {}
    for module, file in latest_files.items():
        if file:
            data = extract_coverage(file)
            results[module] = data
            
            if 'error' in data:
                print(f"❌ {module}: {data['error']}")
            else:
                print(f"✅ {module:15} | 覆盖率: {data['percent_covered']:5.2f}% | "
                      f"语句: {data['num_statements']:5} | 覆盖: {data['covered_lines']:5}")
        else:
            print(f"⚪ {module:15} | 未找到覆盖率文件")
    
    # 计算平均覆盖率
    valid_results = [r for r in results.values() if 'error' not in r and r['percent_covered'] > 0]
    if valid_results:
        avg_coverage = sum(r['percent_covered'] for r in valid_results) / len(valid_results)
        total_stmts = sum(r['num_statements'] for r in valid_results)
        total_covered = sum(r['covered_lines'] for r in valid_results)
        
        print("\n" + "="*80)
        print(f"平均覆盖率: {avg_coverage:.2f}%")
        print(f"总语句数: {total_stmts:,}")
        print(f"覆盖语句数: {total_covered:,}")
        print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())


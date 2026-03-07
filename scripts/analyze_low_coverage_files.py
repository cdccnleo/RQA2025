#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析各模块中覆盖率最低的文件
生成针对性的测试补充建议
"""

import json
from pathlib import Path


def analyze_module_coverage(module_name: str, coverage_file: Path):
    """分析模块覆盖率"""
    try:
        with open(coverage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        files_data = data.get('files', {})
        
        # 提取并排序
        file_list = []
        for filepath, file_data in files_data.items():
            summary = file_data.get('summary', {})
            file_list.append({
                'file': Path(filepath).name,
                'full_path': filepath,
                'statements': summary.get('num_statements', 0),
                'covered': summary.get('covered_lines', 0),
                'missing': summary.get('missing_lines', 0),
                'coverage': summary.get('percent_covered', 0),
            })
        
        # 按覆盖率排序
        file_list.sort(key=lambda x: x['coverage'])
        
        print(f"\n{'='*80}")
        print(f"📊 {module_name.upper()}模块 - 低覆盖率文件TOP10")
        print(f"{'='*80}")
        print(f"总文件数: {len(file_list)}")
        print(f"平均覆盖率: {sum(f['coverage'] for f in file_list) / len(file_list):.2f}%")
        print()
        
        total_missing_top10 = 0
        for i, file_info in enumerate(file_list[:10], 1):
            print(f"{i:2}. {file_info['file'][:50]:50} | "
                  f"{file_info['coverage']:5.1f}% | "
                  f"未覆盖: {file_info['missing']:4}行")
            total_missing_top10 += file_info['missing']
        
        print(f"\nTOP10文件未覆盖总行数: {total_missing_top10:,}行")
        print(f"预计需要新增测试: ~{total_missing_top10 // 10}个")
        
        return file_list
        
    except Exception as e:
        print(f"❌ 分析{module_name}出错: {e}")
        return []


def main():
    """主函数"""
    base_dir = Path(__file__).parent.parent
    test_logs = base_dir / "test_logs"
    
    # 待分析的模块
    modules = {
        "config": "coverage_config_20251102_174816.json",
        "cache": "coverage_cache_20251102_175017.json",
        "logging": "coverage_logging_20251102_172142.json",
    }
    
    print("\n" + "="*80)
    print("🔍 基础设施层低覆盖率文件分析")
    print("="*80)
    
    all_results = {}
    
    for module, filename in modules.items():
        coverage_file = test_logs / filename
        if coverage_file.exists():
            results = analyze_module_coverage(module, coverage_file)
            all_results[module] = results
        else:
            print(f"\n⚠️  {module}覆盖率文件不存在: {filename}")
    
    # 生成总体建议
    print("\n" + "="*80)
    print("📋 测试补充建议")
    print("="*80)
    
    total_tests_needed = 0
    for module, results in all_results.items():
        if results:
            top10_missing = sum(f['missing'] for f in results[:10])
            tests_needed = top10_missing // 10
            total_tests_needed += tests_needed
            
            print(f"\n{module.upper()}模块:")
            print(f"  - TOP10文件未覆盖: {top10_missing}行")
            print(f"  - 建议新增测试: ~{tests_needed}个")
            print(f"  - 重点文件（覆盖率<50%）:")
            
            low_coverage = [f for f in results if f['coverage'] < 50 and f['statements'] > 10]
            for f in low_coverage[:5]:
                print(f"    * {f['file'][:45]:45} ({f['coverage']:.1f}%)")
    
    print(f"\n总计建议新增测试: ~{total_tests_needed}个")
    print(f"预计工作量: {total_tests_needed // 10}-{total_tests_needed // 8}小时")


if __name__ == "__main__":
    main()


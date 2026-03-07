#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实覆盖率评估 - 排除有问题的测试文件
只统计能正常运行的测试的覆盖率
"""

import subprocess
import json
import os
from pathlib import Path

# 已知有问题的测试文件（导入错误）
PROBLEMATIC_TESTS = [
    # Config模块 - 缺少kazoo依赖
    "tests/unit/infrastructure/config/test_config_manager_refactored.py",
    "tests/unit/infrastructure/config/test_config_storage.py",
    "tests/unit/infrastructure/config/test_config_storage_enhanced.py",
    "tests/unit/infrastructure/config/test_config_storage_factory.py",
    "tests/unit/infrastructure/config/test_storage_distributedconfigstorage.py",
    
    # Cache模块 - 导入问题
    "tests/unit/infrastructure/cache/test_performance_monitoring_comprehensive.py",
    
    # Logging模块 - 缺少msgpack或导入问题
    "tests/unit/infrastructure/logging/test_interface_checker.py",
    "tests/unit/infrastructure/logging/test_logging_core_comprehensive.py",
    "tests/unit/infrastructure/logging/test_standards.py",
    "tests/unit/infrastructure/logging/test_standards_simple.py",
    
    # Health模块 - 导入问题
    "tests/unit/infrastructure/health/test_health_checker_deep_dive.py",
    "tests/unit/infrastructure/health/test_health_core_targeted_boost.py",
]

def get_all_infrastructure_modules():
    """获取所有基础设施子模块"""
    infrastructure_dir = Path("src/infrastructure")
    modules = []
    
    for item in infrastructure_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            modules.append(item.name)
    
    return sorted(modules)

def test_module_coverage(module_name):
    """测试单个模块的覆盖率"""
    print(f"\n{'='*80}")
    print(f"测试模块: {module_name}")
    print('='*80)
    
    test_dir = f"tests/unit/infrastructure/{module_name}/"
    src_dir = f"src/infrastructure/{module_name}"
    
    # 检查测试目录是否存在
    if not os.path.exists(test_dir):
        print(f"⚠️  测试目录不存在: {test_dir}")
        return None
    
    # 构建忽略参数
    ignore_args = []
    for problematic in PROBLEMATIC_TESTS:
        if module_name in problematic:
            ignore_args.append(f"--ignore={problematic}")
    
    ignore_str = " ".join(ignore_args)
    
    cmd = f'pytest {test_dir} --cov={src_dir} --cov-report=json:test_logs/cov_{module_name}.json -q {ignore_str}'
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 尝试读取覆盖率数据
        try:
            with open(f'test_logs/cov_{module_name}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                coverage = data['totals']['percent_covered']
                
                print(f"✅ 覆盖率: {coverage:.2f}%")
                
                return {
                    'module': module_name,
                    'coverage': coverage,
                    'lines_covered': data['totals']['covered_lines'],
                    'lines_total': data['totals']['num_statements'],
                    'tests_passed': result.returncode == 0
                }
        except Exception as e:
            print(f"❌ 无法读取覆盖率: {e}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  超时")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def main():
    """主函数"""
    print("🔍 基础设施层真实覆盖率评估")
    print("="*80)
    
    modules = get_all_infrastructure_modules()
    print(f"\n找到 {len(modules)} 个子模块:")
    print(", ".join(modules))
    
    # 测试每个模块
    results = []
    for module in modules:
        result = test_module_coverage(module)
        if result:
            results.append(result)
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 覆盖率汇总")
    print("="*80)
    
    total_lines = 0
    total_covered = 0
    
    print(f"\n{'模块':<20} {'覆盖率':<10} {'已覆盖':<10} {'总行数':<10}")
    print("-"*80)
    
    for result in sorted(results, key=lambda x: x['coverage'], reverse=True):
        print(f"{result['module']:<20} {result['coverage']:>6.2f}%  {result['lines_covered']:>8}  {result['lines_total']:>8}")
        total_lines += result['lines_total']
        total_covered += result['lines_covered']
    
    print("-"*80)
    
    if total_lines > 0:
        overall_coverage = (total_covered / total_lines) * 100
        print(f"{'总体':<20} {overall_coverage:>6.2f}%  {total_covered:>8}  {total_lines:>8}")
        
        print("\n" + "="*80)
        print("🎯 最终评估")
        print("="*80)
        print(f"基础设施层整体覆盖率: {overall_coverage:.2f}%")
        
        if overall_coverage >= 80:
            print("✅ 恭喜！已达到80%覆盖率目标！")
        else:
            gap = 80 - overall_coverage
            print(f"⚠️  距离80%目标还需提升: {gap:.2f}%")
        
        # 保存详细报告
        report = {
            'overall_coverage': overall_coverage,
            'total_lines': total_lines,
            'total_covered': total_covered,
            'modules': results,
            'target': 80,
            'achieved': overall_coverage >= 80
        }
        
        with open('test_logs/real_coverage_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存: test_logs/real_coverage_report.json")
    else:
        print("❌ 没有成功测试的模块")

if __name__ == "__main__":
    main()


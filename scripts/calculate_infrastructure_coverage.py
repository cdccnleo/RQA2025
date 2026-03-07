#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算基础设施层整体覆盖率
忽略有导入问题的测试文件
"""

import subprocess
import json
import sys

# 排除有问题的测试文件
EXCLUDED_TESTS = [
    "test_config_manager_refactored.py",
    "test_config_storage.py",
    "test_config_storage_enhanced.py",
    "test_config_storage_factory.py",
    "test_storage_distributedconfigstorage.py",
    "test_performance_monitoring_comprehensive.py",
    "test_interface_checker.py",
    "test_logging_core_comprehensive.py",
    "test_standards.py",
    "test_standards_simple.py",
]

def run_coverage():
    """运行覆盖率测试"""
    
    # 构建排除参数
    ignore_params = " ".join([f"--ignore=tests/unit/infrastructure/**/{test}" for test in EXCLUDED_TESTS])
    
    cmd = f'pytest tests/unit/infrastructure/ --cov=src/infrastructure --cov-report=json:test_logs/coverage_infrastructure_final.json --cov-report=term-missing -q {ignore_params}'
    
    print(f"运行命令: {cmd}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # 读取JSON报告
        try:
            with open('test_logs/coverage_infrastructure_final.json', 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data['totals']['percent_covered']
            print("\n" + "=" * 80)
            print(f"🎯 基础设施层总体覆盖率: {total_coverage:.2f}%")
            print("=" * 80)
            
            # 判断是否达标
            if total_coverage >= 80:
                print("✅ 恭喜！已达到80%覆盖率目标！")
                return 0
            else:
                gap = 80 - total_coverage
                print(f"⚠️  距离80%目标还需提升: {gap:.2f}%")
                return 1
                
        except FileNotFoundError:
            print("❌ 未找到覆盖率报告文件")
            return 1
        except Exception as e:
            print(f"❌ 读取覆盖率报告失败: {e}")
            return 1
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return 1
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_coverage())



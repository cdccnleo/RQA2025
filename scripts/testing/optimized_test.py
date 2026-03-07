#!/usr/bin/env python3
"""
优化测试脚本 - 解决测试环境配置和超时问题
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

def run_single_test(test_path: str, timeout: int = 60) -> Dict:
    """运行单个测试"""
    print(f"运行测试: {test_path}")
    
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'test', 'python', '-m', 'pytest',
            test_path, '-v', '--tb=short', '--timeout=30'
        ], capture_output=True, text=True, timeout=timeout)
        
        return {
            'test_path': test_path,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'test_path': test_path,
            'returncode': -1,
            'stdout': '',
            'stderr': f'测试超时 ({timeout}秒)',
            'success': False
        }
    except Exception as e:
        return {
            'test_path': test_path,
            'returncode': -1,
            'stdout': '',
            'stderr': f'测试异常: {e}',
            'success': False
        }

def run_feature_tests():
    """运行特征层核心测试"""
    print("=== 运行特征层核心测试 ===")
    
    # 设置环境变量
    os.environ['TESTING'] = 'true'
    os.environ['MOCK_EXTERNAL_DEPENDENCIES'] = 'true'
    
    # 核心测试列表
    core_tests = [
        'tests/unit/features/test_signal_generator.py::TestSignalConfig::test_signal_config_defaults',
        'tests/unit/features/processors/test_technical.py::TestTechnicalProcessor::test_calculate_rsi_basic',
        'tests/unit/features/test_feature_selector.py::TestFeatureSelector::test_initialization',
        'tests/unit/features/test_feature_config.py::test_feature_config_init_and_validate'
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_path in core_tests:
        print(f"\n--- 测试: {test_path} ---")
        result = run_single_test(test_path, timeout=60)
        results.append(result)
        
        if result['success']:
            print(f"✅ 通过: {test_path}")
            passed += 1
        else:
            print(f"❌ 失败: {test_path}")
            print(f"错误: {result['stderr']}")
            failed += 1
    
    # 输出总结
    print(f"\n=== 测试总结 ===")
    print(f"总测试数: {len(core_tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/(passed+failed)*100:.1f}%")
    
    return results

def run_infrastructure_tests():
    """运行基础设施层测试"""
    print("\n=== 运行基础设施层测试 ===")
    
    # 基础设施层核心测试
    infra_tests = [
        'tests/unit/infrastructure/test_cachemanager.py',
        'tests/unit/infrastructure/test_configmanager.py',
        'tests/unit/infrastructure/test_monitormanager.py'
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_path in infra_tests:
        if Path(test_path).exists():
            print(f"\n--- 测试: {test_path} ---")
            result = run_single_test(test_path, timeout=60)
            results.append(result)
            
            if result['success']:
                print(f"✅ 通过: {test_path}")
                passed += 1
            else:
                print(f"❌ 失败: {test_path}")
                print(f"错误: {result['stderr']}")
                failed += 1
        else:
            print(f"⚠️ 跳过: {test_path} (文件不存在)")
    
    # 输出总结
    print(f"\n=== 基础设施层测试总结 ===")
    print(f"总测试数: {len(infra_tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    if passed + failed > 0:
        print(f"通过率: {passed/(passed+failed)*100:.1f}%")
    
    return results

def run_integration_tests():
    """运行集成层测试"""
    print("\n=== 运行集成层测试 ===")
    
    # 集成层核心测试
    integration_tests = [
        'tests/unit/integration/test_systemintegrationmanager.py',
        'tests/unit/integration/test_layerinterface.py',
        'tests/unit/integration/test_dataintegration.py'
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_path in integration_tests:
        if Path(test_path).exists():
            print(f"\n--- 测试: {test_path} ---")
            result = run_single_test(test_path, timeout=60)
            results.append(result)
            
            if result['success']:
                print(f"✅ 通过: {test_path}")
                passed += 1
            else:
                print(f"❌ 失败: {test_path}")
                print(f"错误: {result['stderr']}")
                failed += 1
        else:
            print(f"⚠️ 跳过: {test_path} (文件不存在)")
    
    # 输出总结
    print(f"\n=== 集成层测试总结 ===")
    print(f"总测试数: {len(integration_tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    if passed + failed > 0:
        print(f"通过率: {passed/(passed+failed)*100:.1f}%")
    
    return results

def main():
    """主函数"""
    print("=== RQA2025 优化测试执行 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行各层测试
    feature_results = run_feature_tests()
    infra_results = run_infrastructure_tests()
    integration_results = run_integration_tests()
    
    # 总体统计
    all_results = feature_results + infra_results + integration_results
    total_passed = sum(1 for r in all_results if r['success'])
    total_failed = len(all_results) - total_passed
    
    print(f"\n=== 总体测试总结 ===")
    print(f"总测试数: {len(all_results)}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    if len(all_results) > 0:
        print(f"总体通过率: {total_passed/len(all_results)*100:.1f}%")
    
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    main() 
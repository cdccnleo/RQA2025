#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终覆盖率测试脚本

验证所有层级的修复效果，生成完整的覆盖率报告
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_layer_coverage_test(layer_name, test_path, cov_source):
    """
    运行单个层级的覆盖率测试

    Args:
        layer_name: 层级名称
        test_path: 测试路径
        cov_source: 覆盖率源代码路径

    Returns:
        dict: 测试结果
    """
    print(f"\n🔍 开始测试 {layer_name}...")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent / "src")

    # 构建pytest命令
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v", "--tb=short",
        f"--cov={cov_source}",
        "--cov-report=json:temp_coverage.json",
        "--cov-report=term-missing",
        "--maxfail=3",
        "-x",  # 遇到第一个失败就停止
        "--disable-warnings"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        # 读取覆盖率结果
        coverage_file = Path(__file__).parent.parent / "temp_coverage.json"
        coverage_data = {}

        if coverage_file.exists():
            try:
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取覆盖率数据失败: {e}")
            finally:
                # 清理临时文件
                coverage_file.unlink(missing_ok=True)

        return {
            'layer': layer_name,
            'return_code': result.returncode,
            'tests_run': 'Unknown',  # 从输出中解析
            'coverage': coverage_data.get('totals', {}).get('percent_covered', 0),
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED',
            'error_summary': result.stderr[:500] if result.returncode != 0 else None
        }

    except subprocess.TimeoutExpired:
        return {
            'layer': layer_name,
            'return_code': -1,
            'tests_run': 'Timeout',
            'coverage': 0,
            'status': 'TIMEOUT',
            'error_summary': 'Test execution timed out'
        }
    except Exception as e:
        return {
            'layer': layer_name,
            'return_code': -2,
            'tests_run': 'Error',
            'coverage': 0,
            'status': 'ERROR',
            'error_summary': str(e)
        }

def main():
    """主函数：运行所有层级的覆盖率测试"""

    print("🚀 开始最终覆盖率测试...")

    # 定义要测试的层级
    layers_to_test = [
        {
            'name': '核心服务层',
            'test_path': 'tests/unit/core/',
            'cov_source': 'src.core'
        },
        {
            'name': '数据管理层',
            'test_path': 'tests/unit/data/',
            'cov_source': 'src.data'
        },
        {
            'name': '特征分析层',
            'test_path': 'tests/unit/features/',
            'cov_source': 'src.features'
        },
        {
            'name': '机器学习层',
            'test_path': 'tests/unit/ml/',
            'cov_source': 'src.ml'
        },
        {
            'name': '监控层',
            'test_path': 'tests/unit/monitoring/',
            'cov_source': 'src.monitoring'
        },
        {
            'name': '策略服务层',
            'test_path': 'tests/unit/strategy/',
            'cov_source': 'src.strategy'
        }
    ]

    results = []
    total_coverage = 0
    successful_tests = 0

    for layer in layers_to_test:
        result = run_layer_coverage_test(
            layer['name'],
            layer['test_path'],
            layer['cov_source']
        )
        results.append(result)

        if result['status'] == 'SUCCESS':
            successful_tests += 1
            total_coverage += result['coverage']

        print(f"✅ {layer['name']}: {result['status']} - 覆盖率: {result['coverage']:.1f}%")

        if result['error_summary']:
            print(f"   错误摘要: {result['error_summary'][:100]}...")

    # 计算总体结果
    avg_coverage = total_coverage / successful_tests if successful_tests > 0 else 0

    final_report = {
        'timestamp': datetime.now().isoformat(),
        'test_summary': {
            'total_layers': len(layers_to_test),
            'successful_layers': successful_tests,
            'failed_layers': len(layers_to_test) - successful_tests,
            'average_coverage': round(avg_coverage, 1),
            'overall_status': 'SUCCESS' if successful_tests == len(layers_to_test) else 'PARTIAL_SUCCESS'
        },
        'layer_results': results
    }

    # 保存最终报告
    output_dir = Path(__file__).parent.parent / "test_logs"
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / "final_coverage_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("\n📊 最终测试报告已生成:")
    print(f"   文件: {report_file}")
    print(f"   测试层级: {len(layers_to_test)}")
    print(f"   成功层级: {successful_tests}")
    print(f"   平均覆盖率: {avg_coverage:.1f}%")
    print(f"   总体状态: {final_report['test_summary']['overall_status']}")

    return final_report

if __name__ == "__main__":
    main()

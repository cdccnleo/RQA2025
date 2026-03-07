#!/usr/bin/env python3
"""
优化的数据层测试执行脚本
减少线程创建开销，提高执行效率
"""

import os
import sys
import subprocess
import time


def run_optimized_tests():
    """运行优化的测试"""
    print("🚀 启动优化的数据层测试执行...")

    # 配置优化参数
    test_configs = [
        {
            'name': 'BaseDataLoader Tests',
            'command': f'cd {os.path.dirname(__file__)} && python -m pytest ../tests/unit/data/test_base_loader.py -v --tb=short',
            'description': '基础数据加载器测试'
        },
        {
            'name': 'FinancialDataLoader Tests',
            'command': f'cd {os.path.dirname(__file__)} && python -m pytest ../tests/unit/data/test_financial_loader.py -v --tb=short',
            'description': '金融数据加载器测试'
        },
        {
            'name': 'NewsLoader Tests',
            'command': f'cd {os.path.dirname(__file__)} && python -m pytest ../tests/unit/data/test_news_loader.py -v --tb=short',
            'description': '新闻数据加载器测试'
        },
        {
            'name': 'DataAdapters Tests',
            'command': f'cd {os.path.dirname(__file__)} && python -m pytest ../tests/unit/data/test_data_adapters.py -v --tb=short',
            'description': '数据适配器测试'
        },
        {
            'name': 'DataIntegration Tests',
            'command': f'cd {os.path.dirname(__file__)} && python -m pytest ../tests/unit/data/test_data_integration.py -v --tb=short',
            'description': '数据集成测试'
        }
    ]

    results = []
    total_start_time = time.time()

    for config in test_configs:
        print(f"\n📋 执行: {config['name']}")
        print(f"   {config['description']}")

        start_time = time.time()
        result = subprocess.run(
            config['command'],
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )

        execution_time = time.time() - start_time

        success = result.returncode == 0
        results.append({
            'name': config['name'],
            'success': success,
            'execution_time': execution_time,
            'output': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        })

        status = "✅" if success else "❌"
        print(".2f")

        if not success:
            print(f"   错误信息: {result.stderr[-200:] if result.stderr else '无错误信息'}")

    total_time = time.time() - total_start_time

    # 输出总结
    print("\n" + "="*60)
    print("📊 测试执行总结")
    print("="*60)

    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)

    print(f"总测试套件数: {total_tests}")
    print(f"成功测试套件数: {successful_tests}")
    print(f"失败测试套件数: {total_tests - successful_tests}")
    print(".1f")
    print(".2f")

    # 详细结果
    print("\n📋 详细结果:")
    for result in results:
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(".2f")

    if successful_tests == total_tests:
        print("\n🎉 所有测试套件执行成功！")
        return 0
    else:
        print(f"\n⚠️ {total_tests - successful_tests} 个测试套件执行失败")
        return 1


if __name__ == '__main__':
    sys.exit(run_optimized_tests())

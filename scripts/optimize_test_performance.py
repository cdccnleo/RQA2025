#!/usr/bin/env python3
"""
测试性能优化脚本
优化测试执行性能，减少线程创建开销
"""

import os
import sys
import time
import psutil
import threading

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))


def get_thread_count():
    """获取当前线程数量"""
    return threading.active_count()


def get_memory_usage():
    """获取内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def run_test_with_monitoring(test_command, description):
    """运行测试并监控性能"""
    print(f"\n=== {description} ===")

    # 记录开始状态
    start_time = time.time()
    start_threads = get_thread_count()
    start_memory = get_memory_usage()

    print(".1f")
    print(f"初始线程数: {start_threads}")
    print(".1f")

    # 执行测试
    result = os.system(test_command)

    # 记录结束状态
    end_time = time.time()
    end_threads = get_thread_count()
    end_memory = get_memory_usage()

    execution_time = end_time - start_time
    thread_increase = end_threads - start_threads
    memory_increase = end_memory - start_memory

    print(".2f")
    print(f"线程增加: {thread_increase}")
    print(".1f")
    print(f"执行结果: {'成功' if result == 0 else '失败'}")

    return {
        'execution_time': execution_time,
        'thread_increase': thread_increase,
        'memory_increase': memory_increase,
        'success': result == 0
    }


def optimize_test_configuration():
    """优化测试配置"""
    print("\n=== 测试配置优化建议 ===")

    suggestions = [
        {
            'title': '减少线程创建开销',
            'description': '通过以下方式减少线程创建:',
            'actions': [
                '1. 使用 pytest-xdist 并行执行测试',
                '2. 配置适当的 worker 数量 (-n 4)',
                '3. 使用 --tb=short 减少错误输出',
                '4. 禁用不必要的插件和 fixtures'
            ]
        },
        {
            'title': '优化内存使用',
            'description': '减少内存占用:',
            'actions': [
                '1. 使用 gc.collect() 在测试间清理内存',
                '2. 避免在fixtures中缓存大数据',
                '3. 使用临时文件而不是内存缓存',
                '4. 及时清理测试数据'
            ]
        },
        {
            'title': '提高测试执行速度',
            'description': '优化执行效率:',
            'actions': [
                '1. 使用 --maxfail=5 快速失败',
                '2. 并行运行慢速测试',
                '3. 使用 mocks 替代真实服务调用',
                '4. 优化数据库连接池'
            ]
        }
    ]

    for suggestion in suggestions:
        print(f"\n{suggestion['title']}:")
        print(f"  {suggestion['description']}")
        for action in suggestion['actions']:
            print(f"  {action}")


def create_optimized_test_script():
    """创建优化的测试脚本"""
    script_content = '''#!/usr/bin/env python3
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
            'command': 'python -m pytest tests/unit/data/test_base_loader.py -v --tb=short',
            'description': '基础数据加载器测试'
        },
        {
            'name': 'FinancialDataLoader Tests',
            'command': 'python -m pytest tests/unit/data/test_financial_loader.py -v --tb=short',
            'description': '金融数据加载器测试'
        },
        {
            'name': 'NewsLoader Tests',
            'command': 'python -m pytest tests/unit/data/test_news_loader.py -v --tb=short',
            'description': '新闻数据加载器测试'
        },
        {
            'name': 'DataAdapters Tests',
            'command': 'python -m pytest tests/unit/data/test_data_adapters.py -v --tb=short',
            'description': '数据适配器测试'
        },
        {
            'name': 'DataIntegration Tests',
            'command': 'python -m pytest tests/unit/data/test_data_integration.py -v --tb=short',
            'description': '数据集成测试'
        }
    ]

    results = []
    total_start_time = time.time()

    for config in test_configs:
        print(f"\\n📋 执行: {config['name']}")
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
    print("\\n" + "="*60)
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
    print("\\n📋 详细结果:")
    for result in results:
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(".2f")

    if successful_tests == total_tests:
        print("\\n🎉 所有测试套件执行成功！")
        return 0
    else:
        print(f"\\n⚠️ {total_tests - successful_tests} 个测试套件执行失败")
        return 1

if __name__ == '__main__':
    sys.exit(run_optimized_tests())
'''

    with open(os.path.join(os.path.dirname(__file__), 'run_optimized_data_tests.py'), 'w', encoding='utf-8') as f:
        f.write(script_content)

    print("✅ 已创建优化测试脚本: scripts/run_optimized_data_tests.py")


def main():
    """主函数"""
    print("🚀 测试性能优化工具")
    print("=" * 50)

    # 运行性能监控测试
    performance_tests = [
        ('python -m pytest tests/unit/data/test_base_loader.py::TestBaseDataLoader::test_base_data_loader_initialization -v',
         '单个基础数据加载器测试'),
        ('python -m pytest tests/unit/data/test_financial_loader.py -v',
         '金融数据加载器完整测试套件'),
        ('python -m pytest tests/unit/data/test_data_integration.py -v',
         '数据集成测试套件')
    ]

    results = []
    for command, description in performance_tests:
        result = run_test_with_monitoring(command, description)
        results.append((description, result))

    # 输出性能对比
    print("\n" + "="*60)
    print("📊 性能分析结果")
    print("="*60)

    for description, result in results:
        print(f"\\n{description}:")
        print(".2f")
        print(f"  线程增加: {result['thread_increase']}")
        print(".1f")
        print(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}")

    # 提供优化建议
    optimize_test_configuration()

    # 创建优化的测试脚本
    create_optimized_test_script()

    print("\\n" + "="*60)
    print("✅ 性能优化分析完成")
    print("💡 建议使用新创建的优化脚本提高测试执行效率")


if __name__ == '__main__':
    main()

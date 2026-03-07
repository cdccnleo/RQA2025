#!/usr/bin/env python3
"""
数据层coverage测试运行脚本
专门用于解决数据层coverage统计问题的测试运行器
"""

import os
import sys
import subprocess
import coverage


def setup_coverage_for_data():
    """为数据层设置coverage"""
    # 确保src目录在Python路径中
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # 创建coverage实例，只跟踪数据层
    cov = coverage.Coverage(
        source=['src/data', 'src/loader'],
        omit=[
            '*/tests/*',
            '*/test_*',
            '*/scripts/*',
            '*/temp_*',
            '*/__pycache__/*',
            '*.pyc',
            'setup.py'
        ],
        branch=True,
        concurrency='thread'
    )

    return cov


def run_data_tests(cov):
    """运行数据层的测试"""
    print("Running data layer tests...")

    # 数据层测试文件列表 - 先从核心模块开始
    test_files = [
        'tests/unit/data/test_data_manager.py',
        'tests/unit/data/test_base_loader.py',
        'tests/unit/data/test_stock_loader.py',
        'tests/unit/data/test_news_loader.py',
        'tests/unit/data/test_data_cache.py',
        'tests/unit/data/test_data_model.py',
        'tests/unit/data/test_data_validator.py',
        'tests/unit/data/test_data_quality.py',
    ]

    success_count = 0
    total_count = len(test_files)

    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Testing: {test_file}")
        print(f"{'='*60}")

        # 检查文件是否存在
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            continue

        try:
            # 启动coverage跟踪
            cov.start()

            # 运行单个测试文件
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v', '--tb=short',
                '--maxfail=3'
            ]

            result = subprocess.run(cmd, capture_output=False, cwd=os.getcwd())
            success = result.returncode == 0

            if success:
                print(f"✓ Tests passed for {test_file}")
                success_count += 1
            else:
                print(f"✗ Tests failed for {test_file}")

        finally:
            # 停止coverage并保存
            cov.stop()
            cov.save()

    return success_count, total_count


def check_existing_data_tests():
    """检查现有的数据层测试文件"""
    print("Checking existing data layer test files...")

    test_dirs = [
        'tests/unit/data',
        'tests/unit/loader'
    ]

    existing_tests = []

    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        existing_tests.append(os.path.join(root, file))

    print(f"Found {len(existing_tests)} existing test files:")
    for test in existing_tests[:10]:  # 只显示前10个
        print(f"  - {test}")

    if len(existing_tests) > 10:
        print(f"  ... and {len(existing_tests) - 10} more")

    return existing_tests


def generate_data_report(cov):
    """生成数据层的coverage报告"""
    try:
        print("\nGenerating data coverage report...")

        # 生成报告
        cov.report(show_missing=True)
        cov.html_report(directory='htmlcov_data')
        cov.json_report(outfile='data_coverage.json')

        print("Data coverage report generated:")
        print("- HTML report: htmlcov_data/index.html")
        print("- JSON report: data_coverage.json")

    except Exception as e:
        print(f"Error generating data report: {e}")


def analyze_data_coverage():
    """分析数据层coverage数据"""
    print("\nAnalyzing data coverage data...")

    try:
        # 读取coverage数据
        import json
        if os.path.exists('data_coverage.json'):
            with open('data_coverage.json', 'r') as f:
                coverage_data = json.load(f)

            print("Coverage analysis:")
            data_files = []
            for file_path, file_data in coverage_data.get('files', {}).items():
                if 'src/data' in file_path or 'src/loader' in file_path:
                    data_files.append((file_path, file_data))

            # 按覆盖率排序
            data_files.sort(key=lambda x: x[1].get('summary', {}).get(
                'percent_covered', 0), reverse=True)

            for file_path, file_data in data_files[:15]:  # 只显示前15个
                summary = file_data.get('summary', {})
                lines = summary.get('num_statements', 0)
                covered = summary.get('covered_lines', 0)
                percent = summary.get('percent_covered', 0)

                status = "✓" if percent >= 80 else "⚠" if percent >= 50 else "✗"
                print(".1f")
    except Exception as e:
        print(f"Error analyzing data coverage data: {e}")


def main():
    """主函数"""
    print("RQA2025 Data Layer Coverage Test")
    print("=" * 60)

    # 首先检查现有的测试文件
    existing_tests = check_existing_data_tests()

    # 设置coverage
    print("\nSetting up coverage for data layer...")
    cov = setup_coverage_for_data()

    # 运行测试
    success_count, total_count = run_data_tests(cov)

    # 生成报告
    generate_data_report(cov)

    # 分析数据
    analyze_data_coverage()

    # 统计结果
    print(f"\n{'='*60}")
    print("Data Coverage Test Summary:")
    print(f"{'='*60}")
    print(f"Test files processed: {total_count}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {total_count - success_count}")
    print(f"Existing test files found: {len(existing_tests)}")

    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    print(".1f")
    if success_count == total_count:
        print("\n✓ All data tests completed successfully!")
        return 0
    else:
        print("\n✗ Some data tests failed!")
        print("Note: Data layer is very large. Consider creating tests incrementally.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

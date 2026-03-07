#!/usr/bin/env python3
"""
快速覆盖率检查脚本
"""

import json
from pathlib import Path


def check_coverage():
    """检查当前测试覆盖率状态"""
    print("🎯 RQA2025 测试覆盖率状态检查")
    print("=" * 60)

    # 检查覆盖率文件
    coverage_file = Path("reports/coverage.json")
    if not coverage_file.exists():
        print("❌ 覆盖率文件不存在: reports/coverage.json")
        print("📝 建议先运行测试生成覆盖率报告")
        return

    try:
        with open(coverage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取总体覆盖率
        totals = data.get('totals', {})
        overall_coverage = totals.get('percent_covered', 0.0)
        total_statements = totals.get('num_statements', 0)
        covered_statements = totals.get('covered_lines', 0)
        missing_statements = totals.get('missing_lines', 0)

        print(f"📊 总体覆盖率: {overall_coverage:.2f}%")
        print(f"📈 总语句数: {total_statements:,}")
        print(f"✅ 覆盖语句数: {covered_statements:,}")
        print(f"❌ 未覆盖语句数: {missing_statements:,}")
        print()

        # 分析模块覆盖率
        files = data.get('files', {})
        print("📂 模块覆盖率统计:")
        print("-" * 50)

        module_stats = {}
        for file_path, file_data in files.items():
            if file_path.startswith('src/'):
                parts = file_path.split('/')
                if len(parts) >= 2:
                    module = f"src.{parts[1]}"

                    if module not in module_stats:
                        module_stats[module] = {
                            'total_statements': 0,
                            'covered_statements': 0,
                            'files': 0
                        }

                    # 累计统计
                    summary = file_data.get('summary', {})
                    total = summary.get('num_statements', 0)
                    covered = total - summary.get('missing_lines', 0)

                    module_stats[module]['total_statements'] += total
                    module_stats[module]['covered_statements'] += covered
                    module_stats[module]['files'] += 1

        # 显示模块覆盖率
        for module, stats in sorted(module_stats.items()):
            if stats['total_statements'] > 0:
                coverage = (stats['covered_statements'] / stats['total_statements']) * 100
                print(f"  {module:<25} {coverage:6.2f}% ({stats['files']} files)")
            else:
                print(f"  {module:<25}   0.00% ({stats['files']} files)")

        print()
        print("🎯 覆盖率目标分析:")
        print("-" * 50)

        if overall_coverage >= 90.0:
            print("✅ 达到投产目标 (90%)")
        elif overall_coverage >= 80.0:
            print("⚠️  达到最低要求 (80%) 但未达投产目标 (90%)")
            gap = 90.0 - overall_coverage
            print(f"📈 距离投产目标还差: {gap:.2f}%")
        else:
            print("❌ 未达到最低要求 (80%)")
            gap = 80.0 - overall_coverage
            print(f"📈 距离最低要求还差: {gap:.2f}%")

        # 检查已创建的测试文件
        print()
        print("📁 测试文件状态:")
        print("-" * 50)
        tests_dir = Path("tests")
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            print(f"✅ 已创建测试文件: {len(test_files)} 个")
            for test_file in test_files:
                size = test_file.stat().st_size / 1024
                print(f"  📄 {test_file.name} ({size:.1f}KB)")

    except Exception as e:
        print(f"❌ 分析覆盖率数据失败: {e}")


if __name__ == "__main__":
    check_coverage()

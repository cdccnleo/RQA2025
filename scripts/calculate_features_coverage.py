#!/usr/bin/env python3
"""
计算特征层测试覆盖率的脚本
"""

import re
import subprocess
import sys


def calculate_features_coverage():
    """计算特征层的测试覆盖率"""
    try:
        # 运行测试并获取覆盖率报告
        result = subprocess.run([
            'python', '-m', 'pytest', 'tests/unit/features/',
            '--cov=src/features', '--cov-report=term-missing'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode != 0:
            print("测试运行失败")
            return None

        output = result.stdout

        # 提取特征层模块的覆盖率信息
        features_modules = {}
        total_statements = 0
        total_missed = 0

        # 使用正则表达式匹配特征层模块
        # 匹配模式：模块名 + 语句数 + 未覆盖数 + 覆盖率
        lines = output.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('src\\features') and line.endswith('.py'):
                # 找到模块名行，下一行应该包含覆盖率信息
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # 匹配覆盖率行：数字 + 数字 + 数字.数字%
                    coverage_match = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+\.\d+)%', next_line)
                    if coverage_match:
                        statements = int(coverage_match.group(1))
                        missed = int(coverage_match.group(2))
                        coverage = float(coverage_match.group(3))

                        total_statements += statements
                        total_missed += missed

                        features_modules[line] = {
                            'statements': statements,
                            'missed': missed,
                            'coverage': coverage
                        }
            i += 1

        # 计算总体覆盖率
        if total_statements > 0:
            overall_coverage = ((total_statements - total_missed) / total_statements) * 100
        else:
            overall_coverage = 0

        return {
            'modules': features_modules,
            'total_statements': total_statements,
            'total_missed': total_missed,
            'overall_coverage': overall_coverage
        }

    except Exception as e:
        print(f"计算覆盖率时出错: {e}")
        return None


def main():
    """主函数"""
    print("正在计算特征层测试覆盖率...")

    coverage_data = calculate_features_coverage()

    if coverage_data is None:
        print("无法获取覆盖率数据")
        sys.exit(1)

    print(f"\n特征层测试覆盖率报告")
    print("=" * 50)
    print(f"总体覆盖率: {coverage_data['overall_coverage']:.2f}%")
    print(f"总语句数: {coverage_data['total_statements']}")
    print(f"未覆盖语句数: {coverage_data['total_missed']}")
    print(f"已覆盖语句数: {coverage_data['total_statements'] - coverage_data['total_missed']}")

    print(f"\n各模块覆盖率详情:")
    print("-" * 80)
    print(f"{'模块名':<50} {'语句数':<8} {'未覆盖':<8} {'覆盖率':<8}")
    print("-" * 80)

    # 按覆盖率排序
    sorted_modules = sorted(
        coverage_data['modules'].items(),
        key=lambda x: x[1]['coverage'],
        reverse=True
    )

    for module_name, data in sorted_modules:
        print(
            f"{module_name:<50} {data['statements']:<8} {data['missed']:<8} {data['coverage']:<8.2f}%")

    # 统计覆盖率分布
    coverage_ranges = {
        '100%': 0,
        '90-99%': 0,
        '80-89%': 0,
        '70-79%': 0,
        '60-69%': 0,
        '50-59%': 0,
        '40-49%': 0,
        '30-39%': 0,
        '20-29%': 0,
        '10-19%': 0,
        '0-9%': 0
    }

    for data in coverage_data['modules'].values():
        coverage = data['coverage']
        if coverage == 100:
            coverage_ranges['100%'] += 1
        elif coverage >= 90:
            coverage_ranges['90-99%'] += 1
        elif coverage >= 80:
            coverage_ranges['80-89%'] += 1
        elif coverage >= 70:
            coverage_ranges['70-79%'] += 1
        elif coverage >= 60:
            coverage_ranges['60-69%'] += 1
        elif coverage >= 50:
            coverage_ranges['50-59%'] += 1
        elif coverage >= 40:
            coverage_ranges['40-49%'] += 1
        elif coverage >= 30:
            coverage_ranges['30-39%'] += 1
        elif coverage >= 20:
            coverage_ranges['20-29%'] += 1
        elif coverage >= 10:
            coverage_ranges['10-19%'] += 1
        else:
            coverage_ranges['0-9%'] += 1

    print(f"\n覆盖率分布:")
    print("-" * 30)
    for range_name, count in coverage_ranges.items():
        if count > 0:
            print(f"{range_name}: {count} 个模块")

    # 识别需要重点关注的模块
    low_coverage_modules = []
    for module_name, data in coverage_data['modules'].items():
        if data['coverage'] < 50:
            low_coverage_modules.append((module_name, data['coverage']))

    if low_coverage_modules:
        print(f"\n需要重点关注的模块 (覆盖率 < 50%):")
        print("-" * 50)
        for module_name, coverage in sorted(low_coverage_modules, key=lambda x: x[1]):
            print(f"{module_name}: {coverage:.2f}%")

    return coverage_data


if __name__ == "__main__":
    main()

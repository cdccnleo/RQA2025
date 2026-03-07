#!/usr/bin/env python3
"""
RQA2025 测试覆盖率验证脚本
验证当前测试覆盖率是否达到90%目标
"""

import json
from pathlib import Path


def analyze_coverage_data():
    """分析覆盖率数据并生成验证报告"""

    coverage_file = Path("reports/coverage.json")

    if not coverage_file.exists():
        print("❌ 覆盖率数据文件不存在: reports/coverage.json")
        return False

    try:
        with open(coverage_file, 'r', encoding='utf-8') as f:
            coverage_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取覆盖率数据失败: {e}")
        return False

    print("=" * 60)
    print("🚀 RQA2025 测试覆盖率验证报告")
    print("=" * 60)

    # 元数据信息
    meta = coverage_data.get('meta', {})
    print(f"📅 数据时间戳: {meta.get('timestamp', '未知')}")
    print(f"🔧 Coverage版本: {meta.get('version', '未知')}")
    print(f"🌿 分支覆盖: {'是' if meta.get('branch_coverage') else '否'}")
    print()

    # 统计文件覆盖率
    files = coverage_data.get('files', {})
    total_statements = 0
    total_covered = 0
    total_missing = 0
    total_branches = 0
    covered_branches = 0

    module_stats = {}
    low_coverage_files = []
    high_coverage_files = []
    zero_coverage_files = []

    for file_path, file_data in files.items():
        summary = file_data.get('summary', {})
        covered_lines = summary.get('covered_lines', 0)
        num_statements = summary.get('num_statements', 0)
        missing_lines = summary.get('missing_lines', 0)
        percent_covered = summary.get('percent_covered', 0)

        # 分支覆盖率
        num_branches = summary.get('num_branches', 0)
        branch_covered = summary.get('covered_branches', 0)

        total_statements += num_statements
        total_covered += covered_lines
        total_missing += missing_lines
        total_branches += num_branches
        covered_branches += branch_covered

        # 提取模块名
        if '\\' in file_path:
            parts = file_path.split('\\')
            module = parts[1] if len(parts) > 1 else parts[0]
        else:
            module = 'root'

        if module not in module_stats:
            module_stats[module] = {
                'statements': 0, 'covered': 0, 'files': 0,
                'branches': 0, 'branch_covered': 0
            }

        module_stats[module]['statements'] += num_statements
        module_stats[module]['covered'] += covered_lines
        module_stats[module]['files'] += 1
        module_stats[module]['branches'] += num_branches
        module_stats[module]['branch_covered'] += branch_covered

        # 分类文件
        if percent_covered == 0:
            zero_coverage_files.append(file_path)
        elif percent_covered < 50:
            low_coverage_files.append((file_path, percent_covered))
        elif percent_covered >= 80:
            high_coverage_files.append((file_path, percent_covered))

    # 计算总体覆盖率
    overall_coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0
    branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0

    print("📊 总体覆盖率统计:")
    print(f"   📝 总语句数: {total_statements:,}")
    print(f"   ✅ 已覆盖语句: {total_covered:,}")
    print(f"   ❌ 未覆盖语句: {total_missing:,}")
    print(f"   📈 语句覆盖率: {overall_coverage:.2f}%")

    if total_branches > 0:
        print(f"   🌿 总分支数: {total_branches:,}")
        print(f"   ✅ 已覆盖分支: {covered_branches:,}")
        print(f"   📈 分支覆盖率: {branch_coverage:.2f}%")
    print()

    # 目标验证
    target_coverage = 90.0
    gap = target_coverage - overall_coverage

    print("🎯 目标验证结果:")
    print(f"   🎯 目标覆盖率: {target_coverage}%")
    print(f"   📊 当前覆盖率: {overall_coverage:.2f}%")

    if overall_coverage >= target_coverage:
        print(f"   ✅ 🎉 已达标! 超出目标 {overall_coverage - target_coverage:.2f}%")
        status = "达标"
        color = "🟢"
    elif overall_coverage >= 85:
        print(f"   ⚠️  接近达标! 差距 {gap:.2f}%")
        status = "接近达标"
        color = "🟡"
    else:
        print(f"   ❌ 未达标! 差距 {gap:.2f}%")
        status = "未达标"
        color = "🔴"

    if overall_coverage < target_coverage:
        statements_needed = int(gap / 100 * total_statements)
        print(f"   📋 需要增加覆盖: {statements_needed:,} 语句")
    print()

    # 模块覆盖率详情
    print("📁 各模块覆盖率详情:")
    sorted_modules = sorted(
        module_stats.items(),
        key=lambda x: (x[1]['covered']/x[1]['statements']*100) if x[1]['statements'] > 0 else 0,
        reverse=True
    )

    for i, (module, stats) in enumerate(sorted_modules):
        if i >= 15:  # 显示前15个模块
            break
        module_coverage = (stats['covered'] / stats['statements']
                           * 100) if stats['statements'] > 0 else 0
        if module_coverage >= 90:
            status_icon = "🟢"
        elif module_coverage >= 80:
            status_icon = "🟡"
        elif module_coverage >= 50:
            status_icon = "🟠"
        else:
            status_icon = "🔴"

        print(f"   {status_icon} {module:<20}: {module_coverage:6.1f}% "
              f"({stats['covered']:>4}/{stats['statements']:<4}) [{stats['files']:>2}文件]")

    if len(sorted_modules) > 15:
        print(f"   ... 共 {len(sorted_modules)} 个模块")
    print()

    # 问题文件统计
    print("⚠️  覆盖率问题统计:")
    print(f"   🔴 零覆盖率文件: {len(zero_coverage_files)} 个")
    print(f"   🟠 低覆盖率文件 (<50%): {len(low_coverage_files)} 个")
    print(f"   🟢 高覆盖率文件 (≥80%): {len(high_coverage_files)} 个")

    # 最需要改进的文件
    if low_coverage_files:
        print("\n📋 最需要改进的文件:")
        sorted_low = sorted(low_coverage_files, key=lambda x: x[1])
        for file_path, coverage in sorted_low[:10]:
            file_name = file_path.split('\\')[-1] if '\\' in file_path else file_path
            print(f"   🔴 {file_name:<30}: {coverage:>6.1f}%")

    # 覆盖率最好的文件
    if high_coverage_files:
        print("\n🏆 覆盖率最好的文件:")
        sorted_high = sorted(high_coverage_files, key=lambda x: x[1], reverse=True)
        for file_path, coverage in sorted_high[:5]:
            file_name = file_path.split('\\')[-1] if '\\' in file_path else file_path
            print(f"   🟢 {file_name:<30}: {coverage:>6.1f}%")

    print()
    print("=" * 60)
    print(f"🏁 验证完成 | 状态: {color} {status} | 覆盖率: {overall_coverage:.2f}%")
    print("=" * 60)

    return overall_coverage >= target_coverage


if __name__ == "__main__":
    success = analyze_coverage_data()
    exit(0 if success else 1)

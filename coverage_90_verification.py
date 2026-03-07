#!/usr/bin/env python3
"""
RQA2025项目90%测试覆盖率验证脚本
=============================================

用途：
- 验证当前测试覆盖率是否达到90%目标
- 生成详细的覆盖率分析报告
- 识别覆盖率不足的关键模块

运行方式：
python coverage_90_verification.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_coverage_data(coverage_file):
    """加载覆盖率数据文件"""
    try:
        with open(coverage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ 成功加载覆盖率数据文件: {coverage_file}")
        return data
    except FileNotFoundError:
        logger.error(f"❌ 找不到覆盖率数据文件: {coverage_file}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ 解析覆盖率数据文件失败: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 读取覆盖率数据文件时发生错误: {e}")
        return None


def calculate_overall_coverage(coverage_data):
    """计算总体覆盖率"""
    if not coverage_data or 'files' not in coverage_data:
        return 0.0

    total_statements = 0
    total_covered = 0

    for file_path, file_data in coverage_data['files'].items():
        if 'summary' in file_data:
            summary = file_data['summary']
            total_statements += summary.get('num_statements', 0)
            total_covered += summary.get('covered_lines', 0)

    if total_statements == 0:
        return 0.0

    coverage_percent = (total_covered / total_statements) * 100
    return coverage_percent


def analyze_coverage_by_module(coverage_data):
    """按模块分析覆盖率"""
    if not coverage_data or 'files' not in coverage_data:
        return {}

    module_stats = {}

    for file_path, file_data in coverage_data['files'].items():
        if 'summary' not in file_data:
            continue

        summary = file_data['summary']

        # 提取模块路径
        if '\\' in file_path:
            parts = file_path.split('\\')
        else:
            parts = file_path.split('/')

        if len(parts) >= 2:
            module = parts[1] if parts[0] == 'src' else parts[0]
        else:
            module = 'other'

        if module not in module_stats:
            module_stats[module] = {
                'total_statements': 0,
                'covered_lines': 0,
                'files': []
            }

        module_stats[module]['total_statements'] += summary.get('num_statements', 0)
        module_stats[module]['covered_lines'] += summary.get('covered_lines', 0)
        module_stats[module]['files'].append({
            'file': file_path,
            'coverage': summary.get('percent_covered', 0.0),
            'statements': summary.get('num_statements', 0),
            'covered': summary.get('covered_lines', 0)
        })

    # 计算每个模块的覆盖率百分比
    for module, stats in module_stats.items():
        if stats['total_statements'] > 0:
            stats['coverage_percent'] = (stats['covered_lines'] / stats['total_statements']) * 100
        else:
            stats['coverage_percent'] = 0.0

    return module_stats


def identify_low_coverage_files(coverage_data, threshold=50.0):
    """识别低覆盖率文件"""
    if not coverage_data or 'files' not in coverage_data:
        return []

    low_coverage_files = []

    for file_path, file_data in coverage_data['files'].items():
        if 'summary' not in file_data:
            continue

        summary = file_data['summary']
        coverage = summary.get('percent_covered', 0.0)
        statements = summary.get('num_statements', 0)

        if statements > 0 and coverage < threshold:
            low_coverage_files.append({
                'file': file_path,
                'coverage': coverage,
                'statements': statements,
                'covered': summary.get('covered_lines', 0),
                'missing': summary.get('missing_lines', 0)
            })

    # 按覆盖率排序
    low_coverage_files.sort(key=lambda x: x['coverage'])
    return low_coverage_files


def verify_90_percent_target(coverage_percent):
    """验证90%覆盖率目标"""
    print(f"\n🎯 90%覆盖率目标验证")
    print("=" * 60)

    target = 90.0
    gap = target - coverage_percent

    print(f"📋 目标覆盖率: {target}%")
    print(f"📊 当前覆盖率: {coverage_percent:.2f}%")
    print(f"📈 覆盖率差距: {gap:.2f}%")

    if coverage_percent >= target:
        print("🎉 ✅ 恭喜！已达到90%覆盖率目标！")
        status = "PASSED"
    else:
        print(f"❌ 未达到90%覆盖率目标")
        print(f"📉 还需要提升 {gap:.2f} 个百分点")
        status = "FAILED"

    print(f"✨ 验证状态: {status}")
    return status == "PASSED"


def generate_coverage_report(coverage_data):
    """生成详细的覆盖率报告"""
    print("\n📊 RQA2025项目测试覆盖率详细报告")
    print("=" * 80)

    # 基本信息
    meta = coverage_data.get('meta', {})
    print(f"📅 报告生成时间: {meta.get('timestamp', 'N/A')}")
    print(f"🔧 Coverage.py版本: {meta.get('version', 'N/A')}")
    print(f"🌿 分支覆盖: {'是' if meta.get('branch_coverage', False) else '否'}")

    # 总体覆盖率
    overall_coverage = calculate_overall_coverage(coverage_data)
    print(f"\n📈 总体覆盖率: {overall_coverage:.2f}%")

    # 模块级覆盖率分析
    module_stats = analyze_coverage_by_module(coverage_data)

    print(f"\n📂 模块覆盖率分析")
    print("-" * 60)
    print(f"{'模块名称':<20} {'覆盖率':<10} {'语句数':<8} {'覆盖语句':<10} {'文件数':<6}")
    print("-" * 60)

    for module, stats in sorted(module_stats.items(), key=lambda x: x[1]['coverage_percent'], reverse=True):
        print(
            f"{module:<20} {stats['coverage_percent']:>7.2f}% {stats['total_statements']:>8d} {stats['covered_lines']:>10d} {len(stats['files']):>6d}")

    # 低覆盖率文件
    low_coverage_files = identify_low_coverage_files(coverage_data, threshold=30.0)
    if low_coverage_files:
        print(f"\n⚠️  低覆盖率文件 (< 30%)")
        print("-" * 80)
        print(f"{'文件路径':<50} {'覆盖率':<10} {'语句数':<8} {'覆盖语句':<10}")
        print("-" * 80)

        for file_info in low_coverage_files[:10]:  # 只显示前10个
            print(
                f"{file_info['file']:<50} {file_info['coverage']:>7.2f}% {file_info['statements']:>8d} {file_info['covered']:>10d}")

        if len(low_coverage_files) > 10:
            print(f"... 还有 {len(low_coverage_files) - 10} 个低覆盖率文件")

    return overall_coverage


def main():
    """主函数"""
    print("🚀 启动RQA2025项目90%测试覆盖率验证")
    print("=" * 60)

    # 查找覆盖率数据文件
    project_root = Path(__file__).parent
    coverage_files = [
        project_root / "reports" / "coverage.json",
        project_root / "reports" / "coverage_verification.json",
        project_root / "coverage.json",
        project_root / ".coverage.json"
    ]

    coverage_data = None
    used_file = None

    for coverage_file in coverage_files:
        if coverage_file.exists():
            coverage_data = load_coverage_data(coverage_file)
            if coverage_data:
                used_file = coverage_file
                break

    if not coverage_data:
        print("❌ 找不到有效的覆盖率数据文件！")
        print("请确保已运行测试并生成覆盖率报告。")
        print("\n建议运行：")
        print("pytest --cov=. --cov-report=json:reports/coverage.json")
        return False

    print(f"📁 使用覆盖率数据文件: {used_file}")

    # 生成报告
    overall_coverage = generate_coverage_report(coverage_data)

    # 验证90%目标
    target_achieved = verify_90_percent_target(overall_coverage)

    # 生成总结
    print(f"\n📋 验证总结")
    print("=" * 40)
    print(f"🎯 90%目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
    print(f"📊 当前覆盖率: {overall_coverage:.2f}%")

    if not target_achieved:
        gap = 90.0 - overall_coverage
        print(f"📈 覆盖率缺口: {gap:.2f}%")
        print(f"💡 建议优先提升低覆盖率模块的测试")

    print(f"\n⏰ 验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return target_achieved


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        sys.exit(1)

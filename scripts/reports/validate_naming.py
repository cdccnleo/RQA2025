#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 报告命名规范验证脚本
验证报告文件是否符合新的命名规范（不包含日期和版本信息）
"""

import re
from pathlib import Path


def validate_naming_convention():
    """验证命名规范"""
    reports_dir = Path("reports")
    valid_files = []
    invalid_files = []

    # 定义日期和版本模式
    date_version_patterns = [
        r'_\d{8}',  # YYYYMMDD
        r'_\d{8}_\d{6}',  # YYYYMMDD_HHMMSS
        r'_\d{8}_\d{4}',  # YYYYMMDD_HHMM
        r'_v\d+',  # v1, v2, v3...
        r'_final',  # final
        r'_draft',  # draft
        r'_review',  # review
        r'_\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'_\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
    ]

    # 遍历所有文件
    for file_path in reports_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
            filename = file_path.name

            # 检查是否包含日期或版本信息
            has_date_version = False
            for pattern in date_version_patterns:
                if re.search(pattern, filename):
                    has_date_version = True
                    break

            if has_date_version:
                invalid_files.append(str(file_path))
            else:
                valid_files.append(str(file_path))

    return valid_files, invalid_files


def print_validation_report(valid_files, invalid_files):
    """打印验证报告"""
    print("🔍 报告命名规范验证结果")
    print("=" * 50)

    print(f"✅ 符合规范的文件: {len(valid_files)} 个")
    print(f"❌ 不符合规范的文件: {len(invalid_files)} 个")
    print()

    if invalid_files:
        print("📋 不符合规范的文件列表:")
        for file_path in invalid_files[:10]:  # 只显示前10个
            print(f"  - {file_path}")
        if len(invalid_files) > 10:
            print(f"  ... 还有 {len(invalid_files) - 10} 个文件")
        print()

    # 计算符合率
    total_files = len(valid_files) + len(invalid_files)
    if total_files > 0:
        compliance_rate = (len(valid_files) / total_files) * 100
        print(f"📊 符合率: {compliance_rate:.1f}%")

        if compliance_rate >= 95:
            print("🎉 命名规范执行良好！")
        elif compliance_rate >= 80:
            print("⚠️  命名规范基本符合，建议进一步优化")
        else:
            print("🚨 命名规范需要改进")

    print()


def check_naming_patterns():
    """检查命名模式"""
    reports_dir = Path("reports")
    patterns = {}

    for file_path in reports_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
            filename = file_path.stem  # 不含扩展名

            # 分析命名模式
            parts = filename.split('_')
            if len(parts) >= 2:
                pattern = f"{parts[0]}_{parts[1]}"  # category_type
                patterns[pattern] = patterns.get(pattern, 0) + 1

    print("📊 命名模式统计")
    print("=" * 30)

    # 按出现次数排序
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

    for pattern, count in sorted_patterns[:10]:  # 显示前10个模式
        print(f"{pattern}: {count} 个文件")

    print()


def main():
    """主函数"""
    print("🚀 开始验证报告命名规范...")
    print()

    # 1. 验证命名规范
    valid_files, invalid_files = validate_naming_convention()

    # 2. 打印验证报告
    print_validation_report(valid_files, invalid_files)

    # 3. 检查命名模式
    check_naming_patterns()

    print("✅ 验证完成！")


if __name__ == "__main__":
    main()

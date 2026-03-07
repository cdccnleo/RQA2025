#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 修复剩余不符合命名规范的文件
处理重命名脚本遗漏的文件
"""

import re
from pathlib import Path
from datetime import datetime


def clean_filename(filename):
    """清理文件名，移除日期和版本信息"""
    # 移除常见的日期格式和版本标识
    patterns = [
        r'_\d{8}',  # YYYYMMDD
        r'_\d{8}_\d{6}',  # YYYYMMDD_HHMMSS
        r'_\d{8}_\d{4}',  # YYYYMMDD_HHMM
        r'_v\d+',  # v1, v2, v3...
        r'_final',  # final
        r'_draft',  # draft
        r'_review',  # review
        r'_\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'_\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'_\d{6}',  # HHMMSS
        r'_\d{4}',  # HHMM
    ]

    cleaned_name = filename
    for pattern in patterns:
        cleaned_name = re.sub(pattern, '', cleaned_name)

    # 移除末尾的下划线
    cleaned_name = cleaned_name.rstrip('_')

    return cleaned_name


def fix_remaining_files():
    """修复剩余的不符合规范的文件"""
    reports_dir = Path("reports")
    fixed_count = 0

    # 需要修复的文件列表
    files_to_fix = [
        "reports/project/deployment_completion_final_report.md",
        "reports/project/import_migration_final_report.md",
        "reports/project/progress/coverage_enhancement_report_20250804_105304.md",
        "reports/project/progress/coverage_enhancement_v3_report.md",
        "reports/project/progress/production_readiness_advancement_report_20250804_105304.md",
        "reports/project/progress/production_readiness_advancement_report_v3.md",
        "reports/project/progress/production_readiness_advancement_report_v4.md"
    ]

    for file_path_str in files_to_fix:
        file_path = Path(file_path_str)
        if file_path.exists():
            # 清理文件名
            new_name = clean_filename(file_path.stem) + file_path.suffix

            # 如果文件名没有变化，跳过
            if new_name == file_path.name:
                continue

            # 构建新路径
            new_path = file_path.parent / new_name

            # 如果目标文件已存在，添加时间戳
            if new_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{clean_filename(file_path.stem)}_{timestamp}{file_path.suffix}"
                new_path = file_path.parent / new_name

            # 重命名文件
            try:
                file_path.rename(new_path)
                fixed_count += 1
                print(f"📝 修复: {file_path.name} -> {new_name}")
            except Exception as e:
                print(f"❌ 修复失败: {file_path.name} - {e}")
        else:
            print(f"⚠️  文件不存在: {file_path_str}")

    print(f"✅ 共修复 {fixed_count} 个文件")


def main():
    """主函数"""
    print("🚀 开始修复剩余的不符合命名规范的文件...")

    # 修复文件
    fix_remaining_files()

    print("✅ 修复完成！")


if __name__ == "__main__":
    main()

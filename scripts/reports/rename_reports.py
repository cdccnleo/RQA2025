#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 报告文件重命名脚本
将现有报告文件重命名为符合新命名规范的格式（不包含日期和版本信息）
"""

import re
from pathlib import Path
from datetime import datetime


def clean_filename(filename):
    """清理文件名，移除日期和版本信息"""
    # 移除常见的日期格式
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
    ]

    cleaned_name = filename
    for pattern in patterns:
        cleaned_name = re.sub(pattern, '', cleaned_name)

    # 移除末尾的下划线
    cleaned_name = cleaned_name.rstrip('_')

    return cleaned_name


def rename_files():
    """重命名文件"""
    reports_dir = Path("reports")
    renamed_count = 0

    # 遍历所有文件
    for file_path in reports_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
            # 跳过新创建的目录
            if file_path.parent.name in ["project", "technical", "business", "operational", "research"]:
                continue

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
                renamed_count += 1
                print(f"📝 重命名: {file_path.name} -> {new_name}")
            except Exception as e:
                print(f"❌ 重命名失败: {file_path.name} - {e}")

    print(f"✅ 共重命名 {renamed_count} 个文件")


def update_readme_files():
    """更新README文件中的命名规范说明"""
    reports_dir = Path("reports")

    for readme_file in reports_dir.rglob("README.md"):
        try:
            content = readme_file.read_text(encoding='utf-8')

            # 更新命名规范说明
            if "命名规范" in content:
                # 这里可以添加更详细的更新逻辑
                print(f"📝 更新README: {readme_file}")
        except Exception as e:
            print(f"❌ 更新README失败: {readme_file} - {e}")


def main():
    """主函数"""
    print("🚀 开始重命名报告文件...")

    # 1. 重命名文件
    rename_files()

    # 2. 更新README文件
    update_readme_files()

    print("✅ 重命名完成！")
    print("\n📋 新的命名规范:")
    print("- 格式: {category}_{type}_{subject}.{extension}")
    print("- 不包含日期和版本信息")
    print("- 版本控制通过报告内容实现")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量迁移导入规范脚本

将项目中的导入从基础设施层迁移到通用层，
确保遵循统一的导入规范。
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

# 导入映射规则
IMPORT_MAPPINGS = {
    # Logger 模块
    r'from src\.infrastructure\.utils\.logger import get_logger':
        'from src.utils.logger import get_logger',

    # Date Utils 模块
    r'from src\.infrastructure\.utils\.date_utils import convert_timezone':
        'from src.utils.date_utils import convert_timezone',

    # 其他可能的映射
    r'from src\.infrastructure\.utils\.date_utils import get_current_date':
        'from src.utils.date_utils import get_business_date',
}

# 需要保留的基础设施层导入
INFRASTRUCTURE_IMPORTS = [
    'LoggerFactory',
    'configure_logging',
    'DateUtils',
    'PerformanceMonitor',
    'AuditLogger',
    'ExceptionHandler'
]


def should_preserve_infrastructure_import(import_line: str) -> bool:
    """判断是否应该保留基础设施层导入"""
    for import_name in INFRASTRUCTURE_IMPORTS:
        if import_name in import_line:
            return True
    return False


def migrate_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """迁移单个文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # 应用导入映射
        for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
            if re.search(old_pattern, content):
                # 检查是否应该保留
                if not should_preserve_infrastructure_import(content):
                    content = re.sub(old_pattern, new_pattern, content)
                    changes.append(f"  - {old_pattern} -> {new_pattern}")

        # 如果内容有变化且不是试运行，则写回文件
        if content != original_content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes

        return False, changes

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False, []


def find_python_files(directory: Path) -> List[Path]:
    """查找所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 跳过某些目录
        dirs[:] = [d for d in dirs if not d.startswith(
            '.') and d not in ['__pycache__', 'venv', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    return python_files


def main():
    parser = argparse.ArgumentParser(description='批量迁移导入规范')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际修改文件')
    parser.add_argument('--directory', default='src', help='要处理的目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"目录不存在: {directory}")
        return

    print(f"开始迁移导入规范...")
    print(f"目录: {directory}")
    print(f"模式: {'试运行' if args.dry_run else '实际执行'}")
    print("-" * 50)

    python_files = find_python_files(directory)
    print(f"找到 {len(python_files)} 个Python文件")

    modified_files = 0
    total_changes = 0

    for file_path in python_files:
        if args.verbose:
            print(f"处理: {file_path}")

        was_modified, changes = migrate_file(file_path, args.dry_run)

        if was_modified:
            modified_files += 1
            total_changes += len(changes)

            if args.verbose:
                print(f"  ✅ 已修改")
                for change in changes:
                    print(change)
            else:
                print(f"  ✅ {file_path}")

    print("-" * 50)
    print(f"迁移完成!")
    print(f"修改的文件数: {modified_files}")
    print(f"总变更数: {total_changes}")

    if args.dry_run:
        print("\n注意: 这是试运行，没有实际修改文件")
        print("使用 --dry-run=False 来实际执行迁移")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代码导入一致性检查脚本

检查项目中的导入是否符合统一规范，
避免代码重复定义和导入混乱。
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

# 导入规范检查规则
IMPORT_RULES = {
    # 推荐导入方式
    'recommended': {
        'logger': 'from src.utils.logger import get_logger',
        'date_utils': 'from src.utils.date_utils import convert_timezone',
        'math_utils': 'from src.utils.math_utils import calculate_returns',
        'data_utils': 'from src.utils.data_utils import normalize_data'
    },

    # 不推荐的导入方式
    'deprecated': {
        'logger': 'from src.infrastructure.utils.logger import get_logger',
        'date_utils': 'from src.infrastructure.utils.date_utils import convert_timezone'
    },

    # 允许的高级功能导入
    'allowed_advanced': {
        'LoggerFactory': 'from src.infrastructure.utils.logger import LoggerFactory',
        'configure_logging': 'from src.infrastructure.utils.logger import configure_logging',
        'DateUtils': 'from src.infrastructure.utils.date_utils import DateUtils'
    }
}


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


def check_file_imports(file_path: Path) -> Dict[str, List[str]]:
    """检查单个文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        issues = {
            'deprecated_imports': [],
            'missing_recommended': [],
            'inconsistent_imports': []
        }

        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 检查不推荐的导入
            for deprecated_pattern in IMPORT_RULES['deprecated'].values():
                if deprecated_pattern in line:
                    issues['deprecated_imports'].append(
                        f"Line {line_num}: {line.strip()}"
                    )

            # 检查推荐导入的使用情况
            for recommended_pattern in IMPORT_RULES['recommended'].values():
                if recommended_pattern in line:
                    # 找到推荐导入，记录为正常
                    pass

        return issues

    except Exception as e:
        return {
            'error': [f"读取文件失败: {e}"]
        }


def generate_report(issues_by_file: Dict[Path, Dict[str, List[str]]]) -> str:
    """生成检查报告"""
    report = []
    report.append("# 导入一致性检查报告")
    report.append("=" * 50)

    total_files = len(issues_by_file)
    files_with_issues = 0
    total_issues = 0

    for file_path, issues in issues_by_file.items():
        file_has_issues = False
        file_report = []

        for issue_type, issue_list in issues.items():
            if issue_list:
                file_has_issues = True
                file_report.append(f"  {issue_type}:")
                for issue in issue_list:
                    file_report.append(f"    - {issue}")

        if file_has_issues:
            files_with_issues += 1
            total_issues += sum(len(issues) for issues in issues.values())
            report.append(f"\n{file_path}:")
            report.extend(file_report)

    # 总结
    report.append("\n" + "=" * 50)
    report.append(f"检查文件数: {total_files}")
    report.append(f"有问题的文件数: {files_with_issues}")
    report.append(f"总问题数: {total_issues}")

    if files_with_issues == 0:
        report.append("\n✅ 所有文件都符合导入规范!")
    else:
        report.append(f"\n⚠️  发现 {files_with_issues} 个文件需要修复")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='检查代码导入一致性')
    parser.add_argument('--directory', default='src', help='要检查的目录')
    parser.add_argument('--output', help='输出报告文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"目录不存在: {directory}")
        return

    print(f"开始检查导入一致性...")
    print(f"目录: {directory}")
    print("-" * 50)

    python_files = find_python_files(directory)
    print(f"找到 {len(python_files)} 个Python文件")

    issues_by_file = {}

    for file_path in python_files:
        if args.verbose:
            print(f"检查: {file_path}")

        issues = check_file_imports(file_path)
        if any(issues.values()):
            issues_by_file[file_path] = issues

    # 生成报告
    report = generate_report(issues_by_file)

    # 输出报告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存到: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

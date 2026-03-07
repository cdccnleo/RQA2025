#!/usr/bin/env python3
"""
自动化代码修复脚本
"""

import os
import json
from pathlib import Path


def run_automated_fixes():
    """运行自动化修复"""
    print("🔧 开始自动化代码修复...")

    infra_dir = Path('src/infrastructure')

    fixes_applied = {
        'imports_sorted': sort_all_imports(infra_dir),
        'whitespace_cleaned': clean_whitespace(infra_dir),
        'docstrings_added': add_missing_docstrings(infra_dir)
    }

    # 保存修复结果
    with open('automated_fixes_results.json', 'w', encoding='utf-8') as f:
        json.dump(fixes_applied, f, indent=2, ensure_ascii=False)

    print("✅ 自动化修复完成")
    return fixes_applied


def sort_all_imports(infra_dir):
    """排序所有导入"""
    files_sorted = 0

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    if sort_file_imports(file_path):
                        files_sorted += 1
                except Exception:
                    continue

    return files_sorted


def sort_file_imports(file_path):
    """排序文件导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # 找到导入区域
        import_start = -1
        import_end = -1

        for i, line in enumerate(lines):
            if line.strip().startswith(('from ', 'import ')):
                if import_start == -1:
                    import_start = i
                import_end = i
            elif import_start != -1 and line.strip() and not line.strip().startswith('#'):
                break

        if import_start == -1:
            return False

        # 提取和排序导入行
        import_lines = lines[import_start:import_end + 1]
        import_lines.sort()
        lines[import_start:import_end + 1] = import_lines

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return True

    except Exception:
        return False


def clean_whitespace(infra_dir):
    """清理空白字符"""
    files_cleaned = 0

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content

                    # 清理行尾空格和多余空行
                    lines = content.split('\n')
                    cleaned_lines = []
                    prev_empty = False

                    for line in lines:
                        cleaned_line = line.rstrip()
                        is_empty = not cleaned_line

                        if not (is_empty and prev_empty):
                            cleaned_lines.append(cleaned_line)
                        prev_empty = is_empty

                    content = '\n'.join(cleaned_lines)

                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        files_cleaned += 1

                except Exception:
                    continue

    return files_cleaned


def add_missing_docstrings(infra_dir):
    """添加缺失的文档字符串"""
    files_fixed = 0

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否已有模块文档字符串
                    if '"""' not in content[:200]:
                        lines = content.split('\n')
                        if lines and lines[0].strip():
                            # 添加模块文档字符串
                            module_name = file_path.stem
                            docstring = f'"""\n{module_name} 模块\n\n提供 {module_name} 相关功能\n"""\n\n'
                            lines.insert(0, docstring)

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lines))

                            files_fixed += 1

                except Exception:
                    continue

    return files_fixed


if __name__ == "__main__":
    run_automated_fixes()

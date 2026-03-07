#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def fix_file_precise(filepath):
    """精确修复单个文件的flake8格式问题"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False

    original_content = content
    lines = content.split('\n')

    # 第一遍：修复注释格式 (E265)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 修复块注释格式 - 只处理真正的块注释
        if (stripped.startswith('#') and not stripped.startswith('# ') and
            len(stripped) > 1 and not stripped.startswith('#!') and
            not stripped.startswith('#-*-') and not stripped.startswith('# coding') and
                not stripped.startswith('# vim:')):
            lines[i] = line.replace('#', '# ', 1)

    # 第二遍：修复空行问题 (E302, E305)
    fixed_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()

        # 添加当前行
        fixed_lines.append(lines[i])

        # 检查下一行
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()

            # 处理顶级类或函数定义之间的空行 (E302)
            if ((current_line.startswith('class ') or current_line.startswith('def ')) and
                    (next_line.startswith('class ') or next_line.startswith('def '))):
                # 确保有2个空行
                empty_lines_needed = 2
                # 检查接下来的行
                check_index = i + 1
                actual_empty_lines = 0
                while (check_index < len(lines) and
                       (not lines[check_index].strip() or lines[check_index].strip().startswith('#'))):
                    if not lines[check_index].strip():
                        actual_empty_lines += 1
                    elif lines[check_index].strip().startswith('#'):
                        break  # 遇到注释行停止计数
                    check_index += 1

                # 添加需要的空行
                while actual_empty_lines < empty_lines_needed:
                    fixed_lines.append('')
                    actual_empty_lines += 1

            # 处理函数/类定义后的空行 (E305)
            elif (current_line and (current_line.startswith('class ') or current_line.startswith('def '))):
                # 检查接下来的内容
                check_index = i + 1
                actual_empty_lines = 0

                # 跳过空行和注释
                while (check_index < len(lines) and
                       (not lines[check_index].strip() or lines[check_index].strip().startswith('#'))):
                    if not lines[check_index].strip():
                        actual_empty_lines += 1
                    elif lines[check_index].strip().startswith('#'):
                        break
                    check_index += 1

                # 如果后面有实际代码，需要确保至少有1个空行
                if (check_index < len(lines) and lines[check_index].strip() and
                        not lines[check_index].strip().startswith('#')):
                    if actual_empty_lines < 1:
                        fixed_lines.append('')

        i += 1

    content = '\n'.join(fixed_lines)

    # 写入修复后的内容
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            return False
    else:
        return False


def process_directory(directory):
    """处理目录中的所有Python文件"""
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_file_precise(filepath):
                    count += 1

    print(f"总共修复了 {count} 个文件")


if __name__ == "__main__":
    # 处理src目录
    src_dir = "src"
    if os.path.exists(src_dir):
        print("开始精确修复src目录中的格式问题...")
        process_directory(src_dir)
    else:
        print("src目录不存在")

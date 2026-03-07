#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re


def fix_file_formatting_comprehensive(filepath):
    """全面修复单个文件的格式问题"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"跳过文件 {filepath} - 编码问题")
        return False

    original_content = content
    lines = content.split('\n')

    # 第一遍：修复注释格式
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 修复块注释格式
        if stripped.startswith('#') and not stripped.startswith('# ') and len(stripped) > 1:
            if stripped != '#' and not stripped.startswith('#!') and not stripped.startswith('#-*-'):
                if not re.search(r'#\s*coding[:=]\s*utf-?8', stripped, re.IGNORECASE):
                    lines[i] = line.replace('#', '# ', 1)

    # 第二遍：修复空行问题
    fixed_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()

        # 添加当前行
        fixed_lines.append(lines[i])

        # 检查是否需要添加空行
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()

            # 在顶级类或函数定义之间添加空行 (E302)
            if ((current_line.startswith('class ') or current_line.startswith('def ')) and
                    (next_line.startswith('class ') or next_line.startswith('def '))):
                # 检查是否已经有足够的空行
                empty_count = 0
                check_idx = i + 1
                while check_idx < len(lines) and not lines[check_idx].strip():
                    empty_count += 1
                    check_idx += 1

                # 需要2个空行
                while empty_count < 2:
                    fixed_lines.append('')
                    empty_count += 1

                # 跳过已经处理的空行
                i = check_idx - 1

            # 在类定义后添加空行 (E305)
            elif (current_line.startswith('class ') and next_line and
                  not next_line.startswith(' ') and not next_line.startswith('\t') and
                  not next_line.startswith('#')):
                # 检查是否已经有空行
                if not next_line:
                    pass  # 已经有空行了
                else:
                    fixed_lines.append('')

            # 在函数定义后添加空行 (E305)
            elif (current_line.startswith('def ') and next_line and
                  not next_line.startswith(' ') and not next_line.startswith('\t') and
                  not next_line.startswith('#')):
                if not next_line:
                    pass  # 已经有空行了
                else:
                    fixed_lines.append('')

        i += 1

    # 第三遍：清理多余的空行
    final_lines = []
    i = 0
    while i < len(final_lines):
        current_line = fixed_lines[i].strip()

        # 避免连续的空行超过2个
        if not current_line:
            # 检查前面有多少个连续的空行
            empty_count = 0
            j = i - 1
            while j >= 0 and not fixed_lines[j].strip():
                empty_count += 1
                j -= 1

            if empty_count < 2:
                final_lines.append(fixed_lines[i])
        else:
            final_lines.append(fixed_lines[i])

        i += 1

    content = '\n'.join(fixed_lines)

    # 写入修复后的内容
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"写入文件失败 {filepath}: {e}")
            return False
    else:
        return False


def process_directory_comprehensive(directory):
    """处理目录中的所有Python文件"""
    count = 0
    fixed_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_file_formatting_comprehensive(filepath):
                    count += 1
                    fixed_files.append(os.path.basename(filepath))

    print(f"总共修复了 {count} 个文件的格式问题")
    if fixed_files:
        print(f"修复的文件数量: {len(fixed_files)}")
        if len(fixed_files) > 20:
            print("修复的文件包括:", ', '.join(fixed_files[:10]), f"... 等 {len(fixed_files)} 个文件")
        else:
            print("修复的文件:", ', '.join(fixed_files))


if __name__ == "__main__":
    # 处理src目录
    src_dir = "src"
    if os.path.exists(src_dir):
        print("开始全面修复src目录中的格式问题...")
        print("这将修复E302（期望2个空行）和E305（函数/类后期望空行）错误")
        process_directory_comprehensive(src_dir)
    else:
        print("src目录不存在")

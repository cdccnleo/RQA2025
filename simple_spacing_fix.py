#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re


def fix_simple_spacing(filepath):
    """简单地修复空行问题"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False

    original_content = content

    # 修复顶级类/函数定义之间的空行
    # 在class或def之前添加空行，确保至少有1个空行
    content = re.sub(r'([^\n])\n(\s*)(class|def)', r'\1\n\n\2\3', content)

    # 清理多余的空行（超过3个连续空行）
    content = re.sub(r'\n\n\n\n+', '\n\n\n', content)

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


def process_directory_simple(directory):
    """简单处理目录中的所有Python文件"""
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_simple_spacing(filepath):
                    count += 1

    print(f"简单修复处理了 {count} 个文件")


if __name__ == "__main__":
    # 处理src目录
    src_dir = "src"
    if os.path.exists(src_dir):
        print("开始简单空行修复...")
        process_directory_simple(src_dir)
    else:
        print("src目录不存在")

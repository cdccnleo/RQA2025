#!/usr/bin/env python3

def find_duplicate_methods(file_path):
    """查找重复的方法定义"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    duplicates = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 检查是否是空的方法定义
        if line.startswith('    def ') and line.endswith(':') and not line.endswith('"""'):
            method_name = line.split('(')[0].strip()

            # 检查下一行是否是相同的完整方法定义
            if i + 1 < len(lines):
                next_line = lines[i + 1].rstrip()
                if (next_line.startswith('    def ') and
                    next_line.split('(')[0].strip() == method_name and
                        '"""' in next_line):
                    duplicates.append((i + 1, method_name))
                    i += 1  # 跳过下一个重复定义
        i += 1

    return duplicates


# 查找test_lru_cache.py中的重复方法定义
duplicates = find_duplicate_methods('tests/unit/infrastructure/cache/test_lru_cache.py')

if duplicates:
    print("找到的重复方法定义:")
    for line_num, method_name in duplicates:
        print(f"  第{line_num}行: {method_name}")
else:
    print("未找到重复方法定义")

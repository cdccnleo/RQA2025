#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复剩余的中文注释语法错误
"""

import os


def fix_chinese_comments(file_path):
    """修复单个文件中的中文注释"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单的字符串替换
        replacements = [
            ('"""测试', '"""Test'),
            ('"""初始化', '"""Initialize'),
            ('"""配置', '"""Configure'),
            ('"""执行', '"""Execute'),
            ('"""处理', '"""Process'),
            ('"""检查', '"""Check'),
            ('"""验证', '"""Validate'),
            ('"""创建', '"""Create'),
            ('"""获取', '"""Get'),
            ('"""设置', '"""Setup'),
            ('"""准备', '"""Prepare'),
            ('"""清理', '"""Clean'),
            ('"""连接', '"""Connect'),
            ('"""断开', '"""Disconnect'),
        ]

        original_content = content
        for chinese, english in replacements:
            content = content.replace(chinese, english)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'✅ 修复了文件: {file_path}')
            return True
        else:
            print(f'⚪ 文件无需修复: {file_path}')
            return False

    except Exception as e:
        print(f'❌ 处理文件 {file_path} 时出错: {e}')
        return False


def main():
    """主函数"""
    files_to_fix = [
        'tests/unit/infrastructure/utils/test_dynamic_executor.py',
        'tests/unit/infrastructure/utils/test_file_system.py',
        'tests/unit/infrastructure/utils/test_utils.py'
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_chinese_comments(file_path):
                fixed_count += 1
        else:
            print(f'⚠️  文件不存在: {file_path}')

    print(f"\n修复了 {fixed_count} 个文件")


if __name__ == "__main__":
    main()

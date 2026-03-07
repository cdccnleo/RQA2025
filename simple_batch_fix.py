#!/usr/bin/env python3
"""
简单的批量修复脚本 - 修复最常见的语法错误
"""

import os


def fix_common_syntax_errors(file_path):
    """修复常见的语法错误"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 修复重复的方法定义
        if line.startswith('    def ') and line.endswith(':') and not line.endswith('"""'):
            method_name = line.split('(')[0].strip()
            # 检查下一行是否是相同的完整方法定义
            if i + 1 < len(lines):
                next_line = lines[i + 1].rstrip()
                if (next_line.startswith('    def ') and
                    next_line.split('(')[0].strip() == method_name and
                        '"""' in next_line):
                    # 跳过空的方法定义
                    i += 1
                    continue

        # 修复方法内的缩进问题
        elif line.strip().startswith(('self.', 'assert', 'from ', 'import ', 'try:', 'except', 'if ', 'for ', 'async with')):
            # 检查是否缺少缩进
            if not line.startswith(' ') and fixed_lines:
                # 查找最近的方法定义
                for j in range(len(fixed_lines) - 1, -1, -1):
                    if fixed_lines[j].strip().startswith('def ') or fixed_lines[j].strip().startswith('async def '):
                        # 添加适当的缩进
                        line = '        ' + line.lstrip()
                        break

        fixed_lines.append(line)
        i += 1

    return '\n'.join(fixed_lines)


def process_file(file_path):
    """处理单个文件"""
    try:
        print(f"处理文件: {file_path}")

        # 备份原文件
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

        # 修复常见错误
        content = fix_common_syntax_errors(file_path)

        # 验证语法
        try:
            compile(content, file_path, 'exec')
            print("  ✅ 修复成功")

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True
        except SyntaxError as e:
            print(f"  ❌ 仍需手动修复: {e}")
            return False

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False


def main():
    """主函数"""
    print("开始批量修复基础设施层测试文件的常见语法错误...")

    # 处理所有错误的文件（分批处理）
    error_files = [
        'tests/unit/infrastructure/cache/test_cache_client_sdk.py',
        'tests/unit/infrastructure/cache/test_cache_client_sdk_simple.py',
        'tests/unit/infrastructure/cache/test_cache_core_components.py',
    ]

    fixed_count = 0

    for file_path in error_files:
        if process_file(file_path):
            fixed_count += 1

    print(f"\n处理完成! 成功修复: {fixed_count}/{len(error_files)} 个文件")


if __name__ == "__main__":
    main()

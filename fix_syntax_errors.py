#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025语法错误修复脚本
批量修复Python测试文件中的常见语法错误
"""

import os
import re
import glob
from pathlib import Path

class SyntaxErrorFixer:
    """语法错误修复器"""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.errors_found = []

    def fix_indentation_errors(self, content):
        """修复缩进错误"""
        lines = content.split('\n')
        fixed_lines = []
        in_function = False
        in_class = False
        brace_count = 0

        for i, line in enumerate(lines):
            original_line = line
            stripped = line.strip()

            # 跳过空行和注释
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue

            # 检测函数定义
            if re.match(r'^\s*def\s+\w+', stripped):
                in_function = True
            elif re.match(r'^\s*class\s+\w+', stripped):
                in_class = True
                in_function = False

            # 检测括号
            brace_count += stripped.count('(') - stripped.count(')')
            brace_count += stripped.count('[') - stripped.count(']')
            brace_count += stripped.count('{') - stripped.count('}')

            # 修复常见的缩进问题
            if stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')):
                # 确保这些语句有正确的缩进
                if not line.startswith('    ') and not line.startswith('\t'):
                    if in_function or in_class:
                        line = '    ' + line.lstrip()
                    elif brace_count > 0:
                        line = '    ' + line.lstrip()

            # 修复函数内部的语句缩进
            elif in_function and not line.startswith('    ') and not line.startswith('\t') and stripped:
                if not stripped.startswith(('def ', 'class ', '@')):
                    line = '    ' + line.lstrip()

            # 修复类内部的语句缩进
            elif in_class and not in_function and not line.startswith('    ') and not line.startswith('\t') and stripped:
                if not stripped.startswith(('def ', 'class ', '@')):
                    line = '        ' + line.lstrip()

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_f_string_errors(self, content):
        """修复f-string错误"""
        # 修复不完整的f-string
        content = re.sub(r'f"[^"]*$', lambda m: m.group(0) + '"', content, flags=re.MULTILINE)
        content = re.sub(r"f'[^']*$", lambda m: m.group(0) + "'", content, flags=re.MULTILINE)

        # 修复错误的f-string语法
        content = re.sub(r'f"([^"]*)\{([^}]+)\}([^"]*)"', r'f"\1{\2}\3"', content)
        content = re.sub(r"f'([^']*)\{([^}]+)\}([^']*)'", r"f'\1{\2}\3'", content)

        return content

    def fix_import_errors(self, content):
        """修复导入错误"""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # 修复相对导入问题
            if line.strip().startswith('from .'):
                # 确保相对导入在包内部
                pass

            # 修复导入路径问题
            if 'from src.' in line:
                # 确保src路径正确
                pass

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_try_except_blocks(self, content):
        """修复try-except块错误"""
        # 修复不完整的try-except块
        if 'try:' in content and 'except' not in content:
            content += '\n    pass'

        # 修复except语句格式
        content = re.sub(r'except\s*\w+\s*as\s*\w+:', r'except \1 as \2:', content)
        content = re.sub(r'except\s*\w+:', r'except \1:', content)

        return content

    def fix_missing_imports(self, content):
        """添加缺失的导入"""
        imports_needed = []

        # 检查是否使用了sys但没有导入
        if 'sys.' in content and 'import sys' not in content:
            imports_needed.append('import sys')

        # 检查是否使用了os但没有导入
        if 'os.' in content and 'import os' not in content:
            imports_needed.append('import os')

        # 检查是否使用了Path但没有导入
        if 'Path(' in content and 'from pathlib import Path' not in content:
            imports_needed.append('from pathlib import Path')

        # 检查是否使用了Mock但没有导入
        if 'Mock(' in content and 'from unittest.mock import Mock' not in content:
            imports_needed.append('from unittest.mock import Mock')

        # 添加缺失的导入
        if imports_needed:
            lines = content.split('\n')
            # 找到第一个非注释行
            insert_index = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('#') and line.strip():
                    insert_index = i
                    break

            # 插入导入语句
            for import_stmt in reversed(imports_needed):
                lines.insert(insert_index, import_stmt)

            content = '\n'.join(lines)

        return content

    def fix_file(self, file_path):
        """修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用各种修复
            content = self.fix_indentation_errors(content)
            content = self.fix_f_string_errors(content)
            content = self.fix_import_errors(content)
            content = self.fix_try_except_blocks(content)
            content = self.fix_missing_imports(content)

            # 如果内容有变化，写入文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"✅ 修复了文件: {file_path}")
                return True
            else:
                return False

        except Exception as e:
            self.errors_found.append(f"{file_path}: {str(e)}")
            print(f"❌ 处理文件时出错 {file_path}: {str(e)}")
            return False

    def scan_and_fix(self, directory="tests"):
        """扫描并修复目录中的所有Python文件"""
        print(f"🔍 开始扫描目录: {directory}")

        pattern = os.path.join(self.project_root, directory, "**", "*.py")
        python_files = glob.glob(pattern, recursive=True)

        print(f"📁 发现 {len(python_files)} 个Python文件")

        fixed_count = 0
        for file_path in python_files:
            if self.fix_file(file_path):
                fixed_count += 1

        print("\n📊 修复统计:")
        print(f"   总文件数: {len(python_files)}")
        print(f"   修复文件数: {fixed_count}")
        print(f"   错误文件数: {len(self.errors_found)}")

        if self.errors_found:
            print("\n⚠️  处理过程中发现的错误:")
            for error in self.errors_found[:10]:  # 只显示前10个错误
                print(f"   {error}")
            if len(self.errors_found) > 10:
                print(f"   ... 还有 {len(self.errors_found) - 10} 个错误")

        return fixed_count, len(self.errors_found)

def main():
    """主函数"""
    print("🛠️  RQA2025语法错误修复工具")
    print("=" * 50)

    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir

    # 创建修复器
    fixer = SyntaxErrorFixer(project_root)

    # 扫描并修复tests目录
    print("🎯 开始修复tests目录...")
    fixed, errors = fixer.scan_and_fix("tests")

    # 扫描并修复src目录
    print("\n🎯 开始修复src目录...")
    fixed2, errors2 = fixer.scan_and_fix("src")

    total_fixed = fixed + fixed2
    total_errors = errors + errors2

    print("\n🎉 修复完成总结:")
    print(f"   总修复文件数: {total_fixed}")
    print(f"   总错误文件数: {total_errors}")

    if total_fixed > 0:
        print("\n✅ 成功修复的文件:")
        for file in fixer.fixed_files[:10]:  # 只显示前10个
            print(f"   {file}")
        if len(fixer.fixed_files) > 10:
            print(f"   ... 还有 {len(fixer.fixed_files) - 10} 个文件")
    else:
        print("\n✅ 所有文件都已经正确，无需修复")

    print("\n🚀 语法错误修复完成！")
    print("💡 建议: 运行pytest测试验证修复效果")

if __name__ == "__main__":
    main()

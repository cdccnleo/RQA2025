#!/usr/bin/env python3
"""
RQA2025 批量语法错误修复脚本

批量修复项目中的语法错误和导入问题
"""

import os
import re
import glob
from pathlib import Path


class BatchSyntaxFixer:
    """批量语法错误修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.errors = []

    def find_files_with_errors(self) -> list:
        """查找包含语法错误的文件"""
        error_files = []

        # 基于flake8输出的错误模式查找文件
        error_patterns = [
            "src/infrastructure/**/*.py",
            "src/ml/**/*.py",
            "src/optimization/**/*.py",
            "src/risk/**/*.py",
            "src/strategy/**/*.py",
            "src/streaming/**/*.py",
            "src/testing/**/*.py",
            "src/trading/**/*.py"
        ]

        for pattern in error_patterns:
            for file_path in glob.glob(str(self.project_root / pattern), recursive=True):
                if os.path.isfile(file_path):
                    error_files.append(Path(file_path))

        return error_files

    def fix_indentation_errors(self, file_path: Path) -> bool:
        """修复缩进错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复常见的缩进错误模式
            # 模式1: return {} 后面跟着字典内容但缩进错误
            content = re.sub(
                r'return \{\}\s*(\s*)"([^"]+)":\s*([^,\n]+),',
                r'return {\n\1"\2": \3,',
                content,
                flags=re.MULTILINE
            )

            # 模式2: 函数体中错误的缩进
            lines = content.split('\n')
            fixed_lines = []
            in_function = False
            indent_level = 0

            for i, line in enumerate(lines):
                stripped = line.strip()

                # 检测函数定义
                if stripped.startswith('def ') or stripped.startswith('class '):
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    continue

                # 检测代码块开始
                if stripped.startswith(('if ', 'for ', 'while ', 'try:', 'except:', 'finally:')):
                    if in_function:
                        expected_indent = indent_level + 4
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent != expected_indent:
                            # 修复缩进
                            line = ' ' * expected_indent + stripped

                # 检测return语句后的字典
                if stripped.startswith('return {') and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"') and not next_line.endswith(','):
                        # 修复字典格式
                        lines[i + 1] = ' ' * (len(line) - len(line.lstrip()) + 4) + next_line + ','

                fixed_lines.append(line)

            content = '\n'.join(fixed_lines)

            # 写入修复后的内容
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"✅ 修复文件: {file_path}")
                return True

        except Exception as e:
            self.errors.append(f"修复文件 {file_path} 时出错: {str(e)}")
            return False

        return False

    def fix_missing_imports(self, file_path: Path) -> bool:
        """修复缺失的导入"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            imports_to_add = []

            # 检查是否使用了Dict, Any但没有导入
            if ('Dict[' in content or 'Dict[str' in content) and 'from typing import' not in content:
                if 'Dict' not in content.split('from typing import')[0]:
                    imports_to_add.append('Dict')

            if ('Any' in content) and 'from typing import' not in content:
                if 'Any' not in content.split('from typing import')[0]:
                    imports_to_add.append('Any')

            if ('Optional[' in content) and 'from typing import' not in content:
                if 'Optional' not in content.split('from typing import')[0]:
                    imports_to_add.append('Optional')

            if ('List[' in content) and 'from typing import' not in content:
                if 'List' not in content.split('from typing import')[0]:
                    imports_to_add.append('List')

            # 检查其他常见导入
            if 'logger' in content and 'import logging' not in content:
                imports_to_add.append('logging')

            if 'time' in content and 'import time' not in content:
                imports_to_add.append('time')

            if 'json' in content and 'import json' not in content:
                imports_to_add.append('json')

            if 'ABC' in content and 'from abc import' not in content:
                imports_to_add.append('ABC')

            if 'abstractmethod' in content and 'from abc import' not in content:
                imports_to_add.append('abstractmethod')

            # 添加缺失的导入
            if imports_to_add:
                # 查找现有导入的位置
                lines = content.split('\n')
                insert_index = 0

                # 找到最后一个导入语句的位置
                for i, line in enumerate(lines):
                    if line.startswith(('import ', 'from ')):
                        insert_index = i + 1
                    elif line.strip() and not line.startswith('#'):
                        break

                # 构造导入语句
                if 'Dict' in imports_to_add or 'Any' in imports_to_add or 'Optional' in imports_to_add or 'List' in imports_to_add:
                    typing_imports = [imp for imp in ['Dict', 'Any',
                                                      'Optional', 'List'] if imp in imports_to_add]
                    if typing_imports:
                        lines.insert(
                            insert_index, f"from typing import {', '.join(typing_imports)}")
                        insert_index += 1

                if 'logging' in imports_to_add:
                    lines.insert(insert_index, "import logging")
                    insert_index += 1

                if 'time' in imports_to_add:
                    lines.insert(insert_index, "import time")
                    insert_index += 1

                if 'json' in imports_to_add:
                    lines.insert(insert_index, "import json")
                    insert_index += 1

                if 'ABC' in imports_to_add or 'abstractmethod' in imports_to_add:
                    abc_imports = [imp for imp in [
                        'ABC', 'abstractmethod'] if imp in imports_to_add]
                    if abc_imports:
                        lines.insert(insert_index, f"from abc import {', '.join(abc_imports)}")
                        insert_index += 1

                content = '\n'.join(lines)

            # 写入修复后的内容
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"✅ 添加导入: {file_path}")
                return True

        except Exception as e:
            self.errors.append(f"修复导入 {file_path} 时出错: {str(e)}")
            return False

        return False

    def run_batch_fix(self) -> dict:
        """运行批量修复"""
        print("🔧 开始批量修复语法错误...")

        files = self.find_files_with_errors()
        print(f"📁 找到 {len(files)} 个文件需要检查")

        fixed_count = 0

        for file_path in files:
            # 修复缩进错误
            if self.fix_indentation_errors(file_path):
                fixed_count += 1

            # 修复导入错误
            if self.fix_missing_imports(file_path):
                fixed_count += 1

        print("\n📊 批量修复完成:")
        print(f"   - 处理文件数: {len(files)}")
        print(f"   - 修复文件数: {len(set(self.fixed_files))}")
        print(f"   - 错误数量: {len(self.errors)}")

        if self.errors:
            print("\n❌ 修复错误:")
            for error in self.errors[:5]:  # 只显示前5个错误
                print(f"   - {error}")
            if len(self.errors) > 5:
                print(f"   - ... 还有 {len(self.errors) - 5} 个错误")

        return {
            'total_files': len(files),
            'fixed_files': len(set(self.fixed_files)),
            'errors': len(self.errors),
            'error_details': self.errors
        }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    fixer = BatchSyntaxFixer(project_root)
    result = fixer.run_batch_fix()

    print(f"\n🎯 批量修复结果: {result['fixed_files']}/{result['total_files']} 文件已修复")


if __name__ == "__main__":
    main()

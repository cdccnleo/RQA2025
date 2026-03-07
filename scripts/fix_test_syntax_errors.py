#!/usr/bin/env python3
"""
基础设施层测试语法错误自动修复脚本

自动修复测试文件中的常见语法错误：
1. 错误的导入路径
2. 缺少的变量定义
3. 缩进错误
4. try/except块语法错误
"""

import os
import re
import glob
from pathlib import Path


class TestSyntaxFixer:
    """测试语法错误修复器"""

    def __init__(self, test_dir="tests"):
        self.test_dir = Path(test_dir)
        self.fixed_files = []

    def fix_import_syntax_errors(self, content):
        """修复导入语法错误"""
        # 修复错误的导入路径
        content = re.sub(
            r'from src\.src\\.*import src\\.*',
            lambda m: m.group(0).replace('src.src\\', 'src.').replace('\\', '.'),
            content
        )

        # 修复不完整的try/except块
        content = re.sub(
            r'(\n\s*)try:\s*\n\s*(.*?)\n\s*except ImportError:\s*\n\s*# Module not available\s*\n\s*pass\s*\n\s*(\w+)\s*=',
            r'\1try:\n\2\n\1except ImportError:\n\1    # Module not available\n\1    \3 = None',
            content,
            flags=re.MULTILINE
        )

        return content

    def fix_indentation_errors(self, content):
        """修复缩进错误"""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # 修复except块后的缩进
            if re.match(r'^\s*except ImportError:', line):
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    # 空行，添加正确的缩进
                    fixed_lines.append(line)
                    if i + 2 < len(lines) and lines[i + 2].strip():
                        next_line = lines[i + 2]
                        if not next_line.startswith('    ') and not next_line.startswith('\t'):
                            lines[i + 2] = '    ' + next_line.lstrip()
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_variable_definitions(self, content):
        """修复变量定义问题"""
        # 在try/except块后添加变量定义
        content = re.sub(
            r'(except ImportError:\s*\n\s*# Module not available\s*\n\s*pass)',
            r'\1\n    # Define fallback\n    pass',
            content
        )

        return content

    def process_file(self, file_path):
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用各种修复
            content = self.fix_import_syntax_errors(content)
            content = self.fix_indentation_errors(content)
            content = self.fix_variable_definitions(content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(file_path)
                print(f"✅ Fixed: {file_path}")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    def find_and_fix_test_files(self):
        """查找并修复测试文件"""
        patterns = [
            "tests/unit/infrastructure/**/*.py",
            "tests/unit/infrastructure/*/*.py",
            "tests/unit/infrastructure/**/*.py.backup"
        ]

        for pattern in patterns:
            for file_path in glob.glob(pattern, recursive=True):
                if os.path.isfile(file_path):
                    self.process_file(file_path)

    def generate_report(self):
        """生成修复报告"""
        print(f"\n{'='*60}")
        print("测试文件语法错误修复报告")
        print(f"{'='*60}")
        print(f"修复的文件数量: {len(self.fixed_files)}")
        print("\n修复的文件列表:")
        for file in self.fixed_files:
            print(f"  - {file}")
        print(f"{'='*60}")


def main():
    """主函数"""
    fixer = TestSyntaxFixer()
    print("🔧 开始自动修复测试文件语法错误...")

    fixer.find_and_fix_test_files()
    fixer.generate_report()

    print("\n🎉 语法错误修复完成！")
    print("建议下一步: 运行pytest验证修复效果")


if __name__ == "__main__":
    main()

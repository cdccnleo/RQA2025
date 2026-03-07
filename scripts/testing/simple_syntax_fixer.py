#!/usr/bin/env python3
"""
简单语法修复脚本
快速修复基本的语法错误
"""

import os
import re
import sys
import ast
import glob
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleSyntaxFixer:
    """简单语法修复器"""

    def __init__(self):
        self.fixed_files = []

    def fix_all_syntax_errors(self):
        """修复所有语法错误"""
        test_layers = [
            'tests/unit/infrastructure',
            'tests/unit/features',
            'tests/unit/ml',
            'tests/unit/trading',
            'tests/unit/risk',
            'tests/unit/core'
        ]

        total_fixed = 0

        for layer_path in test_layers:
            if not os.path.exists(layer_path):
                continue

            test_files = glob.glob(f'{layer_path}/**/*comprehensive.py', recursive=True)

            for test_file in test_files:
                if self._fix_single_file(test_file):
                    total_fixed += 1

        return total_fixed

    def _fix_single_file(self, file_path: str) -> bool:
        """修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复常见问题
            content = self._fix_string_issues(content)
            content = self._fix_import_issues(content)

            # 验证语法
            try:
                ast.parse(content)
                syntax_ok = True
            except SyntaxError:
                syntax_ok = False

            if syntax_ok and content != original_content:
                # 备份并写入
                backup_path = f"{file_path}.simple_backup"
                if not os.path.exists(backup_path):
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.fixed_files.append(file_path)
                return True

        except Exception as e:
            logger.error(f"修复文件 {file_path} 时出错: {e}")

        return False

    def _fix_string_issues(self, content: str) -> str:
        """修复字符串问题"""
        # 替换有问题的字符串模式
        content = content.replace('print("\\n===', 'print("\\n===")
        content=content.replace('===\\n")', '===")')
        content=content.replace('print(f"\\n===', 'print(f"\\n===')
        content=content.replace('===\\n")', '===")')

        return content

    def _fix_import_issues(self, content: str) -> str:
        """修复导入问题"""
        lines=content.split('\n')
        fixed_lines=[]

        i=0
        while i < len(lines):
            line=lines[i]

            # 修复不完整的try块
            if line.strip() == 'try:' and i + 1 < len(lines):
                next_line=lines[i + 1]
                if 'from src.' in next_line and 'import' in next_line:
                    # 检查是否有except块
                    has_except=False
                    for j in range(i + 2, min(i + 7, len(lines))):
                        if lines[j].strip().startswith('except'):
                            has_except=True
                            break

                    if not has_except:
                        # 添加except块
                        fixed_lines.append(line)
                        fixed_lines.append(next_line)
                        fixed_lines.append("except ImportError:")
                        fixed_lines.append("    # Module not available")
                        fixed_lines.append("    pass")
                        i += 1  # 跳过下一行
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

            i += 1

        return '\n'.join(fixed_lines)

def main():
    """主函数"""
    fixer=SimpleSyntaxFixer()
    total_fixed=fixer.fix_all_syntax_errors()

    print(f"\n✅ 简单语法修复完成!")
    print(f"修复了 {total_fixed} 个文件")

    if fixer.fixed_files:
        print("\n修复的文件列表:")
        for file in fixer.fixed_files[:10]:
            print(f"  ✓ {file}")

if __name__ == "__main__":
    main()

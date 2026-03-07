#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复缩进错误

将不正确缩进的pass语句修复为正确缩进
"""

import os
import re
from pathlib import Path


class IndentationFixer:
    """缩进修复器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_path = self.project_root / 'tests' / 'unit' / 'infrastructure' / 'health'

    def fix_indentation(self):
        """修复缩进问题"""
        fixed_files = 0

        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # 修复缩进问题：将顶格的pass语句改为正确缩进
                lines = content.split('\n')
                fixed_lines = []

                for i, line in enumerate(lines):
                    # 如果这一行是"pass  # Empty skip replaced"且没有正确缩进
                    if line.strip() == 'pass  # Empty skip replaced':
                        # 查找前一行，确定正确的缩进级别
                        if i > 0:
                            prev_line = lines[i-1]
                            # 计算前一行的缩进
                            indent_match = re.match(r'^(\s*)', prev_line)
                            if indent_match:
                                indent = indent_match.group(1)
                                # 使用相同的缩进
                                fixed_lines.append(indent + 'pass  # Empty skip replaced')
                            else:
                                # 默认4个空格缩进
                                fixed_lines.append('    pass  # Empty skip replaced')
                        else:
                            fixed_lines.append('    pass  # Empty skip replaced')
                    else:
                        fixed_lines.append(line)

                content = '\n'.join(fixed_lines)

                # 如果内容有变化，保存文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_files += 1
                    print(f"✅ 修复缩进: {py_file.relative_to(self.project_root)}")

            except Exception as e:
                print(f"❌ 处理文件 {py_file} 时出错: {e}")

        return fixed_files

    def verify_fixes(self):
        """验证修复结果"""
        import subprocess
        import sys

        try:
            # 尝试运行测试收集
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/infrastructure/health/',
                '--collect-only', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)

            if result.returncode == 0:
                print("✅ 所有缩进错误已修复！")
                return True
            else:
                print("❌ 仍有缩进错误")
                print(result.stderr[:500])
                return False

        except subprocess.TimeoutExpired:
            print("❌ 验证超时")
            return False
        except Exception as e:
            print(f"❌ 验证错误: {e}")
            return False

    def run_fix(self):
        """运行修复"""
        print("🔧 开始修复缩进错误...")
        print("=" * 60)

        # 执行修复
        fixed_files = self.fix_indentation()
        print(f"✅ 修复了 {fixed_files} 个文件的缩进")

        # 验证修复
        print("🔍 验证修复结果...")
        success = self.verify_fixes()

        print("\n" + "=" * 60)
        print("🎉 缩进修复完成！")

        return success


def main():
    """主函数"""
    fixer = IndentationFixer()
    success = fixer.run_fix()

    if success:
        print("\n🎉 缩进错误修复成功！")
        return 0
    else:
        print("\n⚠️ 修复完成但仍需进一步处理")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
安全缩进修复器

安全地修复Python文件的缩进问题，避免破坏代码结构
"""

import os
import ast
from pathlib import Path
from typing import List, Tuple


class SafeIndentationFixer:
    """安全缩进修复器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.backup_created = False

    def create_backup(self) -> bool:
        """创建备份文件"""
        if self.file_path.exists():
            backup_path = str(self.file_path) + '.backup'
            try:
                import shutil
                shutil.copy2(str(self.file_path), backup_path)
                self.backup_created = True
                print(f"📋 已创建备份: {backup_path}")
                return True
            except Exception as e:
                print(f"❌ 创建备份失败: {e}")
                return False
        return False

    def check_syntax(self) -> Tuple[bool, str]:
        """检查文件语法"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"其他错误: {str(e)}"

    def fix_indentation_issues(self) -> bool:
        """修复缩进问题"""
        try:
            # 创建备份
            if not self.create_backup():
                return False

            # 读取文件内容
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 修复缩进问题
            fixed_lines = self._fix_lines_indentation(lines)

            # 检查修复后的语法
            temp_content = ''.join(fixed_lines)
            try:
                ast.parse(temp_content)
            except SyntaxError as e:
                print(f"❌ 修复后语法仍有问题: {e}")
                # 恢复备份
                if self.backup_created:
                    import shutil
                    shutil.copy2(str(self.file_path) + '.backup', str(self.file_path))
                    print("📋 已恢复备份文件")
                return False

            # 保存修复后的内容
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)

            print(f"✅ 修复成功: {self.file_path}")
            return True

        except Exception as e:
            print(f"❌ 修复失败: {e}")
            # 恢复备份
            if self.backup_created:
                import shutil
                shutil.copy2(str(self.file_path) + '.backup', str(self.file_path))
                print("📋 已恢复备份文件")
            return False

    def _fix_lines_indentation(self, lines: List[str]) -> List[str]:
        """修复行缩进"""
        fixed_lines = []
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped:
                # 空行保持不变
                fixed_lines.append(line)
                continue

            # 计算当前行的缩进
            current_indent = len(line) - len(stripped)

            # 分析行内容
            if stripped.startswith('class '):
                # 类定义
                fixed_lines.append('class ' + stripped[6:])
                indent_stack = [4]  # 重置缩进栈

            elif stripped.startswith('def '):
                # 函数定义
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0

                fixed_lines.append(' ' * expected_indent + stripped)
                indent_stack.append(expected_indent + 4)

            elif stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ')):
                # 控制结构开始
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0

                fixed_lines.append(' ' * expected_indent + stripped)
                indent_stack.append(expected_indent + 4)

            elif stripped.startswith(('elif ', 'else:', 'except ', 'finally:')):
                # 控制结构继续
                if len(indent_stack) >= 2:
                    expected_indent = indent_stack[-2]
                elif indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0

                fixed_lines.append(' ' * expected_indent + stripped)

            elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                # 控制结构结束
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0

                fixed_lines.append(' ' * expected_indent + stripped)

            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # 文档字符串
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 4  # 类或模块级别的文档字符串

                fixed_lines.append(' ' * expected_indent + stripped)

            else:
                # 普通语句
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 4  # 假设在类或函数中

                fixed_lines.append(' ' * expected_indent + stripped)

        return fixed_lines

    def cleanup_backup(self):
        """清理备份文件"""
        if self.backup_created:
            backup_path = str(self.file_path) + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)
                print(f"🗑️ 已清理备份: {backup_path}")


def main():
    """主函数"""
    import sys

    if len(sys.argv) != 2:
        print("用法: python safe_indentation_fixer.py <文件路径>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    fixer = SafeIndentationFixer(file_path)

    # 检查原始语法
    syntax_ok, error_msg = fixer.check_syntax()
    if syntax_ok:
        print("✅ 文件语法已正确，无需修复")
        return

    print(f"❌ 发现语法错误: {error_msg}")

    # 尝试修复
    if fixer.fix_indentation_issues():
        print("🎉 修复完成")
        fixer.cleanup_backup()
    else:
        print("❌ 修复失败")


if __name__ == "__main__":
    main()

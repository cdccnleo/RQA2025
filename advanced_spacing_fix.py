#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


class SpacingRules:
    """空行规则定义"""

    @staticmethod
    def is_definition_line(line: str) -> bool:
        """检查是否为函数或类定义行"""
        stripped = line.strip()
        return stripped.startswith('def ') or stripped.startswith('class ')

    @staticmethod
    def is_code_line(line: str) -> bool:
        """检查是否为代码行（非空行、非注释、非文档字符串）"""
        stripped = line.strip()
        return (stripped and
                not stripped.startswith('#') and
                not stripped.startswith('"""') and
                not stripped.startswith("'''"))

    @staticmethod
    def is_indented_code(line: str) -> bool:
        """检查是否为缩进的代码行"""
        stripped = line.strip()
        return stripped and (line.startswith(' ') or line.startswith('\t'))


class SpacingFixer:
    """空行修复器"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.original_content = None
        self.lines = []

    def read_file(self) -> bool:
        """读取文件内容"""
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                self.original_content = f.read()
                self.lines = self.original_content.split('\n')
            return True
        except UnicodeDecodeError:
            return False

    def count_empty_lines_ahead(self, start_pos: int) -> int:
        """计算指定位置前方的空行数"""
        empty_lines = 0
        check_pos = start_pos
        while check_pos < len(self.lines) and not self.lines[check_pos].strip():
            empty_lines += 1
            check_pos += 1
        return empty_lines

    def add_empty_lines(self, fixed_lines: list, count: int, position: int) -> int:
        """添加指定数量的空行"""
        for _ in range(count):
            fixed_lines.append('')
        return position + count - 1  # 返回跳过的位置

    def apply_definition_spacing_rule(self, fixed_lines: list, current_line: str,
                                      next_line: str, position: int) -> int:
        """应用函数/类定义间距规则 (E302)"""
        if (SpacingRules.is_definition_line(current_line) and
                SpacingRules.is_definition_line(next_line)):

            empty_lines = self.count_empty_lines_ahead(position + 1)
            needed = 2 - empty_lines

            if needed > 0:
                return self.add_empty_lines(fixed_lines, needed, position + 1 + empty_lines)

        return position

    def apply_definition_after_spacing_rule(self, fixed_lines: list, current_line: str,
                                            next_line: str, position: int) -> int:
        """应用函数/类定义后间距规则 (E305)"""
        if (SpacingRules.is_definition_line(current_line) and
            SpacingRules.is_code_line(next_line) and
                not SpacingRules.is_indented_code(next_line)):

            empty_lines = self.count_empty_lines_ahead(position + 1)
            if empty_lines < 1:
                fixed_lines.append('')
                return position + 1  # 跳过刚添加的空行

        return position

    def fix_spacing_issues(self) -> bool:
        """修复空行问题"""
        if not self.read_file():
            return False

        fixed_lines = []
        i = 0

        while i < len(self.lines):
            current_line = self.lines[i]
            fixed_lines.append(current_line)

            # 检查下一行
            if i < len(self.lines) - 1:
                next_line = self.lines[i + 1]

                # 应用规则1: 函数/类定义之间的2个空行
                new_pos = self.apply_definition_spacing_rule(
                    fixed_lines, current_line, next_line, i)
                if new_pos != i:
                    i = new_pos
                    continue

                # 应用规则2: 函数/类定义后的1个空行
                new_pos = self.apply_definition_after_spacing_rule(
                    fixed_lines, current_line, next_line, i)
                if new_pos != i:
                    i = new_pos
                    continue

            i += 1

        # 清理多余的连续空行
        final_lines = self._clean_consecutive_empty_lines(fixed_lines)

        return self._write_if_changed('\n'.join(final_lines))

    def _clean_consecutive_empty_lines(self, lines: list) -> list:
        """清理超过2个的连续空行"""
        final_lines = []
        consecutive_empty = 0

        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty <= 2:
                    final_lines.append(line)
            else:
                consecutive_empty = 0
                final_lines.append(line)

        return final_lines

    def _write_if_changed(self, new_content: str) -> bool:
        """仅在内容有变化时写入文件"""
        if new_content != self.original_content:
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            except Exception:
                return False
        return False


def fix_spacing_issues_advanced(filepath):
    """高级空行修复 - 重构后的版本"""
    fixer = SpacingFixer(filepath)
    return fixer.fix_spacing_issues()


def process_directory_advanced(directory):
    """处理目录中的所有Python文件"""
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_spacing_issues_advanced(filepath):
                    count += 1

    print(f"高级修复处理了 {count} 个文件")


if __name__ == "__main__":
    # 处理src目录
    src_dir = "src"
    if os.path.exists(src_dir):
        print("开始高级空行修复...")
        process_directory_advanced(src_dir)
    else:
        print("src目录不存在")

#!/usr/bin/env python3
"""
修复代码质量问题的脚本

主要修复以下问题：
1. F541 f-string is missing placeholders
2. 代码风格问题（缩进、空白等）
3. 语法错误
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


class CodeQualityFixer:
    """代码质量修复器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.fixed_files = []

    def fix_fstring_placeholders(self, file_path: Path) -> bool:
        """修复f字符串缺少占位符的问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 模式1: f"文本 {variable} 更多文本" -> 保持不变，这是正确的
            # 模式2: f"只有文本没有变量" -> "只有文本没有变量"（移除f前缀）

            # 查找所有f字符串
            fstring_pattern = r'f"([^"]*)"'

            def replace_fstring(match):
                fstring_content = match.group(1)
                # 检查是否包含变量占位符（{}包围的内容）
                if '{' in fstring_content and '}' in fstring_content:
                    # 包含变量，保持f前缀
                    return match.group(0)
                else:
                    # 不包含变量，移除f前缀
                    return f'"{fstring_content}"'

            content = re.sub(fstring_pattern, replace_fstring, content)

            # 同样的处理单引号f字符串
            fstring_pattern_single = r"f'([^']*)'"

            def replace_fstring_single(match):
                fstring_content = match.group(1)
                if '{' in fstring_content and '}' in fstring_content:
                    return match.group(0)
                else:
                    return f"'{fstring_content}'"

            content = re.sub(fstring_pattern_single, replace_fstring_single, content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                return True

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

        return False

    def fix_whitespace_issues(self, file_path: Path) -> bool:
        """修复空白字符相关问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_lines = lines.copy()
            modified = False

            for i, line in enumerate(lines):
                original_line = line

                # 修复行尾空白字符 (W291)
                if line.rstrip('\n').endswith((' ', '\t')):
                    lines[i] = line.rstrip() + '\n'
                    modified = True

                # 修复只有空白的行 (W293)
                stripped = line.strip()
                if not stripped and line.strip('\n'):
                    lines[i] = '\n'
                    modified = True

            # 确保文件以换行符结尾，如果需要的话
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
                modified = True

            # 移除文件末尾多余的空行 (W391)
            while len(lines) >= 2 and lines[-1].strip() == '' and lines[-2].strip() == '':
                lines.pop()
                modified = True

            # 如果文件不以换行符结尾，添加一个 (W292)
            if lines and lines[-1] and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
                modified = True

            if modified and lines != original_lines:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                if str(file_path) not in self.fixed_files:
                    self.fixed_files.append(str(file_path))
                return True

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

        return False

    def fix_syntax_errors(self, file_path: Path) -> bool:
        """修复语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 修复f字符串语法错误 (E999)
            # 处理不完整的f字符串
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 查找不完整的f字符串
                if 'f"' in line and not line.count('"') % 2 == 0:
                    # 尝试修复不完整的f字符串
                    if '{' in line and '}' not in line:
                        # 可能是缺少结束大括号的变量
                        continue  # 暂时跳过复杂的情况
                elif "f'" in line and not line.count("'") % 2 == 0:
                    continue

            content = '\n'.join(lines)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                if str(file_path) not in self.fixed_files:
                    self.fixed_files.append(str(file_path))
                modified = True

            return modified

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

        return False

    def fix_indentation_issues(self, file_path: Path) -> bool:
        """修复缩进问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_lines = lines.copy()
            modified = False

            # 这里可以添加更复杂的缩进修复逻辑
            # 目前主要处理一些简单的缩进问题

            if modified and lines != original_lines:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                if str(file_path) not in self.fixed_files:
                    self.fixed_files.append(str(file_path))
                return True

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

        return False

    def process_file(self, file_path: Path) -> bool:
        """处理单个文件"""
        if not file_path.exists() or file_path.suffix != '.py':
            return False

        modified = False

        # 按优先级修复问题
        modified |= self.fix_fstring_placeholders(file_path)
        modified |= self.fix_syntax_errors(file_path)
        modified |= self.fix_whitespace_issues(file_path)
        modified |= self.fix_indentation_issues(file_path)

        return modified

    def find_files_with_issues(self) -> List[Path]:
        """查找有问题的文件"""
        issues_files = []

        # 从flake8输出中提取有问题的文件
        flake8_output = """
tests/deployment_launch_task.py:830:11: F541 f-string is missing placeholders
tests/deployment_launch_task.py:831:11: F541 f-string is missing placeholders
tests/deployment_launch_task.py:832:11: F541 f-string is missing placeholders
tests/deployment_launch_task.py:833:11: F541 f-string is missing placeholders
tests/deployment_launch_task.py:834:11: F541 f-string is missing placeholders
tests/deployment_preparation.py:75:25: E128 continuation line under-indented for visual indent
tests/deployment_preparation.py:177:9: F841 local variable 'required_tasks' is assigned to but never used
tests/deployment_preparation.py:1489:38: E128 continuation line under-indented for visual indent
tests/deployment_preparation.py:1490:38: E128 continuation line under-indented for visual indent
tests/dev_env_setup_task.py:753:11: F541 f-string is missing placeholders
tests/dev_env_setup_task.py:754:11: F541 f-string is missing placeholders
tests/dev_env_setup_task.py:755:11: F541 f-string is missing placeholders
tests/dev_env_setup_task.py:756:11: F541 f-string is missing placeholders
tests/dev_env_setup_task.py:757:11: F541 f-string is missing placeholders
tests/dev_env_setup_task.py:758:11: F541 f-string is missing placeholders
tests/rqa_post_project_documentation_generator.py:7166:4: E999 SyntaxError: f-string: expecting '}'
"""

        for line in flake8_output.strip().split('\n'):
            if line.strip():
                parts = line.split(':')
                if len(parts) >= 2:
                    file_path = parts[0]
                    full_path = self.project_root / file_path
                    if full_path not in issues_files:
                        issues_files.append(full_path)

        return issues_files

    def process_all_issues(self) -> int:
        """处理所有问题"""
        print("🔧 开始修复代码质量问题...")

        # 手动指定有问题的文件列表（基于提供的flake8输出）
        issues_files = [
            self.project_root / "tests/deployment_launch_task.py",
            self.project_root / "tests/deployment_preparation.py",
            self.project_root / "tests/dev_env_setup_task.py",
            self.project_root / "tests/final_coverage_summary.py",
            self.project_root / "tests/rqa_post_project_documentation_generator.py",
            # 添加其他有问题的文件...
        ]

        total_fixed = 0

        for file_path in issues_files:
            if file_path.exists():
                print(f"处理文件: {file_path}")
                if self.process_file(file_path):
                    total_fixed += 1
                    print(f"✅ 已修复: {file_path}")
                else:
                    print(f"ℹ️ 无需修复或修复失败: {file_path}")
            else:
                print(f"⚠️ 文件不存在: {file_path}")

        print(f"\n📊 修复完成! 共处理 {len(issues_files)} 个文件，修复了 {total_fixed} 个文件")
        print(f"📝 已修复的文件: {len(self.fixed_files)}")

        return total_fixed


def main():
    """主函数"""
    project_root = Path(__file__).resolve().parent.parent

    fixer = CodeQualityFixer(project_root)
    fixed_count = fixer.process_all_issues()

    if fixed_count > 0:
        print("\n🎉 建议运行以下命令验证修复结果:")
        print("flake8 tests/ --select=F541,E999,W291,W293,W391 --show-source")
    else:
        print("\nℹ️ 没有发现需要修复的问题")


if __name__ == "__main__":
    main()

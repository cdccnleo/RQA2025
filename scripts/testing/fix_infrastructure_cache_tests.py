#!/usr/bin/env python3
"""
基础设施缓存测试修复脚本

修复基础设施缓存子模块测试文件中的缩进和语法错误
"""

import os
import re
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InfrastructureCacheTestFixer:
    """基础设施缓存测试修复器"""

    def __init__(self, cache_test_dir: str):
        self.cache_test_dir = Path(cache_test_dir)
        self.fixed_files = []
        self.errors = []

    def find_cache_test_files(self) -> list:
        """查找所有缓存测试文件"""
        test_files = []
        if self.cache_test_dir.exists():
            for file_path in self.cache_test_dir.rglob("test_*.py"):
                test_files.append(file_path)
        return test_files

    def fix_indentation_issues(self, file_path: Path) -> bool:
        """修复单个文件的缩进问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            lines = content.split('\n')
            fixed_lines = []
            in_multiline_string = False
            in_function = False
            function_indent = 0

            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # 跳过空行
                if not stripped:
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 处理多行字符串
                if '"""' in line or "'''" in line:
                    in_multiline_string = not in_multiline_string

                # 处理导入语句
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # 确保导入语句没有缩进
                    if line.startswith(' ') or line.startswith('\t'):
                        line = stripped
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 处理类定义
                if stripped.startswith('class '):
                    in_function = False
                    function_indent = 0
                    # 确保类定义没有缩进
                    if line.startswith(' ') or line.startswith('\t'):
                        line = stripped
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 处理函数定义
                if stripped.startswith('def '):
                    in_function = True
                    # 计算函数缩进
                    indent_match = re.match(r'^(\s*)', line)
                    function_indent = len(indent_match.group(1)) if indent_match else 0

                    # 确保函数定义正确缩进（通常是4个空格）
                    if function_indent == 0:
                        # 如果没有缩进，添加4个空格
                        line = '    ' + stripped
                        function_indent = 4
                    fixed_lines.append(line)
                    i += 1
                    continue

                # 处理普通代码行
                if in_function:
                    # 确保函数内部代码正确缩进
                    indent_match = re.match(r'^(\s*)', line)
                    current_indent = len(indent_match.group(1)) if indent_match else 0

                    if current_indent <= function_indent:
                        # 如果缩进不够，添加更多缩进
                        line = '    ' + line.strip()
                    elif current_indent > function_indent + 8:
                        # 如果缩进太多，减少缩进
                        line = '    ' * (function_indent // 4 + 1) + line.strip()
                else:
                    # 类级别的代码
                    if stripped.startswith('#') or in_multiline_string:
                        pass  # 注释和多行字符串保持原样
                    elif not line.startswith('    ') and not line.startswith('\t'):
                        # 如果没有缩进，添加4个空格
                        line = '    ' + stripped

                fixed_lines.append(line)
                i += 1

            # 重新组合内容
            new_content = '\n'.join(fixed_lines)

            # 修复一些常见的语法错误
            new_content = self.fix_common_syntax_errors(new_content)

            if new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixed_files.append(str(file_path))
                logger.info(f"已修复: {file_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"修复文件 {file_path} 时出错: {e}")
            self.errors.append((str(file_path), str(e)))
            return False

    def fix_common_syntax_errors(self, content: str) -> str:
        """修复常见的语法错误"""
        # 修复函数定义中的冒号问题
        content = re.sub(r'def\s+(\w+)\s*\(', r'def \1(', content)

        # 修复函数定义后缺少冒号的问题
        content = re.sub(r'def\s+(\w+)\s*\([^)]*\)\s*$', r'def \1(...):', content)

        # 修复类定义后缺少冒号的问题
        content = re.sub(r'class\s+(\w+)\s*\([^)]*\)\s*$', r'class \1(...):', content)

        # 修复多行导入语句
        lines = content.split('\n')
        fixed_lines = []
        in_import_block = False

        for line in lines:
            if line.strip().startswith('from ') and '(' in line:
                in_import_block = True
                fixed_lines.append(line)
            elif in_import_block and ')' in line:
                in_import_block = False
                fixed_lines.append(line)
            elif in_import_block:
                # 确保导入块内的行正确缩进
                if not line.startswith('    ') and not line.startswith('\t') and line.strip():
                    line = '    ' + line.strip()
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_all_cache_tests(self) -> tuple:
        """修复所有缓存测试文件"""
        test_files = self.find_cache_test_files()
        logger.info(f"找到 {len(test_files)} 个缓存测试文件")

        fixed_count = 0
        for file_path in test_files:
            if self.fix_indentation_issues(file_path):
                fixed_count += 1

        logger.info(f"修复完成: {fixed_count} 个文件已修复，{len(self.errors)} 个文件出错")
        return fixed_count, len(self.errors)


def main():
    """主函数"""
    cache_test_dir = "tests/unit/infrastructure/cache"

    if not os.path.exists(cache_test_dir):
        logger.error(f"缓存测试目录不存在: {cache_test_dir}")
        return

    fixer = InfrastructureCacheTestFixer(cache_test_dir)
    fixed_count, error_count = fixer.fix_all_cache_tests()

    print("\n修复总结:")
    print(f"- 修复的文件: {fixed_count}")
    print(f"- 出错的文件: {error_count}")

    if fixer.fixed_files:
        print("\n修复的文件列表:")
        for file in fixer.fixed_files[:10]:  # 只显示前10个
            print(f"  - {file}")
        if len(fixer.fixed_files) > 10:
            print(f"  ... 还有 {len(fixer.fixed_files) - 10} 个文件")

    if fixer.errors:
        print("\n出错的文件列表:")
        for file, error in fixer.errors[:5]:  # 只显示前5个
            print(f"  - {file}: {error}")
        if len(fixer.errors) > 5:
            print(f"  ... 还有 {len(fixer.errors) - 5} 个错误")


if __name__ == "__main__":
    main()

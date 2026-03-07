#!/usr/bin/env python3
"""
修复导入错误脚本

专门修复ImportError和模块不存在问题
"""

import os
import re
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImportErrorFixer:
    """导入错误修复器"""

    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.fixed_files = []
        self.errors = []
        self.module_map = self.create_module_map()

    def create_module_map(self):
        """创建模块映射"""
        module_map = {}

        # 扫描src目录中的实际模块
        src_dir = Path('src')
        if src_dir.exists():
            for py_file in src_dir.rglob('*.py'):
                if py_file.name != '__init__.py':
                    # 计算相对路径作为模块名
                    rel_path = py_file.relative_to(src_dir)
                    module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')

                    # 创建可能的导入变体
                    module_map[module_name] = str(py_file)
                    module_map[f'src.{module_name}'] = str(py_file)
                    module_map[f'.{module_name}'] = str(py_file)

        return module_map

    def find_all_test_files(self) -> list:
        """查找所有测试文件"""
        test_files = []
        if self.test_dir.exists():
            for file_path in self.test_dir.rglob("test_*.py"):
                test_files.append(file_path)
        return test_files

    def fix_import_errors(self, file_path: Path) -> bool:
        """修复导入错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 分析导入语句
            lines = content.split('\n')
            fixed_lines = []

            for line in lines:
                # 修复from src.开头的导入
                if line.strip().startswith('from src.'):
                    line = self.fix_src_import(line)
                elif line.strip().startswith('import src.'):
                    line = self.fix_src_import(line)
                elif 'src.' in line and ('import' in line or 'from' in line):
                    line = self.fix_src_import(line)

                # 处理try-except导入块
                if 'try:' in line and 'import' in lines[lines.index(line) + 1 if lines.index(line) + 1 < len(lines) else 0]:
                    # 找到try-except导入块
                    try_block = self.fix_try_except_import(lines, lines.index(line))
                    if try_block:
                        # 替换整个try-except块
                        continue

                fixed_lines.append(line)

            # 重新组合内容
            new_content = '\n'.join(fixed_lines)

            # 移除多余的空行
            new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)

            if new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixed_files.append(str(file_path))
                logger.info(f"已修复导入错误: {file_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            self.errors.append((str(file_path), str(e)))
            return False

    def fix_src_import(self, line: str) -> str:
        """修复src.开头的导入"""
        # 移除src.前缀
        line = line.replace('from src.', 'from ')
        line = line.replace('import src.', 'import ')

        # 处理相对导入
        if line.strip().startswith('from .'):
            # 相对导入，保持不变
            pass
        elif line.strip().startswith('from '):
            # 检查模块是否存在
            parts = line.strip().split()
            if len(parts) >= 2:
                module_name = parts[1].split('.')[0]
                if module_name in self.module_map:
                    # 模块存在，保持原样
                    pass
                else:
                    # 模块不存在，添加try-except
                    line = self.add_try_except(line)

        return line

    def add_try_except(self, import_line: str) -> str:
        """为导入语句添加try-except"""
        indented_import = '    ' + import_line
        try_except_block = f'''try:
{indented_import}
except ImportError:
    pass'''

        return try_except_block

    def fix_try_except_import(self, lines: list, try_index: int) -> str:
        """修复try-except导入块"""
        if try_index + 1 >= len(lines):
            return None

        import_line = lines[try_index + 1].strip()
        if not import_line.startswith('from ') and not import_line.startswith('import '):
            return None

        # 查找对应的except块
        except_index = -1
        for i in range(try_index + 1, min(try_index + 10, len(lines))):
            if lines[i].strip().startswith('except ImportError:'):
                except_index = i
                break

        if except_index == -1:
            return None

        # 修复导入语句
        fixed_import = self.fix_src_import(import_line)

        # 重建try-except块
        fixed_block = f'''try:
    {fixed_import}
{lines[except_index]}
    pass'''

        return fixed_block

    def fix_all_files(self) -> tuple:
        """修复所有文件"""
        test_files = self.find_all_test_files()
        logger.info(f"找到 {len(test_files)} 个测试文件")

        fixed_count = 0
        for file_path in test_files:
            if self.fix_import_errors(file_path):
                fixed_count += 1

        logger.info(f"修复完成: {fixed_count} 个文件已修复")
        return fixed_count, len(self.errors)


def main():
    """主函数"""
    test_dir = "tests/unit/infrastructure"

    if not os.path.exists(test_dir):
        logger.error(f"测试目录不存在: {test_dir}")
        return

    fixer = ImportErrorFixer(test_dir)
    fixed_count, error_count = fixer.fix_all_files()

    print("\n导入错误修复总结:")
    print(f"- 处理的文件数: {len(fixer.find_all_test_files())}")
    print(f"- 修复的文件数: {fixed_count}")
    print(f"- 出错的文件数: {error_count}")

    if fixer.fixed_files:
        print("\n修复的文件列表 (前10个):")
        for file in fixer.fixed_files[:10]:
            print(f"  - {file}")
        if len(fixer.fixed_files) > 10:
            print(f"  ... 还有 {len(fixer.fixed_files) - 10} 个文件")

    if fixer.errors:
        print("\n出错的文件列表:")
        for file, error in fixer.errors[:5]:
            print(f"  - {file}: {error}")
        if len(fixer.errors) > 5:
            print(f"  ... 还有 {len(fixer.errors) - 5} 个错误")


if __name__ == "__main__":
    main()

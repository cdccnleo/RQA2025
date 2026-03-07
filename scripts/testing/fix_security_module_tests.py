#!/usr/bin/env python3
"""
基础设施security模块测试修复脚本

专门修复security子模块的测试文件
"""

import os
import re
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityTestFixer:
    """Security模块测试修复器"""

    def __init__(self, security_test_dir: str):
        self.security_test_dir = Path(security_test_dir)
        self.fixed_files = []
        self.errors = []

    def find_security_test_files(self) -> list:
        """查找所有security测试文件"""
        test_files = []
        if self.security_test_dir.exists():
            for file_path in self.security_test_dir.rglob("test_*.py"):
                test_files.append(file_path)
        return test_files

    def fix_file_content(self, file_path: Path) -> bool:
        """修复单个文件的语法和导入问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            lines = content.split('\n')
            fixed_lines = []

            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # 处理导入语句后的缩进问题
                if (stripped.startswith('from ') or stripped.startswith('import ')) and i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if next_line.startswith('except ImportError:') and not line.startswith('try:'):
                        # 添加try语句
                        lines.insert(i, 'try:')
                        i += 1

                # 处理缩进问题
                elif stripped.startswith('except ImportError:'):
                    # 确保except语句正确缩进
                    if not line.startswith('    ') and not line.startswith('\t'):
                        line = '    ' + stripped

                # 处理pytest.main缩进问题
                elif stripped.startswith('pytest.main(['):
                    if not line.startswith('    ') and not line.startswith('\t'):
                        line = '    ' + stripped

                # 处理if __name__ == "__main__"缩进问题
                elif stripped.startswith('if __name__ == "__main__":'):
                    if line.startswith(' ') or line.startswith('\t'):
                        line = stripped

                # 处理函数定义问题
                elif stripped.startswith('def ') and not stripped.endswith(':'):
                    line = stripped + ':'

                # 处理类定义问题
                elif stripped.startswith('class ') and not stripped.endswith(':'):
                    line = stripped + ':'

                # 处理文件头问题
                elif stripped.startswith('#!/usr/bin/env python3') and i > 0:
                    # 如果shebang不在文件开头，移除它
                    line = ''

                fixed_lines.append(line)
                i += 1

            # 重新组合内容
            new_content = '\n'.join(fixed_lines)

            # 修复一些常见的语法错误模式
            new_content = re.sub(r'^\s*except ImportError:\s*$',
                                 r'    except ImportError:\n        pass', new_content, flags=re.MULTILINE)

            # 修复多行导入的缩进
            lines = new_content.split('\n')
            in_try_block = False
            for i, line in enumerate(lines):
                if line.strip().startswith('try:'):
                    in_try_block = True
                elif line.strip().startswith('except ImportError:'):
                    in_try_block = False
                elif in_try_block and (line.strip().startswith('from ') or line.strip().startswith('import ')):
                    if not line.startswith('    ') and not line.startswith('\t'):
                        lines[i] = '    ' + line.strip()

            new_content = '\n'.join(lines)

            # 修复空行和多余的空行
            new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)

            # 确保文件以正确的shebang开头
            if not new_content.startswith('#!/usr/bin/env python3'):
                new_content = '#!/usr/bin/env python3\n' + new_content

            if new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixed_files.append(str(file_path))
                logger.info(f"已修复: {file_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            self.errors.append((str(file_path), str(e)))
            return False

    def fix_all_security_tests(self) -> tuple:
        """修复所有security测试文件"""
        test_files = self.find_security_test_files()
        logger.info(f"找到 {len(test_files)} 个security测试文件")

        fixed_count = 0
        for file_path in test_files:
            if self.fix_file_content(file_path):
                fixed_count += 1

        logger.info(f"修复完成: {fixed_count} 个文件已修复，{len(self.errors)} 个文件出错")
        return fixed_count, len(self.errors)


def main():
    """主函数"""
    security_test_dir = "tests/unit/infrastructure/security"

    if not os.path.exists(security_test_dir):
        logger.error(f"security测试目录不存在: {security_test_dir}")
        return

    fixer = SecurityTestFixer(security_test_dir)
    fixed_count, error_count = fixer.fix_all_security_tests()

    print("\nsecurity模块修复总结:")
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

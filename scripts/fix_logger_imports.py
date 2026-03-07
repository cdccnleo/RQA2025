#!/usr/bin/env python3
"""
RQA2025 Logger导入修复脚本

专门修复logger导入相关的问题
"""

import os
import glob
from pathlib import Path


class LoggerImportFixer:
    """Logger导入修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []

    def find_files_with_logger_issues(self) -> list:
        """查找包含logger问题的文件"""
        error_files = []

        # 基于flake8输出的错误模式查找文件
        error_patterns = [
            "src/**/*.py"
        ]

        for pattern in error_patterns:
            for file_path in glob.glob(str(self.project_root / pattern), recursive=True):
                if os.path.isfile(file_path):
                    error_files.append(Path(file_path))

        return error_files

    def fix_logger_imports(self, file_path: Path) -> bool:
        """修复logger导入问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 模式1: 文件中使用logger但没有导入logging
            if ('logger' in content or 'logging' in content) and 'import logging' not in content:
                # 检查是否已经有logging相关的导入
                has_logging_import = False
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import logging') or 'from logging' in line:
                        has_logging_import = True
                        break

                if not has_logging_import:
                    # 添加logging导入
                    lines = content.split('\n')
                    insert_index = 0

                    # 找到最后一个导入语句的位置
                    for i, line in enumerate(lines):
                        if line.startswith(('import ', 'from ')):
                            insert_index = i + 1
                        elif line.strip() and not line.startswith('#'):
                            break

                    lines.insert(insert_index, "import logging")
                    content = '\n'.join(lines)
                    modified = True

            # 模式2: 修复常见的logger定义模式
            if 'logger =' in content and 'logging.getLogger' not in content:
                # 查找logger定义的位置
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'logger =' in line and 'logging.getLogger' not in line:
                        # 替换为正确的logger定义
                        indent = len(line) - len(line.lstrip())
                        lines[i] = ' ' * indent + f"logger = logging.getLogger(__name__)"
                        modified = True
                        break

                if modified:
                    content = '\n'.join(lines)

            # 模式3: 修复logger调用问题
            if 'logger.' in content:
                # 确保有logger变量定义
                has_logger_var = False
                for line in content.split('\n'):
                    if 'logger =' in line:
                        has_logger_var = True
                        break

                if not has_logger_var:
                    # 在文件末尾添加logger定义
                    lines = content.split('\n')
                    # 找到最后一个非空行
                    last_line_idx = len(lines) - 1
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip():
                            last_line_idx = i
                            break

                    lines.insert(last_line_idx + 1, "")
                    lines.insert(last_line_idx + 2, "# Logger setup")
                    lines.insert(last_line_idx + 3, "logger = logging.getLogger(__name__)")
                    content = '\n'.join(lines)
                    modified = True

            # 写入修复后的内容
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                print(f"✅ 修复logger导入: {file_path}")
                return True

        except Exception as e:
            print(f"❌ 修复文件 {file_path} 时出错: {str(e)}")
            return False

        return False

    def run_fix(self) -> dict:
        """运行修复"""
        print("🔧 开始修复logger导入问题...")

        files = self.find_files_with_logger_issues()
        print(f"📁 找到 {len(files)} 个文件需要检查")

        fixed_count = 0

        for file_path in files:
            if self.fix_logger_imports(file_path):
                fixed_count += 1

        print("\n📊 Logger导入修复完成:")
        print(f"   - 处理文件数: {len(files)}")
        print(f"   - 修复文件数: {len(set(self.fixed_files))}")

        return {
            'total_files': len(files),
            'fixed_files': len(set(self.fixed_files))
        }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    fixer = LoggerImportFixer(project_root)
    result = fixer.run_fix()

    print(f"\n🎯 Logger导入修复结果: {result['fixed_files']}/{result['total_files']} 文件已修复")


if __name__ == "__main__":
    main()

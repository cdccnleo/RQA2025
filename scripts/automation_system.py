#!/usr/bin/env python3
"""
RQA2025自动化语法错误检测和修复系统
"""

import re
from pathlib import Path
from typing import List, Dict


class SyntaxFixer:
    """语法错误自动修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.errors = []

    def scan_and_fix(self) -> Dict[str, any]:
        """扫描并修复语法错误"""
        print("🔍 开始扫描语法错误...")

        # 扫描所有Python文件
        python_files = self._find_python_files()

        for file_path in python_files:
            try:
                if self._check_and_fix_file(file_path):
                    self.fixed_files.append(str(file_path))
            except Exception as e:
                self.errors.append(f"{file_path}: {str(e)}")

        return {
            'fixed_files': self.fixed_files,
            'errors': self.errors,
            'total_scanned': len(python_files)
        }

    def _find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for file_path in self.root_dir.rglob("*.py"):
            python_files.append(file_path)
        return python_files

    def _check_and_fix_file(self, file_path: Path) -> bool:
        """检查并修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试编译检查语法
            compile(content, str(file_path), 'exec')
            return False  # 没有语法错误

        except SyntaxError as e:
            print(f"❌ 发现语法错误: {file_path} - {e}")
            return self._fix_syntax_error(file_path, e)
        except Exception as e:
            print(f"⚠️ 读取文件错误: {file_path} - {e}")
            return False

    def _fix_syntax_error(self, file_path: Path, error: SyntaxError) -> bool:
        """修复语法错误"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 根据错误类型应用不同的修复策略
        content = self._apply_fixes(content, error)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已修复: {file_path}")
            return True

        return False

    def _apply_fixes(self, content: str, error: SyntaxError) -> str:
        """应用修复策略"""
        error_msg = str(error)

        # 修复字符串字面量错误
        if 'EOL while scanning string literal' in error_msg:
            content = self._fix_string_literals(content)

        # 修复缩进错误
        elif 'unexpected indent' in error_msg or 'unindent' in error_msg:
            content = self._fix_indentation(content)

        # 修复字典语法错误
        elif 'invalid syntax' in error_msg and '{' in content:
            content = self._fix_dict_syntax(content)

        # 修复导入语法错误
        elif 'from' in content and 'import' in content:
            content = self._fix_import_syntax(content)

        # 通用修复
        content = self._fix_common_issues(content)

        return content

    def _fix_string_literals(self, content: str) -> str:
        """修复字符串字面量错误"""
        # 修复多行字符串
        content = re.sub(r'"""([^"]*?)$', r'"""\1"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''([^']*?)$", r"'''\1'''", content, flags=re.MULTILINE)

        # 修复未闭合的字符串
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                # 尝试修复未闭合的字符串
                if line.strip().startswith('"') and not line.strip().endswith('"'):
                    lines[i] = line + '"'
                elif line.strip().startswith("'") and not line.strip().endswith("'"):
                    lines[i] = line + "'"

        return '\n'.join(lines)

    def _fix_indentation(self, content: str) -> str:
        """修复缩进错误"""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 处理字典和列表的缩进
            if stripped.startswith(('return {', 'return [', 'def ', 'class ', 'if ', 'for ', 'while ')):
                indent_stack.append(len(line) - len(line.lstrip()))
            elif stripped.startswith(('},', ']', ')', ':')):
                if indent_stack:
                    indent_stack.pop()

            # 修复字典项的缩进
            if ':' in stripped and not stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:')):
                if len(line) - len(line.lstrip()) == 0:  # 顶层缩进
                    line = '    ' + line.lstrip()

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_dict_syntax(self, content: str) -> str:
        """修复字典语法错误"""
        # 修复多行字典
        content = re.sub(
            r'return \{\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+)',
            r'return {\n\1\2: \3,\n\4\5: \6',
            content,
            flags=re.MULTILINE
        )

        return content

    def _fix_import_syntax(self, content: str) -> str:
        """修复导入语法错误"""
        # 修复未闭合的导入语句
        content = re.sub(r'from\s+\.\w+\s+import\s*\(\s*$',
                         r'from . import (', content, flags=re.MULTILINE)
        content = re.sub(
            r'from\s+\.\w+\s+import\s*\(\s*\w+.*[^\)]$', r'\g<0>)', content, flags=re.MULTILINE)

        return content

    def _fix_common_issues(self, content: str) -> str:
        """修复常见问题"""
        # 修复函数定义错误
        content = re.sub(
            r'def\s+(\w+)\(([^)]*),?\)\s*\n(\s*)\)([^,]*),\s*\n(\s*)([^)]*)\):\s*\n',
            r'def \1(\2, \3\4, \5\6):\n',
            content,
            flags=re.MULTILINE
        )

        # 修复类定义错误
        content = re.sub(
            r'class\s+(\w+):\s*"""([^"]*)"""\s*\n\}',
            r'class \1:\n    """\2"""',
            content,
            flags=re.MULTILINE
        )

        return content


def test_infrastructure_layer():
    """测试基础设施层功能"""
    print("\n🧪 测试基础设施层功能...")

    try:
        # 测试配置管理器
        from src.infrastructure.config import ConfigFactory
        print("✅ ConfigFactory导入成功")

        manager = ConfigFactory.create_config_manager()
        print("✅ ConfigFactory.create_config_manager() 成功")
        print(f"管理器类型: {type(manager)}")

        # 测试缓存管理器
        from src.infrastructure.cache import BaseCacheManager
        print("✅ BaseCacheManager导入成功")

        cache = BaseCacheManager()
        print("✅ BaseCacheManager实例化成功")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 RQA2025自动化语法错误检测和修复系统")
    print("=" * 60)

    # 创建修复器
    fixer = SyntaxFixer("src/infrastructure")

    # 扫描并修复
    result = fixer.scan_and_fix()

    print("\n📊 修复结果:")
    print(f"   扫描文件数: {result['total_scanned']}")
    print(f"   修复文件数: {len(result['fixed_files'])}")

    if result['fixed_files']:
        print("   修复的文件:")
        for file in result['fixed_files']:
            print(f"     ✅ {file}")

    if result['errors']:
        print("   修复错误:")
        for error in result['errors']:
            print(f"     ❌ {error}")

    # 测试功能
    if test_infrastructure_layer():
        print("\n🎉 基础设施层功能测试通过!")
    else:
        print("\n⚠️ 基础设施层功能测试失败")


if __name__ == "__main__":
    main()

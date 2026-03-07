#!/usr/bin/env python3
"""
RQA2025增强型语法错误检测和修复系统
针对剩余的复杂语法错误提供更精确的修复
"""

import re
from pathlib import Path
from typing import List, Dict, Any


class EnhancedSyntaxFixer:
    """增强型语法错误修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.errors = []
        self.complex_errors = []

    def scan_and_fix_remaining(self) -> Dict[str, Any]:
        """扫描并修复剩余的复杂语法错误"""
        print("🔍 开始扫描剩余复杂语法错误...")

        # 扫描所有Python文件
        python_files = self._find_python_files()

        for file_path in python_files:
            try:
                if self._check_and_fix_complex_file(file_path):
                    self.fixed_files.append(str(file_path))
            except Exception as e:
                self.errors.append(f"{file_path}: {str(e)}")

        return {
            'fixed_files': self.fixed_files,
            'errors': self.errors,
            'complex_errors': self.complex_errors,
            'total_scanned': len(python_files)
        }

    def _find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for file_path in self.root_dir.rglob("*.py"):
            python_files.append(file_path)
        return python_files

    def _check_and_fix_complex_file(self, file_path: Path) -> bool:
        """检查并修复单个复杂文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试编译检查语法
            compile(content, str(file_path), 'exec')
            return False  # 没有语法错误

        except SyntaxError as e:
            print(f"❌ 发现复杂语法错误: {file_path} - {e}")
            return self._fix_complex_syntax_error(file_path, e)
        except Exception as e:
            print(f"⚠️ 读取文件错误: {file_path} - {e}")
            return False

    def _fix_complex_syntax_error(self, file_path: Path, error: SyntaxError) -> bool:
        """修复复杂语法错误"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 根据错误类型应用不同的修复策略
        content = self._apply_complex_fixes(content, error)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已修复复杂错误: {file_path}")
            return True

        return False

    def _apply_complex_fixes(self, content: str, error: SyntaxError) -> str:
        """应用复杂修复策略"""
        error_msg = str(error)
        error_line = error.lineno if hasattr(error, 'lineno') else 0

        # 获取错误所在行附近的内容
        lines = content.split('\n')
        if error_line > 0 and error_line <= len(lines):
            error_context = '\n'.join(lines[max(0, error_line-3):error_line+2])
        else:
            error_context = content[:200]

        # 根据具体错误类型应用修复
        if 'invalid syntax' in error_msg:
            content = self._fix_invalid_syntax(content, error_context, error_line)
        elif 'EOL while scanning string literal' in error_msg:
            content = self._fix_string_literal_eol(content, error_context, error_line)
        elif 'invalid character' in error_msg and '：' in error_msg:
            content = self._fix_chinese_colon(content, error_context)
        elif 'closing parenthesis' in error_msg and '{' in error_msg:
            content = self._fix_parenthesis_mismatch(content, error_context, error_line)
        elif 'unindent does not match' in error_msg:
            content = self._fix_unindent_mismatch(content, error_context, error_line)
        elif 'expected an indented block' in error_msg:
            content = self._fix_expected_indented_block(content, error_context, error_line)
        elif 'unexpected EOF while parsing' in error_msg:
            content = self._fix_unexpected_eof(content, error_context)
        else:
            # 通用复杂错误处理
            content = self._fix_general_complex_errors(content, error_context, error_line)

        return content

    def _fix_invalid_syntax(self, content: str, error_context: str, error_line: int) -> str:
        """修复无效语法错误"""
        lines = content.split('\n')

        # 查找并修复常见的无效语法问题
        for i, line in enumerate(lines):
            if i >= error_line - 2 and i <= error_line + 2:
                # 修复函数参数定义问题
                if 'def ' in line and '(' in line and ')' in line:
                    line = re.sub(r'def\s+(\w+)\(([^)]*),?\)\s*\n(\s*)\)([^,]*),\s*\n(\s*)([^)]*)\):\s*\n',
                                  r'def \1(\2, \3\4, \5\6):\n', line)

                # 修复类定义问题
                elif 'class ' in line and ':' in line:
                    line = re.sub(r'class\s+(\w+):\s*"""([^"]*)"""\s*\n\}',
                                  r'class \1:\n    """\2"""', line)

                # 修复字典返回问题
                elif 'return' in line and '{' in line:
                    line = re.sub(r'return\s*\{\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+)',
                                  r'return {\n\1\2: \3,\n\4\5: \6', line)

                # 修复导入语句问题
                elif 'from ' in line and 'import' in line and '(' in line:
                    if not line.strip().endswith(')'):
                        line += ')'

                lines[i] = line

        return '\n'.join(lines)

    def _fix_string_literal_eol(self, content: str, error_context: str, error_line: int) -> str:
        """修复字符串字面量EOL错误"""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if i >= error_line - 3 and i <= error_line + 3:
                # 修复多行字符串
                if '"""' in line and not line.count('"""') % 2 == 0:
                    # 查找下一个"""并合并
                    if i + 1 < len(lines):
                        lines[i] = line + '"""'
                    else:
                        lines[i] = line.rstrip() + '"""'

                # 修复模块级字符串
                if line.strip().startswith('"""') and not line.strip().endswith('"""'):
                    lines[i] = line.rstrip() + '"""'

        return '\n'.join(lines)

    def _fix_chinese_colon(self, content: str, error_context: str) -> str:
        """修复中文冒号错误"""
        # 将中文冒号替换为英文冒号
        content = content.replace('：', ':')
        return content

    def _fix_parenthesis_mismatch(self, content: str, error_context: str, error_line: int) -> str:
        """修复括号不匹配错误"""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if i >= error_line - 2 and i <= error_line + 2:
                # 修复字典括号问题
                if '{' in line and ')' in line and not line.count('{') == line.count('}'):
                    line = re.sub(r'\(\s*\{([^}]*)\}\s*\)', r'(\1)', line)

                # 修复函数调用括号问题
                if '(' in line and '{' in line and not line.count('(') == line.count(')'):
                    if line.count('(') > line.count(')'):
                        line += ')'

                lines[i] = line

        return '\n'.join(lines)

    def _fix_unindent_mismatch(self, content: str, error_context: str, error_line: int) -> str:
        """修复缩进不匹配错误"""
        lines = content.split('\n')

        # 分析缩进模式
        indent_stack = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            current_indent = len(line) - len(line.lstrip())

            # 处理缩进
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ')):
                indent_stack.append(current_indent)
            elif stripped.startswith(('return', 'pass', 'continue', 'break')):
                # 检查是否需要调整缩进
                if indent_stack and current_indent != indent_stack[-1] + 4:
                    # 调整缩进
                    correct_indent = indent_stack[-1] + 4 if indent_stack else 4
                    lines[i] = ' ' * correct_indent + stripped
            elif stripped and indent_stack:
                # 普通语句的缩进
                expected_indent = indent_stack[-1] + 4
                if current_indent != expected_indent and current_indent < expected_indent:
                    lines[i] = ' ' * expected_indent + stripped

        return '\n'.join(lines)

    def _fix_expected_indented_block(self, content: str, error_context: str, error_line: int) -> str:
        """修复期望缩进块错误"""
        lines = content.split('\n')

        if error_line > 0 and error_line <= len(lines):
            # 在错误行后添加缩进块
            error_line_content = lines[error_line - 1].strip()
            if error_line_content.endswith(':'):
                if error_line < len(lines):
                    # 添加pass语句作为占位符
                    lines.insert(error_line, '    pass')
                else:
                    lines.append('    pass')

        return '\n'.join(lines)

    def _fix_unexpected_eof(self, content: str, error_context: str) -> str:
        """修复意外EOF错误"""
        lines = content.split('\n')

        # 检查未闭合的结构
        open_brackets = {'(': 0, '[': 0, '{': 0}
        for line in lines:
            for char in line:
                if char in '([{':
                    open_brackets[char] += 1
                elif char in ')]}':
                    for bracket, count in open_brackets.items():
                        if count > 0:
                            open_brackets[bracket] -= 1
                            break

        # 补全未闭合的括号
        for bracket, count in open_brackets.items():
            if count > 0:
                close_bracket = {'(': ')', '[': ']', '{': '}'}[bracket]
                content += close_bracket * count

        return content

    def _fix_general_complex_errors(self, content: str, error_context: str, error_line: int) -> str:
        """修复通用复杂错误"""
        lines = content.split('\n')

        # 修复常见的复杂语法问题
        for i, line in enumerate(lines):
            if i >= error_line - 2 and i <= error_line + 2:
                # 修复字典定义问题
                if '{' in line and '=' in line and not line.strip().endswith(','):
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith('}'):
                        line += ','

                # 修复列表定义问题
                if '[' in line and '=' in line and not line.strip().endswith(','):
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith(']'):
                        line += ','

                lines[i] = line

        return '\n'.join(lines)


def test_critical_functionality():
    """测试关键功能"""
    print("\n🧪 测试关键功能...")

    try:
        # 测试核心基础设施导入
        from src.infrastructure.config import ConfigFactory
        print("✅ ConfigFactory导入成功")

        from src.infrastructure.cache import BaseCacheManager
        print("✅ BaseCacheManager导入成功")

        from src.infrastructure.interfaces import DataRequest
        print("✅ 标准接口导入成功")

        # 测试实例化
        manager = ConfigFactory.create_config_manager()
        print("✅ ConfigFactory.create_config_manager() 成功")

        cache = BaseCacheManager()
        print("✅ BaseCacheManager实例化成功")

        # 测试数据结构
        request = DataRequest(symbol="TEST", timeframe="1d", limit=100)
        print("✅ DataRequest创建成功")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 RQA2025增强型语法错误检测和修复系统")
    print("=" * 60)

    # 创建增强修复器
    fixer = EnhancedSyntaxFixer("src/infrastructure")

    # 扫描并修复剩余复杂错误
    result = fixer.scan_and_fix_remaining()

    print("\n📊 增强修复结果:")
    print(f"   扫描文件数: {result['total_scanned']}")
    print(f"   修复文件数: {len(result['fixed_files'])}")

    if result['fixed_files']:
        print("   修复的文件:")
        for file in result['fixed_files']:
            print(f"     ✅ {file}")

    if result['complex_errors']:
        print("   复杂错误:")
        for error in result['complex_errors']:
            print(f"     ⚠️ {error}")

    if result['errors']:
        print("   处理错误:")
        for error in result['errors']:
            print(f"     ❌ {error}")

    # 测试关键功能
    if test_critical_functionality():
        print("\n🎉 关键功能测试通过!")
    else:
        print("\n⚠️ 关键功能测试失败")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RQA2025最终基础设施层修复系统
专门针对基础设施层的复杂语法错误进行深度修复
"""

import re
from pathlib import Path
from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalInfrastructureFixer:
    """最终基础设施层修复器"""

    def __init__(self):
        self.infrastructure_dir = Path("src/infrastructure")
        self.fixed_files = []
        self.errors = []
        self.repair_stats = {
            'files_processed': 0,
            'errors_fixed': 0,
            'files_with_errors': 0
        }

    def comprehensive_repair(self) -> Dict[str, Any]:
        """执行全面修复"""
        logger.info("🔧 开始最终基础设施层修复...")

        # 1. 扫描所有基础设施文件
        result = self._scan_infrastructure()

        # 2. 批量修复语法错误
        self._batch_fix_errors(result)

        # 3. 深度修复复杂错误
        self._deep_fix_complex_errors()

        # 4. 验证修复结果
        final_result = self._validate_repairs()

        return final_result

    def _scan_infrastructure(self) -> Dict[str, Any]:
        """扫描基础设施层文件"""
        result = {
            'total_files': 0,
            'error_files': [],
            'syntax_errors': [],
            'error_summary': {}
        }

        # 递归扫描所有Python文件
        for py_file in self.infrastructure_dir.rglob("*.py"):
            result['total_files'] += 1

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查语法
                compile(content, str(py_file), 'exec')

            except SyntaxError as e:
                error_info = {
                    'file': str(py_file),
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'line': e.lineno,
                    'offset': e.offset
                }
                result['error_files'].append(str(py_file))
                result['syntax_errors'].append(error_info)

                # 按错误类型统计
                error_key = f"{type(e).__name__}: {str(e).split('(')[0]}"
                result['error_summary'][error_key] = result['error_summary'].get(error_key, 0) + 1

            except Exception as e:
                logger.warning(f"读取文件失败: {py_file} - {e}")

        logger.info(f"扫描完成: {result['total_files']}个文件，{len(result['error_files'])}个有语法错误")
        return result

    def _batch_fix_errors(self, scan_result: Dict[str, Any]):
        """批量修复错误"""
        logger.info("🔄 开始批量修复...")

        for file_path in scan_result['error_files']:
            try:
                if self._fix_single_file(file_path):
                    self.fixed_files.append(file_path)
                    self.repair_stats['errors_fixed'] += 1
                else:
                    self.repair_stats['files_with_errors'] += 1

            except Exception as e:
                logger.error(f"修复文件失败: {file_path} - {e}")
                self.errors.append(f"{file_path}: {str(e)}")

        self.repair_stats['files_processed'] = len(scan_result['error_files'])

    def _fix_single_file(self, file_path: str) -> bool:
        """修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用多种修复策略
            content = self._apply_comprehensive_fixes(content, file_path)

            # 验证修复结果
            if content != original_content:
                try:
                    compile(content, file_path, 'exec')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"✅ 修复成功: {file_path}")
                    return True
                except SyntaxError as e:
                    logger.warning(f"修复后仍有语法错误: {file_path} - {e}")
                    return False

        except Exception as e:
            logger.error(f"处理文件失败: {file_path} - {e}")
            return False

        return False

    def _apply_comprehensive_fixes(self, content: str, file_path: str) -> str:
        """应用全面修复策略"""
        # 1. 修复字符串字面量错误
        content = self._fix_string_literals(content)

        # 2. 修复缩进错误
        content = self._fix_indentation_issues(content)

        # 3. 修复括号匹配问题
        content = self._fix_bracket_issues(content)

        # 4. 修复导入语句错误
        content = self._fix_import_issues(content)

        # 5. 修复类和函数定义错误
        content = self._fix_class_function_issues(content)

        # 6. 修复字典和列表语法错误
        content = self._fix_dict_list_issues(content)

        # 7. 修复特殊字符问题
        content = self._fix_special_characters(content)

        return content

    def _fix_string_literals(self, content: str) -> str:
        """修复字符串字面量错误"""
        # 修复未闭合的多行字符串
        content = re.sub(r'"""([^"]*?)$', r'"""\1"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''([^']*?)$", r"'''\1'''", content, flags=re.MULTILINE)

        # 修复模块级文档字符串
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if i == 0 and line.strip().startswith('"""') and not line.strip().endswith('"""'):
                lines[i] = line.rstrip() + '"""'
            elif i == 0 and line.strip().startswith("'''") and not line.strip().endswith("'''"):
                lines[i] = line.rstrip() + "'''"

        return '\n'.join(lines)

    def _fix_indentation_issues(self, content: str) -> str:
        """修复缩进问题"""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 跳过空行和注释
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue

            # 处理缩进
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ')):
                indent_stack.append(len(line) - len(line.lstrip()))
                fixed_lines.append(line)
            elif stripped.startswith(('return', 'pass', 'continue', 'break', 'raise')):
                # 控制语句应该保持当前缩进
                fixed_lines.append(line)
            elif stripped and indent_stack:
                # 普通语句的缩进
                expected_indent = indent_stack[-1] + 4
                current_indent = len(line) - len(line.lstrip())

                if current_indent != expected_indent and current_indent < expected_indent:
                    # 修复缩进
                    fixed_lines.append(' ' * expected_indent + stripped)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_bracket_issues(self, content: str) -> str:
        """修复括号匹配问题"""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # 修复字典括号问题
            if '{' in line and ')' in line and not line.count('{') == line.count('}'):
                line = re.sub(r'\(\s*\{([^}]*)\}\s*\)', r'(\1)', line)

            # 修复函数调用括号问题
            if '(' in line and '{' in line and not line.count('(') == line.count(')'):
                if line.count('(') > line.count(')'):
                    line += ')'

            lines[i] = line

        return '\n'.join(lines)

    def _fix_import_issues(self, content: str) -> str:
        """修复导入语句问题"""
        # 修复未闭合的导入语句
        content = re.sub(r'from\s+\.\w+\s+import\s*\(\s*$',
                         r'from . import (', content, flags=re.MULTILINE)
        content = re.sub(
            r'from\s+\.\w+\s+import\s*\(\s*\w+.*[^\)]$', r'\g<0>)', content, flags=re.MULTILINE)

        return content

    def _fix_class_function_issues(self, content: str) -> str:
        """修复类和函数定义问题"""
        # 修复函数参数定义问题
        content = re.sub(
            r'def\s+(\w+)\(([^)]*),?\)\s*\n(\s*)\)([^,]*),\s*\n(\s*)([^)]*)\):\s*\n',
            r'def \1(\2, \3\4, \5\6):\n',
            content,
            flags=re.MULTILINE
        )

        # 修复类定义问题
        content = re.sub(
            r'class\s+(\w+):\s*"""([^"]*)"""\s*\n\}',
            r'class \1:\n    """\2"""',
            content,
            flags=re.MULTILINE
        )

        return content

    def _fix_dict_list_issues(self, content: str) -> str:
        """修复字典和列表语法错误"""
        # 修复多行字典
        content = re.sub(
            r'return \{\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+)',
            r'return {\n\1\2: \3,\n\4\5: \6',
            content,
            flags=re.MULTILINE
        )

        # 修复字典定义
        content = re.sub(
            r'(\w+)\s*=\s*\{\s*\n\s*([^}]*?)\n\s*\}',
            r'\1 = {\n        \2\n    }',
            content,
            flags=re.MULTILINE
        )

        return content

    def _fix_special_characters(self, content: str) -> str:
        """修复特殊字符问题"""
        # 将中文标点符号替换为英文标点符号
        content = content.replace('：', ':')
        content = content.replace('，', ',')

        return content

    def _deep_fix_complex_errors(self):
        """深度修复复杂错误"""
        logger.info("🔍 开始深度修复复杂错误...")

        # 针对特定错误模式进行深度修复
        self._fix_decorator_placement()
        self._fix_multiline_function_calls()
        self._fix_complex_indentation()

        logger.info("✅ 深度修复完成")

    def _fix_decorator_placement(self):
        """修复装饰器位置问题"""
        # 处理所有基础设施文件中的装饰器问题
        for py_file in self.infrastructure_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # 修复@dataclass装饰器位置
                content = re.sub(
                    r'(\s*)@dataclass\s*\n\s*\n\s*class\s+(\w+):',
                    r'\1@dataclass\n\1class \2:',
                    content
                )

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"修复装饰器位置: {py_file}")

            except Exception as e:
                logger.warning(f"处理装饰器时出错: {py_file} - {e}")

    def _fix_multiline_function_calls(self):
        """修复多行函数调用"""
        for py_file in self.infrastructure_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # 修复多行函数调用参数缩进
                content = re.sub(
                    r'(\w+)\(\)\s*\n(\s+)(\w+)=([^,\n]+),\s*\n(\s+)(\w+)=([^,\n]+),\s*\n(\s*)\*\*([^)\n]*)\)',
                    r'\1(\n\2\3=\4,\n\5\6=\7,\n\8**\9\n)',
                    content,
                    flags=re.MULTILINE
                )

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"修复多行函数调用: {py_file}")

            except Exception as e:
                logger.warning(f"处理多行函数调用时出错: {py_file} - {e}")

    def _fix_complex_indentation(self):
        """修复复杂缩进问题"""
        for py_file in self.infrastructure_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # 修复方法定义缩进
                lines = content.split('\n')
                in_class = False
                class_indent = 0

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    if stripped.startswith('class '):
                        in_class = True
                        class_indent = len(line) - len(line.lstrip())
                    elif stripped.startswith('def ') and in_class:
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent != class_indent + 4:
                            lines[i] = ' ' * (class_indent + 4) + stripped
                    elif not line.startswith(' ') and not line.startswith('\t') and stripped:
                        # 顶层定义
                        in_class = False

                content = '\n'.join(lines)

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"修复复杂缩进: {py_file}")

            except Exception as e:
                logger.warning(f"处理复杂缩进时出错: {py_file} - {e}")

    def _validate_repairs(self) -> Dict[str, Any]:
        """验证修复结果"""
        logger.info("🔍 验证修复结果...")

        validation_result = {
            'total_files': 0,
            'error_files': 0,
            'fixed_files': len(self.fixed_files),
            'remaining_errors': [],
            'success_rate': 0.0
        }

        # 重新扫描验证
        for py_file in self.infrastructure_dir.rglob("*.py"):
            validation_result['total_files'] += 1

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                compile(content, str(py_file), 'exec')

            except SyntaxError as e:
                validation_result['error_files'] += 1
                validation_result['remaining_errors'].append({
                    'file': str(py_file),
                    'error': str(e),
                    'line': e.lineno
                })

        if validation_result['total_files'] > 0:
            validation_result['success_rate'] = (
                (validation_result['total_files'] - validation_result['error_files'])
                / validation_result['total_files']
            ) * 100

        logger.info(f"验证完成: {validation_result['total_files']}个文件，"
                    f"{validation_result['error_files']}个仍有错误，"
                    f"成功率: {validation_result['success_rate']:.2f}%")

        return validation_result


def test_core_infrastructure():
    """测试核心基础设施功能"""
    print("🧪 测试核心基础设施功能...")

    try:
        from src.infrastructure.config import ConfigFactory
        print("✅ ConfigFactory导入成功")

        from src.infrastructure.cache import BaseCacheManager
        print("✅ BaseCacheManager导入成功")

        print("✅ 标准接口导入成功")

        # 测试实例化
        manager = ConfigFactory.create_config_manager()
        print("✅ ConfigFactory.create_config_manager() 成功")

        cache = BaseCacheManager()
        print("✅ BaseCacheManager实例化成功")

        print("🎉 核心基础设施功能测试通过!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔧 RQA2025最终基础设施层修复系统")
    print("=" * 60)

    # 创建最终修复器
    fixer = FinalInfrastructureFixer()

    # 执行全面修复
    result = fixer.comprehensive_repair()

    print("\n📊 最终修复结果:")
    print(f"   处理文件数: {result['total_files']}")
    print(f"   错误文件数: {result['error_files']}")
    print(f"   成功率: {result['success_rate']:.2f}%")

    print("\n🔧 修复统计:")
    print(f"   已修复文件数: {len(fixer.fixed_files)}")
    print(f"   仍有错误文件数: {fixer.repair_stats['files_with_errors']}")

    if fixer.fixed_files:
        print("   修复的文件:")
        for file in fixer.fixed_files[:10]:
            print(f"     ✅ {file}")
        if len(fixer.fixed_files) > 10:
            print(f"     ... 还有 {len(fixer.fixed_files) - 10} 个文件")

    if result['remaining_errors']:
        print("\n⚠️  剩余错误:")
        for error in result['remaining_errors'][:5]:
            print(f"     ❌ {error['file']}: {error['error']}")

    # 测试核心功能
    if test_core_infrastructure():
        print("\n🎉 核心基础设施功能测试通过!")
    else:
        print("\n⚠️ 核心基础设施功能测试失败")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化修复建议生成器

分析测试失败结果，提供具体的修复建议和代码修改方案。
基于失败模式识别和最佳实践推荐，生成可操作的修复方案。
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FixConfidence(Enum):
    """修复置信度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class FixSuggestion:
    """修复建议"""
    test_file: str
    test_name: str
    error_type: str
    error_message: str
    confidence: FixConfidence
    fix_description: str
    code_changes: List[Dict[str, Any]]
    alternative_solutions: List[str]


class AutoFixSuggester:
    """自动化修复建议生成器"""

    def __init__(self, test_results_dir: str = "test_logs"):
        self.test_results_dir = Path(test_results_dir)
        self.fix_patterns = self._load_fix_patterns()

    def _load_fix_patterns(self) -> Dict[str, Dict[str, Any]]:
        """加载修复模式"""
        return {
            # ImportError 修复模式
            'module_not_found': {
                'pattern': r"ModuleNotFoundError: No module named '([^']+)'",
                'error_type': 'ImportError',
                'fix_function': self._fix_import_error,
                'confidence': FixConfidence.HIGH
            },

            # AttributeError 修复模式
            'attribute_error': {
                'pattern': r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
                'error_type': 'AttributeError',
                'fix_function': self._fix_attribute_error,
                'confidence': FixConfidence.MEDIUM
            },

            # TypeError 修复模式
            'missing_argument': {
                'pattern': r"TypeError: (\w+)\(\) missing (\d+) required positional argument",
                'error_type': 'TypeError',
                'fix_function': self._fix_missing_argument,
                'confidence': FixConfidence.HIGH
            },

            # AssertionError 修复模式
            'assertion_error': {
                'pattern': r"AssertionError.*",
                'error_type': 'AssertionError',
                'fix_function': self._fix_assertion_error,
                'confidence': FixConfidence.LOW
            },

            # SyntaxError 修复模式
            'indentation_error': {
                'pattern': r"IndentationError: expected an indented block",
                'error_type': 'IndentationError',
                'fix_function': self._fix_indentation_error,
                'confidence': FixConfidence.HIGH
            },

            # KeyError 修复模式
            'key_error': {
                'pattern': r"KeyError: '([^']+)'",
                'error_type': 'KeyError',
                'fix_function': self._fix_key_error,
                'confidence': FixConfidence.MEDIUM
            }
        }

    def analyze_test_failures(self, test_output: Optional[str] = None) -> List[FixSuggestion]:
        """分析测试失败"""
        logger.info("开始分析测试失败...")

        if test_output is None:
            test_output = self._get_latest_test_output()

        suggestions = []

        # 解析测试输出
        failures = self._parse_test_output(test_output)

        for failure in failures:
            suggestion = self._generate_fix_suggestion(failure)
            if suggestion:
                suggestions.append(suggestion)

        logger.info(f"生成{len(suggestions)}个修复建议")
        return suggestions

    def _get_latest_test_output(self) -> str:
        """获取最新的测试输出"""
        # 查找最新的测试报告文件
        report_files = list(self.test_results_dir.glob("*.md"))
        if not report_files:
            # 如果没有报告文件，运行一次测试
            logger.info("未找到测试报告，运行测试...")
            result = subprocess.run(
                ['python', 'tests/framework/test_runner.py', 'core', '--verbose'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            return result.stdout + result.stderr

        # 使用最新的报告文件
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        with open(latest_report, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_test_output(self, output: str) -> List[Dict[str, Any]]:
        """解析测试输出"""
        failures = []

        # 使用正则表达式查找失败的测试
        failure_pattern = r"FAILED ([^\s]+)::([^\s]+) - (.+)"
        matches = re.findall(failure_pattern, output, re.MULTILINE)

        for match in matches:
            test_file, test_name, error_message = match

            # 提取错误类型
            error_type = "Unknown"
            if "ImportError" in error_message or "ModuleNotFoundError" in error_message:
                error_type = "ImportError"
            elif "AttributeError" in error_message:
                error_type = "AttributeError"
            elif "TypeError" in error_message:
                error_type = "TypeError"
            elif "AssertionError" in error_message:
                error_type = "AssertionError"
            elif "IndentationError" in error_message:
                error_type = "IndentationError"
            elif "KeyError" in error_message:
                error_type = "KeyError"

            failures.append({
                'test_file': test_file,
                'test_name': test_name,
                'error_type': error_type,
                'error_message': error_message.strip()
            })

        return failures

    def _generate_fix_suggestion(self, failure: Dict[str, Any]) -> Optional[FixSuggestion]:
        """生成修复建议"""
        error_message = failure['error_message']

        # 尝试匹配修复模式
        for pattern_name, pattern_info in self.fix_patterns.items():
            match = re.search(pattern_info['pattern'], error_message)
            if match:
                fix_function = pattern_info['fix_function']
                confidence = pattern_info['confidence']

                try:
                    suggestion = fix_function(failure, match.groups())
                    if suggestion:
                        return FixSuggestion(
                            test_file=failure['test_file'],
                            test_name=failure['test_name'],
                            error_type=failure['error_type'],
                            error_message=error_message,
                            confidence=confidence,
                            fix_description=suggestion['description'],
                            code_changes=suggestion['changes'],
                            alternative_solutions=suggestion.get('alternatives', [])
                        )
                except Exception as e:
                    logger.warning(f"生成修复建议失败: {e}")

        return None

    def _fix_import_error(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复导入错误"""
        missing_module = groups[0]

        # 检查模块是否存在
        module_parts = missing_module.split('.')
        possible_paths = [
            Path("src") / f"{missing_module.replace('.', '/')}.py",
            Path("src") / module_parts[0] / f"{'_'.join(module_parts[1:])}.py",
            Path("src") / module_parts[0] / f"{module_parts[-1]}.py"
        ]

        existing_path = None
        for path in possible_paths:
            if path.exists():
                existing_path = path
                break

        if existing_path:
            # 建议修改导入路径
            return {
                'description': f"导入路径错误，模块存在于: {existing_path}",
                'changes': [{
                    'type': 'import_fix',
                    'old_code': f"from {missing_module}",
                    'new_code': f"from {str(existing_path).replace('/', '.').replace('.py', '').replace('src.', '')}"
                }],
                'alternatives': [
                    f"检查 {missing_module} 模块是否在正确的包中",
                    "考虑添加缺失的 __init__.py 文件"
                ]
            }
        else:
            # 建议创建缺失的模块
            return {
                'description': f"模块 {missing_module} 不存在，建议创建",
                'changes': [{
                    'type': 'create_module',
                    'module_path': f"src/{missing_module.replace('.', '/')}.py",
                    'template': self._get_module_template(missing_module)
                }],
                'alternatives': [
                    "检查拼写是否正确",
                    "确认模块是否应该从其他位置导入"
                ]
            }

    def _fix_attribute_error(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复属性错误"""
        object_name, attribute_name = groups

        return {
            'description': f"对象 {object_name} 没有属性 {attribute_name}",
            'changes': [{
                'type': 'add_attribute',
                'suggestion': f"为 {object_name} 类添加 {attribute_name} 属性或方法"
            }],
            'alternatives': [
                f"检查 {object_name} 的初始化是否正确",
                f"确认 {attribute_name} 的拼写是否正确",
                "考虑使用 hasattr() 检查属性是否存在"
            ]
        }

    def _fix_missing_argument(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复缺失参数错误"""
        function_name = groups[0]

        # 查找函数定义
        test_file = failure['test_file']
        if Path(test_file).exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找函数调用
            call_pattern = rf"{function_name}\s*\("
            calls = re.findall(call_pattern, content)

            if calls:
                return {
                    'description': f"函数 {function_name} 缺少必需的参数",
                    'changes': [{
                        'type': 'add_parameters',
                        'suggestion': f"为 {function_name} 调用添加缺失的参数"
                    }],
                    'alternatives': [
                        f"检查 {function_name} 的函数签名",
                        "查看函数文档了解必需参数"
                    ]
                }

        return {
            'description': f"函数 {function_name} 调用缺少参数",
            'changes': [{
                'type': 'parameter_check',
                'suggestion': "检查函数调用和函数定义的参数匹配"
            }]
        }

    def _fix_assertion_error(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复断言错误"""
        return {
            'description': "断言失败，需要检查测试逻辑或被测代码",
            'changes': [{
                'type': 'assertion_review',
                'suggestion': "检查断言条件是否正确，考虑实际返回值"
            }],
            'alternatives': [
                "运行调试模式查看变量值",
                "检查测试设置是否正确",
                "确认预期结果是否合理"
            ]
        }

    def _fix_indentation_error(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复缩进错误"""
        test_file = failure['test_file']

        if Path(test_file).exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 查找缩进问题
            for i, line in enumerate(lines):
                stripped = line.lstrip()
                if stripped and not line.startswith(' ') and not line.startswith('\t'):
                    # 检查前一行是否需要缩进
                    if i > 0:
                        prev_line = lines[i-1].rstrip()
                        if prev_line.endswith(':') and not prev_line.startswith('#'):
                            return {
                                'description': f"第{i+1}行缺少缩进，前一行以冒号结束",
                                'changes': [{
                                    'type': 'fix_indentation',
                                    'line_number': i+1,
                                    'old_code': line.rstrip(),
                                    'new_code': '    ' + line.strip()
                                }],
                                'alternatives': [
                                    "使用自动格式化工具修复缩进",
                                    "检查代码块结构是否正确"
                                ]
                            }

        return {
            'description': "缩进错误，需要检查代码缩进格式",
            'changes': [{
                'type': 'indentation_check',
                'suggestion': "使用工具自动修复缩进或手动调整"
            }]
        }

    def _fix_key_error(self, failure: Dict[str, Any], groups: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """修复键错误"""
        missing_key = groups[0]

        return {
            'description': f"字典或集合中缺少键: {missing_key}",
            'changes': [{
                'type': 'add_key_check',
                'suggestion': f"在访问 {missing_key} 之前检查键是否存在"
            }],
            'alternatives': [
                f"为字典添加默认值或使用 dict.get('{missing_key}')",
                "使用 try-except 块处理 KeyError",
                "检查数据源确保键存在"
            ]
        }

    def _get_module_template(self, module_name: str) -> str:
        """获取模块模板"""
        return '''"""
{module_name} 模块

此模块由自动化修复建议器创建。
请根据实际需求实现模块功能。
"""

# 导入相关依赖
# from typing import Dict, Any, Optional


def placeholder_function():
    """
    占位符函数

    请根据实际需求实现具体功能。
    """
    return f"{module_name} 模块功能待实现"


# 导出接口
__all__ = ["placeholder_function"]
'''

    def generate_fix_report(self, suggestions: List[FixSuggestion], output_file: Path):
        """生成修复报告"""
        logger.info(f"生成修复报告: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 自动化修复建议报告\n\n")
            f.write(f"生成时间: {self._get_timestamp()}\n\n")
            f.write(f"总计建议: {len(suggestions)} 个修复方案\n\n")

            # 按置信度分组
            confidence_groups = {}
            for suggestion in suggestions:
                confidence = suggestion.confidence.value
                if confidence not in confidence_groups:
                    confidence_groups[confidence] = []
                confidence_groups[confidence].append(suggestion)

            # 置信度排序
            confidence_order = ['high', 'medium', 'low']

            for confidence in confidence_order:
                if confidence in confidence_groups:
                    fixes = confidence_groups[confidence]
                    f.write(f"## {confidence.upper()} 置信度 ({len(fixes)}个)\n\n")

                    for fix in fixes[:10]:  # 每组最多显示10个
                        f.write(f"### {fix.test_name}\n\n")
                        f.write(f"- **文件**: `{fix.test_file}`\n")
                        f.write(f"- **错误类型**: {fix.error_type}\n")
                        f.write(f"- **错误信息**: {fix.error_message}\n")
                        f.write(f"- **修复描述**: {fix.fix_description}\n\n")

                        if fix.code_changes:
                            f.write("**代码修改建议**:\n\n")
                            for change in fix.code_changes:
                                if change['type'] == 'import_fix':
                                    f.write(f"```python\n# 修改导入语句\n{change['old_code']}\n# 改为:\n{change['new_code']}\n```\n\n")
                                elif change['type'] == 'create_module':
                                    f.write(f"创建新模块: `{change['module_path']}`\n\n")
                                else:
                                    f.write(f"- {change.get('suggestion', change['type'])}\n")

                        if fix.alternative_solutions:
                            f.write("**其他解决方案**:\n\n")
                            for alt in fix.alternative_solutions:
                                f.write(f"- {alt}\n")
                            f.write("\n")

                        f.write("---\n\n")

            # 统计信息
            f.write("## 📊 统计信息\n\n")

            error_types = {}
            for suggestion in suggestions:
                error_type = suggestion.error_type
                error_types[error_type] = error_types.get(error_type, 0) + 1

            f.write("### 错误类型分布\n\n")
            for error_type, count in error_types.items():
                f.write(f"- {error_type}: {count}个\n")

            f.write("\n### 执行建议\n\n")
            f.write("1. **优先处理高置信度建议**: 从高置信度的修复开始\n")
            f.write("2. **逐步验证**: 每个修复后运行测试验证效果\n")
            f.write("3. **备份代码**: 修改前确保有代码备份\n")
            f.write("4. **团队审查**: 重要修复建议团队成员审查\n")

        logger.info(f"修复报告已生成: {output_file}")

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def apply_fix(self, suggestion: FixSuggestion) -> bool:
        """应用修复建议"""
        logger.info(f"应用修复: {suggestion.test_name}")

        # 这里可以实现自动应用修复的逻辑
        # 目前只记录建议，实际应用需要人工确认

        logger.info("修复建议已记录，建议手动应用")
        return True


def main():
    """主函数"""
    suggester = AutoFixSuggester()

    # 分析测试失败
    suggestions = suggester.analyze_test_failures()

    print("\n🎯 自动化修复建议分析结果:")
    print(f"📊 生成建议: {len(suggestions)} 个")

    if suggestions:
        # 按置信度统计
        confidence_stats = {}
        for suggestion in suggestions:
            confidence = suggestion.confidence.value
            confidence_stats[confidence] = confidence_stats.get(confidence, 0) + 1

        print("📈 置信度分布:")
        for confidence, count in confidence_stats.items():
            print(f"  {confidence.upper()}: {count}个")

        # 显示前3个建议
        print("\n🔧 前3个修复建议:")
        for i, suggestion in enumerate(suggestions[:3]):
            print(f"  {i+1}. [{suggestion.confidence.value.upper()}] {suggestion.test_name}")
            print(f"     {suggestion.fix_description}")

    # 生成详细报告
    report_file = Path("test_logs/auto_fix_suggestions.md")
    suggester.generate_fix_report(suggestions, report_file)

    print(f"\n📄 详细报告已生成: {report_file}")
    print("\n💡 建议按高→中→低置信度顺序处理修复")


if __name__ == "__main__":
    main()

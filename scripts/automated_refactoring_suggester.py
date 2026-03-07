#!/usr/bin/env python3
"""
自动化重构建议工具

基于代码分析自动生成重构建议和修复方案
"""

import re
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


class AutomatedRefactoringSuggester:
    """自动化重构建议器"""

    def __init__(self):
        self.layer_mapping = {
            'src/core': 'core',
            'src/infrastructure': 'infrastructure',
            'src/data': 'data',
            'src/gateway': 'gateway',
            'src/features': 'features',
            'src/ml': 'ml',
            'src/backtest': 'backtest',
            'src/risk': 'risk',
            'src/trading': 'trading',
            'src/engine': 'engine'
        }

        self.violations = []
        self.suggestions = []

    def analyze_codebase(self):
        """分析整个代码库"""
        print("🔍 分析代码库，生成重构建议...")

        for root_path, layer in self.layer_mapping.items():
            layer_dir = Path(root_path)
            if not layer_dir.exists():
                continue

            print(f"📁 分析 {layer} 层...")
            for file_path in layer_dir.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_suggestions = self.analyze_file(str(file_path), content, layer)
                    self.suggestions.extend(file_suggestions)

                except Exception as e:
                    print(f"⚠️ 无法分析文件 {file_path}: {e}")

        print(f"📋 生成了 {len(self.suggestions)} 条重构建议")

    def analyze_file(self, file_path: str, content: str, layer: str) -> List[Dict]:
        """分析单个文件"""
        suggestions = []

        # 1. 检查长函数
        function_suggestions = self.check_long_functions(file_path, content)
        suggestions.extend(function_suggestions)

        # 2. 检查重复代码
        duplicate_suggestions = self.check_duplicate_code(file_path, content)
        suggestions.extend(duplicate_suggestions)

        # 3. 检查大类
        class_suggestions = self.check_large_classes(file_path, content)
        suggestions.extend(class_suggestions)

        # 4. 检查复杂的条件语句
        condition_suggestions = self.check_complex_conditions(file_path, content)
        suggestions.extend(condition_suggestions)

        # 5. 检查魔法数字
        magic_number_suggestions = self.check_magic_numbers(file_path, content)
        suggestions.extend(magic_number_suggestions)

        # 6. 检查异常处理
        exception_suggestions = self.check_exception_handling(file_path, content)
        suggestions.extend(exception_suggestions)

        # 7. 检查依赖注入
        dependency_suggestions = self.check_dependency_injection(file_path, content, layer)
        suggestions.extend(dependency_suggestions)

        return suggestions

    def check_long_functions(self, file_path: str, content: str) -> List[Dict]:
        """检查长函数"""
        suggestions = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.body[-1].lineno if node.body else start_line
                    func_lines = end_line - start_line + 1

                    if func_lines > 30:
                        suggestions.append({
                            'type': 'long_function',
                            'file': file_path,
                            'function': node.name,
                            'line': start_line,
                            'severity': 'medium' if func_lines > 50 else 'low',
                            'description': f"函数 '{node.name}' 过长 ({func_lines}行)",
                            'suggestion': "考虑将函数拆分为更小的函数，每个函数负责单一职责",
                            'refactoring_type': 'extract_method',
                            'estimated_effort': 'medium'
                        })

        except:
            pass

        return suggestions

    def check_duplicate_code(self, file_path: str, content: str) -> List[Dict]:
        """检查重复代码"""
        suggestions = []

        lines = content.split('\n')

        # 简单的重复代码检测（3行以上重复）
        code_blocks = {}
        for i in range(len(lines) - 2):
            block = '\n'.join(lines[i:i+3])
            if len(block.strip()) > 20:  # 避免空行或注释
                if block in code_blocks:
                    code_blocks[block].append(i)
                else:
                    code_blocks[block] = [i]

        for block, line_numbers in code_blocks.items():
            if len(line_numbers) >= 2:
                suggestions.append({
                    'type': 'duplicate_code',
                    'file': file_path,
                    'lines': line_numbers,
                    'severity': 'medium',
                    'description': f"发现重复代码块 (在第{line_numbers}行)",
                    'suggestion': "考虑提取公共方法或使用策略模式",
                    'refactoring_type': 'extract_method',
                    'estimated_effort': 'high'
                })

        return suggestions

    def check_large_classes(self, file_path: str, content: str) -> List[Dict]:
        """检查大类"""
        suggestions = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_lines = len(content.split(
                        '\n')[node.lineno-1:node.body[-1].lineno]) if node.body else 0

                    if class_lines > 200:
                        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])

                        suggestions.append({
                            'type': 'large_class',
                            'file': file_path,
                            'class': node.name,
                            'line': node.lineno,
                            'severity': 'high' if class_lines > 300 else 'medium',
                            'description': f"类 '{node.name}' 过大 ({class_lines}行, {method_count}个方法)",
                            'suggestion': "考虑将类拆分为更小的类，或使用组合模式",
                            'refactoring_type': 'extract_class',
                            'estimated_effort': 'high'
                        })

        except:
            pass

        return suggestions

    def check_complex_conditions(self, file_path: str, content: str) -> List[Dict]:
        """检查复杂的条件语句"""
        suggestions = []

        # 查找嵌套的if语句
        lines = content.split('\n')
        indent_stack = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith(('if ', 'elif ', 'else:')):
                # 计算缩进级别
                indent_level = len(line) - len(line.lstrip())

                # 清理栈中缩进级别更高的项目
                while indent_stack and indent_stack[-1][1] >= indent_level:
                    indent_stack.pop()

                indent_stack.append((i, indent_level))

                # 检查嵌套深度
                if len(indent_stack) > 3:
                    suggestions.append({
                        'type': 'complex_condition',
                        'file': file_path,
                        'line': i,
                        'severity': 'medium',
                        'description': f"条件语句嵌套过深 (深度: {len(indent_stack)})",
                        'suggestion': "考虑使用卫语句(Early Return)或提取方法",
                        'refactoring_type': 'simplify_condition',
                        'estimated_effort': 'low'
                    })

        return suggestions

    def check_magic_numbers(self, file_path: str, content: str) -> List[Dict]:
        """检查魔法数字"""
        suggestions = []

        # 查找数字常量
        number_pattern = r'\b\d{2,}\b'  # 两位数以上的数字
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            matches = re.finditer(number_pattern, line)

            for match in matches:
                number = match.group()
                # 排除常见非魔法的数字
                if number not in ['100', '1000', '3600', '86400', '80', '443', '22', '21']:
                    suggestions.append({
                        'type': 'magic_number',
                        'file': file_path,
                        'line': i,
                        'number': number,
                        'severity': 'low',
                        'description': f"魔法数字: {number}",
                        'suggestion': "考虑使用常量或配置参数",
                        'refactoring_type': 'extract_constant',
                        'estimated_effort': 'low'
                    })

        return suggestions

    def check_exception_handling(self, file_path: str, content: str) -> List[Dict]:
        """检查异常处理"""
        suggestions = []

        # 检查裸except
        if re.search(r'except\s*:', content):
            suggestions.append({
                'type': 'bare_except',
                'file': file_path,
                'severity': 'medium',
                'description': "使用裸except语句",
                'suggestion': "指定具体的异常类型或使用Exception",
                'refactoring_type': 'improve_exception',
                'estimated_effort': 'low'
            })

        # 检查未处理的异常
        try_blocks = len(re.findall(r'\btry\s*:', content))
        except_blocks = len(re.findall(r'\bexcept\s', content))

        if try_blocks > except_blocks:
            suggestions.append({
                'type': 'unhandled_exception',
                'file': file_path,
                'severity': 'high',
                'description': f"可能存在未处理的异常 ({try_blocks}个try, {except_blocks}个except)",
                'suggestion': "确保所有try块都有对应的except处理",
                'refactoring_type': 'add_exception_handling',
                'estimated_effort': 'medium'
            })

        return suggestions

    def check_dependency_injection(self, file_path: str, content: str, layer: str) -> List[Dict]:
        """检查依赖注入"""
        suggestions = []

        # 检查硬编码的依赖
        hard_coded_imports = re.findall(r'from\s+src\.\w+\s+import\s+\w+', content)

        for import_stmt in hard_coded_imports:
            if 'factory' not in import_stmt.lower() and 'interface' not in import_stmt.lower():
                suggestions.append({
                    'type': 'hard_coded_dependency',
                    'file': file_path,
                    'import': import_stmt,
                    'severity': 'medium',
                    'description': f"可能存在硬编码依赖: {import_stmt}",
                    'suggestion': "考虑使用依赖注入或工厂模式",
                    'refactoring_type': 'dependency_injection',
                    'estimated_effort': 'high'
                })

        return suggestions

    def generate_refactoring_plan(self):
        """生成重构计划"""
        plan = []

        plan.append("# 自动化重构建议报告")
        plan.append("")
        plan.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        plan.append(f"总建议数: {len(self.suggestions)}")
        plan.append("")

        # 按严重程度分组
        by_severity = defaultdict(list)
        for suggestion in self.suggestions:
            by_severity[suggestion['severity']].append(suggestion)

        # 高优先级建议
        if 'high' in by_severity:
            plan.append("## 🚨 高优先级重构建议")
            for suggestion in by_severity['high'][:10]:  # 最多显示10个
                plan.append(f"### {suggestion['type']}")
                plan.append(f"- **文件**: {suggestion['file']}")
                if 'function' in suggestion:
                    plan.append(f"- **函数**: {suggestion['function']}")
                if 'class' in suggestion:
                    plan.append(f"- **类**: {suggestion['class']}")
                if 'line' in suggestion:
                    plan.append(f"- **行号**: {suggestion['line']}")
                plan.append(f"- **问题**: {suggestion['description']}")
                plan.append(f"- **建议**: {suggestion['suggestion']}")
                plan.append(f"- **重构类型**: {suggestion['refactoring_type']}")
                plan.append(f"- **预计工作量**: {suggestion['estimated_effort']}")
                plan.append("")

        # 中优先级建议
        if 'medium' in by_severity:
            plan.append("## ⚠️ 中优先级重构建议")
            for suggestion in by_severity['medium'][:15]:  # 最多显示15个
                plan.append(f"### {suggestion['type']}")
                plan.append(f"- **文件**: {suggestion['file']}")
                if 'line' in suggestion:
                    plan.append(f"- **行号**: {suggestion['line']}")
                plan.append(f"- **问题**: {suggestion['description']}")
                plan.append(f"- **建议**: {suggestion['suggestion']}")
                plan.append("")

        # 低优先级建议
        if 'low' in by_severity:
            plan.append("## 📝 低优先级重构建议")
            plan.append(f"共 {len(by_severity['low'])} 个建议")
            plan.append("")
            plan.append("这些建议可以逐步实施，不影响系统核心功能")

        # 重构实施计划
        plan.append("## 📋 重构实施计划")
        plan.append("")
        plan.append("### 第一阶段 (2周内)")
        plan.append("1. **修复高严重度问题**")
        plan.append("   - 解决长函数问题")
        plan.append("   - 处理未处理的异常")
        plan.append("   - 修复硬编码依赖")
        plan.append("")
        plan.append("### 第二阶段 (4周内)")
        plan.append("1. **重构大类和复杂函数**")
        plan.append("   - 拆分大类")
        plan.append("   - 简化复杂条件")
        plan.append("   - 提取重复代码")
        plan.append("")
        plan.append("### 第三阶段 (持续改进)")
        plan.append("1. **代码质量优化**")
        plan.append("   - 替换魔法数字")
        plan.append("   - 优化异常处理")
        plan.append("   - 实施依赖注入")
        plan.append("")

        # 按文件分组的建议
        plan.append("## 📂 按文件分组的重构建议")
        by_file = defaultdict(list)
        for suggestion in self.suggestions:
            by_file[suggestion['file']].append(suggestion)

        for file_path, file_suggestions in sorted(by_file.items())[:20]:  # 最多显示20个文件
            plan.append(f"### {file_path}")
            plan.append(f"- 建议数: {len(file_suggestions)}")

            severity_count = defaultdict(int)
            for suggestion in file_suggestions:
                severity_count[suggestion['severity']] += 1

            if severity_count['high'] > 0:
                plan.append(f"- 高优先级: {severity_count['high']}")
            if severity_count['medium'] > 0:
                plan.append(f"- 中优先级: {severity_count['medium']}")
            if severity_count['low'] > 0:
                plan.append(f"- 低优先级: {severity_count['low']}")
            plan.append("")

        with open('reports/AUTOMATED_REFACTORING_SUGGESTIONS.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(plan))

    def apply_quick_fixes(self):
        """应用快速修复"""
        print("🔧 应用快速修复...")

        fixable_suggestions = [s for s in self.suggestions if s.get('refactoring_type') in [
            'extract_constant']]

        for suggestion in fixable_suggestions[:5]:  # 限制数量
            try:
                self._apply_magic_number_fix(suggestion)
                print(f"✅ 已修复: {suggestion['description']}")
            except Exception as e:
                print(f"❌ 修复失败: {e}")

    def _apply_magic_number_fix(self, suggestion: Dict):
        """应用魔法数字修复"""
        file_path = suggestion['file']
        line_number = suggestion['line']
        number = suggestion['number']

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        if line_number <= len(lines):
            # 生成常量名
            constant_name = f"CONST_{number}"

            # 在文件开头添加常量定义
            import_lines = []
            code_lines = []

            for i, line in enumerate(lines):
                if i == 0 and not line.startswith('#'):
                    import_lines.append(f"{constant_name} = {number}")
                    import_lines.append("")
                    code_lines.append(line)
                elif line.startswith(('import ', 'from ')):
                    import_lines.append(line)
                else:
                    if not import_lines and not line.startswith('#'):
                        import_lines.extend(['', f"{constant_name} = {number}", ''])
                    code_lines.append(line)

            # 替换数字
            target_line = lines[line_number - 1]
            target_line = re.sub(r'\b' + number + r'\b', constant_name, target_line)
            lines[line_number - 1] = target_line

            # 重新组合文件
            new_content = '\n'.join(import_lines + code_lines)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

    def run_analysis(self):
        """运行分析"""
        print("🚀 开始自动化重构建议分析...")
        print("="*60)

        try:
            # 1. 分析代码库
            self.analyze_codebase()

            # 2. 生成重构计划
            self.generate_refactoring_plan()

            # 3. 应用快速修复
            self.apply_quick_fixes()

            print("\n📋 重构建议报告已生成:")
            print("   - reports/AUTOMATED_REFACTORING_SUGGESTIONS.md")
            print("🎉 自动化重构建议完成！")
            return True

        except Exception as e:
            print(f"\n❌ 分析过程中出错: {e}")
            return False


def main():
    """主函数"""
    suggester = AutomatedRefactoringSuggester()
    success = suggester.run_analysis()

    if success:
        print("\n" + "="*60)
        print("自动化重构建议成功完成！")
        print("✅ 代码分析完成")
        print("✅ 重构建议生成")
        print("✅ 快速修复应用")
        print("="*60)
    else:
        print("\n❌ 自动化重构建议失败！")


if __name__ == "__main__":
    main()

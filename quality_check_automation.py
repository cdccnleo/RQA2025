#!/usr/bin/env python3
"""
基础设施层配置管理自动化质量检查工具
Phase 2.3: 建立自动化质量检查
"""

import os
import re
import ast
from typing import Dict, List, Any
from collections import defaultdict
import datetime


class CodeQualityChecker:
    """代码质量检查器"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.results = {
            'complexity': [],
            'duplicates': [],
            'imports': [],
            'documentation': [],
            'style': []
        }

    def analyze_complexity(self) -> List[Dict[str, Any]]:
        """分析代码复杂度"""
        print("🔍 分析代码复杂度...")

        complexity_results = []

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 解析AST获取更准确的复杂度
                        tree = ast.parse(content)

                        # 统计类和函数
                        classes = len([node for node in ast.walk(tree)
                                      if isinstance(node, ast.ClassDef)])
                        functions = len([node for node in ast.walk(
                            tree) if isinstance(node, ast.FunctionDef)])
                        methods = len([node for node in ast.walk(tree)
                                       if isinstance(node, ast.FunctionDef) and
                                       node.name not in ['__init__', '__str__', '__repr__']])

                        # 计算圈复杂度（简化版）
                        complexity_score = functions + classes * 2 + methods

                        # 计算行数
                        lines = len(content.split('\n'))

                        # 判断复杂度级别
                        level = 'low'
                        if lines > 300 or complexity_score > 30:
                            level = 'high'
                        elif lines > 150 or complexity_score > 15:
                            level = 'medium'

                        complexity_results.append({
                            'file': rel_path,
                            'lines': lines,
                            'classes': classes,
                            'functions': functions,
                            'methods': methods,
                            'complexity_score': complexity_score,
                            'level': level
                        })

                    except Exception as e:
                        complexity_results.append({
                            'file': rel_path,
                            'error': str(e),
                            'level': 'error'
                        })

        self.results['complexity'] = complexity_results
        return complexity_results

    def analyze_duplicates(self) -> List[Dict[str, Any]]:
        """分析重复代码"""
        print("🔍 分析重复代码...")

        duplicate_results = []
        code_blocks = defaultdict(list)

        # 收集所有代码块
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        # 分析连续的代码块（5行）
                        for i in range(len(lines) - 5):
                            block_lines = lines[i:i+5]
                            # 过滤注释和空行
                            code_lines = [line.strip() for line in block_lines
                                          if line.strip() and not line.strip().startswith('#')]

                            if len(code_lines) >= 3:  # 至少3行有效代码
                                block = '\n'.join(code_lines)
                                if len(block) > 50:  # 代码块足够长
                                    code_blocks[block].append((rel_path, i))

                    except Exception as e:
                        continue

        # 找出重复的代码块
        for block, locations in code_blocks.items():
            if len(locations) > 1:
                duplicate_results.append({
                    'block': block[:100] + '...' if len(block) > 100 else block,
                    'occurrences': len(locations),
                    'locations': locations,
                    'severity': 'high' if len(locations) > 3 else 'medium'
                })

        self.results['duplicates'] = duplicate_results
        return duplicate_results

    def analyze_imports(self) -> List[Dict[str, Any]]:
        """分析导入语句"""
        print("🔍 分析导入语句...")

        import_results = []

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')
                        imports = []

                        for i, line in enumerate(lines):
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                imports.append({
                                    'line': i + 1,
                                    'statement': line,
                                    'type': 'star' if '*' in line else 'normal',
                                    'relative': line.startswith('from .') or line.startswith('from ..'),
                                    'unified': 'from infrastructure.config.core.imports import' in line
                                })

                        if imports:
                            import_results.append({
                                'file': rel_path,
                                'imports': imports,
                                'star_imports': len([imp for imp in imports if imp['type'] == 'star']),
                                'relative_imports': len([imp for imp in imports if imp['relative']]),
                                'unified_imports': len([imp for imp in imports if imp['unified']])
                            })

                    except Exception as e:
                        continue

        self.results['imports'] = import_results
        return import_results

    def analyze_documentation(self) -> List[Dict[str, Any]]:
        """分析文档覆盖率"""
        print("🔍 分析文档覆盖...")

        doc_results = []

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 检查模块级文档
                        module_doc = bool(re.search(r'^\s*""".*?"""',
                                          content, re.DOTALL | re.MULTILINE))

                        # 解析AST获取函数信息
                        tree = ast.parse(content)

                        functions = []
                        documented_functions = 0

                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                functions.append(node.name)

                                # 检查是否有文档字符串
                                if ast.get_docstring(node):
                                    documented_functions += 1

                        classes = []
                        documented_classes = 0

                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                classes.append(node.name)

                                # 检查是否有文档字符串
                                if ast.get_docstring(node):
                                    documented_classes += 1

                        doc_results.append({
                            'file': rel_path,
                            'module_doc': module_doc,
                            'functions': len(functions),
                            'documented_functions': documented_functions,
                            'function_coverage': documented_functions / len(functions) if functions else 1.0,
                            'classes': len(classes),
                            'documented_classes': documented_classes,
                            'class_coverage': documented_classes / len(classes) if classes else 1.0
                        })

                    except Exception as e:
                        continue

        self.results['documentation'] = doc_results
        return doc_results

    def analyze_style(self) -> List[Dict[str, Any]]:
        """分析代码风格"""
        print("🔍 分析代码风格...")

        style_results = []

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.base_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')

                        # 检查各种风格问题
                        issues = []

                        for i, line in enumerate(lines):
                            line_num = i + 1
                            stripped = line.strip()

                            # 检查行长度
                            if len(line) > 88:  # PEP8推荐88字符
                                issues.append({
                                    'line': line_num,
                                    'type': 'line_too_long',
                                    'message': f'行长度 {len(line)} 超过88字符',
                                    'severity': 'low'
                                })

                            # 检查连续空行
                            if i > 0 and not stripped and not lines[i-1].strip():
                                # 检查是否在函数或类内部
                                in_function = False
                                for j in range(i-1, -1, -1):
                                    prev_line = lines[j].strip()
                                    if prev_line.startswith('def ') or prev_line.startswith('class '):
                                        in_function = True
                                        break
                                    elif prev_line and not prev_line.startswith(' ') and not prev_line.startswith('\t'):
                                        break

                                if not in_function:
                                    issues.append({
                                        'line': line_num,
                                        'type': 'multiple_blank_lines',
                                        'message': '函数外部有多余的连续空行',
                                        'severity': 'low'
                                    })

                            # 检查结尾空格
                            if line.endswith(' ') or line.endswith('\t'):
                                issues.append({
                                    'line': line_num,
                                    'type': 'trailing_whitespace',
                                    'message': '行尾有空格或制表符',
                                    'severity': 'low'
                                })

                        style_results.append({
                            'file': rel_path,
                            'issues': issues,
                            'issue_count': len(issues),
                            'severity_high': len([i for i in issues if i['severity'] == 'high']),
                            'severity_medium': len([i for i in issues if i['severity'] == 'medium']),
                            'severity_low': len([i for i in issues if i['severity'] == 'low'])
                        })

                    except Exception as e:
                        continue

        self.results['style'] = style_results
        return style_results

    def generate_report(self) -> str:
        """生成质量报告"""
        print("📊 生成质量报告...")

        # 运行所有分析
        self.analyze_complexity()
        self.analyze_duplicates()
        self.analyze_imports()
        self.analyze_documentation()
        self.analyze_style()

        # 计算总体统计
        total_files = len(set([
            item['file'] for category in self.results.values()
            for item in category if 'file' in item
        ]))

        # 复杂度统计
        complexity_stats = {
            'high': len([f for f in self.results['complexity'] if f.get('level') == 'high']),
            'medium': len([f for f in self.results['complexity'] if f.get('level') == 'medium']),
            'low': len([f for f in self.results['complexity'] if f.get('level') == 'low'])
        }

        # 重复代码统计
        duplicate_stats = {
            'blocks': len(self.results['duplicates']),
            'total_occurrences': sum(d['occurrences'] for d in self.results['duplicates'])
        }

        # 导入统计
        import_stats = {
            'files_with_imports': len(self.results['imports']),
            'star_imports': sum(f['star_imports'] for f in self.results['imports']),
            'relative_imports': sum(f['relative_imports'] for f in self.results['imports']),
            'unified_imports': sum(f['unified_imports'] for f in self.results['imports'])
        }

        # 文档统计
        doc_stats = self.results['documentation']
        if doc_stats:
            avg_function_coverage = sum(d['function_coverage'] for d in doc_stats) / len(doc_stats)
            avg_class_coverage = sum(d['class_coverage'] for d in doc_stats) / len(doc_stats)
            module_docs = sum(1 for d in doc_stats if d['module_doc']) / len(doc_stats)
        else:
            avg_function_coverage = avg_class_coverage = module_docs = 0

        # 风格统计
        style_stats = self.results['style']
        total_style_issues = sum(f['issue_count'] for f in style_stats)

        # 生成报告
        report = f"""
# 基础设施层配置管理代码质量报告

**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析文件数**: {total_files}

## 📊 总体概览

| 指标 | 值 | 状态 |
|------|-----|------|
| **复杂度问题文件** | {complexity_stats['high']} 高 + {complexity_stats['medium']} 中 | {'⚠️ 需要关注' if complexity_stats['high'] > 0 else '✅ 良好'} |
| **重复代码块** | {duplicate_stats['blocks']} 个 | {'⚠️ 需要重构' if duplicate_stats['blocks'] > 5 else '✅ 可接受'} |
| **星号导入** | {import_stats['star_imports']} 个 | {'⚠️ 建议修复' if import_stats['star_imports'] > 0 else '✅ 良好'} |
| **相对导入** | {import_stats['relative_imports']} 个 | {'⚠️ 建议优化' if import_stats['relative_imports'] > 10 else '✅ 良好'} |
| **文档覆盖率** | 函数: {avg_function_coverage:.1%}, 类: {avg_class_coverage:.1%} | {'✅ 优秀' if avg_function_coverage > 0.8 else '⚠️ 需要改进'} |
| **代码风格问题** | {total_style_issues} 个 | {'⚠️ 需要清理' if total_style_issues > 50 else '✅ 良好'} |

## 🔍 详细分析

### 1. 代码复杂度分析

**高复杂度文件** ({complexity_stats['high']} 个):
"""
        for item in sorted(self.results['complexity'], key=lambda x: x.get('complexity_score', 0), reverse=True):
            if item.get('level') == 'high':
                report += f"- `{item['file']}`: {item['lines']}行, 复杂度{item['complexity_score']}\n"

        report += f"""

**复杂度分布**:
- 高复杂度: {complexity_stats['high']} 个文件
- 中复杂度: {complexity_stats['medium']} 个文件
- 低复杂度: {complexity_stats['low']} 个文件

### 2. 重复代码分析

**发现重复代码块**: {duplicate_stats['blocks']} 个

"""

        for dup in self.results['duplicates'][:5]:  # 只显示前5个
            report += f"""**重复代码块** (出现 {dup['occurrences']} 次):
```python
{dup['block']}
```

**出现位置**:
"""
            for loc in dup['locations'][:3]:  # 只显示前3个位置
                report += f"- `{loc[0]}` 第{loc[1]}行\n"
            if len(dup['locations']) > 3:
                report += f"- ... 等{len(dup['locations'])-3}个位置\n"
            report += "\n"

        report += f"""
### 3. 导入语句分析

**导入统计**:
- 使用统一导入的文件: {import_stats['files_with_imports']} 个
- 星号导入总数: {import_stats['star_imports']} 个
- 相对导入总数: {import_stats['relative_imports']} 个
- 统一导入语句: {import_stats['unified_imports']} 个

### 4. 文档覆盖分析

**文档覆盖率**:
- 模块文档覆盖: {module_docs:.1%}
- 函数文档覆盖: {avg_function_coverage:.1%}
- 类文档覆盖: {avg_class_coverage:.1%}

"""

        # 文档覆盖详情
        low_doc_files = [d for d in doc_stats if d['function_coverage'] < 0.5]
        if low_doc_files:
            report += "**文档覆盖不足的文件**:\n"
            for doc in low_doc_files[:5]:
                report += f"- `{doc['file']}`: 函数{doc['function_coverage']:.1%}, 类{doc['class_coverage']:.1%}\n"

        report += f"""

### 5. 代码风格分析

**风格问题统计**:
- 总问题数: {total_style_issues} 个
- 高严重度: {sum(f['severity_high'] for f in style_stats)} 个
- 中严重度: {sum(f['severity_medium'] for f in style_stats)} 个
- 低严重度: {sum(f['severity_low'] for f in style_stats)} 个

"""

        # 风格问题最多的文件
        worst_style_files = sorted(style_stats, key=lambda x: x['issue_count'], reverse=True)[:3]
        if worst_style_files and worst_style_files[0]['issue_count'] > 0:
            report += "**风格问题最多的文件**:\n"
            for style in worst_style_files:
                report += f"- `{style['file']}`: {style['issue_count']} 个问题\n"

        report += """

## 🎯 改进建议

### 高优先级
1. **重构高复杂度文件**: 拆分复杂度过高的文件
2. **消除重复代码**: 提取公共代码块到工具函数
3. **修复导入问题**: 替换星号导入为显式导入

### 中优先级
4. **完善文档**: 为函数和类添加文档字符串
5. **统一代码风格**: 修复行长度和空白字符问题
6. **优化导入结构**: 减少相对导入，使用绝对导入

### 持续改进
7. **建立质量门禁**: 集成到CI/CD流程
8. **定期质量检查**: 建立自动化监控
9. **最佳实践分享**: 形成团队编码规范

---

**报告生成时间**: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return report

    def save_report(self, output_path: str = "QUALITY_CHECK_REPORT.md"):
        """保存报告到文件"""
        report = self.generate_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 质量报告已保存到: {output_path}")
        return output_path


def main():
    """主函数"""
    print("=== 🛠️ Phase 2.3: 自动化质量检查 ===\\n")

    # 检查配置管理模块
    checker = CodeQualityChecker('src/infrastructure/config')

    try:
        # 生成报告
        report_path = checker.save_report()

        print("\\n📋 质量检查完成！")
        print(f"   📄 详细报告: {report_path}")

        # 输出关键指标
        results = checker.results

        complexity_high = len([f for f in results['complexity'] if f.get('level') == 'high'])
        duplicates = len(results['duplicates'])
        star_imports = sum(f['star_imports'] for f in results['imports'])

        print("\\n📊 关键指标:")
        print(f"   ⚠️ 高复杂度文件: {complexity_high}")
        print(f"   🔄 重复代码块: {duplicates}")
        print(f"   🌟 星号导入: {star_imports}")

        return True

    except Exception as e:
        print(f"❌ 质量检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    print(f"\\n{'🎉 Phase 2.3 完成！' if success else '❌ Phase 2.3 失败！'}")

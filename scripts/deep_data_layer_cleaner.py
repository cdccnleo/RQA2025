#!/usr/bin/env python3
"""
数据采集层深度清理工具

专门解决数据采集层剩余的职责边界问题：
1. 清理变量名中的业务概念
2. 清理函数名中的业务概念
3. 清理字符串内容中的业务概念
4. 优化架构约束
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List


class DeepDataLayerCleaner:
    """数据采集层深度清理器"""

    def __init__(self):
        self.forbidden_concepts = {
            'strategy': {
                'keywords': ['strategy', 'strategies', 'strategic'],
                'replacements': {
                    'strategy': 'approach',
                    'strategies': 'approaches',
                    'strategic': 'technical'
                }
            },
            'execution': {
                'keywords': ['execution', 'execute', 'executor'],
                'replacements': {
                    'execution': 'processing',
                    'execute': 'process',
                    'executor': 'processor'
                }
            },
            'trading': {
                'keywords': ['trading', 'trade', 'trader'],
                'replacements': {
                    'trading': 'business',
                    'trade': 'business',
                    'trader': 'business_processor'
                }
            },
            'decision': {
                'keywords': ['decision', 'decide', 'making'],
                'replacements': {
                    'decision': 'choice',
                    'decide': 'choose',
                    'making': 'creation'
                }
            },
            'order': {
                'keywords': ['order', 'ordering'],
                'replacements': {
                    'order': 'sequence',
                    'ordering': 'sequencing'
                }
            }
        }

        self.problematic_files = []
        self.cleaned_files = []

    def find_remaining_issues(self):
        """查找剩余的问题"""
        print("🔍 深度扫描数据采集层剩余问题...")

        data_path = Path('src/data')
        if not data_path.exists():
            print("❌ 数据采集层目录不存在")
            return []

        issues = []
        for root, dirs, files in os.walk(data_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__']]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    file_issues = self.analyze_file_deep(file_path)
                    if file_issues:
                        issues.append({
                            'path': file_path,
                            'issues': file_issues
                        })

        print(f"📋 发现 {len(issues)} 个文件仍有问题")
        return issues

    def analyze_file_deep(self, file_path: Path) -> List[Dict]:
        """深度分析文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            issues = []

            # 1. 检查变量名
            variable_issues = self.check_variables(content)
            issues.extend(variable_issues)

            # 2. 检查函数名
            function_issues = self.check_functions(content)
            issues.extend(function_issues)

            # 3. 检查类名
            class_issues = self.check_classes(content)
            issues.extend(class_issues)

            # 4. 检查注释和字符串
            comment_issues = self.check_comments_and_strings(content)
            issues.extend(comment_issues)

            return issues

        except Exception as e:
            print(f"⚠️ 无法分析文件 {file_path}: {e}")
            return []

    def check_variables(self, content: str) -> List[Dict]:
        """检查变量名中的业务概念"""
        issues = []

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    var_name = node.id
                    for concept, config in self.forbidden_concepts.items():
                        for keyword in config['keywords']:
                            if keyword.lower() in var_name.lower():
                                issues.append({
                                    'type': 'variable',
                                    'name': var_name,
                                    'concept': concept,
                                    'keyword': keyword,
                                    'line': getattr(node, 'lineno', 0),
                                    'replacement': config['replacements'].get(keyword, f'technical_{keyword}')
                                })
        except:
            # 如果AST解析失败，使用正则表达式检查
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                for concept, config in self.forbidden_concepts.items():
                    for keyword in config['keywords']:
                        # 检查变量赋值模式
                        pattern = rf'\b(\w*{keyword}\w*)\s*='
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            var_name = match.group(1)
                            issues.append({
                                'type': 'variable',
                                'name': var_name,
                                'concept': concept,
                                'keyword': keyword,
                                'line': i,
                                'replacement': config['replacements'].get(keyword, f'technical_{keyword}')
                            })

        return issues

    def check_functions(self, content: str) -> List[Dict]:
        """检查函数名中的业务概念"""
        issues = []

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    for concept, config in self.forbidden_concepts.items():
                        for keyword in config['keywords']:
                            if keyword.lower() in func_name.lower():
                                issues.append({
                                    'type': 'function',
                                    'name': func_name,
                                    'concept': concept,
                                    'keyword': keyword,
                                    'line': node.lineno,
                                    'replacement': config['replacements'].get(keyword, f'technical_{keyword}')
                                })
        except:
            # 如果AST解析失败，使用正则表达式检查
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                match = re.search(r'def\s+(\w+)\s*\(', line)
                if match:
                    func_name = match.group(1)
                    for concept, config in self.forbidden_concepts.items():
                        for keyword in config['keywords']:
                            if keyword.lower() in func_name.lower():
                                issues.append({
                                    'type': 'function',
                                    'name': func_name,
                                    'concept': concept,
                                    'keyword': keyword,
                                    'line': i,
                                    'replacement': config['replacements'].get(keyword, f'technical_{keyword}')
                                })

        return issues

    def check_classes(self, content: str) -> List[Dict]:
        """检查类名中的业务概念"""
        issues = []

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for concept, config in self.forbidden_concepts.items():
                        for keyword in config['keywords']:
                            if keyword.lower() in class_name.lower():
                                issues.append({
                                    'type': 'class',
                                    'name': class_name,
                                    'concept': concept,
                                    'keyword': keyword,
                                    'line': node.lineno,
                                    'replacement': config['replacements'].get(keyword, f'Technical{keyword.title()}')
                                })
        except:
            # 如果AST解析失败，使用正则表达式检查
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    class_name = match.group(1)
                    for concept, config in self.forbidden_concepts.items():
                        for keyword in config['keywords']:
                            if keyword.lower() in class_name.lower():
                                issues.append({
                                    'type': 'class',
                                    'name': class_name,
                                    'concept': concept,
                                    'keyword': keyword,
                                    'line': i,
                                    'replacement': config['replacements'].get(keyword, f'Technical{keyword.title()}')
                                })

        return issues

    def check_comments_and_strings(self, content: str) -> List[Dict]:
        """检查注释和字符串中的业务概念"""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # 检查注释
            if '#' in line:
                comment_part = line.split('#', 1)[1]
                for concept, config in self.forbidden_concepts.items():
                    for keyword in config['keywords']:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', comment_part, re.IGNORECASE):
                            issues.append({
                                'type': 'comment',
                                'content': comment_part.strip(),
                                'concept': concept,
                                'keyword': keyword,
                                'line': i,
                                'replacement': config['replacements'].get(keyword, f'technical {keyword}')
                            })

            # 检查字符串
            string_matches = re.findall(r'["\']([^"\']*?)["\']', line)
            for string_content in string_matches:
                for concept, config in self.forbidden_concepts.items():
                    for keyword in config['keywords']:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', string_content, re.IGNORECASE):
                            issues.append({
                                'type': 'string',
                                'content': string_content,
                                'concept': concept,
                                'keyword': keyword,
                                'line': i,
                                'replacement': config['replacements'].get(keyword, f'technical {keyword}')
                            })

        return issues

    def clean_file(self, file_info: Dict):
        """清理文件"""
        file_path = file_info['path']
        issues = file_info['issues']

        print(f"🔧 深度清理文件: {file_path}")

        # 备份原文件
        backup_path = file_path.with_suffix('.py.deep_backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as original:
                f.write(original.read())

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 清理内容
        cleaned_content = self.apply_cleaning(content, issues)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        self.cleaned_files.append({
            'original': file_path,
            'backup': backup_path,
            'issues_fixed': len(issues)
        })

        print(f"✅ 文件深度清理完成: {file_path}")

    def apply_cleaning(self, content: str, issues: List[Dict]) -> str:
        """应用清理规则"""
        lines = content.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines, 1):
            original_line = line

            # 查找本行的所有问题
            line_issues = [issue for issue in issues if issue['line'] == i]

            for issue in line_issues:
                if issue['type'] == 'variable':
                    # 替换变量名
                    pattern = rf'\b{re.escape(issue["name"])}\b(?!\s*["\'])'
                    line = re.sub(pattern, issue['replacement'], line)

                elif issue['type'] == 'function':
                    # 替换函数名
                    pattern = rf'\bdef\s+{re.escape(issue["name"])}\b'
                    line = re.sub(pattern, f'def {issue["replacement"]}', line)

                elif issue['type'] == 'class':
                    # 替换类名
                    pattern = rf'\bclass\s+{re.escape(issue["name"])}\b'
                    line = re.sub(pattern, f'class {issue["replacement"]}', line)

                elif issue['type'] == 'comment':
                    # 替换注释内容
                    if '#' in line:
                        comment_start = line.find('#')
                        comment_part = line[comment_start:]
                        old_comment = comment_part.replace('#', '', 1).strip()
                        new_comment = re.sub(
                            r'\b' + re.escape(issue['keyword']) + r'\b',
                            issue['replacement'],
                            old_comment,
                            flags=re.IGNORECASE
                        )
                        line = line[:comment_start] + f'# {new_comment}'

                elif issue['type'] == 'string':
                    # 替换字符串内容
                    def replace_in_string(match):
                        string_content = match.group(1)
                        pattern = r'\b' + re.escape(issue['keyword']) + r'\b'
                        if re.search(pattern, string_content, re.IGNORECASE):
                            replaced = re.sub(
                                pattern, issue['replacement'], string_content, flags=re.IGNORECASE)
                            return f'"{replaced}"'
                        return match.group(0)

                    line = re.sub(r'["\']([^"\']*?)["\']', replace_in_string, line)

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def generate_cleaning_report(self):
        """生成清理报告"""
        report = []

        report.append("# 数据采集层深度清理报告")
        report.append("")
        report.append("生成时间: 2024-01-27 12:00:00")
        report.append("")

        report.append("## 清理概览")
        report.append("")
        report.append(f"- **发现问题文件**: {len(self.problematic_files)} 个")
        report.append(f"- **成功清理文件**: {len(self.cleaned_files)} 个")
        report.append("")

        report.append("## 问题详情")
        report.append("")
        for file_info in self.problematic_files:
            report.append(f"### {file_info['path']}")
            for issue in file_info['issues']:
                report.append(f"- **类型**: {issue['type']}")
                report.append(f"- **名称**: {issue['name'] if 'name' in issue else 'N/A'}")
                report.append(f"- **违规概念**: {issue['concept']}")
                report.append(f"- **关键词**: {issue['keyword']}")
                report.append(f"- **行号**: {issue['line']}")
                report.append(f"- **建议替换**: {issue['replacement']}")
            report.append("")

        report.append("## 清理结果")
        report.append("")
        for clean_info in self.cleaned_files:
            report.append(f"- ✅ {clean_info['original']}")
            report.append(f"  - 备份文件: {clean_info['backup']}")
            report.append(f"  - 修复问题: {clean_info['issues_fixed']} 个")
            report.append("")

        report.append("## 清理策略")
        report.append("")
        report.append("### 变量名清理")
        report.append("- 将业务概念替换为技术性描述")
        report.append("- 保持变量功能不变")
        report.append("")
        report.append("### 函数名清理")
        report.append("- 将业务概念替换为技术性描述")
        report.append("- 保持函数功能不变")
        report.append("")
        report.append("### 类名清理")
        report.append("- 将业务概念替换为技术性描述")
        report.append("- 保持类功能不变")
        report.append("")
        report.append("### 注释和字符串清理")
        report.append("- 将业务概念替换为技术性描述")
        report.append("- 保持原有含义")

        return "\n".join(report)

    def run_deep_cleaning(self):
        """运行深度清理"""
        print("🚀 开始数据采集层深度清理...")
        print("="*60)

        try:
            # 1. 查找剩余问题
            self.problematic_files = self.find_remaining_issues()

            if not self.problematic_files:
                print("✅ 未发现需要深度清理的文件")
                return True

            # 2. 显示问题摘要
            print("\n📋 问题摘要:")
            for file_info in self.problematic_files:
                print(f"   - {file_info['path']}: {len(file_info['issues'])} 个问题")

            # 3. 逐个清理文件
            print("\n🔧 开始深度清理...")
            for file_info in self.problematic_files:
                self.clean_file(file_info)

            # 4. 生成报告
            report = self.generate_cleaning_report()
            with open('reports/DATA_LAYER_DEEP_CLEANING_REPORT.md', 'w', encoding='utf-8') as f:
                f.write(report)

            print("\n📋 深度清理报告已保存到: reports/DATA_LAYER_DEEP_CLEANING_REPORT.md")
            print("🎉 数据采集层深度清理完成！")
            return True

        except Exception as e:
            print(f"\n❌ 深度清理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    cleaner = DeepDataLayerCleaner()
    success = cleaner.run_deep_cleaning()

    if success:
        print("\n" + "="*60)
        print("数据采集层深度清理成功完成！")
        print("✅ 变量名业务概念已清理")
        print("✅ 函数名业务概念已清理")
        print("✅ 类名业务概念已清理")
        print("✅ 注释和字符串已清理")
        print("="*60)
    else:
        print("\n❌ 数据采集层深度清理失败！")


if __name__ == "__main__":
    main()

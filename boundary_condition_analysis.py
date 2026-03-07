#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界条件分析和识别脚本
用于检测系统中的潜在边界条件问题
"""

import sys
import os
import re
import ast
from typing import Dict, List
from collections import defaultdict


class BoundaryConditionAnalyzer:
    """边界条件分析器"""

    def __init__(self):
        self.potential_issues = defaultdict(list)
        self.risk_patterns = self._define_risk_patterns()

    def _define_risk_patterns(self) -> Dict[str, List[str]]:
        """定义边界条件风险模式"""
        return {
            'input_validation': [
                r'def \w+\([^)]*\):',  # 函数定义，检查参数验证
                r'input\(',  # 直接用户输入
                r'request\.(GET|POST|args)',  # Web请求参数
            ],
            'numeric_operations': [
                r'/\s*\w+',  # 除法操作
                r'%\s*\w+',  # 取模操作
                r'\*\*\s*\w+',  # 幂运算
                r'math\.(sqrt|log|exp)',  # 数学函数
            ],
            'resource_access': [
                r'open\(',  # 文件操作
                r'connect\(',  # 网络连接
                r'execute\(',  # 数据库操作
                r'redis\.',  # Redis操作
            ],
            'configuration': [
                r'config\[',  # 配置访问
                r'os\.environ\[',  # 环境变量
                r'getenv\(',  # 环境变量获取
            ],
            'external_services': [
                r'requests\.(get|post|put|delete)',  # HTTP请求
                r'kafka\.',  # Kafka操作
                r'socket\.',  # 网络套接字
            ]
        }

    def analyze_file(self, file_path: str):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST以获取更多结构信息
            tree = ast.parse(content, filename=file_path)

            # 分析不同类型的边界条件风险
            self._analyze_input_validation(content, file_path)
            self._analyze_numeric_operations(content, file_path)
            self._analyze_resource_access(content, file_path)
            self._analyze_configuration(content, file_path)
            self._analyze_external_services(content, file_path)
            self._analyze_ast_patterns(tree, file_path)

        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {e}")

    def _analyze_input_validation(self, content: str, file_path: str):
        """分析输入验证边界条件"""
        issues = []

        # 检查是否有参数验证
        func_defs = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
        for func_name in func_defs:
            # 检查函数是否有基本的输入验证
            func_content = self._get_function_content(content, func_name)
            if not self._has_input_validation(func_content):
                issues.append({
                    'type': 'missing_input_validation',
                    'function': func_name,
                    'description': f'函数 {func_name} 缺少输入参数验证'
                })

        # 检查直接用户输入
        if re.search(r'input\s*\(', content):
            issues.append({
                'type': 'direct_user_input',
                'description': '发现直接用户输入，可能需要验证'
            })

        # 检查Web请求参数访问
        if re.search(r'request\.(GET|POST|args)', content):
            if not re.search(r'(validate|sanitize|clean)', content):
                issues.append({
                    'type': 'unvalidated_web_input',
                    'description': 'Web请求参数缺少验证'
                })

        self.potential_issues['input_validation'].extend(issues)

    def _analyze_numeric_operations(self, content: str, file_path: str):
        """分析数值计算边界条件"""
        issues = []

        # 检查除法操作
        divisions = re.findall(r'(\w+)\s*/\s*(\w+)', content)
        for dividend, divisor in divisions:
            if not re.search(r'if.*' + re.escape(divisor) + r'.*[!=\s]*0', content):
                issues.append({
                    'type': 'division_by_zero',
                    'operation': f'{dividend}/{divisor}',
                    'description': f'除法操作 {dividend}/{divisor} 可能出现除零错误'
                })

        # 检查数组访问
        array_accesses = re.findall(r'(\w+)\s*\[\s*(\w+)\s*\]', content)
        for array, index in array_accesses:
            if not re.search(r'len\s*\(\s*' + re.escape(array) + r'\s*\)', content):
                issues.append({
                    'type': 'array_bounds',
                    'operation': f'{array}[{index}]',
                    'description': f'数组访问 {array}[{index}] 可能出现越界错误'
                })

        # 检查数学函数
        math_funcs = re.findall(r'math\.(sqrt|log|exp|pow)\s*\(\s*([^)]+)\s*\)', content)
        for func, args in math_funcs:
            if func in ['sqrt', 'log'] and not re.search(r'if.*' + re.escape(args) + r'.*>=\s*0', content):
                issues.append({
                    'type': 'math_domain_error',
                    'operation': f'math.{func}({args})',
                    'description': f'数学函数 {func} 可能出现域错误'
                })

        self.potential_issues['numeric_operations'].extend(issues)

    def _analyze_resource_access(self, content: str, file_path: str):
        """分析资源访问边界条件"""
        issues = []

        # 检查文件操作
        file_ops = re.findall(r'with\s+open\s*\(\s*([^,)]+)', content)
        for file_path_expr in file_ops:
            if not re.search(r'try:', content) or not re.search(r'except', content):
                issues.append({
                    'type': 'file_operation',
                    'operation': f'open({file_path_expr})',
                    'description': '文件操作缺少异常处理'
                })

        # 检查网络连接
        if re.search(r'connect\s*\(', content) and not re.search(r'timeout', content):
            issues.append({
                'type': 'network_timeout',
                'description': '网络连接缺少超时设置'
            })

        self.potential_issues['resource_access'].extend(issues)

    def _analyze_configuration(self, content: str, file_path: str):
        """分析配置访问边界条件"""
        issues = []

        # 检查配置访问
        config_accesses = re.findall(r'config\s*\[\s*[\'\"]([^\'\"]+)[\'\"]\s*\]', content)
        for key in config_accesses:
            if not re.search(r'get\s*\(\s*[\'\"]' + re.escape(key) + r'[\'\"]', content):
                issues.append({
                    'type': 'config_key_error',
                    'key': key,
                    'description': f'配置键 {key} 访问可能出现KeyError'
                })

        # 检查环境变量
        env_vars = re.findall(r'os\.environ\s*\[\s*[\'\"]([^\'\"]+)[\'\"]\s*\]', content)
        for var in env_vars:
            if not re.search(r'get\s*\(\s*[\'\"]' + re.escape(var) + r'[\'\"]', content):
                issues.append({
                    'type': 'env_var_error',
                    'variable': var,
                    'description': f'环境变量 {var} 访问可能出现KeyError'
                })

        self.potential_issues['configuration'].extend(issues)

    def _analyze_external_services(self, content: str, file_path: str):
        """分析外部服务调用边界条件"""
        issues = []

        # 检查HTTP请求
        http_requests = re.findall(r'requests\.(get|post|put|delete)\s*\(', content)
        for method in http_requests:
            if not re.search(r'timeout\s*=', content):
                issues.append({
                    'type': 'http_timeout',
                    'method': method.upper(),
                    'description': f'HTTP {method.upper()} 请求缺少超时设置'
                })

        # 检查数据库操作
        if re.search(r'execute\s*\(', content) and not re.search(r'try:', content):
            issues.append({
                'type': 'db_operation',
                'description': '数据库操作缺少异常处理'
            })

        self.potential_issues['external_services'].extend(issues)

    def _analyze_ast_patterns(self, tree: ast.AST, file_path: str):
        """通过AST分析代码结构"""
        issues = []

        for node in ast.walk(tree):
            # 检查函数参数
            if isinstance(node, ast.FunctionDef):
                # 检查是否有默认参数为None的参数
                for arg in node.args.args:
                    if arg.arg in ['config', 'settings', 'params']:
                        issues.append({
                            'type': 'potential_none_param',
                            'function': node.name,
                            'parameter': arg.arg,
                            'description': f'函数 {node.name} 的参数 {arg.arg} 可能为None'
                        })

            # 检查循环中的break/continue使用
            elif isinstance(node, (ast.For, ast.While)):
                has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                has_continue = any(isinstance(n, ast.Continue) for n in ast.walk(node))
                if has_break or has_continue:
                    issues.append({
                        'type': 'loop_control',
                        'description': '循环中使用了break/continue，检查边界条件'
                    })

        self.potential_issues['ast_patterns'].extend(issues)

    def _get_function_content(self, content: str, func_name: str) -> str:
        """获取函数内容"""
        # 简单的函数内容提取（实际实现可能需要更复杂的解析）
        lines = content.split('\n')
        in_function = False
        indent_level = 0
        function_content = []

        for line in lines:
            if re.match(rf'def\s+{func_name}\s*\(', line):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                function_content.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                if current_indent > indent_level or (line.strip() == '' and len(function_content) > 0):
                    function_content.append(line)
                else:
                    break

        return '\n'.join(function_content)

    def _has_input_validation(self, func_content: str) -> bool:
        """检查函数是否包含输入验证"""
        validation_patterns = [
            r'if.*is None',
            r'if.*not',
            r'assert',
            r'raise.*ValueError',
            r'raise.*TypeError',
            r'isinstance',
            r'len\s*\('
        ]

        for pattern in validation_patterns:
            if re.search(pattern, func_content):
                return True
        return False

    def generate_report(self):
        """生成分析报告"""
        print("\n📋 边界条件分析报告")
        print("=" * 50)

        total_issues = sum(len(issues) for issues in self.potential_issues.values())

        print(f"📊 发现潜在边界条件问题: {total_issues} 个")
        print()

        for category, issues in self.potential_issues.items():
            if issues:
                print(f"🔍 {category.replace('_', ' ').title()}: {len(issues)} 个问题")
                for i, issue in enumerate(issues[:5]):  # 只显示前5个
                    print(f"   {i+1}. {issue['description']}")
                    if 'function' in issue:
                        print(f"      函数: {issue['function']}")
                    if 'operation' in issue:
                        print(f"      操作: {issue['operation']}")
                if len(issues) > 5:
                    print(f"   ... 还有 {len(issues) - 5} 个类似问题")
                print()

        print("💡 改进建议:")
        print("   • 为所有用户输入添加验证")
        print("   • 检查数值计算的边界条件")
        print("   • 为外部服务调用添加超时和重试")
        print("   • 使用安全的配置访问方法")
        print("   • 添加适当的异常处理")


def analyze_project(root_path: str):
    """分析整个项目"""
    analyzer = BoundaryConditionAnalyzer()

    # 遍历项目文件
    for root, dirs, files in os.walk(root_path):
        # 跳过一些不需要分析的目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'build', 'dist']]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"分析文件: {file_path}")
                analyzer.analyze_file(file_path)

    # 生成报告
    analyzer.generate_report()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "src"

    print("🧠 RQA2025 边界条件分析工具")
    print("=" * 40)
    analyze_project(project_path)

#!/usr/bin/env python3
"""
RQA2025 AST代码分析器
使用AST深度分析代码结构和数据流，实现跨模块调用关系分析
"""

import os
import sys
import ast
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
import gc
import psutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"【内存监控】{stage} 当前内存占用: {mem_mb:.2f} MB")


class ASTCodeAnalyzer:
    """AST代码分析器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.import_graph = nx.DiGraph()
        self.function_calls = defaultdict(set)
        self.class_inheritance = defaultdict(set)
        self.data_flow = defaultdict(dict)
        self.module_dependencies = defaultdict(set)
        self.analysis_results = {}

    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目"""
        logger.info("🔍 开始AST项目分析...")
        # 只分析src和scripts目录下的Python文件，跳过venv、tests、site-packages等
        include_dirs = [self.project_root / 'src', self.project_root / 'scripts']
        python_files = []
        for include_dir in include_dirs:
            if include_dir.exists():
                for file_path in include_dir.rglob("*.py"):
                    # 跳过无关目录
                    if any(skip in str(file_path) for skip in ['venv', 'site-packages', 'tests', '__pycache__']):
                        continue
                    python_files.append(file_path)
        logger.info(f"发现 {len(python_files)} 个业务相关Python文件")
        log_memory_usage("收集文件后")
        for idx, file_path in enumerate(python_files, 1):
            try:
                self.analyze_file(file_path)
            except Exception as e:
                logger.warning(f"分析文件失败 {file_path}: {e}")
            if idx % 100 == 0:
                log_memory_usage(f"已分析{idx}个文件")
                gc.collect()
        log_memory_usage("文件分析结束")
        gc.collect()
        # 构建调用关系图
        self.build_call_graph()
        log_memory_usage("构建调用关系图后")
        gc.collect()
        # 分析数据流
        self.analyze_data_flow()
        log_memory_usage("分析数据流后")
        gc.collect()
        # 生成分析报告
        self.generate_analysis_report()
        log_memory_usage("生成分析报告后")
        gc.collect()
        return self.analysis_results

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # 获取模块路径
            module_path = str(file_path.relative_to(self.project_root))

            # 分析导入
            imports = self.extract_imports(tree, module_path)

            # 分析函数定义
            functions = self.extract_functions(tree, module_path)

            # 分析类定义
            classes = self.extract_classes(tree, module_path)

            # 分析函数调用
            calls = self.extract_function_calls(tree, module_path)

            # 分析数据流
            data_flow = self.analyze_file_data_flow(tree, module_path)

            file_analysis = {
                'module_path': module_path,
                'imports': imports,
                'functions': functions,
                'classes': classes,
                'calls': calls,
                'data_flow': data_flow,
                'complexity': self.calculate_complexity(tree),
                'lines_of_code': len(content.split('\n'))
            }

            self.analysis_results[module_path] = file_analysis

            return file_analysis

        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {e}")
            return {}

    def extract_imports(self, tree: ast.AST, module_path: str) -> List[Dict[str, Any]]:
        """提取导入信息"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'as_name': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'as_name': alias.asname,
                        'line': node.lineno
                    })

        return imports

    def extract_functions(self, tree: ast.AST, module_path: str) -> List[Dict[str, Any]]:
        """提取函数定义"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 分析函数参数
                args = []
                for arg in node.args.args:
                    args.append({
                        'name': arg.arg,
                        'annotation': self.get_annotation_name(arg.annotation)
                    })

                # 分析函数装饰器
                decorators = []
                for decorator in node.decorator_list:
                    decorators.append(self.get_decorator_name(decorator))

                # 分析函数复杂度
                complexity = self.calculate_function_complexity(node)

                functions.append({
                    'name': node.name,
                    'args': args,
                    'decorators': decorators,
                    'line': node.lineno,
                    'complexity': complexity,
                    'docstring': ast.get_docstring(node)
                })

        return functions

    def extract_classes(self, tree: ast.AST, module_path: str) -> List[Dict[str, Any]]:
        """提取类定义"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 分析基类
                bases = []
                for base in node.bases:
                    bases.append(self.get_base_class_name(base))

                # 分析类方法
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'line': item.lineno,
                            'complexity': self.calculate_function_complexity(item)
                        })

                # 分析类属性
                attributes = []
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append({
                                    'name': target.id,
                                    'line': item.lineno
                                })

                classes.append({
                    'name': node.name,
                    'bases': bases,
                    'methods': methods,
                    'attributes': attributes,
                    'line': node.lineno,
                    'docstring': ast.get_docstring(node)
                })

        return classes

    def extract_function_calls(self, tree: ast.AST, module_path: str) -> List[Dict[str, Any]]:
        """提取函数调用"""
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_info = self.analyze_function_call(node)
                if call_info:
                    call_info['line'] = node.lineno
                    calls.append(call_info)

        return calls

    def analyze_function_call(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """分析函数调用"""
        if isinstance(node.func, ast.Name):
            return {
                'type': 'function_call',
                'name': node.func.id,
                'args_count': len(node.args),
                'keywords_count': len(node.keywords)
            }
        elif isinstance(node.func, ast.Attribute):
            return {
                'type': 'method_call',
                'object': self.get_attribute_object(node.func),
                'method': node.func.attr,
                'args_count': len(node.args),
                'keywords_count': len(node.keywords)
            }

        return None

    def analyze_file_data_flow(self, tree: ast.AST, module_path: str) -> Dict[str, Any]:
        """分析文件数据流"""
        data_flow = {
            'variables': {},
            'assignments': [],
            'returns': [],
            'exceptions': []
        }

        for node in ast.walk(tree):
            # 分析变量赋值
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        data_flow['assignments'].append({
                            'variable': target.id,
                            'line': node.lineno,
                            'value_type': self.get_value_type(node.value)
                        })

            # 分析返回语句
            elif isinstance(node, ast.Return):
                data_flow['returns'].append({
                    'line': node.lineno,
                    'value_type': self.get_value_type(node.value) if node.value else None
                })

            # 分析异常处理
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    data_flow['exceptions'].append({
                        'line': handler.lineno,
                        'exception_type': self.get_exception_type(handler.type) if handler.type else 'Exception'
                    })

        return data_flow

    def build_call_graph(self):
        """构建调用关系图"""
        logger.info("🔗 构建调用关系图...")

        for module_path, analysis in self.analysis_results.items():
            # 添加模块节点
            self.import_graph.add_node(module_path)

            # 添加导入关系
            for import_info in analysis.get('imports', []):
                if import_info['type'] == 'import':
                    target_module = import_info['module']
                    self.import_graph.add_edge(module_path, target_module)
                elif import_info['type'] == 'from_import':
                    target_module = import_info['module']
                    if target_module:
                        self.import_graph.add_edge(module_path, target_module)

            # 添加函数调用关系
            for call in analysis.get('calls', []):
                if call['type'] == 'function_call':
                    # 查找函数定义
                    for other_module, other_analysis in self.analysis_results.items():
                        for func in other_analysis.get('functions', []):
                            if func['name'] == call['name']:
                                self.import_graph.add_edge(module_path, other_module)
                                break

    def analyze_data_flow(self):
        """分析数据流"""
        logger.info("📊 分析数据流...")

        for module_path, analysis in self.analysis_results.items():
            data_flow = analysis.get('data_flow', {})

            # 分析变量使用
            variables = {}
            for assignment in data_flow.get('assignments', []):
                var_name = assignment['variable']
                if var_name not in variables:
                    variables[var_name] = []
                variables[var_name].append({
                    'type': 'assignment',
                    'line': assignment['line'],
                    'value_type': assignment['value_type']
                })

            # 分析函数参数和返回值
            for func in analysis.get('functions', []):
                func_name = func['name']
                self.data_flow[module_path][func_name] = {
                    'parameters': func['args'],
                    'returns': [],
                    'complexity': func['complexity']
                }

    def calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """计算代码复杂度"""
        complexity = {
            'cyclomatic': 0,
            'cognitive': 0,
            'nesting_depth': 0
        }

        for node in ast.walk(tree):
            # 计算圈复杂度
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity['cyclomatic'] += 1

            # 计算认知复杂度
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity['cognitive'] += 1

            # 计算嵌套深度
            depth = self.calculate_nesting_depth(node)
            complexity['nesting_depth'] = max(complexity['nesting_depth'], depth)

        return complexity

    def calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1

        return complexity

    def calculate_nesting_depth(self, node: ast.AST) -> int:
        """计算嵌套深度"""
        depth = 0
        current = node

        while hasattr(current, 'parent'):
            if isinstance(current.parent, (ast.If, ast.While, ast.For, ast.Try)):
                depth += 1
            current = current.parent

        return depth

    def get_annotation_name(self, annotation) -> Optional[str]:
        """获取类型注解名称"""
        if annotation is None:
            return None
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return f"{self.get_attribute_object(annotation)}.{annotation.attr}"
        return str(annotation)

    def get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self.get_attribute_object(decorator)}.{decorator.attr}"
        return str(decorator)

    def get_base_class_name(self, base) -> str:
        """获取基类名称"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self.get_attribute_object(base)}.{base.attr}"
        return str(base)

    def get_attribute_object(self, attr_node: ast.Attribute) -> str:
        """获取属性对象名称"""
        if isinstance(attr_node.value, ast.Name):
            return attr_node.value.id
        elif isinstance(attr_node.value, ast.Attribute):
            return f"{self.get_attribute_object(attr_node.value)}.{attr_node.value.attr}"
        return str(attr_node.value)

    def get_value_type(self, value) -> str:
        """获取值类型"""
        if value is None:
            return 'None'
        elif isinstance(value, ast.Name):
            return 'variable'
        elif isinstance(value, ast.Constant):
            return type(value.value).__name__
        elif isinstance(value, ast.Call):
            return 'function_call'
        elif isinstance(value, ast.List):
            return 'list'
        elif isinstance(value, ast.Dict):
            return 'dict'
        elif isinstance(value, ast.Tuple):
            return 'tuple'
        return 'unknown'

    def get_exception_type(self, exception) -> str:
        """获取异常类型"""
        if exception is None:
            return 'Exception'
        elif isinstance(exception, ast.Name):
            return exception.id
        elif isinstance(exception, ast.Attribute):
            return f"{self.get_attribute_object(exception)}.{exception.attr}"
        return str(exception)

    def find_critical_modules(self) -> List[str]:
        """找出关键模块"""
        logger.info("🎯 识别关键模块...")

        critical_modules = []

        # 计算模块重要性指标
        for module_path, analysis in self.analysis_results.items():
            importance_score = 0

            # 基于函数数量
            functions_count = len(analysis.get('functions', []))
            importance_score += functions_count * 0.1

            # 基于类数量
            classes_count = len(analysis.get('classes', []))
            importance_score += classes_count * 0.2

            # 基于复杂度
            complexity = analysis.get('complexity', {})
            importance_score += complexity.get('cyclomatic', 0) * 0.05

            # 基于被依赖程度
            in_degree = self.import_graph.in_degree(module_path)
            importance_score += in_degree * 0.3

            # 基于代码行数
            lines = analysis.get('lines_of_code', 0)
            importance_score += min(lines / 100, 1.0) * 0.1

            if importance_score > 1.0:
                critical_modules.append({
                    'module': module_path,
                    'score': importance_score,
                    'functions': functions_count,
                    'classes': classes_count,
                    'complexity': complexity.get('cyclomatic', 0),
                    'dependencies': in_degree,
                    'lines': lines
                })

        # 按重要性排序
        critical_modules.sort(key=lambda x: x['score'], reverse=True)

        return critical_modules

    def analyze_dependencies(self) -> Dict[str, Any]:
        """分析模块依赖关系"""
        log_memory_usage("分析模块依赖关系前")
        logger.info("🔗 分析模块依赖关系...")
        dependencies = {
            'imports': defaultdict(set),
            'exports': defaultdict(set),
            'circular_deps': [],
            'critical_paths': []
        }
        # 输出依赖图规模
        num_nodes = self.import_graph.number_of_nodes() if hasattr(self, 'import_graph') else 0
        num_edges = self.import_graph.number_of_edges() if hasattr(self, 'import_graph') else 0
        logger.info(f"【依赖图规模】节点数: {num_nodes}, 边数: {num_edges}")
        max_graph_nodes = 500
        max_graph_edges = 2000
        if num_nodes > max_graph_nodes or num_edges > max_graph_edges:
            logger.warning(f"依赖图过大({num_nodes} nodes, {num_edges} edges)，跳过循环依赖和关键路径分析")
            log_memory_usage("分析模块依赖关系后(跳过大图)")
            gc.collect()
            return dependencies
        # 分析导入关系
        for module_path, analysis in self.analysis_results.items():
            for import_info in analysis.get('imports', []):
                if import_info['type'] == 'import':
                    target_module = import_info['module']
                    dependencies['imports'][module_path].add(target_module)
                elif import_info['type'] == 'from_import':
                    target_module = import_info['module']
                    if target_module:
                        dependencies['imports'][module_path].add(target_module)
        # 检测循环依赖（加数量限制）
        try:
            max_cycles = 50
            cycles = list(nx.simple_cycles(self.import_graph))
            if len(cycles) > max_cycles:
                logger.warning(f"检测到循环依赖数量过多({len(cycles)}), 仅保留前{max_cycles}条")
                cycles = cycles[:max_cycles]
            dependencies['circular_deps'] = cycles
        except Exception as e:
            logger.error(f"循环依赖检测异常: {e}")
        # 找出关键路径（加长度限制）
        try:
            max_path_len = 1000
            sorted_modules = list(nx.topological_sort(self.import_graph))
            if len(sorted_modules) > max_path_len:
                logger.warning(f"关键路径长度过长({len(sorted_modules)}), 仅保留前{max_path_len}个模块")
                sorted_modules = sorted_modules[:max_path_len]
            dependencies['critical_paths'] = sorted_modules
        except Exception as e:
            logger.error(f"关键路径分析异常: {e}")
        log_memory_usage("分析模块依赖关系后")
        gc.collect()
        return dependencies

    def generate_analysis_report(self) -> str:
        """生成分析报告"""
        report_file = "reports/testing/ast_analysis_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 统计信息
        total_files = len(self.analysis_results)
        total_functions = sum(len(analysis.get('functions', []))
                              for analysis in self.analysis_results.values())
        total_classes = sum(len(analysis.get('classes', []))
                            for analysis in self.analysis_results.values())
        total_lines = sum(analysis.get('lines_of_code', 0)
                          for analysis in self.analysis_results.values())

        # 找出关键模块
        critical_modules = self.find_critical_modules()

        # 分析依赖关系
        dependencies = self.analyze_dependencies()

        report_content = f"""# RQA2025 AST代码分析报告

## 📊 分析摘要

**分析时间**: {current_time}
**总文件数**: {total_files}
**总函数数**: {total_functions}
**总类数**: {total_classes}
**总代码行数**: {total_lines}

## 🎯 关键模块

| 模块 | 重要性评分 | 函数数 | 类数 | 复杂度 | 依赖数 | 代码行数 |
|------|------------|--------|------|--------|--------|----------|
"""

        for module_info in critical_modules[:10]:  # 显示前10个关键模块
            report_content += f"| {module_info['module']} | {module_info['score']:.2f} | {module_info['functions']} | {module_info['classes']} | {module_info['complexity']} | {module_info['dependencies']} | {module_info['lines']} |\n"

        report_content += f"""
## 🔗 依赖关系

### 循环依赖
"""

        if dependencies['circular_deps']:
            for cycle in dependencies['circular_deps']:
                report_content += f"- {' -> '.join(cycle)} -> {cycle[0]}\n"
        else:
            report_content += "- 未发现循环依赖\n"

        report_content += f"""
### 关键路径
"""

        for i, module in enumerate(dependencies['critical_paths'][:10]):
            report_content += f"{i+1}. {module}\n"

        report_content += f"""
## 📈 复杂度分析

### 平均复杂度指标
- 圈复杂度: {sum(analysis.get('complexity', {}).get('cyclomatic', 0) for analysis in self.analysis_results.values()) / max(total_files, 1):.2f}
- 认知复杂度: {sum(analysis.get('complexity', {}).get('cognitive', 0) for analysis in self.analysis_results.values()) / max(total_files, 1):.2f}
- 最大嵌套深度: {max(analysis.get('complexity', {}).get('nesting_depth', 0) for analysis in self.analysis_results.values())}

## 🎯 测试建议

基于AST分析结果，建议优先测试以下模块：

"""

        for module_info in critical_modules[:5]:
            report_content += f"1. **{module_info['module']}** (评分: {module_info['score']:.2f})\n"
            report_content += f"   - 函数数: {module_info['functions']}\n"
            report_content += f"   - 复杂度: {module_info['complexity']}\n"
            report_content += f"   - 建议测试覆盖率: 95%\n\n"

        report_content += f"""
## 📋 详细分析

### 模块统计
"""

        for module_path, analysis in self.analysis_results.items():
            functions = len(analysis.get('functions', []))
            classes = len(analysis.get('classes', []))
            complexity = analysis.get('complexity', {})

            report_content += f"- **{module_path}**: {functions} 函数, {classes} 类, 复杂度 {complexity.get('cyclomatic', 0)}\n"

        report_content += f"""
---
**报告版本**: v1.0
**分析时间**: {current_time}
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📄 AST分析报告已生成: {report_file}")
        return report_file

    def get_test_priorities(self) -> List[Dict[str, Any]]:
        """获取测试优先级建议"""
        critical_modules = self.find_critical_modules()

        priorities = []
        for module_info in critical_modules:
            priorities.append({
                'module': module_info['module'],
                'priority': 'high' if module_info['score'] > 2.0 else 'medium' if module_info['score'] > 1.0 else 'low',
                'score': module_info['score'],
                'suggested_coverage': min(95, 70 + module_info['score'] * 10),
                'complexity': module_info['complexity'],
                'functions': module_info['functions'],
                'classes': module_info['classes']
            })

        return priorities


def main():
    """主函数"""
    analyzer = ASTCodeAnalyzer(project_root)
    results = analyzer.analyze_project()

    print("✅ AST代码分析完成")
    print(f"📄 分析报告: {analyzer.generate_analysis_report()}")

    # 显示测试优先级
    priorities = analyzer.get_test_priorities()
    print("\n🎯 测试优先级建议:")
    for priority in priorities[:5]:
        print(
            f"  {priority['module']}: {priority['priority']} (评分: {priority['score']:.2f}, 建议覆盖率: {priority['suggested_coverage']}%)")


if __name__ == "__main__":
    main()

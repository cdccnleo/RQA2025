#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 API文档自动生成器

自动扫描代码库生成API文档
支持多种输出格式和文档模板
"""

import os
import sys
import inspect
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import ast
import re


@dataclass
class APIEndpoint:
    """API端点信息"""
    name: str
    module: str
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    docstring: str = ""
    signature: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    visibility: str = "public"  # public, private, protected


@dataclass
class APIModule:
    """API模块信息"""
    name: str
    path: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    docstring: str = ""


class APIDocumentationGenerator:
    """
    API文档生成器

    自动扫描Python代码库，提取API信息并生成文档
    """

    def __init__(self, source_dir: str, output_dir: str = "docs/api"):
        """
        初始化文档生成器

        Args:
            source_dir: 源代码目录
            output_dir: 输出目录
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.modules: Dict[str, APIModule] = {}

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模块过滤器
        self.module_filters = [
            lambda path: not path.name.startswith('test_'),
            lambda path: not path.name.startswith('_'),
            lambda path: path.suffix == '.py'
        ]

    def scan_modules(self) -> Dict[str, APIModule]:
        """
        扫描模块

        Returns:
            模块字典
        """
        print("🔍 开始扫描模块...")

        for py_file in self.source_dir.rglob('*.py'):
            if self._should_process_file(py_file):
                module_name = self._get_module_name(py_file)
                module = self._analyze_module(py_file)
                if module:
                    self.modules[module_name] = module
                    print(f"  ✅ 分析完成: {module_name}")

        print(f"📊 扫描完成，共发现 {len(self.modules)} 个模块")
        return self.modules

    def _should_process_file(self, file_path: Path) -> bool:
        """检查是否应该处理文件"""
        return all(filter_func(file_path) for filter_func in self.module_filters)

    def _get_module_name(self, file_path: Path) -> str:
        """获取模块名称"""
        relative_path = file_path.relative_to(self.source_dir)
        return str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

    def _analyze_module(self, file_path: Path) -> Optional[APIModule]:
        """分析模块"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content, filename=str(file_path))

            module = APIModule(
                name=self._get_module_name(file_path),
                path=str(file_path)
            )

            # 提取模块文档字符串
            module.docstring = ast.get_docstring(tree) or ""

            # 分析类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class(node, module, content)
                elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                    self._analyze_function(node, module, content, None)
                elif isinstance(node, ast.AsyncFunctionDef):
                    self._analyze_async_function(node, module, content, None)

            return module if module.endpoints else None

        except Exception as e:
            print(f"  ❌ 分析失败 {file_path}: {e}")
            return None

    def _analyze_class(self, node: ast.ClassDef, module: APIModule, content: str):
        """分析类"""
        class_name = node.name
        class_docstring = ast.get_docstring(node) or ""

        # 检查是否是API类
        if self._is_api_class(class_name, class_docstring):
            # 分析类方法
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not isinstance(item, ast.AsyncFunctionDef):
                    self._analyze_function(item, module, content, class_name)
                elif isinstance(item, ast.AsyncFunctionDef):
                    self._analyze_async_function(item, module, content, class_name)

    def _analyze_function(self, node: ast.FunctionDef, module: APIModule,
                         content: str, class_name: Optional[str]):
        """分析函数"""
        func_name = node.name

        # 检查是否是API函数
        if self._is_api_function(func_name, class_name):
            endpoint = self._create_endpoint(node, module.name, content, class_name, func_name, False)
            if endpoint:
                module.endpoints.append(endpoint)

    def _analyze_async_function(self, node: ast.AsyncFunctionDef, module: APIModule,
                               content: str, class_name: Optional[str]):
        """分析异步函数"""
        func_name = node.name

        # 检查是否是API函数
        if self._is_api_function(func_name, class_name):
            endpoint = self._create_endpoint(node, module.name, content, class_name, func_name, True)
            if endpoint:
                module.endpoints.append(endpoint)

    def _create_endpoint(self, node: ast.FunctionDef, module_name: str, content: str,
                        class_name: Optional[str], func_name: str, is_async: bool) -> Optional[APIEndpoint]:
        """创建端点"""
        try:
            # 获取函数签名
            signature = self._get_function_signature(node, content)

            # 获取文档字符串
            docstring = ast.get_docstring(node) or ""

            # 解析参数
            parameters = self._parse_parameters(node)

            # 获取返回类型
            return_type = self._get_return_type(node)

            # 获取装饰器
            decorators = [self._get_decorator_name(d) for d in node.decorator_list]

            # 确定可见性
            visibility = self._get_visibility(func_name)

            return APIEndpoint(
                name=f"{class_name}.{func_name}" if class_name else func_name,
                module=module_name,
                class_name=class_name,
                method_name=func_name,
                docstring=docstring,
                signature=signature,
                parameters=parameters,
                return_type=return_type,
                decorators=decorators,
                is_async=is_async,
                visibility=visibility
            )

        except Exception as e:
            print(f"  ⚠️  创建端点失败 {func_name}: {e}")
            return None

    def _is_api_class(self, class_name: str, docstring: str) -> bool:
        """检查是否是API类"""
        # 可以根据类名或文档字符串判断
        api_indicators = ['manager', 'service', 'handler', 'controller', 'client']
        return any(indicator in class_name.lower() for indicator in api_indicators) or \
               'api' in docstring.lower()

    def _is_api_function(self, func_name: str, class_name: Optional[str]) -> bool:
        """检查是否是API函数"""
        # 排除私有方法
        if func_name.startswith('_'):
            return False

        # 包含公共API方法
        api_methods = ['get_', 'post_', 'put_', 'delete_', 'create_', 'update_',
                      'delete_', 'find_', 'search_', 'list_', 'execute_', 'process_',
                      'authenticate_', 'authorize_', 'validate_', 'check_']

        return any(func_name.startswith(method) for method in api_methods) or \
               len(func_name) > 3  # 简单的启发式方法

    def _get_function_signature(self, node: ast.FunctionDef, content: str) -> str:
        """获取函数签名"""
        try:
            # 从源代码中提取签名
            lines = content.split('\n')
            start_line = node.lineno - 1
            signature_lines = []

            # 查找函数定义行
            for i in range(start_line, len(lines)):
                line = lines[i].strip()
                signature_lines.append(line)
                if line.endswith(':'):
                    break

            return ' '.join(signature_lines)

        except:
            # 回退到简单的签名
            params = [arg.arg for arg in node.args.args]
            return f"def {node.name}({', '.join(params)}):"

    def _parse_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """解析参数"""
        parameters = []

        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg.annotation) if arg.annotation else 'Any',
                'default': None,
                'required': True
            }
            parameters.append(param_info)

        # 处理默认值参数
        defaults = node.args.defaults
        if defaults:
            offset = len(node.args.args) - len(defaults)
            for i, default in enumerate(defaults):
                param_index = offset + i
                if param_index < len(parameters):
                    parameters[param_index]['default'] = self._get_default_value(default)
                    parameters[param_index]['required'] = False

        return parameters

    def _get_type_annotation(self, annotation) -> str:
        """获取类型注解"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return f"{annotation.value.id}[{self._get_type_annotation(annotation.slice)}]"
        elif isinstance(annotation, ast.Str):
            return annotation.s
        else:
            return "Any"

    def _get_default_value(self, default) -> str:
        """获取默认值"""
        if isinstance(default, ast.Str):
            return f"'{default.s}'"
        elif isinstance(default, ast.Num):
            return str(default.n)
        elif isinstance(default, ast.NameConstant):
            return str(default.value)
        else:
            return "..."

    def _get_return_type(self, node: ast.FunctionDef) -> str:
        """获取返回类型"""
        if node.returns:
            return self._get_type_annotation(node.returns)
        return "None"

    def _get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return str(decorator)

    def _get_visibility(self, func_name: str) -> str:
        """获取可见性"""
        if func_name.startswith('__'):
            return 'private'
        elif func_name.startswith('_'):
            return 'protected'
        else:
            return 'public'

    def generate_markdown_docs(self) -> str:
        """
        生成Markdown文档

        Returns:
            Markdown文档内容
        """
        print("📝 生成Markdown文档...")

        content = ["# RQA2025 API Documentation\n", f"Generated on: {self._get_timestamp()}\n"]

        # 按模块分组
        modules_by_category = self._group_modules_by_category()

        for category, modules in modules_by_category.items():
            content.append(f"## {category}\n")

            for module_name, module in modules.items():
                content.append(f"### {module_name}\n")

                if module.docstring:
                    content.append(f"{module.docstring}\n")

                if module.endpoints:
                    content.append("#### API Endpoints\n")

                    for endpoint in module.endpoints:
                        content.append(self._format_endpoint_markdown(endpoint))

                content.append("---\n")

        return "\n".join(content)

    def _group_modules_by_category(self) -> Dict[str, Dict[str, APIModule]]:
        """按类别分组模块"""
        categories = {
            'Core Services': [],
            'Infrastructure': [],
            'Business Logic': [],
            'Security': [],
            'Monitoring': [],
            'Other': []
        }

        for module_name, module in self.modules.items():
            category = self._get_module_category(module_name)
            categories[category].append((module_name, module))

        # 转换为字典
        result = {}
        for category, modules in categories.items():
            if modules:
                result[category] = dict(modules)

        return result

    def _get_module_category(self, module_name: str) -> str:
        """获取模块类别"""
        if 'core' in module_name:
            return 'Core Services'
        elif 'infrastructure' in module_name:
            return 'Infrastructure'
        elif 'business' in module_name:
            return 'Business Logic'
        elif 'security' in module_name:
            return 'Security'
        elif 'monitoring' in module_name:
            return 'Monitoring'
        else:
            return 'Other'

    def _format_endpoint_markdown(self, endpoint: APIEndpoint) -> str:
        """格式化端点为Markdown"""
        content = [f"#### `{endpoint.name}`\n"]

        if endpoint.docstring:
            content.append(f"{endpoint.docstring}\n")

        content.append(f"**Signature:** `{endpoint.signature}`\n")

        if endpoint.parameters:
            content.append("**Parameters:**\n")
            for param in endpoint.parameters:
                required = " (required)" if param['required'] else f" (default: {param['default']})"
                content.append(f"- `{param['name']}: {param['type']}`{required}")
            content.append("")

        if endpoint.return_type:
            content.append(f"**Returns:** `{endpoint.return_type}`\n")

        if endpoint.decorators:
            content.append(f"**Decorators:** {', '.join(f'`{d}`' for d in endpoint.decorators)}\n")

        content.append(f"**Async:** {'Yes' if endpoint.is_async else 'No'} | **Visibility:** {endpoint.visibility}\n")

        return "\n".join(content)

    def generate_json_docs(self) -> str:
        """
        生成JSON文档

        Returns:
            JSON文档内容
        """
        print("📋 生成JSON文档...")

        data = {
            'generated_at': self._get_timestamp(),
            'total_modules': len(self.modules),
            'total_endpoints': sum(len(m.endpoints) for m in self.modules.values()),
            'modules': {}
        }

        for module_name, module in self.modules.items():
            data['modules'][module_name] = {
                'path': module.path,
                'docstring': module.docstring,
                'endpoints': [
                    {
                        'name': e.name,
                        'signature': e.signature,
                        'docstring': e.docstring,
                        'parameters': e.parameters,
                        'return_type': e.return_type,
                        'decorators': e.decorators,
                        'is_async': e.is_async,
                        'visibility': e.visibility
                    }
                    for e in module.endpoints
                ]
            }

        return json.dumps(data, indent=2, ensure_ascii=False)

    def save_docs(self, format_type: str = 'markdown'):
        """
        保存文档

        Args:
            format_type: 文档格式 ('markdown' 或 'json')
        """
        if format_type == 'markdown':
            content = self.generate_markdown_docs()
            filename = "api_documentation.md"
        elif format_type == 'json':
            content = self.generate_json_docs()
            filename = "api_documentation.json"
        else:
            raise ValueError(f"不支持的格式: {format_type}")

        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 文档已保存到: {output_file}")

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python generate_api_docs.py <source_dir> [output_dir]")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/api"

    # 创建文档生成器
    generator = APIDocumentationGenerator(source_dir, output_dir)

    # 扫描模块
    generator.scan_modules()

    # 生成并保存文档
    generator.save_docs('markdown')
    generator.save_docs('json')

    print("🎉 API文档生成完成！")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
基础设施层配置管理API文档自动生成工具
Phase 2.3: 建立自动化质量检查 - API文档生成
"""

import os
import ast
from typing import Dict, List, Any, Optional
import datetime


class APIDocumentationGenerator:
    """API文档生成器"""

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.modules = {}
        self.classes = {}
        self.functions = {}

    def analyze_module(self, file_path: str) -> Dict[str, Any]:
        """分析单个模块"""
        rel_path = os.path.relpath(file_path, self.base_path)

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            # 提取模块信息
            module_info = {
                'path': rel_path,
                'name': os.path.splitext(os.path.basename(rel_path))[0],
                'docstring': ast.get_docstring(tree) or "",
                'classes': [],
                'functions': [],
                'imports': []
            }

            # 提取导入语句
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports = [f"import {alias.name}" for alias in node.names]
                    else:
                        module = node.module or ""
                        imports = [f"from {module} import {alias.name}" for alias in node.names]
                    module_info['imports'].extend(imports)

            # 提取类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node)
                    module_info['classes'].append(class_info)

                elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef)
                                                                   for parent in self._get_parents(node, tree)):
                    func_info = self._extract_function_info(node)
                    module_info['functions'].append(func_info)

            return module_info

        except Exception as e:
            return {
                'path': rel_path,
                'error': str(e),
                'classes': [],
                'functions': []
            }

    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """提取类信息"""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'bases': [self._get_name(base) for base in node.bases],
            'methods': [],
            'attributes': []
        }

        # 提取方法
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item)
                class_info['methods'].append(method_info)

        return class_info

    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """提取函数信息"""
        # 解析参数
        args = []
        defaults = [self._get_literal_value(d) for d in node.args.defaults]
        default_offset = len(node.args.args) - len(defaults)

        for i, arg in enumerate(node.args.args):
            default_value = None
            if i >= default_offset:
                default_value = defaults[i - default_offset]

            arg_info = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation),
                'default': default_value
            }
            args.append(arg_info)

        # 处理*args和**kwargs
        if node.args.vararg:
            args.append({
                'name': f"*{node.args.vararg.arg}",
                'annotation': self._get_annotation(node.args.vararg.annotation),
                'default': None
            })

        if node.args.kwarg:
            args.append({
                'name': f"**{node.args.kwarg.arg}",
                'annotation': self._get_annotation(node.args.kwarg.annotation),
                'default': None
            })

        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'args': args,
            'return_annotation': self._get_annotation(node.returns),
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }

    def _get_name(self, node: ast.AST) -> str:
        """获取AST节点的名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_annotation(self, node: Optional[ast.AST]) -> str:
        """获取类型注解"""
        if node is None:
            return ""
        return self._get_name(node)

    def _get_literal_value(self, node: ast.AST) -> Any:
        """获取字面量值"""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._get_literal_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            keys = [self._get_literal_value(k) for k in node.keys if k]
            values = [self._get_literal_value(v) for v in node.values]
            return dict(zip(keys, values))
        return None

    def _get_parents(self, node: ast.AST, tree: ast.AST) -> List[ast.AST]:
        """获取节点的父节点"""
        parents = []

        def find_parents(current, target, path):
            if current == target:
                parents.extend(path)
                return True

            for child in ast.iter_child_nodes(current):
                if find_parents(child, target, path + [current]):
                    return True
            return False

        find_parents(tree, node, [])
        return parents

    def analyze_all_modules(self):
        """分析所有模块"""
        print("🔍 分析所有模块...")

        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    module_info = self.analyze_module(file_path)
                    self.modules[module_info['path']] = module_info

    def generate_module_doc(self, module_path: str) -> str:
        """生成单个模块的文档"""
        if module_path not in self.modules:
            return f"# 模块 {module_path}\n\n未找到模块信息"

        module = self.modules[module_path]

        module_name = module.get('name', os.path.splitext(os.path.basename(module_path))[0])
        doc = f"# {module_name}\n\n"
        doc += f"**文件路径**: `{module_path}`\n\n"

        if module.get('docstring'):
            doc += f"## 模块描述\n\n{module['docstring']}\n\n"

        # 导入语句
        if module.get('imports'):
            doc += "## 导入语句\n\n```python\n"
            for imp in module['imports'][:10]:  # 只显示前10个
                doc += f"{imp}\n"
            if len(module['imports']) > 10:
                doc += f"# ... 等{len(module['imports'])-10}个导入\n"
            doc += "```\n\n"

        # 类
        if module.get('classes'):
            doc += "## 类\n\n"
            for cls in module['classes']:
                doc += f"### {cls['name']}\n\n"
                if cls.get('docstring'):
                    doc += f"{cls['docstring']}\n\n"

                if cls.get('bases'):
                    doc += f"**继承**: {', '.join(cls['bases'])}\n\n"

                if cls.get('methods'):
                    doc += "**方法**:\n\n"
                    for method in cls['methods'][:5]:  # 只显示前5个方法
                        doc += f"- `{method['name']}`\n"
                    if len(cls['methods']) > 5:
                        doc += f"- ... 等{len(cls['methods'])-5}个方法\n"
                    doc += "\n"

        # 函数
        if module.get('functions'):
            doc += "## 函数\n\n"
            for func in module['functions']:
                doc += f"### {func['name']}\n\n"
                if func.get('docstring'):
                    doc += f"{func['docstring']}\n\n"

                # 参数
                if func.get('args'):
                    doc += "**参数**:\n\n"
                    for arg in func['args']:
                        default = f" = {arg['default']}" if arg['default'] is not None else ""
                        annotation = f": {arg['annotation']}" if arg['annotation'] else ""
                        doc += f"- `{arg['name']}{annotation}{default}`\n"
                    doc += "\n"

                # 返回值
                if func.get('return_annotation'):
                    doc += f"**返回值**: `{func['return_annotation']}`\n\n"

        return doc

    def generate_overview_doc(self) -> str:
        """生成总览文档"""
        doc = "# 基础设施层配置管理API文档总览\n\n"
        doc += f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        doc += f"**模块总数**: {len(self.modules)}\n\n"

        # 按目录分组
        directories = {}
        for path, module in self.modules.items():
            dir_name = os.path.dirname(path).split(os.sep)[0] if os.sep in path else 'root'
            if dir_name not in directories:
                directories[dir_name] = []
            directories[dir_name].append(module)

        # 生成目录结构
        doc += "## 目录结构\n\n"
        for dir_name, modules in sorted(directories.items()):
            if dir_name == 'root':
                doc += f"### 根目录 ({len(modules)} 个模块)\n\n"
            else:
                doc += f"### {dir_name}/ ({len(modules)} 个模块)\n\n"

            for module in sorted(modules, key=lambda x: x.get('name', x.get('path', 'unknown'))):
                module_name = module.get('name', module.get('path', 'unknown'))
                doc += f"- [{module_name}]({module['path'].replace('.py', '.md')}) - {len(module.get('classes', []))}个类, {len(module.get('functions', []))}个函数\n"
            doc += "\n"

        # 统计信息
        total_classes = sum(len(m.get('classes', [])) for m in self.modules.values())
        total_functions = sum(len(m.get('functions', [])) for m in self.modules.values())
        total_methods = sum(sum(len(c.get('methods', []))
                            for c in m.get('classes', [])) for m in self.modules.values())

        doc += "## 统计信息\n\n"
        doc += f"- **总模块数**: {len(self.modules)}\n"
        doc += f"- **总类数**: {total_classes}\n"
        doc += f"- **总函数数**: {total_functions}\n"
        doc += f"- **总方法数**: {total_methods}\n"
        doc += f"- **平均每模块类数**: {total_classes/len(self.modules):.1f}\n"
        doc += f"- **平均每模块函数数**: {total_functions/len(self.modules):.1f}\n\n"

        # 最大的模块
        largest_modules = sorted(self.modules.values(),
                                 key=lambda x: len(x.get('classes', [])) +
                                 len(x.get('functions', [])),
                                 reverse=True)[:5]

        doc += "## 最大模块\n\n"
        for module in largest_modules:
            class_count = len(module.get('classes', []))
            func_count = len(module.get('functions', []))
            total_count = class_count + func_count
            doc += f"- `{module['path']}`: {class_count}个类 + {func_count}个函数 = {total_count}个成员\n"
        doc += "\n"

        return doc

    def generate_all_docs(self, output_dir: str = "api_docs"):
        """生成所有文档"""
        print("📝 生成API文档...")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成总览文档
        overview_doc = self.generate_overview_doc()
        with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(overview_doc)

        # 生成各模块文档
        generated_count = 0
        for module_path in self.modules:
            module_doc = self.generate_module_doc(module_path)
            output_path = os.path.join(output_dir, module_path.replace('.py', '.md'))

            # 创建子目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(module_doc)

            generated_count += 1

        print(f"✅ 生成了 {generated_count} 个模块文档")
        return generated_count


def main():
    """主函数"""
    print("=== 📚 Phase 2.3: API文档自动生成 ===\\n")

    # 生成配置管理模块的API文档
    generator = APIDocumentationGenerator('src/infrastructure/config')

    try:
        # 分析所有模块
        generator.analyze_all_modules()

        # 生成文档
        doc_count = generator.generate_all_docs('config_api_docs')

        print("\\n📋 API文档生成完成！")
        print(f"   📄 生成文档数: {doc_count}")
        print("   📁 输出目录: config_api_docs/")
        print("   📖 总览文档: config_api_docs/README.md")

        # 输出统计信息
        total_classes = sum(len(m.get('classes', [])) for m in generator.modules.values())
        total_functions = sum(len(m.get('functions', [])) for m in generator.modules.values())

        print("\\n📊 文档统计:")
        print(f"   🏗️ 模块总数: {len(generator.modules)}")
        print(f"   🏛️ 类总数: {total_classes}")
        print(f"   🔧 函数总数: {total_functions}")

        return True

    except Exception as e:
        print(f"❌ API文档生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    print(f"\\n{'🎉 Phase 2.3 API文档生成完成！' if success else '❌ Phase 2.3 API文档生成失败！'}")

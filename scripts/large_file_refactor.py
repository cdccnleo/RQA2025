#!/usr/bin/env python3
"""
大文件重构工具

用于分析和重构超过1000行的代码文件，实现模块化拆分。
支持功能模块识别、接口抽象设计、自动化拆分实施。
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeEntity:
    """代码实体"""
    name: str
    type: str  # 'class', 'function', 'method'
    start_line: int
    end_line: int
    dependencies: List[str]
    complexity: int
    docstring: Optional[str] = None


@dataclass
class ModuleGroup:
    """模块分组"""
    name: str
    entities: List[CodeEntity]
    dependencies: List[str]
    cohesion_score: float
    description: str


class LargeFileAnalyzer:
    """大文件分析器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = ""
        self.lines = []
        self.entities = []
        self.imports = []
        self.tree = None

    def analyze_file(self) -> Dict[str, Any]:
        """分析文件"""
        logger.info(f"开始分析文件: {self.file_path}")

        # 读取文件内容
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
            self.lines = self.content.splitlines()

        # 解析AST
        try:
            self.tree = ast.parse(self.content)
        except SyntaxError as e:
            logger.error(f"AST解析失败: {e}")
            return {"error": f"语法错误: {e}"}

        # 提取导入语句
        self._extract_imports()

        # 提取代码实体
        self._extract_entities()

        # 计算复杂度指标
        metrics = self._calculate_metrics()

        return {
            "file_path": str(self.file_path),
            "total_lines": len(self.lines),
            "entities": [entity.__dict__ for entity in self.entities],
            "imports": self.imports,
            "metrics": metrics,
            "analysis_complete": True
        }

    def _extract_imports(self):
        """提取导入语句"""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports.append(f"import {alias.name}")
                else:
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    self.imports.append(f"from {module} import {', '.join(names)}")

    def _extract_entities(self):
        """提取代码实体"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    name=node.name,
                    type="class",
                    start_line=node.lineno,
                    end_line=getattr(node, 'end_lineno', node.lineno),
                    dependencies=self._extract_dependencies(node),
                    complexity=self._calculate_complexity(node),
                    docstring=ast.get_docstring(node)
                )
                self.entities.append(entity)

            elif isinstance(node, ast.FunctionDef):
                # 检查是否是方法
                is_method = False
                for parent in ast.walk(self.tree):
                    if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
                        is_method = True
                        break

                entity = CodeEntity(
                    name=node.name,
                    type="method" if is_method else "function",
                    start_line=node.lineno,
                    end_line=getattr(node, 'end_lineno', node.lineno),
                    dependencies=self._extract_dependencies(node),
                    complexity=self._calculate_complexity(node),
                    docstring=ast.get_docstring(node)
                )
                self.entities.append(entity)

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """提取依赖关系"""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):  # 只处理读取的名称
                    dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.append(f"{child.value.id}.{child.attr}")

        return list(set(dependencies))

    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_metrics(self) -> Dict[str, Any]:
        """计算文件指标"""
        total_lines = len(self.lines)
        code_lines = len([line for line in self.lines if line.strip()
                         and not line.strip().startswith('#')])

        classes = [e for e in self.entities if e.type == "class"]
        functions = [e for e in self.entities if e.type == "function"]
        methods = [e for e in self.entities if e.type == "method"]

        avg_complexity = sum(e.complexity for e in self.entities) / \
            len(self.entities) if self.entities else 0
        max_complexity = max(e.complexity for e in self.entities) if self.entities else 0

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_ratio": (total_lines - code_lines) / total_lines if total_lines > 0 else 0,
            "classes_count": len(classes),
            "functions_count": len(functions),
            "methods_count": len(methods),
            "entities_count": len(self.entities),
            "avg_complexity": avg_complexity,
            "max_complexity": max_complexity,
            "needs_refactor": total_lines > 1000
        }


class ModuleRefactorer:
    """模块重构器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.analyzer = LargeFileAnalyzer(file_path)

    def identify_modules(self) -> List[ModuleGroup]:
        """识别功能模块"""
        analysis = self.analyzer.analyze_file()
        entities = [CodeEntity(**e) for e in analysis["entities"]]

        # 基于名称和依赖关系进行聚类
        modules = self._cluster_entities(entities)

        return modules

    def _cluster_entities(self, entities: List[CodeEntity]) -> List[ModuleGroup]:
        """聚类实体"""
        # 简单的基于名称前缀的聚类
        clusters = defaultdict(list)

        for entity in entities:
            # 基于名称模式识别模块
            if "user" in entity.name.lower() or "auth" in entity.name.lower():
                clusters["user_management"].append(entity)
            elif "order" in entity.name.lower() or "trade" in entity.name.lower():
                clusters["trading"].append(entity)
            elif "market" in entity.name.lower() or "data" in entity.name.lower():
                clusters["market_data"].append(entity)
            elif "position" in entity.name.lower() or "portfolio" in entity.name.lower():
                clusters["portfolio"].append(entity)
            elif "config" in entity.name.lower() or "setting" in entity.name.lower():
                clusters["configuration"].append(entity)
            else:
                clusters["utilities"].append(entity)

        # 转换为ModuleGroup
        modules = []
        for name, entities_list in clusters.items():
            if entities_list:
                cohesion = self._calculate_cohesion(entities_list)
                modules.append(ModuleGroup(
                    name=name,
                    entities=entities_list,
                    dependencies=self._extract_module_dependencies(entities_list),
                    cohesion_score=cohesion,
                    description=f"{name.replace('_', ' ').title()} module"
                ))

        return modules

    def _calculate_cohesion(self, entities: List[CodeEntity]) -> float:
        """计算内聚性"""
        if len(entities) <= 1:
            return 1.0

        # 基于共享依赖计算内聚性
        all_deps = set()
        for entity in entities:
            all_deps.update(entity.dependencies)

        shared_deps = 0
        for dep in all_deps:
            count = sum(1 for e in entities if dep in e.dependencies)
            if count > 1:
                shared_deps += 1

        return shared_deps / len(all_deps) if all_deps else 0.0

    def _extract_module_dependencies(self, entities: List[CodeEntity]) -> List[str]:
        """提取模块依赖"""
        all_deps = set()
        for entity in entities:
            all_deps.update(entity.dependencies)

        # 过滤掉模块内部的依赖
        internal_names = {e.name for e in entities}
        external_deps = [dep for dep in all_deps if dep not in internal_names]

        return external_deps

    def generate_refactor_plan(self) -> Dict[str, Any]:
        """生成重构计划"""
        analysis = self.analyzer.analyze_file()
        modules = self.identify_modules()

        plan = {
            "original_file": str(self.file_path),
            "analysis": analysis,
            "modules": [
                {
                    "name": m.name,
                    "entities_count": len(m.entities),
                    "cohesion_score": m.cohesion_score,
                    "dependencies": m.dependencies,
                    "description": m.description,
                    "estimated_lines": sum(e.end_line - e.start_line + 1 for e in m.entities)
                }
                for m in modules
            ],
            "refactor_strategy": self._generate_strategy(modules),
            "implementation_steps": self._generate_steps(modules)
        }

        return plan

    def _generate_strategy(self, modules: List[ModuleGroup]) -> str:
        """生成重构策略"""
        high_cohesion = [m for m in modules if m.cohesion_score > 0.7]
        low_cohesion = [m for m in modules if m.cohesion_score <= 0.7]

        strategy = f"识别到 {len(modules)} 个功能模块。"
        if high_cohesion:
            strategy += f" {len(high_cohesion)} 个模块内聚性良好，可直接拆分。"
        if low_cohesion:
            strategy += f" {len(low_cohesion)} 个模块需要进一步优化。"

        return strategy

    def _generate_steps(self, modules: List[ModuleGroup]) -> List[str]:
        """生成实施步骤"""
        steps = [
            "1. 创建模块目录结构",
            "2. 提取共享导入和常量",
            "3. 按模块拆分代码实体",
            "4. 创建模块接口和工厂",
            "5. 更新原有文件的导入",
            "6. 添加单元测试",
            "7. 验证功能完整性"
        ]

        return steps

    def implement_refactor(self, target_dir: str = None) -> Dict[str, Any]:
        """实施重构"""
        if target_dir is None:
            target_dir = self.file_path.parent / f"{self.file_path.stem}_modules"

        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True)

        plan = self.generate_refactor_plan()
        modules = self.identify_modules()

        # 读取原始文件内容
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {
            "target_directory": str(target_dir),
            "modules_created": [],
            "files_generated": 0,
            "lines_refactored": 0
        }

        # 为每个模块创建文件
        for module in modules:
            if len(module.entities) > 0:
                module_file = target_dir / f"{module.name}.py"
                module_content = self._generate_module_content(module, content)

                with open(module_file, 'w', encoding='utf-8') as f:
                    f.write(module_content)

                results["modules_created"].append({
                    "name": module.name,
                    "file": str(module_file),
                    "entities_count": len(module.entities),
                    "lines": len(module_content.splitlines())
                })

                results["files_generated"] += 1
                results["lines_refactored"] += len(module_content.splitlines())

        # 生成主文件（保留接口）
        main_file = target_dir / f"{self.file_path.stem}_main.py"
        main_content = self._generate_main_content(modules)

        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(main_content)

        results["main_file"] = str(main_file)

        return results

    def _generate_module_content(self, module: ModuleGroup, original_content: str) -> str:
        """生成模块内容"""
        lines = original_content.splitlines()

        # 提取模块相关的代码
        module_lines = []
        for entity in module.entities:
            start_idx = entity.start_line - 1  # 转换为0索引
            end_idx = min(entity.end_line, len(lines))  # 确保不越界

            # 添加一些上下文行
            context_start = max(0, start_idx - 2)
            context_end = min(len(lines), end_idx + 2)

            if context_start < start_idx:
                module_lines.extend(lines[context_start:start_idx])

            module_lines.extend(lines[start_idx:end_idx])

            if end_idx < context_end:
                module_lines.extend(lines[end_idx:context_end])

            module_lines.append("")  # 添加空行

        # 添加导入
        imports = [
            "import logging",
            "from typing import Dict, List, Any, Optional",
            f"from {self.file_path.parent.name}.{self.file_path.stem} import *"
        ]

        # 合并内容
        content = "\n".join(imports) + "\n\n" + "\n".join(module_lines)

        return content

    def _generate_main_content(self, modules: List[ModuleGroup]) -> str:
        """生成主文件内容"""
        imports = [f"from .{module.name} import *" for module in modules]

        header = f'''"""
{self.file_path.stem} 主文件
拆分后的模块统一接口
"""

'''

        imports_str = "\n".join(imports)
        footer = '''

# 保持原有接口的兼容性
'''

        content = header + imports_str + footer

        return content


def main():
    """主函数"""
    print("🚀 大文件重构和代码规范专项")
    print("=" * 60)

    # 要重构的文件列表
    large_files = [
        "src/mobile/mobile_trading.py",
        "src/features/acceleration/gpu/gpu_scheduler.py",
        "src/core/optimizations/short_term_optimizations.py",
        "src/data/integration/enhanced_data_integration.py"
    ]

    results = {
        "files_analyzed": [],
        "refactor_plans": [],
        "implementation_results": []
    }

    for file_path in large_files:
        print(f"\n📊 分析文件: {file_path}")

        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            continue

        # 分析文件
        refactorer = ModuleRefactorer(file_path)
        analysis = refactorer.analyzer.analyze_file()

        if "error" in analysis:
            print(f"❌ 分析失败: {analysis['error']}")
            continue

        metrics = analysis["metrics"]
        print(f"   📏 总行数: {metrics['total_lines']}")
        print(f"   📚 代码实体: {metrics['entities_count']}")
        print(f"   🔄 平均复杂度: {metrics['avg_complexity']:.1f}")
        print(f"   ⚠️  需要重构: {metrics['needs_refactor']}")

        results["files_analyzed"].append(analysis)

        # 生成重构计划
        plan = refactorer.generate_refactor_plan()
        results["refactor_plans"].append(plan)

        print(f"   🎯 识别模块: {len(plan['modules'])}")

        # 实施重构
        try:
            refactor_result = refactorer.implement_refactor()
            results["implementation_results"].append(refactor_result)

            print(f"   ✅ 创建文件: {refactor_result['files_generated']}")
            print(f"   📝 重构行数: {refactor_result['lines_refactored']}")

        except Exception as e:
            print(f"   ❌ 重构失败: {e}")

    # 保存结果
    with open("large_file_refactor_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # 输出总结
    total_files = len(results["files_analyzed"])
    total_modules = sum(len(plan.get("modules", [])) for plan in results["refactor_plans"])
    total_new_files = sum(len(result.get("modules_created", []))
                          for result in results["implementation_results"])

    print("\n📊 重构总结")
    print("-" * 50)
    print(f"📁 分析文件: {total_files}")
    print(f"🎯 识别模块: {total_modules}")
    print(f"📄 新建文件: {total_new_files}")
    print(f"💾 报告已保存: large_file_refactor_report.json")

    print("\n✅ 大文件重构专项完成！")


if __name__ == "__main__":
    main()

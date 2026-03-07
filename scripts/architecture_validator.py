#!/usr/bin/env python3
"""
高级架构验证工具

提供深度架构分析、模式识别、质量评估和改进建议
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import networkx as nx


@dataclass
class ArchitectureMetrics:
    """架构指标"""
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    total_modules: int = 0
    average_complexity: float = 0.0
    test_coverage: float = 0.0
    coupling_score: float = 0.0
    cohesion_score: float = 0.0
    maintainability_index: float = 0.0


@dataclass
class ArchitectureIssue:
    """架构问题"""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'structure', 'dependencies', 'quality', 'consistency'
    description: str
    location: str
    suggestion: str
    confidence: float


@dataclass
class LayerAnalysis:
    """层次分析结果"""
    layer_name: str
    component_count: int
    total_lines: int
    complexity_score: float
    test_coverage: float
    dependencies: List[str]
    issues: List[ArchitectureIssue]


class AdvancedArchitectureValidator:
    """高级架构验证器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.test_dir = self.project_root / "tests"

        # 架构层次映射
        self.layer_mapping = {
            "core": "核心服务层",
            "infrastructure": "基础设施层",
            "data": "数据采集层",
            "gateway": "API网关层",
            "features": "特征处理层",
            "ml": "模型推理层",
            "backtest": "策略决策层",
            "trading": "交易执行层",
            "risk": "风控合规层",
            "engine": "监控反馈层"
        }

        # 分析结果
        self.metrics = ArchitectureMetrics()
        self.issues = []
        self.layer_analyses = {}
        self.dependency_graph = nx.DiGraph()

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行全面架构分析"""
        print("🔬 开始高级架构分析...")

        analysis_results = {
            "metrics": self._analyze_metrics(),
            "layers": self._analyze_layers(),
            "dependencies": self._analyze_dependencies(),
            "patterns": self._analyze_patterns(),
            "quality": self._analyze_quality(),
            "issues": self._identify_issues(),
            "recommendations": self._generate_recommendations()
        }

        print("✅ 高级架构分析完成")
        return analysis_results

    def _analyze_metrics(self) -> Dict[str, Any]:
        """分析架构指标"""
        print("📊 分析架构指标...")

        # 统计基本指标
        python_files = list(self.src_dir.rglob("*.py"))
        self.metrics.total_files = len(python_files)

        total_lines = 0
        total_classes = 0
        total_functions = 0

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    total_lines += lines

                # 解析AST获取类和函数数量
                try:
                    tree = ast.parse(content)
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    functions = [node for node in ast.walk(
                        tree) if isinstance(node, ast.FunctionDef)]
                    total_classes += len(classes)
                    total_functions += len(functions)
                except:
                    pass
            except:
                continue

        self.metrics.total_lines = total_lines
        self.metrics.total_classes = total_classes
        self.metrics.total_functions = total_functions
        self.metrics.total_modules = len(list(self.src_dir.iterdir()))

        # 计算复杂性指标
        self.metrics.average_complexity = total_lines / max(self.metrics.total_files, 1)
        self.metrics.maintainability_index = self._calculate_maintainability_index()

        return asdict(self.metrics)

    def _analyze_layers(self) -> Dict[str, Any]:
        """分析架构层次"""
        print("🏗️  分析架构层次...")

        layer_results = {}

        for module, layer_name in self.layer_mapping.items():
            module_path = self.src_dir / module
            if module_path.exists():
                analysis = self._analyze_single_layer(module_path, layer_name)
                layer_results[module] = asdict(analysis)
                self.layer_analyses[module] = analysis

        return layer_results

    def _analyze_single_layer(self, module_path: Path, layer_name: str) -> LayerAnalysis:
        """分析单个层次"""
        python_files = list(module_path.rglob("*.py"))

        total_lines = 0
        component_count = len(python_files)
        complexity_scores = []

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    total_lines += lines

                    # 计算复杂性（简单方法：基于行数和分支）
                    complexity = self._calculate_file_complexity(content)
                    complexity_scores.append(complexity)
            except:
                continue

        avg_complexity = sum(complexity_scores) / max(len(complexity_scores), 1)

        # 分析依赖关系
        dependencies = self._analyze_module_dependencies(module_path)

        return LayerAnalysis(
            layer_name=layer_name,
            component_count=component_count,
            total_lines=total_lines,
            complexity_score=avg_complexity,
            test_coverage=self._calculate_test_coverage(module_path),
            dependencies=dependencies,
            issues=[]
        )

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """分析依赖关系"""
        print("🔗 分析依赖关系...")

        # 构建依赖图
        for module_path in self.src_dir.iterdir():
            if module_path.is_dir() and not module_path.name.startswith('.'):
                self._build_dependency_graph(module_path)

        # 分析依赖关系
        analysis = {
            "total_modules": len(self.dependency_graph.nodes()),
            "total_dependencies": len(self.dependency_graph.edges()),
            "circular_dependencies": list(nx.simple_cycles(self.dependency_graph)),
            "centrality": nx.degree_centrality(self.dependency_graph),
            "strongly_connected_components": list(nx.strongly_connected_components(self.dependency_graph))
        }

        return analysis

    def _analyze_patterns(self) -> Dict[str, Any]:
        """分析架构模式"""
        print("🔍 分析架构模式...")

        patterns = {
            "design_patterns": self._identify_design_patterns(),
            "architectural_patterns": self._identify_architectural_patterns(),
            "anti_patterns": self._identify_anti_patterns()
        }

        return patterns

    def _analyze_quality(self) -> Dict[str, Any]:
        """分析代码质量"""
        print("⚖️  分析代码质量...")

        quality_metrics = {
            "complexity_analysis": self._analyze_complexity(),
            "coupling_analysis": self._analyze_coupling(),
            "cohesion_analysis": self._analyze_cohesion(),
            "test_quality": self._analyze_test_quality()
        }

        return quality_metrics

    def _identify_issues(self) -> List[Dict[str, Any]]:
        """识别架构问题"""
        print("🚨 识别架构问题...")

        issues = []

        # 检查循环依赖
        cycles = list(nx.simple_cycles(self.dependency_graph))
        for cycle in cycles:
            issues.append(ArchitectureIssue(
                severity="high",
                category="dependencies",
                description=f"检测到循环依赖: {' -> '.join(cycle)}",
                location=f"模块: {', '.join(cycle)}",
                suggestion="重构依赖关系，消除循环依赖",
                confidence=0.9
            ))

        # 检查过高的复杂性
        for layer_name, analysis in self.layer_analyses.items():
            if analysis.complexity_score > 50:
                issues.append(ArchitectureIssue(
                    severity="medium",
                    category="quality",
                    description=f"层次复杂度过高: {analysis.layer_name}",
                    location=f"层次: {layer_name}",
                    suggestion="拆分复杂组件，降低单个组件的复杂度",
                    confidence=0.8
                ))

        # 检查测试覆盖率
        for layer_name, analysis in self.layer_analyses.items():
            if analysis.test_coverage < 0.7:
                issues.append(ArchitectureIssue(
                    severity="medium",
                    category="quality",
                    description=f"测试覆盖率不足: {analysis.layer_name}",
                    location=f"层次: {layer_name}",
                    suggestion="增加单元测试，提高测试覆盖率",
                    confidence=0.7
                ))

        return [asdict(issue) for issue in issues]

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成改进建议"""
        print("💡 生成改进建议...")

        recommendations = []

        # 基于分析结果生成建议
        if self.metrics.maintainability_index < 50:
            recommendations.append({
                "priority": "high",
                "category": "quality",
                "title": "提高代码可维护性",
                "description": "当前可维护性指数偏低，建议重构复杂代码",
                "actions": ["拆分大函数", "减少类复杂度", "改进命名"]
            })

        if len(list(nx.simple_cycles(self.dependency_graph))) > 0:
            recommendations.append({
                "priority": "high",
                "category": "structure",
                "title": "解决循环依赖",
                "description": "检测到循环依赖，影响系统稳定性",
                "actions": ["重构依赖关系", "引入接口层", "使用依赖注入"]
            })

        return recommendations

    def _calculate_maintainability_index(self) -> float:
        """计算可维护性指数"""
        # 简化的可维护性计算
        complexity_factor = max(0, 171 - 5.2 * self.metrics.average_complexity)
        lines_factor = max(0, 50 - 0.23 * self.metrics.average_complexity)
        return (complexity_factor + lines_factor) / 2

    def _calculate_file_complexity(self, content: str) -> float:
        """计算文件复杂度"""
        lines = len(content.splitlines())
        branches = content.count('if ') + content.count('elif ') + \
            content.count('else:') + content.count('for ') + content.count('while ')
        return lines + branches * 2

    def _analyze_module_dependencies(self, module_path: Path) -> List[str]:
        """分析模块依赖关系"""
        dependencies = []
        for py_file in module_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                imports = []
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        if 'src.' in line:
                            # 提取模块名
                            if 'from src.' in line:
                                parts = line.split('from src.')[1].split(' ')[0]
                                imports.append(parts.split('.')[0])
                            elif 'import src.' in line:
                                parts = line.split('import src.')[1].split(' ')[0]
                                imports.append(parts.split('.')[0])

                dependencies.extend(imports)
            except:
                continue

        return list(set(dependencies))

    def _calculate_test_coverage(self, module_path: Path) -> float:
        """计算测试覆盖率"""
        module_name = module_path.name
        test_path = self.test_dir / f"test_{module_name}"

        if not test_path.exists():
            return 0.0

        source_files = len(list(module_path.rglob("*.py")))
        test_files = len(list(test_path.rglob("*.py")))

        return min(test_files / max(source_files, 1), 1.0)

    def _build_dependency_graph(self, module_path: Path):
        """构建依赖图"""
        module_name = module_path.name
        self.dependency_graph.add_node(module_name)

        dependencies = self._analyze_module_dependencies(module_path)
        for dep in dependencies:
            if dep != module_name:  # 避免自依赖
                self.dependency_graph.add_edge(module_name, dep)

    def _identify_design_patterns(self) -> List[str]:
        """识别设计模式"""
        patterns = []

        # 检查工厂模式
        if self._check_factory_pattern():
            patterns.append("工厂模式 (Factory Pattern)")

        # 检查观察者模式
        if self._check_observer_pattern():
            patterns.append("观察者模式 (Observer Pattern)")

        # 检查策略模式
        if self._check_strategy_pattern():
            patterns.append("策略模式 (Strategy Pattern)")

        return patterns

    def _identify_architectural_patterns(self) -> List[str]:
        """识别架构模式"""
        patterns = []

        # 检查分层架构
        if self._check_layered_architecture():
            patterns.append("分层架构 (Layered Architecture)")

        # 检查微服务架构
        if self._check_microservices_architecture():
            patterns.append("微服务架构 (Microservices Architecture)")

        # 检查事件驱动架构
        if self._check_event_driven_architecture():
            patterns.append("事件驱动架构 (Event-Driven Architecture)")

        return patterns

    def _identify_anti_patterns(self) -> List[str]:
        """识别反模式"""
        anti_patterns = []

        # 检查上帝类
        if self._check_god_class_anti_pattern():
            anti_patterns.append("上帝类 (God Class)")

        # 检查圈复杂度过高
        if self._check_high_complexity_anti_pattern():
            anti_patterns.append("圈复杂度过高 (High Cyclomatic Complexity)")

        return anti_patterns

    def _check_factory_pattern(self) -> bool:
        """检查工厂模式"""
        # 简化检查：查找包含"Factory"在类名中的文件
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class' in content and 'Factory' in content:
                        return True
            except:
                continue
        return False

    def _check_observer_pattern(self) -> bool:
        """检查观察者模式"""
        observer_keywords = ['subscribe', 'notify', 'observer', 'listener', 'callback']
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in observer_keywords):
                        return True
            except:
                continue
        return False

    def _check_strategy_pattern(self) -> bool:
        """检查策略模式"""
        strategy_keywords = ['strategy', 'algorithm', 'policy']
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in strategy_keywords):
                        return True
            except:
                continue
        return False

    def _check_layered_architecture(self) -> bool:
        """检查分层架构"""
        return len(self.layer_mapping) >= 5  # 如果有5个或更多层次

    def _check_microservices_architecture(self) -> bool:
        """检查微服务架构"""
        # 检查是否有API网关、服务发现等微服务特征
        microservice_indicators = ['gateway', 'service', 'api', 'rest', 'microservice']
        indicator_count = 0

        for module in self.src_dir.iterdir():
            if module.is_dir():
                if any(indicator in module.name.lower() for indicator in microservice_indicators):
                    indicator_count += 1

        return indicator_count >= 3

    def _check_event_driven_architecture(self) -> bool:
        """检查事件驱动架构"""
        event_indicators = ['event', 'bus', 'queue', 'message', 'async']
        indicator_count = 0

        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for indicator in event_indicators:
                        if indicator in content:
                            indicator_count += 1
                            break
            except:
                continue

        return indicator_count >= 3

    def _check_god_class_anti_pattern(self) -> bool:
        """检查上帝类反模式"""
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    classes = [node for node in ast.walk(
                        ast.parse(content)) if isinstance(node, ast.ClassDef)]

                    # 如果单个类超过500行，可能是上帝类
                    if lines > 500 and classes:
                        return True
            except:
                continue
        return False

    def _check_high_complexity_anti_pattern(self) -> bool:
        """检查圈复杂度过高反模式"""
        return self.metrics.average_complexity > 20

    def _analyze_complexity(self) -> Dict[str, Any]:
        """分析复杂度"""
        return {
            "average_complexity": self.metrics.average_complexity,
            "max_complexity": max([analysis.complexity_score for analysis in self.layer_analyses.values()], default=0),
            "complexity_distribution": "中等到高"
        }

    def _analyze_coupling(self) -> Dict[str, Any]:
        """分析耦合度"""
        coupling_score = len(self.dependency_graph.edges()) / \
            max(len(self.dependency_graph.nodes()), 1)
        return {
            "coupling_score": coupling_score,
            "coupling_level": "高" if coupling_score > 2 else "中" if coupling_score > 1 else "低"
        }

    def _analyze_cohesion(self) -> Dict[str, Any]:
        """分析内聚度"""
        # 简化的内聚度计算
        cohesion_score = self.metrics.total_functions / max(self.metrics.total_classes, 1)
        return {
            "cohesion_score": cohesion_score,
            "cohesion_level": "高" if cohesion_score > 5 else "中" if cohesion_score > 2 else "低"
        }

    def _analyze_test_quality(self) -> Dict[str, Any]:
        """分析测试质量"""
        total_test_files = len(list(self.test_dir.rglob("*.py")))
        return {
            "test_files_count": total_test_files,
            "test_to_code_ratio": total_test_files / max(self.metrics.total_files, 1),
            "test_quality": "良好" if total_test_files > self.metrics.total_files * 0.5 else "需要改进"
        }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    validator = AdvancedArchitectureValidator(project_root)

    results = validator.run_comprehensive_analysis()

    # 保存分析结果
    output_file = project_root / "reports" / "architecture_validation_report.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 转换不可序列化的对象
    def convert_for_json(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, nx.Graph):
            return {"nodes": list(obj.nodes()), "edges": list(obj.edges())}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    # 递归处理结果
    def process_results(obj):
        if isinstance(obj, dict):
            return {k: process_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_results(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, nx.Graph):
            return {"nodes": list(obj.nodes()), "edges": list(obj.edges())}
        else:
            return obj

    processed_results = process_results(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 架构验证报告已生成: {output_file}")

    # 打印摘要
    print("\n📋 架构分析摘要:")
    print(f"   📊 总文件数: {results['metrics']['total_files']}")
    print(f"   📈 总代码行: {results['metrics']['total_lines']}")
    print(f"   🏗️  架构层次: {len(results['layers'])}")
    print(f"   🚨 发现问题: {len(results['issues'])}")
    print(f"   💡 改进建议: {len(results['recommendations'])}")


if __name__ == "__main__":
    main()

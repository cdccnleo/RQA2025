#!/usr/bin/env python3
"""
依赖关系分析和优化工具

分析和优化各个架构层级间的依赖关系：
1. 检查依赖关系合规性
2. 识别循环依赖
3. 优化依赖倒置
4. 生成依赖关系图
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class DependencyAnalyzer:
    """依赖关系分析器"""

    def __init__(self):
        self.layers = {
            'core': 'src/core',
            'infrastructure': 'src/infrastructure',
            'data': 'src/data',
            'gateway': 'src/gateway',
            'features': 'src/features',
            'ml': 'src/ml',
            'backtest': 'src/backtest',
            'risk': 'src/risk',
            'trading': 'src/trading',
            'engine': 'src/engine'
        }

        # 定义合理的依赖关系
        self.valid_dependencies = {
            'core': [],  # 核心层不依赖其他层
            'infrastructure': ['core'],
            'data': ['infrastructure', 'core'],
            'gateway': ['infrastructure', 'core'],
            'features': ['data', 'infrastructure', 'core'],
            'ml': ['features', 'infrastructure', 'core'],
            'backtest': ['ml', 'features', 'data', 'infrastructure', 'core'],
            'risk': ['backtest', 'infrastructure', 'core'],
            'trading': ['risk', 'backtest', 'infrastructure', 'core'],
            'engine': ['trading', 'risk', 'backtest', 'ml', 'features', 'data', 'infrastructure', 'core']
        }

        self.dependencies = defaultdict(list)
        self.violations = []
        self.cycles = []

    def analyze_dependencies(self):
        """分析依赖关系"""
        print("🔍 分析架构层级依赖关系...")

        # 扫描所有Python文件中的导入语句
        for layer_name, layer_path in self.layers.items():
            layer_dir = Path(layer_path)
            if not layer_dir.exists():
                continue

            for root, dirs, files in os.walk(layer_dir):
                dirs[:] = [d for d in dirs if d not in ['__pycache__']]
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        file_deps = self.analyze_file_dependencies(file_path, layer_name)
                        for dep in file_deps:
                            self.dependencies[layer_name].append(dep)

        print(f"📋 发现 {sum(len(deps) for deps in self.dependencies.values())} 个依赖关系")

    def analyze_file_dependencies(self, file_path: Path, current_layer: str) -> List[Dict]:
        """分析文件依赖关系"""
        dependencies = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # 查找from src.xxx import语句
                from_match = re.search(r'from\s+src\.(\w+)\s+import', line)
                if from_match:
                    imported_layer = from_match.group(1)
                    if imported_layer in self.layers:
                        dependencies.append({
                            'file': str(file_path),
                            'line': i,
                            'type': 'from_import',
                            'from_layer': current_layer,
                            'to_layer': imported_layer,
                            'import_statement': line.strip()
                        })

                # 查找import src.xxx语句
                import_match = re.search(r'import\s+src\.(\w+)', line)
                if import_match:
                    imported_layer = import_match.group(1)
                    if imported_layer in self.layers:
                        dependencies.append({
                            'file': str(file_path),
                            'line': i,
                            'type': 'direct_import',
                            'from_layer': current_layer,
                            'to_layer': imported_layer,
                            'import_statement': line.strip()
                        })

        except Exception as e:
            print(f"⚠️ 无法分析文件 {file_path}: {e}")

        return dependencies

    def check_dependency_violations(self):
        """检查依赖违规"""
        print("🚫 检查依赖关系违规...")

        violations = []

        for from_layer, deps in self.dependencies.items():
            valid_targets = self.valid_dependencies.get(from_layer, [])

            for dep in deps:
                to_layer = dep['to_layer']

                if to_layer not in valid_targets:
                    violations.append({
                        'type': 'invalid_dependency',
                        'from_layer': from_layer,
                        'to_layer': to_layer,
                        'file': dep['file'],
                        'line': dep['line'],
                        'import_statement': dep['import_statement'],
                        'severity': 'high' if to_layer in ['trading', 'backtest'] else 'medium'
                    })

        self.violations = violations
        print(f"📋 发现 {len(violations)} 个依赖违规")

    def detect_cycles(self):
        """检测循环依赖"""
        print("🔄 检测循环依赖...")

        # 构建依赖图
        graph = defaultdict(list)
        for from_layer, deps in self.dependencies.items():
            for dep in deps:
                to_layer = dep['to_layer']
                if to_layer not in graph[from_layer]:
                    graph[from_layer].append(to_layer)

        # DFS检测循环
        cycles = []
        visited = set()
        path = []

        def dfs(node):
            if node in path:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for neighbor in graph[node]:
                dfs(neighbor)

            path.pop()

        # 检查所有节点
        for node in self.layers.keys():
            if node not in visited:
                dfs(node)

        self.cycles = cycles
        print(f"📋 发现 {len(cycles)} 个循环依赖")

    def generate_dependency_graph(self):
        """生成依赖关系图"""
        print("📊 生成依赖关系图...")

        graph_content = []

        # Graphviz DOT格式
        graph_content.append("digraph ArchitectureDependencies {")
        graph_content.append("    rankdir=TB;")
        graph_content.append("    node [shape=box, style=filled];")
        graph_content.append("")

        # 定义节点颜色
        colors = {
            'core': 'lightblue',
            'infrastructure': 'lightgreen',
            'data': 'lightyellow',
            'gateway': 'lightcyan',
            'features': 'lightpink',
            'ml': 'lavender',
            'backtest': 'lightsalmon',
            'risk': 'lightcoral',
            'trading': 'orange',
            'engine': 'lightgray'
        }

        # 添加节点
        for layer, color in colors.items():
            graph_content.append(
                f"    {layer} [label=\"{layer}\\n({len(self.dependencies.get(layer, []))} deps)\", fillcolor={color}];")

        graph_content.append("")

        # 添加边
        edges = set()
        for from_layer, deps in self.dependencies.items():
            for dep in deps:
                to_layer = dep['to_layer']
                edge = (from_layer, to_layer)
                if edge not in edges:
                    edges.add(edge)

                    # 检查是否为违规依赖
                    is_violation = any(v['from_layer'] == from_layer and v['to_layer'] == to_layer
                                       for v in self.violations)

                    if is_violation:
                        graph_content.append(
                            f"    {from_layer} -> {to_layer} [color=red, penwidth=2, label=\"违规\"];")
                    else:
                        graph_content.append(f"    {from_layer} -> {to_layer};")

        graph_content.append("}")

        # 保存图文件
        with open('docs/architecture/DEPENDENCY_GRAPH.dot', 'w', encoding='utf-8') as f:
            f.write('\n'.join(graph_content))

        print("✅ 依赖关系图已生成: docs/architecture/DEPENDENCY_GRAPH.dot")

    def generate_dependency_matrix(self):
        """生成依赖关系矩阵"""
        print("📈 生成依赖关系矩阵...")

        matrix_content = []

        # 表头
        layers = list(self.layers.keys())
        matrix_content.append("# 架构层级依赖关系矩阵\n")
        matrix_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        matrix_content.append("")

        # 依赖矩阵
        matrix_content.append("| From \\ To | " + " | ".join(layers) + " |")
        matrix_content.append("|" + "|".join(["-" * (len(layer) + 2)
                              for layer in ["From \\ To"] + layers]) + "|")

        for from_layer in layers:
            row = [from_layer]
            for to_layer in layers:
                # 检查是否存在依赖
                has_dep = any(dep['to_layer'] == to_layer
                              for dep in self.dependencies.get(from_layer, []))

                # 检查是否为违规
                is_violation = any(v['from_layer'] == from_layer and v['to_layer'] == to_layer
                                   for v in self.violations)

                if has_dep and is_violation:
                    row.append(" ❌ ")
                elif has_dep:
                    row.append(" ✅ ")
                else:
                    row.append(" - ")

            matrix_content.append("| " + " | ".join(row) + " |")

        matrix_content.append("")
        matrix_content.append("**说明**:")
        matrix_content.append("- ✅ : 存在有效依赖")
        matrix_content.append("- ❌ : 存在违规依赖")
        matrix_content.append("- - : 无依赖关系")

        with open('docs/architecture/DEPENDENCY_MATRIX.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(matrix_content))

        print("✅ 依赖关系矩阵已生成: docs/architecture/DEPENDENCY_MATRIX.md")

    def suggest_dependency_improvements(self):
        """建议依赖关系改进"""
        print("💡 生成依赖关系改进建议...")

        suggestions = []

        # 1. 违规依赖修复建议
        if self.violations:
            suggestions.append("## 🚨 违规依赖修复\n")
            for violation in self.violations[:10]:  # 限制数量
                suggestions.append(f"**违规**: {violation['from_layer']} → {violation['to_layer']}")
                suggestions.append(f"- 文件: {violation['file']}:{violation['line']}")
                suggestions.append(f"- 导入: `{violation['import_statement']}`")
                suggestions.append("- **建议**: 考虑使用依赖倒置或接口抽象")
                suggestions.append("")

        # 2. 循环依赖解决建议
        if self.cycles:
            suggestions.append("## 🔄 循环依赖解决\n")
            for cycle in self.cycles:
                suggestions.append(f"**循环**: {' → '.join(cycle)}")
                suggestions.append("- **建议**: 提取共同接口或使用事件驱动模式")
                suggestions.append("")

        # 3. 依赖优化建议
        suggestions.append("## ⚡ 依赖关系优化建议\n")
        suggestions.append("### 1. 依赖倒置原则\n")
        suggestions.append("- 上层模块不应依赖下层模块，应依赖抽象\n")
        suggestions.append("- 具体实现应依赖抽象接口\n")
        suggestions.append("")

        suggestions.append("### 2. 接口分离原则\n")
        suggestions.append("- 为每个模块提供专门的接口\n")
        suggestions.append("- 避免大型通用接口\n")
        suggestions.append("")

        suggestions.append("### 3. 依赖注入\n")
        suggestions.append("- 使用依赖注入替代直接依赖\n")
        suggestions.append("- 通过构造函数或setter注入依赖\n")
        suggestions.append("")

        with open('docs/architecture/DEPENDENCY_IMPROVEMENTS.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(suggestions))

        print("✅ 依赖关系改进建议已生成: docs/architecture/DEPENDENCY_IMPROVEMENTS.md")

    def generate_comprehensive_report(self):
        """生成综合依赖分析报告"""
        report = []

        report.append("# 架构依赖关系分析报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## 📊 分析概览")
        report.append("")
        report.append("### 依赖统计")
        report.append(f"- **总依赖关系**: {sum(len(deps) for deps in self.dependencies.values())}")
        report.append(f"- **违规依赖**: {len(self.violations)}")
        report.append(f"- **循环依赖**: {len(self.cycles)}")
        report.append("")

        # 各层级依赖统计
        report.append("### 各层级依赖统计")
        for layer in self.layers.keys():
            dep_count = len(self.dependencies.get(layer, []))
            violation_count = len([v for v in self.violations if v['from_layer'] == layer])
            report.append(f"- **{layer}**: {dep_count} 个依赖, {violation_count} 个违规")
        report.append("")

        # 严重违规
        if self.violations:
            report.append("## 🚨 严重违规 (需要立即处理)")
            high_severity = [v for v in self.violations if v['severity'] == 'high']
            for violation in high_severity[:5]:  # 显示前5个
                report.append(f"- **{violation['from_layer']} → {violation['to_layer']}**")
                report.append(f"  - 文件: {violation['file']}:{violation['line']}")
                report.append(f"  - 导入: `{violation['import_statement']}`")
                report.append("")

        # 循环依赖
        if self.cycles:
            report.append("## 🔄 循环依赖 (需要重点关注)")
            for cycle in self.cycles:
                report.append(f"- **循环路径**: {' → '.join(cycle)}")
            report.append("")

        report.append("## 📋 改进措施")
        report.append("")
        report.append("### 短期措施 (1周内)")
        report.append("1. **修复高严重度违规依赖**")
        report.append("2. **解决循环依赖问题**")
        report.append("3. **建立依赖关系监控**")
        report.append("")

        report.append("### 中期措施 (2周内)")
        report.append("1. **实施依赖倒置原则**")
        report.append("2. **优化接口设计**")
        report.append("3. **完善依赖注入机制**")
        report.append("")

        report.append("### 长期措施 (1月内)")
        report.append("1. **建立架构治理流程**")
        report.append("2. **完善自动化检查**")
        report.append("3. **加强团队培训**")
        report.append("")

        report.append("## 📊 质量指标")
        report.append("")
        total_deps = sum(len(deps) for deps in self.dependencies.values())
        if total_deps > 0:
            compliance_rate = (total_deps - len(self.violations)) / total_deps * 100
            report.append(f"- **依赖合规率**: {compliance_rate:.1f}%")
        else:
            report.append("- **依赖合规率**: N/A")
        report.append(f"- **循环依赖数**: {len(self.cycles)}")
        report.append("")

        with open('reports/DEPENDENCY_ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    def run_analysis(self):
        """运行依赖关系分析"""
        print("🚀 开始架构依赖关系分析...")
        print("="*60)

        try:
            # 1. 分析依赖关系
            self.analyze_dependencies()

            # 2. 检查依赖违规
            self.check_dependency_violations()

            # 3. 检测循环依赖
            self.detect_cycles()

            # 4. 生成依赖关系图
            self.generate_dependency_graph()

            # 5. 生成依赖关系矩阵
            self.generate_dependency_matrix()

            # 6. 生成改进建议
            self.suggest_dependency_improvements()

            # 7. 生成综合报告
            self.generate_comprehensive_report()

            print("\n📋 依赖分析报告已生成:")
            print("   - docs/architecture/DEPENDENCY_GRAPH.dot")
            print("   - docs/architecture/DEPENDENCY_MATRIX.md")
            print("   - docs/architecture/DEPENDENCY_IMPROVEMENTS.md")
            print("   - reports/DEPENDENCY_ANALYSIS_REPORT.md")
            print("🎉 架构依赖关系分析完成！")
            return True

        except Exception as e:
            print(f"\n❌ 分析过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    analyzer = DependencyAnalyzer()
    success = analyzer.run_analysis()

    if success:
        print("\n" + "="*60)
        print("架构依赖关系分析成功完成！")
        print("✅ 依赖关系已分析")
        print("✅ 违规问题已识别")
        print("✅ 改进建议已生成")
        print("="*60)
    else:
        print("\n❌ 架构依赖关系分析失败！")


if __name__ == "__main__":
    main()

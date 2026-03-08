#!/usr/bin/env python3
"""
RQA2025 架构一致性检查工具

功能:
1. 检查架构层级是否符合规范
2. 检查模块依赖关系是否合规
3. 检查代码是否符合架构标准
4. 生成架构健康度报告

作者: AI Assistant
创建日期: 2026-03-08
版本: 1.0.0
"""

import os
import sys
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ArchitectureViolation:
    """架构违规记录"""
    type: str
    file: str
    line: int
    message: str
    severity: str  # critical, high, medium, low


@dataclass
class ArchitectureMetrics:
    """架构度量指标"""
    total_files: int
    total_violations: int
    layer_count: int
    module_coupling: float
    code_duplication_rate: float
    documentation_coverage: float
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]


class ArchitectureChecker:
    """架构一致性检查器"""

    # 架构层级定义
    LAYERS = {
        '核心业务层': ['strategy', 'trading', 'risk', 'features'],
        '核心支撑层': ['data', 'ml', 'infrastructure', 'streaming'],
        '辅助支撑层': ['core', 'monitoring', 'optimization', 'gateway', 'adapters']
    }

    # 允许的依赖关系 (上层 -> 下层)
    ALLOWED_DEPENDENCIES = {
        'strategy': ['data', 'ml', 'infrastructure', 'core'],
        'trading': ['data', 'infrastructure', 'core', 'risk'],
        'risk': ['data', 'infrastructure', 'core'],
        'features': ['data', 'infrastructure', 'core'],
        'data': ['infrastructure', 'core'],
        'ml': ['data', 'infrastructure', 'core'],
        'infrastructure': ['core'],
        'core': [],
        'monitoring': ['core', 'infrastructure'],
        'optimization': ['core', 'infrastructure'],
        'gateway': ['core', 'infrastructure'],
        'adapters': ['core', 'infrastructure']
    }

    # 禁止的导入模式
    FORBIDDEN_IMPORTS = [
        'from src.resilience',
        'from src.utils',
        'from src.automation',
        'import src.resilience',
        'import src.utils',
        'import src.automation'
    ]

    def __init__(self, project_root: str = None):
        """初始化检查器"""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.src_path = self.project_root / 'src'
        self.violations: List[ArchitectureViolation] = []
        self.metrics = ArchitectureMetrics(
            total_files=0,
            total_violations=0,
            layer_count=0,
            module_coupling=0.0,
            code_duplication_rate=0.0,
            documentation_coverage=0.0,
            violations_by_type={},
            violations_by_severity={}
        )

    def check_all(self) -> ArchitectureMetrics:
        """执行所有检查"""
        print("🔍 开始架构一致性检查...")
        print("=" * 60)

        # 1. 检查架构层级
        print("\n📊 检查架构层级...")
        self._check_layer_structure()

        # 2. 检查模块依赖
        print("\n🔗 检查模块依赖关系...")
        self._check_module_dependencies()

        # 3. 检查禁止的导入
        print("\n🚫 检查禁止的导入语句...")
        self._check_forbidden_imports()

        # 4. 检查代码规范
        print("\n📝 检查代码规范...")
        self._check_code_standards()

        # 5. 计算度量指标
        print("\n📈 计算架构度量指标...")
        self._calculate_metrics()

        print("\n" + "=" * 60)
        print("✅ 架构一致性检查完成")

        return self.metrics

    def _check_layer_structure(self):
        """检查架构层级结构"""
        if not self.src_path.exists():
            self._add_violation(
                "layer_structure",
                str(self.project_root),
                0,
                f"src目录不存在: {self.src_path}",
                "critical"
            )
            return

        # 检查是否有过时的层级目录
        obsolete_layers = ['resilience', 'utils', 'automation']
        for layer in obsolete_layers:
            layer_path = self.src_path / layer
            if layer_path.exists():
                self._add_violation(
                    "layer_structure",
                    str(layer_path),
                    0,
                    f"发现过时的层级目录: {layer}，应该已合并到其他层级",
                    "high"
                )

        # 检查核心层级是否存在
        required_layers = ['core', 'infrastructure', 'data', 'strategy']
        for layer in required_layers:
            layer_path = self.src_path / layer
            if not layer_path.exists():
                self._add_violation(
                    "layer_structure",
                    str(self.src_path),
                    0,
                    f"缺少核心层级目录: {layer}",
                    "critical"
                )

        # 统计层级数量
        actual_layers = [d.name for d in self.src_path.iterdir() if d.is_dir()]
        self.metrics.layer_count = len(actual_layers)

        if self.metrics.layer_count > 13:
            self._add_violation(
                "layer_structure",
                str(self.src_path),
                0,
                f"架构层级过多: {self.metrics.layer_count}层，建议不超过13层",
                "medium"
            )

    def _check_module_dependencies(self):
        """检查模块依赖关系"""
        python_files = list(self.src_path.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析导入语句
                imports = self._parse_imports(content)

                # 检查依赖关系
                current_module = self._get_module_name(file_path)
                if current_module not in self.ALLOWED_DEPENDENCIES:
                    continue

                allowed = self.ALLOWED_DEPENDENCIES[current_module]

                for imp in imports:
                    if imp.startswith('src.'):
                        target_module = imp.split('.')[1]
                        if target_module not in allowed and target_module != current_module:
                            self._add_violation(
                                "module_dependency",
                                str(file_path),
                                0,
                                f"违规依赖: {current_module} -> {target_module}",
                                "high"
                            )

            except Exception as e:
                print(f"⚠️  检查文件失败 {file_path}: {e}")

    def _check_forbidden_imports(self):
        """检查禁止的导入语句"""
        python_files = list(self.src_path.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    for forbidden in self.FORBIDDEN_IMPORTS:
                        if forbidden in line:
                            self._add_violation(
                                "forbidden_import",
                                str(file_path),
                                line_num,
                                f"发现禁止的导入: {line.strip()}",
                                "high"
                            )

            except Exception as e:
                print(f"⚠️  检查文件失败 {file_path}: {e}")

    def _check_code_standards(self):
        """检查代码规范"""
        python_files = list(self.src_path.rglob("*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查文件是否为空
                if not content.strip():
                    self._add_violation(
                        "code_standard",
                        str(file_path),
                        0,
                        "空文件",
                        "low"
                    )
                    continue

                # 检查是否有文档字符串
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    # 跳过 __init__.py 文件
                    if file_path.name != '__init__.py':
                        self._add_violation(
                            "code_standard",
                            str(file_path),
                            1,
                            "缺少模块文档字符串",
                            "low"
                        )

                # 检查类是否有文档字符串
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if not ast.get_docstring(node):
                                self._add_violation(
                                    "code_standard",
                                    str(file_path),
                                    node.lineno,
                                    f"类 {node.name} 缺少文档字符串",
                                    "low"
                                )
                except SyntaxError:
                    pass

            except Exception as e:
                print(f"⚠️  检查文件失败 {file_path}: {e}")

    def _parse_imports(self, content: str) -> List[str]:
        """解析导入语句"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            pass
        return imports

    def _get_module_name(self, file_path: Path) -> str:
        """获取模块名"""
        relative_path = file_path.relative_to(self.src_path)
        return relative_path.parts[0] if relative_path.parts else ''

    def _add_violation(self, type_: str, file: str, line: int, message: str, severity: str):
        """添加违规记录"""
        violation = ArchitectureViolation(
            type=type_,
            file=file,
            line=line,
            message=message,
            severity=severity
        )
        self.violations.append(violation)

    def _calculate_metrics(self):
        """计算度量指标"""
        python_files = list(self.src_path.rglob("*.py"))
        self.metrics.total_files = len(python_files)
        self.metrics.total_violations = len(self.violations)

        # 按类型统计
        for v in self.violations:
            self.metrics.violations_by_type[v.type] = \
                self.metrics.violations_by_type.get(v.type, 0) + 1
            self.metrics.violations_by_severity[v.severity] = \
                self.metrics.violations_by_severity.get(v.severity, 0) + 1

        # 计算模块耦合度（简化计算）
        if self.metrics.total_files > 0:
            self.metrics.module_coupling = min(
                len(self.violations) / self.metrics.total_files, 1.0
            )

        # 估算文档覆盖率
        documented_files = 0
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    documented_files += 1
            except:
                pass

        if self.metrics.total_files > 0:
            self.metrics.documentation_coverage = documented_files / self.metrics.total_files

    def generate_report(self, output_path: str = None) -> str:
        """生成检查报告"""
        if output_path is None:
            output_path = self.project_root / 'reports' / 'architecture_consistency_report.json'
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'metrics': asdict(self.metrics),
            'violations': [asdict(v) for v in self.violations],
            'summary': {
                'status': 'PASS' if self.metrics.total_violations == 0 else 'FAIL',
                'critical_issues': self.metrics.violations_by_severity.get('critical', 0),
                'high_issues': self.metrics.violations_by_severity.get('high', 0),
                'medium_issues': self.metrics.violations_by_severity.get('medium', 0),
                'low_issues': self.metrics.violations_by_severity.get('low', 0)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "=" * 60)
        print("📊 架构一致性检查摘要")
        print("=" * 60)

        print(f"\n📁 总文件数: {self.metrics.total_files}")
        print(f"📊 架构层级数: {self.metrics.layer_count}")
        print(f"⚠️  违规总数: {self.metrics.total_violations}")
        print(f"🔗 模块耦合度: {self.metrics.module_coupling:.2%}")
        print(f"📝 文档覆盖率: {self.metrics.documentation_coverage:.2%}")

        print("\n📈 违规分布（按严重程度）:")
        for severity, count in sorted(self.metrics.violations_by_severity.items()):
            emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
            print(f"  {emoji} {severity.upper()}: {count}")

        print("\n📈 违规分布（按类型）:")
        for type_, count in sorted(self.metrics.violations_by_type.items()):
            print(f"  • {type_}: {count}")

        if self.violations:
            print("\n🔍 详细违规列表（前10条）:")
            for i, v in enumerate(self.violations[:10], 1):
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(v.severity, '⚪')
                print(f"\n  {i}. {emoji} [{v.severity.upper()}] {v.type}")
                print(f"     文件: {v.file}")
                print(f"     位置: 第{v.line}行")
                print(f"     描述: {v.message}")

        print("\n" + "=" * 60)
        if self.metrics.total_violations == 0:
            print("✅ 检查通过：未发现架构违规")
        else:
            print(f"⚠️  检查失败：发现 {self.metrics.total_violations} 个架构违规")
        print("=" * 60)


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 创建检查器
    checker = ArchitectureChecker(project_root)

    # 执行检查
    metrics = checker.check_all()

    # 打印摘要
    checker.print_summary()

    # 生成报告
    report_path = checker.generate_report()
    print(f"\n📄 详细报告已保存: {report_path}")

    # 返回退出码
    return 0 if metrics.total_violations == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

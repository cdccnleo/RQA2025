#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层模块审查脚本
全面检查基础设施层的架构、依赖关系和代码质量
"""

import re
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict
import argparse


class InfrastructureModuleAuditor:
    """基础设施层模块审查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.results = {
            'module_structure': {},
            'dependency_analysis': {},
            'code_quality': {},
            'architecture_compliance': {},
            'security_analysis': {}
        }

        # 定义基础设施层标准模块
        self.standard_modules = {
            'logging': '日志管理模块',
            'config': '配置管理模块',
            'database': '数据库管理模块',
            'cache': '缓存管理模块',
            'messaging': '消息队列模块',
            'monitoring': '监控管理模块',
            'security': '安全管理模块',
            'utils': '工具函数模块'
        }

        # 禁止的依赖模式
        self.forbidden_dependencies = [
            r'from src\.engine\.',
            r'import src\.engine\.',
            r'from src\.trading\.',
            r'import src\.trading\.',
            r'from src\.risk\.',
            r'import src\.risk\.',
        ]

    def analyze_module_structure(self) -> Dict:
        """分析模块结构"""
        print("🔍 分析模块结构...")
        structure = {
            'modules': {},
            'missing_modules': [],
            'extra_modules': [],
            'total_files': 0,
            'total_lines': 0
        }

        if not self.infrastructure_dir.exists():
            print("  ❌ 基础设施层目录不存在")
            return structure

        # 分析现有模块
        for module_dir in self.infrastructure_dir.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('_'):
                module_name = module_dir.name
                module_info = {
                    'name': module_name,
                    'description': self.standard_modules.get(module_name, '自定义模块'),
                    'files': [],
                    'lines': 0,
                    'classes': 0,
                    'functions': 0
                }

                # 统计文件信息
                for py_file in module_dir.rglob("*.py"):
                    if py_file.name != "__pycache__":
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                lines = len(content.split('\n'))

                                # 解析AST
                                try:
                                    tree = ast.parse(content)
                                    classes = len([node for node in ast.walk(tree)
                                                  if isinstance(node, ast.ClassDef)])
                                    functions = len([node for node in ast.walk(
                                        tree) if isinstance(node, ast.FunctionDef)])

                                    module_info['classes'] += classes
                                    module_info['functions'] += functions
                                except:
                                    pass

                                module_info['files'].append({
                                    'name': py_file.name,
                                    'path': str(py_file.relative_to(self.infrastructure_dir)),
                                    'lines': lines
                                })
                                module_info['lines'] += lines
                                structure['total_files'] += 1
                                structure['total_lines'] += lines
                        except Exception as e:
                            print(f"  读取文件失败 {py_file}: {str(e)}")

                structure['modules'][module_name] = module_info

        # 检查缺失的标准模块
        for module_name in self.standard_modules:
            if module_name not in structure['modules']:
                structure['missing_modules'].append(module_name)

        # 检查额外的模块
        for module_name in structure['modules']:
            if module_name not in self.standard_modules:
                structure['extra_modules'].append(module_name)

        return structure

    def analyze_dependencies(self) -> Dict:
        """分析依赖关系"""
        print("🔍 分析依赖关系...")
        dependencies = {
            'internal_deps': {},
            'external_deps': {},
            'forbidden_deps': [],
            'circular_deps': [],
            'dependency_graph': {}
        }

        if not self.infrastructure_dir.exists():
            return dependencies

        # 分析每个模块的依赖
        for module_dir in self.infrastructure_dir.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('_'):
                module_name = module_dir.name
                module_deps = {
                    'imports': [],
                    'from_imports': [],
                    'forbidden_imports': []
                }

                for py_file in module_dir.rglob("*.py"):
                    if py_file.name != "__pycache__":
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read()

                            # 分析导入语句
                            import_pattern = r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
                            from_pattern = r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'

                            for match in re.finditer(import_pattern, content):
                                module_deps['imports'].append(match.group(1))

                            for match in re.finditer(from_pattern, content):
                                module_deps['from_imports'].append(match.group(1))

                            # 检查禁止的依赖
                            for pattern in self.forbidden_dependencies:
                                if re.search(pattern, content):
                                    module_deps['forbidden_imports'].append(pattern)
                                    dependencies['forbidden_deps'].append({
                                        'file': str(py_file),
                                        'pattern': pattern
                                    })

                        except Exception as e:
                            print(f"  分析依赖失败 {py_file}: {str(e)}")

                dependencies['internal_deps'][module_name] = module_deps

        return dependencies

    def analyze_code_quality(self) -> Dict:
        """分析代码质量"""
        print("🔍 分析代码质量...")
        quality = {
            'files_analyzed': 0,
            'total_issues': 0,
            'issues_by_type': {},
            'complexity_analysis': {},
            'documentation_coverage': {}
        }

        if not self.infrastructure_dir.exists():
            return quality

        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name != "__pycache__":
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    quality['files_analyzed'] += 1
                    file_issues = []

                    # 检查编码声明
                    if not content.startswith('# -*- coding: utf-8 -*-'):
                        file_issues.append('缺少编码声明')

                    # 检查文档字符串
                    lines = content.split('\n')
                    has_docstring = False
                    for line in lines[:10]:  # 检查前10行
                        if '"""' in line or "'''" in line:
                            has_docstring = True
                            break

                    if not has_docstring:
                        file_issues.append('缺少模块文档字符串')

                    # 检查行长度
                    long_lines = 0
                    for line in lines:
                        if len(line) > 120:
                            long_lines += 1

                    if long_lines > 0:
                        file_issues.append(f'有{long_lines}行超过120字符')

                    # 检查空行
                    empty_lines = sum(1 for line in lines if line.strip() == '')
                    if empty_lines > len(lines) * 0.3:  # 空行超过30%
                        file_issues.append('空行过多')

                    # 统计问题
                    for issue in file_issues:
                        if issue not in quality['issues_by_type']:
                            quality['issues_by_type'][issue] = 0
                        quality['issues_by_type'][issue] += 1
                        quality['total_issues'] += 1

                    # 复杂度分析
                    try:
                        tree = ast.parse(content)
                        complexity = self._calculate_complexity(tree)
                        quality['complexity_analysis'][str(py_file)] = complexity
                    except:
                        pass

                except Exception as e:
                    print(f"  分析代码质量失败 {py_file}: {str(e)}")

        return quality

    def _calculate_complexity(self, tree: ast.AST) -> Dict:
        """计算代码复杂度"""
        complexity = {
            'cyclomatic': 0,
            'depth': 0,
            'functions': 0,
            'classes': 0
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity['cyclomatic'] += 1
            elif isinstance(node, ast.FunctionDef):
                complexity['functions'] += 1
            elif isinstance(node, ast.ClassDef):
                complexity['classes'] += 1

        return complexity

    def check_architecture_compliance(self) -> Dict:
        """检查架构合规性"""
        print("🔍 检查架构合规性...")
        compliance = {
            'layer_isolation': True,
            'dependency_direction': True,
            'interface_compliance': True,
            'naming_conventions': True,
            'issues': []
        }

        if not self.infrastructure_dir.exists():
            compliance['layer_isolation'] = False
            compliance['issues'].append('基础设施层目录不存在')
            return compliance

        # 检查层隔离
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name != "__pycache__":
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查是否引用了上层模块
                    if re.search(r'from src\.(engine|trading|risk)', content):
                        compliance['layer_isolation'] = False
                        compliance['issues'].append(f'{py_file} 引用了上层模块')

                    # 检查命名规范
                    if not re.match(r'^[a-z_][a-z0-9_]*\.py$', py_file.name):
                        compliance['naming_conventions'] = False
                        compliance['issues'].append(f'{py_file} 命名不符合规范')

                except Exception as e:
                    compliance['issues'].append(f'检查文件失败 {py_file}: {str(e)}')

        return compliance

    def analyze_security(self) -> Dict:
        """分析安全风险"""
        print("🔍 分析安全风险...")
        security = {
            'vulnerabilities': [],
            'hardcoded_secrets': [],
            'insecure_patterns': [],
            'recommendations': []
        }

        if not self.infrastructure_dir.exists():
            return security

        # 安全模式检查
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', '硬编码密码'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', '硬编码API密钥'),
            (r'secret\s*=\s*["\'][^"\']+["\']', '硬编码密钥'),
            (r'eval\s*\(', '使用eval函数'),
            (r'exec\s*\(', '使用exec函数'),
            (r'os\.system\s*\(', '使用os.system'),
            (r'subprocess\.call\s*\(', '使用subprocess.call'),
        ]

        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name != "__pycache__":
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for pattern, description in security_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security['vulnerabilities'].append({
                                'file': str(py_file),
                                'pattern': pattern,
                                'description': description
                            })

                except Exception as e:
                    print(f"  安全分析失败 {py_file}: {str(e)}")

        # 生成安全建议
        if security['vulnerabilities']:
            security['recommendations'].append('移除所有硬编码的敏感信息')
            security['recommendations'].append('使用环境变量或配置文件存储敏感信息')
            security['recommendations'].append('避免使用eval、exec等危险函数')

        return security

    def run_audit(self) -> Dict:
        """运行完整审查"""
        print("🚀 开始基础设施层模块审查...")
        print("=" * 50)

        self.results['module_structure'] = self.analyze_module_structure()
        print()

        self.results['dependency_analysis'] = self.analyze_dependencies()
        print()

        self.results['code_quality'] = self.analyze_code_quality()
        print()

        self.results['architecture_compliance'] = self.check_architecture_compliance()
        print()

        self.results['security_analysis'] = self.analyze_security()
        print()

        return self.results

    def generate_report(self) -> str:
        """生成审查报告"""
        report = []
        report.append("# 基础设施层模块审查报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 模块结构分析
        structure = self.results['module_structure']
        report.append("## 模块结构分析")
        report.append(f"- 总文件数: {structure['total_files']}")
        report.append(f"- 总代码行数: {structure['total_lines']}")
        report.append(f"- 现有模块数: {len(structure['modules'])}")
        report.append("")

        if structure['modules']:
            report.append("### 现有模块")
            for module_name, module_info in structure['modules'].items():
                report.append(f"#### {module_name}")
                report.append(f"- 描述: {module_info['description']}")
                report.append(f"- 文件数: {len(module_info['files'])}")
                report.append(f"- 代码行数: {module_info['lines']}")
                report.append(f"- 类数: {module_info['classes']}")
                report.append(f"- 函数数: {module_info['functions']}")
                report.append("")

        if structure['missing_modules']:
            report.append("### 缺失的标准模块")
            for module_name in structure['missing_modules']:
                report.append(f"- {module_name}: {self.standard_modules[module_name]}")
            report.append("")

        if structure['extra_modules']:
            report.append("### 额外的模块")
            for module_name in structure['extra_modules']:
                report.append(f"- {module_name}")
            report.append("")

        # 依赖分析
        deps = self.results['dependency_analysis']
        report.append("## 依赖关系分析")

        if deps['forbidden_deps']:
            report.append("### 禁止的依赖")
            for dep in deps['forbidden_deps']:
                report.append(f"- {dep['file']}: {dep['pattern']}")
            report.append("")

        # 代码质量分析
        quality = self.results['code_quality']
        report.append("## 代码质量分析")
        report.append(f"- 分析文件数: {quality['files_analyzed']}")
        report.append(f"- 总问题数: {quality['total_issues']}")
        report.append("")

        if quality['issues_by_type']:
            report.append("### 问题类型统计")
            for issue_type, count in quality['issues_by_type'].items():
                report.append(f"- {issue_type}: {count} 个")
            report.append("")

        # 架构合规性
        compliance = self.results['architecture_compliance']
        report.append("## 架构合规性")
        report.append(f"- 层隔离: {'✅ 通过' if compliance['layer_isolation'] else '❌ 失败'}")
        report.append(f"- 依赖方向: {'✅ 通过' if compliance['dependency_direction'] else '❌ 失败'}")
        report.append(f"- 接口合规: {'✅ 通过' if compliance['interface_compliance'] else '❌ 失败'}")
        report.append(f"- 命名规范: {'✅ 通过' if compliance['naming_conventions'] else '❌ 失败'}")
        report.append("")

        if compliance['issues']:
            report.append("### 合规性问题")
            for issue in compliance['issues']:
                report.append(f"- {issue}")
            report.append("")

        # 安全分析
        security = self.results['security_analysis']
        report.append("## 安全风险分析")
        report.append(f"- 发现漏洞: {len(security['vulnerabilities'])} 个")
        report.append(f"- 硬编码密钥: {len(security['hardcoded_secrets'])} 个")
        report.append(f"- 不安全模式: {len(security['insecure_patterns'])} 个")
        report.append("")

        if security['vulnerabilities']:
            report.append("### 安全漏洞")
            for vuln in security['vulnerabilities']:
                report.append(f"- {vuln['file']}: {vuln['description']}")
            report.append("")

        if security['recommendations']:
            report.append("### 安全建议")
            for rec in security['recommendations']:
                report.append(f"- {rec}")
            report.append("")

        # 总体建议
        report.append("## 总体建议")
        if not structure['missing_modules'] and not deps['forbidden_deps'] and compliance['layer_isolation']:
            report.append("✅ 基础设施层架构良好，符合设计规范")
        else:
            report.append("❌ 发现架构问题，需要改进")

        report.append("1. 完善缺失的标准模块")
        report.append("2. 移除禁止的依赖关系")
        report.append("3. 修复代码质量问题")
        report.append("4. 解决安全风险")
        report.append("5. 确保架构合规性")

        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="审查基础设施层模块")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--report", help="输出报告文件路径")

    args = parser.parse_args()

    # 创建审查器
    auditor = InfrastructureModuleAuditor(args.project_root)

    # 运行审查
    results = auditor.run_audit()

    # 生成报告
    report = auditor.generate_report()

    # 输出报告
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 报告已保存到: {args.report}")
    else:
        print("\n" + "="*50)
        print(report)

    # 统计结果
    structure = results['module_structure']
    deps = results['dependency_analysis']
    compliance = results['architecture_compliance']

    print(f"\n✅ 审查完成！")
    print(f"📊 统计信息:")
    print(f"  - 模块数: {len(structure['modules'])}")
    print(f"  - 文件数: {structure['total_files']}")
    print(f"  - 代码行数: {structure['total_lines']}")
    print(f"  - 禁止依赖: {len(deps['forbidden_deps'])}")
    print(f"  - 架构合规: {'是' if compliance['layer_isolation'] else '否'}")


if __name__ == "__main__":
    main()

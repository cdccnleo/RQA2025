#!/usr/bin/env python3
"""
RQA2025 代码与文档一致性检查工具

提供自动化的一致性检查功能，包括：
- 代码与文档接口一致性检查
- 架构描述准确性验证
- 实现与设计文档同步状态检查
- 定期检查报告生成
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import ast

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """代码与文档一致性检查器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.docs_dir = self.project_root / "docs" / "architecture"
        self.src_dir = self.project_root / "src"
        self.reports_dir = self.project_root / "reports" / "technical" / "consistency"

        # 创建报告目录
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 初始化检查规则
        self.check_rules = {
            'interface_consistency': self._check_interface_consistency,
            'architecture_alignment': self._check_architecture_alignment,
            'implementation_completeness': self._check_implementation_completeness,
            'documentation_accuracy': self._check_documentation_accuracy,
            'version_synchronization': self._check_version_synchronization
        }

    def run_full_check(self) -> Dict[str, Any]:
        """运行完整的一致性检查"""
        logger.info("开始运行完整一致性检查...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {},
            'recommendations': []
        }

        # 运行所有检查
        for check_name, check_func in self.check_rules.items():
            logger.info(f"执行检查: {check_name}")
            try:
                results['checks'][check_name] = check_func()
            except Exception as e:
                logger.error(f"检查 {check_name} 失败: {e}")
                results['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

        # 生成摘要
        results['summary'] = self._generate_summary(results['checks'])

        # 生成推荐建议
        results['recommendations'] = self._generate_recommendations(results['checks'])

        # 保存检查结果
        self._save_results(results)

        logger.info("一致性检查完成")
        return results

    def run_quick_check(self, focus_areas: List[str] = None) -> Dict[str, Any]:
        """运行快速检查"""
        focus_areas = focus_areas or ['interface_consistency', 'architecture_alignment']

        logger.info(f"开始运行快速检查，关注领域: {focus_areas}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {},
            'quick_mode': True
        }

        # 只运行指定的检查
        for area in focus_areas:
            if area in self.check_rules:
                logger.info(f"执行快速检查: {area}")
                try:
                    results['checks'][area] = self.check_rules[area]()
                except Exception as e:
                    logger.error(f"快速检查 {area} 失败: {e}")
                    results['checks'][area] = {
                        'status': 'error',
                        'error': str(e)
                    }

        results['summary'] = self._generate_summary(results['checks'])
        self._save_results(results, prefix="quick_check")

        return results

    def _check_interface_consistency(self) -> Dict[str, Any]:
        """检查接口一致性"""
        result = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }

        # 检查各层适配器接口
        layers = ['data', 'features', 'models']
        for layer in layers:
            try:
                adapter_file = self.src_dir / "core" / "integration" / f"{layer}_adapter.py"
                if adapter_file.exists():
                    interfaces = self._extract_interfaces_from_file(adapter_file)
                    result['metrics'][f'{layer}_interfaces'] = len(interfaces)

                    # 检查文档中是否描述了这些接口
                    doc_file = self.docs_dir / f"{layer}_layer_architecture_design.md"
                    if doc_file.exists():
                        doc_content = doc_file.read_text(encoding='utf-8')
                        missing_interfaces = []
                        for interface in interfaces:
                            if interface not in doc_content:
                                missing_interfaces.append(interface)

                        if missing_interfaces:
                            result['issues'].append({
                                'type': 'missing_interface_doc',
                                'layer': layer,
                                'missing_interfaces': missing_interfaces
                            })
                            result['status'] = 'warning'

            except Exception as e:
                result['issues'].append({
                    'type': 'check_error',
                    'layer': layer,
                    'error': str(e)
                })
                result['status'] = 'error'

        return result

    def _check_architecture_alignment(self) -> Dict[str, Any]:
        """检查架构对齐"""
        result = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }

        # 检查架构文档与代码结构的对齐
        architecture_files = list(self.docs_dir.glob("*architecture*.md"))

        for doc_file in architecture_files:
            layer_name = doc_file.stem.split('_')[0]  # 提取层名称
            src_layer_dir = self.src_dir / layer_name

            if src_layer_dir.exists():
                # 检查文档中描述的组件是否在代码中存在
                doc_content = doc_file.read_text(encoding='utf-8')
                components = self._extract_components_from_doc(doc_content)

                missing_components = []
                for component in components:
                    component_path = src_layer_dir / f"{component}.py"
                    if not component_path.exists():
                        # 检查子目录
                        found = False
                        for sub_dir in src_layer_dir.rglob("*.py"):
                            if component in sub_dir.name:
                                found = True
                                break
                        if not found:
                            missing_components.append(component)

                if missing_components:
                    result['issues'].append({
                        'type': 'missing_component',
                        'layer': layer_name,
                        'document': doc_file.name,
                        'missing_components': missing_components
                    })
                    result['status'] = 'warning'

        return result

    def _check_implementation_completeness(self) -> Dict[str, Any]:
        """检查实现完整性"""
        result = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }

        # 检查Phase标记的功能实现情况
        phase_patterns = [
            r'Phase \d+',
            r'⭐.*Phase \d+',
            r'🆕.*Phase \d+'
        ]

        # 扫描所有架构文档
        for doc_file in self.docs_dir.glob("*architecture*.md"):
            doc_content = doc_file.read_text(encoding='utf-8')

            for pattern in phase_patterns:
                matches = re.findall(pattern, doc_content)
                for match in matches:
                    # 提取Phase版本
                    phase_match = re.search(r'Phase (\d+)', match)
                    if phase_match:
                        phase_num = int(phase_match.group(1))

                        # 检查对应的实现文件是否存在
                        layer_name = doc_file.stem.split('_')[0]
                        implementation_files = list((self.src_dir / layer_name).rglob("*.py"))

                        # 简单的启发式检查：查找包含phase相关注释的文件
                        phase_implemented = False
                        for impl_file in implementation_files:
                            try:
                                content = impl_file.read_text(encoding='utf-8')
                                if f'Phase {phase_num}' in content or f'phase {phase_num}' in content:
                                    phase_implemented = True
                                    break
                            except:
                                continue

                        if not phase_implemented:
                            result['issues'].append({
                                'type': 'phase_not_implemented',
                                'document': doc_file.name,
                                'phase': phase_num,
                                'description': match.strip()
                            })
                            result['status'] = 'warning'

        return result

    def _check_documentation_accuracy(self) -> Dict[str, Any]:
        """检查文档准确性"""
        result = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }

        # 检查文档中的代码示例是否与实际代码匹配
        for doc_file in self.docs_dir.glob("*architecture*.md"):
            doc_content = doc_file.read_text(encoding='utf-8')

            # 提取代码块
            code_blocks = re.findall(r'```python\s*(.*?)\s*```', doc_content, re.DOTALL)

            for i, code_block in enumerate(code_blocks):
                # 简单的语法检查
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    result['issues'].append({
                        'type': 'syntax_error_in_doc',
                        'document': doc_file.name,
                        'code_block_index': i,
                        'error': str(e)
                    })
                    result['status'] = 'warning'

                # 检查导入语句是否有效
                import_lines = [line for line in code_block.split('\n') if line.strip(
                ).startswith('from ') or line.strip().startswith('import ')]
                for import_line in import_lines:
                    if not self._validate_import_statement(import_line):
                        result['issues'].append({
                            'type': 'invalid_import_in_doc',
                            'document': doc_file.name,
                            'code_block_index': i,
                            'import_line': import_line.strip()
                        })
                        result['status'] = 'warning'

        return result

    def _check_version_synchronization(self) -> Dict[str, Any]:
        """检查版本同步"""
        result = {
            'status': 'passed',
            'issues': [],
            'metrics': {}
        }

        # 检查文档版本与代码版本的一致性
        version_pattern = r'v(\d+)\.(\d+)\.(\d+)'

        for doc_file in self.docs_dir.glob("*architecture*.md"):
            doc_content = doc_file.read_text(encoding='utf-8')

            # 查找文档版本
            doc_version_match = re.search(version_pattern, doc_content)
            if doc_version_match:
                doc_version = doc_version_match.group(0)

                # 检查对应的代码文件是否有版本信息
                layer_name = doc_file.stem.split('_')[0]
                init_file = self.src_dir / layer_name / "__init__.py"

                if init_file.exists():
                    init_content = init_file.read_text(encoding='utf-8')
                    code_version_match = re.search(version_pattern, init_content)

                    if code_version_match:
                        code_version = code_version_match.group(0)
                        if doc_version != code_version:
                            result['issues'].append({
                                'type': 'version_mismatch',
                                'document': doc_file.name,
                                'doc_version': doc_version,
                                'code_version': code_version
                            })
                            result['status'] = 'warning'
                    else:
                        result['issues'].append({
                            'type': 'missing_code_version',
                            'document': doc_file.name,
                            'doc_version': doc_version
                        })
                        result['status'] = 'warning'

        return result

    def _extract_interfaces_from_file(self, file_path: Path) -> List[str]:
        """从文件中提取接口定义"""
        try:
            content = file_path.read_text(encoding='utf-8')
            # 查找类定义
            class_pattern = r'class (\w+).*?:'
            classes = re.findall(class_pattern, content)
            # 过滤出接口类（通常以I开头）
            interfaces = [cls for cls in classes if cls.startswith('I')]
            return interfaces
        except Exception as e:
            logger.warning(f"提取接口失败 {file_path}: {e}")
            return []

    def _extract_components_from_doc(self, doc_content: str) -> List[str]:
        """从文档中提取组件名称"""
        components = []

        # 查找类名和函数名
        class_pattern = r'class (\w+)'
        function_pattern = r'def (\w+)\('

        classes = re.findall(class_pattern, doc_content)
        functions = re.findall(function_pattern, doc_content)

        components.extend(classes)
        components.extend(functions)

        return list(set(components))  # 去重

    def _validate_import_statement(self, import_line: str) -> bool:
        """验证导入语句"""
        try:
            # 简单的语法检查
            ast.parse(import_line)
            return True
        except:
            return False

    def _generate_summary(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """生成检查摘要"""
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks.values() if check.get('status') == 'passed')
        warning_checks = sum(1 for check in checks.values() if check.get('status') == 'warning')
        error_checks = sum(1 for check in checks.values() if check.get('status') == 'error')

        # 计算一致性评分
        consistency_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        return {
            'total_checks': total_checks,
            'passed': passed_checks,
            'warnings': warning_checks,
            'errors': error_checks,
            'consistency_score': round(consistency_score, 2),
            'overall_status': 'passed' if error_checks == 0 and warning_checks == 0 else 'warning' if error_checks == 0 else 'error'
        }

    def _generate_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """生成推荐建议"""
        recommendations = []

        for check_name, check_result in checks.items():
            if check_result.get('status') != 'passed':
                issues = check_result.get('issues', [])

                if check_name == 'interface_consistency':
                    recommendations.append("🔧 更新架构文档，确保所有接口都在文档中正确描述")

                elif check_name == 'architecture_alignment':
                    recommendations.append("🏗️ 检查缺失的组件实现，或更新文档以反映当前架构状态")

                elif check_name == 'implementation_completeness':
                    recommendations.append("📋 验证Phase功能实现状态，确保文档描述与实际实现一致")

                elif check_name == 'documentation_accuracy':
                    recommendations.append("📝 修复文档中的语法错误和无效导入语句")

                elif check_name == 'version_synchronization':
                    recommendations.append("🔄 同步文档版本与代码版本信息")

        # 通用建议
        recommendations.extend([
            "🔍 建立定期一致性检查机制，每周执行一次",
            "🤖 考虑实现自动化文档更新工具",
            "📊 设置一致性检查的告警机制",
            "👥 培训团队成员了解一致性检查的重要性"
        ])

        return recommendations

    def _save_results(self, results: Dict[str, Any], prefix: str = "full_check"):
        """保存检查结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"

        report_file = self.reports_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"检查结果已保存到: {report_file}")

        # 生成HTML报告
        html_report = self._generate_html_report(results)
        html_file = self.reports_dir / f"{prefix}_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        logger.info(f"HTML报告已生成: {html_file}")

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """生成HTML报告"""
        summary = results.get('summary', {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RQA2025 一致性检查报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .issues {{ background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .recommendations {{ background: #d1ecf1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .status-passed {{ color: #28a745; }}
                .status-warning {{ color: #ffc107; }}
                .status-error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RQA2025 代码与文档一致性检查报告</h1>
                <p>检查时间: {results['timestamp']}</p>
                <p>一致性评分: <strong class="status-{summary.get('overall_status', 'passed')}">{summary.get('consistency_score', 0)}%</strong></p>
            </div>

            <div class="summary">
                <h2>📊 检查摘要</h2>
                <ul>
                    <li>总检查数: {summary.get('total_checks', 0)}</li>
                    <li>通过检查: <span class="status-passed">{summary.get('passed', 0)}</span></li>
                    <li>警告检查: <span class="status-warning">{summary.get('warnings', 0)}</span></li>
                    <li>错误检查: <span class="status-error">{summary.get('errors', 0)}</span></li>
                </ul>
            </div>

            <div class="issues">
                <h2>⚠️ 发现的问题</h2>
        """

        for check_name, check_result in results.get('checks', {}).items():
            if check_result.get('status') != 'passed':
                html += f"<h3>{check_name.replace('_', ' ').title()}</h3>"
                for issue in check_result.get('issues', []):
                    html += f"<p>• {issue.get('type', 'unknown')}: {issue.get('description', 'no description')}</p>"

        html += """
            </div>

            <div class="recommendations">
                <h2>💡 改进建议</h2>
                <ul>
        """

        for rec in results.get('recommendations', []):
            html += f"<li>{rec}</li>"

        html += """
                </ul>
            </div>
        </body>
        </html>
        """

        return html


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 一致性检查工具')
    parser.add_argument('--project-root', help='项目根目录路径')
    parser.add_argument('--quick', action='store_true', help='运行快速检查')
    parser.add_argument('--focus', nargs='+', help='快速检查的关注领域')

    args = parser.parse_args()

    # 创建检查器
    checker = ConsistencyChecker(args.project_root)

    if args.quick:
        # 运行快速检查
        results = checker.run_quick_check(args.focus)
        print(f"快速检查完成，一致性评分: {results['summary'].get('consistency_score', 0)}%")
    else:
        # 运行完整检查
        results = checker.run_full_check()
        print(f"完整检查完成，一致性评分: {results['summary'].get('consistency_score', 0)}%")

    # 输出主要问题
    if results['summary'].get('errors', 0) > 0 or results['summary'].get('warnings', 0) > 0:
        print("\n⚠️ 发现问题:")
        for check_name, check_result in results.get('checks', {}).items():
            if check_result.get('status') != 'passed':
                print(f"  • {check_name}: {len(check_result.get('issues', []))} 个问题")


if __name__ == "__main__":
    main()

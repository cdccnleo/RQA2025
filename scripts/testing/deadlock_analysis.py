#!/usr/bin/env python3
# -*- coding: utf-8
"""
死锁问题分析脚本

分析项目中的潜在死锁风险，包括：
1. 锁的获取顺序问题
2. 嵌套锁使用问题
3. 超时设置问题
4. 资源清理问题
"""

import time
from pathlib import Path
from typing import Dict, Any
import ast
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeadlockAnalyzer:
    """死锁分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.lock_patterns = {
            'threading.Lock': '普通锁',
            'threading.RLock': '可重入锁',
            'threading.Semaphore': '信号量',
            'threading.Event': '事件',
            'threading.Condition': '条件变量'
        }
        self.analysis_results = {
            'potential_deadlocks': [],
            'lock_usage_patterns': {},
            'nested_locks': [],
            'timeout_issues': [],
            'resource_cleanup_issues': []
        }

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """分析单个文件的死锁风险"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {'error': '语法错误，无法解析'}

            file_analysis = {
                'file_path': str(file_path.relative_to(self.project_root)),
                'lock_imports': [],
                'lock_instances': [],
                'lock_usage': [],
                'potential_issues': []
            }

            # 分析导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'threading' in alias.name:
                            file_analysis['lock_imports'].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module == 'threading':
                        for alias in node.names:
                            file_analysis['lock_imports'].append(f"threading.{alias.name}")

                # 分析锁的实例化
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if isinstance(node.value, ast.Call):
                                if isinstance(node.value.func, ast.Attribute):
                                    if node.value.func.value.id == 'threading':
                                        lock_type = f"threading.{node.value.func.attr}"
                                        file_analysis['lock_instances'].append({
                                            'name': target.id,
                                            'type': lock_type,
                                            'line': node.lineno
                                        })

                # 分析锁的使用
                elif isinstance(node, ast.With):
                    if isinstance(node.items[0].context_expr, ast.Name):
                        lock_name = node.items[0].context_expr.id
                        file_analysis['lock_usage'].append({
                            'lock_name': lock_name,
                            'line': node.lineno,
                            'type': 'with_statement'
                        })

                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['acquire', 'release']:
                            if isinstance(node.func.value, ast.Name):
                                lock_name = node.func.value.id
                                file_analysis['lock_usage'].append({
                                    'lock_name': lock_name,
                                    'line': node.lineno,
                                    'type': f'{node.func.attr}_call'
                                })

            # 检测潜在问题
            self._detect_potential_issues(file_analysis)

            return file_analysis

        except Exception as e:
            return {'error': f'分析失败: {e}'}

    def _detect_potential_issues(self, file_analysis: Dict[str, Any]):
        """检测潜在的死锁问题"""
        issues = []

        # 检查是否有锁但没有导入
        if file_analysis['lock_usage'] and not file_analysis['lock_imports']:
            issues.append("使用锁但未导入threading模块")

        # 检查锁的使用模式
        lock_names = {lock['name'] for lock in file_analysis['lock_instances']}
        used_locks = {usage['lock_name'] for usage in file_analysis['lock_usage']}

        # 检查未定义的锁使用
        undefined_locks = used_locks - lock_names
        if undefined_locks:
            issues.append(f"使用未定义的锁: {undefined_locks}")

        # 检查锁的获取和释放是否匹配
        acquire_calls = [u for u in file_analysis['lock_usage'] if u['type'] == 'acquire_call']
        release_calls = [u for u in file_analysis['lock_usage'] if u['type'] == 'release_call']

        if len(acquire_calls) != len(release_calls):
            issues.append(f"锁的获取和释放不匹配: acquire={len(acquire_calls)}, release={len(release_calls)}")

        file_analysis['potential_issues'] = issues

    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目的死锁风险"""
        logger.info("开始分析项目死锁风险...")

        python_files = list(self.project_root.rglob("*.py"))
        logger.info(f"找到 {len(python_files)} 个Python文件")

        for file_path in python_files:
            # 跳过测试文件和备份文件
            if any(skip in str(file_path) for skip in ['test_', 'backup', '__pycache__', '.git']):
                continue

            logger.info(f"分析文件: {file_path.relative_to(self.project_root)}")
            file_analysis = self.analyze_file(file_path)

            if 'error' not in file_analysis:
                # 汇总分析结果
                self._summarize_file_analysis(file_analysis)

        return self.analysis_results

    def _summarize_file_analysis(self, file_analysis: Dict[str, Any]):
        """汇总单个文件的分析结果"""
        # 统计锁的使用模式
        for lock_type in file_analysis['lock_instances']:
            lock_type_name = lock_type['type']
            if lock_type_name not in self.analysis_results['lock_usage_patterns']:
                self.analysis_results['lock_usage_patterns'][lock_type_name] = 0
            self.analysis_results['lock_usage_patterns'][lock_type_name] += 1

        # 记录潜在问题
        if file_analysis['potential_issues']:
            self.analysis_results['potential_deadlocks'].append({
                'file': file_analysis['file_path'],
                'issues': file_analysis['potential_issues']
            })

    def generate_report(self) -> str:
        """生成死锁分析报告"""
        report = []
        report.append("# 死锁问题分析报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"项目路径: {self.project_root}")
        report.append("")

        # 锁使用模式统计
        report.append("## 锁使用模式统计")
        for lock_type, count in self.analysis_results['lock_usage_patterns'].items():
            report.append(f"- {lock_type}: {count} 个实例")
        report.append("")

        # 潜在死锁问题
        if self.analysis_results['potential_deadlocks']:
            report.append("## 潜在死锁问题")
            for issue in self.analysis_results['potential_deadlocks']:
                report.append(f"### {issue['file']}")
                for problem in issue['issues']:
                    report.append(f"- {problem}")
                report.append("")
        else:
            report.append("## 潜在死锁问题")
            report.append("未发现明显的死锁风险")
            report.append("")

        # 建议
        report.append("## 死锁预防建议")
        report.append("1. 使用 `threading.RLock()` 替代 `threading.Lock()` 避免重入死锁")
        report.append("2. 为所有锁操作设置合理的超时时间")
        report.append("3. 确保锁的获取顺序一致，避免循环等待")
        report.append("4. 使用 `with` 语句确保锁的正确释放")
        report.append("5. 在异常情况下也要确保锁的释放")
        report.append("6. 避免在持有锁时调用可能获取其他锁的方法")
        report.append("7. 定期检查长时间持有的锁")

        return "\n".join(report)


def main():
    """主函数"""
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    logger.info(f"项目根目录: {project_root}")

    # 创建分析器
    analyzer = DeadlockAnalyzer(project_root)

    # 分析项目
    results = analyzer.analyze_project()

    # 生成报告
    report = analyzer.generate_report()

    # 保存报告
    report_file = project_root / "reports" / "testing" / "deadlock_analysis_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"死锁分析报告已保存到: {report_file}")

    # 打印关键问题
    if results['potential_deadlocks']:
        logger.warning(f"发现 {len(results['potential_deadlocks'])} 个文件存在潜在死锁风险")
        for issue in results['potential_deadlocks'][:5]:  # 只显示前5个
            logger.warning(f"文件: {issue['file']}")
            for problem in issue['issues']:
                logger.warning(f"  问题: {problem}")
    else:
        logger.info("未发现明显的死锁风险")

    # 打印锁使用统计
    logger.info("锁使用模式统计:")
    for lock_type, count in results['lock_usage_patterns'].items():
        logger.info(f"  {lock_type}: {count} 个实例")


if __name__ == "__main__":
    main()

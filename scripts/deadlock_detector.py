#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
死锁检测工具
用于检测基础设施层代码中的潜在死锁问题
"""

import re
from typing import Dict, List
from pathlib import Path


class DeadlockDetector:
    """死锁检测器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.lock_patterns = [
            r'threading\.Lock\(\)',
            r'threading\.RLock\(\)',
            r'asyncio\.Lock\(\)',
            r'Lock\(\)',
            r'RLock\(\)',
            r'\.lock\s*=',
            r'\._lock\s*=',
            r'self\.lock\s*=',
            r'self\._lock\s*='
        ]

        self.lock_usage_patterns = [
            r'with\s+.*\.lock\s*:',
            r'with\s+.*\._lock\s*:',
            r'with\s+self\.lock\s*:',
            r'with\s+self\._lock\s*:',
            r'\.lock\.acquire\(\)',
            r'\._lock\.acquire\(\)',
            r'self\.lock\.acquire\(\)',
            r'self\._lock\.acquire\(\)',
            r'\.lock\.release\(\)',
            r'\._lock\.release\(\)',
            r'self\.lock\.release\(\)',
            r'self\._lock\.release\(\)'
        ]

        self.potential_deadlock_patterns = [
            r'threading\.Thread\(.*target=.*\)\.start\(\)',
            r'concurrent\.futures\.',
            r'multiprocessing\.',
            r'asyncio\.gather\(.*await',
            r'await.*asyncio\.gather'
        ]

    def scan_files(self) -> Dict[str, Dict]:
        """扫描所有Python文件"""
        results = {}

        for py_file in self.root_path.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(py_file, 'r', encoding='latin-1') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue

            analysis = self.analyze_file(str(py_file), content)
            if analysis:
                results[str(py_file)] = analysis

        return results

    def analyze_file(self, file_path: str, content: str) -> Dict:
        """分析单个文件"""
        analysis = {
            'locks_defined': [],
            'lock_usage': [],
            'potential_deadlocks': [],
            'nested_locks': [],
            'concurrent_operations': [],
            'issues': []
        }

        # 检测锁定义
        for pattern in self.lock_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis['locks_defined'].extend(matches)

        # 检测锁使用
        for pattern in self.lock_usage_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis['lock_usage'].extend(matches)

        # 检测潜在死锁模式
        for pattern in self.potential_deadlock_patterns:
            matches = re.findall(pattern, content)
            if matches:
                analysis['potential_deadlocks'].extend(matches)

        # 检测嵌套锁
        analysis['nested_locks'] = self.detect_nested_locks(content)

        # 检测并发操作
        analysis['concurrent_operations'] = self.detect_concurrent_operations(content)

        # 分析问题
        analysis['issues'] = self.analyze_issues(analysis)

        # 只返回有问题的文件
        if any(analysis.values()):
            return analysis

        return {}

    def detect_nested_locks(self, content: str) -> List[str]:
        """检测嵌套锁使用"""
        nested_locks = []

        # 查找嵌套的with语句
        lines = content.split('\n')
        lock_stack = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 进入锁上下文
            if 'with ' in stripped and ('lock' in stripped or '_lock' in stripped):
                lock_stack.append((i, stripped))

            # 离开锁上下文
            elif stripped.startswith(')') and lock_stack:
                # 检查是否有嵌套
                if len(lock_stack) > 1:
                    nested_info = {
                        'line': i,
                        'nested_locks': lock_stack.copy()
                    }
                    nested_locks.append(nested_info)
                lock_stack.pop()

        return nested_locks

    def detect_concurrent_operations(self, content: str) -> List[str]:
        """检测并发操作"""
        concurrent_ops = []

        # 检测线程创建
        thread_matches = re.findall(r'threading\.Thread\(.*target=.*\)', content)
        concurrent_ops.extend(thread_matches)

        # 检测线程池
        pool_matches = re.findall(r'ThreadPoolExecutor\(.*\)', content)
        concurrent_ops.extend(pool_matches)

        # 检测进程池
        process_matches = re.findall(r'ProcessPoolExecutor\(.*\)', content)
        concurrent_ops.extend(process_matches)

        # 检测异步操作
        async_matches = re.findall(r'asyncio\.gather\(.*\)', content)
        concurrent_ops.extend(async_matches)

        return concurrent_ops

    def analyze_issues(self, analysis: Dict) -> List[str]:
        """分析潜在问题"""
        issues = []

        # 检查锁数量
        if len(analysis['locks_defined']) > 5:
            issues.append(f"过多锁定义 ({len(analysis['locks_defined'])}) - 可能增加死锁风险")

        # 检查嵌套锁
        if analysis['nested_locks']:
            issues.append(f"发现嵌套锁使用 ({len(analysis['nested_locks'])}) - 高死锁风险")

        # 检查并发操作
        if len(analysis['concurrent_operations']) > 0 and len(analysis['locks_defined']) > 0:
            issues.append("并发操作与锁同时使用 - 需要检查死锁风险")

        # 检查锁使用模式
        acquire_count = len([u for u in analysis['lock_usage'] if 'acquire' in u])
        release_count = len([u for u in analysis['lock_usage'] if 'release' in u])

        if acquire_count != release_count:
            issues.append(f"锁获取({acquire_count})和释放({release_count})次数不匹配")

        return issues

    def generate_report(self, results: Dict) -> str:
        """生成报告"""
        report_lines = []
        report_lines.append("# 死锁检测报告")
        report_lines.append("")

        total_files = len(results)
        files_with_issues = len([f for f in results.values() if f.get('issues')])
        total_locks = sum(len(f.get('locks_defined', [])) for f in results.values())
        total_nested = sum(len(f.get('nested_locks', [])) for f in results.values())

        report_lines.append(f"## 概览")
        report_lines.append(f"- 扫描文件数: {total_files}")
        report_lines.append(f"- 有问题文件数: {files_with_issues}")
        report_lines.append(f"- 锁定义总数: {total_locks}")
        report_lines.append(f"- 嵌套锁总数: {total_nested}")
        report_lines.append("")

        if files_with_issues > 0:
            report_lines.append("## 🚨 有问题的文件")
            report_lines.append("")

            for file_path, analysis in results.items():
                if analysis.get('issues'):
                    report_lines.append(f"### {file_path}")
                    for issue in analysis['issues']:
                        report_lines.append(f"- ⚠️ {issue}")

                    if analysis['locks_defined']:
                        report_lines.append("  锁定义:")
                        for lock in analysis['locks_defined'][:5]:  # 只显示前5个
                            report_lines.append(f"    - {lock}")

                    if analysis['nested_locks']:
                        report_lines.append("  嵌套锁:")
                        for nested in analysis['nested_locks'][:3]:  # 只显示前3个
                            report_lines.append(
                                f"    - 第{nested['line']}行: {len(nested['nested_locks'])}层嵌套")

                    report_lines.append("")

        report_lines.append("## 📊 详细统计")
        report_lines.append("")

        for file_path, analysis in results.items():
            if analysis:
                locks = len(analysis.get('locks_defined', []))
                nested = len(analysis.get('nested_locks', []))
                concurrent = len(analysis.get('concurrent_operations', []))
                issues = len(analysis.get('issues', []))

                if locks > 0 or nested > 0 or concurrent > 0 or issues > 0:
                    report_lines.append(f"- {file_path}")
                    report_lines.append(f"  - 锁定义: {locks}")
                    report_lines.append(f"  - 嵌套锁: {nested}")
                    report_lines.append(f"  - 并发操作: {concurrent}")
                    report_lines.append(f"  - 问题数: {issues}")

        return "\n".join(report_lines)


def main():
    """主函数"""
    detector = DeadlockDetector("src/infrastructure")
    results = detector.scan_files()
    report = detector.generate_report(results)

    print(report)

    # 保存报告
    with open("deadlock_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n报告已保存到: deadlock_analysis_report.md")


if __name__ == "__main__":
    main()

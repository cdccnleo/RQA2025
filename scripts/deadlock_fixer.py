#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
死锁修复工具
自动修复基础设施层代码中的死锁问题
"""

import re
from typing import Dict, List
from pathlib import Path


class DeadlockFixer:
    """死锁修复器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.fixed_files = []
        self.backup_files = []

    def scan_and_fix(self) -> Dict[str, Dict]:
        """扫描并修复死锁问题"""
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

            issues = self.analyze_issues(content)
            if issues:
                fixes = self.generate_fixes(content, issues)
                if fixes:
                    results[str(py_file)] = {
                        'issues': issues,
                        'fixes': fixes,
                        'original_content': content
                    }

        return results

    def analyze_issues(self, content: str) -> List[Dict]:
        """分析代码中的死锁问题"""
        issues = []

        # 检测嵌套锁
        nested_locks = self.detect_nested_locks(content)
        if nested_locks:
            issues.extend(nested_locks)

        # 检测锁竞争
        lock_contention = self.detect_lock_contention(content)
        if lock_contention:
            issues.extend(lock_contention)

        # 检测长时间持有锁
        long_holding = self.detect_long_lock_holding(content)
        if long_holding:
            issues.extend(long_holding)

        return issues

    def detect_nested_locks(self, content: str) -> List[Dict]:
        """检测嵌套锁"""
        issues = []
        lines = content.split('\n')

        lock_stack = []
        for i, line in enumerate(lines):
            stripped = line.strip()

            # 检测锁进入
            if self.is_lock_acquire(stripped):
                lock_info = self.extract_lock_info(stripped, i)
                lock_stack.append(lock_info)

            # 检测锁释放
            elif self.is_lock_release(stripped):
                if lock_stack:
                    # 检查是否有嵌套
                    if len(lock_stack) > 1:
                        nested_info = {
                            'type': 'nested_lock',
                            'severity': 'high',
                            'line': i,
                            'description': f'发现嵌套锁使用 ({len(lock_stack)}层)',
                            'nested_locks': lock_stack.copy(),
                            'fix_suggestion': '重构为独立的锁操作或使用不同的锁'
                        }
                        issues.append(nested_info)
                    lock_stack.pop()

        return issues

    def detect_lock_contention(self, content: str) -> List[Dict]:
        """检测锁竞争"""
        issues = []
        lines = content.split('\n')

        lock_usage = {}
        for i, line in enumerate(lines):
            if 'with ' in line and ('lock' in line or '_lock' in line):
                lock_name = self.extract_lock_name(line)
                if lock_name:
                    if lock_name not in lock_usage:
                        lock_usage[lock_name] = []
                    lock_usage[lock_name].append(i)

        # 检查锁使用频率
        for lock_name, usages in lock_usage.items():
            if len(usages) > 10:  # 同一个锁被使用超过10次
                issues.append({
                    'type': 'lock_contention',
                    'severity': 'medium',
                    'line': usages[0],
                    'description': f'锁 {lock_name} 使用过于频繁 ({len(usages)}次)',
                    'usages': usages,
                    'fix_suggestion': '考虑使用读写锁或减少锁的粒度'
                })

        return issues

    def detect_long_lock_holding(self, content: str) -> List[Dict]:
        """检测长时间持有锁"""
        issues = []
        lines = content.split('\n')

        lock_blocks = []
        current_block = None

        for i, line in enumerate(lines):
            if self.is_lock_acquire(line.strip()):
                if current_block is None:
                    current_block = {
                        'start_line': i,
                        'lock_info': self.extract_lock_info(line.strip(), i),
                        'operations': []
                    }

            elif current_block is not None:
                current_block['operations'].append(line.strip())

                if self.is_lock_release(line.strip()):
                    current_block['end_line'] = i
                    current_block['block_size'] = i - current_block['start_line']

                    # 检查锁块大小
                    if current_block['block_size'] > 20:  # 锁块超过20行
                        issues.append({
                            'type': 'long_lock_holding',
                            'severity': 'medium',
                            'line': current_block['start_line'],
                            'description': f'锁持有时间过长 ({current_block["block_size"]}行代码)',
                            'block_info': current_block,
                            'fix_suggestion': '减少锁的持有时间，将非关键操作移到锁外'
                        })

                    lock_blocks.append(current_block)
                    current_block = None

        return issues

    def is_lock_acquire(self, line: str) -> bool:
        """判断是否是锁获取操作"""
        return ('with ' in line and ('lock' in line or '_lock' in line)) or \
               ('.acquire()' in line and ('lock' in line or '_lock' in line))

    def is_lock_release(self, line: str) -> bool:
        """判断是否是锁释放操作"""
        return (line.strip() == ')' and 'with' in line) or \
               ('.release()' in line and ('lock' in line or '_lock' in line))

    def extract_lock_info(self, line: str, line_num: int) -> Dict:
        """提取锁信息"""
        return {
            'line': line_num,
            'statement': line.strip(),
            'lock_type': 'RLock' if 'RLock' in line else 'Lock'
        }

    def extract_lock_name(self, line: str) -> str:
        """提取锁名称"""
        # 匹配 self.lock, self._lock, lock 等模式
        patterns = [
            r'self\.([a-zA-Z_][a-zA-Z0-9_]*lock)',
            r'([a-zA-Z_][a-zA-Z0-9_]*lock)',
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)

        return None

    def generate_fixes(self, content: str, issues: List[Dict]) -> List[Dict]:
        """生成修复建议"""
        fixes = []

        for issue in issues:
            if issue['type'] == 'nested_lock':
                fixes.append(self.generate_nested_lock_fix(issue))
            elif issue['type'] == 'lock_contention':
                fixes.append(self.generate_contention_fix(issue))
            elif issue['type'] == 'long_lock_holding':
                fixes.append(self.generate_long_holding_fix(issue))

        return fixes

    def generate_nested_lock_fix(self, issue: Dict) -> Dict:
        """生成嵌套锁修复建议"""
        return {
            'type': 'refactor',
            'description': '重构嵌套锁为独立的锁操作',
            'code_changes': [
                '# 将嵌套锁重构为独立的锁操作',
                '# 例如:',
                '# with self.lock_a:',
                '#     with self.lock_b:  # 嵌套',
                '#         ...',
                '# 改为:',
                '# with self.lock_a:',
                '#     ...',
                '# with self.lock_b:',
                '#     ...'
            ],
            'risk_level': 'medium',
            'estimated_time': '2-4小时'
        }

    def generate_contention_fix(self, issue: Dict) -> Dict:
        """生成锁竞争修复建议"""
        return {
            'type': 'optimization',
            'description': '减少锁竞争或使用读写锁',
            'code_changes': [
                '# 使用读写锁减少竞争',
                '# from readerwriterlock import rwlock',
                '# self.lock = rwlock.RWLockRead()',
                '# 或减少锁的粒度',
                '# 将大锁拆分为多个小锁'
            ],
            'risk_level': 'low',
            'estimated_time': '1-2小时'
        }

    def generate_long_holding_fix(self, issue: Dict) -> Dict:
        """生成长时间持有锁修复建议"""
        return {
            'type': 'optimization',
            'description': '减少锁持有时间',
            'code_changes': [
                '# 将非关键操作移到锁外',
                '# 例如:',
                '# with self.lock:',
                '#     result = expensive_operation()  # 移到锁外',
                '#     self.data = result',
                '# 改为:',
                '# result = expensive_operation()',
                '# with self.lock:',
                '#     self.data = result'
            ],
            'risk_level': 'low',
            'estimated_time': '30分钟-1小时'
        }

    def apply_fixes(self, results: Dict) -> Dict[str, List[str]]:
        """应用修复（生成修复脚本）"""
        applied_fixes = {}

        for file_path, result in results.items():
            fixes_applied = []
            content = result['original_content']
            issues = result['issues']

            # 这里可以实现自动修复逻辑
            # 目前只生成修复建议

            for issue in issues:
                fixes_applied.append(f"修复 {issue['type']}: {issue['description']}")

            if fixes_applied:
                applied_fixes[file_path] = fixes_applied

        return applied_fixes

    def generate_report(self, results: Dict) -> str:
        """生成修复报告"""
        report_lines = []
        report_lines.append("# 死锁修复报告")
        report_lines.append("")

        total_files = len(results)
        total_issues = sum(len(r['issues']) for r in results.values())
        high_risk = sum(1 for r in results.values()
                        for issue in r['issues'] if issue.get('severity') == 'high')

        report_lines.append(f"## 概览")
        report_lines.append(f"- 需要修复的文件数: {total_files}")
        report_lines.append(f"- 发现的问题总数: {total_issues}")
        report_lines.append(f"- 高风险问题数: {high_risk}")
        report_lines.append("")

        if results:
            report_lines.append("## 🔧 修复建议")
            report_lines.append("")

            for file_path, result in results.items():
                report_lines.append(f"### {file_path}")
                report_lines.append("")

                for issue in result['issues']:
                    report_lines.append(
                        f"#### {issue['type'].upper()} - {issue['severity'].upper()}")
                    report_lines.append(f"**描述**: {issue['description']}")
                    report_lines.append(f"**位置**: 第{issue['line']}行")
                    report_lines.append(f"**建议**: {issue['fix_suggestion']}")
                    report_lines.append("")

                if result['fixes']:
                    report_lines.append("**修复方案**:")
                    for fix in result['fixes']:
                        report_lines.append(f"- **{fix['type']}**: {fix['description']}")
                        report_lines.append(f"  - 风险等级: {fix['risk_level']}")
                        report_lines.append(f"  - 预估时间: {fix['estimated_time']}")
                        report_lines.append("  - 代码变更:")
                        for code in fix['code_changes']:
                            report_lines.append(f"    ```python\n    {code}\n    ```")
                        report_lines.append("")

        report_lines.append("## 📋 修复优先级")
        report_lines.append("")
        report_lines.append("1. **高优先级** (立即修复)")
        report_lines.append("   - 嵌套锁问题 (severity: high)")
        report_lines.append("   - 多层嵌套 (>5层)")
        report_lines.append("")
        report_lines.append("2. **中优先级** (近期修复)")
        report_lines.append("   - 锁竞争问题")
        report_lines.append("   - 长时间持有锁")
        report_lines.append("")
        report_lines.append("3. **低优先级** (优化阶段)")
        report_lines.append("   - 锁使用优化")
        report_lines.append("   - 性能改进")
        report_lines.append("")

        report_lines.append("## ⚠️ 注意事项")
        report_lines.append("")
        report_lines.append("- 修复嵌套锁时需要特别小心，避免引入竞态条件")
        report_lines.append("- 测试所有修复后的并发场景")
        report_lines.append("- 监控修复后的性能表现")
        report_lines.append("- 考虑使用更高级的同步原语")

        return "\n".join(report_lines)


def main():
    """主函数"""
    fixer = DeadlockFixer("src/infrastructure")
    results = fixer.scan_and_fix()

    print("死锁修复分析完成")
    print(f"发现 {len(results)} 个需要修复的文件")

    report = fixer.generate_report(results)

    with open("deadlock_fix_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("修复报告已保存到: deadlock_fix_report.md")


if __name__ == "__main__":
    main()

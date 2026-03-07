#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试超时检查工具
检查基础设施层测试的超时设置情况
"""

import re
from pathlib import Path


class TimeoutChecker:
    """超时检查器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.risk_patterns = [
            # 并发操作
            r'threading\.Thread\(',
            r'concurrent\.futures\.',
            r'multiprocessing\.',
            r'asyncio\.',
            r'await\s+',

            # 网络操作
            r'requests\.',
            r'urllib\.',
            r'http\.',

            # 数据库操作
            r'\.connect\(',
            r'\.execute\(',
            r'\.query\(',

            # 文件操作
            r'open\(',
            r'\.read\(',
            r'\.write\(',

            # 循环操作
            r'while\s+True:',
            r'for\s+\w+\s+in\s+range\(',
            r'while\s+\w+:',

            # 锁操作
            r'with\s+.*lock',
            r'\.acquire\(\)',
            r'\.wait\(\)',
        ]

        self.timeout_levels = {
            'low': 30,      # 简单操作
            'medium': 60,   # 中等复杂度
            'high': 120,    # 高复杂度
            'critical': 300  # 非常复杂
        }

    def check_timeout_settings(self) -> dict:
        """检查超时设置"""
        results = {}

        for py_file in self.root_path.rglob('test_*.py'):
            if 'test' not in str(py_file):
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

            risk_level = self.assess_risk_level(content)
            if risk_level != 'none':
                results[str(py_file)] = {
                    'risk_level': risk_level,
                    'recommended_timeout': self.timeout_levels[risk_level],
                    'current_timeout': self.get_current_timeout(content),
                    'has_concurrent_ops': self.has_concurrent_operations(content),
                    'has_network_ops': self.has_network_operations(content)
                }

        return results

    def assess_risk_level(self, content: str) -> str:
        """评估风险等级"""
        risk_score = 0

        for pattern in self.risk_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                risk_score += matches

        # 根据风险分数确定等级
        if risk_score >= 10:
            return 'critical'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        elif risk_score >= 1:
            return 'low'
        else:
            return 'none'

    def get_current_timeout(self, content: str) -> int:
        """获取当前超时设置"""
        timeout_match = re.search(r'@pytest\.mark\.timeout\((\d+)\)', content)
        if timeout_match:
            return int(timeout_match.group(1))
        return 0

    def has_concurrent_operations(self, content: str) -> bool:
        """检查是否有并发操作"""
        concurrent_patterns = [
            r'threading\.Thread\(',
            r'concurrent\.futures\.',
            r'multiprocessing\.',
            r'asyncio\.',
            r'await\s+'
        ]

        for pattern in concurrent_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def has_network_operations(self, content: str) -> bool:
        """检查是否有网络操作"""
        network_patterns = [
            r'requests\.',
            r'urllib\.',
            r'http\.',
            r'\.connect\('
        ]

        for pattern in network_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def add_timeout_decorators(self, results: dict) -> dict:
        """添加超时装饰器"""
        added_timeouts = {}

        for file_path, info in results.items():
            if info['current_timeout'] == 0:  # 没有设置超时
                success = self._add_timeout_to_file(file_path, info)
                added_timeouts[file_path] = success

        return added_timeouts

    def _add_timeout_to_file(self, file_path: str, info: dict) -> bool:
        """为文件添加超时装饰器"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                return False

        # 为测试类添加超时装饰器
        lines = content.split('\n')
        modified_lines = []

        for i, line in enumerate(lines):
            modified_lines.append(line)

            # 检测测试类开始
            if line.strip().startswith('class ') and 'Test' in line.strip():
                # 在类定义后添加超时装饰器
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() == '' or next_line.strip().startswith('"""'):
                        # 在类定义后添加装饰器
                        indent = '    '  # 标准缩进
                        timeout_decorator = f"{indent}@pytest.mark.timeout({info['recommended_timeout']})"
                        modified_lines.append(timeout_decorator)

        # 写入修改后的内容
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))
            return True
        except Exception as e:
            print(f"写入文件失败 {file_path}: {e}")
            return False

    def generate_report(self, results: dict) -> str:
        """生成报告"""
        report_lines = []
        report_lines.append("# 测试超时设置检查报告")
        report_lines.append("")

        total_files = len(results)
        files_with_timeout = len([f for f in results.values() if f['current_timeout'] > 0])
        files_needing_timeout = len([f for f in results.values() if f['current_timeout'] == 0])
        high_risk_files = len(
            [f for f in results.values() if f['risk_level'] in ['high', 'critical']])

        report_lines.append(f"## 📊 概览")
        report_lines.append(f"- 扫描文件数: {total_files}")
        report_lines.append(f"- 已设置超时文件数: {files_with_timeout}")
        report_lines.append(f"- 需要设置超时文件数: {files_needing_timeout}")
        report_lines.append(f"- 高风险文件数: {high_risk_files}")
        report_lines.append("")

        if results:
            report_lines.append("## 📋 超时设置详情")
            report_lines.append("")

            for file_path, info in results.items():
                risk_emoji = {
                    'critical': '🚨',
                    'high': '⚠️',
                    'medium': '📊',
                    'low': 'ℹ️',
                    'none': '✅'
                }.get(info['risk_level'], '❓')

                report_lines.append(f"### {risk_emoji} {Path(file_path).name}")
                report_lines.append(f"- **风险等级**: {info['risk_level']}")
                report_lines.append(f"- **推荐超时**: {info['recommended_timeout']}秒")
                report_lines.append(f"- **并发操作**: {'✅' if info['has_concurrent_ops'] else '❌'}")
                report_lines.append(f"- **网络操作**: {'✅' if info['has_network_ops'] else '❌'}")

                if info['current_timeout'] > 0:
                    report_lines.append(f"- **当前超时**: {info['current_timeout']}秒 ✅")
                else:
                    report_lines.append(f"- **当前超时**: 未设置 ❌")
                    report_lines.append(
                        f"- **建议**: 添加 `@pytest.mark.timeout({info['recommended_timeout']})`")

                report_lines.append("")

        report_lines.append("## 🔧 超时配置说明")
        report_lines.append("")
        report_lines.append("### 全局配置 (pytest.ini)")
        report_lines.append("```ini")
        report_lines.append("--timeout=60")
        report_lines.append("--timeout-method=thread")
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("### 风险等级说明")
        report_lines.append("- **critical**: 300秒 - 涉及复杂并发、网络、数据库操作")
        report_lines.append("- **high**: 120秒 - 涉及并发、文件操作、循环")
        report_lines.append("- **medium**: 60秒 - 涉及锁操作、异步操作")
        report_lines.append("- **low**: 30秒 - 涉及简单并发、I/O操作")
        report_lines.append("")
        report_lines.append("### 最佳实践")
        report_lines.append("1. 为所有并发测试设置超时")
        report_lines.append("2. 根据操作复杂度选择合适的超时时间")
        report_lines.append("3. 使用thread超时方法处理死锁")
        report_lines.append("4. 定期检查和调整超时设置")
        report_lines.append("")
        report_lines.append("### 死锁预防")
        report_lines.append("- 避免嵌套锁使用")
        report_lines.append("- 使用RLock替代Lock")
        report_lines.append("- 设置合理的锁等待超时")
        report_lines.append("- 在测试中使用thread.join()的timeout参数")

        return "\n".join(report_lines)


def main():
    """主函数"""
    checker = TimeoutChecker("tests/unit/infrastructure")
    results = checker.check_timeout_settings()

    print(f"发现 {len(results)} 个需要检查的文件")

    # 添加超时装饰器
    print("开始添加超时装饰器...")
    added_results = checker.add_timeout_decorators(results)
    added_count = len([r for r in added_results.values() if r])

    print(f"成功为 {added_count} 个文件添加了超时装饰器")

    # 生成报告
    report = checker.generate_report(results)

    with open("timeout_check_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("超时检查报告已保存到: timeout_check_report.md")

    # 显示统计信息
    critical_files = [f for f, info in results.items() if info['risk_level'] == 'critical']
    high_risk_files = [f for f, info in results.items() if info['risk_level'] == 'high']

    if critical_files:
        print(f"\n🚨 发现 {len(critical_files)} 个高风险文件:")
        for file_path in critical_files[:5]:  # 只显示前5个
            print(f"  - {Path(file_path).name}")

    if high_risk_files:
        print(f"\n⚠️ 发现 {len(high_risk_files)} 个中等风险文件:")
        for file_path in high_risk_files[:5]:  # 只显示前5个
            print(f"  - {Path(file_path).name}")


if __name__ == "__main__":
    main()

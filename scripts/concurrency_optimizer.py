#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发优化工具
用于优化基础设施层代码中的并发性能问题
"""

import re
from typing import Dict, List
from pathlib import Path


class ConcurrencyOptimizer:
    """并发优化器"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.optimized_files = []

    def optimize_concurrent_code(self) -> Dict[str, List[str]]:
        """优化并发代码"""
        optimizations = {}

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

            file_optimizations = self.optimize_file(str(py_file), content)
            if file_optimizations:
                optimizations[str(py_file)] = file_optimizations

        return optimizations

    def optimize_file(self, file_path: str, content: str) -> List[str]:
        """优化单个文件"""
        optimizations = []

        # 检测可以优化的模式
        optimizations.extend(self.optimize_lock_patterns(content))
        optimizations.extend(self.optimize_atomic_operations(content))
        optimizations.extend(self.optimize_thread_safety(content))

        return optimizations

    def optimize_lock_patterns(self, content: str) -> List[str]:
        """优化锁模式"""
        optimizations = []

        # 检测不必要的锁使用
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'with self._lock:' in line.strip():
                # 检查锁内的操作是否真正需要同步
                next_lines = lines[i+1:i+10] if i+10 < len(lines) else lines[i+1:]
                lock_content = []
                indent_level = len(line) - len(line.lstrip())

                for next_line in next_lines:
                    if next_line.strip() == '':
                        continue
                    if len(next_line) - len(next_line.lstrip()) <= indent_level:
                        break
                    lock_content.append(next_line.strip())

                # 分析锁内容
                if self._is_simple_operation(lock_content):
                    optimizations.append(f"第{i+1}行: 可以移除不必要的锁")

        return optimizations

    def optimize_atomic_operations(self, content: str) -> List[str]:
        """优化原子操作"""
        optimizations = []

        # 检测可以原子化的操作
        atomic_patterns = [
            r'self\.(\w+)\s*\+=\s*1',  # 计数器递增
            r'self\.(\w+)\s*-=\s*1',  # 计数器递减
            r'self\.(\w+)\s*=\s*\d+',  # 简单赋值
        ]

        for pattern in atomic_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for var in matches:
                    optimizations.append(f"变量 {var} 可以考虑使用原子操作")

        return optimizations

    def optimize_thread_safety(self, content: str) -> List[str]:
        """优化线程安全"""
        optimizations = []

        # 检测潜在的线程安全问题
        if 'threading.Thread' in content and 'self.' in content:
            optimizations.append("检测到多线程访问实例变量，建议检查线程安全")

        # 检测全局变量使用
        global_vars = re.findall(r'\bglobal\s+(\w+)', content)
        if global_vars:
            optimizations.append(f"检测到全局变量: {', '.join(global_vars)}，可能存在线程安全风险")

        return optimizations

    def _is_simple_operation(self, lock_content: List[str]) -> bool:
        """判断是否是简单操作"""
        if not lock_content:
            return False

        # 简单操作模式
        simple_patterns = [
            r'^\w+\s*=\s*\w+',  # 简单赋值
            r'^\w+\s*\.\s*\w+\s*=\s*.*',  # 属性赋值
            r'^return\s+.*',  # 返回语句
            r'^if\s+.*:\s*$',  # 简单条件
            r'^del\s+.*',  # 删除操作
        ]

        for line in lock_content:
            line = line.strip()
            if line.startswith('#'):  # 注释
                continue
            if line == '':  # 空行
                continue

            # 检查是否匹配简单操作
            is_simple = False
            for pattern in simple_patterns:
                if re.match(pattern, line):
                    is_simple = True
                    break

            if not is_simple:
                return False

        return True

    def generate_optimization_report(self, optimizations: Dict[str, List[str]]) -> str:
        """生成优化报告"""
        report_lines = []
        report_lines.append("# 并发优化报告")
        report_lines.append("")

        total_files = len(optimizations)
        total_optimizations = sum(len(opts) for opts in optimizations.values())

        report_lines.append(f"## 概览")
        report_lines.append(f"- 需要优化的文件数: {total_files}")
        report_lines.append(f"- 发现的优化点总数: {total_optimizations}")
        report_lines.append("")

        if optimizations:
            report_lines.append("## 📈 优化建议")
            report_lines.append("")

            for file_path, file_optimizations in optimizations.items():
                report_lines.append(f"### {file_path}")
                report_lines.append("")

                for optimization in file_optimizations:
                    report_lines.append(f"- ✅ {optimization}")

                report_lines.append("")

        report_lines.append("## 🔧 优化策略")
        report_lines.append("")
        report_lines.append("### 1. 锁优化")
        report_lines.append("- 移除不必要的锁")
        report_lines.append("- 使用更细粒度的锁")
        report_lines.append("- 考虑读写锁分离")
        report_lines.append("")
        report_lines.append("### 2. 原子操作")
        report_lines.append("- 使用原子整数类型")
        report_lines.append("- 使用线程安全的队列")
        report_lines.append("- 避免共享可变状态")
        report_lines.append("")
        report_lines.append("### 3. 线程安全")
        report_lines.append("- 使用线程局部存储")
        report_lines.append("- 避免全局变量")
        report_lines.append("- 使用不可变对象")

        return "\n".join(report_lines)


def main():
    """主函数"""
    optimizer = ConcurrencyOptimizer("src/infrastructure")
    optimizations = optimizer.optimize_concurrent_code()

    print("并发优化分析完成")
    print(f"发现 {len(optimizations)} 个可优化文件")

    report = optimizer.generate_optimization_report(optimizations)

    with open("concurrency_optimization_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("优化报告已保存到: concurrency_optimization_report.md")


if __name__ == "__main__":
    main()

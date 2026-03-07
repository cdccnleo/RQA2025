#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康管理模块测试跳过用例全面检查脚本

系统性检查和修复所有可能的跳过测试情况
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class ComprehensiveHealthSkipChecker:
    """全面检查健康管理模块跳过测试"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_path = self.project_root / 'tests' / 'unit' / 'infrastructure' / 'health'

    def check_all_skip_patterns(self) -> Dict[str, Any]:
        """检查所有跳过模式"""
        results = {
            'pytest_skip_calls': [],
            'pytest_mark_skip_decorators': [],
            'pytest_mark_skipif_decorators': [],
            'unittest_skip_decorators': [],
            'runtime_skips': []
        }

        # 检查代码中的跳过调用
        skip_patterns = [
            ('pytest.skip', 'pytest_skip_calls'),
            ('@pytest.mark.skip', 'pytest_mark_skip_decorators'),
            ('@pytest.mark.skipif', 'pytest_mark_skipif_decorators'),
            ('@unittest.skip', 'unittest_skip_decorators')
        ]

        for pattern, key in skip_patterns:
            try:
                result = subprocess.run([
                    'grep', '-r', pattern, str(self.tests_path)
                ], capture_output=True, text=True, cwd=self.project_root)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    results[key] = [line for line in lines if line.strip()]
            except:
                # 如果grep不可用，使用Python查找
                results[key] = self.find_pattern_in_files(pattern, key)

        # 检查运行时跳过
        runtime_result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/unit/infrastructure/health/',
            '--maxfail=10', '--tb=no', '-q'
        ], capture_output=True, text=True, cwd=self.project_root)

        runtime_skips = []
        for line in runtime_result.stdout.split('\n'):
            if 'SKIPPED' in line:
                runtime_skips.append(line.strip())

        results['runtime_skips'] = runtime_skips

        return results

    def find_pattern_in_files(self, pattern: str, key: str) -> List[str]:
        """在文件中查找模式"""
        matches = []
        for py_file in self.tests_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            matches.append(f"{py_file.relative_to(self.project_root)}:{i}:{line.strip()}")
            except:
                continue
        return matches

    def analyze_skip_reasons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析跳过原因"""
        analysis = {
            'total_static_skips': 0,
            'total_runtime_skips': len(results['runtime_skips']),
            'skip_categories': {},
            'critical_issues': []
        }

        # 统计静态跳过
        for key in ['pytest_skip_calls', 'pytest_mark_skip_decorators',
                   'pytest_mark_skipif_decorators', 'unittest_skip_decorators']:
            count = len(results[key])
            analysis['total_static_skips'] += count
            analysis['skip_categories'][key] = count

        # 分析关键问题
        if analysis['total_runtime_skips'] > 0:
            analysis['critical_issues'].append(f"运行时发现{analysis['total_runtime_skips']}个跳过测试")

        if analysis['total_static_skips'] > 0:
            analysis['critical_issues'].append(f"代码中发现{analysis['total_static_skips']}个跳过调用")

        return analysis

    def generate_fix_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成修复计划"""
        plan = {
            'immediate_actions': [],
            'code_fixes': [],
            'test_fixes': [],
            'priority_order': []
        }

        if analysis['total_runtime_skips'] > 0:
            plan['immediate_actions'].append("修复运行时跳过测试")
            plan['priority_order'].append("runtime_fixes")

        if analysis['total_static_skips'] > 0:
            plan['code_fixes'].append("移除或修复静态跳过装饰器")
            plan['test_fixes'].append("实现缺失的功能以支持测试")
            plan['priority_order'].append("static_fixes")

        return plan

    def execute_comprehensive_check(self):
        """执行全面检查"""
        print("🔍 开始健康管理模块跳过测试全面检查...")
        print("=" * 60)

        # 1. 检查所有跳过模式
        results = self.check_all_skip_patterns()

        # 2. 分析跳过原因
        analysis = self.analyze_skip_reasons(results)

        # 3. 生成修复计划
        plan = self.generate_fix_plan(analysis)

        # 输出结果
        print("📊 检查结果:")
        print(f"  运行时跳过测试: {analysis['total_runtime_skips']}")
        print(f"  静态跳过调用: {analysis['total_static_skips']}")
        print(f"  跳过分类: {analysis['skip_categories']}")

        if analysis['critical_issues']:
            print("\n⚠️  关键问题:")
            for issue in analysis['critical_issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ 无跳过测试问题发现")

        print("\n🎯 修复计划:")
        if plan['immediate_actions']:
            print("  立即行动:")
            for action in plan['immediate_actions']:
                print(f"    • {action}")

        if plan['code_fixes']:
            print("  代码修复:")
            for fix in plan['code_fixes']:
                print(f"    • {fix}")

        if plan['test_fixes']:
            print("  测试修复:")
            for fix in plan['test_fixes']:
                print(f"    • {fix}")

        print("\n" + "=" * 60)

        # 保存详细报告
        self.save_detailed_report(results, analysis, plan)

        return analysis['total_runtime_skips'] == 0 and analysis['total_static_skips'] == 0

    def save_detailed_report(self, results: Dict[str, Any],
                           analysis: Dict[str, Any], plan: Dict[str, Any]):
        """保存详细报告"""
        report = f"""# 健康管理模块跳过测试全面检查报告

## 检查时间
{self.get_timestamp()}

## 检查结果

### 运行时跳过测试 ({len(results['runtime_skips'])})
"""
        for skip in results['runtime_skips']:
            report += f"- {skip}\n"

        report += f"""
### 静态跳过调用 ({analysis['total_static_skips']})

#### pytest.skip 调用 ({len(results['pytest_skip_calls'])})
"""
        for call in results['pytest_skip_calls']:
            report += f"- {call}\n"

        report += f"""
#### @pytest.mark.skip 装饰器 ({len(results['pytest_mark_skip_decorators'])})
"""
        for decorator in results['pytest_mark_skip_decorators']:
            report += f"- {decorator}\n"

        report += f"""
#### @pytest.mark.skipif 装饰器 ({len(results['pytest_mark_skipif_decorators'])})
"""
        for decorator in results['pytest_mark_skipif_decorators']:
            report += f"- {decorator}\n"

        report += f"""
#### @unittest.skip 装饰器 ({len(results['unittest_skip_decorators'])})
"""
        for decorator in results['unittest_skip_decorators']:
            report += f"- {decorator}\n"

        report += f"""
## 分析结果

- 总运行时跳过: {analysis['total_runtime_skips']}
- 总静态跳过: {analysis['total_static_skips']}
- 跳过分类: {analysis['skip_categories']}

## 修复计划

### 优先级顺序
{chr(10).join(f"- {priority}" for priority in plan['priority_order'])}

### 立即行动
{chr(10).join(f"- {action}" for action in plan['immediate_actions'])}

### 代码修复
{chr(10).join(f"- {fix}" for fix in plan['code_fixes'])}

### 测试修复
{chr(10).join(f"- {fix}" for fix in plan['test_fixes'])}
"""

        report_path = self.project_root / "HEALTH_SKIP_COMPREHENSIVE_CHECK.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 详细报告已保存: {report_path}")

    def get_timestamp(self):
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def run_final_validation(self):
        """运行最终验证"""
        print("\n🔍 运行最终验证测试...")

        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/unit/infrastructure/health/',
            '--maxfail=5', '--tb=no', '-q'
        ], capture_output=True, text=True, cwd=self.project_root)

        final_skips = 0
        for line in result.stdout.split('\n'):
            if 'SKIPPED' in line:
                final_skips += 1

        print(f"最终验证结果: {final_skips} 个跳过测试")

        if final_skips == 0:
            print("🎉 跳过测试问题已完全解决！")
            return True
        else:
            print(f"⚠️ 仍有 {final_skips} 个跳过测试需要处理")
            return False


def main():
    """主函数"""
    checker = ComprehensiveHealthSkipChecker()

    # 执行全面检查
    is_clean = checker.execute_comprehensive_check()

    # 运行最终验证
    final_result = checker.run_final_validation()

    if is_clean and final_result:
        print("\n✅ 健康管理模块跳过测试问题检查完成 - 无问题")
        return 0
    else:
        print("\n⚠️  发现跳过测试问题，需要修复")
        return 1


if __name__ == "__main__":
    exit(main())

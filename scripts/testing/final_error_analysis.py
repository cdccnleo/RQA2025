#!/usr/bin/env python3
"""
最终错误分析脚本

详细分析基础设施层测试的剩余错误，为100%可收集目标做准备
"""

import subprocess
import sys
from collections import defaultdict, Counter


class FinalErrorAnalyzer:
    """最终错误分析器"""

    def __init__(self):
        self.error_details = {}
        self.error_patterns = defaultdict(list)
        self.error_summary = Counter()

    def analyze_all_errors(self):
        """分析所有错误"""
        print("最终错误深度分析")
        print("=" * 70)

        # 分析各个子模块
        submodules = {
            'cache': 'tests/unit/infrastructure/cache/',
            'config': 'tests/unit/infrastructure/config/',
            'error': 'tests/unit/infrastructure/error/',
            'health': 'tests/unit/infrastructure/health/',
            'logging': 'tests/unit/infrastructure/logging/',
            'resource': 'tests/unit/infrastructure/resource/',
            'security': 'tests/unit/infrastructure/security/'
        }

        for name, path in submodules.items():
            print(f"\n分析 {name} 模块...")
            errors = self.get_detailed_errors(path)
            if errors:
                self.error_details[name] = errors
                self.categorize_errors(errors, name)
                print(f"   找到 {len(errors)} 个具体错误")

        # 分析根目录
        print(f"\n分析根目录...")
        root_errors = self.get_detailed_errors(
            'tests/unit/infrastructure/',
            exclude_filters=['cache', 'config', 'error',
                             'health', 'logging', 'resource', 'security']
        )
        if root_errors:
            self.error_details['root'] = root_errors
            self.categorize_errors(root_errors, 'root')
            print(f"   找到 {len(root_errors)} 个具体错误")

    def get_detailed_errors(self, path, exclude_filters=None):
        """获取详细错误信息"""
        cmd = [sys.executable, '-m', 'pytest', path, '--collect-only', '-v']

        if exclude_filters:
            exclude_expr = ' and '.join([f'not {f}' for f in exclude_filters])
            cmd.extend(['-k', exclude_expr])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30
            )

            errors = []
            for line in result.stdout.split('\n'):
                if 'ERROR' in line and ('collecting' in line or 'failed' in line):
                    errors.append(line.strip())

            return errors

        except subprocess.TimeoutExpired:
            return ['ERROR: 命令超时']
        except Exception as e:
            return [f'ERROR: 执行失败 - {str(e)}']

    def categorize_errors(self, errors, module_name):
        """分类错误"""
        for error in errors:
            error_lower = error.lower()

            # ModuleNotFoundError
            if 'modulenotfounderror' in error_lower or 'no module named' in error_lower:
                self.error_patterns['ModuleNotFoundError'].append((module_name, error))
                self.error_summary['ModuleNotFoundError'] += 1

            # ImportError
            elif 'importerror' in error_lower:
                self.error_patterns['ImportError'].append((module_name, error))
                self.error_summary['ImportError'] += 1

            # SyntaxError
            elif 'syntaxerror' in error_lower:
                self.error_patterns['SyntaxError'].append((module_name, error))
                self.error_summary['SyntaxError'] += 1

            # IndentationError
            elif 'indentationerror' in error_lower:
                self.error_patterns['IndentationError'].append((module_name, error))
                self.error_summary['IndentationError'] += 1

            # AttributeError
            elif 'attributeerror' in error_lower:
                self.error_patterns['AttributeError'].append((module_name, error))
                self.error_summary['AttributeError'] += 1

            # NameError
            elif 'nameerror' in error_lower:
                self.error_patterns['NameError'].append((module_name, error))
                self.error_summary['NameError'] += 1

            # PytestCollectionWarning
            elif 'pytestcollectionwarning' in error_lower:
                self.error_patterns['PytestCollectionWarning'].append((module_name, error))
                self.error_summary['PytestCollectionWarning'] += 1

            # 其他错误
            else:
                self.error_patterns['Other'].append((module_name, error))
                self.error_summary['Other'] += 1

    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "="*70)
        print("最终错误分析报告")
        print("="*70)

        # 总体统计
        total_errors = sum(self.error_summary.values())
        print(f"\n总体统计:")
        print(f"   总错误数: {total_errors}")
        print(f"   错误类型数: {len(self.error_summary)}")
        print(f"   受影响模块数: {len(self.error_details)}")

        # 错误类型分布
        print(f"\n错误类型分布:")
        for error_type, count in self.error_summary.most_common():
            percentage = (count / total_errors) * 100
            print(f"   {error_type}: {count} 个 ({percentage:.1f}%)")

        # 模块错误分布
        print(f"\n模块错误分布:")
        for module, errors in self.error_details.items():
            print(f"   {module}: {len(errors)} 个错误")

        # 详细错误模式分析
        print(f"\n主要错误模式分析:")
        for error_type, patterns in self.error_patterns.items():
            if patterns:
                print(f"\n   {error_type} ({len(patterns)} 个):")

                # 分析模块分布
                module_count = Counter()
                for module, _ in patterns:
                    module_count[module] += 1

                for module, count in module_count.most_common(3):
                    print(f"      {module}: {count} 个")

                # 显示几个示例
                if len(patterns) <= 5:
                    for i, (module, error) in enumerate(patterns[:3]):
                        short_error = error[:80] + "..." if len(error) > 80 else error
                        print(f"        示例 {i+1}: {short_error}")
                else:
                    print(f"      ... 还有 {len(patterns) - 3} 个类似错误")

    def generate_fix_strategy(self):
        """生成修复策略"""
        print("\n" + "="*70)
        print("100%可收集修复策略")
        print("="*70)

        print("\n优先级修复顺序:")

        priority_strategies = [
            ('ModuleNotFoundError', '创建缺失模块', '分析具体缺失模块并创建'),
            ('ImportError', '修复导入路径', '统一导入语句和路径'),
            ('SyntaxError', '修复语法错误', '检查括号、引号、语句结构'),
            ('IndentationError', '修复缩进问题', '统一缩进风格'),
            ('PytestCollectionWarning', '修复测试类', '处理测试类结构问题'),
            ('AttributeError', '修复属性访问', '检查对象属性和方法'),
            ('NameError', '修复名称定义', '检查变量和函数定义'),
            ('Other', '处理其他错误', '逐个分析和修复')
        ]

        for i, (error_type, strategy, detail) in enumerate(priority_strategies, 1):
            count = self.error_summary.get(error_type, 0)
            if count > 0:
                print(f"\n{i}. {error_type} ({count} 个)")
                print(f"   策略: {strategy}")
                print(f"   方法: {detail}")
                print(f"   目标: 全部修复")

        print(f"\n最终目标:")
        print(f"   基础设施层测试100%可收集")
        print(f"   错误数量从 {sum(self.error_summary.values())} 降至 0")
        print(f"   为业务层测试用例补充奠定基础")


def main():
    """主函数"""
    analyzer = FinalErrorAnalyzer()
    analyzer.analyze_all_errors()
    analyzer.generate_comprehensive_report()
    analyzer.generate_fix_strategy()


if __name__ == "__main__":
    main()

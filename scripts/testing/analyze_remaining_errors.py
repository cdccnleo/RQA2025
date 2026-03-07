#!/usr/bin/env python3
"""
分析剩余错误类型的脚本

详细分析基础设施层测试中剩余错误的类型和模式
"""

import subprocess
import sys
from collections import defaultdict, Counter


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self):
        self.error_patterns = defaultdict(list)
        self.error_types = Counter()
        self.module_errors = defaultdict(list)

    def analyze_infrastructure_errors(self):
        """分析基础设施层错误"""
        print("🔍 分析基础设施层错误...")

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
            print(f"\n📁 分析 {name} 模块...")
            errors = self.run_pytest_collect(path)
            if errors:
                self.module_errors[name] = errors
                self.categorize_errors(errors, name)

        # 分析根目录测试
        print(f"\n📁 分析根目录测试...")
        root_errors = self.run_pytest_collect(
            'tests/unit/infrastructure/',
            exclude_filters=['cache', 'config', 'error',
                             'health', 'logging', 'resource', 'security']
        )
        if root_errors:
            self.module_errors['root'] = root_errors
            self.categorize_errors(root_errors, 'root')

    def run_pytest_collect(self, path, exclude_filters=None):
        """运行pytest收集错误"""
        cmd = [sys.executable, '-m', 'pytest', path, '--collect-only', '--quiet']

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
                timeout=60
            )

            errors = []
            for line in result.stdout.split('\n'):
                if line.strip().startswith('ERROR'):
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

            # ModuleNotFoundError - 模块不存在
            if 'modulenotfounderror' in error_lower or 'no module named' in error_lower:
                self.error_patterns['ModuleNotFoundError'].append((module_name, error))
                self.error_types['ModuleNotFoundError'] += 1

            # ImportError - 导入错误
            elif 'importerror' in error_lower:
                self.error_patterns['ImportError'].append((module_name, error))
                self.error_types['ImportError'] += 1

            # SyntaxError - 语法错误
            elif 'syntaxerror' in error_lower:
                self.error_patterns['SyntaxError'].append((module_name, error))
                self.error_types['SyntaxError'] += 1

            # IndentationError - 缩进错误
            elif 'indentationerror' in error_lower:
                self.error_patterns['IndentationError'].append((module_name, error))
                self.error_types['IndentationError'] += 1

            # AttributeError - 属性错误
            elif 'attributeerror' in error_lower:
                self.error_patterns['AttributeError'].append((module_name, error))
                self.error_types['AttributeError'] += 1

            # NameError - 名称错误
            elif 'nameerror' in error_lower:
                self.error_patterns['NameError'].append((module_name, error))
                self.error_types['NameError'] += 1

            # AssertionError - 断言错误
            elif 'assertionerror' in error_lower:
                self.error_patterns['AssertionError'].append((module_name, error))
                self.error_types['AssertionError'] += 1

            # PytestCollectionWarning - 收集警告
            elif 'pytestcollectionwarning' in error_lower:
                self.error_patterns['PytestCollectionWarning'].append((module_name, error))
                self.error_types['PytestCollectionWarning'] += 1

            # 其他错误
            else:
                self.error_patterns['Other'].append((module_name, error))
                self.error_types['Other'] += 1

    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*80)
        print("📊 错误分析报告")
        print("="*80)

        # 总体统计
        total_errors = sum(self.error_types.values())
        print(f"\n🔢 总体统计:")
        print(f"   总错误数: {total_errors}")
        print(f"   错误类型数: {len(self.error_types)}")
        print(f"   受影响模块数: {len(self.module_errors)}")

        # 错误类型分布
        print(f"\n错误类型分布:")
        for error_type, count in self.error_types.most_common():
            percentage = (count / total_errors) * 100
            print(f"   {error_type}: {count} 个 ({percentage:.1f}%)")
        # 模块错误分布
        print(f"\n模块错误分布:")
        for module, errors in self.module_errors.items():
            print(f"   {module}: {len(errors)} 个错误")
        # 详细错误模式分析
        print(f"\n🔍 主要错误模式分析:")
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
                        # 简化错误信息显示
                        short_error = error[:100] + "..." if len(error) > 100 else error
                        print(f"        示例 {i+1}: {short_error}")
                else:
                    print(f"      ... 还有 {len(patterns) - 3} 个类似错误")

        # 修复建议
        print(f"\n💡 修复建议:")
        if self.error_types['ModuleNotFoundError'] > 0:
            print("   1. ModuleNotFoundError: 检查模块路径和导入语句")
        if self.error_types['ImportError'] > 0:
            print("   2. ImportError: 修复导入路径和依赖关系")
        if self.error_types['SyntaxError'] > 0:
            print("   3. SyntaxError: 检查语法错误和代码结构")
        if self.error_types['IndentationError'] > 0:
            print("   4. IndentationError: 修复缩进和代码格式")
        if self.error_types['PytestCollectionWarning'] > 0:
            print("   5. PytestCollectionWarning: 修复测试类结构问题")


def main():
    """主函数"""
    analyzer = ErrorAnalyzer()
    analyzer.analyze_infrastructure_errors()
    analyzer.generate_report()


if __name__ == "__main__":
    main()

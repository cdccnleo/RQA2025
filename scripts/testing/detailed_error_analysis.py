#!/usr/bin/env python3
"""
详细错误分析脚本

获取基础设施层测试中剩余错误的具体信息
"""

import subprocess
import sys
from collections import defaultdict


def get_detailed_errors():
    """获取详细错误信息"""
    print("🔍 获取详细错误信息...")

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

    detailed_errors = {}

    for name, path in submodules.items():
        print(f"\n📁 获取 {name} 模块错误详情...")
        errors = run_pytest_collect(path)
        if errors:
            detailed_errors[name] = errors
            print(f"   找到 {len(errors)} 个错误")

    # 分析根目录测试
    print(f"\n📁 获取根目录测试错误详情...")
    root_errors = run_pytest_collect(
        'tests/unit/infrastructure/',
        exclude_filters=['cache', 'config', 'error', 'health', 'logging', 'resource', 'security']
    )
    if root_errors:
        detailed_errors['root'] = root_errors
        print(f"   找到 {len(root_errors)} 个错误")

    return detailed_errors


def run_pytest_collect(path, exclude_filters=None):
    """运行pytest收集错误"""
    cmd = [sys.executable, '-m', 'pytest', path, '--collect-only']

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

        return result.stdout

    except subprocess.TimeoutExpired:
        return "ERROR: 命令超时\n"
    except Exception as e:
        return f"ERROR: 执行失败 - {str(e)}\n"


def analyze_error_patterns(detailed_errors):
    """分析错误模式"""
    print("\n" + "="*80)
    print("🔍 错误模式分析")
    print("="*80)

    total_errors = 0
    error_categories = defaultdict(int)
    sample_errors = {}

    for module, output in detailed_errors.items():
        lines = output.split('\n')
        module_errors = 0
        module_samples = []

        for line in lines:
            if 'ERROR' in line:
                module_errors += 1
                total_errors += 1

                # 分类错误
                line_lower = line.lower()
                if 'modulenotfounderror' in line_lower or 'no module named' in line_lower:
                    error_categories['ModuleNotFoundError'] += 1
                elif 'importerror' in line_lower:
                    error_categories['ImportError'] += 1
                elif 'syntaxerror' in line_lower:
                    error_categories['SyntaxError'] += 1
                elif 'indentationerror' in line_lower:
                    error_categories['IndentationError'] += 1
                elif 'attributeerror' in line_lower:
                    error_categories['AttributeError'] += 1
                elif 'nameerror' in line_lower:
                    error_categories['NameError'] += 1
                elif 'pytestcollectionwarning' in line_lower:
                    error_categories['PytestCollectionWarning'] += 1
                else:
                    error_categories['Other'] += 1

                # 收集示例错误
                if len(module_samples) < 3:
                    module_samples.append(line.strip())

        sample_errors[module] = module_samples
        print(f"\n📁 {module} 模块 ({module_errors} 个错误):")
        for i, sample in enumerate(module_samples[:2]):
            # 简化错误信息显示
            short_error = sample[:120] + "..." if len(sample) > 120 else sample
            print(f"   示例 {i+1}: {short_error}")

    print(f"\n" + "-"*80)
    print(f"📊 错误统计:")
    print(f"   总错误数: {total_errors}")
    print(f"   错误类型分布:")

    for error_type, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_errors) * 100 if total_errors > 0 else 0
        print(f"   - {error_type}: {count} 个 ({percentage:.1f}%)")

    return error_categories, sample_errors


def generate_fix_strategy(error_categories, sample_errors):
    """生成修复策略"""
    print(f"\n" + "="*80)
    print("🎯 修复策略建议")
    print("="*80)

    # 按优先级排序修复策略
    priority_order = [
        'ModuleNotFoundError',
        'ImportError',
        'SyntaxError',
        'IndentationError',
        'PytestCollectionWarning',
        'AttributeError',
        'NameError',
        'Other'
    ]

    strategy_count = 1
    for error_type in priority_order:
        if error_type in error_categories and error_categories[error_type] > 0:
            count = error_categories[error_type]

            print(f"\n{strategy_count}. {error_type} ({count} 个)")
            strategy_count += 1

            if error_type == 'ModuleNotFoundError':
                print("   策略: 检查并创建缺失的模块")
                print("   方法: 分析导入路径，创建占位模块或mock对象")
            elif error_type == 'ImportError':
                print("   策略: 修复导入路径和依赖关系")
                print("   方法: 统一导入语句格式，处理循环导入")
            elif error_type == 'SyntaxError':
                print("   策略: 修复语法错误")
                print("   方法: 检查括号匹配、字符串引号、语句结束符")
            elif error_type == 'IndentationError':
                print("   策略: 修复缩进问题")
                print("   方法: 统一缩进风格（4空格），修复不一致缩进")
            elif error_type == 'PytestCollectionWarning':
                print("   策略: 修复测试类结构问题")
                print("   方法: 为测试类添加__init__方法或使用__test__=False")
            elif error_type == 'AttributeError':
                print("   策略: 修复属性访问问题")
                print("   方法: 检查对象属性是否存在，添加必要的属性")
            elif error_type == 'NameError':
                print("   策略: 修复名称未定义问题")
                print("   方法: 检查变量作用域，添加必要的导入或定义")
            else:
                print("   策略: 处理其他类型错误")
                print("   方法: 逐个分析和修复")


def main():
    """主函数"""
    # 获取详细错误信息
    detailed_errors = get_detailed_errors()

    # 分析错误模式
    error_categories, sample_errors = analyze_error_patterns(detailed_errors)

    # 生成修复策略
    generate_fix_strategy(error_categories, sample_errors)

    print(f"\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)


if __name__ == "__main__":
    main()

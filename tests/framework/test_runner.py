#!/usr/bin/env python3
"""
测试运行器 - 命令行接口

提供命令行接口来运行统一测试框架，支持分层测试执行和报告生成。
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import sys
from pathlib import Path

# 添加framework目录到路径
_framework_dir = Path(__file__).parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from test_executor import get_test_executor, TestResult


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RQA2025统一测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行基础设施层测试
  python test_runner.py infrastructure

  # 运行多个层级测试
  python test_runner.py infrastructure core data

  # 运行所有层级测试
  python test_runner.py --all

  # 运行测试并生成覆盖率报告
  python test_runner.py infrastructure --coverage

  # 并行执行测试
  python test_runner.py infrastructure --parallel --workers 4

  # 生成详细报告
  python test_runner.py infrastructure --report test_logs/custom_report.md
        """
    )

    parser.add_argument(
        'layers',
        nargs='*',
        help='要测试的层级 (infrastructure, core, data, features, ml, optimization, strategy, trading, risk, monitoring, gateway)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有层级的测试'
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help='生成覆盖率报告'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='启用并行执行'
    )

    parser.add_argument(
        '--workers',
        type=str,
        default='auto',
        help='并行工作进程数 (默认: auto)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default='test_logs/test_execution_report.md',
        help='报告输出文件路径'
    )

    parser.add_argument(
        '--json-report',
        type=str,
        default='test_logs/test_results.json',
        help='JSON报告输出文件路径'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式'
    )

    parser.add_argument(
        '--markers',
        type=str,
        help='pytest标记过滤器'
    )

    parser.add_argument(
        '--list-layers',
        action='store_true',
        help='列出所有可用层级'
    )

    return parser.parse_args()


def list_available_layers():
    """列出所有可用层级"""
    layers = [
        'infrastructure', 'core', 'data', 'features', 'ml',
        'optimization', 'strategy', 'trading', 'risk', 'monitoring', 'gateway'
    ]

    print("可用测试层级:")
    print("=" * 50)
    for i, layer in enumerate(layers, 1):
        print("2d")
    print("=" * 50)


def validate_layers(layers: List[str]) -> List[str]:
    """验证层级名称"""
    valid_layers = {
        'infrastructure', 'core', 'data', 'features', 'ml',
        'optimization', 'strategy', 'trading', 'risk', 'monitoring', 'gateway'
    }

    invalid_layers = [layer for layer in layers if layer not in valid_layers]
    if invalid_layers:
        print(f"错误: 无效的层级名称: {', '.join(invalid_layers)}")
        print("使用 --list-layers 查看可用层级")
        sys.exit(1)

    return layers


def run_tests(args: argparse.Namespace) -> Dict[str, TestResult]:
    """运行测试"""
    executor = get_test_executor()

    # 确定要测试的层级
    if args.all:
        layers = None  # 执行所有层级
    elif args.layers:
        layers = validate_layers(args.layers)
    else:
        print("错误: 请指定要测试的层级或使用 --all")
        print("使用 -h 查看帮助信息")
        sys.exit(1)

    # 构建执行参数
    execution_kwargs = {
        'coverage': args.coverage,
        'parallel': args.parallel,
        'workers': args.workers,
        'verbose': args.verbose,
        'quiet': args.quiet
    }

    if args.markers:
        execution_kwargs['markers'] = args.markers

    # 执行测试
    print("开始执行测试...")
    print(f"测试层级: {layers if layers else '所有层级'}")
    print(f"覆盖率: {'启用' if args.coverage else '禁用'}")
    print(f"并行执行: {'启用' if args.parallel else '禁用'}")
    print("-" * 50)

    results = executor.execute_all_layers(layers, **execution_kwargs)

    return results


def display_results(results: Dict[str, TestResult]):
    """显示测试结果"""
    print("\n" + "=" * 60)
    print("测试执行结果汇总")
    print("=" * 60)

    # 总体统计
    total_tests = sum(r.total_tests for r in results.values())
    total_passed = sum(r.passed_tests for r in results.values())
    total_failed = sum(r.failed_tests for r in results.values())
    total_skipped = sum(r.skipped_tests for r in results.values())
    total_errors = sum(r.errors for r in results.values())
    total_time = sum(r.execution_time for r in results.values())

    print("\n总体统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {total_passed}")
    print(f"  失败测试: {total_failed}")
    print(f"  跳过测试: {total_skipped}")
    print(f"  错误数: {total_errors}")
    print(".2")
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(".1")
    print("\n分层结果:")
    print("-" * 40)

    for layer_name, result in results.items():
        status_icon = "✅" if result.is_success else "❌"
        print("15"
              "4d"
              ".2")
        if result.coverage is not None:
            print(".1")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_arguments()

    # 处理特殊命令
    if args.list_layers:
        list_available_layers()
        return

    try:
        # 运行测试
        results = run_tests(args)

        # 显示结果
        display_results(results)

        # 生成报告
        executor = get_test_executor()
        report_content = executor.generate_report(results, args.report)
        executor.save_results_json(results, args.json_report)

        print(f"\n详细报告已保存到: {args.report}")
        print(f"JSON结果已保存到: {args.json_report}")

        # 返回适当的退出码
        has_failures = any(not result.is_success for result in results.values())
        sys.exit(1 if has_failures else 0)

    except KeyboardInterrupt:
        print("\n测试执行被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

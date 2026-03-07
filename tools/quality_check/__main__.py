"""
质量检查工具主入口

命令行接口，用于运行质量检查。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .core.quality_checker import QualityChecker
from .reporters.console_reporter import ConsoleReporter
from .reporters.json_reporter import JsonReporter
from .reporters.html_reporter import HtmlReporter
from .config.default_config import get_config, merge_config


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args)

        # 创建检查器
        checker = QualityChecker(config)

        # 执行检查
        print("🚀 开始质量检查...")
        results = checker.run_quality_check(args.target)

        # 生成报告
        generate_reports(results, config)

        # 检查是否应该失败
        should_fail = results.get('summary', {}).get('should_fail', False)

        # 输出最终状态
        if should_fail:
            print("\n❌ 检查失败，发现严重问题")
            sys.exit(1)
        else:
            print("\n✅ 检查通过")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断检查")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 检查过程中发生错误: {e}")
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="基础设施层质量检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 检查当前目录
  python -m tools.quality_check .

  # 检查特定目录
  python -m tools.quality_check src/infrastructure/cache

  # 使用基础设施配置
  python -m tools.quality_check --config infrastructure .

  # 生成所有报告
  python -m tools.quality_check --reports all .

  # 详细输出
  python -m tools.quality_check --verbose .
        """
    )

    parser.add_argument(
        'target',
        help='检查目标路径'
    )

    parser.add_argument(
        '--config',
        choices=['default', 'infrastructure'],
        default='infrastructure',
        help='配置类型 (默认: infrastructure)'
    )

    parser.add_argument(
        '--config-file',
        type=str,
        help='自定义配置文件路径'
    )

    parser.add_argument(
        '--reports',
        choices=['console', 'json', 'html', 'all'],
        default='all',
        help='报告类型 (默认: all)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='报告输出目录 (默认: 当前目录)'
    )

    parser.add_argument(
        '--checkers',
        nargs='+',
        choices=['duplicate', 'interface', 'complexity'],
        help='指定要运行的检查器'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--no-color',
        action='store_true',
        help='禁用控制台颜色输出'
    )

    return parser


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    加载配置

    Args:
        args: 命令行参数

    Returns:
        Dict[str, Any]: 配置字典
    """
    # 加载基础配置
    config = get_config(args.config)

    # 如果指定了配置文件，合并配置
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        if file_config:
            config = merge_config(config, file_config)

    # 应用命令行参数覆盖
    apply_cli_overrides(config, args)

    return config


def load_config_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从文件加载配置

    Args:
        file_path: 配置文件路径

    Returns:
        Optional[Dict[str, Any]]: 配置字典
    """
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  加载配置文件失败 {file_path}: {e}")
        return None


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    应用命令行参数覆盖

    Args:
        config: 配置字典
        args: 命令行参数
    """
    # 检查器选择
    if args.checkers:
        config['enabled_checkers'] = args.checkers

    # 报告设置
    if args.reports != 'all':
        # 禁用所有报告
        for reporter_name in config['reporters']:
            config['reporters'][reporter_name]['enabled'] = False

        # 启用指定报告
        if args.reports in config['reporters']:
            config['reporters'][args.reports]['enabled'] = True

    # 报告输出目录
    output_dir = Path(args.output_dir)
    for reporter_name, reporter_config in config['reporters'].items():
        if 'output_file' in reporter_config:
            original_path = Path(reporter_config['output_file'])
            new_path = output_dir / original_path.name
            reporter_config['output_file'] = str(new_path)

    # 控制台设置
    if args.verbose:
        config['reporters']['console']['verbose'] = True

    if args.no_color:
        config['reporters']['console']['colors'] = False


def generate_reports(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    生成报告

    Args:
        results: 检查结果
        config: 配置
    """
    reporters_config = config.get('reporters', {})

    try:
        # 控制台报告
        if reporters_config.get('console', {}).get('enabled', True):
            console_config = reporters_config['console']
            reporter = ConsoleReporter(console_config)
            reporter.report(results['results'])
    except Exception as e:
        print(f"⚠️  控制台报告生成失败: {e}")

    try:
        # JSON报告
        if reporters_config.get('json', {}).get('enabled', False):
            json_config = reporters_config['json']
            reporter = JsonReporter(json_config)
            json_output = reporter.report(results.get('results', {}))
            print(f"📄 JSON报告已生成: {json_config.get('output_file', 'quality_report.json')}")
    except Exception as e:
        print(f"⚠️  JSON报告生成失败: {e}")

    try:
        # HTML报告
        if reporters_config.get('html', {}).get('enabled', False):
            html_config = reporters_config['html']
            reporter = HtmlReporter(html_config)
            html_output = reporter.report(results.get('results', {}))
            print(f"🌐 HTML报告已生成: {html_config.get('output_file', 'quality_report.html')}")
    except Exception as e:
        print(f"⚠️  HTML报告生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

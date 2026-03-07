#!/usr/bin/env python3
"""
自动化重构工具命令行入口

提供命令行接口来执行自动化重构操作。
"""

import sys
import argparse
import json
from pathlib import Path

from .core import AutoRefactorEngine, RefactorConfig
from tools.smart_code_analyzer import SmartCodeAnalyzer


def main():
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # 创建配置
        config = create_config_from_args(args)

        # 验证配置
        config_errors = config.validate_config()
        if config_errors:
            print("❌ 配置错误:", file=sys.stderr)
            for error in config_errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

        # 显示配置信息
        if config.verbose_output:
            print_config_info(config)

        # 执行命令
        if args.command == 'analyze-and-refactor':
            exit_code = handle_analyze_and_refactor(args, config)
        elif args.command == 'refactor':
            exit_code = handle_refactor(args, config)
        elif args.command == 'generate':
            exit_code = handle_generate(args, config)
        elif args.command == 'validate':
            exit_code = handle_validate(args, config)
        elif args.command == 'backup':
            exit_code = handle_backup(args, config)
        else:
            parser.print_help()
            exit_code = 1

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  操作被用户中断", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ 发生错误: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""

    parser = argparse.ArgumentParser(
        description="自动化重构工具 - 智能代码重构执行引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析并自动重构
  python -m tools.auto_refactor analyze-and-refactor src/

  # 仅执行重构（需要先分析）
  python -m tools.auto_refactor refactor --safe src/

  # 生成代码
  python -m tools.auto_refactor generate --type method --name my_method

  # 验证重构结果
  python -m tools.auto_refactor validate src/modified_file.py

  # 备份管理
  python -m tools.auto_refactor backup --create src/file.py
        """
    )

    # 全局选项
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细输出'
    )

    parser.add_argument(
        '--config',
        help='指定配置文件路径'
    )

    parser.add_argument(
        '--preset',
        choices=['safe', 'fast', 'balanced', 'ci', 'experimental'],
        help='使用预设配置'
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # analyze-and-refactor 命令
    analyze_parser = subparsers.add_parser(
        'analyze-and-refactor',
        help='分析代码并执行自动重构'
    )
    analyze_parser.add_argument(
        'path',
        help='要分析的路径'
    )
    analyze_parser.add_argument(
        '--safety',
        choices=['low', 'medium', 'high'],
        default='high',
        help='安全级别 (默认: high)'
    )
    analyze_parser.add_argument(
        '--report',
        help='指定报告输出路径'
    )
    analyze_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，不实际执行重构'
    )

    # refactor 命令
    refactor_parser = subparsers.add_parser(
        'refactor',
        help='执行重构操作'
    )
    refactor_parser.add_argument(
        'path',
        help='目标路径'
    )
    refactor_parser.add_argument(
        '--types',
        help='指定重构类型 (逗号分隔)'
    )
    refactor_parser.add_argument(
        '--safe',
        action='store_true',
        help='安全模式执行'
    )
    refactor_parser.add_argument(
        '--parallel',
        type=int,
        help='并行执行的工作线程数'
    )

    # generate 命令
    generate_parser = subparsers.add_parser(
        'generate',
        help='生成代码'
    )
    generate_parser.add_argument(
        '--type',
        choices=['method', 'class', 'test', 'interface'],
        required=True,
        help='生成代码类型'
    )
    generate_parser.add_argument(
        '--name',
        required=True,
        help='生成的项目名称'
    )
    generate_parser.add_argument(
        '--output',
        help='输出文件路径'
    )

    # validate 命令
    validate_parser = subparsers.add_parser(
        'validate',
        help='验证重构结果'
    )
    validate_parser.add_argument(
        'files',
        nargs='+',
        help='要验证的文件'
    )
    validate_parser.add_argument(
        '--syntax-only',
        action='store_true',
        help='仅验证语法'
    )

    # backup 命令
    backup_parser = subparsers.add_parser(
        'backup',
        help='备份管理'
    )
    backup_parser.add_argument(
        '--create',
        metavar='FILE',
        help='创建文件备份'
    )
    backup_parser.add_argument(
        '--rollback',
        metavar='FILE',
        help='回滚文件到备份版本'
    )
    backup_parser.add_argument(
        '--cleanup',
        action='store_true',
        help='清理旧备份文件'
    )

    return parser


def create_config_from_args(args: argparse.Namespace) -> RefactorConfig:
    """从命令行参数创建配置"""

    # 使用预设配置或默认配置
    if args.preset:
        config = RefactorConfig.from_preset(args.preset)
    else:
        config = RefactorConfig()

    # 从配置文件加载（如果指定）
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = RefactorConfig.from_dict(config_dict)

    # 应用命令行参数覆盖
    config.verbose_output = args.verbose

    # 命令特定的配置
    if hasattr(args, 'safety') and args.safety:
        from .core.config import SafetyLevel
        config.safety_level = SafetyLevel(args.safety)

    if hasattr(args, 'dry_run') and args.dry_run:
        config.dry_run = True

    if hasattr(args, 'parallel') and args.parallel:
        config.parallel_processing = True
        config.max_workers = args.parallel

    return config


def print_config_info(config: RefactorConfig):
    """打印配置信息"""

    print("🔧 当前配置:")
    print(f"  安全级别: {config.safety_level.value}")
    print(f"  备份启用: {config.backup_enabled}")
    print(f"  验证启用: {config.validation_enabled}")
    print(f"  并行处理: {config.parallel_processing}")
    print(f"  最大工作线程: {config.max_workers}")
    print(f"  试运行模式: {config.dry_run}")
    print(f"  执行策略: {config.get_execution_strategy()}")
    print(f"  风险评分: {config.get_risk_score():.2f}")
    print()


def handle_analyze_and_refactor(args: argparse.Namespace, config: RefactorConfig) -> int:
    """处理分析并重构命令"""

    print(f"🔍 开始分析和重构: {args.path}")

    try:
        # 创建分析器和重构引擎
        analyzer = SmartCodeAnalyzer()
        engine = AutoRefactorEngine(config)

        # 执行分析
        print("📊 执行代码分析...")
        analysis_results = analyzer.analyze_project(args.path)

        if not analysis_results:
            print("⚠️  没有找到可分析的文件")
            return 1

        print(f"✅ 分析完成，发现 {len(analysis_results)} 个文件")

        # 显示分析摘要
        total_suggestions = sum(len(r.suggestions) for r in analysis_results.values())
        print(f"📋 发现 {total_suggestions} 个重构建议")

        if total_suggestions == 0:
            print("🎉 代码质量良好，无需重构")
            return 0

        # 执行重构
        print("🔧 开始执行重构...")
        refactor_results = engine.execute_auto_refactor(analysis_results, config.safety_level.value)

        # 显示结果
        successful_refactors = sum(len(results) for results in refactor_results.values()
                                   if results and any(r.success for r in results))
        failed_refactors = sum(len(results) for results in refactor_results.values()
                               if results and any(not r.success for r in results))

        print(f"✅ 重构完成: {successful_refactors} 成功, {failed_refactors} 失败")

        # 生成报告
        if config.generate_report:
            report_path = args.report or f"refactor_report_{Path(args.path).name}.json"
            report = engine.generate_refactor_report(refactor_results)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"📄 报告已保存到: {report_path}")

        return 0 if successful_refactors > 0 else 1

    except Exception as e:
        print(f"❌ 操作失败: {e}", file=sys.stderr)
        return 1


def handle_refactor(args: argparse.Namespace, config: RefactorConfig) -> int:
    """处理重构命令"""

    print(f"🔧 开始执行重构: {args.path}")

    # 这里需要实现从现有分析结果加载建议的逻辑
    # 暂时返回提示信息
    print("⚠️  此命令需要先运行分析命令生成建议文件")
    print("💡 建议使用: analyze-and-refactor 命令")

    return 1


def handle_generate(args: argparse.Namespace, config: RefactorConfig) -> int:
    """处理生成命令"""

    print(f"🎨 生成{args.type}: {args.name}")

    # 这里需要实现代码生成逻辑
    # 暂时返回提示信息
    print("⚠️  代码生成功能正在开发中")
    print(f"📝 将生成 {args.type} 类型的代码: {args.name}")

    if args.output:
        print(f"📄 输出文件: {args.output}")

    return 0


def handle_validate(args: argparse.Namespace, config: RefactorConfig) -> int:
    """处理验证命令"""

    print(f"🔍 开始验证文件: {', '.join(args.files)}")

    from .core.safety_manager import ValidationManager

    validator = ValidationManager(config)
    all_success = True

    for file_path in args.files:
        print(f"  验证: {file_path}")

        if args.syntax_only:
            result = validator.validate_syntax(file_path)
        else:
            result = validator.run_all_validations(file_path)

        if result.success:
            print("    ✅ 验证通过")
            if result.warnings:
                print(f"    ⚠️  警告: {len(result.warnings)} 个")
                for warning in result.warnings[:3]:  # 只显示前3个
                    print(f"      - {warning}")
        else:
            print("    ❌ 验证失败")
            all_success = False
            for error in result.errors[:3]:  # 只显示前3个
                print(f"      - {error}")

    return 0 if all_success else 1


def handle_backup(args: argparse.Namespace, config: RefactorConfig) -> int:
    """处理备份命令"""

    from .core.safety_manager import BackupManager

    backup_manager = BackupManager(config)

    if args.create:
        print(f"💾 创建备份: {args.create}")
        result = backup_manager.create_backup(args.create)

        if result.success:
            print(f"✅ 备份创建成功: {result.backup_path}")
            return 0
        else:
            print(f"❌ 备份创建失败: {result.error}")
            return 1

    elif args.rollback:
        print(f"🔄 回滚文件: {args.rollback}")
        result = backup_manager.rollback_backup(args.rollback)

        if result.success:
            print("✅ 回滚成功")
            return 0
        else:
            print(f"❌ 回滚失败: {result.error}")
            return 1

    elif args.cleanup:
        print("🧹 清理旧备份文件...")
        backup_manager.cleanup_old_backups()
        print("✅ 清理完成")
        return 0

    else:
        print("❌ 请指定备份操作: --create, --rollback 或 --cleanup")
        return 1


if __name__ == '__main__':
    main()

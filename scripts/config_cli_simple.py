#!/usr/bin/env python3
"""
RQA配置管理CLI工具 - 简化版
"""

from src.infrastructure.config import ConfigFactory
import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA配置管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # list命令
    list_parser = subparsers.add_parser('list', help='列出所有配置')

    # get命令
    get_parser = subparsers.add_parser('get', help='获取配置值')
    get_parser.add_argument('key', help='配置键')

    # set命令
    set_parser = subparsers.add_parser('set', help='设置配置值')
    set_parser.add_argument('key', help='配置键')
    set_parser.add_argument('value', help='配置值')

    # validate命令
    validate_parser = subparsers.add_parser('validate', help='验证配置')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 创建配置管理器
    config_manager = ConfigFactory.create_config_manager()

    try:
        if args.command == 'list':
            configs = config_manager.to_dict()
            print(json.dumps(configs, ensure_ascii=False, indent=2))

        elif args.command == 'get':
            value = config_manager.get(args.key)
            if value is not None:
                print(value)
            else:
                print(f"配置键 '{args.key}' 不存在")
                sys.exit(1)

        elif args.command == 'set':
            result = config_manager.set(args.key, args.value)
            if result.success:
                print(f"配置 '{args.key}' 已设置为: {args.value}")
            else:
                print(f"设置配置失败: {result.error}")
                sys.exit(1)

        elif args.command == 'validate':
            configs = config_manager.to_dict()
            is_valid, errors = config_manager.validate(configs)
            if is_valid:
                print("✅ 配置验证通过")
            else:
                print("❌ 配置验证失败:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

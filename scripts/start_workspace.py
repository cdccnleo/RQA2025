#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略工作空间启动脚本
Strategy Workspace Startup Script

启动RQA2025策略工作空间的Web服务。
"""

from src.strategy.workspace.web_server import run_workspace_server
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO"):
    """
    设置日志配置

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/workspace.log', encoding='utf-8')
        ]
    )

    # 设置第三方库的日志级别
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='RQA2025 策略工作空间启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start_workspace.py                    # 使用默认配置启动
  python start_workspace.py --host 127.0.0.1   # 指定主机地址
  python start_workspace.py --port 3000        # 指定端口
  python start_workspace.py --debug           # 启用调试模式
        """
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器监听主机地址 (默认: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务器监听端口 (默认: 8000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='./config/workspace_config.yaml',
        help='配置文件路径 (默认: ./config/workspace_config.yaml)'
    )

    return parser.parse_args()


def create_directories():
    """
    创建必要的目录
    """
    directories = [
        'logs',
        'data/backtest',
        'data/optimization',
        'data/monitoring',
        'config',
        'static'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """
    检查依赖项
    """
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pandas',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少必要的依赖包，请安装以下包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n安装命令:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def print_startup_info(host: str, port: int):
    """
    打印启动信息

    Args:
        host: 主机地址
        port: 端口号
    """
    print("\n" + "="*60)
    print("🚀 RQA2025 策略工作空间启动中...")
    print("="*60)
    print(f"📍 服务地址: http://{host}:{port}")
    print(f"📖 API文档: http://{host}:{port}/docs")
    print(f"🔄 API界面: http://{host}:{port}/redoc")
    print(f"💚 健康检查: http://{host}:{port}/health")
    print("="*60)
    print("💡 使用说明:")
    print("  - 在浏览器中打开上述地址访问Web界面")
    print("  - 使用 Ctrl+C 停止服务器")
    print("  - 查看 logs/workspace.log 获取详细日志")
    print("="*60 + "\n")


def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 创建必要的目录
        create_directories()

        # 设置日志
        setup_logging(args.log_level)

        # 检查依赖
        if not check_dependencies():
            sys.exit(1)

        # 打印启动信息
        print_startup_info(args.host, args.port)

        # 启动服务器
        logger = logging.getLogger(__name__)
        logger.info(f"启动策略工作空间服务器: {args.host}:{args.port}")

        try:
            run_workspace_server(host=args.host, port=args.port)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务器...")
        except Exception as e:
            logger.error(f"服务器运行异常: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

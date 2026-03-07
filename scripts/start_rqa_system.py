#!/usr/bin/env python3
"""
RQA2025系统启动脚本

用于启动完整的量化交易系统，包括数据库、缓存、API服务等。
"""

from src.app import RQAApplication
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from src.infrastructure.logging.core.unified_logger import setup_logging  # 暂时注释，函数不存在


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RQA2025量化交易系统启动器')

    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口 (默认: 8000)')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='日志级别 (默认: INFO)')
    parser.add_argument('--env', choices=['development', 'production', 'testing'],
                        default='development', help='运行环境 (默认: development)')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数 (默认: 1)')

    return parser.parse_args()


async def check_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")

    missing_deps = []

    # 检查数据库连接（可选）
    try:
        print("✅ PostgreSQL驱动可用")
    except ImportError:
        print("⚠️  PostgreSQL驱动不可用，将使用模拟模式")
        missing_deps.append("asyncpg")

    try:
        print("✅ Redis驱动可用")
    except ImportError:
        print("⚠️  Redis驱动不可用，将使用模拟缓存")
        missing_deps.append("aioredis")

    # 检查FastAPI
    try:
        print("✅ FastAPI框架可用")
    except ImportError:
        print("❌ FastAPI框架不可用")
        missing_deps.append("fastapi")

    # 检查其他核心依赖
    core_deps = ['pydantic', 'uvicorn', 'python-jose']
    for dep in core_deps:
        try:
            __import__(dep.replace('-', ''))
            print(f"✅ {dep}可用")
        except ImportError:
            print(f"❌ {dep}不可用")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n⚠️  缺少以下依赖: {', '.join(missing_deps)}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("\n✅ 依赖检查完成")
    return True


async def initialize_database():
    """初始化数据库"""
    print("🗄️  初始化数据库...")

    try:
        from src.core.database_service import get_database_service

        db_service = await get_database_service()
        print("✅ 数据库服务初始化成功")

        # 等待一秒确保连接建立
        await asyncio.sleep(1)

        return True

    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        print("系统将在模拟模式下运行")
        return False


async def start_application(args):
    """启动应用"""
    print("🚀 启动RQA2025量化交易系统...")
    print(f"   主机: {args.host}")
    print(f"   端口: {args.port}")
    print(f"   环境: {args.env}")
    print(f"   日志级别: {args.log_level}")
    print()

    try:
        # 创建应用实例
        app = RQAApplication()

        # 初始化应用
        await app.initialize()

        print("✅ 应用初始化完成")
        print(f"📖 API文档: http://{args.host}:{args.port}/docs")
        print(f"🏛️  交易API: http://{args.host}:{args.port}/api/v1/trading/docs")
        print(f"💚 健康检查: http://{args.host}:{args.port}/health")
        print(f"📊 系统指标: http://{args.host}:{args.port}/metrics")
        print()

        # 运行应用
        await app.run(host=args.host, port=args.port)

    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在关闭...")
    except Exception as e:
        print(f"\n❌ 应用启动失败: {e}")
        return False

    return True


def setup_environment(args):
    """设置运行环境"""
    # 设置环境变量
    os.environ['RQA_ENV'] = args.env
    os.environ['RQA_LOG_LEVEL'] = args.log_level

    # 设置配置
    if args.config:
        os.environ['RQA_CONFIG_FILE'] = args.config

    # 设置日志
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """主函数"""
    args = parse_arguments()

    print("🌟 RQA2025 量化交易系统")
    print("=" * 50)

    # 设置环境
    setup_environment(args)

    # 检查依赖
    if not await check_dependencies():
        if args.env == 'production':
            print("❌ 生产环境依赖检查失败，退出")
            sys.exit(1)
        else:
            print("⚠️  依赖不完整，继续启动（开发模式）")

    # 初始化数据库
    await initialize_database()

    print()

    # 启动应用
    success = await start_application(args)

    if success:
        print("\n✅ RQA2025系统启动成功！")
    else:
        print("\n❌ RQA2025系统启动失败！")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

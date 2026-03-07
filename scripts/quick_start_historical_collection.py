#!/usr/bin/env python3
"""
快速启动历史数据采集脚本

检查环境依赖并启动历史数据采集服务
适用于开发和测试环境
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")

    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查必要依赖
    required_modules = [
        'asyncpg',  # TimescaleDB客户端
        'redis',    # Redis客户端
        'pyyaml',   # YAML配置文件
        'pandas',   # 数据处理
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module} 已安装")
        except ImportError:
            missing_modules.append(module)
            logger.warning(f"✗ {module} 未安装")

    if missing_modules:
        logger.error(f"缺少必要依赖: {', '.join(missing_modules)}")
        logger.info("请运行: pip install asyncpg redis pyyaml pandas")
        return False

    # 检查可选依赖
    optional_modules = [
        'akshare',  # AKShare数据源
        'yfinance', # Yahoo Finance数据源
    ]

    for module in optional_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module} 已安装")
        except ImportError:
            logger.warning(f"⚠ {module} 未安装（可选）")

    # 检查配置文件
    config_files = [
        'config/historical_data_sources.yml',
        '.env.production'
    ]

    for config_file in config_files:
        if (project_root / config_file).exists():
            logger.info(f"✓ 配置文件 {config_file} 存在")
        else:
            logger.warning(f"⚠ 配置文件 {config_file} 不存在")

    # 检查数据库连接（可选）
    try:
        import asyncpg
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'rqa2025_prod'),
            'user': os.getenv('DB_USER', 'rqa2025_admin'),
            'password': os.getenv('DB_PASSWORD', 'SecurePass123!')
        }

        conn = await asyncpg.connect(**db_config)
        await conn.close()
        logger.info("✓ 数据库连接正常")
    except Exception as e:
        logger.warning(f"⚠ 数据库连接失败: {e}（这在没有启动数据库时是正常的）")

    logger.info("环境检查完成")
    return True


async def start_historical_collection():
    """启动历史数据采集"""
    try:
        # 检查环境
        if not await check_environment():
            logger.error("环境检查失败，请解决依赖问题后重试")
            return False

        # 设置环境变量
        os.environ.setdefault('HISTORICAL_START_YEAR', '2014')
        os.environ.setdefault('HISTORICAL_END_YEAR', '2024')
        os.environ.setdefault('MAX_CONCURRENT_BATCHES', '2')  # 降低并发度以适应测试环境
        os.environ.setdefault('QUALITY_THRESHOLD', '0.85')
        os.environ.setdefault('SYMBOL_BATCH_SIZE', '5')  # 减小批次大小

        logger.info("启动历史数据采集服务...")
        logger.info("采集配置:")
        logger.info(f"  - 时间范围: {os.environ['HISTORICAL_START_YEAR']}-{os.environ['HISTORICAL_END_YEAR']}年")
        logger.info(f"  - 并发批次: {os.environ['MAX_CONCURRENT_BATCHES']}")
        logger.info(f"  - 质量阈值: {os.environ['QUALITY_THRESHOLD']}")

        # 导入并运行历史数据采集
        from scripts.start_historical_data_collection import main as historical_main
        exit_code = await historical_main()

        if exit_code == 0:
            logger.info("历史数据采集完成")
            return True
        else:
            logger.error("历史数据采集失败")
            return False

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在优雅关闭...")
        return False
    except Exception as e:
        logger.error(f"启动历史数据采集失败: {e}", exc_info=True)
        return False


def print_usage():
    """打印使用说明"""
    print("=" * 60)
    print("RQA2025 历史数据采集快速启动脚本")
    print("=" * 60)
    print()
    print("功能说明:")
    print("  - 检查运行环境和依赖")
    print("  - 启动核心股票池的历史数据采集")
    print("  - 支持2014-2024年的10年历史数据")
    print()
    print("使用方法:")
    print("  python scripts/quick_start_historical_collection.py")
    print()
    print("环境变量配置:")
    print("  HISTORICAL_START_YEAR: 开始年份（默认2014）")
    print("  HISTORICAL_END_YEAR: 结束年份（默认2024）")
    print("  MAX_CONCURRENT_BATCHES: 最大并发批次数（默认2）")
    print("  QUALITY_THRESHOLD: 数据质量阈值（默认0.85）")
    print()
    print("依赖要求:")
    print("  - Python 3.8+")
    print("  - asyncpg: TimescaleDB客户端")
    print("  - redis: Redis客户端")
    print("  - pyyaml: YAML配置解析")
    print("  - pandas: 数据处理")
    print("  - akshare: AKShare数据源（可选）")
    print("  - yfinance: Yahoo Finance数据源（可选）")
    print()
    print("数据源说明:")
    print("  1. AKShare: 主要数据源，支持A股、港股等")
    print("  2. Yahoo Finance: 备选数据源，支持国际市场")
    print("  3. Local Backup: 本地备份数据源")
    print()
    print("=" * 60)


async def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        return 0

    logger.info("RQA2025 历史数据采集快速启动脚本")
    logger.info("=" * 50)

    success = await start_historical_collection()

    if success:
        logger.info("历史数据采集服务执行完成")
        return 0
    else:
        logger.error("历史数据采集服务执行失败")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
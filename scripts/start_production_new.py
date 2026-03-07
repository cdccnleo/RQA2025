#!/usr/bin/env python3
"""
RQA2025 量化交易系统生产环境启动脚本

基于21层架构设计的分层启动系统：
- 核心业务层：策略层、交易层、风险控制层、特征层
- 核心支撑层：数据管理层、机器学习层、基础设施层、流处理层
- 辅助支撑层：核心服务层、监控层、优化层、网关层、适配器层、自动化层、弹性层、测试层、工具层

职责：
1. 系统环境初始化
2. 架构启动器调用
3. 基础服务启动
4. 优雅关闭处理
"""

import os
import sys
import time
import logging
import uvicorn
import signal
from typing import Optional

# 设置Python路径
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/scripts')

# 可选依赖检查
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/production.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_rate_limit(rate_limit_str):
    """解析频率限制字符串，返回间隔秒数"""
    import re

    if not rate_limit_str or not isinstance(rate_limit_str, str):
        return 60  # 默认60秒

    match = re.match(r'(\d+)\s*次?\s*/?\s*(分钟|小时|时|秒|分)?', rate_limit_str.strip())
    if not match:
        try:
            return max(1, int(float(rate_limit_str)))
        except (ValueError, TypeError):
            return 60

    count = int(match.group(1))
    unit = match.group(2) or '分'

    if unit in ['秒', 's']:
        interval = 1.0 / count if count > 0 else 1
    elif unit in ['分', '分钟', 'm']:
        interval = 60.0 / count if count > 0 else 60
    elif unit in ['时', '小时', 'h']:
        interval = 3600.0 / count if count > 0 else 3600
    else:
        interval = 60.0 / count if count > 0 else 60

    return max(0.1, interval)

def check_environment():
    """检查运行环境"""
    logger.info("检查生产环境配置...")

    required_env_vars = [
        'RQA_ENV',
        'DATABASE_URL',
        'REDIS_URL'
    ]

    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"环境变量 {var} 未设置")

    # 检查数据库连接
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL', '')
        if db_url:
            logger.info("数据库配置检查完成")
        else:
            logger.warning("DATABASE_URL 未配置")
    except ImportError:
        logger.warning("psycopg2 未安装，跳过数据库检查")

    # 检查Redis连接
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', '')
        if redis_url:
            logger.info("Redis配置检查完成")
        else:
            logger.warning("REDIS_URL 未配置")
    except ImportError:
        logger.warning("redis 未安装，跳过Redis检查")

    logger.info("环境检查完成")

def initialize_services():
    """初始化各项服务"""
    logger.info("初始化RQA2025服务...")

    # 初始化数据源配置
    try:
        logger.info("初始化数据源配置...")
        # 设置正确的Python路径
        sys.path.insert(0, '/app')
        sys.path.insert(0, '/app/src')
        sys.path.insert(0, '/app/scripts')
        from src.gateway.web.api import initialize_data_sources_if_needed
        initialize_data_sources_if_needed()
        logger.info("数据源配置初始化完成")
    except Exception as e:
        logger.error(f"数据源配置初始化失败: {e}")
        # 不阻止服务启动，但记录错误

    logger.info("服务初始化完成")

def start_web_server():
    """启动Web服务器 - 使用架构启动器"""
    logger.info("启动RQA2025 Web服务...")

    try:
        # 使用架构启动器启动系统
        from src.core.architecture_startup import start_architecture_system

        logger.info("正在启动架构系统...")
        if not start_architecture_system():
            logger.error("架构系统启动失败")
            return

        # 获取网关层应用
        from src.gateway import create_gateway_app
        app = create_gateway_app()

        # 配置服务器参数
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', '8000'))

        logger.info(f"启动Web服务器在 {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务...")
        # 优雅关闭架构系统
        from src.core.architecture_startup import stop_architecture_system
        stop_architecture_system()
    except Exception as e:
        logger.error(f"启动过程中发生错误: {e}")
        sys.exit(1)
    finally:
        logger.info("RQA2025服务已停止")

def main():
    """主启动函数"""
    print("=" * 60)
    print("🚀 RQA2025 量化交易系统启动")
    print("基于21层架构设计的分层启动系统")
    print("=" * 60)

    try:
        # 1. 检查环境
        check_environment()

        # 2. 初始化服务
        initialize_services()

        # 3. 启动Web服务器
        start_web_server()

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"启动过程中发生错误: {e}")
        sys.exit(1)
    finally:
        logger.info("RQA2025服务已停止")

if __name__ == "__main__":
    main()

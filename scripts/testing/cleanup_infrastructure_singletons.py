#!/usr/bin/env python3
"""
基础设施层单例清理脚本 - 清理全局注册表、缓存等资源
"""

from src.infrastructure.logging.infrastructure_logger import InfrastructureLogger
from src.infrastructure.config.strategies.unified_strategy import cleanup_strategy_registry
import gc
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


def cleanup_infrastructure_singletons():
    """清理基础设施层的所有单例和全局资源"""
    logger.info("开始清理基础设施层单例资源...")

    # 1. 清理策略注册表
    try:
        cleanup_strategy_registry()
        logger.info("✅ 策略注册表已清理")
    except Exception as e:
        logger.error(f"❌ 清理策略注册表失败: {e}")

    # 2. 清理日志系统单例
    try:
        # 清理InfrastructureLogger的单例缓存
        if hasattr(InfrastructureLogger, '_instances'):
            InfrastructureLogger._instances.clear()
        logger.info("✅ 日志系统单例已清理")
    except Exception as e:
        logger.error(f"❌ 清理日志系统单例失败: {e}")

    # 3. 清理监控装饰器缓存
    try:
        # 导入并清理监控装饰器的缓存
        from src.infrastructure.monitoring.decorators import _metric_cache
        if hasattr(_metric_cache, 'clear'):
            _metric_cache.clear()
        logger.info("✅ 监控装饰器缓存已清理")
    except Exception as e:
        logger.error(f"❌ 清理监控装饰器缓存失败: {e}")

    # 4. 强制垃圾回收
    try:
        collected = gc.collect()
        logger.info(f"✅ 垃圾回收完成，清理了 {collected} 个对象")
    except Exception as e:
        logger.error(f"❌ 垃圾回收失败: {e}")

    logger.info("基础设施层单例资源清理完成")


def main():
    """主函数"""
    cleanup_infrastructure_singletons()


if __name__ == '__main__':
    main()

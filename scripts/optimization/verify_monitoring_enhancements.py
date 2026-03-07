#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据持久化优化验证脚本

验证增强的监控数据持久化系统是否正常工作。
"""

import sys
import logging
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def verify_enhanced_persistence():
    """验证增强的持久化功能"""
    logger.info("验证增强的持久化功能...")

    try:
        from src.features.monitoring.metrics_persistence import get_enhanced_persistence_manager

        # 创建测试管理器
        config = {
            'path': './test_monitoring_verification',
            'batch_size': 10,
            'batch_timeout': 1.0
        }

        manager = get_enhanced_persistence_manager(config)

        # 测试存储
        logger.info("测试数据存储...")
        success = manager.store_metric_sync(
            component_name='verification_test',
            metric_name='test_metric',
            metric_value=123.45,
            metric_type='TEST',
            labels={'test': 'verification'}
        )

        if success:
            logger.info("✓ 数据存储测试通过")
        else:
            logger.error("✗ 数据存储测试失败")
            return False

        # 等待批量写入
        time.sleep(2)

        # 测试查询
        logger.info("测试数据查询...")
        import asyncio
        result = asyncio.run(manager.query_metrics_async(component_name='verification_test'))

        if len(result) > 0:
            logger.info("✓ 数据查询测试通过")
        else:
            logger.error("✗ 数据查询测试失败")
            return False

        # 清理
        manager.stop()

        logger.info("✓ 增强持久化功能验证通过")
        return True

    except Exception as e:
        logger.error(f"✗ 增强持久化功能验证失败: {e}")
        return False


def verify_monitoring_service_integration():
    """验证监控服务集成"""
    logger.info("验证监控服务集成...")

    try:
        from src.strategy.monitoring.monitoring_service import MonitoringService

        # 创建监控服务实例
        service = MonitoringService()

        # 检查是否有增强持久化支持
        has_enhanced = hasattr(service, 'enhanced_persistence')

        if has_enhanced:
            logger.info("✓ 监控服务增强持久化集成成功")
            return True
        else:
            logger.warning("! 监控服务未检测到增强持久化支持")
            return False

    except Exception as e:
        logger.error(f"✗ 监控服务集成验证失败: {e}")
        return False


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)

    logger.info("开始监控数据持久化优化验证")
    logger.info("=" * 50)

    all_passed = True

    # 验证增强持久化功能
    if not verify_enhanced_persistence():
        all_passed = False

    # 验证监控服务集成
    if not verify_monitoring_service_integration():
        all_passed = False

    logger.info("=" * 50)
    if all_passed:
        logger.info("✓ 所有验证测试通过")
    else:
        logger.error("✗ 部分验证测试失败")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

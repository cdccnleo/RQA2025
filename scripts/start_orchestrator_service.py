#!/usr/bin/env python3
"""
数据采集编排器服务启动脚本
用于在Docker容器中启动数据采集编排器
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_rate_limit(rate_limit_str: str) -> float:
    """
    解析频率限制字符串，返回采集间隔秒数（使用统一函数）
    
    注意：此函数现在使用data_collectors.py中的统一实现，确保所有调度器使用相同的解析逻辑
    """
    try:
        # 优先使用统一的parse_rate_limit函数（符合架构设计：统一实现）
        from src.gateway.web.data_collectors import parse_rate_limit as unified_parse_rate_limit
        return unified_parse_rate_limit(rate_limit_str)
    except ImportError:
        # 降级方案：如果无法导入，使用本地实现
        logger.warning("无法导入统一的parse_rate_limit函数，使用本地实现")
        try:
            # 默认值
            if not rate_limit_str:
                return 60.0  # 默认60秒

            parts = rate_limit_str.split('/')
            if len(parts) != 2:
                return 60.0

            count_str, period_str = parts

            # 提取数字
            count = int(''.join(filter(str.isdigit, count_str)))
            if count <= 0:
                count = 1

            # 解析时间单位
            if '分钟' in period_str:
                return 60.0 / count if count > 0 else 60.0
            elif '小时' in period_str:
                return 3600.0 / count if count > 0 else 3600.0
            elif '天' in period_str:
                return 86400.0 / count if count > 0 else 86400.0
            else:
                return 60.0

        except Exception as e:
            logger.warning(f"解析频率限制失败: {rate_limit_str}, 使用默认值: {e}")
            return 60.0


async def run_orchestrator():
    """运行编排器服务"""
    try:
        # 导入必要的模块
        from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
        from src.gateway.web.data_source_config_manager import DataSourceConfigManager

        logger.info('启动数据采集编排器服务...')

        # 初始化编排器
        orchestrator = DataCollectionWorkflow()
        data_source_manager = DataSourceConfigManager()

        logger.info('数据采集编排器服务已启动，开始调度数据采集任务...')

        # 开始数据采集调度循环
        last_collection_times = {}
        check_interval = 30  # 30秒检查一次

        while True:
            try:
                # 获取数据源配置
                sources = data_source_manager.get_data_sources()
                current_time = datetime.now().timestamp()

                for source in sources:
                    if not source.get('enabled', False):
                        continue

                    source_id = source['id']
                    rate_limit = source.get('rate_limit', '60次/分钟')

                    # 解析频率限制
                    interval_seconds = parse_rate_limit(rate_limit)

                    # 检查是否到采集时间
                    last_time = last_collection_times.get(source_id, 0)
                    if current_time - last_time >= interval_seconds:
                        # 启动采集流程
                        try:
                            success = await orchestrator.start_collection_process(source_id, source)
                            if success:
                                last_collection_times[source_id] = current_time
                                logger.info(f'数据源 {source_id} 采集任务启动成功')
                            else:
                                logger.warning(f'数据源 {source_id} 采集任务启动失败')
                        except Exception as e:
                            logger.error(f'数据源 {source_id} 采集任务执行异常: {e}')

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f'数据采集调度循环异常: {e}')
                await asyncio.sleep(5)

    except Exception as e:
        logger.error(f'编排器服务启动失败: {e}')
        sys.exit(1)


if __name__ == '__main__':
    # 添加src路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    # 设置环境变量
    os.environ.setdefault('RQA_ENV', 'production')
    os.environ.setdefault('RQA_SERVICE', 'data-collection-orchestrator')

    asyncio.run(run_orchestrator())

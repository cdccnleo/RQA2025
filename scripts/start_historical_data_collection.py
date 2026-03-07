#!/usr/bin/env python3
"""
历史数据采集服务启动脚本

启动历史数据采集服务，采集策略回测所需的10年历史数据：
1. 初始化多数据源适配器（AKShare, Yahoo, Local Backup）
2. 启动核心股票池的历史数据采集
3. 提供实时监控和进度报告
4. 支持并发采集和错误恢复

使用方法：
    python scripts/start_historical_data_collection.py

环境变量：
    HISTORICAL_START_YEAR: 开始年份（默认2014）
    HISTORICAL_END_YEAR: 结束年份（默认2024）
    MAX_CONCURRENT_BATCHES: 最大并发批次数（默认3）
    QUALITY_THRESHOLD: 数据质量阈值（默认0.85）
    FORCE_RESTART: 是否强制重新采集（默认false）
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.historical_data_acquisition_service import HistoricalDataAcquisitionService
from src.core.persistence.timescale_storage import TimescaleStorage
from src.core.cache.redis_cache import RedisCache
from src.infrastructure.logging.core.unified_logger import get_unified_logger

# 配置日志
logger = get_unified_logger(__name__)

# 核心股票池（来自data_collectors.py中的get_core_universe_symbols函数）
CORE_STOCK_SYMBOLS = [
    '600000', '600036', '600519', '600276', '600887', '600000', '600016', '600028', '600030',
    '600031', '600036', '600048', '600050', '600104', '600196', '600276', '600309', '600340',
    '600346', '600352', '600362', '600383', '600390', '600398', '600406', '600436', '600438',
    '600519', '600547', '600570', '600583', '600585', '600588', '600606', '600637', '600690',
    '600703', '600732', '600745', '600754', '600795', '600803', '600809', '600837', '600887',
    '600893', '600900', '600909', '600919', '600926', '600928', '600958', '600989', '600999',
    '601006', '601088', '601166', '601211', '601288', '601318', '601319', '601328', '601336',
    '601360', '601377', '601390', '601398', '601600', '601601', '601628', '601633', '601668',
    '601669', '601688', '601698', '601727', '601766', '601800', '601818', '601857', '601866',
    '601872', '601877', '601878', '601881', '601888', '601898', '601899', '601901', '601916',
    '601918', '601919', '601933', '601939', '601949', '601952', '601958', '601965', '601966',
    '601969', '601975', '601985', '601988', '601989', '601992', '601995', '601998', '603019',
    '603156', '603160', '603259', '603260', '603288', '603369', '603501', '603658', '603799',
    '603806', '603833', '603899', '603986', '603993', '000001', '000002', '000063', '000069',
    '000100', '000157', '000166', '000301', '000338', '000402', '000408', '000425', '000538',
    '000568', '000596', '000617', '000625', '000627', '000629', '000630', '000651', '000661',
    '000671', '000703', '000708', '000723', '000725', '000728', '000738', '000750', '000768',
    '000776', '000783', '000786', '000800', '000807', '000829', '000830', '000831', '000858',
    '000876', '000883', '000895', '000898', '000938', '000961', '000963', '000977', '001979',
    '002007', '002008', '002024', '002027', '002032', '002044', '002049', '002050', '002064',
    '002081', '002085', '002120', '002142', '002146', '002152', '002157', '002179', '002202',
    '002230', '002236', '002241', '002252', '002271', '002294', '002304', '002310', '002352',
    '002371', '002410', '002414', '002415', '002422', '002424', '002426', '002450', '002456',
    '002460', '002463', '002466', '002468', '002475', '002493', '002508', '002555', '002558',
    '002572', '002594', '002600', '002601', '002602', '002607', '002624', '002625', '002648',
    '002709', '002714', '002736', '002739', '002773', '002812', '002821', '002841', '002916',
    '002920', '002938', '002939', '002945', '002958', '003816', '003833', '300003', '300014',
    '300015', '300017', '300024', '300027', '300033', '300058', '300070', '300072', '300122',
    '300124', '300136', '300142', '300144', '300207', '300223', '300274', '300308', '300316',
    '300347', '300408', '300413', '300415', '300433', '300450', '300454', '300496', '300498',
    '300529', '300601', '300628', '300661', '300750', '300751', '300759', '300760', '300782',
    '300832', '300896', '300919', '300957', '300979', '300999', '301269', '301279', '301317',
    '301319', '301338', '301358', '301369', '301380', '301391', '301421', '301488', '301489'
]


class HistoricalDataCollectionManager:
    """历史数据采集管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service: Optional[HistoricalDataAcquisitionService] = None
        self.start_time = None
        self.end_time = None

        # 采集统计
        self.stats = {
            'total_symbols': 0,
            'completed_symbols': 0,
            'failed_symbols': 0,
            'total_batches': 0,
            'completed_batches': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }

    async def initialize(self):
        """初始化服务"""
        logger.info("初始化历史数据采集管理器...")

        # 配置数据源适配器
        adapter_configs = {
            'akshare': {
                'enabled': True,
                'timeout': 30,
                'retry_count': 3
            },
            'yahoo': {
                'enabled': True,
                'timeout': 30,
                'retry_count': 3
            },
            'local_backup': {
                'enabled': True,
                'backup_dir': './data/backup',
                'timeout': 10,
                'retry_count': 1
            }
        }

        # 初始化服务配置
        service_config = {
            'max_concurrent_batches': self.config.get('max_concurrent_batches', 3),
            'quality_threshold': self.config.get('quality_threshold', 0.85),
            'timescale_config': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'rqa2025_prod'),
                'user': os.getenv('DB_USER', 'rqa2025_admin'),
                'password': os.getenv('DB_PASSWORD', 'SecurePass123!')
            },
            'redis_config': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', '6379')),
                'password': os.getenv('REDIS_PASSWORD'),
                'db': 0
            },
            'adapters': adapter_configs
        }

        # 初始化历史数据采集服务
        self.service = HistoricalDataAcquisitionService(service_config)

        logger.info("历史数据采集管理器初始化完成")

    async def start_collection(self, symbols: List[str], start_year: int, end_year: int):
        """启动数据采集"""
        self.start_time = datetime.now()
        self.stats['start_time'] = self.start_time
        self.stats['total_symbols'] = len(symbols)

        logger.info(f"开始历史数据采集: {len(symbols)}个标的, {start_year}-{end_year}年")
        logger.info(f"核心配置: 并发批次={self.config.get('max_concurrent_batches', 3)}, 质量阈值={self.config.get('quality_threshold', 0.85)}")

        # 创建采集任务
        tasks = []
        for symbol in symbols:
            task = self._collect_symbol_data(symbol, start_year, end_year)
            tasks.append(task)

        # 分批执行以避免过载
        batch_size = self.config.get('symbol_batch_size', 10)
        results = []

        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            logger.info(f"执行第{i//batch_size + 1}批采集任务 ({len(batch_tasks)}个标的)")

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

            # 短暂暂停避免过载
            await asyncio.sleep(1)

        # 处理结果
        await self._process_results(results)

        self.end_time = datetime.now()
        self.stats['end_time'] = self.end_time

        # 输出最终统计
        self._print_final_stats()

    async def _collect_symbol_data(self, symbol: str, start_year: int, end_year: int):
        """采集单个标的的历史数据"""
        try:
            logger.info(f"开始采集标的 {symbol} 的历史数据...")

            # 采集多年数据
            batches = await self.service.acquire_multi_year_data(
                symbol=symbol,
                start_year=start_year,
                end_year=end_year,
                data_types=['stock']
            )

            if not batches:
                logger.warning(f"标的 {symbol} 未能采集到任何数据")
                return {'symbol': symbol, 'success': False, 'batches': [], 'error': 'no data collected'}

            # 存储批次结果
            storage_result = await self.service.store_batch_results(batches)

            # 验证数据完整性
            validation_result = await self.service.validate_data_integrity(symbol, start_year, end_year)

            result = {
                'symbol': symbol,
                'success': True,
                'batches': batches,
                'storage_result': storage_result,
                'validation_result': validation_result,
                'total_records': storage_result.get('total_records', 0),
                'quality_score': validation_result.get('avg_quality', 0)
            }

            logger.info(f"标的 {symbol} 采集完成: {len(batches)}个批次, {storage_result.get('total_records', 0)}条记录")
            return result

        except Exception as e:
            error_msg = f"采集标的 {symbol} 失败: {e}"
            logger.error(error_msg)
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def _process_results(self, results: List[Any]):
        """处理采集结果"""
        successful_symbols = []
        failed_symbols = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"采集任务异常: {result}")
                self.stats['errors'].append(str(result))
                continue

            if result.get('success', False):
                successful_symbols.append(result)
                self.stats['completed_symbols'] += 1
                self.stats['total_batches'] += len(result.get('batches', []))
                self.stats['total_records'] += result.get('total_records', 0)
            else:
                failed_symbols.append(result)
                self.stats['failed_symbols'] += 1
                self.stats['errors'].append(result.get('error', 'unknown error'))

        logger.info(f"采集结果处理完成: 成功{len(successful_symbols)}个, 失败{len(failed_symbols)}个")

        # 详细报告失败的标的
        if failed_symbols:
            logger.warning("失败的标的列表:")
            for failed in failed_symbols[:10]:  # 只显示前10个
                logger.warning(f"  - {failed['symbol']}: {failed.get('error', 'unknown')}")

    def _print_final_stats(self):
        """打印最终统计信息"""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else None

        logger.info("=" * 60)
        logger.info("历史数据采集任务完成统计")
        logger.info("=" * 60)
        logger.info(f"采集时间: {self.stats['start_time']} - {self.stats['end_time']}")
        if duration:
            logger.info(f"总耗时: {duration}")
        logger.info(f"标的总数: {self.stats['total_symbols']}")
        logger.info(f"成功采集: {self.stats['completed_symbols']}")
        logger.info(f"采集失败: {self.stats['failed_symbols']}")
        logger.info(f"成功率: {self.stats['completed_symbols']/self.stats['total_symbols']*100:.1f}%" if self.stats['total_symbols'] > 0 else "0%")
        logger.info(f"总批次数: {self.stats['total_batches']}")
        logger.info(f"总记录数: {self.stats['total_records']}")
        if self.stats['completed_symbols'] > 0:
            logger.info(f"平均每标记录数: {self.stats['total_records']/self.stats['completed_symbols']:.0f}")

        if self.stats['errors']:
            logger.info(f"错误数量: {len(self.stats['errors'])}")
            logger.info("主要错误类型:")
            error_counts = {}
            for error in self.stats['errors'][:20]:  # 只统计前20个错误
                error_type = str(error).split(':')[0] if ':' in str(error) else str(error)[:50]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  - {error_type}: {count}次")

        logger.info("=" * 60)

    async def get_collection_status(self) -> Dict[str, Any]:
        """获取采集状态"""
        if not self.service:
            return {'status': 'not_initialized'}

        # 获取数据质量统计
        quality_stats = {}
        if hasattr(self.service, 'get_data_quality_stats'):
            # 这里需要实现获取所有已采集数据的质量统计
            pass

        return {
            'status': 'running' if not self.end_time else 'completed',
            'stats': self.stats,
            'current_time': datetime.now(),
            'quality_stats': quality_stats
        }


async def main():
    """主函数"""
    try:
        logger.info("启动历史数据采集服务...")

        # 获取配置
        config = {
            'start_year': int(os.getenv('HISTORICAL_START_YEAR', '2014')),
            'end_year': int(os.getenv('HISTORICAL_END_YEAR', '2024')),
            'max_concurrent_batches': int(os.getenv('MAX_CONCURRENT_BATCHES', '3')),
            'quality_threshold': float(os.getenv('QUALITY_THRESHOLD', '0.85')),
            'symbol_batch_size': int(os.getenv('SYMBOL_BATCH_SIZE', '10')),
            'force_restart': os.getenv('FORCE_RESTART', 'false').lower() == 'true'
        }

        logger.info(f"采集配置: {config['start_year']}-{config['end_year']}年, 并发{config['max_concurrent_batches']}批次, 质量阈值{config['quality_threshold']}")

        # 初始化管理器
        manager = HistoricalDataCollectionManager(config)
        await manager.initialize()

        # 确定要采集的标的
        symbols_to_collect = CORE_STOCK_SYMBOLS

        # 如果不是强制重启，检查已有的数据
        if not config['force_restart']:
            # 这里可以添加检查已采集数据的逻辑
            logger.info("检查现有数据状态...")
            # TODO: 实现数据存在性检查

        logger.info(f"准备采集 {len(symbols_to_collect)} 个核心股票标的的历史数据")

        # 启动采集
        await manager.start_collection(symbols_to_collect, config['start_year'], config['end_year'])

        # 获取最终状态
        final_status = await manager.get_collection_status()
        logger.info("历史数据采集服务完成")

        # 如果有失败的标的，建议重试
        if final_status['stats']['failed_symbols'] > 0:
            logger.warning(f"有 {final_status['stats']['failed_symbols']} 个标的采集失败，建议检查网络连接或数据源配置后重试")

        return 0

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在优雅关闭...")
        return 0
    except Exception as e:
        logger.error(f"历史数据采集服务异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
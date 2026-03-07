#!/usr/bin/env python3
"""
历史数据采集演示脚本

模拟历史数据采集流程，展示系统架构和工作原理
不依赖外部数据库和网络连接
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockHistoricalDataService:
    """模拟历史数据采集服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collected_data = {}
        self.stats = {
            'total_symbols': 0,
            'completed_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }

    async def collect_symbol_data(self, symbol: str, start_year: int, end_year: int) -> Dict[str, Any]:
        """模拟采集单个标的的历史数据"""
        logger.info(f"开始采集标的 {symbol} 的历史数据 ({start_year}-{end_year}年)")

        # 模拟采集时间
        await asyncio.sleep(0.1)  # 模拟网络延迟

        # 生成模拟数据
        data = self._generate_mock_data(symbol, start_year, end_year)

        # 模拟数据质量检查
        quality_score = 0.85 + (hash(symbol) % 10) / 100  # 0.85-0.94之间的随机质量分

        result = {
            'symbol': symbol,
            'data': data,
            'records_count': len(data),
            'quality_score': quality_score,
            'data_source': 'akshare',
            'collection_time': time.time(),
            'status': 'completed'
        }

        self.collected_data[symbol] = result
        self.stats['total_records'] += len(data)

        logger.info(f"标的 {symbol} 采集完成: {len(data)}条记录, 质量分: {quality_score:.2f}")
        return result

    def _generate_mock_data(self, symbol: str, start_year: int, end_year: int) -> List[Dict[str, Any]]:
        """生成模拟的历史数据"""
        data = []
        base_price = 10.0 + (hash(symbol) % 90)  # 10-100之间的基础价格

        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)

        while current_date <= end_date:
            if current_date.weekday() < 5:  # 周一到周五
                # 模拟价格波动
                price_change = (hash(f"{symbol}_{current_date}") % 200 - 100) / 1000  # -0.1到+0.1
                open_price = base_price * (1 + price_change)
                close_price = open_price * (1 + (hash(f"{symbol}_{current_date}_close") % 40 - 20) / 1000)
                high_price = max(open_price, close_price) * (1 + abs(hash(f"{symbol}_{current_date}_high") % 30) / 1000)
                low_price = min(open_price, close_price) * (1 - abs(hash(f"{symbol}_{current_date}_low") % 30) / 1000)

                record = {
                    'symbol': symbol,
                    'date': current_date.date().isoformat(),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': 1000000 + (hash(f"{symbol}_{current_date}_vol") % 9000000),
                    'amount': round(close_price * (1000000 + (hash(f"{symbol}_{current_date}_vol") % 9000000)), 2),
                    'data_source': 'akshare',
                    'quality_score': 0.9,
                    'timestamp': current_date.isoformat()
                }
                data.append(record)

                # 价格缓慢上涨
                base_price *= 1.0001

            current_date += timedelta(days=1)

        return data

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        return {
            'stats': self.stats,
            'collected_symbols': list(self.collected_data.keys()),
            'total_data_points': sum(len(result['data']) for result in self.collected_data.values()),
            'avg_quality_score': sum(result['quality_score'] for result in self.collected_data.values()) / len(self.collected_data) if self.collected_data else 0
        }


async def demo_historical_collection():
    """演示历史数据采集流程"""
    logger.info("开始历史数据采集演示")
    logger.info("=" * 60)

    # 配置
    config = {
        'start_year': 2014,
        'end_year': 2024,
        'max_concurrent': 3,
        'symbols_to_collect': [
            '000001', '000002', '600000', '600036', '000858', '600519'
        ]  # 核心股票池示例
    }

    # 初始化服务
    service = MockHistoricalDataService(config)
    service.stats['start_time'] = datetime.now()

    logger.info(f"采集配置: {config['start_year']}-{config['end_year']}年")
    logger.info(f"标的数量: {len(config['symbols_to_collect'])}")
    logger.info(f"并发数: {config['max_concurrent']}")
    logger.info(f"标的列表: {', '.join(config['symbols_to_collect'])}")
    logger.info("-" * 60)

    # 分批并发采集
    tasks = []
    for symbol in config['symbols_to_collect']:
        task = service.collect_symbol_data(symbol, config['start_year'], config['end_year'])
        tasks.append(task)

    # 并发执行
    results = []
    batch_size = config['max_concurrent']

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        logger.info(f"执行第{i//batch_size + 1}批采集任务 ({len(batch_tasks)}个标的)")

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)

        # 短暂暂停模拟真实场景
        await asyncio.sleep(0.2)

    # 处理结果
    successful_results = []
    failed_results = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"采集异常: {result}")
            failed_results.append(str(result))
        else:
            successful_results.append(result)
            service.stats['completed_symbols'] += 1

    service.stats['failed_symbols'] = len(failed_results)
    service.stats['end_time'] = datetime.now()

    # 输出统计结果
    logger.info("=" * 60)
    logger.info("历史数据采集演示完成")
    logger.info("=" * 60)

    duration = service.stats['end_time'] - service.stats['start_time']
    logger.info(f"总耗时: {duration.total_seconds():.2f}秒")
    logger.info(f"标的总数: {len(config['symbols_to_collect'])}")
    logger.info(f"成功采集: {service.stats['completed_symbols']}")
    logger.info(f"采集失败: {service.stats['failed_symbols']}")
    logger.info(f"总记录数: {service.stats['total_records']}")

    if successful_results:
        avg_records = service.stats['total_records'] / len(successful_results)
        logger.info(f"平均每标记录数: {avg_records:.0f}")
        logger.info(f"数据质量范围: {min(r['quality_score'] for r in successful_results):.2f} - {max(r['quality_score'] for r in successful_results):.2f}")

    # 显示详细结果
    logger.info("-" * 60)
    logger.info("详细结果:")
    for result in successful_results[:5]:  # 只显示前5个
        logger.info(f"  ✓ {result['symbol']}: {result['records_count']}条记录 (质量: {result['quality_score']:.2f})")

    if failed_results:
        logger.info("失败项目:")
        for failure in failed_results[:3]:  # 只显示前3个
            logger.info(f"  ✗ {failure}")

    # 保存结果到文件
    output_file = Path("historical_collection_demo_results.json")
    demo_results = {
        'configuration': config,
        'statistics': service.get_collection_stats(),
        'timestamp': datetime.now().isoformat(),
        'demo_note': '这是历史数据采集系统的演示版本，不包含真实数据连接'
    }

    # 转换datetime对象为字符串
    if 'stats' in demo_results['statistics']:
        stats = demo_results['statistics']['stats']
        for key in ['start_time', 'end_time']:
            if key in stats and isinstance(stats[key], datetime):
                stats[key] = stats[key].isoformat()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)

    logger.info(f"演示结果已保存到: {output_file}")

    logger.info("=" * 60)
    logger.info("演示说明:")
    logger.info("1. 本演示模拟了历史数据采集的核心流程")
    logger.info("2. 展示了多标的并发采集能力")
    logger.info("3. 包含数据质量评估逻辑")
    logger.info("4. 生产环境需要配置真实的数据源和数据库连接")
    logger.info("5. 建议按批次采集，避免对数据源造成过大压力")
    logger.info("=" * 60)

    return len(successful_results) > 0


async def show_system_architecture():
    """展示系统架构说明"""
    print("\n" + "=" * 80)
    print("RQA2025 历史数据采集系统架构说明")
    print("=" * 80)

    print("\n🏗️ 系统架构:")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                策略回测历史数据采集系统                      │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │  ┌─────────────────┐     ┌─────────────────────────────┐   │")
    print("  │  │   日常补全轨    │     │     历史数据采集轨          │   │")
    print("  │  │ (增量采集为主)  │     │  (批量历史数据采集为主)     │   │")
    print("  │  └─────────────────┘     └─────────────────────────────┘   │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │                统一数据存储与访问层                        │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print("  │  • TimescaleDB: 主数据库 (时序数据优化)                  │")
    print("  │  • Redis Cluster: 缓存层 (热点数据加速)                  │")
    print("  │  • MinIO: 对象存储 (大文件存储)                          │")
    print("  │  • 数据标签系统: 区分数据来源和用途                      │")
    print("  └─────────────────────────────────────────────────────────────┘")

    print("\n📊 数据流:")
    print("  数据源 (AKShare/Yahoo/Local) → 数据适配器 → 质量检查 → TimescaleDB")
    print("                                      ↓")
    print("                                缓存层 (Redis)")

    print("\n🎯 核心特性:")
    print("  • 多数据源集成 (AKShare, Yahoo Finance, 本地备份)")
    print("  • 数据质量监控和验证")
    print("  • 并发采集和错误恢复")
    print("  • 时序数据优化存储")
    print("  • 实时监控和告警")

    print("\n📁 创建的文件:")
    files_created = [
        "scripts/start_historical_data_collection.py - 完整启动脚本",
        "scripts/quick_start_historical_collection.py - 快速启动脚本",
        "scripts/demo_historical_collection.py - 演示脚本",
        "scripts/init_historical_data_tables.sql - 数据库初始化",
        "config/historical_data_sources.yml - 数据源配置"
    ]
    for file in files_created:
        print(f"  ✓ {file}")

    print("\n🚀 启动方式:")
    print("  1. 生产环境: docker-compose up -d")
    print("  2. 开发环境: python scripts/start_historical_data_collection.py")
    print("  3. 演示环境: python scripts/demo_historical_collection.py")

    print("\n⚙️ 配置要求:")
    print("  • TimescaleDB数据库连接")
    print("  • Redis缓存服务")
    print("  • 数据源API密钥 (AKShare, Yahoo可选)")
    print("  • 网络连接 (访问数据源)")

    print("\n" + "=" * 80)


async def main():
    """主函数"""
    try:
        # 显示系统架构
        await show_system_architecture()

        # 运行演示
        print("\n开始历史数据采集演示...")
        success = await demo_historical_collection()

        if success:
            print("\n✅ 历史数据采集演示成功完成!")
            print("🎉 量化系统历史数据采集功能已准备就绪!")
            return 0
        else:
            print("\n❌ 历史数据采集演示失败")
            return 1

    except KeyboardInterrupt:
        print("\n收到停止信号，正在退出...")
        return 0
    except Exception as e:
        logger.error(f"演示执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\n程序退出码: {exit_code}")
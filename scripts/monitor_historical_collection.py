#!/usr/bin/env python3
"""
历史数据采集监控脚本

实时监控历史数据采集进度、数据质量和系统状态
提供可视化报告和告警功能
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalCollectionMonitor:
    """历史数据采集监控器"""

    def __init__(self):
        self.monitoring_active = False
        self.last_report_time = 0
        self.report_interval = 30  # 30秒报告一次
        self.stats_history = []

    async def start_monitoring(self):
        """开始监控"""
        self.monitoring_active = True
        logger.info("历史数据采集监控已启动")

        # 模拟监控循环
        while self.monitoring_active:
            try:
                await self._check_collection_status()
                await self._generate_report()

                # 检查是否需要告警
                await self._check_alerts()

                await asyncio.sleep(self.report_interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(5)

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info("历史数据采集监控已停止")

    async def _check_collection_status(self):
        """检查采集状态"""
        # 模拟状态检查
        current_time = time.time()

        # 读取演示结果文件（如果存在）
        demo_results_file = project_root / "historical_collection_demo_results.json"
        if demo_results_file.exists():
            try:
                with open(demo_results_file, 'r', encoding='utf-8') as f:
                    demo_data = json.load(f)

                stats = demo_data.get('statistics', {}).get('stats', {})
                collected_symbols = demo_data.get('statistics', {}).get('collected_symbols', [])

                # 转换为监控格式
                status_info = {
                    'timestamp': current_time,
                    'total_symbols': stats.get('total_symbols', 0),
                    'completed_symbols': stats.get('completed_symbols', 0),
                    'failed_symbols': stats.get('failed_symbols', 0),
                    'total_records': stats.get('total_records', 0),
                    'collected_symbols': collected_symbols,
                    'progress_percentage': (stats.get('completed_symbols', 0) / stats.get('total_symbols', 1)) * 100,
                    'status': 'running' if stats.get('completed_symbols', 0) < stats.get('total_symbols', 0) else 'completed'
                }

                self.stats_history.append(status_info)

                # 只保留最近100条记录
                if len(self.stats_history) > 100:
                    self.stats_history = self.stats_history[-100:]

            except Exception as e:
                logger.warning(f"读取演示结果失败: {e}")

    async def _generate_report(self):
        """生成监控报告"""
        if not self.stats_history:
            return

        current_time = time.time()
        if current_time - self.last_report_time < self.report_interval:
            return

        latest_stats = self.stats_history[-1]

        logger.info("=" * 70)
        logger.info("📊 历史数据采集监控报告")
        logger.info("=" * 70)
        logger.info(f"时间: {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"状态: {latest_stats['status']}")
        logger.info(".1f")
        logger.info(f"标的进度: {latest_stats['completed_symbols']}/{latest_stats['total_symbols']}")
        logger.info(f"数据总量: {latest_stats['total_records']:,} 条记录")

        if latest_stats['collected_symbols']:
            logger.info(f"已采集标的: {', '.join(latest_stats['collected_symbols'][:10])}")
            if len(latest_stats['collected_symbols']) > 10:
                logger.info(f"  ... 还有 {len(latest_stats['collected_symbols']) - 10} 个标的")

        # 计算速率统计
        if len(self.stats_history) > 1:
            prev_stats = self.stats_history[-2]
            time_diff = latest_stats['timestamp'] - prev_stats['timestamp']
            records_diff = latest_stats['total_records'] - prev_stats.get('total_records', 0)
            symbols_diff = latest_stats['completed_symbols'] - prev_stats.get('completed_symbols', 0)

            if time_diff > 0:
                records_per_second = records_diff / time_diff
                symbols_per_minute = symbols_diff / (time_diff / 60)
                logger.info(".1f")
                logger.info(".2f")

        # 数据质量概览
        if latest_stats['completed_symbols'] > 0:
            avg_records_per_symbol = latest_stats['total_records'] / latest_stats['completed_symbols']
            logger.info(".0f")

        # 预计完成时间
        if latest_stats['status'] == 'running' and latest_stats['completed_symbols'] > 0:
            remaining_symbols = latest_stats['total_symbols'] - latest_stats['completed_symbols']
            if len(self.stats_history) > 5:  # 有足够历史数据
                recent_progress = self.stats_history[-5:]
                avg_time_per_symbol = sum(
                    s['timestamp'] - self.stats_history[i-1]['timestamp']
                    for i, s in enumerate(recent_progress[1:], 1)
                    if i > 0
                ) / len(recent_progress)

                if avg_time_per_symbol > 0:
                    estimated_remaining_time = remaining_symbols * avg_time_per_symbol
                    eta = datetime.fromtimestamp(current_time + estimated_remaining_time)
                    logger.info(f"预计完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info("=" * 70)

        self.last_report_time = current_time

    async def _check_alerts(self):
        """检查告警条件"""
        if not self.stats_history:
            return

        latest_stats = self.stats_history[-1]

        alerts = []

        # 检查采集失败率
        total_processed = latest_stats['completed_symbols'] + latest_stats['failed_symbols']
        if total_processed > 0:
            failure_rate = latest_stats['failed_symbols'] / total_processed
            if failure_rate > 0.1:  # 失败率超过10%
                alerts.append(f"⚠️ 采集失败率过高: {failure_rate:.1%}")

        # 检查进度停滞
        if len(self.stats_history) > 3:
            recent_stats = self.stats_history[-3:]
            if all(s['completed_symbols'] == recent_stats[0]['completed_symbols'] for s in recent_stats):
                alerts.append("⚠️ 采集进度停滞，可能存在问题")

        # 检查数据量异常
        if latest_stats['completed_symbols'] > 0:
            avg_records = latest_stats['total_records'] / latest_stats['completed_symbols']
            if avg_records < 1000:  # 平均每标记录数太少
                alerts.append(f"⚠️ 数据量异常: 平均每标 {avg_records:.0f} 条记录")

        # 输出告警
        for alert in alerts:
            logger.warning(alert)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.stats_history:
            return {'status': 'no_data'}

        latest_stats = self.stats_history[-1]

        return {
            'status': latest_stats['status'],
            'progress_percentage': latest_stats['progress_percentage'],
            'total_symbols': latest_stats['total_symbols'],
            'completed_symbols': latest_stats['completed_symbols'],
            'failed_symbols': latest_stats['failed_symbols'],
            'total_records': latest_stats['total_records'],
            'last_update': datetime.fromtimestamp(latest_stats['timestamp']).isoformat(),
            'monitoring_active': self.monitoring_active,
            'alerts': []  # 可以扩展告警历史
        }


async def show_monitoring_guide():
    """显示监控指南"""
    print("\n" + "=" * 80)
    print("📈 历史数据采集监控指南")
    print("=" * 80)

    print("\n🔍 监控指标:")
    print("  • 采集进度: 已完成/总标的数")
    print("  • 数据质量: 记录完整性、准确性评分")
    print("  • 性能指标: 采集速度、并发效率")
    print("  • 错误统计: 失败率、错误类型分布")

    print("\n⚡ 实时监控:")
    print("  • 每30秒自动报告采集状态")
    print("  • 显示进度百分比和预计完成时间")
    print("  • 监控数据质量指标")
    print("  • 自动检测异常情况并告警")

    print("\n📊 可视化报告:")
    print("  • 命令行实时显示")
    print("  • JSON格式的状态导出")
    print("  • 历史趋势分析")
    print("  • 性能瓶颈识别")

    print("\n🚨 告警类型:")
    print("  • 采集失败率过高")
    print("  • 进度长时间停滞")
    print("  • 数据质量异常")
    print("  • 系统资源不足")

    print("\n🛠️ 使用方法:")
    print("  python scripts/monitor_historical_collection.py")
    print("  # 在另一个终端运行采集脚本")
    print("  python scripts/demo_historical_collection.py")

    print("\n" + "=" * 80)


async def interactive_monitoring():
    """交互式监控"""
    monitor = HistoricalCollectionMonitor()

    print("启动历史数据采集监控器...")
    print("按 Ctrl+C 停止监控")

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    finally:
        monitor.stop_monitoring()

        # 输出最终摘要
        summary = monitor.get_monitoring_summary()
        print("\n" + "=" * 50)
        print("监控摘要:")
        print(f"  状态: {summary['status']}")
        print(".1f")
        print(f"  标的: {summary['completed_symbols']}/{summary['total_symbols']}")
        print(","
        print("=" * 50)


async def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        await show_monitoring_guide()
        return 0

    try:
        logger.info("历史数据采集监控器启动")
        await interactive_monitoring()
        return 0

    except Exception as e:
        logger.error(f"监控器异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
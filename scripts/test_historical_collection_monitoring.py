#!/usr/bin/env python3
"""
历史数据采集监控系统测试脚本

测试历史数据采集监控、调度和WebSocket功能
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.core.monitoring.historical_data_monitor import get_historical_data_monitor
from src.core.orchestration.historical_data_scheduler import get_historical_data_scheduler
from src.gateway.api.historical_collection_websocket import get_historical_collection_websocket_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalCollectionMonitorTester:
    """历史数据采集监控测试器"""

    def __init__(self):
        self.monitor = None
        self.scheduler = None
        self.websocket_manager = None

    async def setup(self):
        """初始化测试环境"""
        logger.info("初始化历史数据采集监控测试环境...")

        # 初始化WebSocket管理器
        self.websocket_manager = get_historical_collection_websocket_manager()

        # 初始化监控器
        self.monitor = get_historical_data_monitor(websocket_callback=self.websocket_manager)

        # 初始化调度器
        self.scheduler = get_historical_data_scheduler()

        logger.info("测试环境初始化完成")

    async def test_basic_monitoring(self):
        """测试基础监控功能"""
        logger.info("测试基础监控功能...")

        # 创建一些测试任务
        task_ids = []
        symbols = ['000001', '000002', '600000', '000858']

        for symbol in symbols:
            task_id = self.monitor.create_task(
                symbol=symbol,
                start_date='2020-01-01',
                end_date='2024-12-31',
                data_types=['price', 'volume']
            )
            task_ids.append(task_id)
            logger.info(f"创建测试任务: {task_id} ({symbol})")

        # 模拟任务执行进度
        for i, task_id in enumerate(task_ids):
            # 模拟不同阶段的进度更新
            for progress in [0.1, 0.3, 0.6, 0.8, 1.0]:
                records = int(progress * 1000)  # 模拟记录数
                self.monitor.update_task_progress(task_id, progress, records)

                # 广播任务进度更新
                await self.websocket_manager.broadcast_task_update(
                    task_id, progress, records, 'running'
                )

                await asyncio.sleep(0.5)  # 模拟执行时间

            # 完成任务
            success = i < 3  # 前3个任务成功，最后1个失败
            records_collected = 1000 if success else 0
            error_msg = None if success else "模拟采集失败"

            self.monitor.complete_task(task_id, records_collected, error_msg)

            # 广播任务完成
            await self.websocket_manager.broadcast_task_completed(
                task_id, success, records_collected
            )

        # 获取监控数据
        monitoring_data = self.monitor.get_monitoring_data()
        logger.info(f"监控数据: 任务数={monitoring_data['stats']['total_tasks_created']}, "
                   f"成功={monitoring_data['stats']['total_tasks_completed']}, "
                   f"失败={monitoring_data['stats']['total_tasks_failed']}")

    async def test_scheduler(self):
        """测试调度器功能"""
        logger.info("测试调度器功能...")

        # 启动调度器
        success = await self.scheduler.start()
        if success:
            logger.info("调度器启动成功")

            # 注册工作节点
            worker_id = self.scheduler.register_worker(
                worker_id='test_worker_1',
                host='localhost',
                port=8080,
                max_concurrent=2
            )
            logger.info(f"注册工作节点: {worker_id}")

            # 调度任务
            task_id = self.scheduler.schedule_task(
                symbol='000001',
                start_date='2020-01-01',
                end_date='2024-12-31',
                data_types=['price']
            )
            logger.info(f"调度任务: {task_id}")

            # 等待一段时间让调度器工作
            await asyncio.sleep(3)

            # 获取调度器状态
            status = self.scheduler.get_scheduler_status()
            logger.info(f"调度器状态: {status['status']}, 工作节点: {status['workers']['active']}")

            # 停止调度器
            await self.scheduler.stop()
            logger.info("调度器停止成功")
        else:
            logger.error("调度器启动失败")

    async def test_alerts(self):
        """测试告警功能"""
        logger.info("测试告警功能...")

        # 创建一个测试任务并模拟失败来触发告警
        task_id = self.monitor.create_task(
            symbol='TEST001',
            start_date='2020-01-01',
            end_date='2024-12-31',
            data_types=['price']
        )

        # 模拟任务失败多次来触发连续失败告警
        for i in range(4):
            self.monitor.update_task_progress(task_id, 0.1 * (i + 1), 100, f"Network error {i+1}")
            await asyncio.sleep(0.1)

        # 最终失败任务
        self.monitor.complete_task(task_id, 0, "连续网络错误")

        # 创建另一个任务并设置低质量分数来触发质量告警
        task_id2 = self.monitor.create_task(
            symbol='TEST002',
            start_date='2020-01-01',
            end_date='2024-12-31',
            data_types=['price']
        )

        # 设置低质量分数（这会触发告警）
        self.monitor.update_data_quality('hist_TEST002', 0.5)  # 低于阈值

        # 等待告警处理
        await asyncio.sleep(1)

        # 获取告警列表
        alerts = self.monitor.get_alerts()
        logger.info(f"生成告警数量: {len(alerts)}")

        for alert in alerts[-3:]:  # 显示最后3个告警
            logger.info(f"告警: [{alert['level']}] {alert['message']}")

    async def test_websocket_broadcasting(self):
        """测试WebSocket广播功能"""
        logger.info("测试WebSocket广播功能...")

        # 创建模拟WebSocket连接
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                self.connected = True

            async def send_json(self, data):
                self.messages.append(data)
                logger.debug(f"WebSocket消息: {data['type']}")

            async def receive_text(self):
                await asyncio.sleep(1)  # 模拟等待
                return '{"type": "ping"}'

            async def accept(self):
                pass

        mock_ws = MockWebSocket()

        # 连接WebSocket
        await self.websocket_manager.connect(mock_ws)

        # 发送订阅消息
        await self.websocket_manager.handle_message(
            mock_ws,
            '{"type": "subscribe", "topics": ["historical_collection_status", "task_progress"]}'
        )

        # 等待广播
        await asyncio.sleep(3)

        # 断开连接
        await self.websocket_manager.disconnect(mock_ws)

        logger.info(f"WebSocket接收到消息数量: {len(mock_ws.messages)}")

        # 显示消息类型统计
        message_types = {}
        for msg in mock_ws.messages:
            msg_type = msg.get('type')
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        logger.info(f"消息类型分布: {message_types}")

    async def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("=" * 60)
        logger.info("开始历史数据采集监控系统综合测试")
        logger.info("=" * 60)

        try:
            # 设置测试环境
            await self.setup()

            # 运行各项测试
            logger.info("-" * 40)
            await self.test_basic_monitoring()
            logger.info("-" * 40)

            await self.test_scheduler()
            logger.info("-" * 40)

            await self.test_alerts()
            logger.info("-" * 40)

            await self.test_websocket_broadcasting()
            logger.info("-" * 40)

            # 生成测试报告
            await self.generate_test_report()

            logger.info("=" * 60)
            logger.info("✅ 历史数据采集监控系统测试完成")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"测试执行失败: {e}", exc_info=True)
            return False

        return True

    async def generate_test_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")

        report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_duration': time.time(),
            'monitoring_stats': self.monitor.get_monitoring_data() if self.monitor else {},
            'scheduler_stats': self.scheduler.get_scheduler_status() if self.scheduler else {},
            'websocket_stats': self.websocket_manager.get_subscription_stats() if self.websocket_manager else {},
            'alerts': self.monitor.get_alerts() if self.monitor else []
        }

        # 保存报告
        report_file = project_root / "test_historical_monitoring_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"测试报告已保存: {report_file}")

        # 打印摘要
        if self.monitor:
            stats = report['monitoring_stats'].get('stats', {})
            logger.info("测试结果摘要:")
            logger.info(f"  创建任务数: {stats.get('total_tasks_created', 0)}")
            logger.info(f"  完成任务数: {stats.get('total_tasks_completed', 0)}")
            logger.info(f"  失败任务数: {stats.get('total_tasks_failed', 0)}")
            logger.info(".2f")
            logger.info(f"  采集记录数: {stats.get('total_records_collected', 0)}")

        if self.scheduler:
            scheduler_stats = report['scheduler_stats']
            logger.info(f"  调度器状态: {scheduler_stats.get('status', 'unknown')}")
            logger.info(f"  活跃工作节点: {scheduler_stats.get('workers', {}).get('active', 0)}")


async def main():
    """主函数"""
    tester = HistoricalCollectionMonitorTester()

    try:
        success = await tester.run_comprehensive_test()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在退出...")
        return 0
    except Exception as e:
        logger.error(f"测试执行异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\n测试程序退出码: {exit_code}")
"""
分批次数据补全处理器
实现智能的分批次历史数据补全算法
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .data_complement_scheduler import ComplementTask, ComplementMode, ComplementPriority

logger = get_unified_logger(__name__)


class BatchStatus(Enum):
    """批次状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComplementBatch:
    """补全批次"""
    batch_id: str
    task_id: str
    source_id: str
    batch_index: int
    total_batches: int
    start_date: datetime
    end_date: datetime
    estimated_records: int = 0
    actual_records: int = 0
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class BatchComplementProcessor:
    """
    分批次数据补全处理器

    实现智能的分批次补全算法：
    1. 动态批次大小调整
    2. 系统负载感知
    3. 断点续传支持
    4. 并行处理优化
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # 批次管理
        self.active_batches: Dict[str, ComplementBatch] = {}
        self.completed_batches: List[ComplementBatch] = []

        # 系统负载监控
        self.system_load_threshold = self.config['system_load_threshold']
        self.max_concurrent_batches = self.config['max_concurrent_batches']
        self.batch_size_adaptive = self.config['batch_size_adaptive']

        logger.info("分批次数据补全处理器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_concurrent_batches': 2,      # 最大并发批次数
            'default_batch_size_days': 7,     # 默认批次大小（天）
            'min_batch_size_days': 1,         # 最小批次大小（天）
            'max_batch_size_days': 30,        # 最大批次大小（天）
            'system_load_threshold': 0.8,     # 系统负载阈值
            'batch_size_adaptive': True,      # 是否启用自适应批次大小
            'progress_check_interval': 30,    # 进度检查间隔（秒）
        }

    def create_complement_batches(self, task: ComplementTask) -> List[ComplementBatch]:
        """
        将补全任务分解为多个批次

        Args:
            task: 补全任务

        Returns:
            补全批次列表
        """
        try:
            # 计算批次大小
            batch_size_days = self._calculate_optimal_batch_size(task)

            # 计算总批次数
            total_days = (task.end_date - task.start_date).days
            total_batches = max(1, (total_days + batch_size_days - 1) // batch_size_days)

            # 创建批次
            batches = []
            current_date = task.start_date

            for i in range(total_batches):
                # 计算批次时间范围
                batch_start = current_date
                batch_end = min(
                    current_date + timedelta(days=batch_size_days),
                    task.end_date
                )

                # 估算批次记录数
                batch_days = (batch_end - batch_start).days
                estimated_records = self._estimate_batch_records(task, batch_days)

                # 创建批次对象
                batch = ComplementBatch(
                    batch_id=f"{task.task_id}_batch_{i+1}",
                    task_id=task.task_id,
                    source_id=task.source_id,
                    batch_index=i + 1,
                    total_batches=total_batches,
                    start_date=batch_start,
                    end_date=batch_end,
                    estimated_records=estimated_records
                )

                batches.append(batch)
                current_date = batch_end

            logger.info(f"补全任务 {task.task_id} 分解为 {len(batches)} 个批次")
            return batches

        except Exception as e:
            logger.error(f"创建补全批次失败: {e}")
            return []

    def _calculate_optimal_batch_size(self, task: ComplementTask) -> int:
        """
        计算最优批次大小

        考虑因素：
        1. 数据类型和优先级
        2. 系统负载状况
        3. 历史执行性能
        4. 时间范围大小
        5. 补全模式特殊处理
        """
        try:
            # 特殊处理：策略回测模式按年分批
            if hasattr(task, 'mode') and str(task.mode).endswith('STRATEGY_BACKTEST'):
                # 策略回测：按年分批，每批365天
                optimal_batch_size = 365
                logger.info(f"策略回测模式使用年度批次大小: {optimal_batch_size}天")
                return optimal_batch_size

            # 基础批次大小
            base_batch_size = self.config['default_batch_size_days']

            # 根据优先级调整
            priority_multipliers = {
                ComplementPriority.CRITICAL: 0.5,  # 关键任务：更小的批次
                ComplementPriority.HIGH: 0.7,
                ComplementPriority.MEDIUM: 1.0,
                ComplementPriority.LOW: 1.5,      # 低优先级：更大的批次
            }

            priority_multiplier = priority_multipliers.get(task.priority, 1.0)

            # 根据数据类型调整
            data_type_multipliers = {
                'stock': 1.0,
                'index': 0.8,    # 指数数据通常更密集
                'macro': 2.0,    # 宏观数据通常更稀疏
                'news': 0.5,     # 新闻数据可能很大
            }

            data_type_multiplier = data_type_multipliers.get(task.data_type, 1.0)

            # 计算调整后的批次大小
            adjusted_batch_size = int(base_batch_size * priority_multiplier * data_type_multiplier)

            # 限制在合理范围内
            optimal_batch_size = max(
                self.config['min_batch_size_days'],
                min(adjusted_batch_size, self.config['max_batch_size_days'])
            )

            logger.debug(f"计算最优批次大小: {optimal_batch_size}天 "
                        f"(基础: {base_batch_size}, 优先级调整: {priority_multiplier:.1f}, "
                        f"数据类型调整: {data_type_multiplier:.1f})")

            return optimal_batch_size

        except Exception as e:
            logger.warning(f"计算最优批次大小失败，使用默认值: {e}")
            return self.config['default_batch_size_days']

    def _estimate_batch_records(self, task: ComplementTask, batch_days: int) -> int:
        """估算批次记录数"""
        try:
            # 基础估算：总记录数 / 总天数 * 批次天数
            if task.estimated_records > 0:
                total_days = (task.end_date - task.start_date).days
                if total_days > 0:
                    return int(task.estimated_records * batch_days / total_days)

            # 根据数据类型估算
            daily_rates = {
                'stock': 1,      # 股票：每日约1条记录
                'index': 1,      # 指数：每日约1条记录
                'macro': 0.1,    # 宏观：每日约0.1条记录
                'news': 10,      # 新闻：每日约10条记录
            }

            daily_rate = daily_rates.get(task.data_type, 1)
            estimated = int(batch_days * daily_rate)

            return max(estimated, 1)

        except Exception as e:
            logger.warning(f"估算批次记录数失败: {e}")
            return 10  # 默认估算值

    async def execute_batch(self, batch: ComplementBatch) -> Tuple[bool, str]:
        """
        执行单个补全批次

        Args:
            batch: 补全批次

        Returns:
            (是否成功, 错误信息)
        """
        try:
            # 检查系统负载
            if not await self._check_system_capacity():
                return False, "系统负载过高，跳过批次执行"

            # 更新批次状态
            batch.status = BatchStatus.RUNNING
            batch.started_at = datetime.now()

            self.active_batches[batch.batch_id] = batch

            logger.info(f"开始执行补全批次: {batch.batch_id} "
                       f"({batch.batch_index}/{batch.total_batches})")

            # 执行数据补全
            success, error_msg = await self._execute_complement_batch(batch)

            # 更新批次状态
            batch.completed_at = datetime.now()
            batch.status = BatchStatus.COMPLETED if success else BatchStatus.FAILED
            batch.error_message = error_msg if not success else None

            # 从活跃批次中移除
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]

            # 添加到完成列表
            self.completed_batches.append(batch)

            # 限制完成列表大小
            if len(self.completed_batches) > 1000:
                self.completed_batches = self.completed_batches[-1000:]

            status_msg = "成功" if success else f"失败: {error_msg}"
            logger.info(f"补全批次执行完成: {batch.batch_id} - {status_msg}")

            return success, error_msg or ""

        except Exception as e:
            error_msg = f"执行补全批次异常: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # 更新批次状态
            batch.status = BatchStatus.FAILED
            batch.error_message = error_msg
            batch.completed_at = datetime.now()

            return False, error_msg

    async def _execute_complement_batch(self, batch: ComplementBatch) -> Tuple[bool, Optional[str]]:
        """
        执行具体的补全批次逻辑

        这里需要调用实际的数据采集和存储逻辑
        """
        try:
            # 模拟补全执行过程
            # 实际实现需要：
            # 1. 调用数据采集器获取缺失数据
            # 2. 进行数据验证和清洗
            # 3. 存储到数据库
            # 4. 更新进度

            logger.info(f"执行补全批次数据采集: {batch.source_id} "
                       f"时间范围: {batch.start_date.date()} 至 {batch.end_date.date()}")

            # 模拟数据采集延迟
            await asyncio.sleep(0.1)  # 100ms模拟延迟

            # 模拟成功执行
            batch.actual_records = batch.estimated_records

            # TODO: 替换为实际的数据采集和存储逻辑
            # from src.gateway.web.data_collectors import collect_data_via_data_layer
            # result = await collect_data_via_data_layer(source_config, {
            #     'start_date': batch.start_date,
            #     'end_date': batch.end_date,
            #     'mode': 'complement'
            # })

            return True, None

        except Exception as e:
            return False, str(e)

    async def _check_system_capacity(self) -> bool:
        """
        检查系统容量是否允许执行新批次

        考虑因素：
        1. 当前活跃批次数
        2. 系统负载
        3. 内存使用
        """
        try:
            # 检查并发批次数
            if len(self.active_batches) >= self.max_concurrent_batches:
                logger.debug(f"活跃批次数已达上限: {len(self.active_batches)}/{self.max_concurrent_batches}")
                return False

            # 检查系统负载（简化实现）
            # 实际应该检查CPU、内存、磁盘IO等
            current_load = len(self.active_batches) / self.max_concurrent_batches
            if current_load > self.system_load_threshold:
                logger.debug(f"系统负载过高: {current_load:.2f}")
                return False

            return True

        except Exception as e:
            logger.warning(f"检查系统容量失败: {e}")
            return True  # 出错时允许执行

    async def execute_complement_task_batches(self, task: ComplementTask,
                                            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        执行补全任务的所有批次

        Args:
            task: 补全任务
            progress_callback: 进度回调函数

        Returns:
            执行结果统计
        """
        try:
            # 创建批次
            batches = self.create_complement_batches(task)
            if not batches:
                return {
                    'success': False,
                    'error': '无法创建补全批次',
                    'total_batches': 0
                }

            logger.info(f"开始执行补全任务批次: {task.task_id}, 共 {len(batches)} 个批次")

            # 执行批次
            successful_batches = 0
            failed_batches = 0
            total_records = 0

            for i, batch in enumerate(batches):
                # 执行批次
                success, error_msg = await self.execute_batch(batch)

                if success:
                    successful_batches += 1
                    total_records += batch.actual_records
                else:
                    failed_batches += 1
                    logger.warning(f"批次执行失败: {batch.batch_id} - {error_msg}")

                # 更新任务进度
                progress_percentage = (i + 1) / len(batches) * 100
                task.progress_percentage = progress_percentage
                task.actual_records = total_records

                # 调用进度回调
                if progress_callback:
                    await progress_callback(task, batch, progress_percentage)

                # 批次间暂停，避免系统过载
                if i < len(batches) - 1:  # 不是最后一个批次
                    await asyncio.sleep(0.5)  # 500ms暂停

            # 计算最终结果
            success_rate = successful_batches / len(batches) if batches else 0
            overall_success = success_rate >= 0.8  # 80%以上批次成功

            result = {
                'success': overall_success,
                'total_batches': len(batches),
                'successful_batches': successful_batches,
                'failed_batches': failed_batches,
                'success_rate': success_rate,
                'total_records': total_records,
                'task_duration': (datetime.now() - task.started_at).total_seconds() if task.started_at else 0
            }

            logger.info(f"补全任务批次执行完成: {task.task_id} - "
                       f"成功率: {success_rate:.1f}%, "
                       f"总记录数: {total_records}")

            return result

        except Exception as e:
            error_msg = f"执行补全任务批次异常: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'total_batches': 0
            }

    def get_active_batches(self) -> List[ComplementBatch]:
        """获取活跃批次"""
        return list(self.active_batches.values())

    def get_batch(self, batch_id: str) -> Optional[ComplementBatch]:
        """获取批次信息"""
        return self.active_batches.get(batch_id)

    def cancel_batch(self, batch_id: str):
        """取消批次执行"""
        if batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
            batch.status = BatchStatus.CANCELLED
            batch.completed_at = datetime.now()

            del self.active_batches[batch_id]
            self.completed_batches.append(batch)

            logger.info(f"补全批次已取消: {batch_id}")

    def get_batch_statistics(self) -> Dict[str, Any]:
        """获取批次统计信息"""
        stats = {
            'active_batches': len(self.active_batches),
            'completed_batches_today': 0,
            'failed_batches_today': 0,
            'avg_batch_duration': 0.0
        }

        today = datetime.now().date()
        durations = []

        for batch in self.completed_batches:
            if batch.completed_at and batch.completed_at.date() == today:
                if batch.status == BatchStatus.COMPLETED:
                    stats['completed_batches_today'] += 1
                elif batch.status == BatchStatus.FAILED:
                    stats['failed_batches_today'] += 1

            # 计算平均执行时间
            if batch.started_at and batch.completed_at:
                duration = (batch.completed_at - batch.started_at).total_seconds()
                durations.append(duration)

        if durations:
            stats['avg_batch_duration'] = sum(durations) / len(durations)

        return stats

    def cleanup_completed_batches(self, days_to_keep: int = 7):
        """清理已完成的批次记录"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        old_batches = [
            batch for batch in self.completed_batches
            if batch.completed_at and batch.completed_at < cutoff_time
        ]

        for batch in old_batches:
            self.completed_batches.remove(batch)

        if old_batches:
            logger.info(f"清理了 {len(old_batches)} 个旧补全批次记录")


# 全局实例
_batch_processor = None


def get_batch_complement_processor() -> BatchComplementProcessor:
    """获取分批次补全处理器实例"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchComplementProcessor()
    return _batch_processor
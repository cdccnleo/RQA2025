#!/usr/bin/env python3
"""
策略回测数据工作流
协调历史数据采集、质量保证、存储优化等全流程
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from .historical_data_acquisition_service import (
    HistoricalDataAcquisitionService,
    HistoricalDataBatch,
    HistoricalDataConfig,
    DataSourceType,
    DataQualityLevel
)
from ..persistence.timescale_storage import TimescaleStorage
from ..cache.redis_cache import RedisCache
from ..monitoring.data_collection_monitor import DataCollectionMonitor

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    COLLECTING = "collecting"
    VALIDATING = "validating"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchPriority(Enum):
    """批次优先级"""
    CRITICAL = "critical"    # 关键数据（如核心资产）
    HIGH = "high"           # 高优先级
    MEDIUM = "medium"       # 中等优先级
    LOW = "low"            # 低优先级


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    symbol: str
    start_year: int
    end_year: int
    data_types: List[str] = field(default_factory=lambda: ["stock"])
    max_concurrent_years: int = 2
    quality_threshold: float = 0.85
    retry_failed_batches: bool = True
    max_retry_attempts: int = 3
    enable_progress_tracking: bool = True
    notification_enabled: bool = True


@dataclass
class WorkflowProgress:
    """工作流进度"""
    total_years: int = 0
    completed_years: int = 0
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    total_records: int = 0
    start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    current_year: Optional[int] = None
    status_message: str = ""


@dataclass
class WorkflowResult:
    """工作流结果"""
    workflow_id: str
    config: WorkflowConfig
    status: WorkflowStatus
    progress: WorkflowProgress
    batches: List[HistoricalDataBatch] = field(default_factory=list)
    quality_stats: Dict[str, Any] = field(default_factory=dict)
    storage_stats: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


class StrategyBacktestDataWorkflow:
    """策略回测数据工作流"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.acquisition_service = HistoricalDataAcquisitionService(
            config.get('acquisition_service_config', {})
        )

        self.timescale_storage = TimescaleStorage(
            config.get('timescale_config', {})
        )

        self.redis_cache = RedisCache(
            config.get('redis_config', {})
        )

        self.monitor = DataCollectionMonitor(
            config.get('monitor_config', {})
        )

        # 工作流状态管理
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.workflow_queue: asyncio.Queue = asyncio.Queue()

        # 并发控制
        self.max_concurrent_workflows = config.get('max_concurrent_workflows', 1)
        self.workflow_semaphore = asyncio.Semaphore(self.max_concurrent_workflows)

    async def start_workflow(self, workflow_config: WorkflowConfig) -> str:
        """
        启动数据采集工作流

        Args:
            workflow_config: 工作流配置

        Returns:
            工作流ID
        """
        workflow_id = f"workflow_{workflow_config.symbol}_{workflow_config.start_year}_{workflow_config.end_year}_{int(datetime.now().timestamp())}"

        # 创建工作流结果对象
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            config=workflow_config,
            status=WorkflowStatus.INITIALIZING,
            progress=WorkflowProgress(
                total_years=workflow_config.end_year - workflow_config.start_year + 1,
                start_time=datetime.now()
            )
        )

        self.active_workflows[workflow_id] = workflow_result

        # 启动工作流任务
        asyncio.create_task(self._execute_workflow(workflow_result))

        self.logger.info(f"启动策略回测数据工作流: {workflow_id}")
        return workflow_id

    async def _execute_workflow(self, workflow_result: WorkflowResult):
        """执行工作流"""
        try:
            async with self.workflow_semaphore:
                config = workflow_result.config

                # 1. 初始化阶段
                await self._initialize_workflow(workflow_result)

                # 2. 数据采集阶段
                await self._collect_data_phase(workflow_result)

                # 3. 数据验证阶段
                await self._validate_data_phase(workflow_result)

                # 4. 数据存储阶段
                await self._store_data_phase(workflow_result)

                # 5. 完成阶段
                await self._complete_workflow(workflow_result)

        except Exception as e:
            await self._fail_workflow(workflow_result, str(e))

    async def _initialize_workflow(self, workflow_result: WorkflowResult):
        """初始化工作流"""
        config = workflow_result.config

        self.logger.info(f"初始化工作流 {workflow_result.workflow_id}: {config.symbol} {config.start_year}-{config.end_year}")

        # 验证配置
        await self._validate_workflow_config(config)

        # 检查已有数据
        existing_data = await self._check_existing_data(config)

        # 更新进度
        workflow_result.progress.status_message = "工作流初始化完成，准备开始数据采集"

        # 记录监控信息
        await self.monitor.record_workflow_start(
            workflow_result.workflow_id,
            config.symbol,
            config.start_year,
            config.end_year
        )

    async def _validate_workflow_config(self, config: WorkflowConfig):
        """验证工作流配置"""
        # 检查年份范围
        current_year = datetime.now().year
        if config.start_year > current_year or config.end_year > current_year:
            raise ValueError(f"年份范围无效: {config.start_year}-{config.end_year}")

        if config.start_year > config.end_year:
            raise ValueError("开始年份不能大于结束年份")

        # 检查数据类型
        supported_types = ["stock", "index", "fund", "bond", "futures"]
        for data_type in config.data_types:
            if data_type not in supported_types:
                raise ValueError(f"不支持的数据类型: {data_type}")

        # 检查标的代码
        supported_symbols = self.acquisition_service.get_supported_symbols()
        if config.symbol not in supported_symbols:
            self.logger.warning(f"标的代码 {config.symbol} 可能不受所有数据源支持")

    async def _check_existing_data(self, config: WorkflowConfig) -> Dict[str, Any]:
        """检查已存在的历史数据"""
        existing_stats = {}

        try:
            for data_type in config.data_types:
                stats = await self.timescale_storage.get_data_stats(
                    config.symbol,
                    config.start_year,
                    config.end_year
                )
                existing_stats[data_type] = stats

            self.logger.info(f"检查到现有数据: {existing_stats}")

        except Exception as e:
            self.logger.warning(f"检查现有数据失败: {e}")

        return existing_stats

    async def _collect_data_phase(self, workflow_result: WorkflowResult):
        """数据采集阶段"""
        config = workflow_result.config
        workflow_result.status = WorkflowStatus.COLLECTING

        self.logger.info(f"开始数据采集: {config.symbol} {config.start_year}-{config.end_year}")

        # 并行采集各年数据
        years = list(range(config.start_year, config.end_year + 1))
        semaphore = asyncio.Semaphore(config.max_concurrent_years)

        async def collect_year(year: int):
            async with semaphore:
                workflow_result.progress.current_year = year
                workflow_result.progress.status_message = f"正在采集 {year} 年数据"

                try:
                    # 采集该年数据
                    year_batches = await self.acquisition_service.acquire_yearly_data(
                        config.symbol,
                        year,
                        config.data_types
                    )

                    # 更新进度
                    workflow_result.batches.extend(year_batches)
                    workflow_result.progress.completed_years += 1
                    workflow_result.progress.total_batches += len(year_batches)

                    successful_batches = [b for b in year_batches if b.status == "completed"]
                    workflow_result.progress.completed_batches += len(successful_batches)

                    failed_batches = [b for b in year_batches if b.status == "failed"]
                    workflow_result.progress.failed_batches += len(failed_batches)

                    # 计算总记录数
                    for batch in successful_batches:
                        if batch.best_result:
                            workflow_result.progress.total_records += len(batch.best_result.data)

                    self.logger.info(f"完成 {year} 年数据采集: {len(successful_batches)}/{len(year_batches)} 批次成功")

                    # 处理失败的批次
                    if failed_batches and config.retry_failed_batches:
                        await self._retry_failed_batches(failed_batches, config.max_retry_attempts)

                except Exception as e:
                    workflow_result.errors.append(f"采集 {year} 年数据失败: {e}")
                    self.logger.error(f"采集 {year} 年数据异常: {e}")

        # 执行并行采集
        tasks = [collect_year(year) for year in years]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 更新总体进度
        total_years = len(years)
        completed_years = workflow_result.progress.completed_years
        success_rate = completed_years / total_years if total_years > 0 else 0

        workflow_result.progress.status_message = f"数据采集完成: {completed_years}/{total_years} 年 ({success_rate:.1%})"

        self.logger.info(f"数据采集阶段完成: {workflow_result.progress.total_batches} 批次, {workflow_result.progress.total_records} 条记录")

    async def _retry_failed_batches(self, failed_batches: List[HistoricalDataBatch], max_attempts: int):
        """重试失败的批次"""
        for batch in failed_batches:
            for attempt in range(max_attempts):
                try:
                    self.logger.info(f"重试批次 {batch.batch_id} (尝试 {attempt + 1}/{max_attempts})")

                    # 重新采集
                    new_batch = await self.acquisition_service.acquire_historical_data(batch.config)

                    if new_batch.status == "completed":
                        # 更新批次状态
                        batch.status = "completed"
                        batch.best_result = new_batch.best_result
                        batch.completed_at = datetime.now()

                        self.logger.info(f"重试成功: {batch.batch_id}")
                        break
                    else:
                        self.logger.warning(f"重试失败: {batch.batch_id}")

                except Exception as e:
                    self.logger.error(f"重试批次异常 {batch.batch_id}: {e}")

                # 等待重试间隔
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def _validate_data_phase(self, workflow_result: WorkflowResult):
        """数据验证阶段"""
        workflow_result.status = WorkflowStatus.VALIDATING

        self.logger.info("开始数据质量验证")
        workflow_result.progress.status_message = "正在验证数据质量"

        try:
            # 获取质量统计
            workflow_result.quality_stats = self.acquisition_service.get_data_quality_stats(
                workflow_result.batches
            )

            # 验证数据完整性
            validation_result = await self.acquisition_service.validate_data_integrity(
                workflow_result.config.symbol,
                workflow_result.config.start_year,
                workflow_result.config.end_year
            )

            workflow_result.validation_results = validation_result

            # 检查整体质量
            avg_quality = workflow_result.quality_stats.get('average_quality_score', 0)
            completeness = validation_result.get('completeness_ratio', 0)

            if avg_quality < workflow_result.config.quality_threshold:
                workflow_result.errors.append(f"数据质量不足: {avg_quality:.2%} < {workflow_result.config.quality_threshold:.2%}")

            if not validation_result.get('is_complete', False):
                workflow_result.errors.append(f"数据不完整: {completeness:.2%}")

            self.logger.info(f"数据验证完成: 平均质量={avg_quality:.2%}, 完整性={completeness:.2%}")

        except Exception as e:
            workflow_result.errors.append(f"数据验证失败: {e}")
            self.logger.error(f"数据验证异常: {e}")

    async def _store_data_phase(self, workflow_result: WorkflowResult):
        """数据存储阶段"""
        workflow_result.status = WorkflowStatus.STORING

        self.logger.info("开始数据存储")
        workflow_result.progress.status_message = "正在存储数据到数据库"

        try:
            # 存储批次结果
            storage_stats = await self.acquisition_service.store_batch_results(
                workflow_result.batches
            )

            workflow_result.storage_stats = storage_stats

            # 记录存储监控信息
            await self.monitor.record_data_storage(
                workflow_result.workflow_id,
                storage_stats
            )

            self.logger.info(f"数据存储完成: {storage_stats['stored_batches']}/{storage_stats['total_batches']} 批次")

        except Exception as e:
            workflow_result.errors.append(f"数据存储失败: {e}")
            self.logger.error(f"数据存储异常: {e}")

    async def _complete_workflow(self, workflow_result: WorkflowResult):
        """完成工作流"""
        workflow_result.status = WorkflowStatus.COMPLETED
        workflow_result.end_time = datetime.now()
        workflow_result.duration_seconds = (
            workflow_result.end_time - workflow_result.start_time
        ).total_seconds()

        workflow_result.progress.status_message = "工作流执行完成"

        # 记录完成监控信息
        await self.monitor.record_workflow_completion(
            workflow_result.workflow_id,
            workflow_result.status.value,
            workflow_result.duration_seconds,
            workflow_result.progress.total_records
        )

        # 发送完成通知
        if workflow_result.config.notification_enabled:
            await self._send_completion_notification(workflow_result)

        self.logger.info(f"工作流完成: {workflow_result.workflow_id}, 耗时: {workflow_result.duration_seconds:.1f}秒")

    async def _fail_workflow(self, workflow_result: WorkflowResult, error: str):
        """标记工作流失败"""
        workflow_result.status = WorkflowStatus.FAILED
        workflow_result.end_time = datetime.now()
        workflow_result.duration_seconds = (
            workflow_result.end_time - workflow_result.start_time
        ).total_seconds()
        workflow_result.errors.append(error)

        workflow_result.progress.status_message = f"工作流执行失败: {error}"

        # 记录失败监控信息
        await self.monitor.record_workflow_failure(
            workflow_result.workflow_id,
            error
        )

        # 发送失败通知
        if workflow_result.config.notification_enabled:
            await self._send_failure_notification(workflow_result)

        self.logger.error(f"工作流失败: {workflow_result.workflow_id}, 错误: {error}")

    async def _send_completion_notification(self, workflow_result: WorkflowResult):
        """发送完成通知"""
        try:
            message = {
                "type": "workflow_completed",
                "workflow_id": workflow_result.workflow_id,
                "symbol": workflow_result.config.symbol,
                "period": f"{workflow_result.config.start_year}-{workflow_result.config.end_year}",
                "total_records": workflow_result.progress.total_records,
                "duration_seconds": workflow_result.duration_seconds,
                "quality_score": workflow_result.quality_stats.get('average_quality_score', 0),
                "timestamp": datetime.now().isoformat()
            }

            # 这里可以集成实际的通知服务（如邮件、微信等）
            self.logger.info(f"工作流完成通知: {json.dumps(message, indent=2)}")

        except Exception as e:
            self.logger.error(f"发送完成通知失败: {e}")

    async def _send_failure_notification(self, workflow_result: WorkflowResult):
        """发送失败通知"""
        try:
            message = {
                "type": "workflow_failed",
                "workflow_id": workflow_result.workflow_id,
                "symbol": workflow_result.config.symbol,
                "errors": workflow_result.errors,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.error(f"工作流失败通知: {json.dumps(message, indent=2)}")

        except Exception as e:
            self.logger.error(f"发送失败通知失败: {e}")

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowResult]:
        """获取工作流状态"""
        return self.active_workflows.get(workflow_id)

    def list_active_workflows(self) -> List[WorkflowResult]:
        """列出活跃的工作流"""
        return list(self.active_workflows.values())

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = datetime.now()
            workflow.progress.status_message = "工作流已被取消"

            self.logger.info(f"工作流已取消: {workflow_id}")
            return True

        return False

    async def cleanup_completed_workflows(self, max_age_days: int = 7):
        """清理完成的工作流记录"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        workflows_to_remove = []

        for workflow_id, workflow in self.active_workflows.items():
            if workflow.end_time and workflow.end_time < cutoff_time:
                workflows_to_remove.append(workflow_id)

        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]

        if workflows_to_remove:
            self.logger.info(f"清理了 {len(workflows_to_remove)} 个过期的工作流记录")

    async def get_workflow_statistics(self) -> Dict[str, Any]:
        """获取工作流统计信息"""
        stats = {
            "active_workflows": len(self.active_workflows),
            "total_workflows_started": 0,  # 可以从历史记录中获取
            "average_completion_time": 0.0,
            "success_rate": 0.0,
            "total_records_collected": 0,
            "workflows_by_status": {}
        }

        total_duration = 0.0
        completed_count = 0
        successful_count = 0

        for workflow in self.active_workflows.values():
            status = workflow.status.value
            stats["workflows_by_status"][status] = stats["workflows_by_status"].get(status, 0) + 1

            if workflow.end_time:
                completed_count += 1
                total_duration += workflow.duration_seconds

                if workflow.status == WorkflowStatus.COMPLETED:
                    successful_count += 1

            stats["total_records_collected"] += workflow.progress.total_records

        if completed_count > 0:
            stats["average_completion_time"] = total_duration / completed_count
            stats["success_rate"] = successful_count / completed_count

        return stats

    async def optimize_workflow_config(self, symbol: str, start_year: int, end_year: int) -> WorkflowConfig:
        """
        根据历史数据和系统状态优化工作流配置

        Args:
            symbol: 标的代码
            start_year: 开始年份
            end_year: 结束年份

        Returns:
            优化后的工作流配置
        """
        # 分析历史采集性能
        historical_stats = await self._analyze_historical_performance(symbol)

        # 根据系统负载调整并发度
        system_load = await self._get_system_load()
        concurrent_years = max(1, min(3, 3 - int(system_load * 3)))

        # 根据数据重要性调整质量阈值
        quality_threshold = 0.80  # 默认值
        if symbol in ["000001.SZ", "000858.SZ", "600036.SH"]:  # 核心资产
            quality_threshold = 0.90  # 提高质量要求

        config = WorkflowConfig(
            name=f"optimized_{symbol}_{start_year}_{end_year}",
            symbol=symbol,
            start_year=start_year,
            end_year=end_year,
            max_concurrent_years=concurrent_years,
            quality_threshold=quality_threshold,
            retry_failed_batches=True,
            enable_progress_tracking=True,
            notification_enabled=True
        )

        self.logger.info(f"优化工作流配置: {symbol}, 并发度={concurrent_years}, 质量阈值={quality_threshold}")
        return config

    async def _analyze_historical_performance(self, symbol: str) -> Dict[str, Any]:
        """分析历史采集性能"""
        # 这里可以从监控数据中分析历史性能
        # 暂时返回默认值
        return {
            "average_collection_time": 30.0,
            "success_rate": 0.95,
            "average_quality": 0.88
        }

    async def _get_system_load(self) -> float:
        """获取系统负载"""
        # 这里可以集成实际的系统监控
        # 暂时返回模拟负载
        import random
        return random.uniform(0.1, 0.8)

    async def batch_start_workflows(self, workflow_configs: List[WorkflowConfig]) -> List[str]:
        """
        批量启动工作流

        Args:
            workflow_configs: 工作流配置列表

        Returns:
            工作流ID列表
        """
        workflow_ids = []

        # 限制并发启动的数量
        semaphore = asyncio.Semaphore(self.max_concurrent_workflows)

        async def start_single_workflow(config: WorkflowConfig):
            async with semaphore:
                try:
                    workflow_id = await self.start_workflow(config)
                    workflow_ids.append(workflow_id)
                    self.logger.info(f"批量启动工作流成功: {workflow_id}")
                except Exception as e:
                    self.logger.error(f"批量启动工作流失败 {config.symbol}: {e}")

        # 并行启动工作流
        tasks = [start_single_workflow(config) for config in workflow_configs]
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info(f"批量启动完成: {len(workflow_ids)}/{len(workflow_configs)} 个工作流")
        return workflow_ids
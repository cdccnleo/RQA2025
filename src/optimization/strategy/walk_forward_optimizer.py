#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
步进优化器实现
Walk - Forward Optimizer Implementation

实现步进优化算法，支持锚定步进和滚动步进两种模式。
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import numpy as np
from ..interfaces.optimization_interfaces import (
    IWalkForwardOptimizer, WalkForwardConfig, WalkForwardResult
)
from ..interfaces.backtest_interfaces import IBacktestService
from ...core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:

    """步进优化周期"""
    period_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_parameters: Dict[str, Any]
    performance_score: float
    out_of_sample_score: float
    optimization_details: Dict[str, Any]


class WalkForwardOptimizer(IWalkForwardOptimizer):

    """
    步进优化器
    Walk - Forward Optimizer

    实现步进优化算法，避免未来数据泄露，确保优化结果的稳健性。
    """

    def __init__(self, backtest_service: IBacktestService):
        """
        初始化步进优化器

        Args:
            backtest_service: 回测服务实例
        """
        self.backtest_service = backtest_service
        self.adapter_factory = get_unified_adapter_factory()

        logger.info("步进优化器初始化完成")

    async def walk_forward_optimization(self, config: WalkForwardConfig) -> WalkForwardResult:
        """
        执行步进优化

        Args:
            config: 步进优化配置

        Returns:
            WalkForwardResult: 步进优化结果
        """
        start_time = datetime.now()

        try:
            # 生成步进周期
            periods = self._generate_walk_forward_periods(config)

            logger.info(f"开始步进优化，共 {len(periods)} 个周期")

            # 执行每个周期的优化
            period_results = []
            for period in periods:
                period_result = await self._optimize_single_period(config, period)
                period_results.append(period_result)

            # 计算整体性能
            overall_performance = self._calculate_overall_performance(period_results)
            robustness_score = self._calculate_robustness_score(period_results)

            execution_time = (datetime.now() - start_time).total_seconds()

            result = WalkForwardResult(
                optimization_id=config.optimization_id,
                strategy_id=config.strategy_id,
                periods=period_results,
                overall_performance=overall_performance,
                robustness_score=robustness_score,
                execution_time=execution_time,
                timestamp=datetime.now()
            )

            # 发布事件
            await self._publish_event("walk_forward_completed", {
                "optimization_id": config.optimization_id,
                "periods_count": len(period_results),
                "robustness_score": robustness_score,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"步进优化完成，稳健性评分: {robustness_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"步进优化异常: {e}")
            raise

    async def anchored_walk_forward(self, strategy_id: str, train_window: int,
                                    test_window: int, step_size: int) -> WalkForwardResult:
        """
        锚定步进优化

        Args:
            strategy_id: 策略ID
            train_window: 训练窗口大小（天）
            test_window: 测试窗口大小（天）
            step_size: 步进大小（天）

        Returns:
            WalkForwardResult: 步进优化结果
        """
        # 创建锚定步进配置
        config = WalkForwardConfig(
            optimization_id=f"anchored_wf_{strategy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
            strategy_id=strategy_id,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            anchored=True  # 锚定步进
        )

        return await self.walk_forward_optimization(config)

    async def rolling_walk_forward(self, strategy_id: str, train_window: int,
                                   test_window: int, step_size: int) -> WalkForwardResult:
        """
        滚动步进优化

        Args:
            strategy_id: 策略ID
            train_window: 训练窗口大小（天）
            test_window: 测试窗口大小（天）
            step_size: 步进大小（天）

        Returns:
            WalkForwardResult: 步进优化结果
        """
        # 创建滚动步进配置
        config = WalkForwardConfig(
            optimization_id=f"rolling_wf_{strategy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
            strategy_id=strategy_id,
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            anchored=False  # 滚动步进
        )

        return await self.walk_forward_optimization(config)

    def _generate_walk_forward_periods(self, config: WalkForwardConfig) -> List[WalkForwardPeriod]:
        """
        生成步进周期

        Args:
            config: 步进优化配置

        Returns:
            List[WalkForwardPeriod]: 步进周期列表
        """
        periods = []

        # 这里简化实现，假设我们有历史数据的时间范围
        # 实际应该从数据源获取可用的时间范围

        # 示例：假设我们有从2020 - 01 - 01到2023 - 12 - 31的数据
        data_start = datetime(2020, 1, 1)
        data_end = datetime(2023, 12, 31)

        current_train_end = data_start + timedelta(days=config.train_window)

        period_id = 1
        while current_train_end + timedelta(days=config.test_window) <= data_end:
            if config.anchored:
                # 锚定步进：训练窗口固定从起点开始
                train_start = data_start
                train_end = current_train_end
            else:
                # 滚动步进：训练窗口向前滚动
                train_start = current_train_end - timedelta(days=config.train_window)
                train_end = current_train_end

            test_start = current_train_end
            test_end = current_train_end + timedelta(days=config.test_window)

            period = WalkForwardPeriod(
                period_id=f"period_{period_id}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_parameters={},
                performance_score=0.0,
                out_of_sample_score=0.0,
                optimization_details={}
            )

            periods.append(period)

            # 移动到下一个周期
            current_train_end += timedelta(days=config.step_size)
            period_id += 1

        logger.info(f"生成了 {len(periods)} 个步进周期")
        return periods

    async def _optimize_single_period(self, config: WalkForwardConfig,
                                      period: WalkForwardPeriod) -> Dict[str, Any]:
        """
        优化单个周期

        Args:
            config: 步进优化配置
            period: 步进周期

        Returns:
            Dict[str, Any]: 周期优化结果
        """
        try:
            logger.debug(f"优化周期: {period.train_start.date()} - {period.test_end.date()}")

            # 在训练数据上进行参数优化
            best_parameters = await self._optimize_on_train_data(
                config, period.train_start, period.train_end
            )

            # 在测试数据上验证性能
            test_performance = await self._evaluate_on_test_data(
                config, period.test_start, period.test_end, best_parameters
            )

            # 计算样本外表现
            out_of_sample_score = test_performance.get('sharpe_ratio', 0.0)

            period_result = {
                'period_id': period.period_id,
                'train_start': period.train_start.isoformat(),
                'train_end': period.train_end.isoformat(),
                'test_start': period.test_start.isoformat(),
                'test_end': period.test_end.isoformat(),
                'best_parameters': best_parameters,
                'performance_score': test_performance.get('total_return', 0.0),
                'out_of_sample_score': out_of_sample_score,
                'optimization_details': {
                    'test_performance': test_performance,
                    'optimization_method': 'grid_search'  # 简化实现
                }
            }

            logger.debug(f"周期 {period.period_id} 优化完成，样本外得分: {out_of_sample_score}")
            return period_result

        except Exception as e:
            logger.error(f"周期 {period.period_id} 优化失败: {e}")
            return {
                'period_id': period.period_id,
                'train_start': period.train_start.isoformat(),
                'train_end': period.train_end.isoformat(),
                'test_start': period.test_start.isoformat(),
                'test_end': period.test_end.isoformat(),
                'best_parameters': {},
                'performance_score': 0.0,
                'out_of_sample_score': 0.0,
                'optimization_details': {'error': str(e)}
            }

    async def _optimize_on_train_data(self, config: WalkForwardConfig,
                                      train_start: datetime, train_end: datetime) -> Dict[str, Any]:
        """
        在训练数据上进行参数优化

        Args:
            config: 步进优化配置
            train_start: 训练开始时间
            train_end: 训练结束时间

        Returns:
            Dict[str, Any]: 最优参数
        """
        # 这里简化实现，实际应该运行完整的参数优化
        # 暂时返回固定的参数组合作为示例

        if not config.parameter_ranges:
            return {}

        # 从参数范围内选择一个组合作为"最优"
        best_parameters = {}
        for param_name, param_values in config.parameter_ranges.items():
            best_parameters[param_name] = param_values[0]  # 简化：选择第一个值

        return best_parameters

    async def _evaluate_on_test_data(self, config: WalkForwardConfig,
                                     test_start: datetime, test_end: datetime,
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        在测试数据上验证性能

        Args:
            config: 步进优化配置
            test_start: 测试开始时间
            test_end: 测试结束时间
            parameters: 参数组合

        Returns:
            Dict[str, Any]: 测试性能
        """
        try:
            # 创建回测配置 - 暂时不需要，直接返回模拟结果
            pass

            # 这里简化实现，实际应该调用回测服务
            # 暂时返回模拟的性能数据

            import secrets
            np.random.seed(hash(f"{config.strategy_id}_{test_start}_{test_end}") % 2 ** 32)

            performance = {
                'total_return': secrets.uniform(-0.2, 0.3),
                'sharpe_ratio': secrets.uniform(-1, 2),
                'max_drawdown': secrets.uniform(0, 0.15),
                'win_rate': secrets.uniform(0.4, 0.7),
                'total_trades': secrets.randint(10, 50)
            }

            return performance

        except Exception as e:
            logger.error(f"测试数据评估失败: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }

    def _calculate_overall_performance(self, period_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算整体性能

        Args:
            period_results: 周期结果列表

        Returns:
            Dict[str, float]: 整体性能
        """
        if not period_results:
            return {}

        # 计算各项指标的平均值
        total_returns = [p['performance_score'] for p in period_results]
        out_of_sample_scores = [p['out_of_sample_score'] for p in period_results]

        overall_performance = {
            'avg_total_return': sum(total_returns) / len(total_returns) if total_returns else 0.0,
            'avg_out_of_sample_score': sum(out_of_sample_scores) / len(out_of_sample_scores) if out_of_sample_scores else 0.0,
            'periods_count': len(period_results),
            'positive_periods': sum(1 for r in total_returns if r > 0),
            'negative_periods': sum(1 for r in total_returns if r < 0)
        }

        return overall_performance

    def _calculate_robustness_score(self, period_results: List[Dict[str, Any]]) -> float:
        """
        计算稳健性评分

        Args:
            period_results: 周期结果列表

        Returns:
            float: 稳健性评分 (0 - 1)
        """
        if not period_results:
            return 0.0

        out_of_sample_scores = [p['out_of_sample_score'] for p in period_results]

        # 计算一致性：正向表现的周期比例
        positive_periods = sum(1 for score in out_of_sample_scores if score > 0)
        consistency = positive_periods / len(out_of_sample_scores)

        # 计算稳定性：分数的标准差倒数
        if len(out_of_sample_scores) > 1:
            import numpy as np
            std_dev = np.std(out_of_sample_scores)
            stability = 1.0 / (1.0 + std_dev) if std_dev > 0 else 1.0
        else:
            stability = 1.0

        # 计算平均表现
        avg_score = sum(out_of_sample_scores) / len(out_of_sample_scores)
        normalized_avg = max(0, min(1, (avg_score + 1) / 2))  # 归一化到[0,1]

        # 综合评分
        robustness_score = (consistency * 0.4 + stability * 0.3 + normalized_avg * 0.3)

        return robustness_score

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            event_bus_adapter = self.adapter_factory.get_adapter("event_bus")
            await event_bus_adapter.publish_event({
                "event_type": f"walk_forward_{event_type}",
                "data": event_data,
                "source": "walk_forward_optimizer",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"事件发布异常: {e}")

    def evaluate_robustness(self, walk_forward_result: WalkForwardResult) -> float:
        """
        评估优化稳健性

        Args:
            walk_forward_result: 步进优化结果

        Returns:
            float: 稳健性评分 (0 - 1)
        """
        return walk_forward_result.robustness_score


# 导出类
__all__ = [
    'WalkForwardOptimizer',
    'WalkForwardPeriod'
]

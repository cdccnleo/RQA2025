#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
执行算法模块
提供各种订单执行算法的实现
"""

import logging
import time
import secrets
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AlgorithmType(Enum):
    """算法类型枚举"""
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"    # 成交量百分比
    ICEBERG = "iceberg"  # 冰山订单
    MARKET = "market"  # 市价执行
    LIMIT = "limit"   # 限价执行


@dataclass
class AlgorithmConfig:
    """算法配置"""
    algo_type: AlgorithmType
    duration: int = 300  # 执行持续时间（秒）
    target_quantity: int = 1000  # 目标数量
    max_participation: float = 0.1  # 最大参与率
    randomize_timing: bool = True  # 是否随机化时机


@dataclass
class ExecutionSlice:
    """执行切片"""
    quantity: int
    price: Optional[float] = None
    timestamp: Optional[float] = None
    venue: str = "default"


class BaseExecutionAlgorithm:
    """基础执行算法类"""

    def __init__(self, config: AlgorithmConfig, metrics_collector=None):
        self.config = config
        self.metrics = metrics_collector
        self.logger = logging.getLogger(__name__)
        self.execution_slices: List[ExecutionSlice] = []

    def execute(self, orders: List[Dict[str, Any]], algo_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行订单

        Args:
            orders: 订单列表
            algo_params: 算法参数

        Returns:
            List[Dict[str, Any]]: 执行结果列表
        """
        if not orders:
            return []

        # 默认实现：处理第一个订单
        order = orders[0]
        slices = self._execute_single_order(order, algo_params)

        # 转换为ExecutionEngine期望的格式
        results = []
        for slice_obj in slices:
            result = {
                'quantity': slice_obj.quantity,
                'price': slice_obj.price,
                'timestamp': slice_obj.timestamp,
                'venue': slice_obj.venue,
                'status': 'completed'
            }
            results.append(result)

        return results

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """
        执行单个订单（子类重写此方法）

        Args:
            order: 订单信息
            algo_params: 算法参数

        Returns:
            List[ExecutionSlice]: 执行切片列表
        """
        raise NotImplementedError("子类必须实现_execute_single_order方法")

    def _record_slice(self, quantity: int, price: Optional[float] = None, venue: str = "default"):
        """记录执行切片"""
        slice_obj = ExecutionSlice(
            quantity=quantity,
            price=price,
            timestamp=time.time(),
            venue=venue
        )
        self.execution_slices.append(slice_obj)
        return slice_obj


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """时间加权平均价格算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行TWAP算法"""
        # 使用算法参数覆盖默认配置
        if algo_params:
            duration = algo_params.get('duration', self.config.duration)
            target_quantity = algo_params.get('target_quantity', order.get(
                'quantity', self.config.target_quantity))
        else:
            duration = self.config.duration
            target_quantity = order.get('quantity', self.config.target_quantity)

        slices_count = min(10, max(1, duration // 30))  # 每30秒一个切片

        slice_quantity = target_quantity // slices_count
        remaining_quantity = target_quantity % slices_count

        slices = []
        base_price = order.get('price', 100.0)

        for i in range(slices_count):
            # 添加一些价格随机性
            price_variation = secrets.uniform(-0.01, 0.01) if self.config.randomize_timing else 0
            price = base_price * (1 + price_variation)

            quantity = slice_quantity + (remaining_quantity if i == slices_count - 1 else 0)

            slice_obj = self._record_slice(quantity, price, "twap_exchange")
            slices.append(slice_obj)

            # 模拟执行间隔（在测试中跳过以加快速度）
            if i < slices_count - 1 and not hasattr(self, '_test_mode'):
                time.sleep(min(0.01, duration / slices_count))  # 限制最短间隔

        self.logger.info(f"TWAP执行完成，共{len(slices)}个切片，总量{target_quantity}")
        return slices


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """成交量加权平均价格算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行VWAP算法"""
        # 使用算法参数覆盖默认配置
        if algo_params:
            target_quantity = algo_params.get('target_quantity', order.get(
                'quantity', self.config.target_quantity))
        else:
            target_quantity = order.get('quantity', self.config.target_quantity)

        # 模拟基于成交量的执行
        volume_profile = [0.1, 0.15, 0.2, 0.25, 0.2, 0.1]  # 6个时间段的成交量分布

        slices = []
        base_price = order.get('price', 100.0)

        for i, volume_ratio in enumerate(volume_profile):
            quantity = int(target_quantity * volume_ratio)

            if quantity > 0:
                # 添加成交量相关的价格影响
                price_impact = secrets.uniform(-0.005, 0.005)
                price = base_price * (1 + price_impact)

                slice_obj = self._record_slice(quantity, price, "vwap_exchange")
                slices.append(slice_obj)

        self.logger.info(f"VWAP执行完成，共{len(slices)}个切片，总量{sum(s.quantity for s in slices)}")
        return slices


class POVAlgorithm(BaseExecutionAlgorithm):
    """成交量百分比算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行POV算法"""
        # 使用算法参数覆盖默认配置
        if algo_params:
            total_quantity = algo_params.get('target_quantity', order.get(
                'quantity', self.config.target_quantity))
            participation_rate = algo_params.get(
                'participation_rate', min(self.config.max_participation, 0.05))
        else:
            total_quantity = order.get('quantity', self.config.target_quantity)
            participation_rate = min(self.config.max_participation, 0.05)  # 最大5%参与率

        # 模拟市场成交量
        market_volume = secrets.randint(5000, 20000)
        target_volume = int(market_volume * participation_rate)

        slices = []
        base_price = order.get('price', 100.0)

        while sum(s.quantity for s in slices) < total_quantity:
            remaining_quantity = total_quantity - sum(s.quantity for s in slices)
            quantity = min(remaining_quantity, target_volume // 10)  # 分批执行

            if quantity <= 0:
                break

            # POV算法的价格通常接近市场价格
            price = base_price * secrets.uniform(0.995, 1.005)

            slice_obj = self._record_slice(quantity, price, "pov_exchange")
            slices.append(slice_obj)

        self.logger.info(f"POV执行完成，共{len(slices)}个切片，总量{sum(s.quantity for s in slices)}")
        return slices


class IcebergAlgorithm(BaseExecutionAlgorithm):
    """冰山订单算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行冰山算法"""
        # 使用算法参数覆盖默认配置
        if algo_params:
            total_quantity = algo_params.get('target_quantity', order.get(
                'quantity', self.config.target_quantity))
            display_quantity = algo_params.get('display_quantity', total_quantity // 10)  # 显示数量
        else:
            total_quantity = order.get('quantity', self.config.target_quantity)
            display_quantity = total_quantity // 10  # 显示数量

        slices = []
        base_price = order.get('price', 100.0)
        remaining_quantity = total_quantity

        while remaining_quantity > 0:
            # 每次只显示部分数量
            visible_quantity = min(display_quantity, remaining_quantity)
            quantity = visible_quantity

            # 冰山算法的价格通常稍低于市场价格以减少冲击
            price = base_price * secrets.uniform(0.99, 1.00)

            slice_obj = self._record_slice(quantity, price, "iceberg_exchange")
            slices.append(slice_obj)

            remaining_quantity -= quantity

            # 模拟执行间隔
            if remaining_quantity > 0 and not hasattr(self, '_test_mode'):
                time.sleep(0.1)  # 短暂延迟

        self.logger.info(f"冰山执行完成，共{len(slices)}个切片，总量{total_quantity}")
        return slices


class MarketAlgorithm(BaseExecutionAlgorithm):
    """市价执行算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行市价算法"""
        total_quantity = order.get('quantity', self.config.target_quantity)

        # 市价执行通常一次性完成
        base_price = order.get('price', 100.0)
        # 市价可能有滑点
        price = base_price * secrets.uniform(0.995, 1.01)

        slice_obj = self._record_slice(total_quantity, price, "market_exchange")

        self.logger.info(f"市价执行完成，数量{total_quantity}，价格{price}")
        return [slice_obj]


class LimitAlgorithm(BaseExecutionAlgorithm):
    """限价执行算法"""

    def _execute_single_order(self, order: Dict[str, Any], algo_params: Optional[Dict[str, Any]] = None) -> List[ExecutionSlice]:
        """执行限价算法"""
        total_quantity = order.get('quantity', self.config.target_quantity)
        limit_price = order.get('price')

        if limit_price is None:
            raise ValueError("限价算法需要指定价格")

        # 限价执行通常一次性完成，但可能部分成交
        executed_quantity = total_quantity  # 简化实现，假设全部成交
        price = limit_price

        slice_obj = self._record_slice(executed_quantity, price, "limit_exchange")

        self.logger.info(f"限价执行完成，数量{executed_quantity}，价格{price}")
        return [slice_obj]


def get_algorithm(algo_type: AlgorithmType, config=None, metrics=None) -> BaseExecutionAlgorithm:
    """
    获取执行算法实例

    Args:
        algo_type: 算法类型
        config: 配置对象
        metrics: 指标收集器

    Returns:
        BaseExecutionAlgorithm: 算法实例
    """
    if config is None:
        config = AlgorithmConfig(algo_type=algo_type)

    if algo_type == AlgorithmType.TWAP:
        return TWAPAlgorithm(config, metrics)
    elif algo_type == AlgorithmType.VWAP:
        return VWAPAlgorithm(config, metrics)
    elif algo_type == AlgorithmType.POV:
        return POVAlgorithm(config, metrics)
    elif algo_type == AlgorithmType.ICEBERG:
        return IcebergAlgorithm(config, metrics)
    elif algo_type == AlgorithmType.MARKET:
        return MarketAlgorithm(config, metrics)
    elif algo_type == AlgorithmType.LIMIT:
        return LimitAlgorithm(config, metrics)
    else:
        logger.warning(f"未知算法类型: {algo_type}，使用TWAP算法")
        return TWAPAlgorithm(config, metrics)


def create_execution_algorithm(algo_type: str) -> BaseExecutionAlgorithm:
    """创建执行算法实例"""
    try:
        algo_enum = AlgorithmType[algo_type.upper()]
        config = AlgorithmConfig(algo_type=algo_enum)
        return get_algorithm(algo_enum, config=config)
    except (KeyError, ValueError):
        # 默认使用TWAP算法
        config = AlgorithmConfig(algo_type=AlgorithmType.TWAP)
        return TWAPAlgorithm(config)

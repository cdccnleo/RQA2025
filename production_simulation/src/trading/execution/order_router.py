#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
订单路由器
负责将订单路由到最佳的执行地点
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """路由策略枚举"""
    BEST_PRICE = "best_price"
    FASTEST_EXECUTION = "fastest_execution"
    LOWEST_LATENCY = "lowest_latency"
    BALANCED = "balanced"


@dataclass
class RoutingResult:
    """路由结果"""
    destination: str
    strategy: RoutingStrategy
    estimated_latency: float
    estimated_cost: float
    confidence: float


class OrderRouter:
    """订单路由器"""

    def __init__(self, config=None, metrics=None):
        """
        初始化订单路由器

        Args:
            config: 配置管理器
            metrics: 指标收集器
        """
        self.config = config
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)

        # 默认路由配置
        self.routing_config = {
            'strategy': RoutingStrategy.BALANCED,
            'max_latency': 1000,  # 毫秒
            'max_cost': 0.01,     # 百分比
            'min_confidence': 0.8
        }

        # 路由目的地列表
        self.destinations = [
            'primary_exchange',
            'secondary_exchange',
            'dark_pool',
            'internal_book'
        ]

        # 目的地性能指标
        self.destination_metrics = {
            dest: {
                'latency': 50.0,  # 毫秒
                'cost': 0.005,    # 百分比
                'reliability': 0.95
            } for dest in self.destinations
        }

        self.logger.info("订单路由器初始化完成")

    def route_order(self, order: Dict[str, Any]) -> RoutingResult:
        """
        路由订单到最佳目的地

        Args:
            order: 订单信息字典

        Returns:
            RoutingResult: 路由结果
        """
        try:
            # 基于订单特征选择最佳路由
            best_destination = self._select_best_destination(order)
            strategy = self.routing_config['strategy']

            # 计算预估指标
            metrics = self.destination_metrics[best_destination]
            estimated_latency = metrics['latency']
            estimated_cost = metrics['cost']
            confidence = metrics['reliability']

            result = RoutingResult(
                destination=best_destination,
                strategy=strategy,
                estimated_latency=estimated_latency,
                estimated_cost=estimated_cost,
                confidence=confidence
            )

            self.logger.info(f"订单路由完成: {order.get('order_id', 'unknown')} -> {best_destination}")
            return result

        except Exception as e:
            self.logger.error(f"订单路由失败: {e}")
            # 返回默认路由
            return RoutingResult(
                destination=self.destinations[0],
                strategy=RoutingStrategy.BALANCED,
                estimated_latency=100.0,
                estimated_cost=0.01,
                confidence=0.5
            )

    def _select_best_destination(self, order: Dict[str, Any]) -> str:
        """
        选择最佳目的地

        Args:
            order: 订单信息

        Returns:
            str: 最佳目的地名称
        """
        # 简单的路由逻辑：基于订单大小和紧急程度选择
        order_size = order.get('quantity', 100)
        urgency = order.get('urgency', 'normal')

        if urgency == 'high':
            return 'primary_exchange'  # 高优先级使用主交易所
        elif order_size > 10000:
            return 'dark_pool'  # 大单使用暗池
        else:
            return 'secondary_exchange'  # 普通订单使用二级市场

    def update_destination_metrics(self, destination: str, metrics: Dict[str, float]):
        """
        更新目的地性能指标

        Args:
            destination: 目的地名称
            metrics: 新的性能指标
        """
        if destination in self.destination_metrics:
            self.destination_metrics[destination].update(metrics)
            self.logger.info(f"更新目的地指标: {destination}")

    def get_available_destinations(self) -> List[str]:
        """
        获取可用目的地列表

        Returns:
            List[str]: 目的地列表
        """
        return self.destinations.copy()

    def get_destination_metrics(self, destination: str) -> Optional[Dict[str, float]]:
        """
        获取目的地性能指标

        Args:
            destination: 目的地名称

        Returns:
            Optional[Dict[str, float]]: 性能指标
        """
        return self.destination_metrics.get(destination)

import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据分片管理器

实现数据分片功能：
- 基于哈希的分片
- 基于范围的分片
- 基于时间的分片
- 自定义分片策略
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):

    """数据分片策略枚举"""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    TIME_BASED = "time_based"
    CUSTOM = "custom"


@dataclass
class ShardInfo:

    """分片信息数据类"""
    shard_id: str
    data_source: str
    approach: ShardingStrategy
    parameters: Dict[str, Any]
    assigned_nodes: List[str]
    status: str
    created_at: datetime
    metadata: Dict[str, Any]


class DataShardingManager:

    """数据分片管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据分片管理器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 分片管理
        self.shards: Dict[str, ShardInfo] = {}

        # 分片策略映射
        self.sharding_strategies = {
            ShardingStrategy.HASH_BASED: self._hash_based_sharding,
            ShardingStrategy.RANGE_BASED: self._range_based_sharding,
            ShardingStrategy.TIME_BASED: self._time_based_sharding
        }

        logger.info("DataShardingManager initialized")

    async def create_shards(self, data_source: str, parameters: Dict[str, Any],
                            strategy: ShardingStrategy,
                            shard_parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        创建分片

        Args:
            data_source: 数据源标识
            parameters: 原始参数
            strategy: 分片策略
            shard_parameters: 分片参数

        Returns:
            List[Dict[str, Any]]: 分片列表
        """
        if strategy in self.sharding_strategies:
            return await self.sharding_strategies[strategy](data_source, parameters, shard_parameters)
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")

    async def _hash_based_sharding(self, data_source: str, parameters: Dict[str, Any],
                                   shard_parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """基于哈希的分片"""
        num_shards = shard_parameters.get('num_shards', 4) if shard_parameters else 4
        shard_key = shard_parameters.get('shard_key', 'id') if shard_parameters else 'id'

        shards = []
        for i in range(num_shards):
            shard_params = parameters.copy()
            shard_params['shard_id'] = i
            shard_params['total_shards'] = num_shards
            shard_params['shard_key'] = shard_key

            shards.append({
                'shard_id': f"{data_source}_hash_shard_{i}",
                'parameters': shard_params,
                "approach": ShardingStrategy.HASH_BASED.value,
                'metadata': {
                    'shard_index': i,
                    'total_shards': num_shards,
                    'shard_key': shard_key
                }
            })

        logger.info(f"Created {num_shards} hash-based shards for {data_source}")
        return shards

    async def _range_based_sharding(self, data_source: str, parameters: Dict[str, Any],
                                    shard_parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """基于范围的分片"""
        ranges = shard_parameters.get('ranges', []) if shard_parameters else []
        if not ranges:
            # 默认按时间范围分片
            ranges = [
                {'start': '2020-01-01', 'end': '2021-01-01'},
                {'start': '2021-01-01', 'end': '2022-01-01'},
                {'start': '2022-01-01', 'end': '2023-01-01'},
                {'start': '2023-01-01', 'end': '2024-01-01'}
            ]

        shards = []
        for i, range_info in enumerate(ranges):
            shard_params = parameters.copy()
            shard_params.update(range_info)
            shard_params['shard_id'] = i

            shards.append({
                'shard_id': f"{data_source}_range_shard_{i}",
                'parameters': shard_params,
                "approach": ShardingStrategy.RANGE_BASED.value,
                'metadata': {
                    'shard_index': i,
                    'range': range_info
                }
            })

        logger.info(f"Created {len(ranges)} range-based shards for {data_source}")
        return shards

    async def _time_based_sharding(self, data_source: str, parameters: Dict[str, Any],
                                   shard_parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """基于时间的分片"""
        time_period = shard_parameters.get(
            'time_period', 'monthly') if shard_parameters else 'monthly'
        start_date = shard_parameters.get(
            'start_date', '2020-01-01') if shard_parameters else '2020-01-01'
        end_date = shard_parameters.get(
            'end_date', '2024-01-01') if shard_parameters else '2024-01-01'

        # 生成时间分片
        shards = []
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        shard_id = 0

        while current_date < end_date:
            if time_period == 'monthly':
                next_date = current_date + pd.DateOffset(months=1)
            elif time_period == 'quarterly':
                next_date = current_date + pd.DateOffset(months=3)
            elif time_period == 'yearly':
                next_date = current_date + pd.DateOffset(years=1)
            else:
                next_date = current_date + pd.DateOffset(months=1)

            shard_params = parameters.copy()
            shard_params['start_date'] = current_date.strftime('%Y-%m-%d')
            shard_params['end_date'] = next_date.strftime('%Y-%m-%d')
            shard_params['shard_id'] = shard_id

            shards.append({
                'shard_id': f"{data_source}_time_shard_{shard_id}",
                'parameters': shard_params,
                "approach": ShardingStrategy.TIME_BASED.value,
                'metadata': {
                    'shard_index': shard_id,
                    'time_period': time_period,
                    'start_date': current_date.strftime('%Y-%m-%d'),
                    'end_date': next_date.strftime('%Y-%m-%d')
                }
            })

            current_date = next_date
            shard_id += 1

        logger.info(f"Created {len(shards)} time-based shards for {data_source}")
        return shards

    def get_shard_by_key(self, key: str, num_shards: int) -> int:
        """根据键获取分片索引"""
        # 使用哈希函数计算分片索引
        hash_value = hashlib.md5(key.encode()).hexdigest()
        hash_int = int(hash_value, 16)
        return hash_int % num_shards

    def get_shard_info(self, shard_id: str) -> Optional[ShardInfo]:
        """获取分片信息"""
        return self.shards.get(shard_id)

    def register_shard(self, shard_info: ShardInfo):
        """注册分片"""
        self.shards[shard_info.shard_id] = shard_info
        logger.info(f"Registered shard {shard_info.shard_id}")

    def unregister_shard(self, shard_id: str):
        """注销分片"""
        if shard_id in self.shards:
            del self.shards[shard_id]
            logger.info(f"Unregistered shard {shard_id}")

    def get_all_shards(self) -> List[ShardInfo]:
        """获取所有分片"""
        return list(self.shards.values())

    def approach(self, strategy: ShardingStrategy) -> List[ShardInfo]:
        """根据策略获取分片"""
        return [shard for shard in self.shards.values() if shard.approach == strategy]

    def get_shards_by_data_source(self, data_source: str) -> List[ShardInfo]:
        """根据数据源获取分片"""
        return [shard for shard in self.shards.values() if shard.data_source == data_source]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import zlib
from typing import Dict, List, Optional, Union
import redis
from redis.cluster import RedisCluster
from ..core import StorageAdapter

class RedisAdapter(StorageAdapter):
    """Redis存储适配器"""

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: str = None,
                 cluster_mode: bool = False,
                 compress_threshold: int = 1024):
        """
        初始化Redis适配器

        Args:
            host: Redis主机地址
            port: Redis端口
            db: 数据库编号
            password: 认证密码
            cluster_mode: 是否集群模式
            compress_threshold: 压缩阈值(字节)
        """
        self.compress_threshold = compress_threshold
        self.cluster_mode = cluster_mode

        if cluster_mode:
            self.client = RedisCluster(
                host=host,
                port=port,
                password=password,
                decode_responses=True
            )
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )

    def _compress(self, data: str) -> bytes:
        """压缩数据"""
        return zlib.compress(data.encode('utf-8'))

    def _decompress(self, data: bytes) -> str:
        """解压数据"""
        return zlib.decompress(data).decode('utf-8')

    def write(self, path: str, data: Dict) -> bool:
        """写入数据到Redis"""
        try:
            serialized = json.dumps(data)

            # 根据大小决定是否压缩
            if len(serialized) > self.compress_threshold:
                compressed = self._compress(serialized)
                return bool(self.client.set(path, compressed))
            return bool(self.client.set(path, serialized))
        except (redis.RedisError, json.JSONEncodeError):
            return False

    def read(self, path: str) -> Optional[Dict]:
        """从Redis读取数据"""
        try:
            data = self.client.get(path)
            if data is None:
                return None

            # 自动检测是否为压缩数据
            try:
                if isinstance(data, bytes):
                    decompressed = self._decompress(data)
                    return json.loads(decompressed)
                return json.loads(data)
            except (zlib.error, UnicodeDecodeError):
                # 如果不是压缩数据，直接解析
                return json.loads(data)
        except (redis.RedisError, json.JSONDecodeError):
            return None

    def bulk_delete(self, keys: List[str]) -> int:
        """
        批量删除键
        Args:
            keys: 要删除的键列表
        Returns:
            int: 成功删除的数量
        """
        if not keys:
            return 0

        try:
            if self.cluster_mode:
                # 集群模式下需要逐个删除
                return sum(1 for k in keys if self.client.delete(k))
            else:
                # 单机模式使用批量删除
                return self.client.delete(*keys)
        except redis.RedisError:
            return 0

    def pipeline(self):
        """获取流水线操作对象"""
        return self.client.pipeline()

class RedisClusterAdapter(RedisAdapter):
    """Redis集群适配器"""
    
    def __init__(self, *args, **kwargs):
        # 强制启用集群模式
        kwargs['cluster_mode'] = True
        super().__init__(*args, **kwargs)
        self._init_cluster_metrics()
    
    def _init_cluster_metrics(self):
        """初始化集群监控指标"""
        self.cluster_metrics = {
            'node_count': 0,
            'slots_coverage': 0.0,
            'failover_count': 0,
            'node_health': {}
        }
    
    def get_cluster_info(self) -> Dict:
        """获取集群信息"""
        try:
            info = self.client.cluster_info()
            nodes = self.client.cluster_nodes()
            
            self.cluster_metrics['node_count'] = len(nodes)
            self.cluster_metrics['slots_coverage'] = info.get('cluster_slots_assigned', 0) / 16384
            
            return {
                'info': info,
                'nodes': nodes,
                'metrics': self.cluster_metrics
            }
        except redis.RedisError:
            return {'error': 'Failed to get cluster info'}


# 导出所有适配器类
__all__ = [
    'RedisAdapter',
    'RedisClusterAdapter', 
    'AShareRedisAdapter'
]


class AShareRedisAdapter(RedisAdapter):
    """A股专用Redis适配器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_keyspace()
        self.metrics = {
            'write_count': 0,
            'read_count': 0,
            'compress_count': 0,
            'delete_count': 0,
            'pipeline_ops': 0
        }

    def _init_keyspace(self):
        """初始化A股专用键空间"""
        if not self.client.exists('ashare:metadata'):
            self.client.hset('ashare:metadata', 'initialized', 'true')

    def save_quote(self, symbol: str, data: Dict) -> bool:
        """
        存储A股行情数据(带市场标记)

        Args:
            symbol: 股票代码(6位数字)
            data: 行情数据字典

        Returns:
            bool: 是否存储成功
        """
        if not symbol.isdigit() or len(symbol) != 6:
            raise ValueError("Invalid A-share symbol")

        # 添加市场标记
        market = 'SH' if symbol.startswith(('6', '3')) else 'SZ'
        data['market'] = market

        # 使用HSET存储多字段
        key = f"ashare:quotes:{symbol}"
        mapping = {
            'price': str(data.get('price', '')),
            'volume': str(data.get('volume', '')),
            'time': data.get('time', ''),
            'status': data.get('limit_status', '')
        }

        try:
            # 主数据存储
            result = self.client.hset(key, mapping=mapping)
            # 更新时间索引
            self.client.zadd(
                'ashare:timestamps',
                {symbol: int(data.get('timestamp', 0))}
            )
            self.metrics['write_count'] += 1
            return bool(result)
        except redis.RedisError:
            return False

    def bulk_save(self, quotes: Dict[str, Dict]) -> bool:
        """批量存储行情数据"""
        pipe = self.client.pipeline()

        for symbol, data in quotes.items():
            key = f"ashare:quotes:{symbol}"
            mapping = {
                'price': str(data.get('price', '')),
                'volume': str(data.get('volume', '')),
                'time': data.get('time', ''),
                'status': data.get('limit_status', '')
            }
            pipe.hset(key, mapping=mapping)
            pipe.zadd(
                'ashare:timestamps',
                {symbol: int(data.get('timestamp', 0))}
            )
            self.metrics['pipeline_ops'] += 2

        try:
            pipe.execute()
            self.metrics['write_count'] += len(quotes)
            return True
        except redis.RedisError:
            return False

    def bulk_delete_quotes(self, symbols: List[str]) -> int:
        """
        批量删除A股行情数据
        Args:
            symbols: 股票代码列表
        Returns:
            int: 成功删除的数量
        """
        if not symbols:
            return 0

        keys = [f"ashare:quotes:{sym}" for sym in symbols]
        try:
            # 删除行情数据
            deleted = self.bulk_delete(keys)
            # 删除时间索引
            self.client.zrem('ashare:timestamps', *symbols)
            self.metrics['delete_count'] += deleted
            return deleted
        except redis.RedisError:
            return 0

    def get_metrics(self) -> Dict:
        """获取监控指标"""
        return {
            'write_count': self.metrics['write_count'],
            'read_count': self.metrics['read_count'],
            'compress_count': self.metrics['compress_count'],
            'delete_count': self.metrics['delete_count'],
            'pipeline_ops': self.metrics['pipeline_ops'],
            'keys_count': len(self.client.keys('ashare:*')),
            'cluster_mode': self.cluster_mode
        }

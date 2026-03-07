"""
distributed_cache_manager 模块

提供 distributed_cache_manager 相关功能和接口。
"""

import json
import logging
import redis

# from ..redis_cache import RedisCache  # TODO: RedisCache not found
import hashlib
import threading
import time

from ..exceptions import CacheConnectionError
from ..interfaces import ConsistencyLevel
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
#!/usr/bin/env python3
"""
分布式缓存管理器 - 实现Redis集群支持和分布式一致性
"""

logger = logging.getLogger(__name__)


class SyncStrategy(Enum):
    """同步策略"""
    WRITE_THROUGH = "write_through"    # 写穿
    WRITE_BEHIND = "write_behind"      # 写回
    WRITE_AHEAD = "write_ahead"        # 写前


class SyncMode(Enum):
    """同步模式"""
    AUTO = "auto"              # 自动同步
    MANUAL = "manual"          # 手动同步
    HYBRID = "hybrid"          # 混合同步
    BATCH = "batch"            # 批量同步
    REAL_TIME = "real_time"    # 实时同步


@dataclass
class ClusterNode:
    """集群节点配置"""
    host: str
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    weight: int = 1  # 节点权重，用于负载均衡


@dataclass
class DistributedConfig:
    """分布式缓存配置"""
    nodes: List[ClusterNode]
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    sync_strategy: str = "write_through"  # 暂时使用字符串
    replication_factor: int = 2
    heartbeat_interval: int = 30
    failover_timeout: int = 10
    max_sync_retry: int = 3
    enable_monitoring: bool = True


class VectorClockManager:
    """向量时钟管理器"""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.vector_clocks: Dict[str, Dict[str, int]] = {}
        self.lock = threading.RLock()

    def update_vector_clock(self, key: str, node_id: str, timestamp: int):
        """更新向量时钟"""
        with self.lock:
            if key not in self.vector_clocks:
                self.vector_clocks[key] = {}
            self.vector_clocks[key][node_id] = max(
                self.vector_clocks[key].get(node_id, 0), timestamp
            )

    def get_vector_clock(self, key: str) -> Dict[str, int]:
        """获取向量时钟"""
        with self.lock:
            return self.vector_clocks.get(key, {}).copy()

    def is_consistent(self, key: str, node_clocks: Dict[str, Dict[str, int]]) -> bool:
        """检查一致性"""
        if self.config.consistency_level == ConsistencyLevel.STRONG:
            return self._check_strong_consistency(key, node_clocks)
        elif self.config.consistency_level == ConsistencyLevel.EVENTUAL:
            return self._check_eventual_consistency(key, node_clocks)
        else:
            return True  # 弱一致性不检查

    def _check_strong_consistency(self, key: str, node_clocks: Dict[str, Dict[str, int]]) -> bool:
        """检查强一致性"""
        # 所有节点的向量时钟必须相同
        if not node_clocks:
            return True

        first_clock = None
        for node_id, clock in node_clocks.items():
            if first_clock is None:
                first_clock = clock
            elif clock != first_clock:
                return False
        return True

    def _check_eventual_consistency(self, key: str, node_clocks: Dict[str, Dict[str, int]]) -> bool:
        """检查最终一致性"""
        # 检查是否有明显的数据冲突
        if len(node_clocks) < 2:
            return True

        # 简单的冲突检测：检查是否有相同key的不同值
        values = set()
        for node_id, clock in node_clocks.items():
            value_hash = hashlib.sha256(str(clock).encode()).hexdigest()[:8]
            values.add(value_hash)

        return len(values) <= 1  # 如果值不冲突，认为是一致的


class ClusterManager:
    """Redis集群管理器"""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.nodes: Dict[str, Any] = {}
        self.active_nodes: Set[str] = set()
        self.node_load: Dict[str, int] = {}
        self.consistency_manager = VectorClockManager(config)
        self.sync_strategy = config.sync_strategy  # 添加同步策略属性
        self._initialize_cluster()

    def _initialize_cluster(self):
        """初始化集群"""
        for i, node in enumerate(self.config.nodes):
            node_id = f"node_{i}"
            try:
                # 创建Redis连接
                redis_client = redis.Redis(
                    host=node.host,
                    port=node.port,
                    db=node.db,
                    password=node.password,
                    decode_responses=True
                )
                self.nodes[node_id] = redis_client
                self.active_nodes.add(node_id)
                self.node_load[node_id] = 0
                logger.info(f"Redis节点 {node_id} 初始化成功: {node.host}:{node.port}")
            except Exception as e:
                logger.error(f"Redis节点 {node_id} 初始化失败: {e}")

        if not self.active_nodes:
            raise CacheConnectionError("无法连接到任何Redis节点")

    def get_node(self, key: str) -> Tuple[str, Any]:
        """根据key选择节点（一致性哈希）"""
        if not self.active_nodes:
            raise CacheConnectionError("没有可用的Redis节点")

        # 简单的一致性哈希实现
        hash_value = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        sorted_nodes = sorted(self.active_nodes)

        # 考虑节点权重
        weighted_nodes = []
        for node_id in sorted_nodes:
            # 获取节点索引
            node_index = int(node_id.split('_')[1])
            if node_index < len(self.config.nodes):
                weight = self.config.nodes[node_index].weight
                weighted_nodes.extend([node_id] * weight)

        if weighted_nodes:
            selected_node = weighted_nodes[hash_value % len(weighted_nodes)]
            return selected_node, self.nodes[selected_node]

        # 降级到简单轮询
        node_id = sorted_nodes[hash_value % len(sorted_nodes)]
        return node_id, self.nodes[node_id]

    def get_replica_nodes(self, key: str, exclude_node: Optional[str] = None) -> List[Tuple[str, Any]]:
        """获取副本节点"""
        replicas = []
        for node_id, cache in self.nodes.items():
            if node_id != exclude_node and node_id in self.active_nodes:
                replicas.append((node_id, cache))
            if len(replicas) >= self.config.replication_factor - 1:
                break
        return replicas

    def sync_data(self, key: str, value: Any, primary_node: str, ttl: Optional[int] = None):
        """同步数据到副本节点"""
        if self.config.sync_strategy == "write_through":
            self._sync_write_through(key, value, primary_node, ttl)
        elif self.config.sync_strategy == "write_behind":
            self._sync_write_behind(key, value, primary_node, ttl)
        else:
            logger.warning(f"不支持的同步策略: {self.config.sync_strategy}")

    def _sync_write_through(self, key: str, value: Any, primary_node: str, ttl: Optional[int]):
        """写穿同步"""
        replica_nodes = self.get_replica_nodes(key, primary_node)

        if not replica_nodes:
            return

        # 并行同步到副本节点
        with ThreadPoolExecutor(max_workers=len(replica_nodes)) as executor:
            futures = []
            for node_id, cache in replica_nodes:
                future = executor.submit(self._sync_to_node, cache, key, value, ttl)
                futures.append((node_id, future))

            # 等待同步完成
            for node_id, future in futures:
                try:
                    future.result(timeout=self.config.failover_timeout)
                    logger.debug(f"数据同步成功: {key} -> {node_id}")
                except Exception as e:
                    logger.error(f"数据同步失败: {key} -> {node_id}: {e}")

    def _sync_write_behind(self, key: str, value: Any, primary_node: str, ttl: Optional[int]):
        """写回同步（异步）"""
        replica_nodes = self.get_replica_nodes(key, primary_node)

        if not replica_nodes:
            return

        # 异步同步，不阻塞主操作
        def async_sync():
            for node_id, cache in replica_nodes:
                try:
                    self._sync_to_node(cache, key, value, ttl)
                    logger.debug(f"异步数据同步成功: {key} -> {node_id}")
                except Exception as e:
                    logger.error(f"异步数据同步失败: {key} -> {node_id}: {e}")

        threading.Thread(target=async_sync, daemon=True).start()

    def _sync_to_node(self, cache: Any, key: str, value: Any, ttl: Optional[int]):
        """同步数据到指定节点"""
        serialized_value = json.dumps(value, default=str) if not isinstance(value, str) else value
        # 使用setex方法设置带过期时间的键值对
        cache.setex(key, ttl or 3600, serialized_value)

    def check_node_health(self, node_id: str) -> bool:
        """检查节点健康状态"""
        if node_id not in self.nodes:
            return False

        try:
            cache = self.nodes[node_id]
            # 发送ping命令检查连接
            return cache.ping()
        except Exception:
            return False

    def handle_node_failure(self, failed_node: str):
        """处理节点故障"""
        if failed_node in self.active_nodes:
            self.active_nodes.remove(failed_node)
            logger.warning(f"节点故障，已从活跃节点列表移除: {failed_node}")

        # 检查是否需要重新分配数据
        if len(self.active_nodes) < len(self.nodes) * 0.5:
            logger.error("可用节点少于50%，集群处于降级模式")

    def handle_node_recovery(self, recovered_node: str):
        """处理节点恢复"""
        if recovered_node in self.nodes and recovered_node not in self.active_nodes:
            if self.check_node_health(recovered_node):
                self.active_nodes.add(recovered_node)
                logger.info(f"节点恢复，已添加到活跃节点列表: {recovered_node}")


class DistributedCacheManager:
    """分布式缓存管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 兼容两种配置类型
        if config is None:
            config = DistributedConfig(nodes=[ClusterNode(host="localhost", port=6379)])
        if hasattr(config, 'nodes'):
            # 使用分布式缓存管理器的配置
            self.config = config
        else:
            # 转换核心缓存配置为分布式配置
            self.config = self._convert_config(config)

        self.cluster_manager = ClusterManager(self.config)
        self.local_cache = {}  # 本地缓存，用于加速访问
        self.creation_times = {}  # 添加创建时间字典
        self.ttl = 3600  # 默认TTL
        self._start_monitoring()

    def _convert_config(self, core_config: Dict[str, Any]):
        """将核心缓存配置转换为分布式配置"""
        # 创建一个模拟的节点列表
        host = getattr(core_config, "redis_host", "localhost")
        port = getattr(core_config, "redis_port", 6379)
        nodes = [ClusterNode(host=host, port=port)]

        # 创建分布式配置
        return DistributedConfig(
            nodes=nodes,
            consistency_level=ConsistencyLevel.EVENTUAL,
            sync_strategy="write_through",
            replication_factor=1,
            heartbeat_interval=30,
            failover_timeout=10,
            max_sync_retry=3,
            enable_monitoring=True
        )

    def _start_monitoring(self):
        """启动监控"""
        if self.config.enable_monitoring:
            def monitor_nodes():
                while True:
                    try:
                        for node_id in list(self.cluster_manager.nodes.keys()):
                            is_healthy = self.cluster_manager.check_node_health(node_id)

                            if node_id in self.cluster_manager.active_nodes and not is_healthy:
                                self.cluster_manager.handle_node_failure(node_id)
                            elif node_id not in self.cluster_manager.active_nodes and is_healthy:
                                self.cluster_manager.handle_node_recovery(node_id)

                        time.sleep(self.config.heartbeat_interval)
                    except Exception as e:
                        logger.error(f"节点监控异常: {e}")
                        time.sleep(self.config.heartbeat_interval)

            monitoring_thread = threading.Thread(target=monitor_nodes, daemon=True)
            monitoring_thread.start()
            logger.info("分布式缓存监控已启动")

    def get(self, key: str) -> Any:
        """分布式获取缓存"""
        # 添加键前缀
        prefixed_key = f"rqa2025_cache:{key}"

        # 先检查本地缓存
        if prefixed_key in self.local_cache:
            local_value = self.local_cache[prefixed_key]
            if not self._is_expired(prefixed_key):
                self._update_access_stats(prefixed_key)
                return local_value

        try:
            # 从分布式缓存获取
            primary_node, cache = self.cluster_manager.get_node(prefixed_key)
            value = cache.get(prefixed_key)

            if value is not None:
                # 反序列化
                try:
                    deserialized_value = json.loads(value) if isinstance(value, str) else value
                except (json.JSONDecodeError, TypeError):
                    # 如果反序列化失败，返回None而不是原始值
                    logger.warning(f"反序列化失败: {value}")
                    return None

                # 更新本地缓存
                self.local_cache[prefixed_key] = deserialized_value
                self.creation_times[prefixed_key] = time.time()

                # 检查一致性
                if self.config.consistency_level != ConsistencyLevel.WEAK:
                    self._check_consistency(prefixed_key, deserialized_value)

                return deserialized_value

        except Exception as e:
            logger.error(f"分布式缓存获取失败: {prefixed_key}: {e}")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """分布式设置缓存"""
        # 添加键前缀
        prefixed_key = f"rqa2025_cache:{key}"

        try:
            # 序列化值
            serialized_value = json.dumps(
                value, default=str) if not isinstance(value, str) else value

            # 获取主节点
            primary_node, cache = self.cluster_manager.get_node(prefixed_key)

            # 设置到主节点
            success = cache.setex(prefixed_key, ttl or 3600, serialized_value)
            if success:
                # 更新本地缓存
                self.local_cache[prefixed_key] = value
                self.creation_times[prefixed_key] = time.time()

                # 同步到副本节点
                if self.config.replication_factor > 1:
                    self.cluster_manager.sync_data(prefixed_key, value, primary_node, ttl)

                # 更新向量时钟
                if self.config.consistency_level != ConsistencyLevel.WEAK:
                    self.cluster_manager.consistency_manager.update_vector_clock(
                        prefixed_key, primary_node, int(time.time() * 1000000)
                    )

                logger.debug(f"分布式缓存设置成功: {prefixed_key}")
                return True

        except Exception as e:
            logger.error(f"分布式缓存设置失败: {prefixed_key}: {e}")

        return False

    def delete(self, key: str) -> bool:
        """分布式删除缓存"""
        # 添加键前缀
        prefixed_key = f"rqa2025_cache:{key}"

        try:
            # 获取主节点
            primary_node, cache = self.cluster_manager.get_node(prefixed_key)

            # 从主节点删除
            success = cache.delete(prefixed_key)
            if success:
                # 从本地缓存删除
                if prefixed_key in self.local_cache:
                    del self.local_cache[prefixed_key]
                if prefixed_key in self.creation_times:
                    del self.creation_times[prefixed_key]

                # 从副本节点删除
                replica_nodes = self.cluster_manager.get_replica_nodes(prefixed_key, primary_node)
                for node_id, replica_cache in replica_nodes:
                    try:
                        replica_cache.delete(prefixed_key)
                    except Exception as e:
                        logger.error(f"副本节点删除失败: {prefixed_key} -> {node_id}: {e}")

                logger.debug(f"分布式缓存删除成功: {prefixed_key}")
                return True

        except Exception as e:
            logger.error(f"分布式缓存删除失败: {prefixed_key}: {e}")

        return False

    def _check_consistency(self, key: str, value: Any):
        """检查数据一致性"""
        try:
            replica_nodes = self.cluster_manager.get_replica_nodes(key)
            node_clocks = {}

            # 从各个副本节点获取数据和时钟
            for node_id, cache in replica_nodes:
                try:
                    replica_value = cache.get(key)
                    if replica_value is not None:
                        # 获取节点的向量时钟（简化实现）
                        node_clocks[node_id] = {node_id: int(time.time() * 1000000)}
                except Exception as e:
                    logger.debug(f"获取副本数据失败: {key} -> {node_id}: {e}")

            # 检查一致性
            if not self.cluster_manager.consistency_manager.is_consistent(key, node_clocks):
                logger.warning(f"检测到数据不一致: {key}")
                # 可以在这里实现冲突解决逻辑

        except Exception as e:
            logger.error(f"一致性检查失败: {key}: {e}")

    def get_cluster_stats(self) -> Dict[str, Any]:
        """获取集群统计信息"""
        return {
            "total_nodes": len(self.cluster_manager.nodes),
            "active_nodes": len(self.cluster_manager.active_nodes),
            "consistency_level": self.config.consistency_level.value,
            "sync_strategy": self.config.sync_strategy,
            "replication_factor": self.config.replication_factor,
            "local_cache_size": len(self.local_cache)
        }

    def get_node_stats(self, node_id: str) -> Dict[str, Any]:
        """获取指定节点的统计信息"""
        if node_id not in self.cluster_manager.nodes:
            return {}

        return {
            "node_id": node_id,
            "is_active": node_id in self.cluster_manager.active_nodes,
            "load": self.cluster_manager.node_load.get(node_id, 0),
            "last_health_check": time.time()
        }

    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self.creation_times:
            return False  # 如果没有创建时间，认为未过期

        creation_time = self.creation_times[key]
        current_time = time.time()
        return current_time - creation_time > self.ttl

    def _update_access_stats(self, key: str):
        """更新访问统计"""
        # 这里可以添加访问统计逻辑

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        # 添加键前缀
        prefixed_key = f"rqa2025_cache:{key}"

        try:
            # 获取主节点
            primary_node, cache = self.cluster_manager.get_node(prefixed_key)
            return cache.exists(prefixed_key) > 0
        except Exception as e:
            logger.error(f"检查键存在性失败: {prefixed_key}: {e}")
            return False

    def clear(self) -> bool:
        """清空所有缓存"""
        try:
            # 清空所有节点
            for node_id, cache in self.cluster_manager.nodes.items():
                try:
                    # 获取所有匹配前缀的键并删除
                    keys = cache.keys("rqa2025_cache:*")
                    if keys:
                        cache.delete(*keys)
                except Exception as e:
                    logger.error(f"清空节点缓存失败: {node_id}: {e}")

            # 清空本地缓存
            self.local_cache.clear()
            self.creation_times.clear()

            logger.debug("分布式缓存清空成功")
            return True

        except Exception as e:
            logger.error(f"分布式缓存清空失败: {e}")
            return False

    def close(self):
        """关闭分布式缓存管理器"""
        # 关闭所有Redis连接
        for node_id, cache in self.cluster_manager.nodes.items():
            try:
                cache.close()
            except Exception as e:
                logger.error(f"关闭Redis连接失败: {node_id}: {e}")

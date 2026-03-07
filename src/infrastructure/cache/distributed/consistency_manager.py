"""
consistency_manager 模块

提供 consistency_manager 相关功能和接口。
"""

import json
import logging

import hashlib
import threading
import time

from ..interfaces import ConsistencyLevel
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
#!/usr/bin/env python3
"""
分布式缓存一致性保证机制 - P1级别改进
提供强一致性、最终一致性和会话一致性等多种一致性保证
"""

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    LAST_WRITE_WINS = "last_write_wins"  # 最后写入获胜
    FIRST_WRITE_WINS = "first_write_wins"  # 首次写入获胜
    MANUAL = "manual"                    # 手动解决
    CUSTOM = "custom"                    # 自定义策略
    MERGE = "merge"                      # 合并策略


@dataclass
class ConsistencyConfig:
    """一致性配置"""
    level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
    sync_timeout: float = 5.0            # 同步超时时间
    max_retries: int = 3                 # 最大重试次数
    read_quorum: int = 1                 # 读仲裁数
    write_quorum: int = 1                # 写仲裁数
    enable_version_vector: bool = True    # 启用版本向量
    enable_read_repair: bool = True       # 启用读修复
    anti_entropy_interval: float = 60.0  # 反熵间隔


@dataclass
class VersionInfo:
    """版本信息"""
    version: int
    timestamp: float
    node_id: str
    checksum: str = ""


@dataclass
class DataEntry:
    """数据条目"""
    key: str
    value: Any
    version_info: VersionInfo
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyMetrics:
    """一致性指标"""
    total_operations: int = 0
    consistent_reads: int = 0
    inconsistent_reads: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    sync_operations: int = 0
    failed_sync_operations: int = 0
    repair_operations: int = 0


class VectorClock:
    """向量时钟实现"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = {node_id: 0}

    def increment(self):
        """递增本地时钟"""
        self.clock[self.node_id] += 1

    def update(self, other_clock: Dict[str, int]):
        """更新时钟向量"""
        for node_id, timestamp in other_clock.items():
            if node_id in self.clock:
                self.clock[node_id] = max(self.clock[node_id], timestamp)
            else:
                self.clock[node_id] = timestamp
        self.increment()  # 增加本地时钟

    def compare(self, other_clock: Dict[str, int]) -> str:
        """比较时钟向量"""
        # 返回 "before", "after", "concurrent" 或 "equal"
        before = after = True

        all_nodes = set(self.clock.keys()) | set(other_clock.keys())

        # 如果没有公共节点，则认为并发
        if not all_nodes:
            return "equal"

        for node_id in all_nodes:
            self_ts = self.clock.get(node_id, 0)
            other_ts = other_clock.get(node_id, 0)

            if self_ts > other_ts:
                before = False
            elif self_ts < other_ts:
                after = False

        if before and after:
            return "equal"
        elif before:
            return "before"
        elif after:
            return "after"
        else:
            return "concurrent"

    def get_clock(self) -> Dict[str, int]:
        """获取时钟向量"""
        return self.clock.copy()


class ConsistencyManager:
    """一致性管理器 - P1级别完善实现"""

    def __init__(self, node_id: str, config: ConsistencyConfig):
        self.node_id = node_id
        self.config = config
        self.vector_clock = VectorClock(node_id)
        self.metrics = ConsistencyMetrics()

        # 缓存节点管理
        self.cache_nodes: Dict[str, Any] = {}
        self.node_status: Dict[str, bool] = {}  # 节点状态

        # 一致性状态跟踪
        self.pending_writes: Dict[str, DataEntry] = {}  # 待写入数据
        self.version_vectors: Dict[str, Dict[str, int]] = {}  # 版本向量存储
        self.conflict_queue: List[Dict[str, Any]] = []  # 冲突队列

        # 会话一致性支持
        self.session_reads: Dict[str, Dict[str, VersionInfo]] = defaultdict(dict)

        # 线程安全
        self.lock = threading.RLock()
        self.anti_entropy_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 自定义冲突解决函数
        self.custom_conflict_resolver: Optional[Callable] = None

        logger.info(f"一致性管理器已初始化: 节点={node_id}, 级别={config.level.value}")

    def register_cache_node(self, node_id: str, cache_manager: Any):
        """注册缓存节点"""
        with self.lock:
            self.cache_nodes[node_id] = cache_manager
            self.node_status[node_id] = True
            logger.info(f"注册缓存节点: {node_id}")

    def unregister_cache_node(self, node_id: str):
        """注销缓存节点"""
        with self.lock:
            if node_id in self.cache_nodes:
                del self.cache_nodes[node_id]
                del self.node_status[node_id]
                logger.info(f"注销缓存节点: {node_id}")

    def start_consistency_manager(self):
        """启动一致性管理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动反熵进程
        if self.config.anti_entropy_interval > 0:
            self.anti_entropy_thread = threading.Thread(
                target=self._anti_entropy_process,
                daemon=True
            )
            self.anti_entropy_thread.start()

        logger.info("一致性管理器已启动")

    def stop_consistency_manager(self):
        """停止一致性管理器"""
        self.is_running = False

        if self.anti_entropy_thread:
            self.anti_entropy_thread.join(timeout=5)

        logger.info("一致性管理器已停止")

    def consistent_read(self, key: str, session_id: Optional[str] = None) -> Optional[Any]:
        """一致性读取"""
        with self.lock:
            self.metrics.total_operations += 1

        if self.config.level == ConsistencyLevel.STRONG:
            return self._strong_consistent_read(key)
        elif self.config.level == ConsistencyLevel.SESSION and session_id:
            return self._session_consistent_read(key, session_id)
        elif self.config.level == ConsistencyLevel.READ_YOUR_WRITES and session_id:
            return self._read_your_writes_read(key, session_id)
        else:  # EVENTUAL, MONOTONIC_READ
            return self._eventual_consistent_read(key)

    def consistent_write(self, key: str, value: Any,
                         session_id: Optional[str] = None,
                         ttl: Optional[int] = None) -> bool:
        """一致性写入"""
        with self.lock:
            self.metrics.total_operations += 1

        # 创建版本信息
        self.vector_clock.increment()
        version_info = VersionInfo(
            version=self.vector_clock.clock[self.node_id],
            timestamp=time.time(),
            node_id=self.node_id,
            checksum=self._calculate_checksum(value)
        )

        # 创建数据条目
        data_entry = DataEntry(
            key=key,
            value=value,
            version_info=version_info,
            ttl=ttl,
            metadata={'session_id': session_id} if session_id else {}
        )

        if self.config.level == ConsistencyLevel.STRONG:
            return self._strong_consistent_write(data_entry)
        elif self.config.level == ConsistencyLevel.SESSION and session_id:
            return self._session_consistent_write(data_entry, session_id)
        else:  # EVENTUAL, MONOTONIC_WRITE
            return self._eventual_consistent_write(data_entry)

    def _strong_consistent_read(self, key: str) -> Optional[Any]:
        """强一致性读取"""
        available_nodes = [node_id for node_id, status in self.node_status.items() if status]

        if len(available_nodes) < self.config.read_quorum:
            logger.warning(f"可用节点不足以满足读仲裁要求: {len(available_nodes)}/{self.config.read_quorum}")
            return None

        # 从多个节点读取
        read_results = {}
        with ThreadPoolExecutor(max_workers=len(available_nodes)) as executor:
            future_to_node = {
                executor.submit(self._read_from_node, node_id, key): node_id
                for node_id in available_nodes[:self.config.read_quorum]
            }

        for future in as_completed(future_to_node):
            node_id = future_to_node[future]
            try:
                result = future.result(timeout=self.config.sync_timeout)
                if result is not None:
                    read_results[node_id] = result
            except Exception as e:
                logger.error(f"从节点{node_id}读取失败: {e}")
                self.node_status[node_id] = False

        if not read_results:
            return None

        # 检查一致性
        if self._check_read_consistency(read_results):
            self.metrics.consistent_reads += 1
            # 返回最新版本的数据
            latest_entry = max(read_results.values(),
                               key=lambda x: x.version_info.timestamp)
            return latest_entry.value
        else:
            self.metrics.inconsistent_reads += 1
            # 触发读修复
            if self.config.enable_read_repair:
                self._perform_read_repair(key, read_results)

        # 返回最新版本数据
        latest_entry = max(read_results.values(),
                           key=lambda x: x.version_info.timestamp)
        return latest_entry.value

    def _session_consistent_read(self, key: str, session_id: str) -> Optional[Any]:
        """会话一致性读取"""
        # 检查会话中是否有该键的读取历史
        if key in self.session_reads[session_id]:
            required_version = self.session_reads[session_id][key]

            # 从能提供足够新版本的节点读取
            for node_id, cache_manager in self.cache_nodes.items():
                if not self.node_status.get(node_id, False):
                    continue

                try:
                    entry = self._read_from_node(node_id, key)
                    if (entry and
                            entry.version_info.timestamp >= required_version.timestamp):
                        return entry.value
                except Exception as e:
                    logger.error(f"会话读取失败: {node_id}: {e}")

        # 如果没有会话历史，进行普通读取
        return self._eventual_consistent_read(key)

    def _read_your_writes_read(self, key: str, session_id: str) -> Optional[Any]:
        """读己之写一致性读取"""
        # 检查是否有该会话的写入
        if key in self.session_reads[session_id]:
            required_version = self.session_reads[session_id][key]

            # 确保读取到至少是自己写入的版本
            for node_id, cache_manager in self.cache_nodes.items():
                if not self.node_status.get(node_id, False):
                    continue

                try:
                    entry = self._read_from_node(node_id, key)
                    if (entry and
                            entry.version_info.timestamp >= required_version.timestamp):
                        return entry.value
                except Exception as e:
                    logger.error(f"读己之写读取失败: {node_id}: {e}")

        return self._eventual_consistent_read(key)

    def _eventual_consistent_read(self, key: str) -> Optional[Any]:
        """最终一致性读取"""
        # 从任意可用节点读取
        for node_id, cache_manager in self.cache_nodes.items():
            if not self.node_status.get(node_id, False):
                continue

            try:
                entry = self._read_from_node(node_id, key)
                if entry:
                    return entry.value
            except Exception as e:
                logger.error(f"最终一致性读取失败: {node_id}: {e}")
                self.node_status[node_id] = False

        return None

    def _strong_consistent_write(self, data_entry: DataEntry) -> bool:
        """强一致性写入"""
        available_nodes = [node_id for node_id, status in self.node_status.items() if status]

        if len(available_nodes) < self.config.write_quorum:
            logger.warning(f"可用节点不足以满足写仲裁要求: {len(available_nodes)}/{self.config.write_quorum}")
            return False

        # 并发写入到多个节点
        successful_writes = 0
        with ThreadPoolExecutor(max_workers=len(available_nodes)) as executor:
            future_to_node = {
                executor.submit(self._write_to_node, node_id, data_entry): node_id
                for node_id in available_nodes
            }

        for future in as_completed(future_to_node):
            node_id = future_to_node[future]
            try:
                success = future.result(timeout=self.config.sync_timeout)
                if success:
                    successful_writes += 1
            except Exception as e:
                logger.error(f"写入节点{node_id}失败: {e}")
                self.node_status[node_id] = False

        # 检查是否满足写仲裁要求
        if successful_writes >= self.config.write_quorum:
            # 更新会话读取记录
            if data_entry.metadata.get('session_id'):
                session_id = data_entry.metadata['session_id']
                self.session_reads[session_id][data_entry.key] = data_entry.version_info

            return True
        else:
            logger.error(f"写入失败，成功节点数不足: {successful_writes}/{self.config.write_quorum}")
            return False

    def _session_consistent_write(self, data_entry: DataEntry, session_id: str) -> bool:
        """会话一致性写入"""
        # 会话一致性写入通常写入到多个节点以保证持久性
        result = self._eventual_consistent_write(data_entry)

        if result:
            # 更新会话读取记录
            self.session_reads[session_id][data_entry.key] = data_entry.version_info

        return result

    def _eventual_consistent_write(self, data_entry: DataEntry) -> bool:
        """最终一致性写入"""
        # 异步写入到所有可用节点
        successful_writes = 0

        for node_id, cache_manager in self.cache_nodes.items():
            if not self.node_status.get(node_id, False):
                continue

            try:
                success = self._write_to_node(node_id, data_entry)
                if success:
                    successful_writes += 1
            except Exception as e:
                logger.error(f"最终一致性写入失败: {node_id}: {e}")
                self.node_status[node_id] = False

        # 只要有一个节点写入成功就认为成功
        return successful_writes > 0

    def _read_from_node(self, node_id: str, key: str) -> Optional[DataEntry]:
        """从指定节点读取数据"""
        cache_manager = self.cache_nodes.get(node_id)
        if not cache_manager:
            return None

        try:
            # 这里假设缓存管理器存储的是DataEntry对象
            # 在实际实现中，可能需要序列化/反序列化
            raw_value = cache_manager.get(key)
            if raw_value is None:
                return None

            # 如果存储的是原始值，需要包装成DataEntry
            if not isinstance(raw_value, DataEntry):
                # 创建一个简单的DataEntry
                version_info = VersionInfo(
                    version=1,
                    timestamp=time.time(),
                    node_id=node_id,
                    checksum=self._calculate_checksum(raw_value)
                )
                return DataEntry(key=key, value=raw_value, version_info=version_info)

            return raw_value

        except Exception as e:
            logger.error(f"从节点{node_id}读取失败: {e}")
            raise

    def _write_to_node(self, node_id: str, data_entry: DataEntry) -> bool:
        """写入数据到指定节点"""
        cache_manager = self.cache_nodes.get(node_id)
        if not cache_manager:
            return False

        try:
            # 在实际实现中，可能需要序列化DataEntry
            return cache_manager.set(data_entry.key, data_entry, data_entry.ttl)

        except Exception as e:
            logger.error(f"写入节点{node_id}失败: {e}")
            raise

    def _check_read_consistency(self, read_results: Dict[str, DataEntry]) -> bool:
        """检查读取结果的一致性"""
        if len(read_results) <= 1:
            return True

        # 比较版本信息
        versions = [entry.version_info for entry in read_results.values()]
        checksums = [version.checksum for version in versions]

        # 如果所有校验和相同，则认为一致
        return len(set(checksums)) == 1

    def _perform_read_repair(self, key: str, read_results: Dict[str, DataEntry]):
        """执行读修复"""
        logger.info(f"执行读修复: {key}")
        self.metrics.repair_operations += 1

        # 找到最新版本
        latest_entry = max(read_results.values(),
                           key=lambda x: x.version_info.timestamp)

        # 更新过时的节点
        for node_id, entry in read_results.items():
            if entry.version_info.timestamp < latest_entry.version_info.timestamp:
                try:
                    self._write_to_node(node_id, latest_entry)
                    logger.info(f"读修复更新节点: {node_id}")
                except Exception as e:
                    logger.error(f"读修复失败: {node_id}: {e}")

    def _anti_entropy_process(self):
        """反熵进程"""
        logger.info("启动反熵进程")

        while self.is_running:
            try:
                self._perform_anti_entropy()
                time.sleep(self.config.anti_entropy_interval)
            except Exception as e:
                logger.error(f"反熵进程异常: {e}")
                time.sleep(10)  # 错误后等待

    def _perform_anti_entropy(self):
        """执行反熵操作"""
        logger.debug("执行反熵同步...")

        # 获取所有可用节点
        available_nodes = [node_id for node_id, status in self.node_status.items() if status]

        if len(available_nodes) < 2:
            return

        # 简化的反熵：比较节点间的数据
        for i, node1_id in enumerate(available_nodes):
            for node2_id in available_nodes[i+1:]:
                try:
                    self._sync_between_nodes(node1_id, node2_id)
                except Exception as e:
                    logger.error(f"节点间同步失败 {node1_id}<->{node2_id}: {e}")

    def _sync_between_nodes(self, node1_id: str, node2_id: str):
        """在两个节点间同步数据"""
        # 这里实现简化的同步逻辑
        # 实际实现中需要更复杂的算法，如Merkle树比较

        node1_cache = self.cache_nodes.get(node1_id)
        node2_cache = self.cache_nodes.get(node2_id)

        if not node1_cache or not node2_cache:
            return

        # 获取节点的键列表（如果支持）
        if hasattr(node1_cache, 'keys') and hasattr(node2_cache, 'keys'):
            keys1 = set(node1_cache.keys())
            keys2 = set(node2_cache.keys())

            # 找到不一致的键
            all_keys = keys1 | keys2
            for key in list(all_keys)[:10]:  # 限制每次同步的键数量
                try:
                    self._sync_key_between_nodes(key, node1_id, node2_id)
                except Exception as e:
                    logger.debug(f"同步键{key}失败: {e}")

    def _sync_key_between_nodes(self, key: str, node1_id: str, node2_id: str):
        """在两个节点间同步特定键"""
        entry1 = self._read_from_node(node1_id, key)
        entry2 = self._read_from_node(node2_id, key)

        # 处理不同情况
        if entry1 and entry2:
            # 两个节点都有数据，比较版本
            if entry1.version_info.timestamp > entry2.version_info.timestamp:
                self._write_to_node(node2_id, entry1)
            elif entry2.version_info.timestamp > entry1.version_info.timestamp:
                self._write_to_node(node1_id, entry2)
            # 版本相同则不需要同步
        elif entry1 and not entry2:
            # 只有node1有数据
            self._write_to_node(node2_id, entry1)
        elif entry2 and not entry1:
            # 只有node2有数据
            self._write_to_node(node1_id, entry2)

    def _calculate_checksum(self, value: Any) -> str:
        """计算数据校验和"""
        try:
            data_str = json.dumps(value, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(value).encode()).hexdigest()

    def get_consistency_metrics(self) -> Dict[str, Any]:
        """获取一致性指标"""
        with self.lock:
            consistency_ratio = 0.0
            if self.metrics.total_operations > 0:
                consistency_ratio = (self.metrics.consistent_reads /
                                     max(1, self.metrics.consistent_reads + self.metrics.inconsistent_reads))

        return {
            'total_operations': self.metrics.total_operations,
            'consistent_reads': self.metrics.consistent_reads,
            'inconsistent_reads': self.metrics.inconsistent_reads,
            'consistency_ratio': consistency_ratio,
            'conflicts_detected': self.metrics.conflicts_detected,
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'sync_operations': self.metrics.sync_operations,
            'failed_sync_operations': self.metrics.failed_sync_operations,
            'repair_operations': self.metrics.repair_operations,
            'active_nodes': len([status for status in self.node_status.values() if status]),
            'total_nodes': len(self.node_status)
        }

    def set_custom_conflict_resolver(self, resolver: Callable):
        """设置自定义冲突解决函数"""
        self.custom_conflict_resolver = resolver
        logger.info("已设置自定义冲突解决函数")

    def cleanup(self):
        """清理资源"""
        self.stop_consistency_manager()

        with self.lock:
            self.cache_nodes.clear()
            self.node_status.clear()
            self.pending_writes.clear()
            self.version_vectors.clear()
            self.conflict_queue.clear()
        self.session_reads.clear()

        logger.info("一致性管理器已清理")

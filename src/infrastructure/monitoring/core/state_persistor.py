#!/usr/bin/env python3
"""
RQA2025 基础设施层状态持久化器

负责组件状态的持久化存储、恢复和管理。
这是从ComponentRegistry中拆分出来的状态管理组件。
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
import threading
from datetime import datetime
import gzip
import hashlib

logger = logging.getLogger(__name__)


class ComponentState:
    """组件状态"""

    def __init__(self, component_name: str, state_data: Dict[str, Any]):
        """
        初始化组件状态

        Args:
            component_name: 组件名称
            state_data: 状态数据
        """
        self.component_name = component_name
        self.state_data = state_data
        self.timestamp = datetime.now()
        self.version = state_data.get('version', '1.0.0')
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """计算状态数据的校验和"""
        data_str = json.dumps(self.state_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'component_name': self.component_name,
            'state_data': self.state_data,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentState':
        """从字典创建状态对象"""
        state = cls(data['component_name'], data['state_data'])
        state.timestamp = datetime.fromisoformat(data['timestamp'])
        state.version = data.get('version', '1.0.0')
        return state

    def validate(self) -> bool:
        """验证状态完整性"""
        return self.checksum == self._calculate_checksum()


class StatePersistor:
    """
    状态持久化器

    负责组件状态的持久化存储、恢复和管理，支持压缩、校验和验证等高级功能。
    """

    def __init__(self, storage_dir: str = "data/component_states",
                 enable_compression: bool = True,
                 max_backup_files: int = 10):
        """
        初始化状态持久化器

        Args:
            storage_dir: 存储目录
            enable_compression: 是否启用压缩
            max_backup_files: 最大备份文件数量
        """
        self.storage_dir = storage_dir
        self.enable_compression = enable_compression
        self.max_backup_files = max_backup_files
        self._lock = threading.RLock()

        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)

        # 内存缓存
        self._state_cache: Dict[str, ComponentState] = {}
        self._cache_dirty = False

        logger.info(f"状态持久化器初始化完成，存储目录: {storage_dir}")

    def save_component_state(self, component_name: str, state_data: Dict[str, Any]) -> bool:
        """
        保存组件状态

        Args:
            component_name: 组件名称
            state_data: 状态数据

        Returns:
            bool: 是否保存成功
        """
        with self._lock:
            try:
                # 创建状态对象
                state = ComponentState(component_name, state_data)

                # 更新缓存
                self._state_cache[component_name] = state
                self._cache_dirty = True

                # 持久化存储
                return self._persist_state(state)

            except Exception as e:
                logger.error(f"保存组件状态失败 {component_name}: {e}")
                return False

    def load_component_state(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        加载组件状态

        Args:
            component_name: 组件名称

        Returns:
            Optional[Dict[str, Any]]: 状态数据，如果不存在则返回None
        """
        with self._lock:
            try:
                # 先检查缓存
                if component_name in self._state_cache:
                    cached_state = self._state_cache[component_name]
                    if cached_state.validate():
                        logger.debug(f"从缓存加载组件状态: {component_name}")
                        return cached_state.state_data.copy()

                # 从文件加载
                state = self._load_state_from_file(component_name)
                if state and state.validate():
                    # 更新缓存
                    self._state_cache[component_name] = state
                    logger.debug(f"从文件加载组件状态: {component_name}")
                    return state.state_data.copy()
                else:
                    logger.warning(f"组件状态校验失败或不存在: {component_name}")
                    return None

            except Exception as e:
                logger.error(f"加载组件状态失败 {component_name}: {e}")
                return None

    def delete_component_state(self, component_name: str) -> bool:
        """
        删除组件状态

        Args:
            component_name: 组件名称

        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            try:
                # 从缓存删除
                self._state_cache.pop(component_name, None)
                self._cache_dirty = True

                # 从文件删除
                file_path = self._get_state_file_path(component_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"删除组件状态文件: {component_name}")
                    return True
                else:
                    logger.debug(f"组件状态文件不存在: {component_name}")
                    return True

            except Exception as e:
                logger.error(f"删除组件状态失败 {component_name}: {e}")
                return False

    def list_component_states(self) -> List[str]:
        """
        列出所有保存的组件状态

        Returns:
            List[str]: 组件名称列表
        """
        try:
            components = []
            if os.path.exists(self.storage_dir):
                for filename in os.listdir(self.storage_dir):
                    if filename.endswith('.json') or filename.endswith('.json.gz'):
                        component_name = filename.rsplit('.', 2)[0]  # 移除扩展名
                        components.append(component_name)
            return components
        except Exception as e:
            logger.error(f"列出组件状态失败: {e}")
            return []

    def backup_all_states(self, backup_name: Optional[str] = None) -> bool:
        """
        备份所有组件状态

        Args:
            backup_name: 备份名称，如果不指定则使用时间戳

        Returns:
            bool: 是否备份成功
        """
        with self._lock:
            try:
                if backup_name is None:
                    backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")

                backup_dir = os.path.join(self.storage_dir, "backups", backup_name)
                os.makedirs(backup_dir, exist_ok=True)

                # 复制所有状态文件
                components = self.list_component_states()
                for component in components:
                    src_path = self._get_state_file_path(component)
                    if os.path.exists(src_path):
                        dst_path = os.path.join(backup_dir, os.path.basename(src_path))
                        with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                            dst.write(src.read())

                # 清理旧备份
                self._cleanup_old_backups()

                logger.info(f"组件状态备份完成: {backup_name}")
                return True

            except Exception as e:
                logger.error(f"备份组件状态失败: {e}")
                return False

    def restore_from_backup(self, backup_name: str) -> bool:
        """
        从备份恢复组件状态

        Args:
            backup_name: 备份名称

        Returns:
            bool: 是否恢复成功
        """
        with self._lock:
            try:
                backup_dir = os.path.join(self.storage_dir, "backups", backup_name)
                if not os.path.exists(backup_dir):
                    logger.error(f"备份不存在: {backup_name}")
                    return False

                # 恢复所有文件
                for filename in os.listdir(backup_dir):
                    src_path = os.path.join(backup_dir, filename)
                    dst_path = os.path.join(self.storage_dir, filename)

                    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                        dst.write(src.read())

                # 清空缓存，强制重新加载
                self._state_cache.clear()
                self._cache_dirty = False

                logger.info(f"从备份恢复组件状态: {backup_name}")
                return True

            except Exception as e:
                logger.error(f"从备份恢复失败: {e}")
                return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            Dict[str, Any]: 存储统计
        """
        try:
            total_size = 0
            file_count = 0
            components = self.list_component_states()

            for component in components:
                file_path = self._get_state_file_path(component)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            # 检查备份
            backup_dir = os.path.join(self.storage_dir, "backups")
            backup_count = 0
            if os.path.exists(backup_dir):
                for root, dirs, files in os.walk(backup_dir):
                    backup_count += len([f for f in files if f.endswith(('.json', '.json.gz'))])

            return {
                'total_components': len(components),
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'backup_count': backup_count,
                'compression_enabled': self.enable_compression,
                'cache_size': len(self._state_cache)
            }

        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {
                'error': str(e)
            }

    def optimize_storage(self) -> Dict[str, Any]:
        """
        优化存储空间

        Returns:
            Dict[str, Any]: 优化结果
        """
        results = {
            'files_cleaned': 0,
            'space_saved': 0,
            'backups_cleaned': 0
        }

        try:
            # 清理无效文件
            components = self.list_component_states()
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(('.json', '.json.gz')) and not filename.startswith('backup'):
                    component_name = filename.rsplit('.', 2)[0]
                    if component_name not in components:
                        file_path = os.path.join(self.storage_dir, filename)
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        results['files_cleaned'] += 1
                        results['space_saved'] += size

            # 清理旧备份
            results['backups_cleaned'] = self._cleanup_old_backups()

            # 整理缓存
            if self._cache_dirty:
                self._state_cache.clear()
                self._cache_dirty = False

            logger.info(f"存储优化完成: {results}")
            return results

        except Exception as e:
            logger.error(f"存储优化失败: {e}")
            return {
                'error': str(e)
            }

    def _persist_state(self, state: ComponentState) -> bool:
        """
        持久化状态到文件

        Args:
            state: 状态对象

        Returns:
            bool: 是否成功
        """
        try:
            file_path = self._get_state_file_path(state.component_name)
            state_dict = state.to_dict()

            # 写入文件
            if self.enable_compression:
                with gzip.open(file_path + '.gz', 'wt', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)

            logger.debug(f"状态持久化成功: {state.component_name}")
            return True

        except Exception as e:
            logger.error(f"状态持久化失败 {state.component_name}: {e}")
            return False

    def _load_state_from_file(self, component_name: str) -> Optional[ComponentState]:
        """
        从文件加载状态

        Args:
            component_name: 组件名称

        Returns:
            Optional[ComponentState]: 状态对象
        """
        try:
            file_path = self._get_state_file_path(component_name)

            # 尝试压缩文件
            if os.path.exists(file_path + '.gz'):
                with gzip.open(file_path + '.gz', 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            elif os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return None

            return ComponentState.from_dict(data)

        except Exception as e:
            logger.error(f"从文件加载状态失败 {component_name}: {e}")
            return None

    def _get_state_file_path(self, component_name: str) -> str:
        """
        获取状态文件路径

        Args:
            component_name: 组件名称

        Returns:
            str: 文件路径
        """
        safe_name = "".join(c for c in component_name if c.isalnum() or c in ('_', '-')).rstrip()
        return os.path.join(self.storage_dir, f"{safe_name}.json")

    def _cleanup_old_backups(self) -> int:
        """
        清理旧备份文件

        Returns:
            int: 清理的备份数量
        """
        try:
            backup_dir = os.path.join(self.storage_dir, "backups")
            if not os.path.exists(backup_dir):
                return 0

            # 获取所有备份目录
            backups = []
            for item in os.listdir(backup_dir):
                item_path = os.path.join(backup_dir, item)
                if os.path.isdir(item_path):
                    backups.append((item_path, os.path.getctime(item_path)))

            # 按创建时间排序
            backups.sort(key=lambda x: x[1], reverse=True)

            # 删除多余的备份
            cleaned = 0
            for backup_path, _ in backups[self.max_backup_files:]:
                import shutil
                shutil.rmtree(backup_path)
                cleaned += 1

            return cleaned

        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")
            return 0

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取状态持久化器的健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            stats = self.get_storage_stats()

            issues = []

            # 检查存储空间
            if stats.get('total_size_mb', 0) > 100:  # 超过100MB
                issues.append(f"存储空间过大: {stats['total_size_mb']}MB")

            # 检查文件完整性
            corrupted_files = []
            for component in self.list_component_states():
                state = self._load_state_from_file(component)
                if state and not state.validate():
                    corrupted_files.append(component)

            if corrupted_files:
                issues.append(f"发现损坏的文件: {corrupted_files}")

            # 检查备份状态
            if stats.get('backup_count', 0) == 0:
                issues.append("没有备份文件")

            return {
                'status': 'healthy' if not issues else 'warning',
                'storage_stats': stats,
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局状态持久化器实例
global_state_persistor = StatePersistor()

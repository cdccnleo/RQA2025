#!/usr/bin/env python3
"""
RQA2025 基础设施层数据持久化器

负责监控数据的持久化存储和管理。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from ..core.parameter_objects import DataPersistenceConfig


class DataPersistor:
    """
    数据持久化器

    负责监控数据的存储、检索和管理，支持文件和数据库存储。
    """

    def __init__(
        self,
        pool_name: str = "default_pool",
        config: Optional[DataPersistenceConfig] = None,
    ):
        """
        初始化数据持久化器

        Args:
            pool_name: 池名称
            config: 持久化配置
        """
        self.pool_name = pool_name
        self.config = config or DataPersistenceConfig()

        # 创建存储目录
        self.storage_path = Path(self.config.storage_path) / pool_name
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 数据文件路径
        self.data_file = self.storage_path / "monitoring_data.json"
        self.metadata_file = self.storage_path / "metadata.json"

        # 内存缓存
        self._data_cache: List[Dict[str, Any]] = []
        self._metadata_cache: Dict[str, Any] = {}

        # 初始化
        self._load_metadata()
        self._cleanup_old_data()

    def persist_data(self, data: Dict[str, Any]) -> bool:
        """
        持久化数据

        Args:
            data: 要持久化的数据

        Returns:
            bool: 是否成功持久化
        """
        try:
            # 添加时间戳和元数据
            data_entry = {
                'timestamp': datetime.now().isoformat(),
                'pool_name': self.pool_name,
                'data': data
            }

            # 添加到缓存
            self._data_cache.append(data_entry)

            # 限制缓存大小
            if len(self._data_cache) > 100:  # 内存中最多保持100条记录
                self._flush_to_disk()

            # 检查是否需要定期刷新
            self._check_periodic_flush()

            return True

        except Exception as e:
            print(f"数据持久化失败: {e}")
            return False

    def retrieve_data(self, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """
        检索数据

        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回记录数

        Returns:
            List[Dict[str, Any]]: 检索到的数据列表
        """
        try:
            # 从磁盘加载数据
            all_data = self._load_all_data()

            # 过滤时间范围
            if start_time or end_time:
                filtered_data = []
                for entry in all_data:
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if start_time and entry_time < start_time:
                        continue
                    if end_time and entry_time > end_time:
                        continue
                    filtered_data.append(entry)
                all_data = filtered_data

            # 排序和限制数量
            all_data.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_data[:limit]

        except Exception as e:
            print(f"数据检索失败: {e}")
            return []

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            Dict[str, Any]: 数据统计信息
        """
        try:
            all_data = self._load_all_data()

            if not all_data:
                return {
                    'total_records': 0,
                    'date_range': None,
                    'storage_size_mb': 0,
                    'last_updated': None
                }

            # 计算统计信息
            timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in all_data]
            min_time = min(timestamps)
            max_time = max(timestamps)

            # 计算存储大小
            storage_size = self._calculate_storage_size()

            return {
                'total_records': len(all_data),
                'date_range': {
                    'start': min_time.isoformat(),
                    'end': max_time.isoformat()
                },
                'storage_size_mb': storage_size,
                'last_updated': max_time.isoformat(),
                'pool_name': self.pool_name
            }

        except Exception as e:
            print(f"获取数据统计失败: {e}")
            return {'error': str(e)}

    def cleanup_old_data(self, days_to_keep: Optional[int] = None) -> int:
        """
        清理旧数据

        Args:
            days_to_keep: 保留天数，默认使用配置值

        Returns:
            int: 删除的记录数量
        """
        try:
            if days_to_keep is None:
                days_to_keep = self.config.max_file_age_days

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # 重新加载所有数据
            all_data = self._load_all_data()
            original_count = len(all_data)

            # 过滤保留的数据
            kept_data = []
            for entry in all_data:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_date:
                    kept_data.append(entry)

            # 保存过滤后的数据
            self._save_all_data(kept_data)

            deleted_count = original_count - len(kept_data)
            print(f"清理了 {deleted_count} 条旧记录")
            return deleted_count

        except Exception as e:
            print(f"清理旧数据失败: {e}")
            return 0

    def export_data(self, file_path: str, format_type: str = 'json') -> bool:
        """
        导出数据

        Args:
            file_path: 导出文件路径
            format_type: 导出格式

        Returns:
            bool: 是否成功导出
        """
        try:
            all_data = self._load_all_data()

            if format_type == 'json':
                export_data = {
                    'metadata': {
                        'pool_name': self.pool_name,
                        'export_time': datetime.now().isoformat(),
                        'total_records': len(all_data)
                    },
                    'data': all_data
                }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            elif format_type == 'csv':
                # CSV格式导出（简化实现）
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if all_data:
                        fieldnames = all_data[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_data)

            else:
                print(f"不支持的导出格式: {format_type}")
                return False

            print(f"数据已导出到: {file_path}")
            return True

        except Exception as e:
            print(f"数据导出失败: {e}")
            return False

    def _flush_to_disk(self):
        """将缓存数据刷新到磁盘"""
        try:
            if not self._data_cache:
                return

            # 加载现有数据
            existing_data = self._load_all_data()

            # 合并数据
            existing_data.extend(self._data_cache)

            # 保存所有数据
            self._save_all_data(existing_data)

            # 清空缓存
            self._data_cache.clear()

        except Exception as e:
            print(f"刷新数据到磁盘失败: {e}")

    def _check_periodic_flush(self):
        """检查是否需要定期刷新"""
        # 简化实现：可以根据配置添加更复杂的刷新逻辑
        pass

    def _load_all_data(self) -> List[Dict[str, Any]]:
        """
        加载所有数据

        Returns:
            List[Dict[str, Any]]: 所有数据
        """
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            return []
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []

    def _save_all_data(self, data: List[Dict[str, Any]]):
        """
        保存所有数据

        Args:
            data: 要保存的数据
        """
        try:
            # 按时间排序
            data.sort(key=lambda x: x['timestamp'])

            # 限制数据大小（如果配置了）
            if self.config.retention_policy == 'size_based':
                max_size_mb = self.config.max_storage_size_mb
                # 简化的实现：可以根据实际大小进行限制
                pass

            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # 更新元数据
            self._metadata_cache['last_updated'] = datetime.now().isoformat()
            self._metadata_cache['total_records'] = len(data)
            self._save_metadata()

        except Exception as e:
            print(f"保存数据失败: {e}")

    def _load_metadata(self):
        """加载元数据"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self._metadata_cache = json.load(f)
        except Exception as e:
            self._metadata_cache = {}

    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._metadata_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存元数据失败: {e}")

    def _calculate_storage_size(self) -> float:
        """
        计算存储大小

        Returns:
            float: 存储大小（MB）
        """
        try:
            total_size = 0
            for file_path in self.storage_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # 转换为MB
        except Exception:
            return 0.0

    def _cleanup_old_data(self):
        """清理初始化时的旧数据"""
        try:
            # 在初始化时执行一次清理
            if self.config.max_file_age_days > 0:
                self.cleanup_old_data()
        except Exception as e:
            print(f"初始化数据清理失败: {e}")

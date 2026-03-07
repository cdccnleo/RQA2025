from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class PartitionStrategy(Enum):

    """分区策略枚举"""
    DATE = "date"
    HASH = "hash"
    CUSTOM = "custom"
    RANGE = "range"


@dataclass
class PartitionConfig:

    """分区配置"""
    approach: PartitionStrategy = PartitionStrategy.DATE
    partition_key: Optional[str] = None
    num_partitions: int = 100
    date_format: str = "%Y-%m-%d"
    range_bins: Optional[List[float]] = None


class PartitionManager:

    """
    分区管理器
    支持多种分区策略：日期分区、哈希分区、自定义分区、范围分区
    """

    def __init__(self, config: PartitionConfig = None):

        self.config = config or PartitionConfig()

    def get_partition_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取分区信息"""
        if not self.config.partition_key or self.config.partition_key not in data.columns:
            return {}

        partition_info = {}

        strategy = self._normalize_strategy(self.config.approach)

        if strategy == PartitionStrategy.DATE:
            partition_info = self._get_date_partition(data)
        elif strategy == PartitionStrategy.HASH:
            partition_info = self._get_hash_partition(data)
        elif strategy == PartitionStrategy.CUSTOM:
            partition_info = self._get_custom_partition(data)
        elif strategy == PartitionStrategy.RANGE:
            partition_info = self._get_range_partition(data)

        return partition_info

    def _normalize_strategy(self, approach) -> PartitionStrategy:
        """将配置的分区策略规范化为 PartitionStrategy 枚举"""
        if isinstance(approach, PartitionStrategy):
            return approach
        if isinstance(approach, str):
            try:
                return PartitionStrategy(approach.lower())
            except ValueError:
                pass
        return PartitionStrategy.DATE

    def _get_date_partition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """日期分区"""
        partition_info = {}

        if self.config.partition_key in data.columns:
            # 获取第一个日期值作为分区
            first_date = data[self.config.partition_key].iloc[0]
            if pd.notna(first_date):
                if isinstance(first_date, str):
                    # 如果是字符串，尝试解析
                    try:
                        parsed_date = pd.to_datetime(first_date)
                        partition_info['date'] = parsed_date.strftime(self.config.date_format)
                    except BaseException:
                        partition_info['date'] = str(first_date)
                else:
                    # 如果是datetime类型
                    partition_info['date'] = first_date.strftime(self.config.date_format)

        return partition_info

    def _get_hash_partition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """哈希分区"""
        partition_info = {}

        if self.config.partition_key in data.columns:
            # 使用第一个值计算哈希
            first_value = data[self.config.partition_key].iloc[0]
            if pd.notna(first_value):
                hash_value = hash(str(first_value)) % self.config.num_partitions
                partition_info['hash'] = f"part_{hash_value:03d}"

        return partition_info

    def _get_custom_partition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """自定义分区"""
        partition_info = {}

        if self.config.partition_key in data.columns:
            first_value = data[self.config.partition_key].iloc[0]
            if pd.notna(first_value):
                partition_info['custom'] = str(first_value)

        return partition_info

    def _get_range_partition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """范围分区"""
        partition_info = {}

        if self.config.partition_key in data.columns and self.config.range_bins:
            first_value = data[self.config.partition_key].iloc[0]
            if pd.notna(first_value):
                # 找到值所属的范围
                for i, bin_value in enumerate(self.config.range_bins):
                    if first_value <= bin_value:
                        partition_info['range'] = f"bin_{i:03d}"
                        break
                else:
                    # 如果超出所有范围，归入最后一个分区
                    partition_info['range'] = f"bin_{len(self.config.range_bins):03d}"

        return partition_info

    def get_partition_path(self, partition_info: Dict[str, Any]) -> str:
        """获取分区路径"""
        if not partition_info:
            return ""

        path_parts = []
        for key, value in partition_info.items():
            path_parts.append(f"{key}={value}")

        return "/".join(path_parts)

    def list_partitions(self, dataset_path: str) -> List[Dict[str, Any]]:
        """列出所有分区"""
        from pathlib import Path

        partitions = []
        dataset_path_obj = Path(dataset_path)

        if not dataset_path_obj.exists():
            return partitions

        for partition_dir in dataset_path_obj.iterdir():
            if partition_dir.is_dir():
                partition_info = self._parse_partition_path(partition_dir.name)
                if partition_info:
                    partition_info['path'] = str(partition_dir)
                    partition_info['file_count'] = len(list(partition_dir.glob("*")))
                    partitions.append(partition_info)

        return partitions

    def _parse_partition_path(self, path_name: str) -> Dict[str, Any]:
        """解析分区路径"""
        partition_info = {}

        if '=' in path_name:
            key, value = path_name.split('=', 1)
            partition_info[key] = value

        return partition_info

    def optimize_partitions(self, data: pd.DataFrame, target_size_mb: int = 100) -> List[pd.DataFrame]:
        """优化分区大小"""
        if data.empty:
            return []

        # 估算每个分区的大小
        estimated_size_mb = len(data) * len(data.columns) * 8 / (1024 * 1024)  # 粗略估算
        num_partitions = max(1, int(estimated_size_mb / target_size_mb))

        # 按行数分割数据
        chunk_size = len(data) // num_partitions
        if chunk_size == 0:
            chunk_size = 1

        partitions = []
        for i in range(0, len(data), chunk_size):
            partition = data.iloc[i:i + chunk_size]
            partitions.append(partition)

        return partitions

    def get_partition_stats(self, dataset_path: str) -> Dict[str, Any]:
        """获取分区统计信息"""
        from pathlib import Path

        stats = {
            'total_partitions': 0,
            'total_files': 0,
            'total_size_mb': 0,
            'partition_distribution': {},
            'largest_partition': None,
            'smallest_partition': None
        }

        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            return stats

        partition_sizes = {}

        for partition_dir in dataset_path_obj.iterdir():
            if partition_dir.is_dir():
                stats['total_partitions'] += 1
                partition_size = 0
                file_count = 0

                for file_path in partition_dir.glob("*"):
                    if file_path.is_file():
                        stats['total_files'] += 1
                        file_count += 1
                        partition_size += file_path.stat().st_size

                partition_size_mb = partition_size / (1024 * 1024)
                stats['total_size_mb'] += partition_size_mb

                partition_name = partition_dir.name
                partition_sizes[partition_name] = {
                    'size_mb': partition_size_mb,
                    'file_count': file_count
                }

        if partition_sizes:
            # 找到最大和最小分区
            largest = max(partition_sizes.items(), key=lambda x: x[1]['size_mb'])
            smallest = min(partition_sizes.items(), key=lambda x: x[1]['size_mb'])

            stats['largest_partition'] = {
                'name': largest[0],
                'size_mb': largest[1]['size_mb'],
                'file_count': largest[1]['file_count']
            }

            stats['smallest_partition'] = {
                'name': smallest[0],
                'size_mb': smallest[1]['size_mb'],
                'file_count': smallest[1]['file_count']
            }

            # 分区分布
            for partition_name, info in partition_sizes.items():
                approach = partition_name.split('=')[0] if '=' in partition_name else 'unknown'
                if approach not in stats['partition_distribution']:
                    stats['partition_distribution'][approach] = 0
                stats['partition_distribution'][approach] += 1

        return stats

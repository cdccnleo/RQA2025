import os
import time
from typing import Any, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import zlib

class DiskCache:
    """基于Parquet的磁盘缓存实现"""

    def __init__(self, root_dir: str = "data_cache"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

    def set(self, key: str, value: Any, ttl: int = 0, compress: bool = False) -> None:
        """设置磁盘缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 存活时间(秒)，0表示永不过期
            compress: 是否压缩存储
        """
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            value = self._compress(value)

        metadata = {
            'created_at': time.time(),
            'expire_at': time.time() + ttl if ttl > 0 else 0,
            'is_compressed': compress,
            'original_type': type(value).__name__
        }

        table = pa.Table.from_pydict({
            'data': [value],
            'metadata': [metadata]
        })

        pq.write_table(table, file_path)

    def get(self, key: str) -> Optional[Any]:
        """获取磁盘缓存"""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        table = pq.read_table(file_path)
        data = table['data'][0].as_py()
        metadata = table['metadata'][0].as_py()

        # 检查过期
        if metadata['expire_at'] > 0 and time.time() > metadata['expire_at']:
            os.remove(file_path)
            return None

        if metadata['is_compressed']:
            data = self._decompress(data)

        return data

    def delete(self, key: str) -> None:
        """删除磁盘缓存"""
        file_path = self._get_file_path(key)
        if file_path.exists():
            os.remove(file_path)

    def clear(self) -> None:
        """清空所有磁盘缓存"""
        for file in self.root_dir.glob('**/*.parquet'):
            file.unlink()

    def _get_file_path(self, key: str) -> Path:
        """根据key生成文件路径"""
        # 使用key的前两个字符作为子目录
        sub_dir = key[:2] if len(key) >= 2 else '00'
        return self.root_dir / sub_dir / f"{key}.parquet"

    def _compress(self, data: Any) -> bytes:
        """压缩数据"""
        return zlib.compress(str(data).encode())

    def _decompress(self, data: bytes) -> Any:
        """解压数据"""
        return zlib.decompress(data).decode()

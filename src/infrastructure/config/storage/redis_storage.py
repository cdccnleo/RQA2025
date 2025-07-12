import json
import zlib
from typing import Dict, Optional
import redis

# 尝试导入RedisCluster，如果失败则使用普通Redis
try:
    from rediscluster import RedisCluster
    REDIS_CLUSTER_AVAILABLE = True
except ImportError:
    REDIS_CLUSTER_AVAILABLE = False

class RedisStorage:
    """Redis存储适配器"""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 password: str = None, db: int = 0,
                 cluster_mode: bool = False, compress_threshold: int = 1024):
        """
        初始化Redis存储
        :param compress_threshold: 压缩阈值(字节)，大于此值的数据会被压缩
        """
        self.compress_threshold = compress_threshold
        self.cluster_mode = cluster_mode

        if cluster_mode:
            self.client = RedisCluster(
                startup_nodes=[{'host': host, 'port': port}],
                password=password,
                decode_responses=False
            )
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=False
            )

    def _compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if len(data) > self.compress_threshold:
            return zlib.compress(data)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """解压数据"""
        try:
            return zlib.decompress(data)
        except zlib.error:
            return data

    def save_version(self, env: str, version_id: str, config: Dict,
                    metadata: Dict, is_rollback: bool = False) -> bool:
        """保存配置版本到Redis"""
        key = f"config:{env}:{version_id}"
        data = {
            'config': config,
            'metadata': metadata,
            'is_rollback': is_rollback,
            'timestamp': metadata.get('timestamp')
        }
        try:
            serialized = json.dumps(data).encode('utf-8')
            compressed = self._compress(serialized)
            return self.client.set(key, compressed)
        except Exception as e:
            raise RuntimeError(f"Redis保存失败: {str(e)}")

    def get_version(self, env: str, version_id: str) -> Optional[Dict]:
        """从Redis获取配置版本"""
        key = f"config:{env}:{version_id}"
        try:
            data = self.client.get(key)
            if data:
                decompressed = self._decompress(data)
                return json.loads(decompressed.decode('utf-8'))
            return None
        except Exception as e:
            raise RuntimeError(f"Redis读取失败: {str(e)}")

    def set_limit_status(self, env: str, version_id: str, status: str) -> bool:
        """设置涨跌停标记"""
        key = f"config:{env}:{version_id}:limit_status"
        try:
            return bool(self.client.set(key, status))
        except Exception as e:
            raise RuntimeError(f"Redis设置涨跌停状态失败: {str(e)}")

    def bulk_save(self, items: Dict[str, Dict]) -> bool:
        """批量保存配置版本"""
        pipeline = self.client.pipeline()
        try:
            for version_key, data in items.items():
                serialized = json.dumps(data).encode('utf-8')
                compressed = self._compress(serialized)
                pipeline.set(version_key, compressed)
            pipeline.execute()
            return True
        except Exception as e:
            pipeline.reset()
            raise RuntimeError(f"Redis批量保存失败: {str(e)}")

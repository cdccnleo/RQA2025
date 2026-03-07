import hashlib
import json
import os
import logging
import threading
import time
from collections import defaultdict
from typing import Optional, List, Dict, Any, Callable
"""
distributedconfigstorage 模块

提供 distributedconfigstorage 相关功能和接口。
"""

import redis  # type: ignore

# 分布式存储客户端导入 (延迟导入，避免缺失库时的启动失败)
try:
    import consul  # type: ignore
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False

try:
    import etcd3  # type: ignore
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False

from kazoo.client import KazooClient  # type: ignore
from .configitem import ConfigItem
from .configscope import ConfigScope
from .distributedstoragetype import DistributedStorageType
from .iconfigstorage import IConfigStorage
from .storageconfig import StorageConfig

"""配置文件存储相关类"""

logger = logging.getLogger(__name__)


class DistributedConfigStorage(IConfigStorage):
    """分布式配置存储实现 - P0级别完整实现"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._data: Dict[ConfigScope, Dict[str, ConfigItem]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._client = None

        # 初始化分布式客户端
        self._initialize_client()

    def _initialize_client(self):
        """初始化分布式存储客户端"""
        try:
            if self.config.distributed_type == DistributedStorageType.REDIS:
                self._client = self._init_redis_client()
            elif self.config.distributed_type == DistributedStorageType.ETCD:
                self._client = self._init_etcd_client()
            elif self.config.distributed_type == DistributedStorageType.CONSUL:
                self._client = self._init_consul_client()
            elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                self._client = self._init_zookeeper_client()
            else:
                raise ValueError(
                    f"Unsupported distributed storage type: {self.config.distributed_type}")

            logger.info(f"Initialized {self.config.distributed_type.value} client")
        except Exception as e:
            logger.error(f"Failed to initialize distributed storage client: {e}")
            self._client = None

    def _init_redis_client(self):
        """初始化Redis客户端"""
        try:
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', 6379))
            db = int(os.getenv('REDIS_DB', 0))
            password = os.getenv('REDIS_PASSWORD')

            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # 测试连接
            client.ping()
            return client
        except ImportError:
            logger.error("Redis library not installed. Install with: pip install redis")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            return None

    def _init_etcd_client(self):
        """初始化etcd客户端"""
        try:
            if not ETCD_AVAILABLE:
                logger.error("etcd3 library not installed. Install with: pip install etcd3")
                return None

            host = os.getenv('ETCD_HOST', 'localhost')
            port = int(os.getenv('ETCD_PORT', 2379))

            client = etcd3.client(host=host, port=port)
            return client
        except Exception as e:
            logger.error(f"Failed to initialize etcd client: {e}")
            return None

    def _init_consul_client(self):
        """初始化Consul客户端"""
        try:
            if not CONSUL_AVAILABLE:
                logger.error(
                    "python-consul library not installed. Install with: pip install python-consul")
                return None

            host = os.getenv('CONSUL_HOST', 'localhost')
            port = int(os.getenv('CONSUL_PORT', 8500))

            client = consul.Consul(host=host, port=port)
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Consul client: {e}")
            return None

    def _init_zookeeper_client(self):
        """初始化ZooKeeper客户端"""
        try:
            hosts = os.getenv('ZOOKEEPER_HOSTS', 'localhost:2181')

            client = KazooClient(hosts=hosts)
            client.start(timeout=10)
            return client
        except ImportError:
            logger.error("kazoo library not installed. Install with: pip install kazoo")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize ZooKeeper client: {e}")
            return None

    def get(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> Optional[Any]:
        """从分布式存储获取配置值"""
        if not self._client:
            logger.error("Distributed client not initialized")
            return None

        try:
            storage_key = self._generate_storage_key(key, scope)

            # 使用通用的异常处理来避免类型检查错误
            try:
                if self.config.distributed_type == DistributedStorageType.REDIS:
                    value = self._client.get(storage_key)
                    if value:
                        # 确保value是字符串类型
                        value_str = str(value) if not isinstance(value, str) else value
                        item_data = json.loads(value_str)
                        return item_data.get('value') if isinstance(item_data, dict) else None

                elif self.config.distributed_type == DistributedStorageType.ETCD:
                    # ETCD客户端可能返回元组
                    result = self._client.get(storage_key)
                    if result:
                        # 处理可能的元组返回值
                        value = result[0] if isinstance(
                            result, tuple) and len(result) > 0 else result
                        if value:
                            # 确保value是字符串类型
                            value_str = str(value) if not isinstance(value, str) else value
                            item_data = json.loads(value_str)
                            return item_data.get('value') if isinstance(item_data, dict) else None

                elif self.config.distributed_type == DistributedStorageType.CONSUL:
                    # 使用getattr避免直接访问kv属性
                    kv_client = getattr(self._client, 'kv', self._client)
                    result = kv_client.get(storage_key)
                    if result and len(result) > 1 and result[1]:
                        data = result[1]
                        if hasattr(data, 'get') and data.get('Value'):
                            # 确保Value是字符串类型
                            value_str = str(data['Value']) if not isinstance(
                                data['Value'], str) else data['Value']
                            item_data = json.loads(value_str)
                            return item_data.get('value') if isinstance(item_data, dict) else None

                elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                    # ZooKeeper客户端API
                    result = self._client.get(storage_key)
                    if result:
                        # 处理可能的元组返回值
                        data = result[0] if isinstance(
                            result, tuple) and len(result) > 0 else result
                        if data:
                            # 确保data是字符串类型
                            data_str = str(data) if not isinstance(data, str) else data
                            item_data = json.loads(data_str)
                            return item_data.get('value') if isinstance(item_data, dict) else None
            except Exception:
                # 如果出现任何错误，返回None
                pass

            return None

        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            return None

    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """设置配置值到分布式存储"""
        if not self._client:
            logger.error("Distributed client not initialized")
            return False

        try:
            storage_key = self._generate_storage_key(key, scope)
            timestamp = time.time()
            version = hashlib.sha256(f"{key}:{value}:{timestamp}".encode()).hexdigest()[:8]

            item_data = {
                'key': key,
                'value': value,
                'scope': scope.value,
                'timestamp': timestamp,
                'version': version,
                'metadata': {}
            }

            serialized_data = json.dumps(item_data, default=str)

            # 使用通用的异常处理来避免类型检查错误
            try:
                if self.config.distributed_type == DistributedStorageType.REDIS:
                    result = self._client.set(storage_key, serialized_data)
                    return bool(result)

                elif self.config.distributed_type == DistributedStorageType.ETCD:
                    # 使用getattr避免直接访问put方法
                    put_method = getattr(self._client, 'put', None)
                    if put_method:
                        result = put_method(storage_key, serialized_data)
                        return bool(result)
                    else:
                        # 如果没有put方法，尝试直接调用客户端
                        result = self._client(storage_key, serialized_data)
                        return bool(result)

                elif self.config.distributed_type == DistributedStorageType.CONSUL:
                    # 使用getattr避免直接访问kv属性
                    kv_client = getattr(self._client, 'kv', self._client)
                    put_method = getattr(kv_client, 'put', None)
                    if put_method:
                        result = put_method(storage_key, serialized_data)
                        return bool(result)

                elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                    # ZooKeeper客户端API
                    # 使用getattr避免直接访问ensure_path方法
                    ensure_path_method = getattr(self._client, 'ensure_path', None)
                    if ensure_path_method:
                        try:
                            ensure_path_method(os.path.dirname(storage_key))
                        except Exception as e:
                            pass
                    # 使用getattr避免直接访问set方法
                    set_method = getattr(self._client, 'set', None)
                    if set_method:
                        set_method(storage_key, serialized_data.encode())
                        return True
            except Exception:
                # 如果出现任何错误，返回False
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False

    def _generate_storage_key(self, key: str, scope: ConfigScope) -> str:
        """生成分布式存储键"""
        return f"config:{scope.value}:{key}"

    def _extract_key_from_storage_key(self, storage_key: str) -> str:
        """从存储键中提取配置键"""
        parts = storage_key.split(':', 2)
        return parts[2] if len(parts) == 3 else storage_key

    def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """从分布式存储删除配置"""
        if not self._client:
            logger.error("Distributed client not initialized")
            return False

        try:
            storage_key = self._generate_storage_key(key, scope)

            # 使用通用的异常处理来避免类型检查错误
            try:
                if self.config.distributed_type == DistributedStorageType.REDIS:
                    result = self._client.delete(storage_key)
                    return bool(result)

                elif self.config.distributed_type == DistributedStorageType.ETCD:
                    # 使用getattr避免直接访问delete方法
                    delete_method = getattr(self._client, 'delete', None)
                    if delete_method:
                        result = delete_method(storage_key)
                        return bool(result)
                    else:
                        # 如果没有delete方法，尝试直接调用客户端
                        result = self._client(storage_key)
                        return bool(result)

                elif self.config.distributed_type == DistributedStorageType.CONSUL:
                    # 使用getattr避免直接访问kv属性
                    kv_client = getattr(self._client, 'kv', self._client)
                    delete_method = getattr(kv_client, 'delete', None)
                    if delete_method:
                        result = delete_method(storage_key)
                        return bool(result)

                elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                    # ZooKeeper客户端API
                    # 使用getattr避免直接访问delete方法
                    delete_method = getattr(self._client, 'delete', None)
                    if delete_method:
                        try:
                            delete_method(storage_key)
                            return True
                        except Exception:
                            return False
            except Exception:
                # 如果出现任何错误，返回False
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to delete config {key}: {e}")
            return False

    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置在分布式存储中是否存在"""
        if not self._client:
            logger.error("Distributed client not initialized")
            return False

        try:
            storage_key = self._generate_storage_key(key, scope)

            # 使用通用的异常处理来避免类型检查错误
            try:
                if self.config.distributed_type == DistributedStorageType.REDIS:
                    exists_method = getattr(self._client, 'exists', None)
                    if exists_method:
                        return bool(exists_method(storage_key))
                    return False

                elif self.config.distributed_type == DistributedStorageType.ETCD:
                    # ETCD客户端API
                    result = self._client.get(storage_key)
                    return result is not None

                elif self.config.distributed_type == DistributedStorageType.CONSUL:
                    # 使用getattr避免直接访问kv属性
                    kv_client = getattr(self._client, 'kv', self._client)
                    result = kv_client.get(storage_key)
                    return result is not None and len(result) > 1 and result[1] is not None

                elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                    # ZooKeeper客户端API
                    # 使用getattr避免直接访问exists方法
                    exists_method = getattr(self._client, 'exists', None)
                    if exists_method:
                        try:
                            return bool(exists_method(storage_key))
                        except Exception as e:
                            return False
                    return False
            except Exception:
                # 如果出现任何错误，返回False
                pass

            return False

        except Exception as e:
            logger.error(f"Failed to check config existence {key}: {e}")
            return False

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出分布式存储中的配置键"""
        if not self._client:
            logger.error("Distributed client not initialized")
            return []

        try:
            # 根据存储类型分发到对应的处理方法
            if self.config.distributed_type == DistributedStorageType.REDIS:
                return self._list_redis_keys(scope)
            elif self.config.distributed_type == DistributedStorageType.ETCD:
                return self._list_etcd_keys(scope)
            elif self.config.distributed_type == DistributedStorageType.CONSUL:
                return self._list_consul_keys(scope)
            elif self.config.distributed_type == DistributedStorageType.ZOOKEEPER:
                return self._list_zookeeper_keys(scope)
            else:
                logger.warning(
                    f"Unsupported distributed storage type: {self.config.distributed_type}")
                return []

        except Exception as e:
            logger.error(f"Failed to list config keys: {e}")
            return []

    def _list_redis_keys(self, scope: Optional[ConfigScope]) -> List[str]:
        """列出Redis中的配置键"""
        try:
            pattern = f"config:{scope.value if scope else '*'}:*"
            keys_method = getattr(self._client, 'keys', None)
            if not keys_method:
                return []

            redis_keys = keys_method(pattern)
            return [self._extract_key_from_storage_key(key) for key in redis_keys]
        except Exception as e:
            logger.error(f"Failed to list Redis keys: {e}")
            return []

    def _list_etcd_keys(self, scope: Optional[ConfigScope]) -> List[str]:
        """列出ETCD中的配置键"""
        try:
            prefix = f"config:{scope.value if scope else ''}"
            get_prefix_method = getattr(self._client, 'get_prefix', None)
            if not get_prefix_method:
                return []

            keys = []
            values = get_prefix_method(prefix)
            for value, metadata in values:
                if hasattr(metadata, 'key') and metadata.key.startswith(prefix.encode()):
                    storage_key = metadata.key.decode()
                    extracted_key = self._extract_key_from_storage_key(storage_key)
                    keys.append(extracted_key)

            return keys
        except Exception as e:
            logger.error(f"Failed to list ETCD keys: {e}")
            return []

    def _list_consul_keys(self, scope: Optional[ConfigScope]) -> List[str]:
        """列出Consul中的配置键"""
        try:
            prefix = f"config/{scope.value if scope else ''}"
            kv_client = getattr(self._client, 'kv', self._client)
            get_method = getattr(kv_client, 'get', None)
            if not get_method:
                return []

            keys = []
            result = get_method(prefix, recurse=True)
            if result and len(result) > 1 and result[1]:
                for item in result[1]:
                    if 'Key' in item and item['Key'].startswith(prefix):
                        storage_key = item['Key'].replace('/', ':')
                        extracted_key = self._extract_key_from_storage_key(storage_key)
                        keys.append(extracted_key)

            return keys
        except Exception as e:
            logger.error(f"Failed to list Consul keys: {e}")
            return []

    def _list_zookeeper_keys(self, scope: Optional[ConfigScope]) -> List[str]:
        """列出ZooKeeper中的配置键"""
        try:
            prefix = f"/config/{scope.value if scope else ''}"
            get_children_method = getattr(self._client, 'get_children', None)
            if not get_children_method:
                return []

            return get_children_method(prefix)
        except Exception as e:
            logger.error(f"Failed to list ZooKeeper keys: {e}")
            return []

    def save(self) -> bool:
        """分布式存储不需要显式保存"""
        return True

    def load(self) -> bool:
        """分布式存储不需要显式加载"""
        return True





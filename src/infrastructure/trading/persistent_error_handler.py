import sqlite3
import threading
import queue
import json
import zlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def save(self, error_id: str, record: Dict[str, Any]) -> bool:
        """保存错误记录"""
        pass

    @abstractmethod
    def load(self, error_id: str) -> Optional[Dict[str, Any]]:
        """加载错误记录"""
        pass

    @abstractmethod
    def search(self, **filters) -> list:
        """搜索错误记录"""
        pass

class SQLiteStorage(StorageBackend):
    """SQLite存储实现"""

    def __init__(self, db_path: str = "errors.db", compress: bool = True):
        """
        Args:
            db_path: 数据库文件路径
            compress: 是否压缩存储上下文数据
        """
        self.db_path = db_path
        self.compress = compress
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # 提高并发写入性能
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS error_logs (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            error_type TEXT NOT NULL,
            context BLOB,
            stack_trace TEXT,
            processed INTEGER DEFAULT 0
        )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON error_logs(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON error_logs(error_type)")
        self.conn.commit()

    def _compress_data(self, data: str) -> bytes:
        """压缩数据"""
        return zlib.compress(data.encode('utf-8'))

    def _decompress_data(self, data: bytes) -> str:
        """解压数据"""
        return zlib.decompress(data).decode('utf-8')

    def save(self, error_id: str, record: Dict[str, Any]) -> bool:
        """保存错误记录"""
        try:
            context = json.dumps(record.get('context', {}))
            if self.compress:
                context = self._compress_data(context)

            self.conn.execute(
                """
                INSERT OR REPLACE INTO error_logs 
                (id, timestamp, error_type, context, stack_trace)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    error_id,
                    record['timestamp'],
                    record['type'],
                    context,
                    record.get('stack_trace', '')
                )
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"保存错误记录失败: {e}")
            return False

    def load(self, error_id: str) -> Optional[Dict[str, Any]]:
        """加载错误记录"""
        try:
            cursor = self.conn.execute(
                "SELECT timestamp, error_type, context, stack_trace FROM error_logs WHERE id = ?",
                (error_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            timestamp, error_type, context, stack_trace = row
            if context and self.compress:
                context = self._decompress_data(context)
            context = json.loads(context) if context else {}

            return {
                'id': error_id,
                'timestamp': timestamp,
                'type': error_type,
                'context': context,
                'stack_trace': stack_trace
            }
        except Exception as e:
            logger.error(f"加载错误记录失败: {e}")
            return None

    def search(self, **filters) -> list:
        """搜索错误记录"""
        query = "SELECT id FROM error_logs WHERE 1=1"
        params = []

        if 'start_time' in filters:
            query += " AND timestamp >= ?"
            params.append(filters['start_time'])

        if 'end_time' in filters:
            query += " AND timestamp <= ?"
            params.append(filters['end_time'])

        if 'error_type' in filters:
            query += " AND error_type = ?"
            params.append(filters['error_type'])

        if 'processed' in filters:
            query += " AND processed = ?"
            params.append(int(filters['processed']))

        try:
            cursor = self.conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"搜索错误记录失败: {e}")
            return []

class PersistentErrorHandler:
    """持久化错误处理器"""

    def __init__(self,
                 storage: StorageBackend,
                 batch_size: int = 100,
                 max_queue_size: int = 1000):
        """
        Args:
            storage: 存储后端实例
            batch_size: 批量写入大小
            max_queue_size: 内存队列最大大小
        """
        self.storage = storage
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # 内存缓存
        self.error_cache = {}
        self.pending_queue = queue.Queue(maxsize=max_queue_size)

        # 后台线程
        self._persistence_thread = threading.Thread(
            target=self._persistence_worker,
            daemon=True
        )
        self._running = True
        self._persistence_thread.start()

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """处理错误并返回错误ID"""
        error_id = self._generate_error_id()
        record = {
            'id': error_id,
            'timestamp': datetime.now().timestamp(),
            'type': type(error).__name__,
            'context': context or {},
            'stack_trace': self._get_stack_trace(error)
        }

        # 存入内存缓存
        self.error_cache[error_id] = record

        # 加入持久化队列
        try:
            self.pending_queue.put_nowait(record)
        except queue.Full:
            logger.warning("持久化队列已满，丢弃错误记录")

        return error_id

    def get_error(self, error_id: str) -> Optional[Dict[str, Any]]:
        """获取错误记录"""
        # 先查内存缓存
        if error_id in self.error_cache:
            return self.error_cache[error_id]

        # 查持久化存储
        record = self.storage.load(error_id)
        if record:
            self.error_cache[error_id] = record  # 回填缓存
        return record

    def shutdown(self):
        """关闭处理器"""
        self._running = False
        self._persistence_thread.join(timeout=5)

    def _persistence_worker(self):
        """后台持久化工作线程"""
        batch = []

        while self._running:
            try:
                # 从队列获取记录
                record = self.pending_queue.get(timeout=1)
                batch.append(record)

                # 批量写入
                if len(batch) >= self.batch_size:
                    self._save_batch(batch)
                    batch = []

            except queue.Empty:
                # 队列为空时写入剩余记录
                if batch:
                    self._save_batch(batch)
                    batch = []
            except Exception as e:
                logger.error(f"持久化工作线程异常: {e}")

        # 退出前写入剩余记录
        if batch:
            self._save_batch(batch)

    def _save_batch(self, batch: list):
        """批量保存记录"""
        try:
            for record in batch:
                if not self.storage.save(record['id'], record):
                    logger.warning(f"保存错误记录失败: {record['id']}")
        except Exception as e:
            logger.error(f"批量保存失败: {e}")

    def _generate_error_id(self) -> str:
        """生成唯一错误ID"""
        return f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(threading.get_ident()))}"

    def _get_stack_trace(self, error: Exception) -> str:
        """获取异常堆栈"""
        import traceback
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))

# Redis存储实现示例
class RedisStorage(StorageBackend):
    """Redis存储实现"""

    def __init__(self, host='localhost', port=6379, db=0, compress=True):
        import redis
        self.client = redis.Redis(host=host, port=port, db=db)
        self.compress = compress

    def save(self, error_id: str, record: Dict[str, Any]) -> bool:
        try:
            import pickle
            data = pickle.dumps(record)
            if self.compress:
                data = zlib.compress(data)
            return self.client.set(f"error:{error_id}", data)
        except Exception as e:
            logger.error(f"Redis保存失败: {e}")
            return False

    def load(self, error_id: str) -> Optional[Dict[str, Any]]:
        try:
            import pickle
            data = self.client.get(f"error:{error_id}")
            if not data:
                return None
            if self.compress:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis加载失败: {e}")
            return None

    def search(self, **filters) -> list:
        # Redis需要扫描所有键，性能较低，建议使用其他存储后端进行搜索
        return []

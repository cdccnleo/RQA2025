
# 数据库迁移器常量
import time
import logging

# 可选导入 InfluxDB 客户端（符合基础设施层架构设计：可选依赖）
try:
    from influxdb_client import Point
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    Point = None
    logger = logging.getLogger(__name__)
    logger.warning("influxdb_client not installed, InfluxDB migration features will be disabled")

from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
from tqdm import tqdm
from typing import Dict, List, Optional, Any
"""数据库迁移工具"""

# 初始化logger（如果还没有初始化）
if 'logger' not in globals():
    logger = logging.getLogger(__name__)


class MigrationConstants:
    """数据库迁移器相关常量"""

    # 默认批处理配置
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_DELAY = 5

    # 统计初始化值
    DEFAULT_MIGRATED_COUNT = 0
    DEFAULT_FAILED_COUNT = 0

    # 数据验证配置
    EMPTY_DATA_LENGTH = 0
    FINAL_RETRY_ATTEMPT = 1  # retry_count - 1

    # 采样配置
    SAMPLE_SIZE = 10

    # 时间配置 (秒)
    PROGRESS_UPDATE_INTERVAL = 1.0

    # 并发配置
    MAX_WORKERS = 4

    # 文件配置
    LOG_FILE_SUFFIX = ".log"
    BACKUP_SUFFIX = ".backup"


class DatabaseMigrator:
    """数据库迁移器"""

    def __init__(self, source_adapter: IDatabaseAdapter, target_adapter: IDatabaseAdapter):
        """
        初始化数据库迁移器
        Args:
        source_adapter: 源数据库适配器
        target_adapter: 目标数据库适配器
        """
        self.source_adapter = source_adapter
        self.target_adapter = target_adapter
        self.batch_size = MigrationConstants.DEFAULT_BATCH_SIZE
        self.retry_count = MigrationConstants.DEFAULT_RETRY_COUNT
        self.retry_delay = MigrationConstants.DEFAULT_RETRY_DELAY

    def migrate_table(
        self,
        table_name: str,
        condition: str = "",
        batch_callback: Optional[callable] = None,
    ):
        """迁移指定表的数据"""
        # 初始化迁移状态
        migration_state = self._initialize_migration(table_name, condition)

        # 执行迁移过程
        max_iterations = (migration_state["total_count"] // self.batch_size + 1) * 2 + 100  # 防止无限循环
        iteration_count = 0
        consecutive_empty_batches = 0  # 连续空批次计数
        
        with tqdm(total=migration_state["total_count"], desc=f"Migrating {table_name}") as pbar:
            while migration_state["migrated"] < migration_state["total_count"]:
                # 防止无限循环的保护机制
                iteration_count += 1
                if iteration_count > max_iterations:
                    print(f"Warning: Max iterations ({max_iterations}) reached, breaking loop to prevent deadlock")
                    break
                
                # 构建批查询
                query = self._build_batch_query(table_name, condition, migration_state["migrated"])

                # 带重试的批迁移
                batch_result = self._migrate_batch_with_retry(query, table_name, migration_state)

                # 检测连续空批次以防止死锁
                if batch_result["processed"] == 0 and batch_result["failed"] == 0:
                    consecutive_empty_batches += 1
                    # 连续3次空批次则退出，避免死循环
                    if consecutive_empty_batches >= 3:
                        print(f"Warning: {consecutive_empty_batches} consecutive empty batches detected, breaking to prevent deadlock")
                        break
                else:
                    consecutive_empty_batches = 0  # 重置计数器

                # 更新进度
                self._update_progress(pbar, batch_callback, table_name,
                                      batch_result, migration_state)

                # 更新迁移状态
                migration_state["migrated"] += batch_result["processed"]
                migration_state["failed"] += batch_result["failed"]

                # 如果所有数据都已经处理（成功或失败），则退出
                if migration_state["migrated"] + migration_state["failed"] >= migration_state["total_count"]:
                    break

        # 构建迁移结果
        return self._build_migration_result(table_name, migration_state)

    def _initialize_migration(self, table_name: str, condition: str) -> Dict[str, Any]:
        """初始化迁移状态"""
        return {
            "total_count": self._get_table_count(table_name, condition),
            "migrated": MigrationConstants.DEFAULT_MIGRATED_COUNT,
            "failed": MigrationConstants.DEFAULT_FAILED_COUNT,
            "start_time": time.time(),
        }

    def _build_batch_query(self, table_name: str, condition: str, offset: int) -> str:
        """构建批查询"""
        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        query += f" LIMIT {self.batch_size} OFFSET {offset}"
        return query

    def _migrate_batch_with_retry(self, query: str, table_name: str, migration_state: Dict[str, Any]) -> Dict[str, int]:
        """带重试的批迁移"""
        data = []  # 初始化data变量
        
        # 首先获取数据（只执行一次）
        try:
            result = self.source_adapter.execute_query(query)
            if isinstance(result, dict):
                # 兼容旧的字典格式返回
                data = result.get('data', [])
            else:
                # QueryResult对象格式
                data = result.data if result.success else []

            if not data or len(data) == MigrationConstants.EMPTY_DATA_LENGTH:
                return {"processed": 0, "failed": 0}
        except Exception as e:
            print(f"Failed to query source data: {str(e)}")
            return {"processed": 0, "failed": 0}

        # 重试写入目标数据库
        data_count = len(data)  # 保存数据数量
        for attempt in range(self.retry_count):
            try:
                self._batch_insert(table_name, data)
                return {"processed": data_count, "failed": 0}
            except Exception as e:
                if attempt == self.retry_count - MigrationConstants.FINAL_RETRY_ATTEMPT:
                    print(f"Migration failed after {self.retry_count} attempts: {str(e)}")
                    return {"processed": 0, "failed": data_count}
                time.sleep(self.retry_delay)

        # 不应该到达这里，但作为后备
        return {"processed": 0, "failed": data_count}

    def _update_progress(
        self,
        pbar: tqdm,
        batch_callback: Optional[callable],
        table_name: str,
        batch_result: Dict[str, int],
        migration_state: Dict[str, Any]
    ):
        """更新进度"""
        pbar.update(batch_result["processed"])

        # 回调通知
        if batch_callback:
            batch_callback({
                "table": table_name,
                "processed": migration_state["migrated"] + batch_result["processed"],
                "total": migration_state["total_count"],
            })

    def _build_migration_result(self, table_name: str, migration_state: Dict[str, Any]) -> Dict[str, Any]:
        """构建迁移结果"""
        total_processed = migration_state["migrated"] + migration_state["failed"]
        success = migration_state["failed"] == 0 and total_processed > 0

        return {
            "success": success,
            "table": table_name,
            "total": migration_state["total_count"],
            "migrated": migration_state["migrated"],
            "failed": migration_state["failed"],
            "total_processed": total_processed,
            "duration": time.time() - migration_state["start_time"],
        }

    def _get_table_count(self, table_name: str, condition: str) -> int:
        """获取表数据总数"""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        result = self.source_adapter.execute_query(query)
        if isinstance(result, dict):
            # 兼容旧的字典格式返回
            return int(result[0]["count"]) if result.get('data') else 0
        else:
            # QueryResult对象格式
            return int(result.data[0]["count"]) if result.success and result.data else 0

    def _batch_insert(self, table_name: str, data: List[Dict]) -> None:
        """批量插入数据"""
        if not data:
            return
        # 构建INSERT语句
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        # 准备批量数据
        values = [[row[col] for col in columns] for row in data]
        # 执行批量插入
        self.target_adapter.batch_execute(query, values)

    def validate_migration(self, table_name: str) -> bool:
        """验证迁移结果"""
        source_count = self._get_table_count(table_name, "")
        target_count = self._get_table_count(table_name, "")
        if source_count != target_count:
            print(f"Count mismatch: source={source_count}, target={target_count}")
            return False
        # 抽样验证数据一致性
        sample_query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {MigrationConstants.SAMPLE_SIZE}"
        source_samples = self.source_adapter.execute_query(sample_query)
        target_samples = self.target_adapter.execute_query(sample_query)
        return self._compare_samples(source_samples, target_samples)

    def _compare_samples(self, source: List, target: List) -> bool:
        """比较样本数据"""
        if len(source) != len(target):
            return False
        for s, t in zip(source, target):
            if s != t:
                return False
        return True


class DataMigrator:
    """数据迁移工具"""

    def __init__(self, source: IDatabaseAdapter, target: IDatabaseAdapter):
        self.source = source
        self.target = target
        self.batch_size = MigrationConstants.DEFAULT_BATCH_SIZE
        self.retry_count = MigrationConstants.DEFAULT_RETRY_COUNT
        self.retry_delay = MigrationConstants.DEFAULT_RETRY_DELAY

    def set_batch_size(self, size: int):
        """设置批量大小"""
        self.batch_size = size

    def migrate_measurement(self, measurement: str, condition: str = "", batch_callback=None):
        """迁移指定measurement的数据"""
        # 初始化迁移状态
        migration_state = self._initialize_measurement_migration(measurement, condition)

        # 执行迁移过程
        with tqdm(total=migration_state["total_count"], desc=f"Migrating {measurement}") as pbar:
            while migration_state["migrated"] < migration_state["total_count"]:
                # 构建查询
                query = self._build_measurement_query(
                    measurement, condition, migration_state["migrated"])

                # 带重试的批迁移
                batch_result = self._migrate_measurement_batch_with_retry(
                    query, measurement, migration_state)

                # 处理批结果
                self._process_measurement_batch(pbar, batch_callback, batch_result, migration_state)

                # 更新迁移状态
                migration_state["migrated"] += batch_result["processed"]
                migration_state["failed"] += batch_result["failed"]

                # 检查是否完成
                if migration_state["migrated"] >= migration_state["total_count"]:
                    break

        # 构建迁移结果
        return self._build_measurement_migration_result(measurement, migration_state)

    def _initialize_measurement_migration(self, measurement: str, condition: str) -> Dict[str, Any]:
        """初始化measurement迁移状态"""
        return {
            "total_count": self._get_count(measurement, condition),
            "migrated": MigrationConstants.DEFAULT_MIGRATED_COUNT,
            "failed": 0,
            "start_time": time.time(),
        }

    def _build_measurement_query(self, measurement: str, condition: str, offset: int) -> str:
        """构建measurement查询"""
        query = (
            f'from(bucket:"source") '
            f"|> range(start: 0) "
            f'|> filter(fn: (r) => r._measurement == "{measurement}")'
        )
        if condition:
            query += f" |> filter(fn: (r) => {condition})"
        query += f" |> limit(n: {self.batch_size}, offset: {offset})"
        return query

    def _migrate_measurement_batch_with_retry(
        self,
        query: str,
        measurement: str,
        migration_state: Dict[str, Any]
    ) -> Dict[str, int]:
        """带重试的measurement批迁移"""
        for attempt in range(self.retry_count):
            try:
                result = self.source.execute_query(query)
                if isinstance(result, dict):
                    data = result.get('data', [])
                else:
                    data = result.data if hasattr(result, 'data') else []
                if not data:
                    return {"processed": 0, "failed": 0}

                # 转换数据格式
                points = self._transform_data(data, measurement)
                # 批量写入目标
                self.target.batch_write(points)  # 或者根据实际接口调整

                return {"processed": len(data), "failed": 0}

            except Exception as e:
                if attempt == self.retry_count - MigrationConstants.FINAL_RETRY_ATTEMPT:
                    failed_count = len(data) if "data" in locals() and data else 0
                    print(f"Migration failed: {str(e)}")
                    return {"processed": 0, "failed": failed_count}
                time.sleep(self.retry_delay)

        return {"processed": 0, "failed": 0}

    def _process_measurement_batch(
        self,
        pbar: tqdm,
        batch_callback,
        batch_result: Dict[str, int],
        migration_state: Dict[str, Any]
    ):
        """处理measurement批结果"""
        pbar.update(batch_result["processed"])

        # 回调通知
        if batch_callback:
            batch_callback({"total": migration_state["total_count"]})

    def _build_measurement_migration_result(self, measurement: str, migration_state: Dict[str, Any]) -> Dict[str, Any]:
        """构建measurement迁移结果"""
        return {
            "measurement": measurement,
            "total": migration_state["total_count"],
            "migrated": migration_state["migrated"],
            "failed": migration_state["failed"],
            "duration": time.time() - migration_state["start_time"],
        }

    def _get_count(self, measurement: str, condition: str) -> int:
        """获取数据总数"""
        query = (
            f'from(bucket:"source") ' f"|> range(start: 0) " f'|> filter(fn: (r) => r._measurement == "{measurement}")'
        )
        if condition:
            query += f" |> filter(fn: (r) => {condition})"
        query += " |> count()"
        result = self.source.execute_query(query)
        if isinstance(result, dict):
            # 处理字典格式返回
            return int(result.get('data', [{}])[0].get("values", {}).get("_value", 0)) if result.get('data') else 0
        elif isinstance(result, list) and result:
            # 处理列表格式返回
            return int(result[0]["values"]["_value"])
        else:
            return 0

    def _transform_data(self, data: List[Dict], measurement: str) -> List:
        """转换数据格式"""
        if not INFLUXDB_AVAILABLE or Point is None:
            logger.warning("influxdb_client not available, cannot transform data to InfluxDB Point format")
            # 返回原始数据格式，让调用者处理
            return data
        
        points = []
        for record in data:
            point = Point(measurement)
            # 添加标签
            for key, value in record["values"].items():
                if key.startswith("_"):
                    point.tag(key[1:], str(value))
            # 添加字段
            for key, value in record["values"].items():
                if not key.startswith("_"):
                    point.field(key, value)
            # 设置时间戳
            if "time" in record:
                point.time(record["time"])
            points.append(point)
        return points

    def validate_migration(self, measurement: str) -> bool:
        """验证迁移结果"""
        source_count = self._get_count(measurement, "")
        target_count = self._get_count(measurement, "")
        if source_count != target_count:
            print(f"Count mismatch: source={source_count}, target={target_count}")
            return False
        # 抽样验证数据一致性
        sample_query = (
            f'from(bucket:"source") '
            f"|> range(start: 0) "
            f'|> filter(fn: (r) => r._measurement == "{measurement}") '
            f"|> sample(n: 10)"
        )
        source_samples = self.source.query(sample_query)
        target_samples = self.target.query(sample_query)
        return self._compare_samples(source_samples, target_samples)

    def _compare_samples(self, source: List, target: List) -> bool:
        """比较样本数据"""
        if len(source) != len(target):
            return False
        for s, t in zip(source, target):
            if s["values"] != t["values"]:
                return False
        return True

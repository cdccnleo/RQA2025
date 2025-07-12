"""数据库迁移工具"""
import time
from typing import Dict, List
from tqdm import tqdm
from . import DatabaseAdapter

class DataMigrator:
    """数据迁移工具"""

    def __init__(self, source: DatabaseAdapter, target: DatabaseAdapter):
        self.source = source
        self.target = target
        self.batch_size = 1000
        self.retry_count = 3
        self.retry_delay = 5

    def set_batch_size(self, size: int):
        """设置批量大小"""
        self.batch_size = size

    def migrate_measurement(self, measurement: str,
                          condition: str = "",
                          batch_callback=None) -> Dict:
        """迁移指定measurement的数据"""
        total_count = self._get_count(measurement, condition)
        migrated = 0
        failed = 0
        start_time = time.time()

        with tqdm(total=total_count, desc=f"Migrating {measurement}") as pbar:
            while migrated < total_count:
                # 分批查询数据
                query = f'from(bucket:"source") ' \
                        f'|> range(start: 0) ' \
                        f'|> filter(fn: (r) => r._measurement == "{measurement}")'

                if condition:
                    query += f' |> filter(fn: (r) => {condition})'

                query += f' |> limit(n: {self.batch_size}, offset: {migrated})'

                # 带重试机制的迁移
                for attempt in range(self.retry_count):
                    try:
                        data = self.source.query(query)
                        if not data:
                            break

                        # 转换数据格式
                        points = self._transform_data(data, measurement)

                        # 批量写入目标
                        self.target.batch_write(points)

                        migrated += len(data)
                        pbar.update(len(data))

                        # 回调通知
                        if batch_callback:
                            batch_callback({
                                'measurement': measurement,
                                'processed': migrated,
                                'total': total_count
                            })

                        break
                    except Exception as e:
                        if attempt == self.retry_count - 1:
                            failed += len(data)
                            print(f"Migration failed: {str(e)}")
                        time.sleep(self.retry_delay)

        return {
            'measurement': measurement,
            'total': total_count,
            'migrated': migrated,
            'failed': failed,
            'duration': time.time() - start_time
        }

    def _get_count(self, measurement: str, condition: str) -> int:
        """获取数据总数"""
        query = f'from(bucket:"source") ' \
                f'|> range(start: 0) ' \
                f'|> filter(fn: (r) => r._measurement == "{measurement}")'

        if condition:
            query += f' |> filter(fn: (r) => {condition})'

        query += ' |> count()'

        result = self.source.query(query)
        return int(result[0]['values']['_value']) if result else 0

    def _transform_data(self, data: List[Dict], measurement: str) -> List:
        """转换数据格式"""
        from influxdb_client import Point
        points = []

        for record in data:
            point = Point(measurement)

            # 添加标签
            for key, value in record['values'].items():
                if key.startswith('_'):
                    point.tag(key[1:], str(value))

            # 添加字段
            for key, value in record['values'].items():
                if not key.startswith('_'):
                    point.field(key, value)

            # 设置时间戳
            if 'time' in record:
                point.time(record['time'])

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
        sample_query = f'from(bucket:"source") ' \
                      f'|> range(start: 0) ' \
                      f'|> filter(fn: (r) => r._measurement == "{measurement}") ' \
                      f'|> sample(n: 10)'

        source_samples = self.source.query(sample_query)
        target_samples = self.target.query(sample_query)

        return self._compare_samples(source_samples, target_samples)

    def _compare_samples(self, source: List, target: List) -> bool:
        """比较样本数据"""
        if len(source) != len(target):
            return False

        for s, t in zip(source, target):
            if s['values'] != t['values']:
                return False

        return True

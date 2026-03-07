"""
AI质量保障数据管理与基础设施

构建高质量数据的收集、存储、处理和治理体系：
1. 数据管道架构 - 实时数据流处理和批处理
2. 数据存储优化 - 时序数据库和特征存储
3. 数据质量保证 - 数据验证、清洗和异常检测
4. 数据治理框架 - 数据生命周期管理和合规性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles
import os
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float  # 完整性
    accuracy: float      # 准确性
    consistency: float   # 一致性
    timeliness: float    # 时效性
    validity: float      # 有效性
    uniqueness: float    # 唯一性
    overall_score: float # 综合评分

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataPipelineManager:
    """数据管道管理器"""

    def __init__(self, pipeline_config: Dict[str, Any] = None):
        self.config = pipeline_config or self._get_default_config()
        self.data_sources = {}
        self.data_sinks = {}
        self.transformations = {}
        self.quality_checks = {}
        self.pipeline_stats = {
            'total_processed': 0,
            'successful_transforms': 0,
            'failed_transforms': 0,
            'quality_violations': 0,
            'processing_time_avg': 0.0
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'batch_size': 1000,
            'processing_timeout': 300,  # 5分钟
            'max_retries': 3,
            'quality_check_enabled': True,
            'data_retention_days': 365,
            'compression_enabled': True
        }

    def register_data_source(self, name: str, source_config: Dict[str, Any]):
        """注册数据源"""
        self.data_sources[name] = source_config
        logger.info(f"已注册数据源: {name}")

    def register_data_sink(self, name: str, sink_config: Dict[str, Any]):
        """注册数据汇"""
        self.data_sinks[name] = sink_config
        logger.info(f"已注册数据汇: {name}")

    def register_transformation(self, name: str, transform_func: callable):
        """注册数据转换"""
        self.transformations[name] = transform_func
        logger.info(f"已注册数据转换: {name}")

    def register_quality_check(self, name: str, check_func: callable):
        """注册质量检查"""
        self.quality_checks[name] = check_func
        logger.info(f"已注册质量检查: {name}")

    async def execute_pipeline(self, pipeline_definition: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据管道"""
        start_time = datetime.now()

        try:
            # 1. 数据提取
            raw_data = await self._extract_data(pipeline_definition.get('sources', []))

            # 2. 数据转换
            transformed_data = await self._transform_data(
                raw_data, pipeline_definition.get('transformations', [])
            )

            # 3. 质量检查
            quality_results = await self._check_data_quality(
                transformed_data, pipeline_definition.get('quality_checks', [])
            )

            # 4. 数据加载
            load_results = await self._load_data(
                transformed_data, pipeline_definition.get('sinks', [])
            )

            # 更新统计
            processing_time = (datetime.now() - start_time).total_seconds()
            self.pipeline_stats['total_processed'] += len(transformed_data) if hasattr(transformed_data, '__len__') else 1
            self.pipeline_stats['processing_time_avg'] = (
                self.pipeline_stats['processing_time_avg'] +
                (processing_time - self.pipeline_stats['processing_time_avg']) /
                max(1, self.pipeline_stats['total_processed'])
            )

            result = {
                'success': True,
                'records_processed': len(transformed_data) if hasattr(transformed_data, '__len__') else 1,
                'processing_time': processing_time,
                'quality_results': quality_results,
                'load_results': load_results,
                'timestamp': datetime.now()
            }

            if quality_results.get('passed', True):
                self.pipeline_stats['successful_transforms'] += 1
            else:
                self.pipeline_stats['quality_violations'] += 1

            return result

        except Exception as e:
            logger.error(f"数据管道执行失败: {e}")
            self.pipeline_stats['failed_transforms'] += 1

            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now()
            }

    async def _extract_data(self, sources: List[str]) -> pd.DataFrame:
        """数据提取"""
        all_data = []

        for source_name in sources:
            if source_name in self.data_sources:
                source_config = self.data_sources[source_name]

                try:
                    # 根据源类型提取数据
                    if source_config['type'] == 'database':
                        data = await self._extract_from_database(source_config)
                    elif source_config['type'] == 'api':
                        data = await self._extract_from_api(source_config)
                    elif source_config['type'] == 'file':
                        data = await self._extract_from_file(source_config)
                    else:
                        logger.warning(f"不支持的数据源类型: {source_config['type']}")
                        continue

                    all_data.append(data)

                except Exception as e:
                    logger.error(f"从数据源 {source_name} 提取数据失败: {e}")
                    continue

        # 合并所有数据
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()

    async def _extract_from_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从数据库提取数据"""
        # 这里应该实现具体的数据库连接和查询逻辑
        # 为了演示，返回模拟数据
        return pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=1),
                                      periods=60, freq='1min'),
            'metric_value': np.random.normal(100, 10, 60),
            'source': config.get('name', 'database')
        })

    async def _extract_from_api(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从API提取数据"""
        # 这里应该实现API调用逻辑
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'api_response_time': [150.5],
            'source': config.get('name', 'api')
        })

    async def _extract_from_file(self, config: Dict[str, Any]) -> pd.DataFrame:
        """从文件提取数据"""
        file_path = config.get('path')
        if file_path and os.path.exists(file_path):
            try:
                if file_path.endswith('.csv'):
                    return pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    return pd.read_json(file_path)
                elif file_path.endswith('.parquet'):
                    return pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")

        return pd.DataFrame()

    async def _transform_data(self, data: pd.DataFrame,
                            transformations: List[str]) -> pd.DataFrame:
        """数据转换"""
        transformed_data = data.copy()

        for transform_name in transformations:
            if transform_name in self.transformations:
                try:
                    transform_func = self.transformations[transform_name]
                    transformed_data = await self._execute_transform(
                        transform_func, transformed_data
                    )
                except Exception as e:
                    logger.error(f"数据转换失败 {transform_name}: {e}")
                    continue

        return transformed_data

    async def _execute_transform(self, transform_func: callable,
                               data: pd.DataFrame) -> pd.DataFrame:
        """执行转换函数"""
        if asyncio.iscoroutinefunction(transform_func):
            return await transform_func(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, transform_func, data)

    async def _check_data_quality(self, data: pd.DataFrame,
                                quality_checks: List[str]) -> Dict[str, Any]:
        """数据质量检查"""
        quality_results = {
            'passed': True,
            'checks': {},
            'violations': []
        }

        for check_name in quality_checks:
            if check_name in self.quality_checks:
                try:
                    check_func = self.quality_checks[check_name]
                    check_result = await self._execute_quality_check(
                        check_func, data
                    )

                    quality_results['checks'][check_name] = check_result

                    if not check_result.get('passed', True):
                        quality_results['passed'] = False
                        quality_results['violations'].append({
                            'check': check_name,
                            'details': check_result
                        })

                except Exception as e:
                    logger.error(f"质量检查失败 {check_name}: {e}")
                    quality_results['passed'] = False
                    quality_results['violations'].append({
                        'check': check_name,
                        'error': str(e)
                    })

        return quality_results

    async def _execute_quality_check(self, check_func: callable,
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """执行质量检查"""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, check_func, data)

    async def _load_data(self, data: pd.DataFrame, sinks: List[str]) -> Dict[str, Any]:
        """数据加载"""
        load_results = {}

        for sink_name in sinks:
            if sink_name in self.data_sinks:
                sink_config = self.data_sinks[sink_name]

                try:
                    if sink_config['type'] == 'database':
                        result = await self._load_to_database(data, sink_config)
                    elif sink_config['type'] == 'file':
                        result = await self._load_to_file(data, sink_config)
                    elif sink_config['type'] == 'stream':
                        result = await self._load_to_stream(data, sink_config)
                    else:
                        logger.warning(f"不支持的数据汇类型: {sink_config['type']}")
                        continue

                    load_results[sink_name] = {'success': True, 'details': result}

                except Exception as e:
                    logger.error(f"数据加载失败 {sink_name}: {e}")
                    load_results[sink_name] = {'success': False, 'error': str(e)}

        return load_results

    async def _load_to_database(self, data: pd.DataFrame,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """加载到数据库"""
        # 这里应该实现具体的数据库写入逻辑
        # 为了演示，记录操作信息
        return {
            'records_loaded': len(data),
            'table': config.get('table', 'quality_data'),
            'timestamp': datetime.now()
        }

    async def _load_to_file(self, data: pd.DataFrame,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """加载到文件"""
        file_path = config.get('path')
        if file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                if file_path.endswith('.csv'):
                    data.to_csv(file_path, index=False)
                elif file_path.endswith('.json'):
                    data.to_json(file_path, orient='records', date_format='iso')
                elif file_path.endswith('.parquet'):
                    data.to_parquet(file_path, index=False)

                return {
                    'file_path': file_path,
                    'records_saved': len(data),
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                }
            except Exception as e:
                raise Exception(f"保存文件失败: {e}")

        return {'error': '未指定文件路径'}

    async def _load_to_stream(self, data: pd.DataFrame,
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """加载到数据流"""
        # 这里应该实现流处理逻辑，如Kafka、Kinesis等
        return {
            'stream_name': config.get('stream_name', 'quality_stream'),
            'records_streamed': len(data),
            'timestamp': datetime.now()
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计"""
        return self.pipeline_stats.copy()


class TimeSeriesDataStore:
    """时序数据存储"""

    def __init__(self, db_path: str = "data/timeseries_quality.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.max_connections = 5
        self._ensure_db_setup()

    def _ensure_db_setup(self):
        """确保数据库设置"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # 创建时序数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS timeseries_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    tags TEXT,  -- JSON格式的标签
                    timestamp DATETIME NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    source TEXT
                )
            ''')

            # 创建索引以提高查询性能
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metric_timestamp
                ON timeseries_metrics (metric_name, timestamp)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON timeseries_metrics (timestamp)
            ''')

            # 创建特征存储表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_type TEXT,  -- numeric, categorical, text
                    metadata TEXT,  -- JSON格式的元数据
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def store_timeseries_data(self, metric_name: str, data_points: List[Dict[str, Any]],
                            tags: Dict[str, Any] = None):
        """存储时序数据"""
        with sqlite3.connect(self.db_path) as conn:
            for point in data_points:
                try:
                    conn.execute('''
                        INSERT INTO timeseries_metrics
                        (metric_name, metric_value, tags, timestamp, quality_score, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        metric_name,
                        point.get('value'),
                        json.dumps(tags or {}),
                        point.get('timestamp'),
                        point.get('quality_score', 1.0),
                        point.get('source', 'unknown')
                    ))
                except Exception as e:
                    logger.error(f"存储时序数据点失败: {e}")
                    continue

            conn.commit()

    def query_timeseries_data(self, metric_name: str,
                            start_time: datetime = None,
                            end_time: datetime = None,
                            limit: int = 1000) -> pd.DataFrame:
        """查询时序数据"""
        query = "SELECT * FROM timeseries_metrics WHERE metric_name = ?"
        params = [metric_name]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['tags'] = df['tags'].apply(json.loads)

        return df

    def store_feature_data(self, feature_name: str, feature_data: Dict[str, Any]):
        """存储特征数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO feature_store
                (feature_name, feature_value, feature_type, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                feature_name,
                feature_data.get('value'),
                feature_data.get('type', 'numeric'),
                json.dumps(feature_data.get('metadata', {}))
            ))
            conn.commit()

    def get_feature_data(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """获取特征数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM feature_store
                WHERE feature_name = ?
                ORDER BY updated_at DESC LIMIT 1
            ''', (feature_name,))

            row = cursor.fetchone()
            if row:
                return {
                    'feature_name': row[1],
                    'value': row[2],
                    'type': row[3],
                    'metadata': json.loads(row[4]),
                    'created_at': row[5],
                    'updated_at': row[6]
                }

        return None

    def cleanup_old_data(self, retention_days: int = 365):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        with sqlite3.connect(self.db_path) as conn:
            # 删除旧的时序数据
            cursor = conn.execute('''
                DELETE FROM timeseries_metrics
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))

            deleted_count = cursor.rowcount

            # 删除旧的特征数据（保留最新的）
            conn.execute('''
                DELETE FROM feature_store
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM feature_store
                    GROUP BY feature_name
                )
            ''')

            conn.commit()

        logger.info(f"已清理 {deleted_count} 条旧时序数据")
        return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        with sqlite3.connect(self.db_path) as conn:
            # 时序数据统计
            timeseries_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_records,
                    COUNT(DISTINCT metric_name) as unique_metrics,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp,
                    AVG(quality_score) as avg_quality_score
                FROM timeseries_metrics
            ''').fetchone()

            # 特征数据统计
            feature_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_features,
                    COUNT(DISTINCT feature_name) as unique_features
                FROM feature_store
            ''').fetchone()

        return {
            'timeseries_data': {
                'total_records': timeseries_stats[0],
                'unique_metrics': timeseries_stats[1],
                'date_range': {
                    'earliest': timeseries_stats[2],
                    'latest': timeseries_stats[3]
                },
                'avg_quality_score': timeseries_stats[4]
            },
            'feature_store': {
                'total_features': feature_stats[0],
                'unique_features': feature_stats[1]
            },
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }


class DataQualityManager:
    """数据质量管理器"""

    def __init__(self):
        self.quality_rules = {}
        self.quality_history = []
        self.alert_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.90,
            'timeliness': 0.95,
            'overall_score': 0.85
        }

    def register_quality_rule(self, rule_name: str, rule_func: callable,
                            description: str = ""):
        """注册质量规则"""
        self.quality_rules[rule_name] = {
            'function': rule_func,
            'description': description,
            'violation_count': 0,
            'last_checked': None
        }
        logger.info(f"已注册质量规则: {rule_name}")

    async def assess_data_quality(self, data: pd.DataFrame,
                                rules: List[str] = None) -> DataQualityMetrics:
        """评估数据质量"""
        if rules is None:
            rules = list(self.quality_rules.keys())

        quality_scores = {}

        # 执行每个质量规则
        for rule_name in rules:
            if rule_name in self.quality_rules:
                try:
                    rule_config = self.quality_rules[rule_name]
                    rule_func = rule_config['function']

                    if asyncio.iscoroutinefunction(rule_func):
                        score = await rule_func(data)
                    else:
                        loop = asyncio.get_event_loop()
                        score = await loop.run_in_executor(None, rule_func, data)

                    quality_scores[rule_name] = score

                    # 更新规则统计
                    rule_config['last_checked'] = datetime.now()
                    if score < 0.8:  # 质量分数低于阈值
                        rule_config['violation_count'] += 1

                except Exception as e:
                    logger.error(f"质量规则评估失败 {rule_name}: {e}")
                    quality_scores[rule_name] = 0.0

        # 计算综合质量指标
        completeness = self._calculate_completeness(data)
        accuracy = quality_scores.get('accuracy', self._calculate_accuracy(data))
        consistency = quality_scores.get('consistency', self._calculate_consistency(data))
        timeliness = quality_scores.get('timeliness', self._calculate_timeliness(data))
        validity = quality_scores.get('validity', self._calculate_validity(data))
        uniqueness = quality_scores.get('uniqueness', self._calculate_uniqueness(data))

        # 计算综合评分
        weights = {
            'completeness': 0.2,
            'accuracy': 0.25,
            'consistency': 0.15,
            'timeliness': 0.15,
            'validity': 0.15,
            'uniqueness': 0.1
        }

        overall_score = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency'] +
            timeliness * weights['timeliness'] +
            validity * weights['validity'] +
            uniqueness * weights['uniqueness']
        )

        quality_metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness,
            overall_score=overall_score
        )

        # 记录质量历史
        self.quality_history.append({
            'timestamp': datetime.now(),
            'metrics': quality_metrics.to_dict(),
            'data_shape': data.shape,
            'rules_evaluated': rules
        })

        # 保持历史记录大小
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-1000:]

        # 检查是否需要告警
        await self._check_quality_alerts(quality_metrics)

        return quality_metrics

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """计算完整性"""
        if data.empty:
            return 0.0

        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.count().sum()

        return non_null_cells / total_cells if total_cells > 0 else 0.0

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """计算准确性"""
        # 这里应该基于业务规则验证数据的准确性
        # 为了演示，使用简单的启发式方法
        try:
            # 检查数值列的合理性
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            accuracy_scores = []

            for col in numeric_columns:
                values = data[col].dropna()
                if len(values) > 0:
                    # 检查是否有异常值（使用IQR方法）
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    normal_values = ((values >= lower_bound) & (values <= upper_bound)).sum()
                    accuracy_scores.append(normal_values / len(values))

            return np.mean(accuracy_scores) if accuracy_scores else 0.8

        except Exception:
            return 0.8

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """计算一致性"""
        # 检查数据类型一致性和格式一致性
        try:
            consistency_scores = []

            # 检查每列的数据类型一致性
            for col in data.columns:
                if data[col].dtype == 'object':
                    # 对于字符串列，检查格式一致性
                    non_null_values = data[col].dropna()
                    if len(non_null_values) > 1:
                        # 简单的格式一致性检查
                        first_value = str(non_null_values.iloc[0])
                        consistent_count = non_null_values.astype(str).str.match(
                            r'^' + first_value.replace('(', r'\(').replace(')', r'\)') + r'$'
                        ).sum()
                        consistency_scores.append(consistent_count / len(non_null_values))
                    else:
                        consistency_scores.append(1.0)
                else:
                    # 对于数值列，检查值域一致性
                    consistency_scores.append(0.9)  # 假设数值列基本一致

            return np.mean(consistency_scores) if consistency_scores else 0.8

        except Exception:
            return 0.8

    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """计算时效性"""
        try:
            if 'timestamp' in data.columns:
                # 检查数据的时间分布
                timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
                valid_timestamps = timestamps.dropna()

                if len(valid_timestamps) > 1:
                    # 计算数据的时间跨度与期望间隔的比值
                    time_span = (valid_timestamps.max() - valid_timestamps.min()).total_seconds()
                    expected_span = len(valid_timestamps) * 3600  # 假设每小时一个数据点

                    timeliness = min(1.0, expected_span / time_span) if time_span > 0 else 0.0
                    return timeliness
                else:
                    return 0.5
            else:
                # 如果没有时间戳列，假设数据是及时的
                return 0.8

        except Exception:
            return 0.8

    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """计算有效性"""
        try:
            validity_scores = []

            for col in data.columns:
                valid_count = 0
                total_count = len(data)

                if data[col].dtype == 'object':
                    # 字符串列：检查非空且非空白
                    valid_count = data[col].astype(str).str.strip().str.len().gt(0).sum()
                elif np.issubdtype(data[col].dtype, np.number):
                    # 数值列：检查非NaN且有限
                    valid_count = data[col].notna().sum()
                else:
                    # 其他类型：检查非空
                    valid_count = data[col].notna().sum()

                validity_scores.append(valid_count / total_count if total_count > 0 else 0.0)

            return np.mean(validity_scores) if validity_scores else 0.8

        except Exception:
            return 0.8

    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """计算唯一性"""
        try:
            uniqueness_scores = []

            for col in data.columns:
                if len(data) > 0:
                    unique_ratio = data[col].nunique() / len(data)
                    uniqueness_scores.append(unique_ratio)

            # 对于大多数指标，过高的唯一性可能表明数据质量问题
            # 这里使用一个平衡函数
            avg_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0.0
            optimal_uniqueness = 0.7  # 假设最佳唯一性为70%

            uniqueness_score = 1.0 - abs(avg_uniqueness - optimal_uniqueness) / optimal_uniqueness
            return max(0.0, min(1.0, uniqueness_score))

        except Exception:
            return 0.8

    async def _check_quality_alerts(self, quality_metrics: DataQualityMetrics):
        """检查质量告警"""
        try:
            alerts = []

            # 检查各项指标是否低于阈值
            if quality_metrics.completeness < self.alert_thresholds['completeness']:
                alerts.append(f"数据完整性过低: {quality_metrics.completeness:.2%}")

            if quality_metrics.accuracy < self.alert_thresholds['accuracy']:
                alerts.append(f"数据准确性过低: {quality_metrics.accuracy:.2%}")

            if quality_metrics.timeliness < self.alert_thresholds['timeliness']:
                alerts.append(f"数据时效性过低: {quality_metrics.timeliness:.2%}")

            if quality_metrics.overall_score < self.alert_thresholds['overall_score']:
                alerts.append(f"数据整体质量过低: {quality_metrics.overall_score:.2%}")

            if alerts:
                logger.warning(f"数据质量告警: {'; '.join(alerts)}")

                # 这里可以集成告警系统发送通知

        except Exception as e:
            logger.error(f"质量告警检查失败: {e}")

    def get_quality_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取质量历史"""
        return self.quality_history[-limit:] if self.quality_history else []

    def get_quality_stats(self) -> Dict[str, Any]:
        """获取质量统计"""
        if not self.quality_history:
            return {}

        recent_history = self.quality_history[-50:]  # 最近50次评估

        # 计算各项指标的趋势
        completeness_scores = [h['metrics']['completeness'] for h in recent_history]
        accuracy_scores = [h['metrics']['accuracy'] for h in recent_history]
        overall_scores = [h['metrics']['overall_score'] for h in recent_history]

        def calculate_trend(scores):
            if len(scores) < 2:
                return 'stable'
            slope = (scores[-1] - scores[0]) / len(scores)
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'

        return {
            'total_assessments': len(self.quality_history),
            'recent_assessments': len(recent_history),
            'current_quality': self.quality_history[-1]['metrics'] if self.quality_history else {},
            'trends': {
                'completeness': calculate_trend(completeness_scores),
                'accuracy': calculate_trend(accuracy_scores),
                'overall': calculate_trend(overall_scores)
            },
            'avg_scores': {
                'completeness': np.mean(completeness_scores),
                'accuracy': np.mean(accuracy_scores),
                'overall': np.mean(overall_scores)
            },
            'quality_rules': {
                rule_name: {
                    'description': config['description'],
                    'violation_count': config['violation_count'],
                    'last_checked': config['last_checked'].isoformat() if config['last_checked'] else None
                }
                for rule_name, config in self.quality_rules.items()
            }
        }


class DataGovernanceFramework:
    """数据治理框架"""

    def __init__(self):
        self.data_catalog = {}
        self.retention_policies = {}
        self.access_policies = {}
        self.compliance_rules = {}
        self.audit_log = []

    def register_data_asset(self, asset_id: str, asset_metadata: Dict[str, Any]):
        """注册数据资产"""
        self.data_catalog[asset_id] = {
            'metadata': asset_metadata,
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'access_count': 0,
            'quality_score': 1.0,
            'compliance_status': 'compliant'
        }
        logger.info(f"已注册数据资产: {asset_id}")

    def set_retention_policy(self, asset_type: str, retention_days: int,
                           archive_policy: str = 'delete'):
        """设置保留策略"""
        self.retention_policies[asset_type] = {
            'retention_days': retention_days,
            'archive_policy': archive_policy,
            'created_at': datetime.now()
        }
        logger.info(f"已设置保留策略: {asset_type} - {retention_days}天")

    def set_access_policy(self, asset_id: str, policy_rules: Dict[str, Any]):
        """设置访问策略"""
        self.access_policies[asset_id] = {
            'rules': policy_rules,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        logger.info(f"已设置访问策略: {asset_id}")

    def add_compliance_rule(self, rule_id: str, rule_definition: Dict[str, Any]):
        """添加合规规则"""
        self.compliance_rules[rule_id] = {
            'definition': rule_definition,
            'created_at': datetime.now(),
            'violation_count': 0
        }
        logger.info(f"已添加合规规则: {rule_id}")

    def check_data_access(self, asset_id: str, user_context: Dict[str, Any]) -> bool:
        """检查数据访问权限"""
        if asset_id not in self.access_policies:
            return True  # 默认允许访问

        policy = self.access_policies[asset_id]

        # 简单的访问控制逻辑（可以扩展为更复杂的RBAC/ABAC）
        user_role = user_context.get('role', 'user')
        required_role = policy['rules'].get('min_role', 'user')

        role_hierarchy = {'admin': 3, 'analyst': 2, 'user': 1}
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        has_access = user_level >= required_level

        # 记录访问审计
        self.audit_log.append({
            'timestamp': datetime.now(),
            'asset_id': asset_id,
            'user_context': user_context,
            'access_granted': has_access,
            'reason': 'insufficient_permissions' if not has_access else 'granted'
        })

        if asset_id in self.data_catalog:
            self.data_catalog[asset_id]['access_count'] += 1

        return has_access

    def enforce_retention_policy(self, asset_type: str) -> List[str]:
        """执行保留策略"""
        if asset_type not in self.retention_policies:
            return []

        policy = self.retention_policies[asset_type]
        cutoff_date = datetime.now() - timedelta(days=policy['retention_days'])

        # 这里应该实现具体的清理逻辑
        # 返回需要清理的资产ID列表
        assets_to_cleanup = []  # 在实际实现中需要查询数据库

        logger.info(f"执行保留策略: {asset_type}, 清理 {len(assets_to_cleanup)} 个资产")

        return assets_to_cleanup

    def check_compliance(self, asset_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """检查合规性"""
        compliance_results = {
            'asset_id': asset_id,
            'overall_compliant': True,
            'violations': [],
            'checked_rules': []
        }

        for rule_id, rule_config in self.compliance_rules.items():
            compliance_results['checked_rules'].append(rule_id)

            try:
                # 执行合规检查
                rule_result = self._execute_compliance_rule(rule_config['definition'], data)

                if not rule_result['passed']:
                    compliance_results['overall_compliant'] = False
                    compliance_results['violations'].append({
                        'rule_id': rule_id,
                        'description': rule_result.get('description', '合规检查失败'),
                        'severity': rule_result.get('severity', 'medium')
                    })

                    rule_config['violation_count'] += 1

            except Exception as e:
                logger.error(f"合规检查失败 {rule_id}: {e}")
                compliance_results['overall_compliant'] = False
                compliance_results['violations'].append({
                    'rule_id': rule_id,
                    'description': f'检查执行失败: {e}',
                    'severity': 'high'
                })

        # 更新资产合规状态
        if asset_id in self.data_catalog:
            self.data_catalog[asset_id]['compliance_status'] = \
                'compliant' if compliance_results['overall_compliant'] else 'non_compliant'

        return compliance_results

    def _execute_compliance_rule(self, rule_definition: Dict[str, Any],
                               data: pd.DataFrame) -> Dict[str, Any]:
        """执行合规规则"""
        rule_type = rule_definition.get('type')

        if rule_type == 'pii_check':
            # PII数据检查
            pii_columns = rule_definition.get('pii_columns', [])
            has_pii = any(col in data.columns for col in pii_columns)

            return {
                'passed': not has_pii or rule_definition.get('allow_pii', False),
                'description': '检测到PII数据' if has_pii else '无PII数据',
                'severity': 'high' if has_pii else 'low'
            }

        elif rule_type == 'retention_check':
            # 保留期检查
            max_age_days = rule_definition.get('max_age_days', 365)
            if 'timestamp' in data.columns:
                oldest_data = pd.to_datetime(data['timestamp']).min()
                age_days = (datetime.now() - oldest_data).days

                return {
                    'passed': age_days <= max_age_days,
                    'description': f'数据年龄 {age_days} 天，超过最大保留期 {max_age_days} 天' if age_days > max_age_days else '数据在保留期内',
                    'severity': 'medium' if age_days > max_age_days else 'low'
                }

            return {'passed': True, 'description': '无法检查数据年龄', 'severity': 'low'}

        # 默认通过
        return {'passed': True, 'description': '规则检查通过', 'severity': 'low'}

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志"""
        return self.audit_log[-limit:] if self.audit_log else []

    def get_governance_stats(self) -> Dict[str, Any]:
        """获取治理统计"""
        return {
            'total_assets': len(self.data_catalog),
            'total_policies': len(self.access_policies),
            'total_rules': len(self.compliance_rules),
            'total_audit_events': len(self.audit_log),
            'compliance_violations': sum(rule['violation_count'] for rule in self.compliance_rules.values()),
            'asset_stats': {
                asset_id: {
                    'access_count': info['access_count'],
                    'compliance_status': info['compliance_status'],
                    'quality_score': info['quality_score']
                }
                for asset_id, info in self.data_catalog.items()
            }
        }

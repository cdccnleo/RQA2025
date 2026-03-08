"""
元数据存储模块

提供管道执行历史、阶段状态和执行时间的记录与管理。
支持执行追踪、性能分析和审计日志功能。
"""

import logging
import sqlite3
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from threading import Lock
from contextlib import contextmanager

from ..exceptions import PipelineException, PipelineErrorCode

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StageStatus(Enum):
    """阶段状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLBACK = "rollback"


@dataclass
class StageExecutionRecord:
    """
    阶段执行记录
    
    Attributes:
        stage_id: 阶段执行ID
        pipeline_execution_id: 所属管道执行ID
        stage_name: 阶段名称
        stage_type: 阶段类型
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        duration_seconds: 执行时长（秒）
        input_data_info: 输入数据信息
        output_data_info: 输出数据信息
        error_info: 错误信息
        metrics: 阶段指标
        logs: 执行日志
    """
    stage_id: str
    pipeline_execution_id: str
    stage_name: str
    stage_type: str
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    input_data_info: Dict[str, Any] = field(default_factory=dict)
    output_data_info: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage_id': self.stage_id,
            'pipeline_execution_id': self.pipeline_execution_id,
            'stage_name': self.stage_name,
            'stage_type': self.stage_type,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'input_data_info': self.input_data_info,
            'output_data_info': self.output_data_info,
            'error_info': self.error_info,
            'metrics': self.metrics,
            'logs': self.logs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageExecutionRecord':
        """从字典创建"""
        return cls(
            stage_id=data['stage_id'],
            pipeline_execution_id=data['pipeline_execution_id'],
            stage_name=data['stage_name'],
            stage_type=data['stage_type'],
            status=StageStatus(data.get('status', 'pending')),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            duration_seconds=data.get('duration_seconds', 0.0),
            input_data_info=data.get('input_data_info', {}),
            output_data_info=data.get('output_data_info', {}),
            error_info=data.get('error_info'),
            metrics=data.get('metrics', {}),
            logs=data.get('logs', [])
        )


@dataclass
class PipelineExecutionRecord:
    """
    管道执行记录
    
    Attributes:
        execution_id: 执行ID
        pipeline_name: 管道名称
        pipeline_version: 管道版本
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        duration_seconds: 执行时长（秒）
        trigger_type: 触发类型 (manual, scheduled, api)
        triggered_by: 触发者
        config_snapshot: 配置快照
        global_metrics: 全局指标
        tags: 标签
    """
    execution_id: str
    pipeline_name: str
    pipeline_version: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    trigger_type: str = "manual"
    triggered_by: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    global_metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'pipeline_name': self.pipeline_name,
            'pipeline_version': self.pipeline_version,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'trigger_type': self.trigger_type,
            'triggered_by': self.triggered_by,
            'config_snapshot': self.config_snapshot,
            'global_metrics': self.global_metrics,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineExecutionRecord':
        """从字典创建"""
        return cls(
            execution_id=data['execution_id'],
            pipeline_name=data['pipeline_name'],
            pipeline_version=data['pipeline_version'],
            status=ExecutionStatus(data.get('status', 'pending')),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None,
            duration_seconds=data.get('duration_seconds', 0.0),
            trigger_type=data.get('trigger_type', 'manual'),
            triggered_by=data.get('triggered_by'),
            config_snapshot=data.get('config_snapshot', {}),
            global_metrics=data.get('global_metrics', {}),
            tags=data.get('tags', [])
        )


@dataclass
class ExecutionSummary:
    """
    执行摘要统计
    
    Attributes:
        total_executions: 总执行次数
        successful_count: 成功次数
        failed_count: 失败次数
        cancelled_count: 取消次数
        avg_duration_seconds: 平均执行时长
        min_duration_seconds: 最短执行时长
        max_duration_seconds: 最长执行时长
        success_rate: 成功率
    """
    total_executions: int = 0
    successful_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_executions': self.total_executions,
            'successful_count': self.successful_count,
            'failed_count': self.failed_count,
            'cancelled_count': self.cancelled_count,
            'avg_duration_seconds': self.avg_duration_seconds,
            'min_duration_seconds': self.min_duration_seconds,
            'max_duration_seconds': self.max_duration_seconds,
            'success_rate': self.success_rate
        }


@dataclass
class MetadataStoreConfig:
    """
    元数据存储配置
    
    Attributes:
        storage_path: 存储路径
        max_history_days: 最大历史保留天数
        max_executions: 最大执行记录数
        enable_archiving: 是否启用归档
        archive_threshold_days: 归档阈值天数
    """
    storage_path: str = "./metadata_store"
    max_history_days: int = 90
    max_executions: int = 10000
    enable_archiving: bool = True
    archive_threshold_days: int = 30


class MetadataStore:
    """
    元数据存储管理器
    
    提供管道执行历史、阶段状态和执行时间的记录与管理。
    支持执行追踪、性能分析和审计日志功能。
    
    Attributes:
        config: 存储配置
        _lock: 线程锁
    """
    
    def __init__(self, config: Optional[MetadataStoreConfig] = None):
        """
        初始化元数据存储
        
        Args:
            config: 存储配置，为None时使用默认配置
        """
        self.config = config or MetadataStoreConfig()
        self._lock = Lock()
        
        # 初始化存储路径
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._archive_path = self._storage_path / "archive"
        self._archive_path.mkdir(exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info("元数据存储初始化完成")
    
    def _init_database(self) -> None:
        """初始化SQLite数据库"""
        self._db_path = self._storage_path / "metadata_store.db"
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 创建管道执行记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_executions (
                    execution_id TEXT PRIMARY KEY,
                    pipeline_name TEXT NOT NULL,
                    pipeline_version TEXT,
                    status TEXT DEFAULT 'pending',
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL DEFAULT 0.0,
                    trigger_type TEXT DEFAULT 'manual',
                    triggered_by TEXT,
                    config_snapshot TEXT,
                    global_metrics TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建阶段执行记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stage_executions (
                    stage_id TEXT PRIMARY KEY,
                    pipeline_execution_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    stage_type TEXT,
                    status TEXT DEFAULT 'pending',
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL DEFAULT 0.0,
                    input_data_info TEXT,
                    output_data_info TEXT,
                    error_info TEXT,
                    metrics TEXT,
                    logs TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pipeline_execution_id) 
                        REFERENCES pipeline_executions(execution_id)
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pipeline_name 
                ON pipeline_executions(pipeline_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_execution_status 
                ON pipeline_executions(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_execution_time 
                ON pipeline_executions(start_time)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stage_execution 
                ON stage_executions(pipeline_execution_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stage_status 
                ON stage_executions(status)
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_execution_id(self) -> str:
        """生成执行ID"""
        return f"exec_{uuid.uuid4().hex[:16]}"
    
    def _generate_stage_id(self) -> str:
        """生成阶段ID"""
        return f"stage_{uuid.uuid4().hex[:16]}"
    
    def start_pipeline_execution(
        self,
        pipeline_name: str,
        pipeline_version: str = "1.0.0",
        trigger_type: str = "manual",
        triggered_by: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        开始管道执行记录
        
        Args:
            pipeline_name: 管道名称
            pipeline_version: 管道版本
            trigger_type: 触发类型
            triggered_by: 触发者
            config_snapshot: 配置快照
            tags: 标签
            
        Returns:
            执行ID
        """
        try:
            with self._lock:
                execution_id = self._generate_execution_id()
                
                record = PipelineExecutionRecord(
                    execution_id=execution_id,
                    pipeline_name=pipeline_name,
                    pipeline_version=pipeline_version,
                    status=ExecutionStatus.RUNNING,
                    start_time=datetime.now(),
                    trigger_type=trigger_type,
                    triggered_by=triggered_by,
                    config_snapshot=config_snapshot or {},
                    tags=tags or []
                )
                
                # 保存到数据库
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO pipeline_executions
                        (execution_id, pipeline_name, pipeline_version, status,
                         start_time, trigger_type, triggered_by, config_snapshot, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.execution_id,
                        record.pipeline_name,
                        record.pipeline_version,
                        record.status.value,
                        record.start_time,
                        record.trigger_type,
                        record.triggered_by,
                        json.dumps(record.config_snapshot),
                        json.dumps(record.tags)
                    ))
                    conn.commit()
                
                logger.info(f"管道执行开始: {pipeline_name} ({execution_id})")
                return execution_id
                
        except Exception as e:
            logger.error(f"开始管道执行记录失败: {e}")
            raise PipelineException(
                message=f"开始管道执行记录失败: {e}",
                error_code=PipelineErrorCode.UNKNOWN_ERROR,
                context={'pipeline_name': pipeline_name}
            )
    
    def end_pipeline_execution(
        self,
        execution_id: str,
        status: ExecutionStatus,
        global_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        结束管道执行记录
        
        Args:
            execution_id: 执行ID
            status: 执行状态
            global_metrics: 全局指标
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                end_time = datetime.now()
                
                # 获取开始时间
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT start_time FROM pipeline_executions
                        WHERE execution_id = ?
                    ''', (execution_id,))
                    row = cursor.fetchone()
                    
                    if not row:
                        logger.warning(f"执行记录不存在: {execution_id}")
                        return False
                    
                    start_time = datetime.fromisoformat(row[0])
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    # 更新记录
                    cursor.execute('''
                        UPDATE pipeline_executions SET
                        status = ?, end_time = ?, duration_seconds = ?, global_metrics = ?
                        WHERE execution_id = ?
                    ''', (
                        status.value,
                        end_time,
                        duration_seconds,
                        json.dumps(global_metrics or {}),
                        execution_id
                    ))
                    conn.commit()
                
                logger.info(f"管道执行结束: {execution_id} -> {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"结束管道执行记录失败: {e}")
            return False
    
    def start_stage_execution(
        self,
        pipeline_execution_id: str,
        stage_name: str,
        stage_type: str,
        input_data_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        开始阶段执行记录
        
        Args:
            pipeline_execution_id: 管道执行ID
            stage_name: 阶段名称
            stage_type: 阶段类型
            input_data_info: 输入数据信息
            
        Returns:
            阶段执行ID
        """
        try:
            with self._lock:
                stage_id = self._generate_stage_id()
                
                record = StageExecutionRecord(
                    stage_id=stage_id,
                    pipeline_execution_id=pipeline_execution_id,
                    stage_name=stage_name,
                    stage_type=stage_type,
                    status=StageStatus.RUNNING,
                    start_time=datetime.now(),
                    input_data_info=input_data_info or {}
                )
                
                # 保存到数据库
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO stage_executions
                        (stage_id, pipeline_execution_id, stage_name, stage_type,
                         status, start_time, input_data_info)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.stage_id,
                        record.pipeline_execution_id,
                        record.stage_name,
                        record.stage_type,
                        record.status.value,
                        record.start_time,
                        json.dumps(record.input_data_info)
                    ))
                    conn.commit()
                
                logger.debug(f"阶段执行开始: {stage_name} ({stage_id})")
                return stage_id
                
        except Exception as e:
            logger.error(f"开始阶段执行记录失败: {e}")
            raise PipelineException(
                message=f"开始阶段执行记录失败: {e}",
                error_code=PipelineErrorCode.UNKNOWN_ERROR,
                context={'stage_name': stage_name}
            )
    
    def end_stage_execution(
        self,
        stage_id: str,
        status: StageStatus,
        output_data_info: Optional[Dict[str, Any]] = None,
        error_info: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        结束阶段执行记录
        
        Args:
            stage_id: 阶段执行ID
            status: 执行状态
            output_data_info: 输出数据信息
            error_info: 错误信息
            metrics: 阶段指标
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                end_time = datetime.now()
                
                # 获取开始时间
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT start_time FROM stage_executions
                        WHERE stage_id = ?
                    ''', (stage_id,))
                    row = cursor.fetchone()
                    
                    if not row:
                        logger.warning(f"阶段执行记录不存在: {stage_id}")
                        return False
                    
                    start_time = datetime.fromisoformat(row[0])
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    # 更新记录
                    cursor.execute('''
                        UPDATE stage_executions SET
                        status = ?, end_time = ?, duration_seconds = ?,
                        output_data_info = ?, error_info = ?, metrics = ?
                        WHERE stage_id = ?
                    ''', (
                        status.value,
                        end_time,
                        duration_seconds,
                        json.dumps(output_data_info or {}),
                        error_info,
                        json.dumps(metrics or {}),
                        stage_id
                    ))
                    conn.commit()
                
                logger.debug(f"阶段执行结束: {stage_id} -> {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"结束阶段执行记录失败: {e}")
            return False
    
    def add_stage_log(self, stage_id: str, log_message: str) -> bool:
        """
        添加阶段执行日志
        
        Args:
            stage_id: 阶段执行ID
            log_message: 日志消息
            
        Returns:
            是否成功
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 获取现有日志
                cursor.execute('''
                    SELECT logs FROM stage_executions WHERE stage_id = ?
                ''', (stage_id,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    logs = json.loads(row[0])
                else:
                    logs = []
                
                # 添加新日志
                logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': log_message
                })
                
                # 更新日志
                cursor.execute('''
                    UPDATE stage_executions SET logs = ? WHERE stage_id = ?
                ''', (json.dumps(logs), stage_id))
                conn.commit()
                
                return True
        except Exception as e:
            logger.error(f"添加阶段日志失败: {e}")
            return False
    
    def get_pipeline_execution(self, execution_id: str) -> Optional[PipelineExecutionRecord]:
        """
        获取管道执行记录
        
        Args:
            execution_id: 执行ID
            
        Returns:
            执行记录或None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM pipeline_executions WHERE execution_id = ?
                ''', (execution_id,))
                row = cursor.fetchone()
                
                if row:
                    return PipelineExecutionRecord(
                        execution_id=row[0],
                        pipeline_name=row[1],
                        pipeline_version=row[2],
                        status=ExecutionStatus(row[3]),
                        start_time=datetime.fromisoformat(row[4]) if row[4] else None,
                        end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                        duration_seconds=row[6],
                        trigger_type=row[7],
                        triggered_by=row[8],
                        config_snapshot=json.loads(row[9]) if row[9] else {},
                        global_metrics=json.loads(row[10]) if row[10] else {},
                        tags=json.loads(row[11]) if row[11] else []
                    )
                return None
        except Exception as e:
            logger.error(f"获取管道执行记录失败: {e}")
            return None
    
    def get_stage_executions(
        self,
        pipeline_execution_id: str
    ) -> List[StageExecutionRecord]:
        """
        获取阶段执行记录
        
        Args:
            pipeline_execution_id: 管道执行ID
            
        Returns:
            阶段执行记录列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM stage_executions
                    WHERE pipeline_execution_id = ?
                    ORDER BY start_time
                ''', (pipeline_execution_id,))
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    records.append(StageExecutionRecord(
                        stage_id=row[0],
                        pipeline_execution_id=row[1],
                        stage_name=row[2],
                        stage_type=row[3],
                        status=StageStatus(row[4]),
                        start_time=datetime.fromisoformat(row[5]) if row[5] else None,
                        end_time=datetime.fromisoformat(row[6]) if row[6] else None,
                        duration_seconds=row[7],
                        input_data_info=json.loads(row[8]) if row[8] else {},
                        output_data_info=json.loads(row[9]) if row[9] else {},
                        error_info=row[10],
                        metrics=json.loads(row[11]) if row[11] else {},
                        logs=json.loads(row[12]) if row[12] else []
                    ))
                
                return records
        except Exception as e:
            logger.error(f"获取阶段执行记录失败: {e}")
            return []
    
    def list_pipeline_executions(
        self,
        pipeline_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PipelineExecutionRecord]:
        """
        列出管道执行记录
        
        Args:
            pipeline_name: 管道名称过滤
            status: 状态过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            limit: 返回记录数限制
            
        Returns:
            执行记录列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM pipeline_executions WHERE 1=1"
                params = []
                
                if pipeline_name:
                    query += " AND pipeline_name = ?"
                    params.append(pipeline_name)
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                if start_time:
                    query += " AND start_time >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND start_time <= ?"
                    params.append(end_time)
                
                query += " ORDER BY start_time DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    records.append(PipelineExecutionRecord(
                        execution_id=row[0],
                        pipeline_name=row[1],
                        pipeline_version=row[2],
                        status=ExecutionStatus(row[3]),
                        start_time=datetime.fromisoformat(row[4]) if row[4] else None,
                        end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                        duration_seconds=row[6],
                        trigger_type=row[7],
                        triggered_by=row[8],
                        config_snapshot=json.loads(row[9]) if row[9] else {},
                        global_metrics=json.loads(row[10]) if row[10] else {},
                        tags=json.loads(row[11]) if row[11] else []
                    ))
                
                return records
        except Exception as e:
            logger.error(f"列出管道执行记录失败: {e}")
            return []
    
    def get_execution_summary(
        self,
        pipeline_name: Optional[str] = None,
        days: int = 7
    ) -> ExecutionSummary:
        """
        获取执行摘要统计
        
        Args:
            pipeline_name: 管道名称过滤
            days: 统计天数
            
        Returns:
            执行摘要
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 计算时间范围
                start_date = datetime.now() - timedelta(days=days)
                
                query = '''
                    SELECT status, duration_seconds 
                    FROM pipeline_executions 
                    WHERE start_time >= ?
                '''
                params = [start_date]
                
                if pipeline_name:
                    query += " AND pipeline_name = ?"
                    params.append(pipeline_name)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return ExecutionSummary()
                
                # 统计
                total = len(rows)
                successful = sum(1 for r in rows if r[0] == 'success')
                failed = sum(1 for r in rows if r[0] == 'failed')
                cancelled = sum(1 for r in rows if r[0] == 'cancelled')
                
                durations = [r[1] for r in rows if r[1] > 0]
                avg_duration = sum(durations) / len(durations) if durations else 0.0
                min_duration = min(durations) if durations else 0.0
                max_duration = max(durations) if durations else 0.0
                
                success_rate = successful / total if total > 0 else 0.0
                
                return ExecutionSummary(
                    total_executions=total,
                    successful_count=successful,
                    failed_count=failed,
                    cancelled_count=cancelled,
                    avg_duration_seconds=avg_duration,
                    min_duration_seconds=min_duration,
                    max_duration_seconds=max_duration,
                    success_rate=success_rate
                )
        except Exception as e:
            logger.error(f"获取执行摘要失败: {e}")
            return ExecutionSummary()
    
    def get_stage_statistics(
        self, pipeline_name: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """
        获取阶段统计信息
        
        Args:
            pipeline_name: 管道名称过滤
            days: 统计天数
            
        Returns:
            阶段统计字典
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                if pipeline_name:
                    cursor.execute('''
                        SELECT se.stage_name, se.status, se.duration_seconds
                        FROM stage_executions se
                        JOIN pipeline_executions pe 
                            ON se.pipeline_execution_id = pe.execution_id
                        WHERE pe.pipeline_name = ? AND se.start_time >= ?
                    ''', (pipeline_name, start_date))
                else:
                    cursor.execute('''
                        SELECT stage_name, status, duration_seconds
                        FROM stage_executions
                        WHERE start_time >= ?
                    ''', (start_date,))
                
                rows = cursor.fetchall()
                
                # 按阶段统计
                stage_stats = {}
                for row in rows:
                    stage_name, status, duration = row
                    if stage_name not in stage_stats:
                        stage_stats[stage_name] = {
                            'total': 0,
                            'successful': 0,
                            'failed': 0,
                            'durations': []
                        }
                    
                    stage_stats[stage_name]['total'] += 1
                    if status == 'success':
                        stage_stats[stage_name]['successful'] += 1
                    elif status == 'failed':
                        stage_stats[stage_name]['failed'] += 1
                    
                    if duration > 0:
                        stage_stats[stage_name]['durations'].append(duration)
                
                # 计算平均值
                for name, stats in stage_stats.items():
                    durations = stats['durations']
                    stats['avg_duration'] = sum(durations) / len(durations) if durations else 0.0
                    stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
                    del stats['durations']
                
                return stage_stats
        except Exception as e:
            logger.error(f"获取阶段统计失败: {e}")
            return {}
    
    def delete_old_executions(self, days: Optional[int] = None) -> int:
        """
        删除旧执行记录
        
        Args:
            days: 保留天数，为None时使用配置值
            
        Returns:
            删除的记录数
        """
        try:
            retention_days = days or self.config.max_history_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 删除阶段记录
                cursor.execute('''
                    DELETE FROM stage_executions
                    WHERE pipeline_execution_id IN (
                        SELECT execution_id FROM pipeline_executions
                        WHERE start_time < ?
                    )
                ''', (cutoff_date,))
                
                # 删除管道记录
                cursor.execute('''
                    DELETE FROM pipeline_executions
                    WHERE start_time < ?
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"删除旧执行记录: {deleted_count} 条")
                return deleted_count
        except Exception as e:
            logger.error(f"删除旧执行记录失败: {e}")
            return 0
    
    def get_performance_trends(
        self,
        pipeline_name: str,
        metric_name: str,
        days: int = 30
    ) -> List[Tuple[datetime, float]]:
        """
        获取性能趋势
        
        Args:
            pipeline_name: 管道名称
            metric_name: 指标名称
            days: 统计天数
            
        Returns:
            (时间, 指标值) 列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                cursor.execute('''
                    SELECT start_time, global_metrics
                    FROM pipeline_executions
                    WHERE pipeline_name = ? AND status = 'success'
                        AND start_time >= ?
                    ORDER BY start_time
                ''', (pipeline_name, start_date))
                
                rows = cursor.fetchall()
                
                trends = []
                for row in rows:
                    timestamp = datetime.fromisoformat(row[0])
                    metrics = json.loads(row[1]) if row[1] else {}
                    if metric_name in metrics:
                        trends.append((timestamp, metrics[metric_name]))
                
                return trends
        except Exception as e:
            logger.error(f"获取性能趋势失败: {e}")
            return []
    
    def export_execution_history(
        self,
        output_path: Union[str, Path],
        pipeline_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        导出执行历史
        
        Args:
            output_path: 输出路径
            pipeline_name: 管道名称过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            
        Returns:
            是否成功
        """
        try:
            executions = self.list_pipeline_executions(
                pipeline_name=pipeline_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'executions': [e.to_dict() for e in executions]
            }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"执行历史导出成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出执行历史失败: {e}")
            return False
    
    def close(self) -> None:
        """关闭存储"""
        logger.info("元数据存储已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

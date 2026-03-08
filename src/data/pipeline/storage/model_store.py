"""
模型存储模块

提供模型版本管理、元数据存储、部署状态跟踪、性能指标记录和回滚历史管理。
支持模型全生命周期管理，包括训练、验证、部署和回滚。
"""

import logging
import sqlite3
import json
import hashlib
import pickle
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from threading import Lock
from contextlib import contextmanager

import numpy as np

from ..exceptions import PipelineException, PipelineErrorCode

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    CANARY = "canary"
    ROLLBACKED = "rollbacked"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStrategy(Enum):
    """部署策略"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class PerformanceMetrics:
    """
    模型性能指标
    
    Attributes:
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
        auc_roc: AUC-ROC
        mse: 均方误差
        rmse: 均方根误差
        mae: 平均绝对误差
        sharpe_ratio: 夏普比率
        max_drawdown: 最大回撤
        custom_metrics: 自定义指标
    """
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'custom_metrics': self.custom_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """从字典创建"""
        custom = data.get('custom_metrics', {})
        return cls(
            accuracy=data.get('accuracy'),
            precision=data.get('precision'),
            recall=data.get('recall'),
            f1_score=data.get('f1_score'),
            auc_roc=data.get('auc_roc'),
            mse=data.get('mse'),
            rmse=data.get('rmse'),
            mae=data.get('mae'),
            sharpe_ratio=data.get('sharpe_ratio'),
            max_drawdown=data.get('max_drawdown'),
            custom_metrics=custom
        )


@dataclass
class DeploymentInfo:
    """
    部署信息
    
    Attributes:
        deployment_id: 部署ID
        model_id: 模型ID
        strategy: 部署策略
        status: 部署状态
        deployed_at: 部署时间
        traffic_percentage: 流量百分比（金丝雀部署）
        target_environment: 目标环境
        deployed_by: 部署者
        deployment_config: 部署配置
    """
    deployment_id: str
    model_id: str
    strategy: DeploymentStrategy
    status: ModelStatus
    deployed_at: datetime = field(default_factory=datetime.now)
    traffic_percentage: float = 100.0
    target_environment: str = "production"
    deployed_by: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'strategy': self.strategy.value,
            'status': self.status.value,
            'deployed_at': self.deployed_at.isoformat(),
            'traffic_percentage': self.traffic_percentage,
            'target_environment': self.target_environment,
            'deployed_by': self.deployed_by,
            'deployment_config': self.deployment_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentInfo':
        """从字典创建"""
        return cls(
            deployment_id=data['deployment_id'],
            model_id=data['model_id'],
            strategy=DeploymentStrategy(data.get('strategy', 'blue_green')),
            status=ModelStatus(data.get('status', 'deployed')),
            deployed_at=datetime.fromisoformat(data['deployed_at']),
            traffic_percentage=data.get('traffic_percentage', 100.0),
            target_environment=data.get('target_environment', 'production'),
            deployed_by=data.get('deployed_by'),
            deployment_config=data.get('deployment_config', {})
        )


@dataclass
class RollbackRecord:
    """
    回滚记录
    
    Attributes:
        rollback_id: 回滚ID
        from_model_id: 源模型ID
        to_model_id: 目标模型ID
        reason: 回滚原因
        rolled_back_at: 回滚时间
        rolled_back_by: 回滚执行者
        trigger_metrics: 触发回滚的指标
        is_successful: 是否成功
    """
    rollback_id: str
    from_model_id: str
    to_model_id: str
    reason: str
    rolled_back_at: datetime = field(default_factory=datetime.now)
    rolled_back_by: Optional[str] = None
    trigger_metrics: Dict[str, Any] = field(default_factory=dict)
    is_successful: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rollback_id': self.rollback_id,
            'from_model_id': self.from_model_id,
            'to_model_id': self.to_model_id,
            'reason': self.reason,
            'rolled_back_at': self.rolled_back_at.isoformat(),
            'rolled_back_by': self.rolled_back_by,
            'trigger_metrics': self.trigger_metrics,
            'is_successful': self.is_successful
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackRecord':
        """从字典创建"""
        return cls(
            rollback_id=data['rollback_id'],
            from_model_id=data['from_model_id'],
            to_model_id=data['to_model_id'],
            reason=data['reason'],
            rolled_back_at=datetime.fromisoformat(data['rolled_back_at']),
            rolled_back_by=data.get('rolled_back_by'),
            trigger_metrics=data.get('trigger_metrics', {}),
            is_successful=data.get('is_successful', True)
        )


@dataclass
class ModelMetadata:
    """
    模型元数据
    
    Attributes:
        model_id: 模型ID
        model_name: 模型名称
        version: 版本号
        model_type: 模型类型
        status: 模型状态
        created_at: 创建时间
        updated_at: 更新时间
        author: 创建者
        description: 描述
        tags: 标签列表
        hyperparameters: 超参数
        feature_columns: 特征列
        target_column: 目标列
        training_data_info: 训练数据信息
        model_size_bytes: 模型大小
        checksum: 模型校验和
    """
    model_id: str
    model_name: str
    version: str
    model_type: str
    status: ModelStatus = ModelStatus.TRAINING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    model_size_bytes: int = 0
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'model_type': self.model_type,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author,
            'description': self.description,
            'tags': self.tags,
            'hyperparameters': self.hyperparameters,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_data_info': self.training_data_info,
            'model_size_bytes': self.model_size_bytes,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        return cls(
            model_id=data['model_id'],
            model_name=data['model_name'],
            version=data['version'],
            model_type=data['model_type'],
            status=ModelStatus(data.get('status', 'training')),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            author=data.get('author'),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            hyperparameters=data.get('hyperparameters', {}),
            feature_columns=data.get('feature_columns', []),
            target_column=data.get('target_column'),
            training_data_info=data.get('training_data_info', {}),
            model_size_bytes=data.get('model_size_bytes', 0),
            checksum=data.get('checksum', '')
        )


@dataclass
class ModelStoreConfig:
    """
    模型存储配置
    
    Attributes:
        storage_path: 存储路径
        max_models: 最大模型数量
        compression: 是否启用压缩
        enable_versioning: 是否启用版本控制
        retention_days: 保留天数
    """
    storage_path: str = "./model_store"
    max_models: int = 100
    compression: bool = True
    enable_versioning: bool = True
    retention_days: int = 90


class ModelStore:
    """
    模型存储管理器
    
    提供模型版本管理、部署状态跟踪、性能指标记录和回滚历史管理。
    支持模型全生命周期管理。
    
    Attributes:
        config: 存储配置
        _lock: 线程锁
    """
    
    def __init__(self, config: Optional[ModelStoreConfig] = None):
        """
        初始化模型存储
        
        Args:
            config: 存储配置，为None时使用默认配置
        """
        self.config = config or ModelStoreConfig()
        self._lock = Lock()
        
        # 初始化存储路径
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._models_path = self._storage_path / "models"
        self._metadata_path = self._storage_path / "metadata"
        self._deployments_path = self._storage_path / "deployments"
        self._rollbacks_path = self._storage_path / "rollbacks"
        
        for path in [self._models_path, self._metadata_path, 
                     self._deployments_path, self._rollbacks_path]:
            path.mkdir(exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info("模型存储初始化完成")
    
    def _init_database(self) -> None:
        """初始化SQLite数据库"""
        self._db_path = self._storage_path / "model_store.db"
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 创建模型元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT,
                    status TEXT DEFAULT 'training',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    author TEXT,
                    description TEXT,
                    tags TEXT,
                    hyperparameters TEXT,
                    feature_columns TEXT,
                    target_column TEXT,
                    training_data_info TEXT,
                    model_size_bytes INTEGER,
                    checksum TEXT,
                    UNIQUE(model_name, version)
                )
            ''')
            
            # 创建性能指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    custom_metrics TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')
            
            # 创建部署信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    strategy TEXT,
                    status TEXT,
                    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    traffic_percentage REAL DEFAULT 100.0,
                    target_environment TEXT DEFAULT 'production',
                    deployed_by TEXT,
                    deployment_config TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            ''')
            
            # 创建回滚历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rollback_history (
                    rollback_id TEXT PRIMARY KEY,
                    from_model_id TEXT NOT NULL,
                    to_model_id TEXT NOT NULL,
                    reason TEXT,
                    rolled_back_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rolled_back_by TEXT,
                    trigger_metrics TEXT,
                    is_successful INTEGER DEFAULT 1,
                    FOREIGN KEY (from_model_id) REFERENCES models(model_id),
                    FOREIGN KEY (to_model_id) REFERENCES models(model_id)
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_name ON models(model_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_status ON models(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_deployment_model ON deployments(model_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_rollback_from ON rollback_history(from_model_id)
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
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """生成模型ID"""
        content = f"{model_name}:{version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, model_data: bytes) -> str:
        """计算模型校验和"""
        return hashlib.sha256(model_data).hexdigest()[:16]
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_type: str,
        model_object: Any,
        author: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        training_data_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        注册模型
        
        Args:
            model_name: 模型名称
            version: 版本号
            model_type: 模型类型
            model_object: 模型对象
            author: 创建者
            description: 描述
            tags: 标签列表
            hyperparameters: 超参数
            feature_columns: 特征列
            target_column: 目标列
            training_data_info: 训练数据信息
            
        Returns:
            模型ID
        """
        try:
            with self._lock:
                model_id = self._generate_model_id(model_name, version)
                
                # 序列化模型
                model_bytes = pickle.dumps(model_object)
                if self.config.compression:
                    import gzip
                    model_bytes = gzip.compress(model_bytes)
                
                # 计算校验和
                checksum = self._calculate_checksum(model_bytes)
                
                # 保存模型文件
                model_file = self._models_path / f"{model_id}.pkl"
                with open(model_file, 'wb') as f:
                    f.write(model_bytes)
                
                # 创建元数据
                metadata = ModelMetadata(
                    model_id=model_id,
                    model_name=model_name,
                    version=version,
                    model_type=model_type,
                    status=ModelStatus.TRAINED,
                    author=author,
                    description=description,
                    tags=tags or [],
                    hyperparameters=hyperparameters or {},
                    feature_columns=feature_columns or [],
                    target_column=target_column,
                    training_data_info=training_data_info or {},
                    model_size_bytes=len(model_bytes),
                    checksum=checksum
                )
                
                # 保存到数据库
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO models
                        (model_id, model_name, version, model_type, status, author,
                         description, tags, hyperparameters, feature_columns,
                         target_column, training_data_info, model_size_bytes, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.model_id, metadata.model_name, metadata.version,
                        metadata.model_type, metadata.status.value, metadata.author,
                        metadata.description, json.dumps(metadata.tags),
                        json.dumps(metadata.hyperparameters),
                        json.dumps(metadata.feature_columns), metadata.target_column,
                        json.dumps(metadata.training_data_info),
                        metadata.model_size_bytes, metadata.checksum
                    ))
                    conn.commit()
                
                # 保存元数据文件
                metadata_file = self._metadata_path / f"{model_id}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                logger.info(f"模型注册成功: {model_name}@{version}")
                return model_id
                
        except Exception as e:
            logger.error(f"注册模型失败: {e}")
            raise PipelineException(
                message=f"注册模型失败: {e}",
                error_code=PipelineErrorCode.MODEL_SAVE_ERROR,
                context={'model_name': model_name, 'version': version}
            )
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        加载模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型对象或None
        """
        try:
            model_file = self._models_path / f"{model_id}.pkl"
            if not model_file.exists():
                logger.warning(f"模型文件不存在: {model_id}")
                return None
            
            with open(model_file, 'rb') as f:
                model_bytes = f.read()
            
            # 解压
            if self.config.compression:
                import gzip
                model_bytes = gzip.decompress(model_bytes)
            
            model = pickle.loads(model_bytes)
            logger.info(f"模型加载成功: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        获取模型元数据
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型元数据或None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM models WHERE model_id = ?
                ''', (model_id,))
                row = cursor.fetchone()
                
                if row:
                    return ModelMetadata(
                        model_id=row[0],
                        model_name=row[1],
                        version=row[2],
                        model_type=row[3],
                        status=ModelStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        updated_at=datetime.fromisoformat(row[6]),
                        author=row[7],
                        description=row[8],
                        tags=json.loads(row[9]),
                        hyperparameters=json.loads(row[10]),
                        feature_columns=json.loads(row[11]),
                        target_column=row[12],
                        training_data_info=json.loads(row[13]),
                        model_size_bytes=row[14],
                        checksum=row[15]
                    )
                return None
        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            return None
    
    def update_status(self, model_id: str, status: ModelStatus) -> bool:
        """
        更新模型状态
        
        Args:
            model_id: 模型ID
            status: 新状态
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE models SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE model_id = ?
                    ''', (status.value, model_id))
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"模型状态更新: {model_id} -> {status.value}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"更新模型状态失败: {e}")
            return False
    
    def record_metrics(
        self,
        model_id: str,
        metrics: PerformanceMetrics
    ) -> bool:
        """
        记录性能指标
        
        Args:
            model_id: 模型ID
            metrics: 性能指标
            
        Returns:
            是否成功
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics
                    (model_id, accuracy, precision, recall, f1_score, auc_roc,
                     mse, rmse, mae, sharpe_ratio, max_drawdown, custom_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, metrics.accuracy, metrics.precision,
                    metrics.recall, metrics.f1_score, metrics.auc_roc,
                    metrics.mse, metrics.rmse, metrics.mae,
                    metrics.sharpe_ratio, metrics.max_drawdown,
                    json.dumps(metrics.custom_metrics)
                ))
                conn.commit()
                
                logger.info(f"性能指标记录成功: {model_id}")
                return True
        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
            return False
    
    def get_metrics(self, model_id: str) -> Optional[PerformanceMetrics]:
        """
        获取性能指标
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能指标或None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM performance_metrics
                    WHERE model_id = ?
                    ORDER BY recorded_at DESC LIMIT 1
                ''', (model_id,))
                row = cursor.fetchone()
                
                if row:
                    return PerformanceMetrics(
                        accuracy=row[2],
                        precision=row[3],
                        recall=row[4],
                        f1_score=row[5],
                        auc_roc=row[6],
                        mse=row[7],
                        rmse=row[8],
                        mae=row[9],
                        sharpe_ratio=row[10],
                        max_drawdown=row[11],
                        custom_metrics=json.loads(row[12]) if row[12] else {}
                    )
                return None
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return None
    
    def deploy_model(
        self,
        model_id: str,
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        traffic_percentage: float = 100.0,
        target_environment: str = "production",
        deployed_by: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        部署模型
        
        Args:
            model_id: 模型ID
            strategy: 部署策略
            traffic_percentage: 流量百分比
            target_environment: 目标环境
            deployed_by: 部署者
            deployment_config: 部署配置
            
        Returns:
            部署ID或None
        """
        try:
            with self._lock:
                # 生成部署ID
                deployment_id = hashlib.sha256(
                    f"{model_id}:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
                
                # 确定部署状态
                if strategy == DeploymentStrategy.CANARY:
                    status = ModelStatus.CANARY
                else:
                    status = ModelStatus.DEPLOYED
                
                # 创建部署信息
                deployment = DeploymentInfo(
                    deployment_id=deployment_id,
                    model_id=model_id,
                    strategy=strategy,
                    status=status,
                    traffic_percentage=traffic_percentage,
                    target_environment=target_environment,
                    deployed_by=deployed_by,
                    deployment_config=deployment_config or {}
                )
                
                # 保存到数据库
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO deployments
                        (deployment_id, model_id, strategy, status, traffic_percentage,
                         target_environment, deployed_by, deployment_config)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        deployment.deployment_id, deployment.model_id,
                        deployment.strategy.value, deployment.status.value,
                        deployment.traffic_percentage, deployment.target_environment,
                        deployment.deployed_by, json.dumps(deployment.deployment_config)
                    ))
                    conn.commit()
                
                # 更新模型状态
                self.update_status(model_id, status)
                
                # 保存部署文件
                deployment_file = self._deployments_path / f"{deployment_id}.json"
                with open(deployment_file, 'w', encoding='utf-8') as f:
                    json.dump(deployment.to_dict(), f, indent=2)
                
                logger.info(f"模型部署成功: {model_id} -> {target_environment}")
                return deployment_id
                
        except Exception as e:
            logger.error(f"部署模型失败: {e}")
            return None
    
    def get_deployment(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """
        获取部署信息
        
        Args:
            deployment_id: 部署ID
            
        Returns:
            部署信息或None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM deployments WHERE deployment_id = ?
                ''', (deployment_id,))
                row = cursor.fetchone()
                
                if row:
                    return DeploymentInfo(
                        deployment_id=row[0],
                        model_id=row[1],
                        strategy=DeploymentStrategy(row[2]),
                        status=ModelStatus(row[3]),
                        deployed_at=datetime.fromisoformat(row[4]),
                        traffic_percentage=row[5],
                        target_environment=row[6],
                        deployed_by=row[7],
                        deployment_config=json.loads(row[8]) if row[8] else {}
                    )
                return None
        except Exception as e:
            logger.error(f"获取部署信息失败: {e}")
            return None
    
    def get_active_deployment(self, model_name: str) -> Optional[DeploymentInfo]:
        """
        获取模型的活跃部署
        
        Args:
            model_name: 模型名称
            
        Returns:
            部署信息或None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT d.* FROM deployments d
                    JOIN models m ON d.model_id = m.model_id
                    WHERE m.model_name = ? AND d.status IN ('deployed', 'canary')
                    ORDER BY d.deployed_at DESC LIMIT 1
                ''', (model_name,))
                row = cursor.fetchone()
                
                if row:
                    return DeploymentInfo(
                        deployment_id=row[0],
                        model_id=row[1],
                        strategy=DeploymentStrategy(row[2]),
                        status=ModelStatus(row[3]),
                        deployed_at=datetime.fromisoformat(row[4]),
                        traffic_percentage=row[5],
                        target_environment=row[6],
                        deployed_by=row[7],
                        deployment_config=json.loads(row[8]) if row[8] else {}
                    )
                return None
        except Exception as e:
            logger.error(f"获取活跃部署失败: {e}")
            return None
    
    def record_rollback(
        self,
        from_model_id: str,
        to_model_id: str,
        reason: str,
        rolled_back_by: Optional[str] = None,
        trigger_metrics: Optional[Dict[str, Any]] = None,
        is_successful: bool = True
    ) -> Optional[str]:
        """
        记录回滚
        
        Args:
            from_model_id: 源模型ID
            to_model_id: 目标模型ID
            reason: 回滚原因
            rolled_back_by: 回滚执行者
            trigger_metrics: 触发回滚的指标
            is_successful: 是否成功
            
        Returns:
            回滚ID或None
        """
        try:
            with self._lock:
                # 生成回滚ID
                rollback_id = hashlib.sha256(
                    f"{from_model_id}:{to_model_id}:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
                
                # 创建回滚记录
                rollback = RollbackRecord(
                    rollback_id=rollback_id,
                    from_model_id=from_model_id,
                    to_model_id=to_model_id,
                    reason=reason,
                    rolled_back_by=rolled_back_by,
                    trigger_metrics=trigger_metrics or {},
                    is_successful=is_successful
                )
                
                # 保存到数据库
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO rollback_history
                        (rollback_id, from_model_id, to_model_id, reason,
                         rolled_back_by, trigger_metrics, is_successful)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rollback.rollback_id, rollback.from_model_id,
                        rollback.to_model_id, rollback.reason,
                        rollback.rolled_back_by, json.dumps(rollback.trigger_metrics),
                        int(rollback.is_successful)
                    ))
                    conn.commit()
                
                # 更新模型状态
                self.update_status(from_model_id, ModelStatus.ROLLBACKED)
                
                # 保存回滚文件
                rollback_file = self._rollbacks_path / f"{rollback_id}.json"
                with open(rollback_file, 'w', encoding='utf-8') as f:
                    json.dump(rollback.to_dict(), f, indent=2)
                
                logger.info(f"回滚记录成功: {from_model_id} -> {to_model_id}")
                return rollback_id
                
        except Exception as e:
            logger.error(f"记录回滚失败: {e}")
            return None
    
    def get_rollback_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 100
    ) -> List[RollbackRecord]:
        """
        获取回滚历史
        
        Args:
            model_id: 模型ID过滤
            limit: 返回记录数限制
            
        Returns:
            回滚记录列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if model_id:
                    cursor.execute('''
                        SELECT * FROM rollback_history
                        WHERE from_model_id = ? OR to_model_id = ?
                        ORDER BY rolled_back_at DESC LIMIT ?
                    ''', (model_id, model_id, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM rollback_history
                        ORDER BY rolled_back_at DESC LIMIT ?
                    ''', (limit,))
                
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    records.append(RollbackRecord(
                        rollback_id=row[0],
                        from_model_id=row[1],
                        to_model_id=row[2],
                        reason=row[3],
                        rolled_back_at=datetime.fromisoformat(row[4]),
                        rolled_back_by=row[5],
                        trigger_metrics=json.loads(row[6]) if row[6] else {},
                        is_successful=bool(row[7])
                    ))
                
                return records
        except Exception as e:
            logger.error(f"获取回滚历史失败: {e}")
            return []
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        列出模型
        
        Args:
            model_name: 模型名称过滤
            status: 状态过滤
            tags: 标签过滤
            
        Returns:
            模型元数据列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM models WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                models = []
                for row in rows:
                    model = ModelMetadata(
                        model_id=row[0],
                        model_name=row[1],
                        version=row[2],
                        model_type=row[3],
                        status=ModelStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        updated_at=datetime.fromisoformat(row[6]),
                        author=row[7],
                        description=row[8],
                        tags=json.loads(row[9]),
                        hyperparameters=json.loads(row[10]),
                        feature_columns=json.loads(row[11]),
                        target_column=row[12],
                        training_data_info=json.loads(row[13]),
                        model_size_bytes=row[14],
                        checksum=row[15]
                    )
                    
                    # 标签过滤
                    if tags and not any(tag in model.tags for tag in tags):
                        continue
                    
                    models.append(model)
                
                return models
        except Exception as e:
            logger.error(f"列出模型失败: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """
        删除模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                # 删除模型文件
                model_file = self._models_path / f"{model_id}.pkl"
                if model_file.exists():
                    model_file.unlink()
                
                # 删除元数据文件
                metadata_file = self._metadata_path / f"{model_id}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                # 从数据库删除
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
                    cursor.execute('DELETE FROM performance_metrics WHERE model_id = ?', (model_id,))
                    cursor.execute('DELETE FROM deployments WHERE model_id = ?', (model_id,))
                    conn.commit()
                
                logger.info(f"模型删除成功: {model_id}")
                return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        获取模型统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 总模型数
                cursor.execute('SELECT COUNT(*) FROM models')
                total_models = cursor.fetchone()[0]
                
                # 按状态统计
                cursor.execute('''
                    SELECT status, COUNT(*) FROM models GROUP BY status
                ''')
                status_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                # 总存储大小
                cursor.execute('SELECT SUM(model_size_bytes) FROM models')
                total_size = cursor.fetchone()[0] or 0
                
                # 部署统计
                cursor.execute('SELECT COUNT(*) FROM deployments')
                total_deployments = cursor.fetchone()[0]
                
                # 回滚统计
                cursor.execute('SELECT COUNT(*) FROM rollback_history')
                total_rollbacks = cursor.fetchone()[0]
                
                return {
                    'total_models': total_models,
                    'status_distribution': status_counts,
                    'total_size_bytes': total_size,
                    'total_size_mb': total_size / (1024 * 1024),
                    'total_deployments': total_deployments,
                    'total_rollbacks': total_rollbacks
                }
        except Exception as e:
            logger.error(f"获取模型统计失败: {e}")
            return {}
    
    def close(self) -> None:
        """关闭存储"""
        logger.info("模型存储已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

"""
管道存储模块

提供特征存储、模型存储和元数据存储功能，支持ML管道的完整数据管理。
"""

from .feature_store import (
    FeatureStore,
    FeatureStoreConfig,
    FeatureVersion,
    FeatureMetadata,
    FeatureStatus,
    StorageBackend
)

from .model_store import (
    ModelStore,
    ModelStoreConfig,
    ModelMetadata,
    ModelStatus,
    DeploymentStrategy,
    DeploymentInfo,
    PerformanceMetrics,
    RollbackRecord
)

from .metadata_store import (
    MetadataStore,
    MetadataStoreConfig,
    PipelineExecutionRecord,
    StageExecutionRecord,
    ExecutionStatus,
    StageStatus,
    ExecutionSummary
)

__all__ = [
    # Feature Store
    'FeatureStore',
    'FeatureStoreConfig',
    'FeatureVersion',
    'FeatureMetadata',
    'FeatureStatus',
    'StorageBackend',
    
    # Model Store
    'ModelStore',
    'ModelStoreConfig',
    'ModelMetadata',
    'ModelStatus',
    'DeploymentStrategy',
    'DeploymentInfo',
    'PerformanceMetrics',
    'RollbackRecord',
    
    # Metadata Store
    'MetadataStore',
    'MetadataStoreConfig',
    'PipelineExecutionRecord',
    'StageExecutionRecord',
    'ExecutionStatus',
    'StageStatus',
    'ExecutionSummary'
]

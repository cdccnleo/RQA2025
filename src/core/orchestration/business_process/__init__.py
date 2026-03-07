"""
业务流程编排模块

包含：
- 数据采集业务流程编排器
- 服务调度器
- 监控和告警
- 服务发现和治理
"""

from .data_collection_orchestrator import DataCollectionWorkflow, DataCollectionState, DataCollectionEvent

# 可选导入：service_scheduler（如果模块导入失败则跳过）
try:
    from .service_scheduler import (
        DataCollectionServiceScheduler,
        get_data_collection_scheduler,
        start_data_collection_scheduler,
        stop_data_collection_scheduler
    )
    SERVICE_SCHEDULER_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"service_scheduler模块导入失败: {e}，跳过相关功能")
    SERVICE_SCHEDULER_AVAILABLE = False
    # 提供占位符避免NameError
    DataCollectionServiceScheduler = None
    get_data_collection_scheduler = None
    start_data_collection_scheduler = None
    stop_data_collection_scheduler = None

from .monitoring_alerts import AlertManager, AlertLevel, DataCollectionMonitor
from .service_discovery import get_service_discovery, ServiceDiscovery

__all__ = [
    'DataCollectionWorkflow',
    'DataCollectionState',
    'DataCollectionEvent',
    'DataCollectionServiceScheduler',
    'get_data_collection_scheduler',
    'start_data_collection_scheduler',
    'stop_data_collection_scheduler',
    'AlertManager',
    'AlertLevel',
    'DataCollectionMonitor',
    'get_service_discovery',
    'ServiceDiscovery',
]

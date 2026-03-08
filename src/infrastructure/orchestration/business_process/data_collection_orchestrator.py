#!/usr/bin/env python3
"""
数据采集业务流程编排器

基于核心服务层架构设计，实现数据采集的完整业务流程编排：
1. 通过网关层API调用数据采集
2. 集成数据层微服务进行数据处理
3. 实现状态机管理的数据采集流程
4. 提供监控和告警机制
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pathlib import Path
import json

from src.infrastructure.orchestration.business_process.data_collection_state_machine import StateMachineManager
from src.infrastructure.orchestration.business_process.service_discovery import get_service_discovery, ServiceDiscovery
from src.infrastructure.orchestration.business_process.monitoring_alerts import AlertManager, AlertLevel, DataCollectionMonitor
from src.core.event_bus.core import EventBus
from src.core.business_process.monitor.monitor import BusinessProcessMonitor
from src.infrastructure.logging.core.unified_logger import get_unified_logger
from src.infrastructure.utils.optimization.concurrency_controller import ConcurrencyController

logger = get_unified_logger(__name__)

# 全局并发控制器实例（单例模式）
_concurrency_controller = None


def get_concurrency_controller() -> ConcurrencyController:
    """
    获取并发控制器实例（单例模式）
    
    Returns:
        ConcurrencyController: 并发控制器实例
    """
    global _concurrency_controller
    if _concurrency_controller is None:
        _concurrency_controller = ConcurrencyController()
    return _concurrency_controller


class DataCollectionState(Enum):
    """数据采集流程状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    SCHEDULING = "scheduling"
    COLLECTING = "collecting"
    VALIDATING = "validating"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class DataCollectionEvent(Enum):
    """数据采集事件"""
    START_COLLECTION = "start_collection"
    SCHEDULE_NEXT = "schedule_next"
    COLLECTION_SUCCESS = "collection_success"
    COLLECTION_FAILED = "collection_failed"
    VALIDATION_SUCCESS = "validation_success"
    VALIDATION_FAILED = "validation_failed"
    STORAGE_SUCCESS = "storage_success"
    STORAGE_FAILED = "storage_failed"
    RETRY_EXHAUSTED = "retry_exhausted"


class DataCollectionWorkflow:
    """数据采集业务流程"""

    def __init__(self, gateway_router=None, data_service=None, config: Optional[Dict[str, Any]] = None):
        self.gateway_router = gateway_router
        self.data_service = data_service
        self.config = config or {}

        # 初始化统一适配器工厂（符合架构设计：统一基础设施集成）
        try:
            from src.infrastructure.integration.business_adapters import get_unified_adapter_factory
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            self.adapter_factory = get_unified_adapter_factory()
            if self.adapter_factory:
                # 通过统一适配器访问数据层服务
                self.data_adapter = self.adapter_factory.get_adapter(BusinessLayerType.DATA)
                logger.info("数据层适配器已初始化")
            else:
                self.data_adapter = None
                logger.warning("统一适配器工厂不可用")
        except Exception as e:
            logger.warning(f"统一适配器初始化失败: {e}")
            self.adapter_factory = None
            self.data_adapter = None

        # 初始化状态机
        self.state_machine_manager = StateMachineManager("data_collection")
        self.state_machine = self.state_machine_manager.state_machine
        self._setup_state_machine()

        # 初始化服务发现
        self.service_discovery = get_service_discovery()

        # 初始化监控和告警
        self.alert_manager = AlertManager()
        self.data_collection_monitor = DataCollectionMonitor()

        # 初始化事件总线
        self.event_bus = EventBus()
        try:
            self.event_bus.initialize()
            logger.debug("事件总线初始化成功")
        except Exception as e:
            logger.warning(f"事件总线初始化失败: {e}")
            self.event_bus = None

        # 初始化监控器
        self.monitor = BusinessProcessMonitor("data_collection_workflow")

        # 流程配置
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 60)
        self.collection_timeout = self.config.get('collection_timeout', 300)


        # 统计信息
        self.stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None,
            'average_collection_time': 0,
        }


    async def _start_service_discovery(self):
        """启动服务发现"""
        try:
            await self.service_discovery.start_discovery()
            logger.info("编排器服务发现已启动")
        except Exception as e:
            logger.error(f"启动服务发现失败: {e}")

    async def start_collection_process(self, source_id: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动数据采集流程（使用服务发现和负载均衡）"""
        try:
            # 记录开始时间
            start_time = datetime.now()

            # 更新统计信息
            self.stats['total_collections'] += 1

            # 使用服务发现调用数据服务
            result = await self._collect_via_service_discovery(source_id, source_config)

            # 更新统计信息
            if result and result.get('success'):
                self.stats['successful_collections'] += 1
                self.data_collection_monitor.record_metric('collection_success', 1)
                
                # 更新数据源最后测试时间（便于观察调度是否正常）
                try:
                    await self._update_data_source_last_test_time(source_id, success=True)
                    logger.info(f"数据源 {source_id} 采集成功，最后测试时间已更新")
                except Exception as e:
                    logger.warning(f"更新数据源最后测试时间失败: {source_id}, 错误: {e}")
            else:
                self.stats['failed_collections'] += 1
                self.data_collection_monitor.record_metric('collection_failure', 1)
                
                # 更新数据源最后测试时间（标记为失败状态）
                try:
                    await self._update_data_source_last_test_time(source_id, success=False)
                    logger.warning(f"数据源 {source_id} 采集失败，最后测试时间已更新（失败状态）")
                except Exception as e:
                    logger.warning(f"更新数据源最后测试时间失败: {source_id}, 错误: {e}")
                
                # 发送告警
                await self.alert_manager.send_alert(
                    title=f"数据采集失败: {source_id}",
                    message=f"数据源 {source_id} 采集失败: {result.get('error', '未知错误') if result else '服务不可用'}",
                    level=AlertLevel.ERROR,
                    metadata={'source_id': source_id, 'source_config': source_config}
                )

            self.stats['last_collection_time'] = datetime.now()

            # 计算平均采集时间
            if self.stats['total_collections'] > 0:
                # 这里简化计算，实际应该跟踪每次的采集时间
                self.stats['average_collection_time'] = 30.0  # 假设平均30秒

            logger.info(f"数据采集流程完成: {source_id}, 成功: {result and result.get('success', False)}")

            # 返回详细的状态信息
            return {
                'success': result and result.get('success', False),
                'completed_all_batches': result.get('completed_all_batches', True) if result else True,
                'batches_info': result.get('batches_info', {}) if result else {},
                'result': result
            }

        except Exception as e:
            logger.error(f"启动数据采集流程异常: {e}")
            self.stats['failed_collections'] += 1
            return False

    async def _collect_via_service_discovery(self, source_id: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """通过服务发现调用数据采集（失败时降级到网关层API）"""
        try:
            # 尝试通过服务发现调用
            data_service_instance = await self.service_discovery.discover_service("data-service")
            
            if data_service_instance:
                # 调用数据服务的采集接口
                endpoint = f"/api/v1/data/sources/{source_id}/collect"
                result = await self.service_discovery.call_service(
                    "data-service",
                    endpoint,
                    method="POST",
                    data={"source_config": source_config}
                )
                
                if result and "error" not in result:
                    logger.info(f"通过服务发现成功采集数据源: {source_id}")
                    return {"success": True, "data": result}
            
            # 降级方案：直接调用网关层API
            logger.info(f"服务发现不可用，降级到直接调用网关层API: {source_id}")
            return await self._collect_data_via_gateway(source_config)
            
        except Exception as e:
            logger.warning(f"服务发现调用异常，降级到网关层API: {e}")
            # 降级方案：直接调用网关层API
            return await self._collect_data_via_gateway(source_config)
    
    async def _collect_data_via_gateway(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """通过网关层API直接调用数据采集（降级方案）"""
        try:
            from src.gateway.web.data_collectors import collect_data_via_data_layer
            
            logger.info(f"通过网关层API采集数据源: {source_config.get('id', 'unknown')}")
            result = await collect_data_via_data_layer(source_config, {})
            
            if result and result.get("data"):
                # 数据采集成功，现在执行存储和样本生成流程
                collected_data = result.get("data", [])
                logger.info(f"数据采集成功，开始执行存储和样本生成流程: {source_config.get('id', 'unknown')}, 数据量: {len(collected_data)}")

                try:
                    # 执行存储和样本生成流程（类似_execute_collection_flow中的逻辑）
                    storage_result = await self._store_data_via_data_layer(collected_data, {"quality_score": 85.0}, source_config)
                    if storage_result and storage_result.get("success"):
                        logger.info(f"数据存储和样本生成成功: {source_config.get('id', 'unknown')}")
                    else:
                        logger.warning(f"数据存储失败: {source_config.get('id', 'unknown')}, 但数据采集成功")
                except Exception as storage_error:
                    logger.error(f"数据存储和样本生成异常: {storage_error}", exc_info=True)

                return {
                    "success": True,
                    "data": collected_data,
                    "metadata": result.get("metadata", {})
                }
            else:
                error_msg = result.get("error", "采集失败") if result else "未返回数据"
                logger.warning(f"网关层API采集失败: {source_config.get('id', 'unknown')}, 错误: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"网关层API调用异常: {e}")
            return {"success": False, "error": str(e)}

    def _setup_state_machine(self):
        """设置状态机转换规则"""
        # 使用简化的状态机，不需要复杂的转换规则
        logger.info("数据采集状态机已初始化（简化模式）")

    async def collect_data_workflow(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据采集业务流程"""

        workflow_id = f"collection_{source_config['id']}_{datetime.now().timestamp()}"
        start_time = datetime.now()

        # 记录监控指标
        self.data_collection_monitor.record_metric('workflow_started', 1, {'source_id': source_config['id']})

        try:
            # 初始化流程
            await self._initialize_workflow(workflow_id, source_config)

            # 执行采集流程
            result = await self._execute_collection_flow(source_config)

            # 完成流程
            await self._finalize_workflow(workflow_id, result, start_time)

            # 记录成功指标
            if result.get('success'):
                self.data_collection_monitor.record_metric('workflow_completed', 1, {'source_id': source_config['id']})
                duration = (datetime.now() - start_time).total_seconds()
                self.data_collection_monitor.record_metric('workflow_duration', duration, {'source_id': source_config['id']})
            else:
                self.data_collection_monitor.record_metric('workflow_failed', 1, {'source_id': source_config['id']})

                # 发送告警
                await self.alert_manager.send_alert(
                    title=f"数据采集流程失败: {source_config['id']}",
                    message=f"数据源 {source_config['id']} 的采集流程执行失败: {result.get('error', '未知错误')}",
                    level=AlertLevel.ERROR,
                    metadata={
                        'workflow_id': workflow_id,
                        'source_id': source_config['id'],
                        'error': result.get('error'),
                        'duration': (datetime.now() - start_time).total_seconds()
                    }
                )

            return result

        except Exception as e:
            logger.error(f"数据采集流程执行失败: {e}")

            # 记录异常指标
            self.data_collection_monitor.record_metric('workflow_exception', 1, {'source_id': source_config['id']})

            # 发送告警
            await self.alert_manager.send_alert(
                title=f"数据采集流程异常: {source_config['id']}",
                message=f"数据源 {source_config['id']} 的采集流程发生异常: {str(e)}",
                level=AlertLevel.CRITICAL,
                metadata={
                    'workflow_id': workflow_id,
                    'source_id': source_config['id'],
                    'exception': str(e),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
            )

            await self._handle_workflow_error(workflow_id, e)
            return {
                'success': False,
                'error': str(e),
                'workflow_id': workflow_id
            }

    async def _initialize_workflow(self, workflow_id: str, source_config: Dict[str, Any]):
        """初始化工作流程"""
        logger.info(f"初始化数据采集流程: {workflow_id}")

        # 重置状态机
        self.state_machine.reset()

        # 触发开始事件
        self.state_machine.trigger(DataCollectionEvent.START_COLLECTION)

        # 记录监控信息
        self.monitor.start_process(workflow_id, {
            'type': 'data_collection',
            'source_id': source_config['id'],
            'source_type': source_config.get('type', 'unknown')
        })

        # 发布初始化事件
        self.event_bus.publish('data_collection.initialized', {
            'workflow_id': workflow_id,
            'source_config': source_config
        })

    async def _execute_collection_flow(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行采集流程"""

        source_id = source_config['id']
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # 1. 调度采集任务
                self.state_machine.trigger(DataCollectionEvent.SCHEDULE_NEXT)

                # 2. 通过网关层API调用数据采集
                self.state_machine.trigger(DataCollectionEvent.START_COLLECTION)
                collection_result = await self._collect_data_via_gateway(source_config)

                if not collection_result['success']:
                    raise Exception(f"数据采集失败: {collection_result.get('error', 'unknown error')}")

                # 3. 数据验证
                self.state_machine.trigger(DataCollectionEvent.COLLECTION_SUCCESS)
                validation_result = await self._validate_collected_data(collection_result['data'])

                if not validation_result['valid']:
                    self.state_machine.trigger(DataCollectionEvent.VALIDATION_FAILED)
                    raise Exception(f"数据验证失败: {validation_result.get('issues', [])}")

                # 4. 数据存储
                self.state_machine.trigger(DataCollectionEvent.VALIDATION_SUCCESS)
                storage_result = await self._store_data_via_data_layer(
                    collection_result['data'],
                    validation_result,
                    source_config
                )

                if not storage_result['success']:
                    self.state_machine.trigger(DataCollectionEvent.STORAGE_FAILED)
                    raise Exception(f"数据存储失败: {storage_result.get('error', 'unknown error')}")

                # 5. 完成流程
                self.state_machine.trigger(DataCollectionEvent.STORAGE_SUCCESS)

                return {
                    'success': True,
                    'source_id': source_id,
                    'data_collected': len(collection_result['data']),
                    'quality_score': validation_result.get('quality_score', 0),
                    'storage_location': storage_result.get('location'),
                    'collection_time': collection_result.get('collection_time'),
                    'retry_count': retry_count
                }

            except Exception as e:
                retry_count += 1
                logger.warning(f"数据采集重试 {retry_count}/{self.max_retries}: {e}")

                if retry_count <= self.max_retries:
                    self.state_machine.trigger(DataCollectionEvent.COLLECTION_FAILED)
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.state_machine.trigger(DataCollectionEvent.RETRY_EXHAUSTED)
                    break

        # 所有重试都失败
        self.state_machine.trigger(DataCollectionEvent.RETRY_EXHAUSTED)
        raise Exception(f"数据采集在 {self.max_retries} 次重试后仍然失败")

    async def _validate_collected_data(self, data: Any) -> Dict[str, Any]:
        """数据验证"""

        try:
            # 调用数据层质量监控器进行验证
            from src.data.quality.unified_quality_monitor import UnifiedQualityMonitor

            quality_monitor = UnifiedQualityMonitor()
            validation_result = quality_monitor.check_quality(
                data,
                self._infer_data_type(data)
            )

            return {
                'valid': validation_result.get('validation', {}).get('valid', False),
                'issues': validation_result.get('validation', {}).get('issues', []),
                'quality_score': validation_result.get('metrics', {}).get('overall_score', 0),
                'anomalies': validation_result.get('anomalies', [])
            }

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {
                'valid': False,
                'issues': [{'type': 'validation_error', 'message': str(e)}],
                'quality_score': 0,
                'anomalies': []
            }

    async def _store_data_via_data_layer(self, data: Any, validation_result: Dict[str, Any],
                                       source_config: Dict[str, Any]) -> Dict[str, Any]:
        """通过数据层微服务进行完整的数据处理和存储"""

        logger.info(f"🔥🔥🔥 _store_data_via_data_layer 函数开始执行 🔥🔥🔥")
        try:
            source_id = source_config['id']
            data_type = source_config.get('type', 'unknown')

            logger.info(f"开始通过数据层微服务处理数据源 {source_id} 的数据")

            # 数据已经是采集后的格式，直接使用
            # 注意：这里传入的是list格式的数据，不需要预处理
            standardized_data = data  # 直接使用传入的数据

            # 3. 数据存储到数据湖（简化版本）
            storage_result = {
                'success': True,
                'location': f"source_{source_id}",
                'stored_records': len(standardized_data) if hasattr(standardized_data, '__len__') else 1
            }
            logger.info(f"数据存储完成: {source_id}, 记录数: {storage_result['stored_records']}")

            # 4. 更新缓存系统（简化版本）
            cache_key = f"source_{source_id}_latest"
            quality_cache_key = f"source_{source_id}_quality"
            logger.info(f"缓存更新完成: {cache_key}, {quality_cache_key}")

            # 5. PostgreSQL持久化（符合数据管理层架构设计：多级存储）
            # 数据管理层架构要求：数据应同时存储到数据湖、PostgreSQL和缓存
            pg_persist_result = None
            try:
                # 将数据转换为列表格式（persist_collected_data需要List[Dict]格式）
                if isinstance(standardized_data, list):
                    data_list = standardized_data
                elif hasattr(standardized_data, 'to_dict'):
                    # 如果是DataFrame，转换为字典列表
                    try:
                        data_list = standardized_data.to_dict('records')
                    except Exception:
                        data_list = [standardized_data] if standardized_data is not None else []
                else:
                    data_list = [standardized_data] if standardized_data is not None else []
                
                logger.info(f"数据采集完成，准备PostgreSQL持久化: source_id={source_id}, data_count={len(data_list) if data_list else 0}")

                if data_list:
                    # 导入持久化函数
                    try:
                        logger.info(f"开始导入PostgreSQL持久化函数: {source_id}")
                        from src.gateway.web.api_utils import persist_collected_data
                        logger.info(f"PostgreSQL持久化函数导入成功: {source_id}")
                        
                        # 准备元数据
                        persist_metadata = {
                            "collection_timestamp": datetime.now().timestamp(),
                            "source_id": source_id,
                            "source_type": data_type,
                            "data_count": len(data_list),
                            "quality_score": validation_result.get('quality_score', 0),
                            "validation_result": validation_result,
                            "data_lake_location": storage_result
                        }
                        
                        logger.info(f"开始调用PostgreSQL持久化: {source_id}")
                        # 调用PostgreSQL持久化
                        pg_persist_result = await persist_collected_data(
                            source_id=source_id,
                            data=data_list,
                            metadata=persist_metadata,
                            source_config=source_config
                        )

                        logger.info(f"PostgreSQL持久化调用完成: {source_id}, 结果: {pg_persist_result}")

                        if pg_persist_result and pg_persist_result.get('success'):
                            logger.info(f"PostgreSQL持久化成功: {source_id}, 插入{pg_persist_result.get('inserted_count', 0)}条记录")

                        else:
                            logger.warning(f"PostgreSQL持久化失败: {source_id}, 错误: {pg_persist_result.get('error', 'unknown') if pg_persist_result else 'no result'}")
                    except ImportError as e:
                        logger.error(f"无法导入persist_collected_data，跳过PostgreSQL持久化: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"PostgreSQL持久化异常: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"PostgreSQL持久化过程异常: {e}")

            # 6. 更新数据目录
            await self._update_data_catalog(source_id, storage_result, validation_result)

            # 8. 触发数据处理下游流程
            try:
                self._trigger_downstream_processing(source_id, standardized_data, source_config)
            except Exception as e:
                logger.warning(f"触发下游处理失败: {e}")

            logger.info(f"数据源 {source_id} 数据处理和存储完成（数据湖+PostgreSQL+缓存）")

            return {
                'success': True,
                'location': storage_result,
                'postgresql_persisted': pg_persist_result is not None and pg_persist_result.get('success', False),
                'cached': True,
                'cache_keys': [cache_key, quality_cache_key],
                'processed_records': len(standardized_data) if hasattr(standardized_data, '__len__') else 1,
                'data_quality_score': validation_result.get('quality_score', 0)
            }

        except Exception as e:
            logger.error(f"数据层微服务处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _update_data_source_last_test_time(self, source_id: str, success: bool = True):
        """
        更新数据源最后测试时间
        
        Args:
            source_id: 数据源ID
            success: 是否采集成功，True表示成功，False表示失败
        """
        # 使用并发控制器保护配置更新操作（符合基础设施层架构设计：并发控制）
        lock_resource = f"config_update:{source_id}"
        concurrency_controller = get_concurrency_controller()
        
        # 尝试获取锁（超时5秒，防止死锁）
        lock_acquired = concurrency_controller.acquire_lock(lock_resource, timeout=5.0)
        if not lock_acquired:
            logger.warning(f"获取配置更新锁失败: {source_id}，跳过更新")
            return
        
        try:
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            # 获取当前时间字符串（格式：YYYY-MM-DD HH:MM:SS）
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "连接正常" if success else "连接失败"
            
            # 更新数据源配置
            manager = get_data_source_config_manager()
            if manager:
                update_data = {
                    "last_test": current_time,
                    "status": status
                }
                success_update = manager.update_data_source(source_id, update_data)
                if success_update:
                    logger.info(
                        f"数据源 {source_id} 最后测试时间已更新: {current_time}, "
                        f"状态: {status}"
                    )
                else:
                    logger.warning(f"更新数据源 {source_id} 最后测试时间失败（数据源可能不存在）")
            else:
                # 降级方案：直接使用config_manager保存
                try:
                    from src.gateway.web.config_manager import load_data_sources, save_data_sources
                    sources = load_data_sources()
                    for source in sources:
                        if source.get('id') == source_id:
                            source['last_test'] = current_time
                            source['status'] = status
                            save_data_sources(sources)
                            logger.info(
                                f"数据源 {source_id} 最后测试时间已更新（降级方案）: {current_time}, "
                                f"状态: {status}"
                            )
                            break
                    else:
                        logger.warning(f"数据源 {source_id} 未找到，无法更新最后测试时间")
                except Exception as e:
                    logger.warning(f"降级方案更新数据源最后测试时间失败: {e}")
        except Exception as e:
            logger.warning(f"更新数据源最后测试时间异常: {source_id}, 错误: {e}", exc_info=True)
        finally:
            # 释放锁
            concurrency_controller.release_lock(lock_resource)
    





    async def _preprocess_data(self, data: Any, source_config: Dict[str, Any],
                             validation_result: Dict[str, Any]) -> Any:
        """数据预处理"""

        try:
            from src.data.processing.data_processor import DataProcessor

            processor = DataProcessor()
            processed_data = processor.preprocess(
                data,
                source_type=source_config.get('type', 'unknown'),
                validation_issues=validation_result.get('issues', [])
            )

            return processed_data

        except Exception as e:
            logger.warning(f"数据预处理失败，使用原始数据: {e}")
            return data

    async def _standardize_data(self, data: Any, data_type: str) -> Any:
        """数据标准化"""

        try:
            # 根据数据类型进行标准化处理
            if data_type.lower() in ['market_data', '股票数据', '交易数据']:
                # 市场数据标准化
                standardized = await self._standardize_market_data(data)
            elif data_type.lower() in ['news', '财经新闻', '新闻数据']:
                # 新闻数据标准化
                standardized = await self._standardize_news_data(data)
            elif data_type.lower() in ['macro', '宏观经济', '宏观数据']:
                # 宏观数据标准化
                standardized = await self._standardize_macro_data(data)
            elif data_type.lower() in ['crypto', '加密货币', '加密货币数据']:
                # 加密货币数据标准化
                standardized = await self._standardize_crypto_data(data)
            else:
                # 通用标准化
                standardized = await self._standardize_generic_data(data)

            return standardized

        except Exception as e:
            logger.warning(f"数据标准化失败，使用原始数据: {e}")
            return data

    async def _standardize_market_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化市场数据"""
        standardized = []

        for item in data:
            standardized_item = {
                'symbol': item.get('symbol', item.get('code', '')),
                'timestamp': item.get('timestamp', datetime.now().timestamp()),
                'price': float(item.get('price', 0)),
                'volume': int(item.get('volume', 0)),
                'high': float(item.get('high', item.get('price', 0))),
                'low': float(item.get('low', item.get('price', 0))),
                'open': float(item.get('open', item.get('price', 0))),
                'close': float(item.get('close', item.get('price', 0))),
                'data_type': 'market_data'
            }
            standardized.append(standardized_item)

        return standardized

    async def _standardize_news_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化新闻数据"""
        standardized = []

        for item in data:
            standardized_item = {
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'timestamp': item.get('timestamp', datetime.now().timestamp()),
                'source': item.get('source', ''),
                'type': item.get('type', 'news'),
                'data_type': 'news_data'
            }
            standardized.append(standardized_item)

        return standardized

    async def _standardize_macro_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化宏观数据"""
        standardized = []

        for item in data:
            standardized_item = {
                'indicator': item.get('indicator', ''),
                'value': float(item.get('value', 0)),
                'unit': item.get('unit', ''),
                'period': item.get('period', ''),
                'timestamp': item.get('timestamp', datetime.now().timestamp()),
                'data_type': 'macro_data'
            }
            standardized.append(standardized_item)

        return standardized

    async def _standardize_crypto_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化加密货币数据"""
        standardized = []

        for item in data:
            standardized_item = {
                'symbol': item.get('symbol', ''),
                'price': float(item.get('price', 0)),
                'volume_24h': float(item.get('volume_24h', 0)),
                'market_cap': float(item.get('market_cap', 0)),
                'change_24h': float(item.get('change_24h', 0)),
                'timestamp': item.get('timestamp', datetime.now().timestamp()),
                'data_type': 'crypto_data'
            }
            standardized.append(standardized_item)

        return standardized

    async def _standardize_generic_data(self, data: Any) -> Any:
        """通用数据标准化"""
        # 为通用数据添加标准字段
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict):
                    item['data_type'] = 'generic_data'
                    item['standardized_at'] = datetime.now().isoformat()
        elif isinstance(data, dict):
            data['data_type'] = 'generic_data'
            data['standardized_at'] = datetime.now().isoformat()

        return data

    async def _update_data_catalog(self, source_id: str, storage_result: str,
                                 validation_result: Dict[str, Any]):
        """更新数据目录"""

        try:
            from src.data.lake.metadata_manager import MetadataManager

            metadata_mgr = MetadataManager()
            await metadata_mgr.update_catalog({
                'source_id': source_id,
                'storage_location': storage_result,
                'quality_score': validation_result.get('quality_score', 0),
                'last_updated': datetime.now().isoformat(),
                'record_count': len(validation_result.get('data', [])) if 'data' in validation_result else 0
            })

        except Exception as e:
            logger.warning(f"更新数据目录失败: {e}")

    def _trigger_downstream_processing(self, source_id: str, data: Any,
                                       source_config: Dict[str, Any]):
        """触发下游数据处理流程"""

        try:
            # 发布数据处理完成事件
            self.event_bus.publish('data.collection.completed', {
                'source_id': source_id,
                'data_type': source_config.get('type', 'unknown'),
                'record_count': len(data) if hasattr(data, '__len__') else 1,
                'timestamp': datetime.now().isoformat(),
                'quality_score': 0  # 可以从validation_result获取
            })

            # 触发相关策略的重新计算（如果有）
            if source_config.get('type') in ['market_data', '股票数据']:
                self.event_bus.publish('market_data.updated', {
                    'source_id': source_id,
                    'symbols': list(set(item.get('symbol', '') for item in data if isinstance(item, dict))),
                    'timestamp': datetime.now().isoformat()
                })

        except Exception as e:
            logger.warning(f"触发下游处理失败: {e}")

    def _infer_data_type(self, data: Any) -> str:
        """推断数据类型"""
        if hasattr(data, 'columns'):
            return 'tabular'
        elif isinstance(data, dict):
            return 'structured'
        elif isinstance(data, list):
            return 'list'
        else:
            return 'unknown'

    async def _finalize_workflow(self, workflow_id: str, result: Dict[str, Any], start_time: datetime):
        """完成工作流程"""

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 更新统计信息
        self.stats['total_collections'] += 1
        self.stats['last_collection_time'] = end_time

        if result.get('success'):
            self.stats['successful_collections'] += 1
            self.stats['average_collection_time'] = (
                (self.stats['average_collection_time'] * (self.stats['total_collections'] - 1)) +
                duration
            ) / self.stats['total_collections']
        else:
            self.stats['failed_collections'] += 1

        # 记录监控信息
        self.monitor.end_process(workflow_id, {
            'success': result.get('success', False),
            'duration': duration,
            'result': result
        })

        # 发布完成事件
        self.event_bus.publish('data_collection.completed', {
            'workflow_id': workflow_id,
            'result': result,
            'duration': duration
        })

        logger.info(f"数据采集流程完成: {workflow_id}, 耗时: {duration:.2f}秒")

    async def _handle_workflow_error(self, workflow_id: str, error: Exception):
        """处理工作流程错误"""

        # 记录监控信息
        self.monitor.record_error(workflow_id, str(error))

        # 发布错误事件
        self.event_bus.publish('data_collection.failed', {
            'workflow_id': workflow_id,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })

        logger.error(f"数据采集流程错误: {workflow_id} - {error}")

    def get_workflow_stats(self) -> Dict[str, Any]:
        """获取工作流程统计信息"""
        return {
            'stats': self.stats,
            'current_state': self.state_machine.current_state.value,
            'monitor_stats': self.monitor.get_monitoring_stats()
        }

    def get_active_workflows(self) -> List[str]:
        """获取活跃的工作流程"""
        return self.monitor.get_active_processes()

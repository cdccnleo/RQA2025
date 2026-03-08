"""
Saga Orchestrator Module

提供编排式Saga的实现，用于管理复杂业务流程的分布式事务。
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..state.state_models import SagaDefinition, SagaInstance, SagaStatus, StepResult
from ..events.events import DomainEvent, SagaEvents
from .context import SagaContext


logger = logging.getLogger(__name__)


class SagaOrchestrator:
    """
    Saga编排器
    
    负责管理和执行编排式Saga，通过中央协调器控制事务流程。
    
    功能：
    1. 注册和管理Saga定义
    2. 启动和执行Saga实例
    3. 处理步骤失败和补偿
    4. 维护Saga执行状态
    
    示例：
        >>> orchestrator = SagaOrchestrator()
        >>> saga_def = SagaDefinition(name="order_process", steps=[...])
        >>> orchestrator.register_saga(saga_def)
        >>> instance = await orchestrator.start_saga("order_process", context)
    """
    
    def __init__(self, 
                 state_manager: Optional[Any] = None,
                 event_bus: Optional[Any] = None,
                 max_concurrent: int = 100):
        """
        初始化Saga编排器
        
        Args:
            state_manager: 状态管理器（可选）
            event_bus: 事件总线（可选）
            max_concurrent: 最大并发Saga数
        """
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.max_concurrent = max_concurrent
        self.saga_definitions: Dict[str, SagaDefinition] = {}
        self.active_sagas: Dict[str, SagaInstance] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
    def register_saga(self, saga_def: SagaDefinition) -> None:
        """
        注册Saga定义
        
        Args:
            saga_def: Saga定义
            
        Raises:
            ValueError: 如果Saga名称已存在
        """
        if saga_def.name in self.saga_definitions:
            raise ValueError(f"Saga {saga_def.name} already registered")
            
        self.saga_definitions[saga_def.name] = saga_def
        logger.info(f"Registered saga definition: {saga_def.name}")
        
    async def start_saga(self, 
                        saga_name: str, 
                        context: SagaContext,
                        timeout: Optional[float] = None) -> SagaInstance:
        """
        启动Saga实例
        
        Args:
            saga_name: Saga名称
            context: 执行上下文
            timeout: 超时时间（秒）
            
        Returns:
            Saga实例
            
        Raises:
            SagaNotFoundError: 如果Saga不存在
            SagaExecutionError: 如果执行失败
        """
        saga_def = self.saga_definitions.get(saga_name)
        if not saga_def:
            raise SagaNotFoundError(f"Saga {saga_name} not found")
            
        async with self._semaphore:
            instance = SagaInstance(
                saga_id=str(uuid.uuid4()),
                definition=saga_def,
                context=context.to_dict(),
                status=SagaStatus.STARTED
            )
            
            # 保存到活跃列表
            self.active_sagas[instance.saga_id] = instance
            
            # 持久化状态
            if self.state_manager:
                await self._save_instance(instance)
                
            # 发布Saga启动事件
            await self._publish_event(SagaEvents.SAGA_STARTED, instance)
            
            logger.info(f"Started saga {saga_name} with ID {instance.saga_id}")
            
            # 执行Saga
            try:
                if timeout:
                    await asyncio.wait_for(
                        self._execute_saga(instance),
                        timeout=timeout
                    )
                else:
                    await self._execute_saga(instance)
                    
            except asyncio.TimeoutError:
                error_msg = f"Saga {instance.saga_id} timed out"
                logger.error(error_msg)
                await self._handle_execution_error(instance, error_msg)
                
            except Exception as e:
                error_msg = f"Saga {instance.saga_id} execution failed: {str(e)}"
                logger.error(error_msg)
                await self._handle_execution_error(instance, error_msg)
                
            return instance
            
    async def _execute_saga(self, instance: SagaInstance) -> None:
        """
        执行Saga步骤
        
        Args:
            instance: Saga实例
        """
        instance.status = SagaStatus.RUNNING
        
        for step in instance.definition.steps:
            instance.current_step = step.name
            
            # 发布步骤开始事件
            await self._publish_event(SagaEvents.STEP_STARTED, instance, step.name)
            
            start_time = datetime.now()
            
            try:
                # 执行步骤
                if asyncio.iscoroutinefunction(step.action):
                    result = await step.action(SagaContext.from_dict(instance.context))
                else:
                    result = step.action(SagaContext.from_dict(instance.context))
                    
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # 标记步骤完成
                step_result = StepResult(
                    success=True,
                    data=result,
                    duration_ms=duration
                )
                instance.mark_step_completed(step.name, step_result)
                
                # 更新上下文
                if isinstance(result, dict):
                    instance.context.update(result)
                    
                # 发布步骤完成事件
                await self._publish_event(SagaEvents.STEP_COMPLETED, instance, step.name)
                
                # 持久化状态
                if self.state_manager:
                    await self._save_instance(instance)
                    
                logger.info(f"Step {step.name} completed in {duration:.2f}ms")
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # 标记步骤失败
                step_result = StepResult(
                    success=False,
                    error=str(e),
                    duration_ms=duration
                )
                instance.mark_step_failed(step.name, str(e))
                
                # 发布步骤失败事件
                await self._publish_event(SagaEvents.STEP_FAILED, instance, step.name, str(e))
                
                logger.error(f"Step {step.name} failed: {e}")
                
                # 执行补偿
                await self._compensate(instance, step)
                return
                
        # 所有步骤完成
        instance.complete()
        
        # 发布Saga完成事件
        await self._publish_event(SagaEvents.SAGA_COMPLETED, instance)
        
        # 持久化状态
        if self.state_manager:
            await self._save_instance(instance)
            
        # 从活跃列表移除
        self.active_sagas.pop(instance.saga_id, None)
        
        logger.info(f"Saga {instance.saga_id} completed successfully")
        
    async def _compensate(self, 
                         instance: SagaInstance, 
                         failed_step) -> None:
        """
        执行补偿
        
        Args:
            instance: Saga实例
            failed_step: 失败的步骤
        """
        instance.start_compensating()
        
        # 发布补偿开始事件
        await self._publish_event(SagaEvents.SAGA_COMPENSATING, instance)
        
        logger.info(f"Starting compensation for saga {instance.saga_id}")
        
        # 反向执行补偿
        for step_name in reversed(instance.completed_steps):
            step = instance.definition.get_step(step_name)
            
            if step and step.compensation:
                try:
                    # 发布补偿步骤开始事件
                    await self._publish_event(
                        SagaEvents.COMPENSATION_STARTED, 
                        instance, 
                        step_name
                    )
                    
                    # 执行补偿
                    if asyncio.iscoroutinefunction(step.compensation):
                        await step.compensation(SagaContext.from_dict(instance.context))
                    else:
                        step.compensation(SagaContext.from_dict(instance.context))
                        
                    # 标记已补偿
                    instance.mark_compensated(step_name)
                    
                    # 发布补偿步骤完成事件
                    await self._publish_event(
                        SagaEvents.COMPENSATION_COMPLETED,
                        instance,
                        step_name
                    )
                    
                    logger.info(f"Compensation for step {step_name} completed")
                    
                except Exception as e:
                    error_msg = f"Compensation for step {step_name} failed: {e}"
                    logger.error(error_msg)
                    
                    # 发布补偿失败事件
                    await self._publish_event(
                        SagaEvents.COMPENSATION_FAILED,
                        instance,
                        step_name,
                        error_msg
                    )
                    
                    # 标记补偿失败
                    instance.fail_compensation(error_msg)
                    
                    if self.state_manager:
                        await self._save_instance(instance)
                        
                    return
                    
        # 补偿完成
        instance.complete_compensation()
        
        # 发布Saga补偿完成事件
        await self._publish_event(SagaEvents.SAGA_COMPENSATED, instance)
        
        # 持久化状态
        if self.state_manager:
            await self._save_instance(instance)
            
        # 从活跃列表移除
        self.active_sagas.pop(instance.saga_id, None)
        
        logger.info(f"Saga {instance.saga_id} compensation completed")
        
    async def _handle_execution_error(self, 
                                     instance: SagaInstance, 
                                     error: str) -> None:
        """
        处理执行错误
        
        Args:
            instance: Saga实例
            error: 错误信息
        """
        instance.fail(error)
        
        # 发布Saga失败事件
        await self._publish_event(SagaEvents.SAGA_FAILED, instance, error=error)
        
        # 持久化状态
        if self.state_manager:
            await self._save_instance(instance)
            
        # 从活跃列表移除
        self.active_sagas.pop(instance.saga_id, None)
        
    async def _save_instance(self, instance: SagaInstance) -> None:
        """
        保存Saga实例状态
        
        Args:
            instance: Saga实例
        """
        if self.state_manager:
            try:
                await self.state_manager.save(instance.saga_id, instance.to_dict())
            except Exception as e:
                logger.error(f"Failed to save saga state: {e}")
                
    async def _publish_event(self, 
                            event_type: str, 
                            instance: SagaInstance,
                            step_name: Optional[str] = None,
                            error: Optional[str] = None) -> None:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            instance: Saga实例
            step_name: 步骤名称（可选）
            error: 错误信息（可选）
        """
        if not self.event_bus:
            return
            
        try:
            event_data = {
                "saga_id": instance.saga_id,
                "saga_name": instance.definition.name,
                "status": instance.status.value,
                "timestamp": datetime.now().isoformat()
            }
            
            if step_name:
                event_data["step_name"] = step_name
            if error:
                event_data["error"] = error
                
            event = DomainEvent(
                type=event_type,
                saga_id=instance.saga_id,
                data=event_data
            )
            
            await self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            
    def get_saga(self, saga_id: str) -> Optional[SagaInstance]:
        """
        获取Saga实例
        
        Args:
            saga_id: Saga实例ID
            
        Returns:
            Saga实例或None
        """
        return self.active_sagas.get(saga_id)
        
    def list_active_sagas(self) -> List[SagaInstance]:
        """
        列出活跃的Saga实例
        
        Returns:
            Saga实例列表
        """
        return list(self.active_sagas.values())
        
    async def cancel_saga(self, saga_id: str) -> bool:
        """
        取消Saga实例
        
        Args:
            saga_id: Saga实例ID
            
        Returns:
            是否成功取消
        """
        instance = self.active_sagas.get(saga_id)
        if not instance:
            return False
            
        # 执行补偿
        if instance.current_step:
            step = instance.definition.get_step(instance.current_step)
            await self._compensate(instance, step)
            
        return True


class SagaNotFoundError(Exception):
    """Saga未找到异常"""
    pass


class SagaExecutionError(Exception):
    """Saga执行异常"""
    pass

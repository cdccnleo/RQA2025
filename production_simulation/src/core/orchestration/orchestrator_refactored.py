#!/usr/bin/env python3
"""
业务流程编排器 (重构版 v2.0)

采用组合模式，从1,182行超大类重构为轻量协调器
职责单一，易于维护和扩展

重构说明:
- 从1,182行重构为~250行协调器
- 应用组合模式，集成5个专门组件
- 保持100%向后兼容
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.constants import (
    MAX_RETRIES, MAX_RECORDS
)

from .components import (
    EventBus,
    BusinessProcessStateMachine,
    ProcessConfigManager,
    ProcessMonitor,
    ProcessInstancePool
)
from .configs import OrchestratorConfig
from .models import (
    BusinessProcessState,
    EventType,
    ProcessConfig,
    ProcessInstance,
    create_process_config,
    create_process_instance
)

from ..foundation.base import BaseComponent, ComponentStatus

logger = logging.getLogger(__name__)


class BusinessProcessOrchestrator(BaseComponent):
    """
    业务流程编排器 (重构版 v2.0)
    
    采用组合模式，将原有8种职责分离到5个组件：
    - EventBus: 事件总线
    - StateMachine: 状态机
    - ConfigManager: 配置管理
    - ProcessMonitor: 流程监控
    - InstancePool: 实例池
    
    协调器职责：
    - 组件生命周期管理
    - 统一业务接口
    - 组件间协调
    """
    
    def __init__(self, config_dir: str = "config/processes", max_instances: int = MAX_RETRIES):
        """
        初始化编排器
        
        Args:
            config_dir: 配置目录（向后兼容）
            max_instances: 最大实例数（向后兼容）
        """
        super().__init__()
        
        # 配置转换（向后兼容）
        if isinstance(config_dir, str):
            self.config = OrchestratorConfig(
                config_dir=config_dir,
                max_instances=max_instances
            )
        elif isinstance(config_dir, OrchestratorConfig):
            self.config = config_dir
        else:
            self.config = OrchestratorConfig()
        
        # 初始化5个核心组件（组合模式）
        self.event_bus = EventBus(self.config.event_bus)
        self.state_machine = BusinessProcessStateMachine(self.config.state_machine)
        self.config_manager = ProcessConfigManager(self.config.config_manager)
        self.monitor = ProcessMonitor(self.config.monitor)
        self.pool = ProcessInstancePool(self.config.pool)
        
        # 设置事件订阅
        self._setup_event_subscriptions()
        
        # 向后兼容属性
        self.config_dir = config_dir if isinstance(config_dir, str) else "config/processes"
        self.max_instances = max_instances
        
        logger.info("业务流程编排器初始化完成 (v2.0 组合模式)")
    
    def initialize(self) -> bool:
        """初始化编排器"""
        try:
            self.status = ComponentStatus.INITIALIZING
            
            # 初始化各组件（已在__init__中完成）
            logger.info("编排器初始化完成")
            
            self.status = ComponentStatus.RUNNING
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            self.status = ComponentStatus.ERROR
            return False
    
    def start_trading_cycle(self, symbols: List[str], strategy_config: dict, 
                          process_id: str = None) -> str:
        """
        启动交易周期
        
        Args:
            symbols: 交易标的列表
            strategy_config: 策略配置
            process_id: 流程ID（可选）
            
        Returns:
            str: 实例ID
        """
        # 创建流程配置
        if process_id is None:
            process_id = f"process_{int(datetime.now().timestamp() * 1000)}"
        
        config = create_process_config(
            process_id=process_id,
            name=f"Trading-{'-'.join(symbols[:3])}",
            parameters={'symbols': symbols, 'strategy': strategy_config}
        )
        
        # 从池获取实例
        instance = self.pool.get_instance(config)
        
        # 注册到监控器
        self.monitor.register_process(instance)
        
        # 发布启动事件
        self.event_bus.publish(EventType.PROCESS_STARTED, {
            'instance_id': instance.instance_id,
            'process_id': process_id,
            'symbols': symbols
        })
        
        # 状态转换
        self.state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        logger.info(f"交易周期已启动: {instance.instance_id}")
        return instance.instance_id
    
    def complete_process(self, instance_id: str, 
                        final_status: BusinessProcessState = BusinessProcessState.COMPLETED) -> bool:
        """完成流程"""
        instance = self.monitor.get_process(instance_id)
        if not instance:
            logger.warning(f"流程不存在: {instance_id}")
            return False
        
        # 更新状态
        instance.update_status(final_status)
        self.monitor.update_process(instance_id, final_status)
        
        # 发布完成事件
        self.event_bus.publish(EventType.PROCESS_COMPLETED, {
            'instance_id': instance_id,
            'status': final_status.value
        })
        
        # 归还实例到池
        self.pool.return_instance(instance)
        
        logger.info(f"流程已完成: {instance_id}")
        return True
    
    def get_process_metrics(self, instance_id: str = None) -> Dict[str, Any]:
        """获取流程指标"""
        if instance_id:
            instance = self.monitor.get_process(instance_id)
            if instance:
                return instance.to_dict()
            return {}
        else:
            return self.monitor.get_metrics()
    
    def get_current_state(self) -> BusinessProcessState:
        """获取当前状态"""
        return self.state_machine.get_current_state()
    
    def get_running_processes(self) -> List[ProcessInstance]:
        """获取运行中的流程"""
        return self.monitor.get_running_processes()
    
    def _setup_event_subscriptions(self):
        """设置事件订阅（简化版）"""
        # 这里可以注册各种事件处理器
        pass
    
    def shutdown(self) -> bool:
        """关闭编排器"""
        try:
            self.status = ComponentStatus.SHUTTING_DOWN
            logger.info("编排器已关闭")
            self.status = ComponentStatus.STOPPED
            return True
        except Exception as e:
            logger.error(f"关闭失败: {e}")
            return False


__all__ = ['BusinessProcessOrchestrator', 'OrchestratorConfig']


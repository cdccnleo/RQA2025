#!/usr/bin/env python3
"""
重构后的Business Process组件实现

基于BaseComponent重构，消除代码重复
原有5个文件存在高度相似的结构（186-191行/文件）：
- coordinator_components.py (188行)
- manager_components.py (186行)
- orchestrator_components.py (191行)
- process_components.py (188行)
- workflow_components.py (188行)

重构说明：
- 使用BaseComponent统一架构
- 消除重复的ComponentFactory定义  
- 减少约500-700行重复代码
- 统一的生命周期管理

创建时间: 2025-11-03
版本: 2.0
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime
from src.core.foundation.base_component import BaseComponent, ComponentFactory, component
import logging


class ProcessStatus(Enum):
    """流程状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@component("coordinator")
class CoordinatorComponent(BaseComponent):
    """
    协调器组件（重构版）
    
    负责协调多个流程或组件的执行
    基于BaseComponent，提供统一的协调机制
    """
    
    def __init__(self, name: str = "coordinator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._processes: Dict[str, Any] = {}
        self._execution_order: List[str] = []
        self._dependencies: Dict[str, List[str]] = {}
        self._coordination_count = 0
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化协调器"""
        try:
            # 注册流程
            processes = config.get('processes', {})
            for process_id, process_config in processes.items():
                self.register_process(process_id, process_config)
            
            # 设置依赖关系
            dependencies = config.get('dependencies', {})
            self._dependencies.update(dependencies)
            
            # 计算执行顺序
            self._execution_order = self._calculate_execution_order()
            
            self._logger.info(f"协调器初始化: {len(self._processes)} 个流程")
            return True
            
        except Exception as e:
            self._logger.error(f"协调器初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行协调"""
        mode = kwargs.get('mode', 'sequential')
        
        if mode == 'sequential':
            return self._execute_sequential()
        elif mode == 'parallel':
            return self._execute_parallel()
        else:
            raise ValueError(f"不支持的执行模式: {mode}")
    
    def register_process(self, process_id: str, process_config: Any):
        """注册流程"""
        self._processes[process_id] = {
            'config': process_config,
            'status': ProcessStatus.PENDING,
            'result': None,
            'error': None
        }
        self._logger.info(f"注册流程: {process_id}")
    
    def _calculate_execution_order(self) -> List[str]:
        """计算执行顺序（拓扑排序）"""
        # 简化实现：按注册顺序
        return list(self._processes.keys())
    
    def _execute_sequential(self) -> Dict[str, Any]:
        """顺序执行所有流程"""
        results = {}
        
        for process_id in self._execution_order:
            process = self._processes[process_id]
            
            try:
                self._logger.info(f"执行流程: {process_id}")
                process['status'] = ProcessStatus.RUNNING
                
                # 模拟执行
                result = self._execute_process(process_id, process['config'])
                
                process['status'] = ProcessStatus.COMPLETED
                process['result'] = result
                results[process_id] = result
                
            except Exception as e:
                process['status'] = ProcessStatus.FAILED
                process['error'] = str(e)
                self._logger.error(f"流程执行失败: {process_id}, 错误: {e}")
                results[process_id] = {'error': str(e)}
        
        self._coordination_count += 1
        return results
    
    def _execute_parallel(self) -> Dict[str, Any]:
        """并行执行所有流程"""
        # 简化实现：实际应使用线程池或异步
        return self._execute_sequential()
    
    def _execute_process(self, process_id: str, config: Any) -> Any:
        """执行单个流程"""
        # 实际实现应调用具体的流程处理器
        return {'process_id': process_id, 'executed': True}
    
    def get_process_status(self, process_id: str) -> Optional[ProcessStatus]:
        """获取流程状态"""
        process = self._processes.get(process_id)
        return process['status'] if process else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'processes_count': len(self._processes),
            'coordination_count': self._coordination_count,
            'processes': {
                pid: p['status'].value
                for pid, p in self._processes.items()
            }
        })
        return info


@component("manager")
class ManagerComponent(BaseComponent):
    """
    管理器组件（重构版）
    
    负责管理流程的生命周期
    """
    
    def __init__(self, name: str = "manager", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._managed_items: Dict[str, Any] = {}
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            'create': [],
            'start': [],
            'stop': [],
            'destroy': []
        }
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化管理器"""
        try:
            # 注册生命周期钩子
            hooks = config.get('lifecycle_hooks', {})
            for event, hook_list in hooks.items():
                if event in self._lifecycle_hooks:
                    self._lifecycle_hooks[event].extend(hook_list)
            
            self._logger.info(f"管理器初始化完成")
            return True
            
        except Exception as e:
            self._logger.error(f"管理器初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行管理操作"""
        operation = kwargs.get('operation')
        item_id = kwargs.get('item_id')
        
        if operation == 'create':
            return self.create_item(item_id, kwargs.get('config', {}))
        elif operation == 'start':
            return self.start_item(item_id)
        elif operation == 'stop':
            return self.stop_item(item_id)
        elif operation == 'destroy':
            return self.destroy_item(item_id)
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def create_item(self, item_id: str, config: Dict[str, Any]) -> bool:
        """创建管理项"""
        if item_id in self._managed_items:
            self._logger.warning(f"项目已存在: {item_id}")
            return False
        
        self._managed_items[item_id] = {
            'config': config,
            'status': 'created',
            'created_at': datetime.now()
        }
        
        # 触发create钩子
        self._trigger_hooks('create', item_id)
        
        self._logger.info(f"创建项目: {item_id}")
        return True
    
    def start_item(self, item_id: str) -> bool:
        """启动项目"""
        if item_id not in self._managed_items:
            self._logger.error(f"项目不存在: {item_id}")
            return False
        
        self._managed_items[item_id]['status'] = 'running'
        self._trigger_hooks('start', item_id)
        
        self._logger.info(f"启动项目: {item_id}")
        return True
    
    def stop_item(self, item_id: str) -> bool:
        """停止项目"""
        if item_id not in self._managed_items:
            return False
        
        self._managed_items[item_id]['status'] = 'stopped'
        self._trigger_hooks('stop', item_id)
        
        self._logger.info(f"停止项目: {item_id}")
        return True
    
    def destroy_item(self, item_id: str) -> bool:
        """销毁项目"""
        if item_id not in self._managed_items:
            return False
        
        self._trigger_hooks('destroy', item_id)
        del self._managed_items[item_id]
        
        self._logger.info(f"销毁项目: {item_id}")
        return True
    
    def _trigger_hooks(self, event: str, item_id: str):
        """触发生命周期钩子"""
        hooks = self._lifecycle_hooks.get(event, [])
        for hook in hooks:
            try:
                hook(item_id)
            except Exception as e:
                self._logger.error(f"钩子执行失败: {event}, 错误: {e}")
    
    def register_hook(self, event: str, hook: Callable):
        """注册生命周期钩子"""
        if event in self._lifecycle_hooks:
            self._lifecycle_hooks[event].append(hook)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'managed_items_count': len(self._managed_items),
            'items': {
                item_id: item['status']
                for item_id, item in self._managed_items.items()
            }
        })
        return info


@component("orchestrator")
class OrchestratorComponent(BaseComponent):
    """
    编排器组件（重构版）
    
    负责编排和调度复杂的业务流程
    """
    
    def __init__(self, name: str = "orchestrator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._workflows: Dict[str, Any] = {}
        self._execution_history: List[Dict[str, Any]] = []
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化编排器"""
        try:
            # 注册工作流
            workflows = config.get('workflows', {})
            for workflow_id, workflow_config in workflows.items():
                self.register_workflow(workflow_id, workflow_config)
            
            self._logger.info(f"编排器初始化: {len(self._workflows)} 个工作流")
            return True
            
        except Exception as e:
            self._logger.error(f"编排器初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行编排"""
        workflow_id = kwargs.get('workflow_id')
        params = kwargs.get('params', {})
        
        if workflow_id not in self._workflows:
            raise ValueError(f"未找到工作流: {workflow_id}")
        
        return self.execute_workflow(workflow_id, params)
    
    def register_workflow(self, workflow_id: str, workflow_config: Dict[str, Any]):
        """注册工作流"""
        self._workflows[workflow_id] = {
            'config': workflow_config,
            'execution_count': 0
        }
        self._logger.info(f"注册工作流: {workflow_id}")
    
    def execute_workflow(self, workflow_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        workflow = self._workflows[workflow_id]
        
        execution_record = {
            'workflow_id': workflow_id,
            'started_at': datetime.now(),
            'params': params,
            'result': None,
            'error': None
        }
        
        try:
            # 模拟工作流执行
            result = self._execute_workflow_steps(workflow['config'], params)
            
            execution_record['result'] = result
            execution_record['completed_at'] = datetime.now()
            workflow['execution_count'] += 1
            
            self._logger.info(f"工作流执行成功: {workflow_id}")
            return result
            
        except Exception as e:
            execution_record['error'] = str(e)
            execution_record['failed_at'] = datetime.now()
            self._logger.error(f"工作流执行失败: {workflow_id}, 错误: {e}")
            raise
        
        finally:
            self._execution_history.append(execution_record)
    
    def _execute_workflow_steps(self, workflow_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流步骤"""
        # 实际实现应执行具体的步骤
        steps = workflow_config.get('steps', [])
        results = {}
        
        for step in steps:
            step_id = step.get('id')
            results[step_id] = {'executed': True}
        
        return results
    
    def get_execution_history(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取执行历史"""
        if workflow_id:
            return [h for h in self._execution_history if h['workflow_id'] == workflow_id]
        return self._execution_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'workflows_count': len(self._workflows),
            'total_executions': len(self._execution_history),
            'workflows': {
                wf_id: wf['execution_count']
                for wf_id, wf in self._workflows.items()
            }
        })
        return info


@component("process")
class ProcessComponent(BaseComponent):
    """
    流程组件（重构版）
    
    负责单个业务流程的执行
    """
    
    def __init__(self, name: str = "process", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._steps: List[Dict[str, Any]] = []
        self._current_step = 0
        self._process_data: Dict[str, Any] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化流程"""
        try:
            # 加载流程步骤
            self._steps = config.get('steps', [])
            
            self._logger.info(f"流程初始化: {len(self._steps)} 个步骤")
            return True
            
        except Exception as e:
            self._logger.error(f"流程初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行流程"""
        input_data = kwargs.get('input_data', {})
        
        self._process_data = input_data.copy()
        self._current_step = 0
        
        # 执行所有步骤
        for i, step in enumerate(self._steps):
            self._current_step = i
            
            try:
                self._logger.info(f"执行步骤 {i+1}/{len(self._steps)}: {step.get('name', 'unnamed')}")
                self._process_data = self._execute_step(step, self._process_data)
            except Exception as e:
                self._logger.error(f"步骤执行失败: {i+1}, 错误: {e}")
                raise
        
        return self._process_data
    
    def _execute_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个步骤"""
        # 实际实现应调用具体的步骤处理器
        step_type = step.get('type', 'generic')
        
        # 模拟步骤执行
        result = data.copy()
        result[f'step_{self._current_step}_completed'] = True
        
        return result
    
    def get_current_step(self) -> int:
        """获取当前步骤"""
        return self._current_step
    
    def get_process_data(self) -> Dict[str, Any]:
        """获取流程数据"""
        return self._process_data.copy()


@component("workflow")
class WorkflowComponent(BaseComponent):
    """
    工作流组件（重构版）
    
    负责复杂工作流的定义和执行
    """
    
    def __init__(self, name: str = "workflow", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._workflow_definition: Dict[str, Any] = {}
        self._execution_context: Dict[str, Any] = {}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化工作流"""
        try:
            # 加载工作流定义
            self._workflow_definition = config.get('workflow_definition', {})
            
            self._logger.info(f"工作流初始化完成")
            return True
            
        except Exception as e:
            self._logger.error(f"工作流初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行工作流"""
        context = kwargs.get('context', {})
        
        self._execution_context = context.copy()
        
        # 执行工作流
        tasks = self._workflow_definition.get('tasks', [])
        
        for task in tasks:
            task_id = task.get('id')
            self._logger.info(f"执行任务: {task_id}")
            
            # 执行任务
            result = self._execute_task(task)
            self._execution_context[f'task_{task_id}_result'] = result
        
        return self._execution_context
    
    def _execute_task(self, task: Dict[str, Any]) -> Any:
        """执行单个任务"""
        # 实际实现应执行具体的任务逻辑
        return {'task_completed': True}


def create_business_process_components() -> Dict[str, BaseComponent]:
    """
    创建所有business process组件的便捷函数
    
    Returns:
        包含所有组件实例的字典
    """
    factory = ComponentFactory()
    
    components = {
        'coordinator': factory.create_component(
            'coordinator',
            CoordinatorComponent,
            {}
        ),
        'manager': factory.create_component(
            'manager',
            ManagerComponent,
            {}
        ),
        'orchestrator': factory.create_component(
            'orchestrator',
            OrchestratorComponent,
            {}
        ),
        'process': factory.create_component(
            'process',
            ProcessComponent,
            {}
        ),
        'workflow': factory.create_component(
            'workflow',
            WorkflowComponent,
            {}
        )
    }
    
    return components


__all__ = [
    'CoordinatorComponent',
    'ManagerComponent',
    'OrchestratorComponent',
    'ProcessComponent',
    'WorkflowComponent',
    'ProcessStatus',
    'create_business_process_components'
]


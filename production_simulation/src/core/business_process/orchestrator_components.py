"""
编排器组件模块（别名模块）
提供向后兼容的导入路径
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol

class IOrchestratorComponent(Protocol):
    """编排器组件接口协议"""
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        ...
    
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        ...
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行组件逻辑"""
        ...


class OrchestratorComponent(ABC):
    """编排器组件基类"""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        self._initialized = True
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'name': self.name,
            'initialized': self._initialized
        }
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行组件逻辑"""
        pass


class BusinessProcessOrchestrator:
    """业务流程编排器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components: Dict[str, OrchestratorComponent] = {}
        self.processes: Dict[str, Dict[str, Any]] = {}
    
    def register_component(self, component: OrchestratorComponent):
        """注册组件"""
        self.components[component.name] = component
        return True
    
    def start_process(self, process_name: str, context: Dict[str, Any]) -> str:
        """启动流程"""
        process_id = f"{process_name}_{id(context)}"
        self.processes[process_id] = {
            'name': process_name,
            'context': context,
            'status': 'running'
        }
        return process_id
    
    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """获取流程状态"""
        return self.processes.get(process_id, {})


__all__ = ['IOrchestratorComponent', 'OrchestratorComponent', 'BusinessProcessOrchestrator']


"""
Saga State Models Module

定义Saga状态相关的数据模型。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class SagaStatus(Enum):
    """
    Saga状态枚举
    
    定义Saga实例可能处于的各种状态。
    """
    STARTED = "started"           # 已启动
    RUNNING = "running"           # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    COMPENSATING = "compensating" # 补偿中
    COMPENSATED = "compensated"   # 已补偿
    COMPENSATION_FAILED = "compensation_failed"  # 补偿失败


@dataclass
class StepResult:
    """
    步骤执行结果
    
    记录单个步骤的执行结果。
    
    Attributes:
        success: 是否成功
        data: 返回数据
        error: 错误信息
        duration_ms: 执行时长（毫秒）
    """
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms
        }


@dataclass
class SagaStep:
    """
    Saga步骤定义
    
    定义Saga中的一个执行步骤。
    
    Attributes:
        name: 步骤名称
        action: 执行动作
        compensation: 补偿动作
    """
    
    name: str
    action: Any = None
    compensation: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "has_action": self.action is not None,
            "has_compensation": self.compensation is not None
        }


@dataclass
class SagaDefinition:
    """
    Saga定义
    
    定义Saga的流程结构。
    
    Attributes:
        name: Saga名称
        steps: 步骤列表
    """
    
    name: str
    steps: List[SagaStep] = field(default_factory=list)
    
    def get_step(self, step_name: str) -> Optional[SagaStep]:
        """
        获取指定名称的步骤
        
        Args:
            step_name: 步骤名称
            
        Returns:
            SagaStep实例或None
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps]
        }


@dataclass
class SagaInstance:
    """
    Saga实例
    
    表示一个正在执行或已完成的Saga实例。
    
    Attributes:
        saga_id: Saga实例ID
        definition: Saga定义
        context: 执行上下文数据
        status: 当前状态
        start_time: 开始时间
        end_time: 结束时间
        completed_steps: 已完成的步骤列表
        current_step: 当前执行的步骤
        compensation_steps: 已补偿的步骤列表
        error_info: 错误信息
    """
    
    saga_id: str
    definition: SagaDefinition
    context: Dict[str, Any] = field(default_factory=dict)
    status: SagaStatus = SagaStatus.STARTED
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    completed_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    compensation_steps: List[str] = field(default_factory=list)
    error_info: Optional[str] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含实例数据的字典
        """
        return {
            "saga_id": self.saga_id,
            "definition": self.definition.to_dict(),
            "context": self.context,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "compensation_steps": self.compensation_steps,
            "error_info": self.error_info,
            "step_results": {
                name: result.to_dict() 
                for name, result in self.step_results.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagaInstance":
        """
        从字典创建实例
        
        Args:
            data: 包含实例数据的字典
            
        Returns:
            SagaInstance实例
        """
        # 重建SagaDefinition
        def_data = data.get("definition", {})
        steps = [
            SagaStep(name=s["name"])
            for s in def_data.get("steps", [])
        ]
        definition = SagaDefinition(
            name=def_data.get("name", ""),
            steps=steps
        )
        
        # 重建StepResult
        step_results = {}
        for name, result_data in data.get("step_results", {}).items():
            step_results[name] = StepResult(
                success=result_data.get("success", False),
                data=result_data.get("data"),
                error=result_data.get("error"),
                duration_ms=result_data.get("duration_ms", 0.0)
            )
        
        return cls(
            saga_id=data["saga_id"],
            definition=definition,
            context=data.get("context", {}),
            status=SagaStatus(data.get("status", "started")),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            completed_steps=data.get("completed_steps", []),
            current_step=data.get("current_step"),
            compensation_steps=data.get("compensation_steps", []),
            error_info=data.get("error_info"),
            step_results=step_results
        )
        
    def mark_step_completed(self, step_name: str, result: StepResult) -> None:
        """
        标记步骤完成
        
        Args:
            step_name: 步骤名称
            result: 执行结果
        """
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        self.step_results[step_name] = result
        
    def mark_step_failed(self, step_name: str, error: str) -> None:
        """
        标记步骤失败
        
        Args:
            step_name: 步骤名称
            error: 错误信息
        """
        self.step_results[step_name] = StepResult(
            success=False,
            error=error
        )
        self.error_info = error
        
    def mark_compensated(self, step_name: str) -> None:
        """
        标记步骤已补偿
        
        Args:
            step_name: 步骤名称
        """
        if step_name not in self.compensation_steps:
            self.compensation_steps.append(step_name)
            
    def complete(self) -> None:
        """标记Saga完成"""
        self.status = SagaStatus.COMPLETED
        self.end_time = datetime.now()
        
    def fail(self, error: str) -> None:
        """
        标记Saga失败
        
        Args:
            error: 错误信息
        """
        self.status = SagaStatus.FAILED
        self.error_info = error
        self.end_time = datetime.now()
        
    def start_compensating(self) -> None:
        """开始补偿"""
        self.status = SagaStatus.COMPENSATING
        
    def complete_compensation(self) -> None:
        """完成补偿"""
        self.status = SagaStatus.COMPENSATED
        self.end_time = datetime.now()
        
    def fail_compensation(self, error: str) -> None:
        """
        标记补偿失败
        
        Args:
            error: 错误信息
        """
        self.status = SagaStatus.COMPENSATION_FAILED
        self.error_info = error
        self.end_time = datetime.now()

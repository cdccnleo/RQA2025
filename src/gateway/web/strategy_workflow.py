"""
策略工作流引擎模块
提供策略生命周期管理和工作流状态转换
"""

import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# 工作流状态定义
class WorkflowStatus(Enum):
    """工作流状态枚举"""
    DESIGN = "design"           # 设计阶段
    BACKTEST = "backtest"       # 回测阶段
    OPTIMIZE = "optimize"       # 优化阶段
    APPLY = "apply"             # 应用阶段
    READY = "ready"             # 就绪阶段
    FAILED = "failed"           # 失败状态
    PAUSED = "paused"           # 暂停状态

# 状态转换规则
STATE_TRANSITIONS = {
    WorkflowStatus.DESIGN: [WorkflowStatus.BACKTEST],
    WorkflowStatus.BACKTEST: [WorkflowStatus.OPTIMIZE, WorkflowStatus.READY, WorkflowStatus.FAILED],
    WorkflowStatus.OPTIMIZE: [WorkflowStatus.APPLY, WorkflowStatus.READY, WorkflowStatus.FAILED],
    WorkflowStatus.APPLY: [WorkflowStatus.READY, WorkflowStatus.FAILED],
    WorkflowStatus.READY: [WorkflowStatus.BACKTEST, WorkflowStatus.OPTIMIZE],
    WorkflowStatus.FAILED: [WorkflowStatus.DESIGN, WorkflowStatus.BACKTEST, WorkflowStatus.OPTIMIZE],
    WorkflowStatus.PAUSED: [WorkflowStatus.DESIGN, WorkflowStatus.BACKTEST, WorkflowStatus.OPTIMIZE]
}


@dataclass
class WorkflowStep:
    """工作流步骤"""
    step_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowStep':
        return cls(**data)


@dataclass
class StrategyWorkflow:
    """策略工作流实例"""
    workflow_id: str
    strategy_id: str
    strategy_name: str
    current_status: WorkflowStatus
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'workflow_id': self.workflow_id,
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'current_status': self.current_status.value,
            'steps': [step.to_dict() for step in self.steps],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'completed_at': self.completed_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyWorkflow':
        workflow = cls(
            workflow_id=data['workflow_id'],
            strategy_id=data['strategy_id'],
            strategy_name=data['strategy_name'],
            current_status=WorkflowStatus(data['current_status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            completed_at=data.get('completed_at'),
            metadata=data.get('metadata', {})
        )
        workflow.steps = [WorkflowStep.from_dict(step) for step in data.get('steps', [])]
        return workflow


class StrategyWorkflowEngine:
    """策略工作流引擎"""
    
    def __init__(self, workflow_dir: str = "data/workflows"):
        self.workflow_dir = workflow_dir
        self.active_workflows: Dict[str, StrategyWorkflow] = {}
        self._ensure_directory()
        self._load_active_workflows()
    
    def _ensure_directory(self):
        """确保工作流目录存在"""
        if not os.path.exists(self.workflow_dir):
            os.makedirs(self.workflow_dir)
            logger.info(f"创建工作流目录: {self.workflow_dir}")
    
    def _load_active_workflows(self):
        """加载活跃的工作流"""
        try:
            for filename in os.listdir(self.workflow_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.workflow_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            workflow = StrategyWorkflow.from_dict(data)
                            # 只加载未完成的
                            if workflow.current_status not in [WorkflowStatus.READY, WorkflowStatus.FAILED]:
                                self.active_workflows[workflow.workflow_id] = workflow
                    except Exception as e:
                        logger.warning(f"加载工作流文件失败 {filename}: {e}")
        except Exception as e:
            logger.error(f"加载工作流目录失败: {e}")
    
    def _save_workflow(self, workflow: StrategyWorkflow):
        """保存工作流到文件"""
        try:
            filepath = os.path.join(self.workflow_dir, f"{workflow.workflow_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存工作流失败 {workflow.workflow_id}: {e}")
    
    def create_workflow(self, strategy_id: str, strategy_name: str) -> StrategyWorkflow:
        """创建新工作流"""
        workflow_id = f"wf_{int(time.time())}_{strategy_id[:8]}"
        workflow = StrategyWorkflow(
            workflow_id=workflow_id,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            current_status=WorkflowStatus.DESIGN
        )
        
        # 添加初始步骤
        workflow.steps.append(WorkflowStep(
            step_name="design",
            status="completed",
            start_time=time.time(),
            end_time=time.time(),
            result={"message": "策略设计完成"}
        ))
        
        self.active_workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        logger.info(f"创建工作流: {workflow_id} for strategy: {strategy_id}")
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[StrategyWorkflow]:
        """获取工作流"""
        return self.active_workflows.get(workflow_id)
    
    def get_strategy_workflows(self, strategy_id: str) -> List[StrategyWorkflow]:
        """获取策略的所有工作流"""
        return [wf for wf in self.active_workflows.values() if wf.strategy_id == strategy_id]
    
    def get_latest_workflow(self, strategy_id: str) -> Optional[StrategyWorkflow]:
        """获取策略最新的工作流"""
        workflows = self.get_strategy_workflows(strategy_id)
        if not workflows:
            return None
        return max(workflows, key=lambda w: w.created_at)
    
    def can_transition(self, workflow: StrategyWorkflow, new_status: WorkflowStatus) -> bool:
        """检查状态转换是否允许"""
        return new_status in STATE_TRANSITIONS.get(workflow.current_status, [])
    
    def transition_status(self, workflow_id: str, new_status: WorkflowStatus, 
                         step_result: Optional[Dict] = None) -> bool:
        """转换工作流状态"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"工作流不存在: {workflow_id}")
            return False
        
        if not self.can_transition(workflow, new_status):
            logger.warning(f"状态转换不允许: {workflow.current_status.value} -> {new_status.value}")
            return False
        
        # 完成当前步骤
        if workflow.steps:
            current_step = workflow.steps[-1]
            if current_step.status == "running":
                current_step.status = "completed" if new_status != WorkflowStatus.FAILED else "failed"
                current_step.end_time = time.time()
                if step_result:
                    current_step.result = step_result
        
        # 更新状态
        old_status = workflow.current_status
        workflow.current_status = new_status
        workflow.updated_at = time.time()
        
        # 添加新步骤
        step_name = new_status.value
        workflow.steps.append(WorkflowStep(
            step_name=step_name,
            status="running" if new_status not in [WorkflowStatus.READY, WorkflowStatus.FAILED] else "completed",
            start_time=time.time()
        ))
        
        # 如果完成或失败，记录完成时间
        if new_status in [WorkflowStatus.READY, WorkflowStatus.FAILED]:
            workflow.completed_at = time.time()
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        self._save_workflow(workflow)
        
        logger.info(f"工作流状态转换: {workflow_id} {old_status.value} -> {new_status.value}")
        return True
    
    def start_step(self, workflow_id: str, step_name: str) -> bool:
        """开始一个步骤"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        workflow.steps.append(WorkflowStep(
            step_name=step_name,
            status="running",
            start_time=time.time()
        ))
        workflow.updated_at = time.time()
        self._save_workflow(workflow)
        return True
    
    def complete_step(self, workflow_id: str, result: Optional[Dict] = None) -> bool:
        """完成当前步骤"""
        workflow = self.get_workflow(workflow_id)
        if not workflow or not workflow.steps:
            return False
        
        current_step = workflow.steps[-1]
        if current_step.status == "running":
            current_step.status = "completed"
            current_step.end_time = time.time()
            if result:
                current_step.result = result
            workflow.updated_at = time.time()
            self._save_workflow(workflow)
            return True
        return False
    
    def fail_step(self, workflow_id: str, error_message: str) -> bool:
        """标记当前步骤为失败"""
        workflow = self.get_workflow(workflow_id)
        if not workflow or not workflow.steps:
            return False
        
        current_step = workflow.steps[-1]
        if current_step.status == "running":
            current_step.status = "failed"
            current_step.end_time = time.time()
            current_step.error_message = error_message
            workflow.updated_at = time.time()
            self._save_workflow(workflow)
            
            # 自动转换到失败状态
            self.transition_status(workflow_id, WorkflowStatus.FAILED)
            return True
        return False
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流"""
        return self.transition_status(workflow_id, WorkflowStatus.PAUSED)
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """恢复工作流"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        # 恢复到之前的状态
        if len(workflow.steps) >= 2:
            previous_step = workflow.steps[-2]
            previous_status = WorkflowStatus(previous_step.step_name)
            return self.transition_status(workflow_id, previous_status)
        return False
    
    def get_workflow_progress(self, workflow_id: str) -> Dict:
        """获取工作流进度"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return {"error": "工作流不存在"}
        
        total_steps = 4  # design, backtest, optimize, apply
        completed_steps = len([s for s in workflow.steps if s.status == "completed"])
        progress_percent = (completed_steps / total_steps) * 100
        
        return {
            "workflow_id": workflow_id,
            "strategy_id": workflow.strategy_id,
            "strategy_name": workflow.strategy_name,
            "current_status": workflow.current_status.value,
            "progress_percent": round(progress_percent, 1),
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "steps": [step.to_dict() for step in workflow.steps],
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "completed_at": workflow.completed_at
        }
    
    def list_workflows(self, strategy_id: Optional[str] = None, 
                      status: Optional[WorkflowStatus] = None) -> List[StrategyWorkflow]:
        """列出工作流"""
        workflows = list(self.active_workflows.values())
        
        if strategy_id:
            workflows = [w for w in workflows if w.strategy_id == strategy_id]
        
        if status:
            workflows = [w for w in workflows if w.current_status == status]
        
        return sorted(workflows, key=lambda w: w.updated_at, reverse=True)


# 全局工作流引擎实例
workflow_engine = StrategyWorkflowEngine()


# 便捷的API函数
def create_strategy_workflow(strategy_id: str, strategy_name: str) -> StrategyWorkflow:
    """创建策略工作流"""
    return workflow_engine.create_workflow(strategy_id, strategy_name)


def get_strategy_workflow_progress(workflow_id: str) -> Dict:
    """获取策略工作流进度"""
    return workflow_engine.get_workflow_progress(workflow_id)


def get_latest_strategy_workflow(strategy_id: str) -> Optional[StrategyWorkflow]:
    """获取策略最新的工作流"""
    return workflow_engine.get_latest_workflow(strategy_id)


def transition_workflow_status(workflow_id: str, new_status: str, 
                               step_result: Optional[Dict] = None) -> bool:
    """转换工作流状态"""
    try:
        status = WorkflowStatus(new_status)
        return workflow_engine.transition_status(workflow_id, status, step_result)
    except ValueError:
        logger.error(f"无效的工作流状态: {new_status}")
        return False

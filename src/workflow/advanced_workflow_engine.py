"""
高级工作流引擎 - 支持多级审批、条件分支、审批委托

本模块提供企业级的工作流引擎，支持：
1. 多级审批流程（串行、并行、混合）
2. 条件分支路由
3. 审批委托和代理
4. 超时处理和提醒
5. 流程监控和追踪

作者: 后端团队
创建日期: 2026-02-21
版本: 2.0.0
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict

from src.common.exceptions import WorkflowError


# 配置日志
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """工作流节点类型"""
    START = auto()           # 开始节点
    END = auto()             # 结束节点
    SERIAL_APPROVAL = auto() # 串行审批
    PARALLEL_APPROVAL = auto() # 并行审批
    CONDITIONAL = auto()     # 条件分支
    DELEGATION = auto()      # 委托审批
    NOTIFICATION = auto()    # 通知节点
    DELAY = auto()           # 延迟节点


class ApprovalStatus(Enum):
    """审批状态"""
    PENDING = "pending"      # 待审批
    APPROVED = "approved"    # 已通过
    REJECTED = "rejected"    # 已拒绝
    DELEGATED = "delegated"  # 已委托
    TIMEOUT = "timeout"      # 已超时
    SKIPPED = "skipped"      # 已跳过


class WorkflowStatus(Enum):
    """工作流状态"""
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    REJECTED = "rejected"    # 已拒绝
    TERMINATED = "terminated" # 已终止
    SUSPENDED = "suspended"  # 已暂停


@dataclass
class ApprovalRecord:
    """审批记录"""
    approver_id: str
    status: ApprovalStatus
    comment: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    delegated_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "approver_id": self.approver_id,
            "status": self.status.value,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat(),
            "delegated_to": self.delegated_to
        }


@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    node_type: NodeType
    name: str
    approvers: List[str] = field(default_factory=list)
    conditions: Optional[Dict[str, Any]] = None
    timeout_hours: int = 24
    delegation_allowed: bool = True
    next_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.name,
            "name": self.name,
            "approvers": self.approvers,
            "conditions": self.conditions,
            "timeout_hours": self.timeout_hours,
            "delegation_allowed": self.delegation_allowed,
            "next_nodes": self.next_nodes,
            "metadata": self.metadata
        }


@dataclass
class WorkflowInstance:
    """工作流实例"""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    context: Dict[str, Any]
    current_node_id: Optional[str] = None
    node_history: List[Dict] = field(default_factory=list)
    approval_records: Dict[str, List[ApprovalRecord]] = field(default_factory=lambda: defaultdict(list))
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "context": self.context,
            "current_node_id": self.current_node_id,
            "node_history": self.node_history,
            "approval_records": {
                k: [r.to_dict() for r in v]
                for k, v in self.approval_records.items()
            },
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class ConditionEvaluator:
    """条件评估器"""
    
    @staticmethod
    def evaluate(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        评估条件
        
        参数:
            condition: 条件定义
            context: 上下文数据
            
        返回:
            bool: 条件是否满足
            
        示例:
            >>> condition = {
            ...     "operator": "AND",
            ...     "conditions": [
            ...         {"field": "amount", "operator": ">", "value": 10000},
            ...         {"field": "type", "operator": "==", "value": "urgent"}
            ...     ]
            ... }
            >>> context = {"amount": 15000, "type": "urgent"}
            >>> ConditionEvaluator.evaluate(condition, context)
            True
        """
        if not condition:
            return True
        
        operator = condition.get("operator", "AND")
        
        if operator == "AND":
            return all(
                ConditionEvaluator.evaluate(c, context)
                for c in condition.get("conditions", [])
            )
        elif operator == "OR":
            return any(
                ConditionEvaluator.evaluate(c, context)
                for c in condition.get("conditions", [])
            )
        elif operator == "NOT":
            return not ConditionEvaluator.evaluate(
                condition.get("conditions", [{}])[0], context
            )
        else:
            # 简单条件
            field = condition.get("field")
            op = condition.get("operator", "==")
            value = condition.get("value")
            
            if field not in context:
                return False
            
            field_value = context[field]
            
            operators = {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                "in": lambda a, b: a in b,
                "not_in": lambda a, b: a not in b,
                "contains": lambda a, b: b in a if isinstance(a, (str, list)) else False,
            }
            
            if op in operators:
                return operators[op](field_value, value)
            else:
                logger.warning(f"未知的操作符: {op}")
                return False


class AdvancedWorkflowEngine:
    """
    高级工作流引擎
    
    功能:
    1. 多级审批流程（串行、并行、混合）
    2. 条件分支路由
    3. 审批委托和代理
    4. 超时处理和提醒
    5. 流程监控和追踪
    
    使用示例:
        # 定义工作流
        workflow_def = {
            "workflow_id": "strategy_approval",
            "nodes": [
                {
                    "node_id": "start",
                    "node_type": "START",
                    "name": "开始",
                    "next_nodes": ["level1_approval"]
                },
                {
                    "node_id": "level1_approval",
                    "node_type": "SERIAL_APPROVAL",
                    "name": "一级审批",
                    "approvers": ["manager_001"],
                    "next_nodes": ["condition_check"]
                },
                {
                    "node_id": "condition_check",
                    "node_type": "CONDITIONAL",
                    "name": "条件判断",
                    "conditions": {
                        "high_value": {"field": "amount", "operator": ">", "value": 100000},
                        "normal": {"field": "amount", "operator": "<=", "value": 100000}
                    },
                    "next_nodes": ["level2_approval", "end"]
                },
                {
                    "node_id": "level2_approval",
                    "node_type": "SERIAL_APPROVAL",
                    "name": "二级审批",
                    "approvers": ["director_001"],
                    "next_nodes": ["end"]
                },
                {
                    "node_id": "end",
                    "node_type": "END",
                    "name": "结束"
                }
            ]
        }
        
        # 启动工作流
        engine = AdvancedWorkflowEngine()
        instance = engine.start_workflow("strategy_approval", {"amount": 150000})
    """
    
    def __init__(self):
        """初始化工作流引擎"""
        self._workflows: Dict[str, Dict] = {}
        self._instances: Dict[str, WorkflowInstance] = {}
        self._node_definitions: Dict[str, WorkflowNode] = {}
        self._condition_evaluator = ConditionEvaluator()
        self._delegation_rules: Dict[str, str] = {}  # 委托规则
        self._timeout_handlers: Dict[str, asyncio.Task] = {}
        
        # 事件回调
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_workflow(self, workflow_definition: Dict[str, Any]):
        """
        注册工作流定义
        
        参数:
            workflow_definition: 工作流定义
            
        示例:
            >>> engine.register_workflow({
            ...     "workflow_id": "approval_flow",
            ...     "name": "审批流程",
            ...     "nodes": [...]
            ... })
        """
        workflow_id = workflow_definition.get("workflow_id")
        if not workflow_id:
            raise WorkflowError("工作流ID不能为空")
        
        # 验证节点定义
        nodes = workflow_definition.get("nodes", [])
        if not nodes:
            raise WorkflowError("工作流必须包含至少一个节点")
        
        # 存储工作流定义
        self._workflows[workflow_id] = workflow_definition
        
        # 解析节点定义
        for node_def in nodes:
            node = self._parse_node_definition(node_def)
            self._node_definitions[node.node_id] = node
        
        logger.info(f"工作流已注册: {workflow_id}")
    
    def start_workflow(
        self,
        workflow_id: str,
        context: Dict[str, Any],
        instance_id: Optional[str] = None
    ) -> WorkflowInstance:
        """
        启动工作流实例
        
        参数:
            workflow_id: 工作流ID
            context: 上下文数据
            instance_id: 实例ID（可选，自动生成）
            
        返回:
            WorkflowInstance: 工作流实例
        """
        if workflow_id not in self._workflows:
            raise WorkflowError(f"工作流不存在: {workflow_id}")
        
        # 创建实例
        instance = WorkflowInstance(
            instance_id=instance_id or str(uuid.uuid4()),
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            context=context
        )
        
        # 找到开始节点
        workflow_def = self._workflows[workflow_id]
        start_node = None
        for node_def in workflow_def.get("nodes", []):
            if node_def.get("node_type") == "START":
                start_node = node_def
                break
        
        if not start_node:
            raise WorkflowError("工作流缺少开始节点")
        
        # 存储实例
        self._instances[instance.instance_id] = instance
        
        # 执行开始节点
        self._execute_node(instance, start_node["node_id"])
        
        # 触发事件
        self._trigger_event("workflow_started", instance)
        
        logger.info(f"工作流实例已启动: {instance.instance_id}")
        
        return instance
    
    def approve(
        self,
        instance_id: str,
        approver_id: str,
        comment: str = "",
        delegate_to: Optional[str] = None
    ) -> WorkflowInstance:
        """
        审批通过
        
        参数:
            instance_id: 实例ID
            approver_id: 审批人ID
            comment: 审批意见
            delegate_to: 委托给（可选）
            
        返回:
            WorkflowInstance: 更新后的工作流实例
        """
        instance = self._get_instance(instance_id)
        
        if instance.status != WorkflowStatus.RUNNING:
            raise WorkflowError(f"工作流实例状态不正确: {instance.status}")
        
        current_node = self._node_definitions.get(instance.current_node_id)
        if not current_node:
            raise WorkflowError("当前节点不存在")
        
        # 检查审批人权限
        if approver_id not in current_node.approvers:
            # 检查是否是委托人
            if not self._is_delegated_approver(approver_id, current_node.approvers):
                raise WorkflowError(f"用户 {approver_id} 无权审批此节点")
        
        # 创建审批记录
        record = ApprovalRecord(
            approver_id=approver_id,
            status=ApprovalStatus.DELEGATED if delegate_to else ApprovalStatus.APPROVED,
            comment=comment,
            delegated_to=delegate_to
        )
        
        instance.approval_records[instance.current_node_id].append(record)
        
        # 处理委托
        if delegate_to:
            logger.info(f"审批已委托: {approver_id} -> {delegate_to}")
            return instance
        
        # 检查是否所有审批人都已审批（并行审批）
        if current_node.node_type == NodeType.PARALLEL_APPROVAL:
            approved_count = sum(
                1 for r in instance.approval_records[instance.current_node_id]
                if r.status == ApprovalStatus.APPROVED
            )
            if approved_count < len(current_node.approvers):
                # 还有审批人未审批
                return instance
        
        # 记录节点历史
        instance.node_history.append({
            "node_id": instance.current_node_id,
            "action": "approved",
            "approver_id": approver_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # 执行下一个节点
        self._execute_next_node(instance, current_node)
        
        # 触发事件
        self._trigger_event("node_approved", instance, current_node)
        
        return instance
    
    def reject(
        self,
        instance_id: str,
        approver_id: str,
        comment: str = ""
    ) -> WorkflowInstance:
        """
        审批拒绝
        
        参数:
            instance_id: 实例ID
            approver_id: 审批人ID
            comment: 拒绝原因
            
        返回:
            WorkflowInstance: 更新后的工作流实例
        """
        instance = self._get_instance(instance_id)
        
        if instance.status != WorkflowStatus.RUNNING:
            raise WorkflowError(f"工作流实例状态不正确: {instance.status}")
        
        current_node = self._node_definitions.get(instance.current_node_id)
        if not current_node:
            raise WorkflowError("当前节点不存在")
        
        # 创建审批记录
        record = ApprovalRecord(
            approver_id=approver_id,
            status=ApprovalStatus.REJECTED,
            comment=comment
        )
        
        instance.approval_records[instance.current_node_id].append(record)
        
        # 记录节点历史
        instance.node_history.append({
            "node_id": instance.current_node_id,
            "action": "rejected",
            "approver_id": approver_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # 结束工作流
        instance.status = WorkflowStatus.REJECTED
        instance.completed_at = datetime.now()
        
        # 触发事件
        self._trigger_event("workflow_rejected", instance, current_node)
        
        logger.info(f"工作流实例已拒绝: {instance_id}")
        
        return instance
    
    def set_delegation(self, approver_id: str, delegate_to: str, expire_at: Optional[datetime] = None):
        """
        设置审批委托
        
        参数:
            approver_id: 原审批人ID
            delegate_to: 被委托人ID
            expire_at: 过期时间（可选）
        """
        self._delegation_rules[approver_id] = {
            "delegate_to": delegate_to,
            "expire_at": expire_at
        }
        logger.info(f"审批委托已设置: {approver_id} -> {delegate_to}")
    
    def get_workflow_status(self, instance_id: str) -> Dict[str, Any]:
        """
        获取工作流状态
        
        参数:
            instance_id: 实例ID
            
        返回:
            Dict: 工作流状态信息
        """
        instance = self._get_instance(instance_id)
        current_node = self._node_definitions.get(instance.current_node_id)
        
        return {
            "instance_id": instance_id,
            "workflow_id": instance.workflow_id,
            "status": instance.status.value,
            "current_node": current_node.name if current_node else None,
            "current_node_id": instance.current_node_id,
            "progress": self._calculate_progress(instance),
            "created_at": instance.created_at.isoformat(),
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "context": instance.context
        }
    
    def on_event(self, event_name: str, handler: Callable):
        """
        注册事件处理器
        
        参数:
            event_name: 事件名称
            handler: 处理函数
        """
        self._event_handlers[event_name].append(handler)
    
    def _parse_node_definition(self, node_def: Dict[str, Any]) -> WorkflowNode:
        """解析节点定义"""
        return WorkflowNode(
            node_id=node_def["node_id"],
            node_type=NodeType[node_def["node_type"]],
            name=node_def["name"],
            approvers=node_def.get("approvers", []),
            conditions=node_def.get("conditions"),
            timeout_hours=node_def.get("timeout_hours", 24),
            delegation_allowed=node_def.get("delegation_allowed", True),
            next_nodes=node_def.get("next_nodes", []),
            metadata=node_def.get("metadata", {})
        )
    
    def _get_instance(self, instance_id: str) -> WorkflowInstance:
        """获取工作流实例"""
        if instance_id not in self._instances:
            raise WorkflowError(f"工作流实例不存在: {instance_id}")
        return self._instances[instance_id]
    
    def _is_delegated_approver(self, approver_id: str, allowed_approvers: List[str]) -> bool:
        """检查是否是委托审批人"""
        for allowed in allowed_approvers:
            if allowed in self._delegation_rules:
                delegation = self._delegation_rules[allowed]
                if delegation["delegate_to"] == approver_id:
                    # 检查是否过期
                    if delegation.get("expire_at") and datetime.now() > delegation["expire_at"]:
                        continue
                    return True
        return False
    
    def _execute_node(self, instance: WorkflowInstance, node_id: str):
        """执行节点"""
        node = self._node_definitions.get(node_id)
        if not node:
            raise WorkflowError(f"节点不存在: {node_id}")
        
        instance.current_node_id = node_id
        
        logger.info(f"执行节点: {node.name} ({node_id})")
        
        # 根据节点类型执行
        if node.node_type == NodeType.END:
            self._complete_workflow(instance)
        elif node.node_type == NodeType.CONDITIONAL:
            self._execute_conditional_node(instance, node)
        elif node.node_type in [NodeType.SERIAL_APPROVAL, NodeType.PARALLEL_APPROVAL]:
            self._setup_approval_timeout(instance, node)
        elif node.node_type == NodeType.DELEGATION:
            self._execute_delegation_node(instance, node)
        else:
            # 其他节点类型直接执行下一个
            self._execute_next_node(instance, node)
    
    def _execute_conditional_node(self, instance: WorkflowInstance, node: WorkflowNode):
        """执行条件节点"""
        conditions = node.conditions or {}
        
        for condition_name, condition_def in conditions.items():
            if self._condition_evaluator.evaluate(condition_def, instance.context):
                # 找到对应的下一个节点
                if node.next_nodes:
                    next_node_id = node.next_nodes[0]  # 简化处理，实际应该映射到具体节点
                    self._execute_node(instance, next_node_id)
                    return
        
        # 没有条件满足，执行默认分支
        if len(node.next_nodes) > 1:
            self._execute_node(instance, node.next_nodes[1])
        elif node.next_nodes:
            self._execute_node(instance, node.next_nodes[0])
    
    def _execute_delegation_node(self, instance: WorkflowInstance, node: WorkflowNode):
        """执行委托节点"""
        # 自动委托给指定人
        if node.approvers and len(node.approvers) > 0:
            delegate_to = node.approvers[0]
            logger.info(f"自动委托给: {delegate_to}")
            # 继续执行下一个节点
            self._execute_next_node(instance, node)
    
    def _execute_next_node(self, instance: WorkflowInstance, current_node: WorkflowNode):
        """执行下一个节点"""
        if not current_node.next_nodes:
            # 没有下一个节点，结束工作流
            self._complete_workflow(instance)
            return
        
        # 找到下一个节点
        next_node_id = current_node.next_nodes[0]
        self._execute_node(instance, next_node_id)
    
    def _setup_approval_timeout(self, instance: WorkflowInstance, node: WorkflowNode):
        """设置审批超时"""
        if node.timeout_hours <= 0:
            return
        
        async def timeout_handler():
            await asyncio.sleep(node.timeout_hours * 3600)
            await self._handle_timeout(instance.instance_id, node.node_id)
        
        # 创建超时任务
        task = asyncio.create_task(timeout_handler())
        self._timeout_handlers[f"{instance.instance_id}:{node.node_id}"] = task
    
    async def _handle_timeout(self, instance_id: str, node_id: str):
        """处理超时"""
        try:
            instance = self._get_instance(instance_id)
            if instance.status != WorkflowStatus.RUNNING:
                return
            
            if instance.current_node_id != node_id:
                return
            
            # 记录超时
            record = ApprovalRecord(
                approver_id="system",
                status=ApprovalStatus.TIMEOUT,
                comment="审批超时"
            )
            instance.approval_records[node_id].append(record)
            
            logger.warning(f"节点审批超时: {instance_id}:{node_id}")
            
            # 触发事件
            self._trigger_event("node_timeout", instance, self._node_definitions.get(node_id))
            
        except Exception as e:
            logger.error(f"处理超时失败: {str(e)}")
    
    def _complete_workflow(self, instance: WorkflowInstance):
        """完成工作流"""
        instance.status = WorkflowStatus.COMPLETED
        instance.completed_at = datetime.now()
        
        # 取消所有超时任务
        for key, task in list(self._timeout_handlers.items()):
            if key.startswith(instance.instance_id):
                task.cancel()
                del self._timeout_handlers[key]
        
        # 触发事件
        self._trigger_event("workflow_completed", instance)
        
        logger.info(f"工作流实例已完成: {instance.instance_id}")
    
    def _calculate_progress(self, instance: WorkflowInstance) -> float:
        """计算工作流进度"""
        workflow_def = self._workflows.get(instance.workflow_id)
        if not workflow_def:
            return 0.0
        
        total_nodes = len(workflow_def.get("nodes", []))
        completed_nodes = len(instance.node_history)
        
        return min(100.0, (completed_nodes / total_nodes) * 100) if total_nodes > 0 else 0.0
    
    def _trigger_event(self, event_name: str, instance: WorkflowInstance, node: Optional[WorkflowNode] = None):
        """触发事件"""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    if node:
                        handler(instance, node)
                    else:
                        handler(instance)
                except Exception as e:
                    logger.error(f"事件处理失败: {str(e)}")


# 便捷函数
def create_simple_approval_workflow(
    workflow_id: str,
    approvers: List[str],
    name: str = "简单审批流程"
) -> Dict[str, Any]:
    """
    创建简单审批工作流
    
    参数:
        workflow_id: 工作流ID
        approvers: 审批人列表
        name: 工作流名称
        
    返回:
        Dict: 工作流定义
    """
    return {
        "workflow_id": workflow_id,
        "name": name,
        "nodes": [
            {
                "node_id": "start",
                "node_type": "START",
                "name": "开始",
                "next_nodes": ["approval"]
            },
            {
                "node_id": "approval",
                "node_type": "SERIAL_APPROVAL",
                "name": "审批",
                "approvers": approvers,
                "next_nodes": ["end"]
            },
            {
                "node_id": "end",
                "node_type": "END",
                "name": "结束"
            }
        ]
    }


def create_multi_level_workflow(
    workflow_id: str,
    levels: List[Dict[str, Any]],
    name: str = "多级审批流程"
) -> Dict[str, Any]:
    """
    创建多级审批工作流
    
    参数:
        workflow_id: 工作流ID
        levels: 审批层级配置列表
        name: 工作流名称
        
    返回:
        Dict: 工作流定义
        
    示例:
        >>> levels = [
        ...     {"name": "一级审批", "approvers": ["manager_001"]},
        ...     {"name": "二级审批", "approvers": ["director_001"]}
        ... ]
        >>> workflow = create_multi_level_workflow("approval_001", levels)
    """
    nodes = [
        {
            "node_id": "start",
            "node_type": "START",
            "name": "开始",
            "next_nodes": ["level_1"]
        }
    ]
    
    for i, level in enumerate(levels, 1):
        next_node = f"level_{i+1}" if i < len(levels) else "end"
        nodes.append({
            "node_id": f"level_{i}",
            "node_type": "SERIAL_APPROVAL",
            "name": level["name"],
            "approvers": level["approvers"],
            "next_nodes": [next_node]
        })
    
    nodes.append({
        "node_id": "end",
        "node_type": "END",
        "name": "结束"
    })
    
    return {
        "workflow_id": workflow_id,
        "name": name,
        "nodes": nodes
    }

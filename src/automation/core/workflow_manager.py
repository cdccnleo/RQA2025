"""
Workflow Manager Module
工作流管理器模块

This module provides workflow management capabilities for automation processes
此模块为自动化流程提供工作流管理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import threading
import time
import networkx as nx

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):

    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):

    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowTask:

    """
    Workflow Task Class
    工作流任务类

    Represents a single task within a workflow
    表示工作流中的单个任务
    """

    def __init__(self,


                 task_id: str,
                 name: str,
                 task_type: str,
                 config: Dict[str, Any],
                 dependencies: Optional[List[str]] = None):
        """
        Initialize workflow task
        初始化工作流任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Human - readable task name
                 人类可读的任务名称
            task_type: Type of task to execute
                      要执行的任务类型
            config: Task configuration
                   任务配置
            dependencies: List of task IDs this task depends on
                        此任务依赖的任务ID列表
        """
        self.task_id = task_id
        self.name = name
        self.task_type = task_type
        self.config = config
        self.dependencies = dependencies or []

        # Runtime state
        self.status = TaskStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.execution_time = 0.0
        self.retry_count = 0

        # Performance tracking
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 300)  # 5 minutes default


class Workflow:

    """
    Workflow Class
    工作流类

    Represents a complete automation workflow with tasks and dependencies
    表示具有任务和依赖关系的完整自动化工作流
    """

    def __init__(self,


                 workflow_id: str,
                 name: str,
                 description: str = "",
                 tasks: Optional[Dict[str, WorkflowTask]] = None):
        """
        Initialize workflow
        初始化工作流

        Args:
            workflow_id: Unique workflow identifier
                        唯一工作流标识符
            name: Human - readable workflow name
                 人类可读的工作流名称
            description: Workflow description
                        工作流描述
            tasks: Dictionary of workflow tasks
                  工作流任务字典
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.tasks = tasks or {}

        # Workflow state
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.execution_time = 0.0

        # Execution tracking
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.running_tasks = set()

        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()

    def add_task(self, task: WorkflowTask) -> None:
        """
        Add a task to the workflow
        将任务添加到工作流中

        Args:
            task: Task to add
                 要添加的任务
        """
        self.tasks[task.task_id] = task
        self.dependency_graph = self._build_dependency_graph()
        logger.info(f"Added task {task.task_id} to workflow {self.workflow_id}")

    def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the workflow
        从工作流中移除任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.dependency_graph = self._build_dependency_graph()
            logger.info(f"Removed task {task_id} from workflow {self.workflow_id}")
            return True
        return False

    def get_executable_tasks(self) -> List[str]:
        """
        Get tasks that are ready to execute (all dependencies satisfied)
        获取准备执行的任务（所有依赖关系都满足）

        Returns:
            list: List of executable task IDs
                  可执行任务ID列表
        """
        executable = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_satisfied = all(
                    dep_task_id in self.completed_tasks
                    for dep_task_id in task.dependencies
                )

                if deps_satisfied:
                    executable.append(task_id)

        return executable

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get comprehensive workflow status
        获取全面的工作流状态

        Returns:
            dict: Workflow status information
                  工作流状态信息
        """
        total_tasks = len(self.tasks)
        pending_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        running_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time': self.execution_time,
            'total_tasks': total_tasks,
            'pending_tasks': pending_tasks,
            'running_tasks': running_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'progress_percentage': (completed_tasks / max(total_tasks, 1)) * 100
        }

    def _build_dependency_graph(self) -> nx.DiGraph:
        """
        Build dependency graph for the workflow
        为工作流构建依赖关系图

        Returns:
            networkx.DiGraph: Dependency graph
                             依赖关系图
        """
        graph = nx.DiGraph()

        # Add all tasks as nodes
        for task_id in self.tasks:
            graph.add_node(task_id)

        # Add dependencies as edges
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep in self.tasks:  # Only add edge if dependency exists
                    graph.add_edge(dep, task_id)

        return graph

    def validate_workflow(self) -> Dict[str, Any]:
        """
        Validate workflow structure and dependencies
        验证工作流结构和依赖关系

        Returns:
            dict: Validation results
                  验证结果
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'task_count': len(self.tasks)
        }

        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    validation_result['errors'].append(
                        f"Task {task_id} depends on non - existent task {dep}"
                    )
                    validation_result['valid'] = False

        # Check for circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                validation_result['errors'].append(f"Circular dependencies detected: {cycles}")
                validation_result['valid'] = False
        except nx.NetworkXError:
            pass  # No cycles found

        # Check for isolated tasks (warning)
        isolated = [node for node in self.dependency_graph.nodes()
                    if self.dependency_graph.degree(node) == 0]
        if len(isolated) > 1:  # More than one isolated task
            validation_result['warnings'].append(
                f"Multiple isolated tasks detected: {isolated}"
            )

        return validation_result


class WorkflowManager:

    """
    Workflow Manager Class
    工作流管理器类

    Manages the execution of automation workflows
    管理工作流的执行
    """

    def __init__(self, manager_name: str = "default_workflow_manager"):
        """
        Initialize workflow manager
        初始化工作流管理器

        Args:
            manager_name: Name of the workflow manager
                        工作流管理器的名称
        """
        self.manager_name = manager_name
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, threading.Thread] = {}

        # Execution settings
        self.max_concurrent_workflows = 5
        self.task_execution_timeout = 300  # 5 minutes
        self.workflow_check_interval = 10  # seconds

        # Statistics
        self.execution_stats = {
            'total_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0
        }

        logger.info(f"Workflow manager {manager_name} initialized")

    def create_workflow(self,


                        workflow_id: str,
                        name: str,
                        description: str = "") -> str:
        """
        Create a new workflow
        创建新工作流

        Args:
            workflow_id: Unique workflow identifier
                        唯一工作流标识符
            name: Workflow name
                 工作流名称
            description: Workflow description
                        工作流描述

        Returns:
            str: Created workflow ID
                 创建的工作流ID
        """
        workflow = Workflow(workflow_id, name, description)
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow_id

    def add_task_to_workflow(self,


                             workflow_id: str,
                             task: WorkflowTask) -> bool:
        """
        Add a task to an existing workflow
        将任务添加到现有工作流中

        Args:
            workflow_id: Workflow identifier
                        工作流标识符
            task: Task to add
                 要添加的任务

        Returns:
            bool: True if added successfully, False otherwise
                  添加成功返回True，否则返回False
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False

        self.workflows[workflow_id].add_task(task)
        return True

    def execute_workflow(self,


                         workflow_id: str,
                         async_execution: bool = True) -> Dict[str, Any]:
        """
        Execute a workflow
        执行工作流

        Args:
            workflow_id: Workflow identifier
                        工作流标识符
            async_execution: Whether to execute asynchronously
                           是否异步执行

        Returns:
            dict: Execution result
                  执行结果
        """
        if workflow_id not in self.workflows:
            return {'success': False, 'error': f'Workflow {workflow_id} not found'}

        workflow = self.workflows[workflow_id]

        # Validate workflow before execution
        validation = workflow.validate_workflow()
        if not validation['valid']:
            return {
                'success': False,
                'error': 'Workflow validation failed',
                'validation_errors': validation['errors']
            }

        # Check concurrent workflow limit
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            return {
                'success': False,
                'error': 'Maximum concurrent workflows reached'
            }

        if async_execution:
            # Start async execution
            execution_thread = threading.Thread(
                target=self._execute_workflow_sync,
                args=(workflow_id,),
                daemon=True
            )
            self.active_workflows[workflow_id] = execution_thread
            execution_thread.start()

            return {
                'success': True,
                'execution_mode': 'async',
                'workflow_id': workflow_id
            }
        else:
            # Execute synchronously
            return self._execute_workflow_sync(workflow_id)

    def _execute_workflow_sync(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute workflow synchronously
        同步执行工作流

        Args:
            workflow_id: Workflow identifier
                        工作流标识符

        Returns:
            dict: Execution result
                  执行结果
        """
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()

        execution_result = {
            'workflow_id': workflow_id,
            'success': True,
            'start_time': workflow.started_at,
            'executed_tasks': [],
            'failed_tasks': [],
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            while True:
                # Get executable tasks
                executable_tasks = workflow.get_executable_tasks()

                if not executable_tasks:
                    # Check if workflow is complete
                    if all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]
                           for task in workflow.tasks.values()):
                        break

                    # Wait for running tasks to complete
                    time.sleep(1)
                    continue

                # Execute tasks (for now, just mark as completed)
                # In a real implementation, this would execute actual tasks
                for task_id in executable_tasks:
                    task = workflow.tasks[task_id]
                    task.status = TaskStatus.RUNNING
                    task.start_time = datetime.now()

                    try:
                        # Simulate task execution
                        time.sleep(0.1)  # Simulate processing time

                        task.status = TaskStatus.COMPLETED
                        task.end_time = datetime.now()
                        task.execution_time = (task.end_time - task.start_time).total_seconds()
                        workflow.completed_tasks.add(task_id)

                        execution_result['executed_tasks'].append({
                            'task_id': task_id,
                            'execution_time': task.execution_time
                        })

                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = e
                        workflow.failed_tasks.add(task_id)

                        execution_result['failed_tasks'].append({
                            'task_id': task_id,
                            'error': str(e)
                        })

            # Determine final workflow status
            if workflow.failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                execution_result['success'] = False
            else:
                workflow.status = WorkflowStatus.COMPLETED

            workflow.completed_at = datetime.now()
            workflow.execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
            execution_result['execution_time'] = workflow.execution_time
            execution_result['end_time'] = workflow.completed_at

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            execution_result['success'] = False
            execution_result['error'] = str(e)
            logger.error(f"Workflow execution failed: {str(e)}")

        # Update statistics
        self._update_execution_stats(
            execution_result['success'], execution_result['execution_time'])

        # Clean up
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]

        return execution_result

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow
        取消正在运行的工作流

        Args:
            workflow_id: Workflow identifier
                    工作流标识符

        Returns:
            bool: True if cancelled successfully, False otherwise
                  取消成功返回True，否则返回False
        """
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]

        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED

            # Cancel running tasks
            for task in workflow.tasks.values():
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.CANCELLED

            logger.info(f"Cancelled workflow: {workflow_id}")
            return True

        return False

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific workflow
        获取特定工作流的状态

        Args:
            workflow_id: Workflow identifier
                        工作流标识符

        Returns:
            dict: Workflow status or None if not found
                  工作流状态，如果未找到则返回None
        """
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]
        return workflow.get_workflow_status()

    def list_workflows(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all workflows with optional status filter
        列出所有工作流，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of workflow summaries
                  工作流摘要列表
        """
        workflows = []

        for workflow_id, workflow in self.workflows.items():
            status = workflow.get_workflow_status()

            if status_filter is None or status['status'] == status_filter:
                workflows.append({
                    'workflow_id': workflow_id,
                    'name': workflow.name,
                    'status': status['status'],
                    'created_at': workflow.created_at.isoformat(),
                    'execution_time': workflow.execution_time,
                    'progress_percentage': status['progress_percentage']
                })

        return workflows

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get workflow manager statistics
        获取工作流管理器统计信息

        Returns:
            dict: Manager statistics
                  管理器统计信息
        """
        active_workflows = len(self.active_workflows)
        total_workflows = len(self.workflows)

        return {
            'manager_name': self.manager_name,
            'total_workflows': total_workflows,
            'active_workflows': active_workflows,
            'max_concurrent_workflows': self.max_concurrent_workflows,
            'execution_stats': self.execution_stats,
            'workflow_summary': {
                status: len([w for w in self.workflows.values() if w.status.value == status])
                for status in WorkflowStatus.__members__.keys()
            }
        }

    def _update_execution_stats(self, success: bool, execution_time: float) -> None:
        """
        Update execution statistics
        更新执行统计信息

        Args:
            success: Whether execution was successful
                    执行是否成功
            execution_time: Execution time
                           执行时间
        """
        self.execution_stats['total_workflows'] += 1

        if success:
            self.execution_stats['completed_workflows'] += 1
        else:
            self.execution_stats['failed_workflows'] += 1

        # Update average execution time
        total_workflows = self.execution_stats['total_workflows']
        current_avg = self.execution_stats['average_execution_time']
        self.execution_stats['average_execution_time'] = (
            (current_avg * (total_workflows - 1)) + execution_time
        ) / total_workflows

    def export_workflow(self, workflow_id: str, filepath: str) -> bool:
        """
        Export workflow definition to file
        将工作流定义导出到文件

        Args:
            workflow_id: Workflow identifier
                        工作流标识符
            filepath: Export file path
                     导出文件路径

        Returns:
            bool: True if export successful
                  导出成功返回True
        """
        try:
            import json

            if workflow_id not in self.workflows:
                return False

            workflow = self.workflows[workflow_id]

            export_data = {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'created_at': workflow.created_at.isoformat(),
                'tasks': {}
            }

            for task_id, task in workflow.tasks.items():
                export_data['tasks'][task_id] = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'task_type': task.task_type,
                    'config': task.config,
                    'dependencies': task.dependencies
                }

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Workflow {workflow_id} exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export workflow: {str(e)}")
            return False


# Global workflow manager instance
# 全局工作流管理器实例
workflow_manager = WorkflowManager()

__all__ = [
    'WorkflowStatus',
    'TaskStatus',
    'WorkflowTask',
    'Workflow',
    'WorkflowManager',
    'workflow_manager'
]

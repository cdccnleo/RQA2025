#!/usr/bin/env python3
"""
RQA2025 自动化层简化引擎
Automation Layer Simple Engine

提供简化的自动化引擎实现。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .automation_models import AutomationRule, AutomationTask, ExecutionStatus, AutomationMetrics
from .rule_manager import RuleManager
from .rule_executor import RuleExecutor

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
except Exception as e:
    models_adapter = None

# 日志记录
try:
    from src.infrastructure.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)


class SimpleAutomationEngine:

    """
    简化自动化引擎
    提供基本的自动化功能
    """

    def __init__(self, engine_id: str, config: Optional[Dict[str, Any]] = None):

        self.engine_id = engine_id
        self.config = config or {}

        # 组件初始化
        self.rule_manager = RuleManager(self.config.get('rule_manager_config', {}))
        self.rule_executor = RuleExecutor(self.config.get('rule_executor_config', {}))

        # 任务存储
        self.tasks: Dict[str, AutomationTask] = {}

        # 引擎状态
        self.is_running = False

        # 指标收集
        self.metrics = AutomationMetrics()

        logger.info(f"简化自动化引擎 {engine_id} 已初始化")

    async def start_engine(self):
        """启动引擎"""
        if self.is_running:
            return

        self.is_running = True
        logger.info(f"启动简化自动化引擎 {self.engine_id}")

    async def stop_engine(self):
        """停止引擎"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info(f"停止简化自动化引擎 {self.engine_id}")

    async def create_rule(self, rule_data: Dict[str, Any]) -> AutomationRule:
        """创建规则"""
        return self.rule_manager.create_rule(rule_data)

    async def trigger_rules(self, context: Dict[str, Any]) -> List[str]:
        """触发规则评估"""
        if not self.is_running:
            logger.warning("引擎未运行，无法触发规则")
            return []

        # 评估规则
        triggered_rules = await self.rule_manager.evaluate_rules(context)

        # 创建并执行任务
        task_ids = []
        for triggered_rule in triggered_rules:
            rule = triggered_rule['rule']
            task_context = triggered_rule['context']

            task = AutomationTask(
                task_id=f"task_{rule.rule_id}_{datetime.now().strftime('%Y % m % d % H % M % S % f')}",
                rule_id=rule.rule_id,
                name=f"{rule.name} - 自动任务",
                description=f"根据规则 {rule.name} 自动生成的任务",
                parameters=task_context,
                priority=rule.priority
            )

            # 存储任务
            self.tasks[task.task_id] = task
            task_ids.append(task.task_id)

            # 异步执行任务
            asyncio.create_task(self._execute_task_async(task, rule))

            self.metrics.total_tasks += 1

        logger.info(f"触发了 {len(task_ids)} 个自动化任务")
        return task_ids

    async def _execute_task_async(self, task: AutomationTask, rule: AutomationRule):
        """异步执行任务"""
        task.started_at = datetime.now()
        task.status = ExecutionStatus.RUNNING
        self.metrics.running_tasks += 1

        try:
            # 执行规则动作
            start_time = datetime.now()
            result = await self.rule_executor.execute_actions(task, rule)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # 更新任务状态
            task.status = ExecutionStatus.COMPLETED
            task.completed_at = datetime.now()
            task.execution_time_ms = execution_time
            task.result = result

            self.metrics.completed_tasks += 1

        except Exception as e:
            task.status = ExecutionStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)

            self.metrics.failed_tasks += 1

            logger.error(f"任务 {task.task_id} 执行失败: {str(e)}")

        finally:
            self.metrics.running_tasks -= 1

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'task_id': task.task_id,
            'status': task.status.value,
            'rule_id': task.rule_id,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'execution_time_ms': task.execution_time_ms,
            'error_message': task.error_message
        }

    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'engine_id': self.engine_id,
            'is_running': self.is_running,
            'total_tasks': len(self.tasks),
            'metrics': {
                'total_tasks': self.metrics.total_tasks,
                'completed_tasks': self.metrics.completed_tasks,
                'failed_tasks': self.metrics.failed_tasks,
                'running_tasks': self.metrics.running_tasks
            },
            'rule_manager_status': self.rule_manager.get_metrics()
        }


__all__ = [
    'SimpleAutomationEngine'
]

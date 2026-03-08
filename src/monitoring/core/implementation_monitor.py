#!/usr/bin/env python3
"""
RQA2025 实施进度监控器
Implementation Progress Monitor

监控项目实施进度和质量指标。
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

# 获取统一基础设施集成层的日志适配器
try:
    from src.infrastructure.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception as e:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except Exception:
        logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:

    """任务进度"""
    task_id: str
    task_name: str
    description: str
    status: str  # pending, in_progress, completed, blocked, cancelled
    priority: str  # low, medium, high, critical
    assignee: Optional[str] = None
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    progress_percent: int = 0
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Milestone:

    """里程碑"""
    milestone_id: str
    name: str
    description: str
    target_date: datetime
    status: str  # pending, achieved, delayed, cancelled
    tasks: List[str] = field(default_factory=list)
    achieved_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetric:

    """质量指标"""
    metric_id: str
    name: str
    description: str
    category: str  # code_quality, test_coverage, performance, security, etc.
    target_value: Any
    current_value: Any
    unit: str
    trend: str  # improving, stable, declining
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ImplementationDashboard:

    """实施仪表板"""
    dashboard_id: str
    name: str
    description: str
    tasks: List[TaskProgress] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    metrics: List[QualityMetric] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class ImplementationMonitor:

    """
    实施进度监控器
    跟踪项目实施进度、质量指标和里程碑达成情况
    """

    def __init__(self, data_dir: str = "data/monitoring"):

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.dashboards: Dict[str, ImplementationDashboard] = {}

        # 数据文件
        self.dashboard_file = self.data_dir / "implementation_dashboard.json"

        # 加载现有数据
        self._load_dashboard_data()

        logger.info(f"实施进度监控器已初始化，数据目录: {self.data_dir}")

    def create_dashboard(self, dashboard_id: str, name: str, description: str) -> ImplementationDashboard:
        """创建实施仪表板"""
        dashboard = ImplementationDashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description
        )

        self.dashboards[dashboard_id] = dashboard
        self._save_dashboard_data()

        logger.info(f"创建实施仪表板: {dashboard_id} - {name}")
        return dashboard

    def add_task(self, dashboard_id: str, task: TaskProgress) -> bool:
        """添加任务"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]
        dashboard.tasks.append(task)
        dashboard.last_updated = datetime.now()

        self._save_dashboard_data()

        logger.info(f"添加任务: {task.task_id} 到仪表板 {dashboard_id}")
        return True

    def update_task_progress(self, dashboard_id: str, task_id: str,


                             progress_percent: int, status: Optional[str] = None) -> bool:
        """更新任务进度"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]

        for task in dashboard.tasks:
            if task.task_id == task_id:
                task.progress_percent = progress_percent
                task.updated_at = datetime.now()

                if status:
                    task.status = status
                    if status == "completed" and not task.completed_date:
                        task.completed_date = datetime.now()

                dashboard.last_updated = datetime.now()
                self._save_dashboard_data()

                logger.info(f"更新任务进度: {task_id} - {progress_percent}%")
                return True

        return False

    def add_milestone(self, dashboard_id: str, milestone: Milestone) -> bool:
        """添加里程碑"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]
        dashboard.milestones.append(milestone)
        dashboard.last_updated = datetime.now()

        self._save_dashboard_data()

        logger.info(f"添加里程碑: {milestone.milestone_id} 到仪表板 {dashboard_id}")
        return True

    def update_milestone_status(self, dashboard_id: str, milestone_id: str, status: str) -> bool:
        """更新里程碑状态"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]

        for milestone in dashboard.milestones:
            if milestone.milestone_id == milestone_id:
                milestone.status = status
                if status == "achieved" and not milestone.achieved_date:
                    milestone.achieved_date = datetime.now()

                dashboard.last_updated = datetime.now()
                self._save_dashboard_data()

                logger.info(f"更新里程碑状态: {milestone_id} - {status}")
                return True

        return False

    def add_quality_metric(self, dashboard_id: str, metric: QualityMetric) -> bool:
        """添加质量指标"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]
        dashboard.metrics.append(metric)
        dashboard.last_updated = datetime.now()

        self._save_dashboard_data()

        logger.info(f"添加质量指标: {metric.metric_id} 到仪表板 {dashboard_id}")
        return True

    def update_quality_metric(self, dashboard_id: str, metric_id: str,


                              current_value: Any, trend: Optional[str] = None) -> bool:
        """更新质量指标"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]

        for metric in dashboard.metrics:
            if metric.metric_id == metric_id:
                metric.current_value = current_value
                metric.last_updated = datetime.now()

                if trend:
                    metric.trend = trend

                dashboard.last_updated = datetime.now()
                self._save_dashboard_data()

                logger.info(f"更新质量指标: {metric_id} - {current_value}")
                return True

        return False

    def get_dashboard_summary(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """获取仪表板摘要"""
        if dashboard_id not in self.dashboards:
            return None

        dashboard = self.dashboards[dashboard_id]

        # 计算任务统计
        total_tasks = len(dashboard.tasks)
        completed_tasks = len([t for t in dashboard.tasks if t.status == "completed"])
        in_progress_tasks = len([t for t in dashboard.tasks if t.status == "in_progress"])
        blocked_tasks = len([t for t in dashboard.tasks if t.status == "blocked"])

        # 计算里程碑统计
        total_milestones = len(dashboard.milestones)
        achieved_milestones = len([m for m in dashboard.milestones if m.status == "achieved"])
        delayed_milestones = len([m for m in dashboard.milestones if m.status == "delayed"])

        # 计算整体进度
        if total_tasks > 0:
            overall_progress = sum(t.progress_percent for t in dashboard.tasks) / total_tasks
        else:
            overall_progress = 0

        return {
            'dashboard_id': dashboard_id,
            'name': dashboard.name,
            'description': dashboard.description,
            'overall_progress_percent': round(overall_progress, 2),
            'task_summary': {
                'total': total_tasks,
                'completed': completed_tasks,
                'in_progress': in_progress_tasks,
                'blocked': blocked_tasks,
                'completion_rate': round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0
            },
            'milestone_summary': {
                'total': total_milestones,
                'achieved': achieved_milestones,
                'delayed': delayed_milestones,
                'achievement_rate': round(achieved_milestones / total_milestones * 100, 2) if total_milestones > 0 else 0
            },
            'quality_metrics': [
                {
                    'metric_id': m.metric_id,
                    'name': m.name,
                    'category': m.category,
                    'current_value': m.current_value,
                    'target_value': m.target_value,
                    'trend': m.trend,
                    'last_updated': m.last_updated.isoformat()
                }
                for m in dashboard.metrics
            ],
            'last_updated': dashboard.last_updated.isoformat()
        }

    def get_overdue_tasks(self, dashboard_id: str) -> List[Dict[str, Any]]:
        """获取逾期任务"""
        if dashboard_id not in self.dashboards:
            return []

        dashboard = self.dashboards[dashboard_id]
        current_time = datetime.now()

        overdue_tasks = []
        for task in dashboard.tasks:
            if (task.due_date
                and task.status != "completed"
                    and current_time > task.due_date):
                overdue_tasks.append({
                    'task_id': task.task_id,
                    'task_name': task.task_name,
                    'due_date': task.due_date.isoformat(),
                    'days_overdue': (current_time - task.due_date).days,
                    'status': task.status,
                    'assignee': task.assignee
                })

        return overdue_tasks

    def get_upcoming_milestones(self, dashboard_id: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """获取即将到来的里程碑"""
        if dashboard_id not in self.dashboards:
            return []

        dashboard = self.dashboards[dashboard_id]
        current_time = datetime.now()
        future_date = current_time + timedelta(days=days_ahead)

        upcoming_milestones = []
        for milestone in dashboard.milestones:
            if (milestone.target_date >= current_time
                and milestone.target_date <= future_date
                    and milestone.status == "pending"):
                upcoming_milestones.append({
                    'milestone_id': milestone.milestone_id,
                    'name': milestone.name,
                    'target_date': milestone.target_date.isoformat(),
                    'days_until': (milestone.target_date - current_time).days,
                    'task_count': len(milestone.tasks)
                })

        return sorted(upcoming_milestones, key=lambda x: x['days_until'])

    def generate_progress_report(self, dashboard_id: str) -> str:
        """生成进度报告"""
        summary = self.get_dashboard_summary(dashboard_id)
        if not summary:
            return "仪表板不存在"

        overdue_tasks = self.get_overdue_tasks(dashboard_id)
        upcoming_milestones = self.get_upcoming_milestones(dashboard_id)

        report = f"""# 实施进度报告

生成时间: {datetime.now().isoformat()}

# # 仪表板概览
- **项目名称**: {summary['name']}
- **描述**: {summary['description']}
- **整体进度**: {summary['overall_progress_percent']}%

# # 任务统计
- **总任务数**: {summary['task_summary']['total']}
- **已完成**: {summary['task_summary']['completed']}
- **进行中**: {summary['task_summary']['in_progress']}
- **受阻**: {summary['task_summary']['blocked']}
- **完成率**: {summary['task_summary']['completion_rate']}%

# # 里程碑统计
- **总里程碑数**: {summary['milestone_summary']['total']}
- **已达成**: {summary['milestone_summary']['achieved']}
- **延期**: {summary['milestone_summary']['delayed']}
- **达成率**: {summary['milestone_summary']['achievement_rate']}%

"""

        if overdue_tasks:
            report += "## 逾期任务\n"
            for task in overdue_tasks:
                report += f"- **{task['task_name']}** (ID: {task['task_id']})\n"
                report += f"  - 逾期天数: {task['days_overdue']}天\n"
                report += f"  - 当前状态: {task['status']}\n"
                report += f"  - 负责人: {task['assignee'] or '未分配'}\n\n"

        if upcoming_milestones:
            report += "## 即将到来的里程碑\n"
            for milestone in upcoming_milestones:
                report += f"- **{milestone['name']}** (ID: {milestone['milestone_id']})\n"
                report += f"  - 目标日期: {milestone['target_date']}\n"
                report += f"  - 剩余天数: {milestone['days_until']}天\n"
                report += f"  - 关联任务数: {milestone['task_count']}\n\n"

        if summary['quality_metrics']:
            report += "## 质量指标\n"
            for metric in summary['quality_metrics']:
                report += f"- **{metric['name']}**\n"
                report += f"  - 当前值: {metric['current_value']}\n"
                report += f"  - 目标值: {metric['target_value']}\n"
                report += f"  - 趋势: {metric['trend']}\n\n"

        return report

    def export_dashboard_data(self, dashboard_id: str, export_path: str) -> bool:
        """导出仪表板数据"""
        if dashboard_id not in self.dashboards:
            return False

        dashboard = self.dashboards[dashboard_id]

        # 准备导出数据
        export_data = {
            'dashboard_id': dashboard.dashboard_id,
            'name': dashboard.name,
            'description': dashboard.description,
            'created_at': dashboard.created_at.isoformat(),
            'last_updated': dashboard.last_updated.isoformat(),
            'tasks': [
                {
                    'task_id': t.task_id,
                    'task_name': t.task_name,
                    'description': t.description,
                    'status': t.status,
                    'priority': t.priority,
                    'assignee': t.assignee,
                    'start_date': t.start_date.isoformat() if t.start_date else None,
                    'due_date': t.due_date.isoformat() if t.due_date else None,
                    'completed_date': t.completed_date.isoformat() if t.completed_date else None,
                    'progress_percent': t.progress_percent,
                    'dependencies': t.dependencies,
                    'subtasks': t.subtasks,
                    'tags': t.tags,
                    'created_at': t.created_at.isoformat(),
                    'updated_at': t.updated_at.isoformat()
                }
                for t in dashboard.tasks
            ],
            'milestones': [
                {
                    'milestone_id': m.milestone_id,
                    'name': m.name,
                    'description': m.description,
                    'target_date': m.target_date.isoformat(),
                    'status': m.status,
                    'tasks': m.tasks,
                    'achieved_date': m.achieved_date.isoformat() if m.achieved_date else None,
                    'created_at': m.created_at.isoformat()
                }
                for m in dashboard.milestones
            ],
            'metrics': [
                {
                    'metric_id': m.metric_id,
                    'name': m.name,
                    'description': m.description,
                    'category': m.category,
                    'target_value': m.target_value,
                    'current_value': m.current_value,
                    'unit': m.unit,
                    'trend': m.trend,
                    'last_updated': m.last_updated.isoformat()
                }
                for m in dashboard.metrics
            ]
        }

        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"导出仪表板数据到: {export_path}")
            return True

        except Exception as e:
            logger.error(f"导出仪表板数据失败: {str(e)}")
            return False

    def _load_dashboard_data(self):
        """加载仪表板数据"""
        if self.dashboard_file.exists():
            try:
                with open(self.dashboard_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for dashboard_data in data.get('dashboards', []):
                    dashboard = ImplementationDashboard(
                        dashboard_id=dashboard_data['dashboard_id'],
                        name=dashboard_data['name'],
                        description=dashboard_data['description'],
                        created_at=datetime.fromisoformat(dashboard_data['created_at']),
                        last_updated=datetime.fromisoformat(dashboard_data['last_updated'])
                    )

                    # 恢复任务
                    for task_data in dashboard_data.get('tasks', []):
                        task = TaskProgress(
                            task_id=task_data['task_id'],
                            task_name=task_data['task_name'],
                            description=task_data['description'],
                            status=task_data['status'],
                            priority=task_data['priority'],
                            assignee=task_data.get('assignee'),
                            start_date=datetime.fromisoformat(
                                task_data['start_date']) if task_data.get('start_date') else None,
                            due_date=datetime.fromisoformat(
                                task_data['due_date']) if task_data.get('due_date') else None,
                            completed_date=datetime.fromisoformat(
                                task_data['completed_date']) if task_data.get('completed_date') else None,
                            progress_percent=task_data['progress_percent'],
                            dependencies=task_data['dependencies'],
                            subtasks=task_data['subtasks'],
                            tags=task_data['tags'],
                            created_at=datetime.fromisoformat(task_data['created_at']),
                            updated_at=datetime.fromisoformat(task_data['updated_at'])
                        )
                        dashboard.tasks.append(task)

                    # 恢复里程碑
                    for milestone_data in dashboard_data.get('milestones', []):
                        milestone = Milestone(
                            milestone_id=milestone_data['milestone_id'],
                            name=milestone_data['name'],
                            description=milestone_data['description'],
                            target_date=datetime.fromisoformat(milestone_data['target_date']),
                            status=milestone_data['status'],
                            tasks=milestone_data['tasks'],
                            achieved_date=datetime.fromisoformat(
                                milestone_data['achieved_date']) if milestone_data.get('achieved_date') else None,
                            created_at=datetime.fromisoformat(milestone_data['created_at'])
                        )
                        dashboard.milestones.append(milestone)

                    # 恢复质量指标
                    for metric_data in dashboard_data.get('metrics', []):
                        metric = QualityMetric(
                            metric_id=metric_data['metric_id'],
                            name=metric_data['name'],
                            description=metric_data['description'],
                            category=metric_data['category'],
                            target_value=metric_data['target_value'],
                            current_value=metric_data['current_value'],
                            unit=metric_data['unit'],
                            trend=metric_data['trend'],
                            last_updated=datetime.fromisoformat(metric_data['last_updated'])
                        )
                        dashboard.metrics.append(metric)

                    self.dashboards[dashboard.dashboard_id] = dashboard

                logger.info(f"加载了 {len(self.dashboards)} 个仪表板")

            except Exception as e:
                logger.error(f"加载仪表板数据失败: {str(e)}")

    def _save_dashboard_data(self):
        """保存仪表板数据"""
        data = {
            'dashboards': [
                {
                    'dashboard_id': dashboard.dashboard_id,
                    'name': dashboard.name,
                    'description': dashboard.description,
                    'created_at': dashboard.created_at.isoformat(),
                    'last_updated': dashboard.last_updated.isoformat(),
                    'tasks': [
                        {
                            'task_id': t.task_id,
                            'task_name': t.task_name,
                            'description': t.description,
                            'status': t.status,
                            'priority': t.priority,
                            'assignee': t.assignee,
                            'start_date': t.start_date.isoformat() if t.start_date else None,
                            'due_date': t.due_date.isoformat() if t.due_date else None,
                            'completed_date': t.completed_date.isoformat() if t.completed_date else None,
                            'progress_percent': t.progress_percent,
                            'dependencies': t.dependencies,
                            'subtasks': t.subtasks,
                            'tags': t.tags,
                            'created_at': t.created_at.isoformat(),
                            'updated_at': t.updated_at.isoformat()
                        }
                        for t in dashboard.tasks
                    ],
                    'milestones': [
                        {
                            'milestone_id': m.milestone_id,
                            'name': m.name,
                            'description': m.description,
                            'target_date': m.target_date.isoformat(),
                            'status': m.status,
                            'tasks': m.tasks,
                            'achieved_date': m.achieved_date.isoformat() if m.achieved_date else None,
                            'created_at': m.created_at.isoformat()
                        }
                        for m in dashboard.milestones
                    ],
                    'metrics': [
                        {
                            'metric_id': m.metric_id,
                            'name': m.name,
                            'description': m.description,
                            'category': m.category,
                            'target_value': m.target_value,
                            'current_value': m.current_value,
                            'unit': m.unit,
                            'trend': m.trend,
                            'last_updated': m.last_updated.isoformat()
                        }
                        for m in dashboard.metrics
                    ]
                }
                for dashboard in self.dashboards.values()
            ],
            'last_saved': datetime.now().isoformat()
        }

        with open(self.dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 创建全局实施监控器实例
_implementation_monitor = None


def get_implementation_monitor() -> ImplementationMonitor:
    """获取全局实施监控器实例"""
    global _implementation_monitor
    if _implementation_monitor is None:
        _implementation_monitor = ImplementationMonitor()
    return _implementation_monitor


__all__ = [
    'ImplementationMonitor', 'TaskProgress', 'Milestone', 'QualityMetric',
    'ImplementationDashboard', 'get_implementation_monitor'
]

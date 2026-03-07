#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化任务跟踪脚本

用于管理和跟踪各层优化任务的执行情况，包括任务状态、进度、负责人等信息。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SubTask:
    """子任务数据结构"""
    id: str
    name: str
    description: str
    status: TaskStatus
    assignee: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    progress: int = 0  # 0-100
    notes: str = ""


@dataclass
class Task:
    """任务数据结构"""
    id: str
    name: str
    description: str
    layer: str  # features, infrastructure, integration
    category: str  # short_term, mid_term
    priority: TaskPriority
    status: TaskStatus
    assignee: str
    start_date: str
    end_date: str
    progress: int = 0  # 0-100
    acceptance_criteria: str = ""
    subtasks: List[SubTask] = None
    notes: str = ""

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []


class OptimizationTaskTracker:
    """优化任务跟踪器"""

    def __init__(self, data_file: str = "optimization_tasks.json"):
        self.data_file = data_file
        self.tasks: Dict[str, Task] = {}
        self.load_tasks()

    def load_tasks(self):
        """加载任务数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for task_data in data.get('tasks', []):
                        # 转换字符串为枚举
                        task_data['priority'] = TaskPriority(task_data['priority'])
                        task_data['status'] = TaskStatus(task_data['status'])

                        # 转换子任务的字符串为枚举
                        for subtask_data in task_data.get('subtasks', []):
                            subtask_data['status'] = TaskStatus(subtask_data['status'])

                        task = Task(**task_data)
                        task.subtasks = [SubTask(**st) for st in task_data.get('subtasks', [])]
                        self.tasks[task.id] = task
            except Exception as e:
                print(f"加载任务数据失败: {e}")

    def save_tasks(self):
        """保存任务数据"""
        try:
            # 转换枚举为字符串
            tasks_data = []
            for task in self.tasks.values():
                task_dict = asdict(task)
                task_dict['priority'] = task_dict['priority'].value
                task_dict['status'] = task_dict['status'].value

                # 转换子任务的枚举
                subtasks_data = []
                for subtask in task.subtasks:
                    subtask_dict = asdict(subtask)
                    subtask_dict['status'] = subtask_dict['status'].value
                    subtasks_data.append(subtask_dict)
                task_dict['subtasks'] = subtasks_data

                tasks_data.append(task_dict)

            data = {
                'tasks': tasks_data,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存任务数据失败: {e}")

    def add_task(self, task: Task):
        """添加任务"""
        self.tasks[task.id] = task
        self.save_tasks()

    def update_task(self, task_id: str, **kwargs):
        """更新任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            self.save_tasks()

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)

    def get_tasks_by_layer(self, layer: str) -> List[Task]:
        """按层获取任务"""
        return [task for task in self.tasks.values() if task.layer == layer]

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """按状态获取任务"""
        return [task for task in self.tasks.values() if task.status == status]

    def get_tasks_by_priority(self, priority: TaskPriority) -> List[Task]:
        """按优先级获取任务"""
        return [task for task in self.tasks.values() if task.priority == priority]

    def calculate_overall_progress(self) -> Dict[str, float]:
        """计算整体进度"""
        progress = {}
        for layer in ['features', 'infrastructure', 'integration']:
            layer_tasks = self.get_tasks_by_layer(layer)
            if layer_tasks:
                total_progress = sum(task.progress for task in layer_tasks)
                avg_progress = total_progress / len(layer_tasks)
                progress[layer] = avg_progress
            else:
                progress[layer] = 0.0
        return progress

    def generate_report(self) -> str:
        """生成报告"""
        report = []
        report.append("# 优化任务执行报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 整体进度
        overall_progress = self.calculate_overall_progress()
        report.append("## 整体进度")
        for layer, progress in overall_progress.items():
            report.append(f"- {layer}层: {progress:.1f}%")
        report.append("")

        # 按层统计
        for layer in ['features', 'infrastructure', 'integration']:
            layer_tasks = self.get_tasks_by_layer(layer)
            if layer_tasks:
                report.append(f"## {layer}层任务")

                # 短期目标
                short_term_tasks = [t for t in layer_tasks if t.category == 'short_term']
                if short_term_tasks:
                    report.append("### 短期目标")
                    for task in short_term_tasks:
                        status_emoji = {
                            TaskStatus.NOT_STARTED: "⏳",
                            TaskStatus.IN_PROGRESS: "🔄",
                            TaskStatus.COMPLETED: "✅",
                            TaskStatus.BLOCKED: "🚫",
                            TaskStatus.CANCELLED: "❌"
                        }
                        report.append(
                            f"- {status_emoji[task.status]} {task.name} ({task.progress}%)")
                        if task.assignee != "待定":
                            report.append(f"  负责人: {task.assignee}")
                        if task.notes:
                            report.append(f"  备注: {task.notes}")
                    report.append("")

                # 中期目标
                mid_term_tasks = [t for t in layer_tasks if t.category == 'mid_term']
                if mid_term_tasks:
                    report.append("### 中期目标")
                    for task in mid_term_tasks:
                        status_emoji = {
                            TaskStatus.NOT_STARTED: "⏳",
                            TaskStatus.IN_PROGRESS: "🔄",
                            TaskStatus.COMPLETED: "✅",
                            TaskStatus.BLOCKED: "🚫",
                            TaskStatus.CANCELLED: "❌"
                        }
                        report.append(
                            f"- {status_emoji[task.status]} {task.name} ({task.progress}%)")
                        if task.assignee != "待定":
                            report.append(f"  负责人: {task.assignee}")
                        if task.notes:
                            report.append(f"  备注: {task.notes}")
                    report.append("")

        return "\n".join(report)

    def initialize_default_tasks(self):
        """初始化默认任务"""
        default_tasks = [
            # 特征层短期目标
            Task(
                id="features_short_1",
                name="完善特征层单元测试",
                description="补充所有核心组件的单元测试，确保测试覆盖率>90%",
                layer="features",
                category="short_term",
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                assignee="待定",
                start_date="2025-08-01",
                end_date="2025-08-15",
                acceptance_criteria="测试覆盖率 > 90%，所有核心功能都有测试覆盖",
                subtasks=[
                    SubTask("st_1", "补充 FeatureEngineer 单元测试",
                            "实现 FeatureEngineer 的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_2", "补充 FeatureProcessor 单元测试",
                            "实现 FeatureProcessor 的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_3", "补充 FeatureSelector 单元测试",
                            "实现 FeatureSelector 的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_4", "补充 FeatureStandardizer 单元测试",
                            "实现 FeatureStandardizer 的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_5", "补充 FeatureSaver 单元测试",
                            "实现 FeatureSaver 的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                ]
            ),
            Task(
                id="features_short_2",
                name="补充集成测试",
                description="验证特征层与其他层的协作，确保接口兼容性",
                layer="features",
                category="short_term",
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                assignee="待定",
                start_date="2025-08-01",
                end_date="2025-08-20",
                acceptance_criteria="所有集成测试通过，无接口兼容性问题",
                subtasks=[
                    SubTask("st_6", "验证与数据层的协作", "测试特征层与数据层的接口兼容性", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_7", "验证与模型层的协作", "测试特征层与模型层的接口兼容性", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_8", "验证与基础设施层的协作", "测试特征层与基础设施层的接口兼容性",
                            TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_9", "创建端到端特征工程流程测试", "实现完整的特征工程流程测试", TaskStatus.NOT_STARTED, "待定"),
                ]
            ),
            # 基础设施层短期目标
            Task(
                id="infrastructure_short_1",
                name="完善监控指标",
                description="补充更多业务指标收集点，优化指标采集性能",
                layer="infrastructure",
                category="short_term",
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                assignee="待定",
                start_date="2025-08-01",
                end_date="2025-08-15",
                acceptance_criteria="监控指标完整，告警规则有效",
                subtasks=[
                    SubTask("st_10", "补充更多业务指标收集点", "添加交易、风控、数据、模型相关指标",
                            TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_11", "优化指标采集性能", "提升指标采集效率，减少性能影响", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_12", "完善告警规则配置", "配置合理的告警规则和阈值", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_13", "添加自定义指标支持", "支持用户自定义监控指标", TaskStatus.NOT_STARTED, "待定"),
                ]
            ),
            # 系统集成层短期目标
            Task(
                id="integration_short_1",
                name="完善系统集成单元测试",
                description="补充所有核心集成组件的单元测试",
                layer="integration",
                category="short_term",
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                assignee="待定",
                start_date="2025-08-01",
                end_date="2025-08-15",
                acceptance_criteria="测试覆盖率 > 90%，所有核心功能都有测试覆盖",
                subtasks=[
                    SubTask("st_14", "补充 SystemIntegrationManager 单元测试",
                            "实现系统集成管理器的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_15", "补充 LayerInterface 单元测试",
                            "实现层接口的完整单元测试", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_16", "补充接口兼容性测试", "测试各层接口的兼容性", TaskStatus.NOT_STARTED, "待定"),
                    SubTask("st_17", "补充配置一致性测试", "测试配置管理的一致性", TaskStatus.NOT_STARTED, "待定"),
                ]
            ),
        ]

        for task in default_tasks:
            self.add_task(task)

    def print_status(self):
        """打印当前状态"""
        print("=" * 60)
        print("优化任务跟踪器状态")
        print("=" * 60)

        overall_progress = self.calculate_overall_progress()
        print(f"整体进度:")
        for layer, progress in overall_progress.items():
            print(f"  {layer}层: {progress:.1f}%")

        print(f"\n任务统计:")
        total_tasks = len(self.tasks)
        completed_tasks = len(self.get_tasks_by_status(TaskStatus.COMPLETED))
        in_progress_tasks = len(self.get_tasks_by_status(TaskStatus.IN_PROGRESS))
        not_started_tasks = len(self.get_tasks_by_status(TaskStatus.NOT_STARTED))

        print(f"  总任务数: {total_tasks}")
        print(f"  已完成: {completed_tasks}")
        print(f"  进行中: {in_progress_tasks}")
        print(f"  未开始: {not_started_tasks}")

        print(f"\n高优先级任务:")
        high_priority_tasks = self.get_tasks_by_priority(TaskPriority.HIGH)
        for task in high_priority_tasks:
            status_emoji = {
                TaskStatus.NOT_STARTED: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.CANCELLED: "❌"
            }
            print(f"  {status_emoji[task.status]} {task.name} ({task.progress}%)")

    def update_task_progress(self, task_id: str, progress: float, status: TaskStatus = None):
        """更新任务进度"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.progress = progress
            if status:
                task.status = status
            self.save_tasks()
            # self.logger.info(f"任务 {task_id} 进度更新: {progress}%, 状态: {task.status.value}") # Original code had this line commented out
        else:
            # self.logger.warning(f"任务 {task_id} 不存在") # Original code had this line commented out
            print(f"任务 {task_id} 不存在，无法更新进度。")

    def mark_task_completed(self, task_id: str):
        """标记任务为已完成"""
        self.update_task_progress(task_id, 100.0, TaskStatus.COMPLETED)

    def mark_task_in_progress(self, task_id: str):
        """标记任务为进行中"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.IN_PROGRESS
            self.save_tasks()


def main():
    """主函数"""
    tracker = OptimizationTaskTracker()

    # 如果任务文件不存在，初始化默认任务
    if not os.path.exists(tracker.data_file):
        print("初始化默认任务...")
        tracker.initialize_default_tasks()
        print("默认任务已初始化完成！")

    # 打印当前状态
    tracker.print_status()

    # 生成报告
    report = tracker.generate_report()
    report_file = "optimization_task_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n详细报告已生成: {report_file}")


if __name__ == "__main__":
    main()

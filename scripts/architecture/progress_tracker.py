#!/usr/bin/env python3
"""
RQA2025 架构实施进度跟踪工具

该工具用于跟踪业务流程驱动架构实施计划的执行进度，
包括任务完成情况、里程碑检查、风险评估等功能。

作者: 架构组
版本: 1.0.0
日期: 2025-01-27
"""

from src.infrastructure.core.monitoring.monitor_factory import create_monitor
from src.infrastructure.core.config.config_factory import create_config_manager
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class Task:
    """任务数据结构"""
    id: str
    name: str
    description: str
    phase: str
    estimated_days: int
    actual_days: Optional[int] = None
    status: str = "pending"  # pending, in_progress, completed, blocked
    assignee: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dependencies: List[str] = None
    progress: float = 0.0
    notes: str = ""


@dataclass
class Phase:
    """阶段数据结构"""
    name: str
    description: str
    start_date: str
    end_date: str
    status: str = "pending"  # pending, in_progress, completed
    tasks: List[Task] = None
    progress: float = 0.0
    risks: List[str] = None


@dataclass
class ProgressReport:
    """进度报告数据结构"""
    phase: str
    time_period: str
    planned_tasks: int
    completed_tasks: int
    completion_rate: float
    key_achievements: List[str]
    issues_encountered: List[str]
    next_phase_plan: List[str]
    risk_assessment: str
    generated_date: str


class ArchitectureProgressTracker:
    """架构实施进度跟踪器"""

    def __init__(self, config_path: str = "config/architecture/progress_tracker.json"):
        """初始化进度跟踪器"""
        self.config_path = config_path
        self.phases = self._load_phases()
        self.tasks = self._load_tasks()
        self.reports = []

        # 初始化配置和监控
        try:
            self.config_manager = create_config_manager("unified")
            self.monitor = create_monitor("unified")
        except Exception as e:
            print(f"警告: 无法初始化配置管理器和监控器: {e}")
            self.config_manager = None
            self.monitor = None

    def _load_phases(self) -> List[Phase]:
        """加载阶段配置"""
        phases_data = [
            {
                "name": "第一阶段: 核心服务完善",
                "description": "完善事件总线、依赖注入容器、服务容器",
                "start_date": "2025-01-27",
                "end_date": "2025-02-10",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            },
            {
                "name": "第二阶段: 业务流程编排",
                "description": "完善业务流程编排器、状态机逻辑、事件处理器",
                "start_date": "2025-02-10",
                "end_date": "2025-03-03",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            },
            {
                "name": "第三阶段: 风控合规层完善",
                "description": "完善风险检查机制、合规验证体系、实时监控系统",
                "start_date": "2025-03-03",
                "end_date": "2025-03-24",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            },
            {
                "name": "第四阶段: 交易执行层完善",
                "description": "完善订单管理系统、执行引擎、成交回报处理",
                "start_date": "2025-03-24",
                "end_date": "2025-04-14",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            },
            {
                "name": "第五阶段: 监控反馈层完善",
                "description": "完善性能监控系统、业务监控系统、告警反馈机制",
                "start_date": "2025-04-14",
                "end_date": "2025-04-28",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            },
            {
                "name": "第六阶段: 集成测试和优化",
                "description": "端到端测试、性能优化、文档完善",
                "start_date": "2025-04-28",
                "end_date": "2025-05-12",
                "status": "pending",
                "progress": 0.0,
                "risks": []
            }
        ]

        return [Phase(**phase_data) for phase_data in phases_data]

    def _load_tasks(self) -> List[Task]:
        """加载任务配置"""
        tasks_data = [
            # 第一阶段任务
            {
                "id": "1.1.1",
                "name": "事件类型扩展",
                "description": "完善EventType枚举，添加缺失的事件类型",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 2,
                "dependencies": []
            },
            {
                "id": "1.1.2",
                "name": "事件处理器优化",
                "description": "实现异步事件处理、事件重试机制、事件优先级管理",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 2,
                "dependencies": ["1.1.1"]
            },
            {
                "id": "1.1.3",
                "name": "事件历史管理",
                "description": "实现事件持久化存储、事件查询和过滤功能",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 1,
                "dependencies": ["1.1.2"]
            },
            {
                "id": "1.2.1",
                "name": "容器功能扩展",
                "description": "实现生命周期管理、作用域管理、循环依赖检测",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 2,
                "dependencies": []
            },
            {
                "id": "1.2.2",
                "name": "服务注册优化",
                "description": "实现自动服务发现、服务健康检查、服务版本管理",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 1,
                "dependencies": ["1.2.1"]
            },
            {
                "id": "1.3.1",
                "name": "服务管理接口",
                "description": "实现服务注册和注销、服务状态监控、服务配置管理",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 2,
                "dependencies": ["1.2.2"]
            },
            {
                "id": "1.3.2",
                "name": "服务发现机制",
                "description": "实现服务发现功能、负载均衡支持、故障转移机制",
                "phase": "第一阶段: 核心服务完善",
                "estimated_days": 1,
                "dependencies": ["1.3.1"]
            }
        ]

        return [Task(**task_data) for task_data in tasks_data]

    def get_overall_progress(self) -> Dict[str, Any]:
        """获取整体进度"""
        total_phases = len(self.phases)
        completed_phases = sum(1 for phase in self.phases if phase.status == "completed")
        in_progress_phases = sum(1 for phase in self.phases if phase.status == "in_progress")

        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks if task.status == "completed")
        in_progress_tasks = sum(1 for task in self.tasks if task.status == "in_progress")

        overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        return {
            "total_phases": total_phases,
            "completed_phases": completed_phases,
            "in_progress_phases": in_progress_phases,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "overall_progress": overall_progress,
            "current_phase": self._get_current_phase(),
            "next_milestone": self._get_next_milestone()
        }

    def _get_current_phase(self) -> Optional[str]:
        """获取当前阶段"""
        today = datetime.now().date()
        for phase in self.phases:
            start_date = datetime.strptime(phase.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(phase.end_date, "%Y-%m-%d").date()
            if start_date <= today <= end_date:
                return phase.name
        return None

    def _get_next_milestone(self) -> Optional[str]:
        """获取下一个里程碑"""
        today = datetime.now().date()
        for phase in self.phases:
            start_date = datetime.strptime(phase.start_date, "%Y-%m-%d").date()
            if start_date > today:
                return f"{phase.name} - {phase.start_date}"
        return None

    def update_task_status(self, task_id: str, status: str, progress: float = None,
                           assignee: str = None, notes: str = None) -> bool:
        """更新任务状态"""
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            return False

        task.status = status
        if progress is not None:
            task.progress = progress
        if assignee:
            task.assignee = assignee
        if notes:
            task.notes = notes

        # 更新开始和结束日期
        if status == "in_progress" and not task.start_date:
            task.start_date = datetime.now().strftime("%Y-%m-%d")
        elif status == "completed" and not task.end_date:
            task.end_date = datetime.now().strftime("%Y-%m-%d")
            if task.start_date:
                start_date = datetime.strptime(task.start_date, "%Y-%m-%d")
                end_date = datetime.strptime(task.end_date, "%Y-%m-%d")
                task.actual_days = (end_date - start_date).days

        # 更新阶段进度
        self._update_phase_progress()

        return True

    def _update_phase_progress(self):
        """更新阶段进度"""
        for phase in self.phases:
            phase_tasks = [t for t in self.tasks if t.phase == phase.name]
            if phase_tasks:
                completed_tasks = sum(1 for t in phase_tasks if t.status == "completed")
                phase.progress = (completed_tasks / len(phase_tasks)) * 100

                # 更新阶段状态
                if phase.progress == 100:
                    phase.status = "completed"
                elif phase.progress > 0:
                    phase.status = "in_progress"

    def generate_progress_report(self, phase_name: str = None) -> ProgressReport:
        """生成进度报告"""
        if phase_name:
            phase = next((p for p in self.phases if p.name == phase_name), None)
            if not phase:
                raise ValueError(f"阶段 '{phase_name}' 不存在")

            phase_tasks = [t for t in self.tasks if t.phase == phase_name]
        else:
            phase = self._get_current_phase()
            if not phase:
                phase = self.phases[0]
            phase_tasks = [t for t in self.tasks if t.phase == phase]

        planned_tasks = len(phase_tasks)
        completed_tasks = sum(1 for t in phase_tasks if t.status == "completed")
        completion_rate = (completed_tasks / planned_tasks * 100) if planned_tasks > 0 else 0

        # 获取关键成果
        key_achievements = [
            f"完成任务: {t.name}" for t in phase_tasks
            if t.status == "completed"
        ]

        # 获取遇到的问题
        issues_encountered = [
            f"任务 {t.name}: {t.notes}" for t in phase_tasks
            if t.notes and t.status in ["blocked", "in_progress"]
        ]

        # 获取下阶段计划
        next_phase_plan = [
            f"准备开始: {t.name}" for t in phase_tasks
            if t.status == "pending" and not t.dependencies
        ]

        # 风险评估
        blocked_tasks = sum(1 for t in phase_tasks if t.status == "blocked")
        if blocked_tasks > 0:
            risk_assessment = f"高风险: {blocked_tasks} 个任务被阻塞"
        elif completion_rate < 50:
            risk_assessment = "中风险: 进度落后于计划"
        else:
            risk_assessment = "低风险: 进度正常"

        report = ProgressReport(
            phase=phase,
            time_period=f"{phase.start_date} - {phase.end_date}",
            planned_tasks=planned_tasks,
            completed_tasks=completed_tasks,
            completion_rate=completion_rate,
            key_achievements=key_achievements,
            issues_encountered=issues_encountered,
            next_phase_plan=next_phase_plan,
            risk_assessment=risk_assessment,
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self.reports.append(report)
        return report

    def print_progress_report(self, report: ProgressReport):
        """打印进度报告"""
        print("=" * 80)
        print("📊 架构实施进度报告")
        print("=" * 80)
        print(f"阶段: {report.phase}")
        print(f"时间: {report.time_period}")
        print(f"计划任务: {report.planned_tasks}")
        print(f"完成任务: {report.completed_tasks}")
        print(f"完成率: {report.completion_rate:.1f}%")
        print()

        if report.key_achievements:
            print("主要成果:")
            for achievement in report.key_achievements:
                print(f"  ✅ {achievement}")
            print()

        if report.issues_encountered:
            print("遇到的问题:")
            for issue in report.issues_encountered:
                print(f"  ⚠️  {issue}")
            print()

        if report.next_phase_plan:
            print("下阶段计划:")
            for plan in report.next_phase_plan:
                print(f"  📋 {plan}")
            print()

        print(f"风险评估: {report.risk_assessment}")
        print(f"生成时间: {report.generated_date}")
        print("=" * 80)

    def print_overall_progress(self):
        """打印整体进度"""
        progress = self.get_overall_progress()

        print("=" * 80)
        print("🏗️  架构实施整体进度")
        print("=" * 80)
        print(f"总体进度: {progress['overall_progress']:.1f}%")
        print(f"阶段进度: {progress['completed_phases']}/{progress['total_phases']}")
        print(f"任务进度: {progress['completed_tasks']}/{progress['total_tasks']}")
        print()

        if progress['current_phase']:
            print(f"当前阶段: {progress['current_phase']}")
        if progress['next_milestone']:
            print(f"下一个里程碑: {progress['next_milestone']}")
        print()

        print("各阶段进度:")
        for phase in self.phases:
            status_icon = "✅" if phase.status == "completed" else "🔄" if phase.status == "in_progress" else "⏳"
            print(f"  {status_icon} {phase.name}: {phase.progress:.1f}%")

        print("=" * 80)

    def save_progress(self, file_path: str = None):
        """保存进度数据"""
        if not file_path:
            file_path = "data/progress/architecture_progress.json"

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data = {
            "phases": [asdict(phase) for phase in self.phases],
            "tasks": [asdict(task) for task in self.tasks],
            "reports": [asdict(report) for report in self.reports],
            "last_updated": datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"进度数据已保存到: {file_path}")

    def load_progress(self, file_path: str = None):
        """加载进度数据"""
        if not file_path:
            file_path = "data/progress/architecture_progress.json"

        if not os.path.exists(file_path):
            print(f"进度文件不存在: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复阶段数据
            for phase_data in data.get("phases", []):
                phase = next((p for p in self.phases if p.name == phase_data["name"]), None)
                if phase:
                    phase.status = phase_data.get("status", "pending")
                    phase.progress = phase_data.get("progress", 0.0)
                    phase.risks = phase_data.get("risks", [])

            # 恢复任务数据
            for task_data in data.get("tasks", []):
                task = next((t for t in self.tasks if t.id == task_data["id"]), None)
                if task:
                    task.status = task_data.get("status", "pending")
                    task.progress = task_data.get("progress", 0.0)
                    task.assignee = task_data.get("assignee")
                    task.start_date = task_data.get("start_date")
                    task.end_date = task_data.get("end_date")
                    task.actual_days = task_data.get("actual_days")
                    task.notes = task_data.get("notes", "")

            print(f"进度数据已从 {file_path} 加载")

        except Exception as e:
            print(f"加载进度数据失败: {e}")


def main():
    """主函数"""
    print("🏗️  RQA2025 架构实施进度跟踪工具")
    print("=" * 50)

    tracker = ArchitectureProgressTracker()

    while True:
        print("\n请选择操作:")
        print("1. 查看整体进度")
        print("2. 生成阶段进度报告")
        print("3. 更新任务状态")
        print("4. 保存进度")
        print("5. 加载进度")
        print("6. 退出")

        choice = input("\n请输入选择 (1-6): ").strip()

        if choice == "1":
            tracker.print_overall_progress()

        elif choice == "2":
            print("\n可用阶段:")
            for i, phase in enumerate(tracker.phases, 1):
                print(f"{i}. {phase.name}")

            try:
                phase_choice = int(input("\n请选择阶段编号: ")) - 1
                if 0 <= phase_choice < len(tracker.phases):
                    phase_name = tracker.phases[phase_choice].name
                    report = tracker.generate_progress_report(phase_name)
                    tracker.print_progress_report(report)
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入有效的数字")

        elif choice == "3":
            print("\n可用任务:")
            for task in tracker.tasks:
                print(f"{task.id}: {task.name} ({task.status})")

            task_id = input("\n请输入任务ID: ").strip()
            if any(t.id == task_id for t in tracker.tasks):
                print("\n可用状态: pending, in_progress, completed, blocked")
                status = input("请输入新状态: ").strip()

                progress = input("请输入进度 (0-100, 回车跳过): ").strip()
                progress = float(progress) if progress else None

                assignee = input("请输入负责人 (回车跳过): ").strip()
                assignee = assignee if assignee else None

                notes = input("请输入备注 (回车跳过): ").strip()
                notes = notes if notes else None

                if tracker.update_task_status(task_id, status, progress, assignee, notes):
                    print("任务状态更新成功")
                else:
                    print("任务状态更新失败")
            else:
                print("无效的任务ID")

        elif choice == "4":
            tracker.save_progress()

        elif choice == "5":
            tracker.load_progress()

        elif choice == "6":
            print("感谢使用架构实施进度跟踪工具！")
            break

        else:
            print("无效的选择，请重新输入")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实施监控器扩展测试
补充ImplementationMonitor的更多方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import patch, Mock
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    core_implementation_monitor_module = importlib.import_module('src.monitoring.core.implementation_monitor')
    ImplementationMonitor = getattr(core_implementation_monitor_module, 'ImplementationMonitor', None)
    TaskProgress = getattr(core_implementation_monitor_module, 'TaskProgress', None)
    Milestone = getattr(core_implementation_monitor_module, 'Milestone', None)
    QualityMetric = getattr(core_implementation_monitor_module, 'QualityMetric', None)
    ImplementationDashboard = getattr(core_implementation_monitor_module, 'ImplementationDashboard', None)
    
    if ImplementationMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestImplementationMonitorExtended:
    """测试ImplementationMonitor扩展功能"""

    @pytest.fixture
    def monitor(self, tmp_path):
        """创建实施监控器实例"""
        return ImplementationMonitor(data_dir=str(tmp_path / "monitoring"))

    @pytest.fixture
    def dashboard(self, monitor):
        """创建测试仪表板"""
        return monitor.create_dashboard('test_dashboard', 'Test Dashboard', 'Test Description')

    @pytest.fixture
    def sample_task(self):
        """创建示例任务"""
        return TaskProgress(
            task_id='task_1',
            task_name='Test Task',
            description='Test task description',
            status='in_progress',
            priority='high'
        )

    @pytest.fixture
    def sample_milestone(self):
        """创建示例里程碑"""
        return Milestone(
            milestone_id='milestone_1',
            name='Test Milestone',
            description='Test milestone description',
            target_date=datetime.now() + timedelta(days=30),
            status='pending'
        )

    @pytest.fixture
    def sample_metric(self):
        """创建示例质量指标"""
        return QualityMetric(
            metric_id='metric_1',
            name='Test Metric',
            description='Test metric description',
            category='code_quality',
            target_value=80.0,
            current_value=75.0,
            unit='percent',
            trend='improving'
        )

    def test_add_task_success(self, monitor, dashboard, sample_task):
        """测试添加任务成功"""
        result = monitor.add_task('test_dashboard', sample_task)
        
        assert result == True
        assert len(monitor.dashboards['test_dashboard'].tasks) == 1

    def test_add_task_dashboard_not_found(self, monitor, sample_task):
        """测试添加任务到不存在的仪表板"""
        result = monitor.add_task('nonexistent_dashboard', sample_task)
        
        assert result == False

    def test_update_task_progress_success(self, monitor, dashboard, sample_task):
        """测试更新任务进度成功"""
        monitor.add_task('test_dashboard', sample_task)
        
        result = monitor.update_task_progress('test_dashboard', 'task_1', 50, 'in_progress')
        
        assert result == True
        task = monitor.dashboards['test_dashboard'].tasks[0]
        assert task.progress_percent == 50

    def test_update_task_progress_completed(self, monitor, dashboard, sample_task):
        """测试更新任务进度为已完成"""
        monitor.add_task('test_dashboard', sample_task)
        
        result = monitor.update_task_progress('test_dashboard', 'task_1', 100, 'completed')
        
        assert result == True
        task = monitor.dashboards['test_dashboard'].tasks[0]
        assert task.status == 'completed'
        assert task.completed_date is not None

    def test_update_task_progress_task_not_found(self, monitor, dashboard):
        """测试更新不存在的任务进度"""
        result = monitor.update_task_progress('test_dashboard', 'nonexistent_task', 50)
        
        assert result == False

    def test_update_task_progress_dashboard_not_found(self, monitor):
        """测试更新任务进度仪表板不存在"""
        result = monitor.update_task_progress('nonexistent_dashboard', 'task_1', 50)
        
        assert result == False

    def test_add_milestone_success(self, monitor, dashboard, sample_milestone):
        """测试添加里程碑成功"""
        result = monitor.add_milestone('test_dashboard', sample_milestone)
        
        assert result == True
        assert len(monitor.dashboards['test_dashboard'].milestones) == 1

    def test_add_milestone_dashboard_not_found(self, monitor, sample_milestone):
        """测试添加里程碑到不存在的仪表板"""
        result = monitor.add_milestone('nonexistent_dashboard', sample_milestone)
        
        assert result == False

    def test_update_milestone_status_success(self, monitor, dashboard, sample_milestone):
        """测试更新里程碑状态成功"""
        monitor.add_milestone('test_dashboard', sample_milestone)
        
        result = monitor.update_milestone_status('test_dashboard', 'milestone_1', 'achieved')
        
        assert result == True
        milestone = monitor.dashboards['test_dashboard'].milestones[0]
        assert milestone.status == 'achieved'
        assert milestone.achieved_date is not None

    def test_update_milestone_status_achieved(self, monitor, dashboard, sample_milestone):
        """测试更新里程碑状态为已达成"""
        monitor.add_milestone('test_dashboard', sample_milestone)
        
        result = monitor.update_milestone_status('test_dashboard', 'milestone_1', 'achieved')
        
        assert result == True
        milestone = monitor.dashboards['test_dashboard'].milestones[0]
        assert milestone.achieved_date is not None

    def test_update_milestone_status_not_found(self, monitor, dashboard):
        """测试更新不存在的里程碑状态"""
        result = monitor.update_milestone_status('test_dashboard', 'nonexistent_milestone', 'achieved')
        
        assert result == False

    def test_add_quality_metric_success(self, monitor, dashboard, sample_metric):
        """测试添加质量指标成功"""
        result = monitor.add_quality_metric('test_dashboard', sample_metric)
        
        assert result == True
        assert len(monitor.dashboards['test_dashboard'].metrics) == 1

    def test_add_quality_metric_dashboard_not_found(self, monitor, sample_metric):
        """测试添加质量指标到不存在的仪表板"""
        result = monitor.add_quality_metric('nonexistent_dashboard', sample_metric)
        
        assert result == False

    def test_update_quality_metric_success(self, monitor, dashboard, sample_metric):
        """测试更新质量指标成功"""
        monitor.add_quality_metric('test_dashboard', sample_metric)
        
        result = monitor.update_quality_metric('test_dashboard', 'metric_1', 80.0, 'stable')
        
        assert result == True
        metric = monitor.dashboards['test_dashboard'].metrics[0]
        assert metric.current_value == 80.0
        assert metric.trend == 'stable'

    def test_update_quality_metric_without_trend(self, monitor, dashboard, sample_metric):
        """测试更新质量指标不指定趋势"""
        monitor.add_quality_metric('test_dashboard', sample_metric)
        
        result = monitor.update_quality_metric('test_dashboard', 'metric_1', 80.0)
        
        assert result == True
        metric = monitor.dashboards['test_dashboard'].metrics[0]
        assert metric.current_value == 80.0
        # 趋势应该保持不变
        assert metric.trend == 'improving'

    def test_update_quality_metric_not_found(self, monitor, dashboard):
        """测试更新不存在的质量指标"""
        result = monitor.update_quality_metric('test_dashboard', 'nonexistent_metric', 80.0)
        
        assert result == False

    def test_get_dashboard_summary_with_data(self, monitor, dashboard, sample_task, sample_milestone, sample_metric):
        """测试获取有数据的仪表板摘要"""
        monitor.add_task('test_dashboard', sample_task)
        monitor.add_milestone('test_dashboard', sample_milestone)
        monitor.add_quality_metric('test_dashboard', sample_metric)
        
        summary = monitor.get_dashboard_summary('test_dashboard')
        
        assert summary is not None
        assert summary['dashboard_id'] == 'test_dashboard'
        assert 'task_summary' in summary
        assert 'milestone_summary' in summary
        assert 'quality_metrics' in summary
        assert summary['task_summary']['total'] == 1

    def test_get_dashboard_summary_empty(self, monitor, dashboard):
        """测试获取空仪表板摘要"""
        summary = monitor.get_dashboard_summary('test_dashboard')
        
        assert summary is not None
        assert summary['task_summary']['total'] == 0
        assert summary['overall_progress_percent'] == 0

    def test_get_dashboard_summary_not_found(self, monitor):
        """测试获取不存在的仪表板摘要"""
        summary = monitor.get_dashboard_summary('nonexistent_dashboard')
        
        assert summary is None

    def test_get_dashboard_summary_with_completed_task(self, monitor, dashboard, sample_task):
        """测试获取有已完成任务的仪表板摘要"""
        monitor.add_task('test_dashboard', sample_task)
        monitor.update_task_progress('test_dashboard', 'task_1', 100, 'completed')
        
        summary = monitor.get_dashboard_summary('test_dashboard')
        
        assert summary['task_summary']['completed'] == 1
        assert summary['task_summary']['completion_rate'] == 100.0

    def test_get_dashboard_summary_with_multiple_tasks(self, monitor, dashboard):
        """测试获取多任务仪表板摘要"""
        task1 = TaskProgress(
            task_id='task_1', task_name='Task 1', description='', status='completed', priority='high'
        )
        task2 = TaskProgress(
            task_id='task_2', task_name='Task 2', description='', status='in_progress', priority='medium'
        )
        task3 = TaskProgress(
            task_id='task_3', task_name='Task 3', description='', status='blocked', priority='low'
        )
        
        monitor.add_task('test_dashboard', task1)
        monitor.add_task('test_dashboard', task2)
        monitor.add_task('test_dashboard', task3)
        
        summary = monitor.get_dashboard_summary('test_dashboard')
        
        assert summary['task_summary']['total'] == 3
        assert summary['task_summary']['completed'] == 1
        assert summary['task_summary']['in_progress'] == 1
        assert summary['task_summary']['blocked'] == 1

    def test_get_overdue_tasks_with_overdue(self, monitor, dashboard):
        """测试获取逾期任务"""
        task = TaskProgress(
            task_id='overdue_task',
            task_name='Overdue Task',
            description='',
            status='in_progress',
            priority='high',
            due_date=datetime.now() - timedelta(days=5)
        )
        monitor.add_task('test_dashboard', task)
        
        overdue_tasks = monitor.get_overdue_tasks('test_dashboard')
        
        assert len(overdue_tasks) == 1
        assert overdue_tasks[0]['task_id'] == 'overdue_task'
        assert overdue_tasks[0]['days_overdue'] >= 5

    def test_get_overdue_tasks_completed_not_included(self, monitor, dashboard):
        """测试已完成任务不包含在逾期任务中"""
        task = TaskProgress(
            task_id='completed_task',
            task_name='Completed Task',
            description='',
            status='completed',
            priority='high',
            due_date=datetime.now() - timedelta(days=5)
        )
        monitor.add_task('test_dashboard', task)
        
        overdue_tasks = monitor.get_overdue_tasks('test_dashboard')
        
        assert len(overdue_tasks) == 0

    def test_get_overdue_tasks_no_due_date(self, monitor, dashboard):
        """测试没有截止日期的任务不包含在逾期任务中"""
        task = TaskProgress(
            task_id='no_due_date_task',
            task_name='No Due Date Task',
            description='',
            status='in_progress',
            priority='high'
        )
        monitor.add_task('test_dashboard', task)
        
        overdue_tasks = monitor.get_overdue_tasks('test_dashboard')
        
        assert len(overdue_tasks) == 0

    def test_get_overdue_tasks_dashboard_not_found(self, monitor):
        """测试获取不存在的仪表板的逾期任务"""
        overdue_tasks = monitor.get_overdue_tasks('nonexistent_dashboard')
        
        assert overdue_tasks == []

    def test_get_upcoming_milestones(self, monitor, dashboard):
        """测试获取即将到来的里程碑"""
        milestone = Milestone(
            milestone_id='upcoming_milestone',
            name='Upcoming Milestone',
            description='',
            target_date=datetime.now() + timedelta(days=3),
            status='pending'
        )
        monitor.add_milestone('test_dashboard', milestone)
        
        upcoming = monitor.get_upcoming_milestones('test_dashboard', days_ahead=7)
        
        assert len(upcoming) == 1
        assert upcoming[0]['milestone_id'] == 'upcoming_milestone'

    def test_get_upcoming_milestones_custom_days(self, monitor, dashboard):
        """测试获取即将到来的里程碑自定义天数"""
        milestone = Milestone(
            milestone_id='future_milestone',
            name='Future Milestone',
            description='',
            target_date=datetime.now() + timedelta(days=10),
            status='pending'
        )
        monitor.add_milestone('test_dashboard', milestone)
        
        # 设置days_ahead为5，应该找不到
        upcoming = monitor.get_upcoming_milestones('test_dashboard', days_ahead=5)
        assert len(upcoming) == 0
        
        # 设置days_ahead为15，应该找到
        upcoming = monitor.get_upcoming_milestones('test_dashboard', days_ahead=15)
        assert len(upcoming) == 1

    def test_get_upcoming_milestones_not_pending(self, monitor, dashboard):
        """测试非pending状态的里程碑不包含在即将到来的里程碑中"""
        milestone = Milestone(
            milestone_id='achieved_milestone',
            name='Achieved Milestone',
            description='',
            target_date=datetime.now() + timedelta(days=3),
            status='achieved'
        )
        monitor.add_milestone('test_dashboard', milestone)
        
        upcoming = monitor.get_upcoming_milestones('test_dashboard')
        
        assert len(upcoming) == 0

    def test_get_upcoming_milestones_dashboard_not_found(self, monitor):
        """测试获取不存在的仪表板的即将到来的里程碑"""
        upcoming = monitor.get_upcoming_milestones('nonexistent_dashboard')
        
        assert upcoming == []

    def test_generate_progress_report(self, monitor, dashboard, sample_task, sample_milestone):
        """测试生成进度报告"""
        monitor.add_task('test_dashboard', sample_task)
        monitor.add_milestone('test_dashboard', sample_milestone)
        
        report = monitor.generate_progress_report('test_dashboard')
        
        assert isinstance(report, str)
        assert '实施进度报告' in report
        assert 'test_dashboard' in report or 'Test Dashboard' in report

    def test_generate_progress_report_not_found(self, monitor):
        """测试生成不存在的仪表板的进度报告"""
        report = monitor.generate_progress_report('nonexistent_dashboard')
        
        assert report == "仪表板不存在"

    def test_generate_progress_report_with_overdue_tasks(self, monitor, dashboard):
        """测试生成包含逾期任务的进度报告"""
        task = TaskProgress(
            task_id='overdue_task',
            task_name='Overdue Task',
            description='',
            status='in_progress',
            priority='high',
            due_date=datetime.now() - timedelta(days=5),
            assignee='test_user'
        )
        monitor.add_task('test_dashboard', task)
        
        report = monitor.generate_progress_report('test_dashboard')
        
        assert '逾期任务' in report
        assert 'overdue_task' in report or 'Overdue Task' in report

    def test_generate_progress_report_with_upcoming_milestones(self, monitor, dashboard):
        """测试生成包含即将到来的里程碑的进度报告"""
        milestone = Milestone(
            milestone_id='upcoming_milestone',
            name='Upcoming Milestone',
            description='',
            target_date=datetime.now() + timedelta(days=3),
            status='pending'
        )
        monitor.add_milestone('test_dashboard', milestone)
        
        report = monitor.generate_progress_report('test_dashboard')
        
        assert '即将到来的里程碑' in report
        assert 'upcoming_milestone' in report or 'Upcoming Milestone' in report

    def test_generate_progress_report_with_quality_metrics(self, monitor, dashboard, sample_metric):
        """测试生成包含质量指标的进度报告"""
        monitor.add_quality_metric('test_dashboard', sample_metric)
        
        report = monitor.generate_progress_report('test_dashboard')
        
        assert '质量指标' in report
        assert 'metric_1' in report or 'Test Metric' in report

    def test_export_dashboard_data_success(self, monitor, dashboard, sample_task, sample_milestone, sample_metric, tmp_path):
        """测试导出仪表板数据成功"""
        monitor.add_task('test_dashboard', sample_task)
        monitor.add_milestone('test_dashboard', sample_milestone)
        monitor.add_quality_metric('test_dashboard', sample_metric)
        
        export_path = tmp_path / "exported_dashboard.json"
        result = monitor.export_dashboard_data('test_dashboard', str(export_path))
        
        assert result == True
        assert export_path.exists()

    def test_export_dashboard_data_dashboard_not_found(self, monitor, tmp_path):
        """测试导出不存在的仪表板数据"""
        export_path = tmp_path / "exported_dashboard.json"
        result = monitor.export_dashboard_data('nonexistent_dashboard', str(export_path))
        
        assert result == False

    def test_export_dashboard_data_file_write_error(self, monitor, dashboard, tmp_path):
        """测试导出仪表板数据文件写入错误"""
        # 使用一个无效的路径（例如父目录不存在）
        invalid_path = str(tmp_path / "nonexistent_dir" / "dashboard.json")
        
        result = monitor.export_dashboard_data('test_dashboard', invalid_path)
        
        assert result == False


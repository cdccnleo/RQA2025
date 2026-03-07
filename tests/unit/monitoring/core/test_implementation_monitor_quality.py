#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实施监控器质量测试
测试覆盖 ImplementationMonitor 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
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


@pytest.fixture
def implementation_monitor(tmp_path):
    """创建实施监控器实例"""
    # 使用临时目录避免路径问题
    return ImplementationMonitor(data_dir=str(tmp_path / "monitoring"))


@pytest.fixture
def sample_task():
    """创建示例任务"""
    return TaskProgress(
        task_id='task_1',
        task_name='Test Task',
        description='Test task description',
        status='in_progress',
        priority='high'
    )


@pytest.fixture
def sample_milestone():
    """创建示例里程碑"""
    return Milestone(
        milestone_id='milestone_1',
        name='Test Milestone',
        description='Test milestone description',
        target_date=datetime.now() + timedelta(days=30),
        status='pending'
    )


class TestTaskProgress:
    """TaskProgress测试类"""

    def test_task_creation(self, sample_task):
        """测试任务创建"""
        assert sample_task.task_id == 'task_1'
        assert sample_task.task_name == 'Test Task'
        assert sample_task.status == 'in_progress'
        assert sample_task.priority == 'high'

    def test_task_update(self, sample_task):
        """测试任务更新"""
        sample_task.progress_percent = 50
        sample_task.updated_at = datetime.now()
        assert sample_task.progress_percent == 50


class TestMilestone:
    """Milestone测试类"""

    def test_milestone_creation(self, sample_milestone):
        """测试里程碑创建"""
        assert sample_milestone.milestone_id == 'milestone_1'
        assert sample_milestone.name == 'Test Milestone'
        assert sample_milestone.status == 'pending'

    def test_milestone_achieved(self, sample_milestone):
        """测试里程碑达成"""
        sample_milestone.status = 'achieved'
        sample_milestone.achieved_date = datetime.now()
        assert sample_milestone.status == 'achieved'
        assert sample_milestone.achieved_date is not None


class TestQualityMetric:
    """QualityMetric测试类"""

    def test_quality_metric_creation(self):
        """测试质量指标创建"""
        metric = QualityMetric(
            metric_id='metric_1',
            name='Test Coverage',
            description='Code test coverage',
            category='code_quality',
            target_value=80.0,
            current_value=75.0,
            unit='percent',
            trend='improving'
        )
        assert metric.metric_id == 'metric_1'
        assert metric.current_value == 75.0
        assert metric.trend == 'improving'


class TestImplementationMonitor:
    """ImplementationMonitor测试类"""

    def test_initialization(self, implementation_monitor):
        """测试初始化"""
        assert implementation_monitor.dashboards == {}
        assert hasattr(implementation_monitor, 'data_dir')

    def test_add_task(self, implementation_monitor, sample_task):
        """测试添加任务"""
        # 先创建仪表板
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        # 添加任务到仪表板
        result = implementation_monitor.add_task('dashboard_1', sample_task)
        assert result is True
        
        # 验证任务已添加（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        assert len(dashboard.tasks) > 0
        assert dashboard.tasks[0].task_id == 'task_1'

    def test_update_task(self, implementation_monitor, sample_task):
        """测试更新任务"""
        # 先创建仪表板并添加任务
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_task('dashboard_1', sample_task)
        
        # 使用update_task_progress更新任务
        result = implementation_monitor.update_task_progress('dashboard_1', 'task_1', 50, 'in_progress')
        assert result is True
        
        # 验证任务已更新（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        task = next((t for t in dashboard.tasks if t.task_id == 'task_1'), None)
        assert task is not None
        assert task.progress_percent == 50

    def test_get_task(self, implementation_monitor, sample_task):
        """测试获取任务（通过仪表板）"""
        # 先创建仪表板并添加任务
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_task('dashboard_1', sample_task)
        
        # 通过仪表板获取任务（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        task = next((t for t in dashboard.tasks if t.task_id == 'task_1'), None)
        assert task is not None
        assert task.task_id == 'task_1'

    def test_add_milestone(self, implementation_monitor, sample_milestone):
        """测试添加里程碑"""
        # 先创建仪表板
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        # 添加里程碑到仪表板
        result = implementation_monitor.add_milestone('dashboard_1', sample_milestone)
        assert result is True
        
        # 验证里程碑已添加（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        assert len(dashboard.milestones) > 0
        assert dashboard.milestones[0].milestone_id == 'milestone_1'

    def test_add_quality_metric(self, implementation_monitor):
        """测试添加质量指标"""
        # 先创建仪表板
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        metric = QualityMetric(
            metric_id='metric_1',
            name='Test Coverage',
            description='Code test coverage',
            category='code_quality',
            target_value=80.0,
            current_value=75.0,
            unit='percent',
            trend='improving'
        )
        
        # 添加质量指标到仪表板
        result = implementation_monitor.add_quality_metric('dashboard_1', metric)
        assert result is True
        
        # 验证指标已添加（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        assert len(dashboard.metrics) > 0
        assert dashboard.metrics[0].metric_id == 'metric_1'

    def test_get_dashboard(self, implementation_monitor, sample_task, sample_milestone):
        """测试获取仪表板"""
        # 创建仪表板并添加任务和里程碑
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_task('dashboard_1', sample_task)
        implementation_monitor.add_milestone('dashboard_1', sample_milestone)
        
        # 获取仪表板（直接访问dashboards字典）
        dashboard = implementation_monitor.dashboards['dashboard_1']
        assert isinstance(dashboard, ImplementationDashboard)
        assert len(dashboard.tasks) > 0
        assert len(dashboard.milestones) > 0

    def test_get_dashboard_summary(self, implementation_monitor, sample_task, sample_milestone):
        """测试获取仪表板摘要"""
        # 创建仪表板并添加任务和里程碑
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_task('dashboard_1', sample_task)
        implementation_monitor.add_milestone('dashboard_1', sample_milestone)
        
        # 获取摘要
        summary = implementation_monitor.get_dashboard_summary('dashboard_1')
        assert summary is not None
        assert 'dashboard_id' in summary
        assert 'overall_progress_percent' in summary
        assert 'task_summary' in summary
        assert 'milestone_summary' in summary

    def test_get_overdue_tasks(self, implementation_monitor, sample_task):
        """测试获取逾期任务"""
        # 创建仪表板并添加逾期任务
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        # 创建逾期任务
        overdue_task = TaskProgress(
            task_id='overdue_task',
            task_name='Overdue Task',
            description='Overdue task description',
            status='in_progress',
            priority='high',
            due_date=datetime.now() - timedelta(days=1)  # 昨天到期
        )
        implementation_monitor.add_task('dashboard_1', overdue_task)
        
        # 获取逾期任务
        overdue_tasks = implementation_monitor.get_overdue_tasks('dashboard_1')
        assert len(overdue_tasks) > 0
        assert overdue_tasks[0]['task_id'] == 'overdue_task'

    def test_get_upcoming_milestones(self, implementation_monitor, sample_milestone):
        """测试获取即将到来的里程碑"""
        # 创建仪表板并添加即将到来的里程碑
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        # 创建即将到来的里程碑（3天后）
        upcoming_milestone = Milestone(
            milestone_id='upcoming_milestone',
            name='Upcoming Milestone',
            description='Upcoming milestone description',
            target_date=datetime.now() + timedelta(days=3),
            status='pending'
        )
        implementation_monitor.add_milestone('dashboard_1', upcoming_milestone)
        
        # 获取即将到来的里程碑（7天内）
        upcoming = implementation_monitor.get_upcoming_milestones('dashboard_1', days_ahead=7)
        assert len(upcoming) > 0
        assert upcoming[0]['milestone_id'] == 'upcoming_milestone'

    def test_update_milestone_status(self, implementation_monitor, sample_milestone):
        """测试更新里程碑状态"""
        # 创建仪表板并添加里程碑
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_milestone('dashboard_1', sample_milestone)
        
        # 更新里程碑状态
        result = implementation_monitor.update_milestone_status('dashboard_1', 'milestone_1', 'achieved')
        assert result is True
        
        # 验证状态已更新
        dashboard = implementation_monitor.dashboards['dashboard_1']
        milestone = next((m for m in dashboard.milestones if m.milestone_id == 'milestone_1'), None)
        assert milestone is not None
        assert milestone.status == 'achieved'

    def test_update_quality_metric(self, implementation_monitor):
        """测试更新质量指标"""
        # 创建仪表板并添加质量指标
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        
        metric = QualityMetric(
            metric_id='metric_1',
            name='Test Coverage',
            description='Code test coverage',
            category='code_quality',
            target_value=80.0,
            current_value=75.0,
            unit='percent',
            trend='improving'
        )
        implementation_monitor.add_quality_metric('dashboard_1', metric)
        
        # 更新质量指标
        result = implementation_monitor.update_quality_metric('dashboard_1', 'metric_1', 80.0, 'stable')
        assert result is True
        
        # 验证指标已更新
        dashboard = implementation_monitor.dashboards['dashboard_1']
        updated_metric = next((m for m in dashboard.metrics if m.metric_id == 'metric_1'), None)
        assert updated_metric is not None
        assert updated_metric.current_value == 80.0
        assert updated_metric.trend == 'stable'

    def test_generate_progress_report(self, implementation_monitor, sample_task, sample_milestone):
        """测试生成进度报告"""
        # 创建仪表板并添加任务和里程碑
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        implementation_monitor.add_task('dashboard_1', sample_task)
        implementation_monitor.add_milestone('dashboard_1', sample_milestone)
        
        # 生成报告
        report = implementation_monitor.generate_progress_report('dashboard_1')
        assert isinstance(report, str)
        assert '实施进度报告' in report
        assert 'Test Dashboard' in report

    def test_get_dashboard_summary_nonexistent(self, implementation_monitor):
        """测试获取不存在的仪表板摘要"""
        summary = implementation_monitor.get_dashboard_summary('nonexistent')
        assert summary is None

    def test_get_overdue_tasks_nonexistent(self, implementation_monitor):
        """测试获取不存在仪表板的逾期任务"""
        overdue_tasks = implementation_monitor.get_overdue_tasks('nonexistent')
        assert overdue_tasks == []

    def test_get_upcoming_milestones_nonexistent(self, implementation_monitor):
        """测试获取不存在仪表板的即将到来的里程碑"""
        upcoming = implementation_monitor.get_upcoming_milestones('nonexistent')
        assert upcoming == []

    def test_update_task_progress_nonexistent_dashboard(self, implementation_monitor, sample_task):
        """测试更新不存在仪表板的任务进度"""
        result = implementation_monitor.update_task_progress('nonexistent', 'task_1', 50)
        assert result is False

    def test_update_task_progress_nonexistent_task(self, implementation_monitor, sample_task):
        """测试更新不存在任务的进度"""
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        result = implementation_monitor.update_task_progress('dashboard_1', 'nonexistent_task', 50)
        assert result is False

    def test_update_milestone_status_nonexistent_dashboard(self, implementation_monitor):
        """测试更新不存在仪表板的里程碑状态"""
        result = implementation_monitor.update_milestone_status('nonexistent', 'milestone_1', 'achieved')
        assert result is False

    def test_update_quality_metric_nonexistent_dashboard(self, implementation_monitor):
        """测试更新不存在仪表板的质量指标"""
        result = implementation_monitor.update_quality_metric('nonexistent', 'metric_1', 80.0)
        assert result is False

    def test_update_quality_metric_nonexistent_metric(self, implementation_monitor):
        """测试更新不存在质量指标"""
        dashboard = implementation_monitor.create_dashboard('dashboard_1', 'Test Dashboard', 'Test description')
        result = implementation_monitor.update_quality_metric('dashboard_1', 'nonexistent_metric', 80.0)
        assert result is False

    def test_add_task_nonexistent_dashboard(self, implementation_monitor, sample_task):
        """测试添加任务到不存在的仪表板"""
        result = implementation_monitor.add_task('nonexistent', sample_task)
        assert result is False

    def test_add_milestone_nonexistent_dashboard(self, implementation_monitor, sample_milestone):
        """测试添加里程碑到不存在的仪表板"""
        result = implementation_monitor.add_milestone('nonexistent', sample_milestone)
        assert result is False

    def test_add_quality_metric_nonexistent_dashboard(self, implementation_monitor):
        """测试添加质量指标到不存在的仪表板"""
        metric = QualityMetric(
            metric_id='metric_1',
            name='Test Coverage',
            description='Code test coverage',
            category='code_quality',
            target_value=80.0,
            current_value=75.0,
            unit='percent',
            trend='improving'
        )
        result = implementation_monitor.add_quality_metric('nonexistent', metric)
        assert result is False


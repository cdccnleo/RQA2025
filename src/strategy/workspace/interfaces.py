#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略工作空间接口定义
Strategy Workspace Interfaces

定义策略工作空间的核心接口，支持策略的可视化编辑、分析和模拟。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkspaceMode(Enum):

    """工作空间模式枚举"""
    DESIGN = "design"      # 设计模式
    ANALYSIS = "analysis"  # 分析模式
    SIMULATION = "simulation"  # 模拟模式
    MONITORING = "monitoring"  # 监控模式


@dataclass
class WorkspaceConfig:

    """工作空间配置"""
    workspace_id: str
    name: str
    description: str
    mode: WorkspaceMode
    strategy_ids: List[str]
    settings: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class VisualElement:

    """可视化元素"""
    element_id: str
    element_type: str  # 'chart', 'table', 'metric', 'control'
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    data_source: str  # 数据源标识
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


class IWorkspaceService(ABC):

    """
    工作空间服务接口
    Workspace Service Interface

    定义工作空间的核心功能接口。
    """

    @abstractmethod
    def create_workspace(self, config: WorkspaceConfig) -> str:
        """
        创建工作空间

        Args:
            config: 工作空间配置

        Returns:
            str: 工作空间ID
        """

    @abstractmethod
    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """
        获取工作空间

        Args:
            workspace_id: 工作空间ID

        Returns:
            Optional[WorkspaceConfig]: 工作空间配置
        """

    @abstractmethod
    def update_workspace(self, workspace_id: str, config: WorkspaceConfig) -> bool:
        """
        更新工作空间

        Args:
            workspace_id: 工作空间ID
            config: 新的工作空间配置

        Returns:
            bool: 更新是否成功
        """

    @abstractmethod
    def delete_workspace(self, workspace_id: str) -> bool:
        """
        删除工作空间

        Args:
            workspace_id: 工作空间ID

        Returns:
            bool: 删除是否成功
        """

    @abstractmethod
    def list_workspaces(self, mode: Optional[WorkspaceMode] = None) -> List[WorkspaceConfig]:
        """
        列出工作空间

        Args:
            mode: 工作空间模式过滤器

        Returns:
            List[WorkspaceConfig]: 工作空间配置列表
        """


class IWorkspaceEditor(ABC):

    """
    工作空间编辑器接口
    Workspace Editor Interface

    定义工作空间的可视化编辑功能接口。
    """

    @abstractmethod
    def add_visual_element(self, workspace_id: str, element: VisualElement) -> bool:
        """
        添加可视化元素

        Args:
            workspace_id: 工作空间ID
            element: 可视化元素

        Returns:
            bool: 添加是否成功
        """

    @abstractmethod
    def update_visual_element(self, workspace_id: str, element_id: str,


                              updates: Dict[str, Any]) -> bool:
        """
        更新可视化元素

        Args:
            workspace_id: 工作空间ID
            element_id: 元素ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """

    @abstractmethod
    def remove_visual_element(self, workspace_id: str, element_id: str) -> bool:
        """
        移除可视化元素

        Args:
            workspace_id: 工作空间ID
            element_id: 元素ID

        Returns:
            bool: 移除是否成功
        """

    @abstractmethod
    def get_workspace_layout(self, workspace_id: str) -> List[VisualElement]:
        """
        获取工作空间布局

        Args:
            workspace_id: 工作空间ID

        Returns:
            List[VisualElement]: 可视化元素列表
        """

    @abstractmethod
    def export_workspace_layout(self, workspace_id: str, format: str = "json") -> str:
        """
        导出工作空间布局

        Args:
            workspace_id: 工作空间ID
            format: 导出格式 ('json', 'xml', 'yaml')

        Returns:
            str: 导出的布局数据
        """


class IWorkspaceAnalyzer(ABC):

    """
    工作空间分析器接口
    Workspace Analyzer Interface

    定义工作空间的策略分析功能接口。
    """

    @abstractmethod
    def analyze_strategy_performance(self, strategy_id: str,


                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        分析策略性能

        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, Any]: 性能分析结果
        """

    @abstractmethod
    def compare_strategies(self, strategy_ids: List[str],


                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个策略

        Args:
            strategy_ids: 策略ID列表
            metrics: 比较指标列表

        Returns:
            Dict[str, Any]: 策略比较结果
        """

    @abstractmethod
    def generate_performance_report(self, strategy_id: str,


                                    report_type: str = "comprehensive") -> str:
        """
        生成性能报告

        Args:
            strategy_id: 策略ID
            report_type: 报告类型 ('summary', 'detailed', 'comprehensive')

        Returns:
            str: 报告文件路径
        """

    @abstractmethod
    def detect_performance_anomalies(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        检测性能异常

        Args:
            strategy_id: 策略ID

        Returns:
            List[Dict[str, Any]]: 异常检测结果
        """


class IWorkspaceSimulator(ABC):

    """
    工作空间模拟器接口
    Workspace Simulator Interface

    定义工作空间的策略模拟功能接口。
    """

    @abstractmethod
    def run_simulation(self, strategy_id: str, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行策略模拟

        Args:
            strategy_id: 策略ID
            simulation_config: 模拟配置

        Returns:
            Dict[str, Any]: 模拟结果
        """

    @abstractmethod
    def run_monte_carlo_simulation(self, strategy_id: str,


                                   n_simulations: int = 1000,
                                   time_horizon: int = 252) -> Dict[str, Any]:
        """
        运行蒙特卡洛模拟

        Args:
            strategy_id: 策略ID
            n_simulations: 模拟次数
            time_horizon: 时间跨度（天）

        Returns:
            Dict[str, Any]: 蒙特卡洛模拟结果
        """

    @abstractmethod
    def stress_test_strategy(self, strategy_id: str,


                             stress_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对策略进行压力测试

        Args:
            strategy_id: 策略ID
            stress_scenarios: 压力测试场景

        Returns:
            Dict[str, Any]: 压力测试结果
        """

    @abstractmethod
    def sensitivity_analysis(self, strategy_id: str,


                             parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        敏感性分析

        Args:
            strategy_id: 策略ID
            parameter_ranges: 参数范围

        Returns:
            Dict[str, Any]: 敏感性分析结果
        """


class IWorkspacePersistence(ABC):

    """
    工作空间持久化接口
    Workspace Persistence Interface

    处理工作空间配置和数据的持久化存储。
    """

    @abstractmethod
    def save_workspace_config(self, config: WorkspaceConfig) -> bool:
        """
        保存工作空间配置

        Args:
            config: 工作空间配置

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_workspace_config(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """
        加载工作空间配置

        Args:
            workspace_id: 工作空间ID

        Returns:
            Optional[WorkspaceConfig]: 工作空间配置
        """

    @abstractmethod
    def save_visual_layout(self, workspace_id: str, layout: List[VisualElement]) -> bool:
        """
        保存可视化布局

        Args:
            workspace_id: 工作空间ID
            layout: 可视化布局

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_visual_layout(self, workspace_id: str) -> List[VisualElement]:
        """
        加载可视化布局

        Args:
            workspace_id: 工作空间ID

        Returns:
            List[VisualElement]: 可视化元素列表
        """

    @abstractmethod
    def backup_workspace(self, workspace_id: str) -> str:
        """
        备份工作空间

        Args:
            workspace_id: 工作空间ID

        Returns:
            str: 备份文件路径
        """


# 导出所有接口
__all__ = [
    'WorkspaceMode',
    'WorkspaceConfig',
    'VisualElement',
    'IWorkspaceService',
    'IWorkspaceEditor',
    'IWorkspaceAnalyzer',
    'IWorkspaceSimulator',
    'IWorkspacePersistence'
]

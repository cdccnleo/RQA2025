"""
Strategy Lifecycle Automation Module
策略生命周期自动化模块

This module provides automated strategy lifecycle management for quantitative trading
此模块为量化交易提供自动化策略生命周期管理

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):

    """Strategy status enumeration"""
    DRAFT = "draft"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class LifecycleEvent(Enum):

    """Lifecycle event types"""
    CREATED = "created"
    UPDATED = "updated"
    TESTED = "tested"
    DEPLOYED = "deployed"
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class StrategyVersion:

    """
    Strategy version data class
    策略版本数据类
    """
    version_id: str
    strategy_id: str
    version_number: str
    status: str
    created_at: datetime
    created_by: str
    description: str
    performance_metrics: Dict[str, Any] = None
    risk_metrics: Dict[str, Any] = None
    code_hash: Optional[str] = None
    config_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class LifecycleEvent:

    """
    Lifecycle event data class
    生命周期事件数据类
    """
    event_id: str
    strategy_id: str
    event_type: str
    timestamp: datetime
    user: str
    details: Dict[str, Any]
    previous_status: Optional[str] = None
    new_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class StrategyLifecycleManager:

    """
    Strategy Lifecycle Manager Class
    策略生命周期管理器类

    Manages the complete lifecycle of trading strategies
    管理交易策略的完整生命周期
    """

    def __init__(self, manager_name: str = "default_strategy_lifecycle_manager"):
        """
        Initialize strategy lifecycle manager
        初始化策略生命周期管理器

        Args:
            manager_name: Name of the lifecycle manager
                        生命周期管理器名称
        """
        self.manager_name = manager_name
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_versions: Dict[str, List[StrategyVersion]] = defaultdict(list)
        self.lifecycle_events: Dict[str, List[LifecycleEvent]] = defaultdict(list)

        # Lifecycle policies
        self.auto_promotion_enabled = True
        self.performance_thresholds = {
            'min_sharpe_ratio': 1.5,
            'max_drawdown': 0.1,
            'min_win_rate': 0.55
        }
        self.testing_requirements = {
            'min_backtest_period_days': 365,
            'required_walk_forward_tests': 3,
            'stress_test_required': True
        }

        # Status transition rules
        self.transition_rules = self._define_transition_rules()

        logger.info(f"Strategy lifecycle manager {manager_name} initialized")

    def _define_transition_rules(self) -> Dict[str, List[str]]:
        """
        Define allowed status transitions
        定义允许的状态转换

        Returns:
            dict: Transition rules mapping
                  转换规则映射
        """
        return {
            StrategyStatus.DRAFT.value: [
                StrategyStatus.DEVELOPMENT.value
            ],
            StrategyStatus.DEVELOPMENT.value: [
                StrategyStatus.TESTING.value,
                StrategyStatus.DRAFT.value
            ],
            StrategyStatus.TESTING.value: [
                StrategyStatus.STAGING.value,
                StrategyStatus.DEVELOPMENT.value
            ],
            StrategyStatus.STAGING.value: [
                StrategyStatus.PRODUCTION.value,
                StrategyStatus.TESTING.value
            ],
            StrategyStatus.PRODUCTION.value: [
                StrategyStatus.DEPRECATED.value,
                StrategyStatus.STAGING.value
            ],
            StrategyStatus.DEPRECATED.value: [
                StrategyStatus.ARCHIVED.value,
                StrategyStatus.PRODUCTION.value
            ],
            StrategyStatus.ARCHIVED.value: []
        }

    def register_strategy(self,


                          strategy_id: str,
                          name: str,
                          description: str,
                          created_by: str,
                          initial_config: Dict[str, Any]) -> str:
        """
        Register a new strategy
        注册新策略

        Args:
            strategy_id: Unique strategy identifier
                        唯一策略标识符
            name: Strategy name
                 策略名称
            description: Strategy description
                        策略描述
            created_by: User who created the strategy
                       创建策略的用户
            initial_config: Initial strategy configuration
                           初始策略配置

        Returns:
            str: Registered strategy ID
                 已注册的策略ID
        """
        strategy = {
            'strategy_id': strategy_id,
            'name': name,
            'description': description,
            'status': StrategyStatus.DRAFT.value,
            'created_at': datetime.now(),
            'created_by': created_by,
            'updated_at': datetime.now(),
            'updated_by': created_by,
            'current_version': None,
            'config': initial_config,
            'performance_history': [],
            'risk_history': [],
            'deployment_history': []
        }

        self.strategies[strategy_id] = strategy

        # Create initial lifecycle event
        self._log_lifecycle_event(
            strategy_id=strategy_id,
            event_type=LifecycleEvent.CREATED,
            user=created_by,
            details={'initial_config': initial_config}
        )

        logger.info(f"Registered strategy: {name} ({strategy_id})")
        return strategy_id

    def create_version(self,


                       strategy_id: str,
                       version_number: str,
                       created_by: str,
                       description: str,
                       code_hash: Optional[str] = None,
                       config_hash: Optional[str] = None) -> str:
        """
        Create a new version of a strategy
        创建策略的新版本

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            version_number: Version number (e.g., "1.0.0")
                           版本号（如"1.0.0"）
            created_by: User creating the version
                       创建版本的用户
            description: Version description
                        版本描述
            code_hash: Hash of the strategy code
                      策略代码的哈希
            config_hash: Hash of the strategy configuration
                        策略配置的哈希

        Returns:
            str: Created version ID
                 创建的版本ID
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        version_id = f"{strategy_id}_v{version_number}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        version = StrategyVersion(
            version_id=version_id,
            strategy_id=strategy_id,
            version_number=version_number,
            status=StrategyStatus.DRAFT.value,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            code_hash=code_hash,
            config_hash=config_hash
        )

        self.strategy_versions[strategy_id].append(version)
        self.strategies[strategy_id]['current_version'] = version_id

        # Log version creation event
        self._log_lifecycle_event(
            strategy_id=strategy_id,
            event_type=LifecycleEvent.UPDATED,
            user=created_by,
            details={
                'action': 'version_created',
                'version_id': version_id,
                'version_number': version_number
            }
        )

        logger.info(f"Created version {version_number} for strategy {strategy_id}")
        return version_id

    def update_strategy_status(self,


                               strategy_id: str,
                               new_status: StrategyStatus,
                               updated_by: str,
                               reason: str = "",
                               performance_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update strategy status
        更新策略状态

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            new_status: New status
                       新状态
            updated_by: User making the update
                       进行更新的用户
            reason: Reason for status change
                   状态变更原因
            performance_data: Performance data for validation
                            用于验证的性能数据

        Returns:
            bool: True if status updated successfully
                  状态更新成功返回True
        """
        if strategy_id not in self.strategies:
            return False

        current_status = self.strategies[strategy_id]['status']

        # Check if transition is allowed
        if new_status.value not in self.transition_rules.get(current_status, []):
            logger.error(f"Invalid status transition from {current_status} to {new_status.value}")
            return False

        # Validate status change requirements
        if not self._validate_status_change(strategy_id, current_status, new_status.value, performance_data):
            logger.error(f"Status change validation failed for strategy {strategy_id}")
            return False

        # Update strategy
        self.strategies[strategy_id]['status'] = new_status.value
        self.strategies[strategy_id]['updated_at'] = datetime.now()
        self.strategies[strategy_id]['updated_by'] = updated_by

        # Update current version status if exists
        current_version_id = self.strategies[strategy_id]['current_version']
        if current_version_id:
            for version in self.strategy_versions[strategy_id]:
                if version.version_id == current_version_id:
                    version.status = new_status.value
                    break

        # Log status change event
        self._log_lifecycle_event(
            strategy_id=strategy_id,
            event_type=LifecycleEvent.PROMOTED if self._is_promotion(
                current_status, new_status.value) else LifecycleEvent.DEMOTED,
            user=updated_by,
            details={
                'reason': reason,
                'performance_data': performance_data
            },
            previous_status=current_status,
            new_status=new_status.value
        )

        logger.info(f"Updated strategy {strategy_id} status to {new_status.value}")
        return True

    def _validate_status_change(self,


                                strategy_id: str,
                                current_status: str,
                                new_status: str,
                                performance_data: Optional[Dict[str, Any]]) -> bool:
        """
        Validate status change requirements
        验证状态变更要求

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            current_status: Current status
                          当前状态
            new_status: New status
                       新状态
            performance_data: Performance data
                            性能数据

        Returns:
            bool: True if validation passes
                  验证通过返回True
        """
        # Skip validation for demotion or draft status
        if new_status in [StrategyStatus.DRAFT.value, StrategyStatus.DEPRECATED.value]:
            return True

        # Require performance data for production promotion
        if new_status == StrategyStatus.PRODUCTION.value:
            if not performance_data:
                return False

            # Check performance thresholds
            sharpe_ratio = performance_data.get('sharpe_ratio', 0)
            max_drawdown = performance_data.get('max_drawdown', 1)
            win_rate = performance_data.get('win_rate', 0)

            if (sharpe_ratio < self.performance_thresholds['min_sharpe_ratio']
                or max_drawdown > self.performance_thresholds['max_drawdown']
                    or win_rate < self.performance_thresholds['min_win_rate']):
                return False

        # Require testing completion for staging promotion
        if new_status == StrategyStatus.STAGING.value:
            if current_status == StrategyStatus.TESTING.value:
                # Check if testing requirements are met
                return self._validate_testing_completion(strategy_id)

        return True

    def _validate_testing_completion(self, strategy_id: str) -> bool:
        """
        Validate that testing requirements are completed
        验证测试要求是否完成

        Args:
            strategy_id: Strategy identifier
                        策略标识符

        Returns:
            bool: True if testing is complete
                  测试完成返回True
        """
        # This would typically check test results, backtest reports, etc.
        # For now, return True as placeholder
        return True

    def _is_promotion(self, current_status: str, new_status: str) -> bool:
        """
        Check if status change is a promotion
        检查状态变更是否为晋升

        Args:
            current_status: Current status
                          当前状态
            new_status: New status
                       新状态

        Returns:
            bool: True if promotion
                  如果是晋升则返回True
        """
        status_order = [
            StrategyStatus.DRAFT.value,
            StrategyStatus.DEVELOPMENT.value,
            StrategyStatus.TESTING.value,
            StrategyStatus.STAGING.value,
            StrategyStatus.PRODUCTION.value
        ]

        try:
            current_index = status_order.index(current_status)
            new_index = status_order.index(new_status)
            return new_index > current_index
        except ValueError:
            return False

    def _log_lifecycle_event(self,


                             strategy_id: str,
                             event_type: LifecycleEvent,
                             user: str,
                             details: Dict[str, Any],
                             previous_status: Optional[str] = None,
                             new_status: Optional[str] = None) -> None:
        """
        Log a lifecycle event
        记录生命周期事件

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            event_type: Type of event
                       事件类型
            user: User who triggered the event
                 触发事件的用户
            details: Event details
                    事件详情
            previous_status: Previous status
                           之前的状态
            new_status: New status
                       新状态
        """
        event = LifecycleEvent(
            event_id=f"event_{strategy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S_ % f')}",
            strategy_id=strategy_id,
            event_type=event_type.value,
            timestamp=datetime.now(),
            user=user,
            details=details,
            previous_status=previous_status,
            new_status=new_status
        )

        self.lifecycle_events[strategy_id].append(event)

    def get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive strategy information
        获取全面的策略信息

        Args:
            strategy_id: Strategy identifier
                        策略标识符

        Returns:
            dict: Strategy information or None if not found
                  策略信息，如果未找到则返回None
        """
        if strategy_id not in self.strategies:
            return None

        strategy = self.strategies[strategy_id].copy()
        strategy['versions'] = [v.to_dict() for v in self.strategy_versions[strategy_id]]
        strategy['lifecycle_events'] = [e.to_dict() for e in self.lifecycle_events[strategy_id]]

        return strategy

    def list_strategies(self,


                        status_filter: Optional[str] = None,
                        created_by_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List strategies with optional filters
        列出策略，可选过滤器

        Args:
            status_filter: Status filter
                          状态过滤器
            created_by_filter: Creator filter
                             创建者过滤器

        Returns:
            list: List of strategies
                  策略列表
        """
        strategies = []

        for strategy_id, strategy in self.strategies.items():
            if status_filter and strategy['status'] != status_filter:
                continue
            if created_by_filter and strategy['created_by'] != created_by_filter:
                continue

            strategies.append(self.get_strategy_info(strategy_id))

        return strategies

    def archive_strategy(self, strategy_id: str, archived_by: str, reason: str = "") -> bool:
        """
        Archive a deprecated strategy
        归档已弃用的策略

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            archived_by: User performing the archive
                        执行归档的用户
            reason: Reason for archiving
                   归档原因

        Returns:
            bool: True if archived successfully
                  归档成功返回True
        """
        if strategy_id not in self.strategies:
            return False

        strategy = self.strategies[strategy_id]

        if strategy['status'] != StrategyStatus.DEPRECATED.value:
            logger.error(f"Cannot archive strategy {strategy_id} - not deprecated")
            return False

        # Update status
        strategy['status'] = StrategyStatus.ARCHIVED.value
        strategy['updated_at'] = datetime.now()
        strategy['updated_by'] = archived_by

        # Log archiving event
        self._log_lifecycle_event(
            strategy_id=strategy_id,
            event_type=LifecycleEvent.ARCHIVED,
            user=archived_by,
            details={'reason': reason}
        )

        logger.info(f"Archived strategy: {strategy_id}")
        return True

    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """
        Get lifecycle management statistics
        获取生命周期管理统计信息

        Returns:
            dict: Lifecycle statistics
                  生命周期统计信息
        """
        total_strategies = len(self.strategies)
        status_counts = defaultdict(int)

        for strategy in self.strategies.values():
            status_counts[strategy['status']] += 1

        total_versions = sum(len(versions) for versions in self.strategy_versions.values())
        total_events = sum(len(events) for events in self.lifecycle_events.values())

        return {
            'total_strategies': total_strategies,
            'status_distribution': dict(status_counts),
            'total_versions': total_versions,
            'total_lifecycle_events': total_events,
            'average_versions_per_strategy': total_versions / max(total_strategies, 1),
            'average_events_per_strategy': total_events / max(total_strategies, 1)
        }

    def export_strategy_history(self, strategy_id: str, filepath: str) -> bool:
        """
        Export strategy lifecycle history to file
        将策略生命周期历史导出到文件

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            filepath: Export file path
                     导出文件路径

        Returns:
            bool: True if export successful
                  导出成功返回True
        """
        try:
            strategy_info = self.get_strategy_info(strategy_id)
            if not strategy_info:
                return False

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(strategy_info, f, indent=2, default=str)

            logger.info(f"Exported strategy history for {strategy_id} to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export strategy history: {str(e)}")
            return False

    def check_auto_promotion(self, strategy_id: str, performance_data: Dict[str, Any]) -> Optional[str]:
        """
        Check if strategy qualifies for automatic promotion
        检查策略是否符合自动晋升条件

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            performance_data: Latest performance data
                            最新性能数据

        Returns:
            str: Recommended new status or None
                 推荐的新状态或None
        """
        if not self.auto_promotion_enabled:
            return None

        if strategy_id not in self.strategies:
            return None

        current_status = self.strategies[strategy_id]['status']

        # Define auto - promotion rules
        if current_status == StrategyStatus.TESTING.value:
            # Check if performance meets staging criteria
            if self._meets_performance_thresholds(performance_data):
                return StrategyStatus.STAGING.value

        elif current_status == StrategyStatus.STAGING.value:
            # Check if performance is consistently good
            recent_performance = self._get_recent_performance(strategy_id, days=30)
            if len(recent_performance) >= 7:  # At least a week of data
                avg_sharpe = np.mean([p.get('sharpe_ratio', 0) for p in recent_performance])
                if avg_sharpe >= self.performance_thresholds['min_sharpe_ratio']:
                    return StrategyStatus.PRODUCTION.value

        return None

    def _meets_performance_thresholds(self, performance_data: Dict[str, Any]) -> bool:
        """
        Check if performance meets required thresholds
        检查性能是否符合所需阈值

        Args:
            performance_data: Performance data
                            性能数据

        Returns:
            bool: True if thresholds met
                  达到阈值返回True
        """
        sharpe_ratio = performance_data.get('sharpe_ratio', 0)
        max_drawdown = performance_data.get('max_drawdown', 1)
        win_rate = performance_data.get('win_rate', 0)

        return (sharpe_ratio >= self.performance_thresholds['min_sharpe_ratio']
                and max_drawdown <= self.performance_thresholds['max_drawdown']
                and win_rate >= self.performance_thresholds['min_win_rate'])

    def _get_recent_performance(self, strategy_id: str, days: int) -> List[Dict[str, Any]]:
        """
        Get recent performance data for a strategy
        获取策略的近期性能数据

        Args:
            strategy_id: Strategy identifier
                        策略标识符
            days: Number of days to look back
                 回溯天数

        Returns:
            list: Recent performance data
                  近期性能数据
        """
        # This would typically query a performance database
        # For now, return empty list as placeholder
        return []


# Global strategy lifecycle manager instance
# 全局策略生命周期管理器实例
strategy_lifecycle_manager = StrategyLifecycleManager()

__all__ = [
    'StrategyStatus',
    'LifecycleEvent',
    'StrategyVersion',
    'LifecycleEvent',
    'StrategyLifecycleManager',
    'strategy_lifecycle_manager'
]

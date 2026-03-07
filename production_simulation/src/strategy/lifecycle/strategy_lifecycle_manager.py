#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略生命周期管理器
Strategy Lifecycle Manager

基于业务流程驱动架构，实现完整的策略生命周期管理：
1. 策略创建与初始化
2. 策略开发与测试
3. 策略部署与运行
4. 策略监控与优化
5. 策略退市与清理
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from ..interfaces.strategy_interfaces import (
    IStrategyService, IStrategyFactory, IStrategyPersistence,
    StrategyConfig, StrategyStatus
)

logger = logging.getLogger(__name__)


class LifecycleStage(Enum):

    """生命周期阶段枚举"""
    CREATED = "created"           # 已创建
    DESIGNING = "designing"       # 设计中
    DEVELOPING = "developing"     # 开发中
    TESTING = "testing"          # 测试中
    BACKTESTING = "backtesting"   # 回测中
    OPTIMIZING = "optimizing"     # 优化中
    VALIDATING = "validating"     # 验证中
    DEPLOYING = "deploying"       # 部署中
    RUNNING = "running"          # 运行中
    MONITORING = "monitoring"     # 监控中
    MAINTAINING = "maintaining"   # 维护中
    RETIRING = "retiring"         # 退市中
    RETIRED = "retired"           # 已退市


@dataclass
class LifecycleEvent:

    """生命周期事件"""
    event_id: str
    strategy_id: str
    stage: LifecycleStage
    action: str
    details: Dict[str, Any]
    timestamp: datetime = None
    user: str = ""

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StrategyLifecycle:

    """策略生命周期状态"""
    strategy_id: str
    current_stage: LifecycleStage
    stage_history: List[LifecycleEvent]
    next_allowed_actions: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


class StrategyLifecycleManager:

    """
    策略生命周期管理器
    Strategy Lifecycle Manager

    基于业务流程驱动架构，实现策略的完整生命周期管理。
    """

    def __init__(self, strategy_service: IStrategyService,


                 strategy_factory: IStrategyFactory,
                 persistence: IStrategyPersistence):
        """
        初始化生命周期管理器

        Args:
            strategy_service: 策略服务实例
            strategy_factory: 策略工厂实例
            persistence: 持久化服务实例
        """
        self.strategy_service = strategy_service
        self.strategy_factory = strategy_factory
        self.persistence = persistence
        self.lifecycle_states: Dict[str, StrategyLifecycle] = {}
        self.stage_transitions = self._define_stage_transitions()

        logger.info("策略生命周期管理器已初始化")

    def _define_stage_transitions(self) -> Dict[LifecycleStage, List[LifecycleStage]]:
        """定义阶段转换规则"""
        return {
            LifecycleStage.CREATED: [
                LifecycleStage.DESIGNING,
                LifecycleStage.RETIRED
            ],
            LifecycleStage.DESIGNING: [
                LifecycleStage.DEVELOPING,
                LifecycleStage.CREATED
            ],
            LifecycleStage.DEVELOPING: [
                LifecycleStage.TESTING,
                LifecycleStage.DESIGNING
            ],
            LifecycleStage.TESTING: [
                LifecycleStage.BACKTESTING,
                LifecycleStage.DEVELOPING
            ],
            LifecycleStage.BACKTESTING: [
                LifecycleStage.OPTIMIZING,
                LifecycleStage.TESTING
            ],
            LifecycleStage.OPTIMIZING: [
                LifecycleStage.VALIDATING,
                LifecycleStage.BACKTESTING
            ],
            LifecycleStage.VALIDATING: [
                LifecycleStage.DEPLOYING,
                LifecycleStage.OPTIMIZING
            ],
            LifecycleStage.DEPLOYING: [
                LifecycleStage.RUNNING,
                LifecycleStage.VALIDATING
            ],
            LifecycleStage.RUNNING: [
                LifecycleStage.MONITORING,
                LifecycleStage.MAINTAINING
            ],
            LifecycleStage.MONITORING: [
                LifecycleStage.MAINTAINING,
                LifecycleStage.RETIRING
            ],
            LifecycleStage.MAINTAINING: [
                LifecycleStage.RUNNING,
                LifecycleStage.RETIRING
            ],
            LifecycleStage.RETIRING: [
                LifecycleStage.RETIRED
            ],
            LifecycleStage.RETIRED: []
        }

    def create_strategy_lifecycle(self, config: StrategyConfig, user: str = "") -> str:
        """
        创建策略生命周期

        Args:
            config: 策略配置
            user: 操作用户

        Returns:
            str: 策略ID
        """
        # 创建策略
        strategy_id = config.strategy_id

        # 初始化生命周期状态
        lifecycle = StrategyLifecycle(
            strategy_id=strategy_id,
            current_stage=LifecycleStage.CREATED,
            stage_history=[],
            next_allowed_actions=self._get_allowed_actions(LifecycleStage.CREATED),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # 记录初始事件
        initial_event = LifecycleEvent(
            event_id=f"{strategy_id}_created_{datetime.now().isoformat()}",
            strategy_id=strategy_id,
            stage=LifecycleStage.CREATED,
            action="strategy_created",
            details={"config": config.__dict__},
            user=user
        )

        lifecycle.stage_history.append(initial_event)
        self.lifecycle_states[strategy_id] = lifecycle

        logger.info(f"策略生命周期已创建: {strategy_id}")
        return strategy_id

    def transition_stage(self, strategy_id: str, target_stage: LifecycleStage,


                         action_details: Dict[str, Any] = None, user: str = "") -> bool:
        """
        转换生命周期阶段

        Args:
            strategy_id: 策略ID
            target_stage: 目标阶段
            action_details: 行动详情
            user: 操作用户

        Returns:
            bool: 转换是否成功
        """
        if strategy_id not in self.lifecycle_states:
            logger.error(f"策略不存在: {strategy_id}")
            return False

        lifecycle = self.lifecycle_states[strategy_id]

        # 检查转换是否允许
        if target_stage not in self.stage_transitions.get(lifecycle.current_stage, []):
            logger.error(f"不允许的阶段转换: {lifecycle.current_stage} -> {target_stage}")
            return False

        # 执行阶段转换逻辑
        if not self._execute_stage_transition(lifecycle, target_stage, action_details):
            logger.error(f"阶段转换执行失败: {strategy_id}")
            return False

        # 更新生命周期状态
        lifecycle.current_stage = target_stage
        lifecycle.next_allowed_actions = self._get_allowed_actions(target_stage)
        lifecycle.updated_at = datetime.now()

        # 记录转换事件
        transition_event = LifecycleEvent(
            event_id=f"{strategy_id}_{target_stage.value}_{datetime.now().isoformat()}",
            strategy_id=strategy_id,
            stage=target_stage,
            action=f"transition_to_{target_stage.value}",
            details=action_details or {},
            user=user
        )

        lifecycle.stage_history.append(transition_event)

        logger.info(f"策略阶段转换成功: {strategy_id} -> {target_stage.value}")
        return True

    def _execute_stage_transition(self, lifecycle: StrategyLifecycle,


                                  target_stage: LifecycleStage,
                                  action_details: Dict[str, Any]) -> bool:
        """
        执行阶段转换逻辑

        Args:
            lifecycle: 策略生命周期
            target_stage: 目标阶段
            action_details: 行动详情

        Returns:
            bool: 执行是否成功
        """
        strategy_id = lifecycle.strategy_id

        try:
            if target_stage == LifecycleStage.DEVELOPING:
                # 开始开发阶段
                return self._start_development(strategy_id, action_details)

            elif target_stage == LifecycleStage.TESTING:
                # 开始测试阶段
                return self._start_testing(strategy_id, action_details)

            elif target_stage == LifecycleStage.BACKTESTING:
                # 开始回测阶段
                return self._start_backtesting(strategy_id, action_details)

            elif target_stage == LifecycleStage.DEPLOYING:
                # 开始部署阶段
                return self._start_deployment(strategy_id, action_details)

            elif target_stage == LifecycleStage.RUNNING:
                # 开始运行阶段
                return self._start_running(strategy_id, action_details)

            elif target_stage == LifecycleStage.RETIRING:
                # 开始退市阶段
                return self._start_retirement(strategy_id, action_details)

            return True

        except Exception as e:
            logger.error(f"阶段转换执行异常: {strategy_id} -> {target_stage.value}, 错误: {e}")
            return False

    def _start_development(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始开发阶段"""
        # 验证策略配置
        config = self.strategy_service.get_strategy(strategy_id)
        if not config:
            return False

        # 设置开发环境
        # 这里可以添加具体的开发环境设置逻辑

        logger.info(f"策略开发环境已设置: {strategy_id}")
        return True

    def _start_testing(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始测试阶段"""
        # 执行单元测试
        # 这里可以添加具体的测试逻辑

        logger.info(f"策略测试已开始: {strategy_id}")
        return True

    def _start_backtesting(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始回测阶段"""
        # 启动回测任务
        # 这里可以添加具体的回测逻辑

        logger.info(f"策略回测已开始: {strategy_id}")
        return True

    def _start_deployment(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始部署阶段"""
        # 执行部署逻辑
        success = self.strategy_service.start_strategy(strategy_id)
        if success:
            logger.info(f"策略部署成功: {strategy_id}")
        return success

    def _start_running(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始运行阶段"""
        # 更新策略状态
        success = self.strategy_service.get_strategy_status(strategy_id) == StrategyStatus.RUNNING
        if success:
            logger.info(f"策略运行已开始: {strategy_id}")
        return success

    def _start_retirement(self, strategy_id: str, action_details: Dict[str, Any]) -> bool:
        """开始退市阶段"""
        # 执行退市逻辑
        success = self.strategy_service.stop_strategy(strategy_id)
        if success:
            logger.info(f"策略退市已开始: {strategy_id}")
        return success

    def _get_allowed_actions(self, stage: LifecycleStage) -> List[str]:
        """获取允许的行动"""
        actions_map = {
            LifecycleStage.CREATED: ["start_design", "retire"],
            LifecycleStage.DESIGNING: ["start_development", "cancel"],
            LifecycleStage.DEVELOPING: ["start_testing", "back_to_design"],
            LifecycleStage.TESTING: ["start_backtesting", "back_to_development"],
            LifecycleStage.BACKTESTING: ["start_optimization", "back_to_testing"],
            LifecycleStage.OPTIMIZING: ["start_validation", "back_to_backtesting"],
            LifecycleStage.VALIDATING: ["deploy", "back_to_optimization"],
            LifecycleStage.DEPLOYING: ["start_running", "back_to_validation"],
            LifecycleStage.RUNNING: ["start_monitoring", "maintenance"],
            LifecycleStage.MONITORING: ["maintenance", "retire"],
            LifecycleStage.MAINTAINING: ["resume_running", "retire"],
            LifecycleStage.RETIRING: ["complete_retirement"],
            LifecycleStage.RETIRED: []
        }
        return actions_map.get(stage, [])

    def get_lifecycle_status(self, strategy_id: str) -> Optional[StrategyLifecycle]:
        """
        获取策略生命周期状态

        Args:
            strategy_id: 策略ID

        Returns:
            Optional[StrategyLifecycle]: 生命周期状态
        """
        return self.lifecycle_states.get(strategy_id)

    def get_stage_history(self, strategy_id: str) -> List[LifecycleEvent]:
        """
        获取阶段历史

        Args:
            strategy_id: 策略ID

        Returns:
            List[LifecycleEvent]: 阶段历史事件列表
        """
        lifecycle = self.lifecycle_states.get(strategy_id)
        if lifecycle:
            return lifecycle.stage_history
        return []

    def get_available_actions(self, strategy_id: str) -> List[str]:
        """
        获取可用的行动

        Args:
            strategy_id: 策略ID

        Returns:
            List[str]: 可用行动列表
        """
        lifecycle = self.lifecycle_states.get(strategy_id)
        if lifecycle:
            return lifecycle.next_allowed_actions
        return []

    def validate_stage_transition(self, strategy_id: str, target_stage: LifecycleStage) -> Dict[str, Any]:
        """
        验证阶段转换

        Args:
            strategy_id: 策略ID
            target_stage: 目标阶段

        Returns:
            Dict[str, Any]: 验证结果
        """
        if strategy_id not in self.lifecycle_states:
            return {
                "valid": False,
                "reason": "策略不存在"
            }

        lifecycle = self.lifecycle_states[strategy_id]

        if target_stage not in self.stage_transitions.get(lifecycle.current_stage, []):
            return {
                "valid": False,
                "reason": f"不允许的阶段转换: {lifecycle.current_stage.value} -> {target_stage.value}"
            }

        # 检查前置条件
        preconditions_result = self._check_transition_preconditions(lifecycle, target_stage)
        if not preconditions_result["valid"]:
            return preconditions_result

        return {
            "valid": True,
            "current_stage": lifecycle.current_stage.value,
            "target_stage": target_stage.value
        }

    def _check_transition_preconditions(self, lifecycle: StrategyLifecycle,


                                        target_stage: LifecycleStage) -> Dict[str, Any]:
        """检查转换前置条件"""
        strategy_id = lifecycle.strategy_id

        if target_stage == LifecycleStage.RUNNING:
            # 检查策略是否已正确部署
            status = self.strategy_service.get_strategy_status(strategy_id)
        if status != StrategyStatus.RUNNING:
            return {
                "valid": False,
                "reason": "策略未正确部署"
            }

        elif target_stage == LifecycleStage.RETIRED:
            # 检查是否有未完成的交易
            # 这里可以添加具体的检查逻辑

            pass

        return {"valid": True}

    def get_lifecycle_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        获取生命周期摘要

        Args:
            strategy_id: 策略ID

        Returns:
            Dict[str, Any]: 生命周期摘要
        """
        lifecycle = self.lifecycle_states.get(strategy_id)
        if not lifecycle:
            return {}

        # 计算阶段停留时间
        stage_durations = self._calculate_stage_durations(lifecycle)

        return {
            "strategy_id": strategy_id,
            "current_stage": lifecycle.current_stage.value,
            "total_stages": len(lifecycle.stage_history),
            "stage_durations": stage_durations,
            "created_at": lifecycle.created_at.isoformat(),
            "updated_at": lifecycle.updated_at.isoformat(),
            "available_actions": lifecycle.next_allowed_actions
        }

    def _calculate_stage_durations(self, lifecycle: StrategyLifecycle) -> Dict[str, float]:
        """计算各阶段停留时间"""
        durations = {}
        events = lifecycle.stage_history

        for i, event in enumerate(events):
            if i < len(events) - 1:
                next_event = events[i + 1]
                duration = (next_event.timestamp - event.timestamp).total_seconds()
                stage_name = event.stage.value
                if stage_name not in durations:
                    durations[stage_name] = 0
                durations[stage_name] += duration

        # 添加当前阶段的停留时间
        if events:
            current_duration = (datetime.now() - events[-1].timestamp).total_seconds()
            current_stage = lifecycle.current_stage.value
            durations[current_stage] = current_duration

        return durations


# 导出类
__all__ = [
    'LifecycleStage',
    'LifecycleEvent',
    'StrategyLifecycle',
    'StrategyLifecycleManager'
]

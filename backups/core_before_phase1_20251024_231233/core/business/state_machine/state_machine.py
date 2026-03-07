"""
业务流程状态机
负责业务流程的状态转换和生命周期管理
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict

from .business_process_enums import BusinessProcessState
from .business_process_models import ProcessInstance, ProcessConfig

logger = logging.getLogger(__name__)


class BusinessProcessStateMachine:

    """业务流程状态机 - 增强版"""

    def __init__(self, process_config: ProcessConfig):
        self.process_config = process_config
        self.state_enter_time: Optional[datetime] = None
        self.state_timeouts: Dict[BusinessProcessState, int] = {}
        self.state_listeners: Dict[BusinessProcessState, List[Callable]] = defaultdict(list)
        self.transition_hooks: Dict[tuple, List[Callable]] = defaultdict(list)

        # 初始化状态超时配置
        self._initialize_state_timeouts()

    def _initialize_state_timeouts(self) -> None:
        """初始化状态超时配置"""
        # 从配置中读取超时设置，如果没有则使用默认值
        config_timeouts = getattr(self.process_config, 'state_timeouts', {})

        default_timeouts = {
            BusinessProcessState.DATA_COLLECTING: 60,      # 1分钟
            BusinessProcessState.DATA_QUALITY_CHECKING: 30,  # 30秒
            BusinessProcessState.FEATURE_EXTRACTING: 120,   # 2分钟
            BusinessProcessState.GPU_ACCELERATING: 300,     # 5分钟
            BusinessProcessState.MODEL_PREDICTING: 180,     # 3分钟
            BusinessProcessState.MODEL_ENSEMBLING: 60,      # 1分钟
            BusinessProcessState.SIGNAL_GENERATING: 30,     # 30秒
            BusinessProcessState.STRATEGY_DECIDING: 60,     # 1分钟
            BusinessProcessState.RISK_CHECKING: 45,         # 45秒
            BusinessProcessState.ORDER_GENERATING: 30,      # 30秒
            BusinessProcessState.ORDER_ROUTING: 15,         # 15秒
            BusinessProcessState.EXECUTING: 120,            # 2分钟
            BusinessProcessState.MONITORING: 300,           # 5分钟
        }

        # 合并配置和默认值
        for state, default_timeout in default_timeouts.items():
            self.state_timeouts[state] = config_timeouts.get(state.name.lower(), default_timeout)

    def add_state_listener(self, state: BusinessProcessState, listener: Callable) -> None:
        """添加状态监听器"""
        self.state_listeners[state].append(listener)

    def remove_state_listener(self, state: BusinessProcessState, listener: Callable) -> None:
        """移除状态监听器"""
        if listener in self.state_listeners[state]:
            self.state_listeners[state].remove(listener)

    def add_transition_hook(self, from_state: BusinessProcessState, to_state: BusinessProcessState, hook: Callable) -> None:
        """添加状态转换钩子"""
        self.transition_hooks[(from_state, to_state)].append(hook)

    def check_state_timeout(self, current_state: BusinessProcessState) -> bool:
        """检查状态是否超时"""
        if not self.state_enter_time:
            return False

        timeout_seconds = self.state_timeouts.get(current_state, 300)  # 默认5分钟
        elapsed = (datetime.now() - self.state_enter_time).total_seconds()

        return elapsed > timeout_seconds

    def get_state_duration(self, state: BusinessProcessState) -> Optional[float]:
        """获取状态持续时间"""
        if self.state_enter_time and self.current_state == state:
            return (datetime.now() - self.state_enter_time).total_seconds()
        return None

    def get_state_history_summary(self) -> Dict[str, Any]:
        """获取状态历史摘要"""
        # 这里可以实现状态历史的统计和分析
        return {
            'total_transitions': 0,  # 需要实际实现
            'average_state_duration': 0.0,
            'most_frequent_state': None,
            'timeout_states': []
        }

    def transition_to(self, process_instance: ProcessInstance, new_state: BusinessProcessState,
                      event_data: Optional[Dict[str, Any]] = None) -> bool:
        """转换到新状态"""
        old_state = process_instance.current_state

        # 检查状态转换是否有效
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(f"无效的状态转换: {old_state} -> {new_state}")
            return False

        try:
            # 执行转换前钩子
            for hook in self.transition_hooks[(old_state, new_state)]:
                try:
                    hook(process_instance, old_state, new_state, event_data)
                except Exception as e:
                    logger.error(f"状态转换钩子执行失败: {e}")

            # 更新状态和时间
            old_enter_time = self.state_enter_time
            process_instance.update_state(new_state)
            self.state_enter_time = datetime.now()

            # 计算状态持续时间
            duration = None
            if old_enter_time:
                duration = (self.state_enter_time - old_enter_time).total_seconds()

            # 触发状态监听器
            for listener in self.state_listeners[new_state]:
                try:
                    listener(process_instance, new_state, event_data, duration)
                except Exception as e:
                    logger.error(f"状态监听器执行失败: {e}")

            # 记录状态转换事件
            logger.info(
                f"流程 {process_instance.instance_id} 状态转换: {old_state.name} -> {new_state.name}")

            # 发布状态转换事件
            self._publish_state_change_event(process_instance, old_state, new_state, duration)

            return True

        except Exception as e:
            logger.error(f"状态转换失败: {e}")
            # 可以在这里添加错误恢复逻辑
            return False

    def _is_valid_transition(self, from_state: BusinessProcessState, to_state: BusinessProcessState) -> bool:
        """检查状态转换是否有效"""
        # 定义有效的状态转换规则
        valid_transitions = {
            BusinessProcessState.IDLE: [BusinessProcessState.DATA_COLLECTING],
            BusinessProcessState.DATA_COLLECTING: [BusinessProcessState.DATA_QUALITY_CHECKING, BusinessProcessState.ERROR],
            BusinessProcessState.DATA_QUALITY_CHECKING: [BusinessProcessState.FEATURE_EXTRACTING, BusinessProcessState.ERROR],
            BusinessProcessState.FEATURE_EXTRACTING: [BusinessProcessState.GPU_ACCELERATING, BusinessProcessState.MODEL_PREDICTING, BusinessProcessState.ERROR],
            BusinessProcessState.GPU_ACCELERATING: [BusinessProcessState.MODEL_PREDICTING, BusinessProcessState.ERROR],
            BusinessProcessState.MODEL_PREDICTING: [BusinessProcessState.MODEL_ENSEMBLING, BusinessProcessState.SIGNAL_GENERATING, BusinessProcessState.ERROR],
            BusinessProcessState.MODEL_ENSEMBLING: [BusinessProcessState.SIGNAL_GENERATING, BusinessProcessState.ERROR],
            BusinessProcessState.SIGNAL_GENERATING: [BusinessProcessState.STRATEGY_DECIDING, BusinessProcessState.ERROR],
            BusinessProcessState.STRATEGY_DECIDING: [BusinessProcessState.RISK_CHECKING, BusinessProcessState.ERROR],
            BusinessProcessState.RISK_CHECKING: [BusinessProcessState.ORDER_GENERATING, BusinessProcessState.ERROR],
            BusinessProcessState.ORDER_GENERATING: [BusinessProcessState.ORDER_ROUTING, BusinessProcessState.ERROR],
            BusinessProcessState.ORDER_ROUTING: [BusinessProcessState.EXECUTING, BusinessProcessState.ERROR],
            BusinessProcessState.EXECUTING: [BusinessProcessState.MONITORING, BusinessProcessState.COMPLETED, BusinessProcessState.ERROR],
            BusinessProcessState.MONITORING: [BusinessProcessState.COMPLETED],
        }

        # 任何状态都可以转换到取消状态
        if to_state == BusinessProcessState.CANCELLED:
            return True

        return to_state in valid_transitions.get(from_state, [])

    def _publish_state_change_event(self, process_instance: ProcessInstance,
                                    old_state: BusinessProcessState, new_state: BusinessProcessState,
                                    duration: Optional[float]) -> None:
        """发布状态转换事件"""
        # 这里可以集成事件总线来发布状态转换事件
        event_data = {
            'process_id': process_instance.instance_id,
            'old_state': old_state.name,
            'new_state': new_state.name,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }

        # 如果有事件总线集成，可以在这里发布事件
        # self.event_bus.publish(EventType.PROCESS_STATE_CHANGED, event_data)
        logger.debug(f"状态转换事件: {event_data}")

    # 属性访问器
    @property
    def current_state(self) -> Optional[BusinessProcessState]:
        """获取当前状态"""
        return getattr(self, '_current_state', None)

    @current_state.setter
    def current_state(self, state: BusinessProcessState) -> None:
        """设置当前状态"""
        self._current_state = state

"""
业务流程状态机组件

职责:
- 管理业务流程状态转换
- 验证状态转换的合法性
- 提供状态监听和钩子机制
- 超时检测
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque, defaultdict

from ..models.process_models import BusinessProcessState

logger = logging.getLogger(__name__)


class BusinessProcessStateMachine:
    """
    业务流程状态机组件

    提供完整的状态管理能力：
    - 状态转换和验证
    - 状态监听器
    - 转换钩子
    - 超时检测
    """

    def __init__(self, config: 'StateMachineConfig'):
        """
        初始化状态机

        Args:
            config: 状态机配置
        """
        self.config = config
        self.current_state = BusinessProcessState.IDLE
        self.state_history: deque = deque(maxlen=100)
        self.transition_rules = self._initialize_transition_rules()
        self._lock = threading.RLock()

        # 状态管理增强功能
        self.state_enter_time = time.time()
        self.state_timeouts = self._initialize_state_timeouts()
        self.state_listeners: Dict[BusinessProcessState, List[Callable]] = defaultdict(list)
        self.transition_hooks: Dict[tuple, List[Callable]] = defaultdict(list)

        logger.info("业务流程状态机初始化完成")

    def transition_to(self, new_state: BusinessProcessState,
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        状态转换

        Args:
            new_state: 新状态
            context: 上下文数据

        Returns:
            bool: 转换是否成功
        """
        with self._lock:
            if self._is_valid_transition(self.current_state, new_state):
                old_state = self.current_state
                old_enter_time = self.state_enter_time

                # 执行转换前钩子
                if self.config.enable_hooks:
                    self._execute_transition_hooks(old_state, new_state, context)

                # 更新状态
                self.current_state = new_state
                self.state_enter_time = time.time()

                # 计算上一状态持续时间
                duration = self.state_enter_time - old_enter_time

                # 记录状态历史
                self.state_history.append({
                    'from_state': old_state,
                    'to_state': new_state,
                    'timestamp': time.time(),
                    'duration': duration,
                    'context': context or {}
                })

                # 触发状态监听器
                if self.config.enable_listeners:
                    self._notify_state_listeners(new_state, context)

                if self.config.enable_state_logging:
                    logger.debug(f"状态转换: {old_state.value} -> {new_state.value} (持续{duration:.1f}s)")

                return True
            else:
                logger.warning(f"无效的状态转换: {self.current_state.value} -> {new_state.value}")
                return False

    def get_current_state(self) -> BusinessProcessState:
        """获取当前状态"""
        return self.current_state

    def get_state_history(self) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return list(self.state_history)

    def get_state_history_summary(self) -> List[Dict[str, Any]]:
        """获取状态历史摘要"""
        return [{
            'transition': f"{record['from_state'].value} -> {record['to_state'].value}",
            'timestamp': record['timestamp'],
            'duration': record.get('duration', 0),
            'context': record.get('context', {})
        } for record in self.state_history]

    def add_state_listener(self, state: BusinessProcessState, listener: Callable):
        """添加状态监听器"""
        self.state_listeners[state].append(listener)
        logger.debug(f"添加状态监听器: {state.value}")

    def remove_state_listener(self, state: BusinessProcessState, listener: Callable):
        """移除状态监听器"""
        if state in self.state_listeners and listener in self.state_listeners[state]:
            self.state_listeners[state].remove(listener)
            logger.debug(f"移除状态监听器: {state.value}")

    def add_transition_hook(self, from_state: BusinessProcessState,
                           to_state: BusinessProcessState, hook: Callable):
        """添加转换钩子"""
        self.transition_hooks[(from_state, to_state)].append(hook)
        logger.debug(f"添加转换钩子: {from_state.value} -> {to_state.value}")

    def check_state_timeout(self) -> Optional[BusinessProcessState]:
        """检查状态是否超时"""
        if not self.config.enable_timeout_check:
            return None

        if self.current_state not in self.state_timeouts:
            return None

        timeout_seconds = self.state_timeouts[self.current_state]
        elapsed = time.time() - self.state_enter_time

        if elapsed > timeout_seconds:
            logger.warning(f"状态超时: {self.current_state.value} ({elapsed:.1f}s > {timeout_seconds}s)")
            return BusinessProcessState.ERROR

        return None

    def get_state_duration(self) -> float:
        """获取当前状态持续时间（秒）"""
        return time.time() - self.state_enter_time

    def reset(self):
        """重置状态机"""
        with self._lock:
            self.current_state = BusinessProcessState.IDLE
            self.state_history.clear()
            self.state_enter_time = time.time()
            logger.info("状态机已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取状态机状态"""
        return {
            'current_state': self.current_state.value,
            'state_duration': self.get_state_duration(),
            'history_size': len(self.state_history),
            'listeners_count': sum(len(listeners) for listeners in self.state_listeners.values()),
            'hooks_count': sum(len(hooks) for hooks in self.transition_hooks.values())
        }

    # ==================== 私有方法 ====================

    def _initialize_state_timeouts(self) -> Dict[BusinessProcessState, int]:
        """初始化状态超时配置（秒）"""
        default_timeout = self.config.default_state_timeout

        return {
            BusinessProcessState.DATA_COLLECTING: default_timeout,
            BusinessProcessState.DATA_QUALITY_CHECKING: 180,
            BusinessProcessState.FEATURE_EXTRACTING: 600,
            BusinessProcessState.GPU_ACCELERATING: 1800,
            BusinessProcessState.MODEL_PREDICTING: 900,
            BusinessProcessState.MODEL_ENSEMBLING: 600,
            BusinessProcessState.STRATEGY_DECIDING: 300,
            BusinessProcessState.SIGNAL_GENERATING: 180,
            BusinessProcessState.RISK_CHECKING: 120,
            BusinessProcessState.COMPLIANCE_VERIFYING: 300,
            BusinessProcessState.ORDER_GENERATING: 60,
            BusinessProcessState.ORDER_EXECUTING: 30,
            BusinessProcessState.MONITORING_FEEDBACK: 60,
        }

    def _initialize_transition_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """初始化状态转换规则"""
        rules = {}
        rules.update(self._get_initialization_rules())
        rules.update(self._get_data_processing_rules())
        rules.update(self._get_model_processing_rules())
        rules.update(self._get_strategy_execution_rules())
        rules.update(self._get_order_execution_rules())
        rules.update(self._get_finalization_rules())
        rules.update(self._get_error_handling_rules())
        return rules

    def _get_initialization_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """初始化阶段转换规则"""
        return {
            BusinessProcessState.IDLE: [
                BusinessProcessState.DATA_COLLECTING,
                BusinessProcessState.ERROR
            ]
        }

    def _get_data_processing_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """数据处理阶段转换规则"""
        return {
            BusinessProcessState.DATA_COLLECTING: [
                BusinessProcessState.DATA_QUALITY_CHECKING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.DATA_QUALITY_CHECKING: [
                BusinessProcessState.FEATURE_EXTRACTING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.FEATURE_EXTRACTING: [
                BusinessProcessState.GPU_ACCELERATING,
                BusinessProcessState.MODEL_PREDICTING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.GPU_ACCELERATING: [
                BusinessProcessState.MODEL_PREDICTING,
                BusinessProcessState.ERROR
            ]
        }

    def _get_model_processing_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """模型处理阶段转换规则"""
        return {
            BusinessProcessState.MODEL_PREDICTING: [
                BusinessProcessState.MODEL_ENSEMBLING,
                BusinessProcessState.STRATEGY_DECIDING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.MODEL_ENSEMBLING: [
                BusinessProcessState.STRATEGY_DECIDING,
                BusinessProcessState.ERROR
            ]
        }

    def _get_strategy_execution_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """策略执行阶段转换规则"""
        return {
            BusinessProcessState.STRATEGY_DECIDING: [
                BusinessProcessState.SIGNAL_GENERATING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.SIGNAL_GENERATING: [
                BusinessProcessState.RISK_CHECKING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.RISK_CHECKING: [
                BusinessProcessState.COMPLIANCE_VERIFYING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.COMPLIANCE_VERIFYING: [
                BusinessProcessState.ORDER_GENERATING,
                BusinessProcessState.ERROR
            ]
        }

    def _get_order_execution_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """订单执行阶段转换规则"""
        return {
            BusinessProcessState.ORDER_GENERATING: [
                BusinessProcessState.ORDER_EXECUTING,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.ORDER_EXECUTING: [
                BusinessProcessState.MONITORING_FEEDBACK,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.MONITORING_FEEDBACK: [
                BusinessProcessState.COMPLETED,
                BusinessProcessState.ERROR
            ]
        }

    def _get_finalization_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """完成阶段转换规则"""
        return {
            BusinessProcessState.COMPLETED: [
                BusinessProcessState.IDLE
            ]
        }

    def _get_error_handling_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """错误处理和暂停恢复转换规则"""
        return {
            BusinessProcessState.ERROR: [
                BusinessProcessState.IDLE,
                BusinessProcessState.PAUSED
            ],
            BusinessProcessState.PAUSED: [
                BusinessProcessState.RESUMED,
                BusinessProcessState.ERROR
            ],
            BusinessProcessState.RESUMED: [
                BusinessProcessState.DATA_COLLECTING,
                BusinessProcessState.FEATURE_EXTRACTING,
                BusinessProcessState.MODEL_PREDICTING,
                BusinessProcessState.STRATEGY_DECIDING,
                BusinessProcessState.SIGNAL_GENERATING,
                BusinessProcessState.RISK_CHECKING,
                BusinessProcessState.COMPLIANCE_VERIFYING,
                BusinessProcessState.ORDER_GENERATING,
                BusinessProcessState.ORDER_EXECUTING,
                BusinessProcessState.MONITORING_FEEDBACK
            ]
        }

    def _execute_transition_hooks(self, from_state: BusinessProcessState,
                                  to_state: BusinessProcessState,
                                  context: Optional[Dict[str, Any]]):
        """执行转换钩子"""
        hooks = self.transition_hooks.get((from_state, to_state), [])
        for hook in hooks:
            try:
                hook(from_state, to_state, context)
            except Exception as e:
                logger.error(f"转换钩子执行失败: {e}")

    def _notify_state_listeners(self, state: BusinessProcessState,
                               context: Optional[Dict[str, Any]]):
        """通知状态监听器"""
        listeners = self.state_listeners.get(state, [])
        for listener in listeners:
            try:
                listener(state, context)
            except Exception as e:
                logger.error(f"状态监听器执行失败: {e}")

    def _is_valid_transition(self, from_state: BusinessProcessState,
                            to_state: BusinessProcessState) -> bool:
        """检查状态转换是否有效"""
        try:
            # 边界情况检查
            if from_state is None or to_state is None:
                logger.warning("状态转换参数为空")
                return False

            if not isinstance(from_state, BusinessProcessState) or not isinstance(to_state, BusinessProcessState):
                logger.warning("状态转换参数类型不正确")
                return False

            # 获取允许的转换状态
            allowed_transitions = self.transition_rules.get(from_state, [])

            # 检查目标状态是否在允许列表中
            if to_state not in allowed_transitions:
                logger.warning(f"无效转换: {from_state.value} -> {to_state.value}")
                return False

            # 特殊规则验证
            if not self._validate_special_transition_rules(from_state, to_state):
                return False

            return True

        except Exception as e:
            logger.error(f"状态转换验证异常: {e}")
            return False

    def _validate_special_transition_rules(self, from_state: BusinessProcessState,
                                          to_state: BusinessProcessState) -> bool:
        """验证特殊状态转换规则"""
        try:
            # 防止从完成状态到错误状态
            if from_state == BusinessProcessState.COMPLETED and to_state == BusinessProcessState.ERROR:
                logger.warning("不允许从完成状态直接转换到错误状态")
                return False

            # 防止从错误状态到完成状态
            if from_state == BusinessProcessState.ERROR and to_state == BusinessProcessState.COMPLETED:
                logger.warning("不允许从错误状态直接转换到完成状态")
                return False

            # 验证暂停/恢复逻辑
            if (from_state == BusinessProcessState.PAUSED and
                to_state != BusinessProcessState.RESUMED and
                to_state != BusinessProcessState.ERROR):
                logger.warning("从暂停状态只能转换到恢复或错误状态")
                return False

            return True

        except Exception as e:
            logger.error(f"特殊规则验证异常: {e}")
            return False


# 类型注解导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..configs.orchestrator_configs import StateMachineConfig

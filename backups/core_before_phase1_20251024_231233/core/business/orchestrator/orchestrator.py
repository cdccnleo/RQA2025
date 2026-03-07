#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程编排器
基于业务流程及依赖关系的架构编排实现 - 优化版 3.0.0
"""

import time
import logging
import threading
import json
import os
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
import gc

from ...foundation.base import BaseComponent, ComponentStatus, ComponentHealth, generate_id
from ...foundation.exceptions.core_exceptions import OrchestratorException
from ...architecture_layers import (
    CoreServicesLayer,
    InfrastructureLayer,
    DataManagementLayer,
    FeatureProcessingLayer,
    ModelInferenceLayer,
    StrategyDecisionLayer,
    RiskComplianceLayer,
    TradingExecutionLayer,
    MonitoringFeedbackLayer
)
from ...business_process.pool.process_instance_pool import ProcessInstancePool

logger = logging.getLogger(__name__)


class BusinessProcessState(Enum):

    """业务流程状态 - 增强版"""
    IDLE = "idle"                           # 空闲状态
    DATA_COLLECTING = "data_collecting"     # 数据采集中
    DATA_QUALITY_CHECKING = "data_quality_checking"  # 数据质量检查中
    FEATURE_EXTRACTING = "feature_extracting"  # 特征提取中
    GPU_ACCELERATING = "gpu_accelerating"   # GPU加速中
    MODEL_PREDICTING = "model_predicting"   # 模型预测中
    MODEL_ENSEMBLING = "model_ensembling"   # 模型集成中
    STRATEGY_DECIDING = "strategy_deciding"  # 策略决策中
    SIGNAL_GENERATING = "signal_generating"  # 信号生成中
    RISK_CHECKING = "risk_checking"         # 风险检查中
    COMPLIANCE_VERIFYING = "compliance_verifying"  # 合规验证中
    ORDER_GENERATING = "order_generating"   # 订单生成中
    ORDER_EXECUTING = "order_executing"     # 订单执行中
    MONITORING_FEEDBACK = "monitoring_feedback"  # 监控反馈中
    COMPLETED = "completed"                 # 完成状态
    ERROR = "error"                         # 错误状态
    PAUSED = "paused"                       # 暂停状态
    RESUMED = "resumed"                     # 恢复状态


class EventType(Enum):

    """事件类型定义 - 增强版"""
    # 数据层事件
    DATA_COLLECTION_STARTED = "data_collection_started"
    DATA_COLLECTED = "data_collected"
    DATA_QUALITY_CHECKED = "data_quality_checked"
    DATA_QUALITY_ALERT = "data_quality_alert"
    DATA_STORED = "data_stored"
    DATA_VALIDATED = "data_validated"

    # 特征层事件
    FEATURE_EXTRACTION_STARTED = "feature_extraction_started"
    FEATURES_EXTRACTED = "features_extracted"
    GPU_ACCELERATION_STARTED = "gpu_acceleration_started"
    GPU_ACCELERATION_COMPLETED = "gpu_acceleration_completed"
    FEATURE_PROCESSING_COMPLETED = "feature_processing_completed"

    # 模型层事件
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_PREDICTION_STARTED = "model_prediction_started"
    MODEL_PREDICTION_READY = "model_prediction_ready"
    MODEL_ENSEMBLE_STARTED = "model_ensemble_started"
    MODEL_ENSEMBLE_READY = "model_ensemble_ready"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_EVALUATED = "model_evaluated"

    # 策略层事件
    STRATEGY_DECISION_STARTED = "strategy_decision_started"
    STRATEGY_DECISION_READY = "strategy_decision_ready"
    SIGNAL_GENERATION_STARTED = "signal_generation_started"
    SIGNALS_GENERATED = "signals_generated"
    PARAMETER_OPTIMIZATION_STARTED = "parameter_optimization_started"
    PARAMETER_OPTIMIZATION_COMPLETED = "parameter_optimization_completed"

    # 风控层事件
    RISK_CHECK_STARTED = "risk_check_started"
    RISK_CHECK_COMPLETED = "risk_check_completed"
    RISK_REJECTED = "risk_rejected"
    COMPLIANCE_VERIFICATION_STARTED = "compliance_verification_started"
    COMPLIANCE_VERIFIED = "compliance_verified"
    COMPLIANCE_REJECTED = "compliance_rejected"
    REAL_TIME_MONITORING_ALERT = "real_time_monitoring_alert"

    # 交易层事件
    ORDER_GENERATION_STARTED = "order_generation_started"
    ORDERS_GENERATED = "orders_generated"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_MODIFIED = "order_modified"
    TRADE_CONFIRMED = "trade_confirmed"

    # 监控层事件
    PERFORMANCE_ALERT = "performance_alert"
    BUSINESS_ALERT = "business_alert"
    TRADING_CYCLE_COMPLETED = "trading_cycle_completed"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"

    # 流程控制事件
    PROCESS_STARTED = "process_started"
    PROCESS_PAUSED = "process_paused"
    PROCESS_RESUMED = "process_resumed"
    PROCESS_COMPLETED = "process_completed"
    PROCESS_ERROR = "process_error"
    PROCESS_ROLLBACK = "process_rollback"


@dataclass
class ProcessConfig:

    """流程配置 - 优化版"""
    process_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 3600  # 秒
    auto_rollback: bool = True
    parallel_execution: bool = False
    steps: List[Dict[str, Any]] = None
    parameters: Dict[str, Any] = None
    memory_limit: int = 100  # MB，内存限制

    def __post_init__(self):

        if self.steps is None:
            self.steps = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ProcessInstance:

    """流程实例 - 优化版"""
    instance_id: str
    process_config: ProcessConfig
    status: BusinessProcessState
    start_time: float
    end_time: float = None
    current_step: str = ""
    progress: float = 0.0
    error_message: str = ""
    context: Dict[str, Any] = None
    memory_usage: float = 0.0  # MB
    last_updated: float = None

    def __post_init__(self):

        if self.context is None:
            self.context = {}
        if self.last_updated is None:
            self.last_updated = time.time()

    def update_memory_usage(self, usage: float):
        """更新内存使用"""
        self.memory_usage = usage
        self.last_updated = time.time()


class EventBus:

    """事件总线 - 简化版"""

    def __init__(self):

        self.handlers = defaultdict(list)
        self.event_history = deque(maxlen=1000)  # 限制历史记录大小

    def subscribe(self, event_type: EventType, handler: callable):
        """订阅事件"""
        self.handlers[event_type].append(handler)

    def publish(self, event_type: EventType, data: dict):
        """发布事件"""
        event = {'type': event_type, 'data': data, 'timestamp': time.time()}
        self.event_history.append(event)

        for handler in self.handlers[event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件处理器异常: {e}")

    def get_event_history(self) -> List[dict]:
        """获取事件历史"""
        return list(self.event_history)


class BusinessProcessStateMachine:

    """业务流程状态机 - 优化版"""

    def __init__(self):

        self.current_state = BusinessProcessState.IDLE
        self.state_history = deque(maxlen=100)  # 限制历史记录大小
        self.transition_rules = self._initialize_transition_rules()
        self._lock = threading.RLock()

        # 状态管理增强功能
        self.state_enter_time = time.time()  # 当前状态进入时间
        self.state_timeouts = self._initialize_state_timeouts()  # 状态超时配置
        self.state_listeners: Dict[BusinessProcessState,
                                   List[Callable]] = defaultdict(list)  # 状态监听器
        self.transition_hooks: Dict[tuple, List[Callable]] = defaultdict(list)  # 转换钩子

    def _initialize_state_timeouts(self) -> Dict[BusinessProcessState, int]:
        """初始化状态超时配置（秒）"""
        return {
            BusinessProcessState.DATA_COLLECTING: 300,      # 5分钟
            BusinessProcessState.DATA_QUALITY_CHECKING: 180,  # 3分钟
            BusinessProcessState.FEATURE_EXTRACTING: 600,    # 10分钟
            BusinessProcessState.GPU_ACCELERATING: 1800,     # 30分钟
            BusinessProcessState.MODEL_PREDICTING: 900,      # 15分钟
            BusinessProcessState.MODEL_ENSEMBLING: 600,      # 10分钟
            BusinessProcessState.STRATEGY_DECIDING: 300,     # 5分钟
            BusinessProcessState.SIGNAL_GENERATING: 180,     # 3分钟
            BusinessProcessState.RISK_CHECKING: 120,         # 2分钟
            BusinessProcessState.COMPLIANCE_VERIFYING: 300,  # 5分钟
            BusinessProcessState.ORDER_GENERATING: 60,       # 1分钟
            BusinessProcessState.ORDER_EXECUTING: 30,        # 30秒
            BusinessProcessState.MONITORING_FEEDBACK: 60,    # 1分钟
        }

    def add_state_listener(self, state: BusinessProcessState, listener: Callable):
        """添加状态监听器"""
        self.state_listeners[state].append(listener)

    def remove_state_listener(self, state: BusinessProcessState, listener: Callable):
        """移除状态监听器"""
        if state in self.state_listeners and listener in self.state_listeners[state]:
            self.state_listeners[state].remove(listener)

    def add_transition_hook(self, from_state: BusinessProcessState, to_state: BusinessProcessState, hook: Callable):
        """添加转换钩子"""
        self.transition_hooks[(from_state, to_state)].append(hook)

    def check_state_timeout(self) -> Optional[BusinessProcessState]:
        """检查状态是否超时"""
        if self.current_state not in self.state_timeouts:
            return None

        timeout_seconds = self.state_timeouts[self.current_state]
        elapsed = time.time() - self.state_enter_time

        if elapsed > timeout_seconds:
            logger.warning(
                f"状态 {self.current_state.value} 超时 ({elapsed:.1f}s > {timeout_seconds}s)")
            return BusinessProcessState.ERROR

        return None

    def get_state_duration(self) -> float:
        """获取当前状态持续时间"""
        return time.time() - self.state_enter_time

    def get_state_history_summary(self) -> List[Dict[str, Any]]:
        """获取状态历史摘要"""
        return [{
            'transition': f"{record['from_state'].value} -> {record['to_state'].value}",
            'timestamp': record['timestamp'],
            'duration': record.get('duration', 0),
            'context': record.get('context', {})
        } for record in self.state_history]

    def _initialize_transition_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """初始化状态转换规则 - 重构版"""
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
        """获取初始化阶段的状态转换规则"""
        return {
            BusinessProcessState.IDLE: [
                BusinessProcessState.DATA_COLLECTING,
                BusinessProcessState.ERROR
            ]
        }

    def _get_data_processing_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """获取数据处理阶段的状态转换规则"""
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
        """获取模型处理阶段的状态转换规则"""
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
        """获取策略执行阶段的状态转换规则"""
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
        """获取订单执行阶段的状态转换规则"""
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
        """获取完成阶段的状态转换规则"""
        return {
            BusinessProcessState.COMPLETED: [
                BusinessProcessState.IDLE
            ]
        }

    def _get_error_handling_rules(self) -> Dict[BusinessProcessState, List[BusinessProcessState]]:
        """获取错误处理和暂停恢复的状态转换规则"""
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

    def transition_to(self, new_state: BusinessProcessState, context: dict = None) -> bool:
        """状态转换 - 增强版"""
        with self._lock:
            if self._is_valid_transition(self.current_state, new_state):
                old_state = self.current_state
                old_enter_time = self.state_enter_time

                # 执行转换前钩子
                self._execute_transition_hooks(old_state, new_state, context)

                # 更新状态
                self.current_state = new_state
                self.state_enter_time = time.time()

                # 计算上一状态持续时间
                duration = self.state_enter_time - old_enter_time

                # 记录状态历史（限制大小）
                if len(self.state_history) >= self.state_history.maxlen:
                    self.state_history.popleft()

                self.state_history.append({
                    'from_state': old_state,
                    'to_state': new_state,
                    'timestamp': time.time(),
                    'duration': duration,
                    'context': context or {}
                })

                # 触发状态监听器
                self._notify_state_listeners(new_state, context)

                logger.debug(f"状态转换: {old_state.value} -> {new_state.value} (持续{ duration:.1f}s)")
                return True
            else:
                logger.warning(f"无效的状态转换: {self.current_state.value} -> {new_state.value}")
                return False

    def _execute_transition_hooks(self, from_state: BusinessProcessState, to_state: BusinessProcessState, context: dict = None):
        """执行转换钩子"""
        hooks = self.transition_hooks.get((from_state, to_state), [])
        for hook in hooks:
            try:
                hook(from_state, to_state, context)
            except Exception as e:
                logger.error(f"转换钩子执行失败: {e}")

    def _notify_state_listeners(self, state: BusinessProcessState, context: dict = None):
        """通知状态监听器"""
        listeners = self.state_listeners.get(state, [])
        for listener in listeners:
            try:
                listener(state, context)
            except Exception as e:
                logger.error(f"状态监听器执行失败: {e}")

    def _is_valid_transition(self, from_state: BusinessProcessState, to_state: BusinessProcessState) -> bool:
        """检查状态转换是否有效 - 增强边界情况处理"""
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
                logger.warning(f"无效的状态转换: {from_state} -> {to_state}, 允许的转换: {allowed_transitions}")
                return False

            # 特殊状态转换规则验证
            if not self._validate_special_transition_rules(from_state, to_state):
                return False

            return True

        except Exception as e:
            logger.error(f"状态转换验证异常: {e}")
            return False

    def _validate_special_transition_rules(self, from_state: BusinessProcessState, to_state: BusinessProcessState) -> bool:
        """验证特殊状态转换规则"""
        try:
            # 防止从完成状态到错误状态的转换（除非有特殊情况）
            if from_state == BusinessProcessState.COMPLETED and to_state == BusinessProcessState.ERROR:
                logger.warning("不允许从完成状态直接转换到错误状态")
                return False

            # 防止从错误状态到完成状态的转换（必须先回到空闲状态）
            if from_state == BusinessProcessState.ERROR and to_state == BusinessProcessState.COMPLETED:
                logger.warning("不允许从错误状态直接转换到完成状态")
                return False

            # 验证暂停 / 恢复状态的转换逻辑
            if from_state == BusinessProcessState.PAUSED and to_state != BusinessProcessState.RESUMED and to_state != BusinessProcessState.ERROR:
                logger.warning("从暂停状态只能转换到恢复或错误状态")
                return False

            # 验证恢复状态只能转换到有效的业务状态
            if from_state == BusinessProcessState.RESUMED:
                valid_resume_states = [
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
                if to_state not in valid_resume_states:
                    logger.warning(f"从恢复状态只能转换到业务处理状态: {valid_resume_states}")
                    return False

            return True

        except Exception as e:
            logger.error(f"特殊状态转换规则验证异常: {e}")
            return False

    def get_current_state(self) -> BusinessProcessState:
        """获取当前状态"""
        return self.current_state

    def get_state_history(self) -> List[dict]:
        """获取状态历史"""
        return list(self.state_history)

    def reset(self):
        """重置状态机"""
        with self._lock:
            self.current_state = BusinessProcessState.IDLE
            self.state_history.clear()


class ProcessConfigManager:

    """流程配置管理器 - 优化版"""

    def __init__(self, config_dir: str = "config / processes"):

        self.config_dir = config_dir
        self.configs = {}
        self._load_configs()

    def _load_configs(self):
        """加载配置"""
        try:
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir, exist_ok=True)
                return

            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    config_path = os.path.join(self.config_dir, filename)
                    try:
                        with open(config_path, 'r', encoding='utf - 8') as f:
                            config_data = json.load(f)
                            config = ProcessConfig(**config_data)
                            self.configs[config.process_id] = config
                    except Exception as e:
                        logger.error(f"加载配置失败: {filename}, 错误: {e}")
        except Exception as e:
            logger.error(f"加载配置目录失败: {e}")

    def get_config(self, process_id: str) -> Optional[ProcessConfig]:
        """获取配置"""
        return self.configs.get(process_id)

    def save_config(self, config: ProcessConfig):
        """保存配置"""
        try:
            config_path = os.path.join(self.config_dir, f"{config.process_id}.json")
            with open(config_path, 'w', encoding='utf - 8') as f:
                json.dump(asdict(config), f, indent=2, ensure_ascii=False)
            self.configs[config.process_id] = config
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def list_configs(self) -> List[ProcessConfig]:
        """列出所有配置"""
        return list(self.configs.values())


class ProcessMonitor:

    """流程监控器 - 优化版"""

    def __init__(self):

        self.processes = {}
        self.metrics = {
            'total_processes': 0,
            'running_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'total_memory_usage': 0.0
        }
        self._lock = threading.RLock()
        self._cleanup_timer = None
        self._start_cleanup_timer()

    def _start_cleanup_timer(self):
        """启动清理定时器"""

        def cleanup_old_processes():

            while True:
                try:
                    time.sleep(300)  # 5分钟清理一次
                    self._cleanup_old_processes()
                except Exception as e:
                    logger.error(f"清理进程失败: {e}")

        self._cleanup_timer = threading.Thread(target=cleanup_old_processes, daemon=True)
        self._cleanup_timer.start()

    def _cleanup_old_processes(self):
        """清理旧进程"""
        with self._lock:
            current_time = time.time()
            to_remove = []

            for instance_id, process in self.processes.items():
                # 清理完成超过1小时的进程
                if (process.status in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]
                        and process.end_time and current_time - process.end_time > 3600):
                    to_remove.append(instance_id)

            for instance_id in to_remove:
                del self.processes[instance_id]
                logger.debug(f"清理旧进程: {instance_id}")

    def register_process(self, instance: ProcessInstance):
        """注册进程"""
        with self._lock:
            self.processes[instance.instance_id] = instance
            self.metrics['total_processes'] += 1
            # 只要不是完成或错误状态，都算作运行中
            if instance.status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                self.metrics['running_processes'] += 1

    def update_process(self, instance_id: str, status: BusinessProcessState, **kwargs):
        """更新进程状态"""
        with self._lock:
            if instance_id in self.processes:
                process = self.processes[instance_id]
                old_status = process.status
                process.status = status
                process.last_updated = time.time()

                # 更新统计
                if old_status == BusinessProcessState.IDLE and status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                    # 从空闲状态转换到运行状态，增加运行中计数
                    self.metrics['running_processes'] += 1
                elif old_status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR] and status in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                    # 从运行状态转换到完成或错误状态，减少运行中计数
                    self.metrics['running_processes'] -= 1

                if status == BusinessProcessState.COMPLETED:
                    self.metrics['completed_processes'] += 1
                    process.end_time = time.time()
                elif status == BusinessProcessState.ERROR:
                    self.metrics['failed_processes'] += 1
                    process.end_time = time.time()

                # 更新其他属性
                for key, value in kwargs.items():
                    if hasattr(process, key):
                        setattr(process, key, value)

    def get_process(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取进程"""
        return self.processes.get(instance_id)

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        with self._lock:
            # 计算总内存使用
            total_memory = sum(p.memory_usage for p in self.processes.values())
            self.metrics['total_memory_usage'] = total_memory
            return dict(self.metrics)

    def get_running_processes(self) -> List[ProcessInstance]:
        """获取运行中的进程"""
        with self._lock:
            return [
                process for process in self.processes.values()
                if process.status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]
            ]


class ProcessInstancePool:

    """流程实例池 - 新增"""

    def __init__(self, max_size: int = 100):

        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def get_instance(self, process_config: ProcessConfig) -> ProcessInstance:
        """获取实例"""
        with self._lock:
            if self.pool:
                instance = self.pool.popleft()
                # 重置实例
                instance.process_config = process_config
                instance.status = BusinessProcessState.IDLE
                instance.start_time = time.time()
                instance.end_time = None
                instance.current_step = ""
                instance.progress = 0.0
                instance.error_message = ""
                instance.context.clear()
                instance.memory_usage = 0.0
                instance.last_updated = time.time()
                return instance
            else:
                # 创建新实例
                return ProcessInstance(
                    instance_id=generate_id("process"),
                    process_config=process_config,
                    status=BusinessProcessState.IDLE,
                    start_time=time.time()
                )

    def return_instance(self, instance: ProcessInstance):
        """归还实例"""
        with self._lock:
            if len(self.pool) < self.max_size:
                self.pool.append(instance)


class BusinessProcessOrchestrator(BaseComponent):

    """业务流程编排器 - 优化版 3.0.0"""

    def __init__(self, config_dir: str = "config / processes", max_instances: int = 100):

        super().__init__("BusinessProcessOrchestrator", "3.0.0", "业务流程编排器核心组件")

        self.config_dir = config_dir
        self.max_instances = max_instances

        # 事件总线和状态机
        self._event_bus = None
        self._state_machine = None

        # 架构层
        self._layers = {}

        # 流程管理
        self._process_configs = {}
        self._process_instances = {}
        self._process_monitor = None
        self.config_manager = None
        self._instance_pool = None  # 将在initialize中初始化

        # 线程安全
        self._lock = threading.RLock()

        # 统计信息
        self._stats = {
            'total_processes': 0,
            'running_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'total_events': 0,
            'memory_usage': 0.0
        }

    def initialize(self) -> bool:
        """初始化业务流程编排器"""
        try:
            self._begin_initialization()
            self._initialize_core_components()
            self._initialize_monitoring_components()
            self._initialize_process_components()
            self._finalize_initialization()
            return True
        except Exception as e:
            self._handle_initialization_failure(e)
            raise

    def _begin_initialization(self) -> None:
        """开始初始化过程"""
        self.set_status(ComponentStatus.INITIALIZING)
        logger.info("开始初始化业务流程编排器")

    def _initialize_core_components(self) -> None:
        """初始化核心组件"""
        # 初始化事件总线
        self._event_bus = EventBus()
        logger.debug("事件总线初始化完成")

        # 初始化状态机
        self._state_machine = BusinessProcessStateMachine()
        logger.debug("状态机初始化完成")

        # 初始化架构层
        self._initialize_layers()
        logger.debug("架构层初始化完成")

    def _initialize_monitoring_components(self) -> None:
        """初始化监控组件"""
        # 初始化流程监控
        self._process_monitor = ProcessMonitor()
        logger.debug("流程监控初始化完成")

    def _initialize_process_components(self) -> None:
        """初始化流程相关组件"""
        # 初始化配置管理器
        self.config_manager = ProcessConfigManager(self.config_dir)
        logger.debug("配置管理器初始化完成")

        # 初始化流程实例池
        self._initialize_process_instance_pool()
        logger.debug("流程实例池初始化完成")

        # 加载流程配置
        self._load_process_configs()
        logger.debug("流程配置加载完成")

        # 设置事件处理器
        self._setup_event_handlers()
        logger.debug("事件处理器设置完成")

    def _initialize_process_instance_pool(self) -> None:
        """初始化流程实例池"""
        self._instance_pool = ProcessInstancePool(
            pool_name="BusinessProcessPool",
            config={
                'min_size': 5,
                'max_size': self.max_instances,
                'idle_timeout': 600,  # 10分钟
                'creation_timeout': 60  # 1分钟
            }
        )
        self._instance_pool.initialize()

    def _finalize_initialization(self) -> None:
        """完成初始化过程"""
        self._initialized = True
        self.set_status(ComponentStatus.INITIALIZED)
        self.set_health(ComponentHealth.HEALTHY)
        logger.info("业务流程编排器初始化完成")

    def _handle_initialization_failure(self, error: Exception) -> None:
        """处理初始化失败"""
        self.set_status(ComponentStatus.ERROR)
        self.set_health(ComponentHealth.UNHEALTHY)
        logger.error(f"业务流程编排器初始化失败: {error}")
        raise OrchestratorException(f"业务流程编排器初始化失败: {error}")

    def shutdown(self) -> bool:
        """关闭业务流程编排器"""
        try:
            self.set_status(ComponentStatus.STOPPING)

            # 停止所有运行中的流程
            running_processes = self.get_running_processes()
            for process in running_processes:
                try:
                    self.pause_process(process.instance_id)
                except Exception as e:
                    logger.warning(f"暂停流程失败: {process.instance_id}, 错误: {e}")

            # 关闭事件总线
            if self._event_bus:
                self._event_bus.shutdown()

            # 关闭流程实例池
            if self._instance_pool:
                self._instance_pool.shutdown()

            # 清理资源
            self._process_instances.clear()
            self._layers.clear()

            # 强制垃圾回收
            gc.collect()

            self.set_status(ComponentStatus.STOPPED)
            logger.info("业务流程编排器已关闭")
            return True

        except Exception as e:
            logger.error(f"关闭业务流程编排器失败: {e}")
            return False

    def _initialize_layers(self):
        """初始化架构层"""
        try:
            # 按照依赖顺序初始化各层
            self._layers = {}

            # 1. 初始化核心服务层
            self._layers['core_services'] = CoreServicesLayer()

            # 2. 初始化基础设施层（依赖核心服务层）
            self._layers['infrastructure'] = InfrastructureLayer(self._layers['core_services'])

            # 3. 初始化数据管理层（依赖基础设施层）
            self._layers['data_management'] = DataManagementLayer(self._layers['infrastructure'])

            # 4. 初始化特征处理层（依赖数据管理层）
            self._layers['feature_processing'] = FeatureProcessingLayer(
                self._layers['data_management'])

            # 5. 初始化模型推理层（依赖特征处理层）
            self._layers['model_inference'] = ModelInferenceLayer(
                self._layers['feature_processing'])

            # 6. 初始化策略决策层（依赖模型推理层）
            self._layers['strategy_decision'] = StrategyDecisionLayer(
                self._layers['model_inference'])

            # 7. 初始化风控合规层（依赖策略决策层）
            self._layers['risk_compliance'] = RiskComplianceLayer(self._layers['strategy_decision'])

            # 8. 初始化交易执行层（依赖风控合规层）
            self._layers['trading_execution'] = TradingExecutionLayer(
                self._layers['risk_compliance'])

            # 9. 初始化监控反馈层（依赖交易执行层）
            self._layers['monitoring_feedback'] = MonitoringFeedbackLayer(
                self._layers['trading_execution'])

            # 初始化各层
            for layer_name, layer in self._layers.items():
                if hasattr(layer, 'initialize'):
                    layer.initialize()

            logger.info("架构层初始化完成")

        except Exception as e:
            logger.error(f"初始化架构层失败: {e}")
            raise OrchestratorException(f"初始化架构层失败: {e}")

    def _load_process_configs(self):
        """加载流程配置"""
        try:
            if self.config_manager:
                self._process_configs = {
                    config.process_id: config
                    for config in self.config_manager.list_configs()
                }
        except Exception as e:
            logger.error(f"加载流程配置失败: {e}")

    def _setup_event_handlers(self):
        """设置事件处理器"""
        try:
            if self._event_bus:
                # 数据层事件
                self._event_bus.subscribe(EventType.DATA_COLLECTED, self._on_data_collected)
                self._event_bus.subscribe(EventType.DATA_QUALITY_CHECKED,
                                          self._on_data_quality_checked)

                # 特征层事件
                self._event_bus.subscribe(EventType.FEATURES_EXTRACTED, self._on_features_extracted)
                self._event_bus.subscribe(EventType.GPU_ACCELERATION_COMPLETED,
                                          self._on_gpu_acceleration_completed)

                # 模型层事件
                self._event_bus.subscribe(EventType.MODEL_PREDICTION_READY,
                                          self._on_model_prediction_ready)
                self._event_bus.subscribe(EventType.MODEL_ENSEMBLE_READY,
                                          self._on_model_ensemble_ready)

                # 策略层事件
                self._event_bus.subscribe(EventType.STRATEGY_DECISION_READY,
                                          self._on_strategy_decision_ready)
                self._event_bus.subscribe(EventType.SIGNALS_GENERATED, self._on_signals_generated)

                # 风控层事件
                self._event_bus.subscribe(EventType.RISK_CHECK_COMPLETED,
                                          self._on_risk_check_completed)
                self._event_bus.subscribe(EventType.COMPLIANCE_VERIFIED,
                                          self._on_compliance_verified)

                # 交易层事件
                self._event_bus.subscribe(EventType.ORDERS_GENERATED, self._on_orders_generated)
                self._event_bus.subscribe(EventType.EXECUTION_COMPLETED,
                                          self._on_execution_completed)

                # 监控层事件
                self._event_bus.subscribe(EventType.PERFORMANCE_ALERT, self._on_performance_alert)
                self._event_bus.subscribe(EventType.BUSINESS_ALERT, self._on_business_alert)

                # 流程控制事件
                self._event_bus.subscribe(EventType.PROCESS_STARTED, self._on_process_started)
                self._event_bus.subscribe(EventType.PROCESS_PAUSED, self._on_process_paused)
                self._event_bus.subscribe(EventType.PROCESS_RESUMED, self._on_process_resumed)
                self._event_bus.subscribe(EventType.PROCESS_COMPLETED, self._on_process_completed)
                self._event_bus.subscribe(EventType.PROCESS_ERROR, self._on_process_error)

        except Exception as e:
            logger.error(f"设置事件处理器失败: {e}")

    def start_trading_cycle(self, symbols: List[str], strategy_config: dict, process_id: str = None) -> str:
        """启动交易周期 - 重构版"""
        try:
            self._validate_initialization()
            process_config = self._prepare_process_config(symbols, strategy_config, process_id)
            instance = self._acquire_process_instance(symbols, strategy_config, process_config)
            self._register_and_initialize_process(instance, process_config)
            self._publish_process_start_event(instance, process_config, symbols, strategy_config)
            self._update_process_statistics()
            return instance.instance_id
        except Exception as e:
            logger.error(f"启动交易周期失败: {e}")
            raise OrchestratorException(f"启动交易周期失败: {e}")

    def _validate_initialization(self) -> None:
        """验证编排器初始化状态"""
        if not self._initialized:
            raise OrchestratorException("编排器未初始化")

    def _prepare_process_config(self, symbols: List[str], strategy_config: dict, process_id: str = None) -> ProcessConfig:
        """准备流程配置"""
        if process_id and process_id in self._process_configs:
            return self._process_configs[process_id]

        return ProcessConfig(
            process_id=process_id or generate_id("trading_cycle"),
            name="Trading Cycle",
            description="交易周期流程",
            steps=[
                {"name": "data_collection", "type": "data"},
                {"name": "feature_extraction", "type": "feature"},
                {"name": "model_prediction", "type": "model"},
                {"name": "strategy_decision", "type": "strategy"},
                {"name": "risk_check", "type": "risk"},
                {"name": "order_execution", "type": "trading"}
            ],
            parameters={
                "symbols": symbols,
                "strategy_config": strategy_config
            }
        )

    def _acquire_process_instance(self, symbols: List[str], strategy_config: dict, process_config: ProcessConfig) -> Any:
        """获取流程实例"""
        instance = self._instance_pool.acquire_instance(
            process_type="trading_cycle",
            context_data={
                'symbols': symbols,
                'strategy_config': strategy_config,
                'process_config': process_config
            },
            priority=1
        )

        if not instance:
            raise OrchestratorException("无法获取流程实例，池已满")

        return instance

    def _register_and_initialize_process(self, instance: Any, process_config: ProcessConfig) -> None:
        """注册并初始化流程实例"""
        # 注册流程实例
        self._process_monitor.register_process(instance)
        self._process_instances[instance.instance_id] = instance

        # 更新状态
        self._process_monitor.update_process(
            instance.instance_id,
            BusinessProcessState.DATA_COLLECTING,
            current_step="data_collection",
            progress=0.0
        )

    def _publish_process_start_event(self, instance: Any, process_config: ProcessConfig,
                                     symbols: List[str], strategy_config: dict) -> None:
        """发布流程开始事件"""
        if self._event_bus:
            self._event_bus.publish(EventType.PROCESS_STARTED, {
                'instance_id': instance.instance_id,
                'process_id': process_config.process_id,
                'symbols': symbols,
                'strategy_config': strategy_config
            })

    def _update_process_statistics(self) -> None:
        """更新流程统计信息"""
        self._stats['total_processes'] += 1
        self._stats['running_processes'] += 1

    def validate_process_config(self, config: ProcessConfig) -> List[str]:
        """验证流程配置"""
        errors = []

        if not config.process_id:
            errors.append("流程ID不能为空")

        if not config.name:
            errors.append("流程名称不能为空")

        if not config.steps:
            errors.append("流程步骤不能为空")

        # 验证步骤依赖关系
        step_names = [step.get('name') for step in config.steps if isinstance(step, dict)]
        for step in config.steps:
            if isinstance(step, dict):
                step_name = step.get('name')
                dependencies = step.get('dependencies', [])
                for dep in dependencies:
                    if dep not in step_names:
                        errors.append(f"步骤 '{step_name}' 的依赖 '{dep}' 不存在")

        # 验证参数
        if config.parameters:
            required_params = config.parameters.get('required', [])
            for param in required_params:
                if param not in config.parameters:
                    errors.append(f"缺少必需参数: {param}")

        return errors

    def update_process_config(self, process_id: str, updates: Dict[str, Any]) -> bool:
        """动态更新流程配置"""
        try:
            with self._lock:
                if process_id not in self._process_configs:
                    logger.warning(f"流程配置不存在: {process_id}")
                    return False

                config = self._process_configs[process_id]

                # 验证更新
                for key, value in updates.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        logger.warning(f"无效的配置属性: {key}")

                # 重新验证配置
                errors = self.validate_process_config(config)
                if errors:
                    logger.error(f"配置更新后验证失败: {errors}")
                    return False

                # 保存配置
                if self.config_manager:
                    self.config_manager.save_config(config)

                logger.info(f"流程配置已更新: {process_id}")
                return True

        except Exception as e:
            logger.error(f"更新流程配置失败: {e}")
            return False

    def get_process_metrics(self, instance_id: str = None) -> Dict[str, Any]:
        """获取流程监控指标 - 重构版"""
        try:
            metrics = self._build_basic_metrics()

            if instance_id:
                metrics['instance'] = self._get_specific_instance_metrics(instance_id)
            else:
                metrics['running_instances'] = self._get_running_instances_summary()

            metrics['pool'] = self._get_pool_statistics()
            return metrics

        except Exception as e:
            logger.error(f"获取流程指标失败: {e}")
            return {'error': str(e)}

    def _build_basic_metrics(self) -> Dict[str, Any]:
        """构建基础指标"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall': dict(self._stats)
        }

    def _get_specific_instance_metrics(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取特定实例的指标"""
        if instance_id in self._process_instances:
            instance = self._process_instances[instance_id]
            return {
                'instance_id': instance.instance_id,
                'status': instance.status.value,
                'progress': instance.progress,
                'start_time': instance.start_time,
                'end_time': instance.end_time,
                'current_step': getattr(instance, 'current_step', None),
                'memory_usage': getattr(instance, 'memory_usage', 0.0),
                'cpu_usage': getattr(instance, 'cpu_usage', 0.0)
            }
        return None

    def _get_running_instances_summary(self) -> List[Dict[str, Any]]:
        """获取所有运行中实例的摘要"""
        running_instances = []
        for inst_id, instance in self._process_instances.items():
            if instance.status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                running_instances.append({
                    'instance_id': inst_id,
                    'status': instance.status.value,
                    'progress': instance.progress,
                    'current_step': getattr(instance, 'current_step', None),
                    'start_time': instance.start_time
                })
        return running_instances

    def _get_pool_statistics(self) -> Optional[Dict[str, Any]]:
        """获取池统计信息"""
        if self._instance_pool:
            pool_stats = self._instance_pool.get_stats()
            return {
                'total_instances': pool_stats.total_instances,
                'active_instances': pool_stats.active_instances,
                'idle_instances': pool_stats.idle_instances,
                'pool_utilization': pool_stats.resource_utilization,
                'pool_hit_rate': pool_stats.pool_hit_rate
            }
        return None

    def complete_process(self, instance_id: str, final_status: BusinessProcessState = BusinessProcessState.COMPLETED) -> bool:
        """完成流程并释放实例 - 重构版"""
        try:
            with self._lock:
                if not self._validate_process_instance(instance_id):
                    return False

                instance = self._process_instances[instance_id]
                self._update_instance_final_status(instance, final_status)
                self._update_process_statistics(final_status)
                self._update_process_monitor(instance_id, instance, final_status)
                self._publish_completion_event(instance_id, instance, final_status)
                self._release_process_instance(instance_id)

                logger.info(f"流程完成: {instance_id} 状态={final_status.value}")
                return True

        except Exception as e:
            logger.error(f"完成流程失败 {instance_id}: {e}")
            return False

    def _validate_process_instance(self, instance_id: str) -> bool:
        """验证流程实例是否存在"""
        if instance_id not in self._process_instances:
            logger.warning(f"流程实例不存在: {instance_id}")
            return False
        return True

    def _update_instance_final_status(self, instance: Any, final_status: BusinessProcessState) -> None:
        """更新实例的最终状态"""
        instance.status = final_status
        instance.end_time = time.time()
        instance.progress = 1.0

    def _update_process_statistics(self, final_status: BusinessProcessState) -> None:
        """更新流程统计信息"""
        if final_status == BusinessProcessState.COMPLETED:
            self._stats['completed_processes'] += 1
            self._stats['running_processes'] -= 1
        elif final_status == BusinessProcessState.ERROR:
            self._stats['failed_processes'] += 1
            self._stats['running_processes'] -= 1

    def _update_process_monitor(self, instance_id: str, instance: Any, final_status: BusinessProcessState) -> None:
        """更新流程监控器"""
        self._process_monitor.update_process(
            instance_id, final_status, instance.current_step, instance.progress)

    def _publish_completion_event(self, instance_id: str, instance: Any, final_status: BusinessProcessState) -> None:
        """发布流程完成事件"""
        if not self._event_bus:
            return

        duration = instance.end_time - instance.start_time
        event_data = {
            'instance_id': instance_id,
            'process_id': instance.process_config.process_id,
            'duration': duration
        }

        if final_status == BusinessProcessState.COMPLETED:
            event_data['final_status'] = final_status.value
            self._event_bus.publish(EventType.PROCESS_COMPLETED, event_data)
        else:
            event_data['error_message'] = getattr(instance, 'error_message', None)
            self._event_bus.publish(EventType.PROCESS_ERROR, event_data)

    def _release_process_instance(self, instance_id: str):
        """释放流程实例回池中"""
        try:
            if not self._instance_pool:
                return

            # 从池中释放实例
            success = self._instance_pool.release_instance(instance_id)
            if success:
                logger.debug(f"流程实例已释放回池中: {instance_id}")
            else:
                logger.warning(f"释放流程实例失败: {instance_id}")

        except Exception as e:
            logger.error(f"释放流程实例异常 {instance_id}: {e}")

    def pause_process(self, instance_id: str) -> bool:
        """暂停流程 - 优化版"""
        try:
            if instance_id not in self._process_instances:
                raise OrchestratorException(f"流程实例不存在: {instance_id}")

            instance = self._process_instances[instance_id]

            # 检查是否可以暂停
            if instance.status in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                raise OrchestratorException(f"流程已完成或出错，无法暂停: {instance_id}")

            # 更新状态
            self._process_monitor.update_process(
                instance_id,
                BusinessProcessState.PAUSED,
                current_step=instance.current_step
            )

            # 发布暂停事件
            if self._event_bus:
                self._event_bus.publish(EventType.PROCESS_PAUSED, {
                    'instance_id': instance_id,
                    'current_step': instance.current_step
                })

            logger.info(f"暂停流程: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"暂停流程失败: {e}")
            return False

    def resume_process(self, instance_id: str) -> bool:
        """恢复流程 - 优化版"""
        try:
            if instance_id not in self._process_instances:
                raise OrchestratorException(f"流程实例不存在: {instance_id}")

            instance = self._process_instances[instance_id]

            # 检查是否可以恢复
            if instance.status != BusinessProcessState.PAUSED:
                raise OrchestratorException(f"流程未暂停，无法恢复: {instance_id}")

            # 更新状态 - 恢复到数据采集状态
            self._process_monitor.update_process(
                instance_id,
                BusinessProcessState.DATA_COLLECTING,
                current_step="data_collection"
            )

            # 同时更新实例状态
            instance.current_state = BusinessProcessState.DATA_COLLECTING
            instance.current_step = "data_collection"

            # 发布恢复事件
            if self._event_bus:
                self._event_bus.publish(EventType.PROCESS_RESUMED, {
                    'instance_id': instance_id,
                    'current_step': instance.current_step
                })

            logger.info(f"恢复流程: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"恢复流程失败: {e}")
            return False

    def _on_data_collected(self, event):
        """数据收集完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.FEATURE_EXTRACTING,
                    current_step="feature_extraction",
                    progress=30.0
                )
        except Exception as e:
            logger.error(f"处理数据收集完成事件失败: {e}")

    def _on_data_quality_checked(self, event):
        """数据质量检查完成事件处理"""
        try:
            instance_id = event.data.get('instance_id') if event.data else None
            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.FEATURE_EXTRACTING,
                    current_step="feature_extraction",
                    progress=30.0
                )
        except Exception as e:
            logger.error(f"处理数据质量检查完成事件失败: {e}")

    def _on_features_extracted(self, event):
        """特征提取完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.MODEL_PREDICTING,
                    current_step="model_prediction",
                    progress=50.0
                )
        except Exception as e:
            logger.error(f"处理特征提取完成事件失败: {e}")

    def _on_gpu_acceleration_completed(self, event):
        """GPU加速完成事件处理"""
        try:
            instance_id = event.data.get('instance_id') if event.data else None
            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.MODEL_PREDICTING,
                    current_step="model_prediction",
                    progress=50.0
                )
        except Exception as e:
            logger.error(f"处理GPU加速完成事件失败: {e}")

    def _on_model_prediction_ready(self, event):
        """模型预测完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.STRATEGY_DECIDING,
                    current_step="strategy_decision",
                    progress=60.0
                )
        except Exception as e:
            logger.error(f"处理模型预测完成事件失败: {e}")

    def _on_model_ensemble_ready(self, event):
        """模型集成完成事件处理"""
        try:
            instance_id = event.get('data', {}).get('instance_id')
            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.STRATEGY_DECIDING,
                    current_step="strategy_decision",
                    progress=60.0
                )
        except Exception as e:
            logger.error(f"处理模型集成完成事件失败: {e}")

    def _on_strategy_decision_ready(self, event):
        """策略决策完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.SIGNAL_GENERATING,
                    current_step="signal_generation",
                    progress=70.0
                )
        except Exception as e:
            logger.error(f"处理策略决策完成事件失败: {e}")

    def _on_signals_generated(self, event):
        """信号生成完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.RISK_CHECKING,
                    current_step="risk_check",
                    progress=80.0
                )
        except Exception as e:
            logger.error(f"处理信号生成完成事件失败: {e}")

    def _on_risk_check_completed(self, event):
        """风险检查完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.ORDER_GENERATING,
                    current_step="order_generation",
                    progress=90.0
                )
        except Exception as e:
            logger.error(f"处理风险检查完成事件失败: {e}")

    def _on_compliance_verified(self, event):
        """合规验证完成事件处理"""
        try:
            instance_id = event.data.get('instance_id') if event.data else None
            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.ORDER_GENERATING,
                    current_step="order_generation",
                    progress=90.0
                )
        except Exception as e:
            logger.error(f"处理合规验证完成事件失败: {e}")

    def _on_orders_generated(self, event):
        """订单生成完成事件处理"""
        try:
            # 从事件数据中获取实例ID，如果没有则从当前运行的流程中获取
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.ORDER_EXECUTING,
                    current_step="order_execution",
                    progress=95.0
                )
        except Exception as e:
            logger.error(f"处理订单生成完成事件失败: {e}")

    def _on_execution_completed(self, event):
        """执行完成事件处理"""
        try:
            instance_id = event.data.get('instance_id') if event.data else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                instance = self._process_instances[instance_id]
                self._process_monitor.update_process(
                    instance_id,
                    BusinessProcessState.COMPLETED,
                    current_step="completed",
                    progress=100.0
                )

                # 归还实例到池中
                self._instance_pool.return_instance(instance)

                # 更新统计
                self._stats['running_processes'] -= 1
                self._stats['completed_processes'] += 1
        except Exception as e:
            logger.error(f"处理执行完成事件失败: {e}")

    def _on_performance_alert(self, event):
        """性能告警事件处理"""
        try:
            logger.warning(f"性能告警: {event.data if hasattr(event, 'data') else str(event)}")
        except Exception as e:
            logger.error(f"处理性能告警事件失败: {e}")

    def _on_business_alert(self, event):
        """业务告警事件处理"""
        try:
            logger.warning(f"业务告警: {event.data if hasattr(event, 'data') else str(event)}")
        except Exception as e:
            logger.error(f"处理业务告警事件失败: {e}")

    def _on_process_started(self, event):
        """流程开始事件处理"""
        try:
            data = event.data if hasattr(event, 'data') else str(event)
            logger.info(f"流程开始: {data}")
        except Exception as e:
            logger.error(f"处理流程开始事件失败: {e}")

    def _on_process_paused(self, event):
        """流程暂停事件处理"""
        try:
            data = event.data if hasattr(event, 'data') else str(event)
            logger.info(f"流程暂停: {data}")
        except Exception as e:
            logger.error(f"处理流程暂停事件失败: {e}")

    def _on_process_resumed(self, event):
        """流程恢复事件处理"""
        try:
            data = event.data if hasattr(event, 'data') else str(event)
            logger.info(f"流程恢复: {data}")
        except Exception as e:
            logger.error(f"处理流程恢复事件失败: {e}")

    def _on_process_completed(self, event):
        """流程完成事件处理"""
        try:
            data = event.data if hasattr(event, 'data') else str(event)
            logger.info(f"流程完成: {data}")
        except Exception as e:
            logger.error(f"处理流程完成事件失败: {e}")

    def _on_process_error(self, event):
        """流程错误事件处理"""
        try:
            data = event.data if hasattr(event, 'data') else str(event)

            instance_id = data.get('instance_id') if isinstance(data, dict) else None
            if not instance_id:
                # 如果没有指定实例ID，使用第一个运行的流程
                running_processes = self.get_running_processes()
                if running_processes:
                    instance_id = running_processes[0].instance_id

            if instance_id and instance_id in self._process_instances:
                instance = self._process_instances[instance_id]
                error_message = data.get('error', 'Unknown error') if isinstance(
                    data, dict) else str(data)

                if self._process_monitor:
                    self._process_monitor.update_process(
                        instance_id,
                        BusinessProcessState.ERROR,
                        error_message=error_message
                    )

                # 同时更新实例状态
                instance.status = BusinessProcessState.ERROR
                instance.error_message = error_message

                # 归还实例到池中
                self._instance_pool.return_instance(instance)

                # 更新统计
                self._stats['running_processes'] -= 1
                self._stats['failed_processes'] += 1

            logger.error(f"流程错误: {data}")
        except Exception as e:
            logger.error(f"处理流程错误事件失败: {e}")

    def get_current_state(self) -> BusinessProcessState:
        """获取当前状态"""
        if self._state_machine:
            return self._state_machine.get_current_state()
        return BusinessProcessState.IDLE

    def get_event_history(self) -> List[dict]:
        """获取事件历史"""
        if self._event_bus:
            return self._event_bus.get_event_history()
        return []

    def get_state_history(self) -> List[dict]:
        """获取状态历史"""
        if self._state_machine:
            return self._state_machine.get_state_history()
        return []

    def get_process_metrics(self) -> Dict[str, Any]:
        """获取流程指标 - 优化版"""
        try:
            metrics = self._process_monitor.get_metrics() if self._process_monitor else {}

            # 计算内存使用
            total_memory = sum(
                instance.memory_usage
                for instance in self._process_instances.values()
            )

            # 计算平均执行时间
            completed_processes = [p for p in self._process_instances.values(
            ) if p.status == BusinessProcessState.COMPLETED]
            if completed_processes:
                total_time = sum(
                    p.end_time - p.start_time for p in completed_processes if p.end_time)
                average_execution_time = total_time / \
                    len(completed_processes) if total_time > 0 else 0
            else:
                average_execution_time = 0

            # 确保包含测试期望的指标名称
            metrics.update({
                'total_memory_usage_mb': total_memory,
                'instance_pool_size': len(self._instance_pool.pool),
                'max_instances': self.max_instances,
                'memory_efficiency': (total_memory / self.max_instances) if self.max_instances > 0 else 0,
                'error_processes': metrics.get('failed_processes', 0),  # 兼容测试期望
                'total_processes': metrics.get('total_processes', 0),
                'running_processes': metrics.get('running_processes', 0),
                'completed_processes': metrics.get('completed_processes', 0),
                'average_execution_time': average_execution_time
            })

            return metrics
        except Exception as e:
            logger.error(f"获取流程指标失败: {e}")
            return {}

    def get_process(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取流程实例 - 新增方法"""
        try:
            # 首先从本地实例字典中查找
            if instance_id in self._process_instances:
                return self._process_instances[instance_id]

            # 从流程监控器中查找
            if self._process_monitor:
                return self._process_monitor.get_process(instance_id)

            return None
        except Exception as e:
            logger.error(f"获取流程实例失败: {instance_id}, 错误: {e}")
            return None

    def get_running_processes(self) -> List[ProcessInstance]:
        """获取运行中的流程 - 优化版"""
        try:
            if self._process_monitor:
                return self._process_monitor.get_running_processes()
            return []
        except Exception as e:
            logger.error(f"获取运行中的流程失败: {e}")
            return []

    def reset(self):
        """重置编排器 - 优化版"""
        try:
            # 停止所有流程
            running_processes = self.get_running_processes()
            for process in running_processes:
                try:
                    self.pause_process(process.instance_id)
                except Exception as e:
                    logger.warning(f"暂停流程失败: {process.instance_id}, 错误: {e}")

            # 清理资源
            self._process_instances.clear()
            self._instance_pool.pool.clear()

            # 重置状态机
            if self._state_machine:
                self._state_machine.reset()

            # 重置统计
            self._stats = {
                'total_processes': 0,
                'running_processes': 0,
                'completed_processes': 0,
                'failed_processes': 0,
                'total_events': 0,
                'memory_usage': 0.0
            }

            # 强制垃圾回收
            gc.collect()

            logger.info("编排器已重置")

        except Exception as e:
            logger.error(f"重置编排器失败: {e}")

    def get_memory_usage(self) -> float:
        """获取内存使用情况"""
        try:
            total_memory = sum(
                instance.memory_usage
                for instance in self._process_instances.values()
            )
            return total_memory
        except Exception as e:
            logger.error(f"获取内存使用失败: {e}")
            return 0.0

    def optimize_memory(self):
        """内存优化"""
        try:
            # 清理完成超过1小时的进程
            current_time = time.time()
            to_remove = []

            for instance_id, instance in self._process_instances.items():
                if (instance.status in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]
                        and instance.end_time and current_time - instance.end_time > 3600):
                    to_remove.append(instance_id)

            for instance_id in to_remove:
                instance = self._process_instances.pop(instance_id)
                self._instance_pool.return_instance(instance)
                logger.debug(f"内存优化：清理旧进程: {instance_id}")

            # 强制垃圾回收
            gc.collect()

            logger.info(f"内存优化完成，清理了 {len(to_remove)} 个旧进程")

        except Exception as e:
            logger.error(f"内存优化失败: {e}")

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查 - 重构版"""
        try:
            health_status = self._build_basic_health_status()
            self._check_component_initialization(health_status)
            self._check_pool_utilization(health_status)
            self._check_failure_rate(health_status)
            return health_status
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return self._build_error_health_status(str(e))

    def _build_basic_health_status(self) -> Dict[str, Any]:
        """构建基础健康状态"""
        return {
            'component_name': self.name,
            'status': 'healthy',
            'initialized': self._initialized,
            'total_processes': self._stats.get('total_processes', 0),
            'running_processes': self._stats.get('running_processes', 0),
            'completed_processes': self._stats.get('completed_processes', 0),
            'failed_processes': self._stats.get('failed_processes', 0),
            'memory_usage_mb': self._stats.get('memory_usage', 0.0),
            'last_check': datetime.now().isoformat()
        }

    def _check_component_initialization(self, health_status: Dict[str, Any]) -> None:
        """检查关键组件初始化状态"""
        component_checks = [
            ('_event_bus', 'event_bus_not_initialized'),
            ('_state_machine', 'state_machine_not_initialized'),
            ('_process_monitor', 'process_monitor_not_initialized'),
            ('_instance_pool', 'instance_pool_not_initialized')
        ]

        for attr_name, issue in component_checks:
            if not getattr(self, attr_name, None):
                health_status['status'] = 'warning'
                health_status.setdefault('issues', []).append(issue)

    def _check_pool_utilization(self, health_status: Dict[str, Any]) -> None:
        """检查流程池利用率"""
        if self._instance_pool:
            pool_stats = self._instance_pool.get_stats()
            pool_utilization = pool_stats.resource_utilization
            if pool_utilization > 0.9:
                health_status['status'] = 'warning'
                health_status.setdefault('issues', []).append('pool_high_utilization')

    def _check_failure_rate(self, health_status: Dict[str, Any]) -> None:
        """检查流程失败率"""
        total_processes = health_status['total_processes']
        if total_processes > 0:
            failure_rate = health_status['failed_processes'] / total_processes
            if failure_rate > 0.1:  # 超过10%的失败率
                health_status['status'] = 'warning'
                health_status.setdefault('issues', []).append('high_failure_rate')

    def _build_error_health_status(self, error_message: str) -> Dict[str, Any]:
        """构建错误健康状态"""
        return {
            'component_name': self.name,
            'status': 'error',
            'error': error_message,
            'last_check': datetime.now().isoformat()
        }

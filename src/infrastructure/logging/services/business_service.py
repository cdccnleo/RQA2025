"""
business_service 模块

提供 business_service 相关功能和接口。
"""

import logging

# 合理跨层级导入：基础设施层日志组件需要核心业务逻辑进行日志分类
# 跨层级导入：层组件
import time

from .base_service import BaseService
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol
    class Event(Protocol):
        """事件协议"""
        pass
    class EventBus(Protocol):
        """事件总线协议"""
        pass
    class ServiceContainer(Protocol):
        """服务容器协议"""
        pass
else:
    # 运行时使用Any作为占位符
    Event = Any
    EventBus = Any
    ServiceContainer = Any
"""
业务服务实现
负责业务流程编排和协调各个服务之间的交互
"""

logger = logging.getLogger(__name__)


class _NoopEventBus:
    """兼容性事件总线占位实现"""

    def __init__(self):  # pragma: no cover - 简化实现
        pass

    def subscribe(self, *_, **__):  # pragma: no cover
        return None

    def unsubscribe(self, *_, **__):  # pragma: no cover
        return None

    def publish(self, *_, **__):  # pragma: no cover
        return None


class _NoopContainer:
    """兼容性依赖容器占位实现"""

    def has(self, *_args, **_kwargs) -> bool:
        return False

    def get(self, *_args, **_kwargs):  # pragma: no cover
        raise KeyError("Service not registered in test container")


class BusinessService(BaseService):

    """
    business_service - 日志系统

    职责说明：
    负责系统日志记录、日志格式化、日志存储和日志分析

    核心职责：
    - 日志记录和格式化
    - 日志级别管理
    - 日志存储和轮转
    - 日志分析和监控
    - 日志搜索和过滤
    - 日志性能优化

    相关接口：
    - ILoggingComponent
    - ILogger
    - ILogHandler

    业务服务 - 业务流程编排器

    负责协调各个服务之间的交互，实现完整的业务流程：
    - 数据获取 -> 特征提取 -> 模型预测 -> 信号生成 -> 风控检查 -> 交易执行
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        container: Optional[Any] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "BusinessService")
        self.event_bus = event_bus or _NoopEventBus()
        self.container = container or _NoopContainer()
        self.logger = logging.getLogger(__name__)

        # 业务流程配置
        self.workflow_configs = {}
        self.workflows = self.workflow_configs  # 兼容性别名
        self.active_workflows = {}
        self.workflow_metrics = {}

        # 服务依赖映射
        self.service_dependencies = {
            "data_service": ["market_data", "historical_data"],
            "feature_service": ["feature_extraction", "feature_validation"],
            "model_service": ["model_prediction", "ab_testing"],
            "trading_service": ["strategy_execution", "risk_control"],
            "validation_service": ["data_validation", "quality_check"]
        }

        # 订阅业务事件
        self._subscribe_to_events()

        # 初始化默认工作流
        self._start_default_workflows()

    def _subscribe_to_events(self):
        """订阅相关业务事件"""
        self.event_bus.subscribe("DATA_READY", self._on_data_ready)
        self.event_bus.subscribe("FEATURE_EXTRACTED", self._on_feature_extracted)
        self.event_bus.subscribe("MODEL_PREDICTED", self._on_model_predicted)
        self.event_bus.subscribe("SIGNAL_GENERATED", self._on_signal_generated)
        self.event_bus.subscribe("RISK_CHECKED", self._on_risk_checked)
        self.event_bus.subscribe("EXECUTION_COMPLETED", self._on_execution_completed)
        self.event_bus.subscribe("VALIDATION_COMPLETED", self._on_validation_completed)

    def _start(self) -> bool:
        """启动业务服务"""
        try:
            self.logger.info("启动业务服务")

            # 初始化服务依赖检查
            for service_name in self.service_dependencies.keys():
                if not self.container.has(service_name):
                    self.logger.warning(f"服务依赖缺失: {service_name}")

            # 启动默认业务流程
            self._start_default_workflows()

            return True
        except Exception as e:
            self.logger.error(f"业务服务启动失败: {e}")
            return False

    def _stop(self) -> bool:
        """停止业务服务"""
        try:
            self.logger.info("停止业务服务")

            # 停止所有活跃的工作流
            for workflow_id in list(self.active_workflows.keys()):
                self.stop_workflow(workflow_id)

            # 取消事件订阅
            self.event_bus.unsubscribe("DATA_READY", self._on_data_ready)
            self.event_bus.unsubscribe("FEATURE_EXTRACTED", self._on_feature_extracted)
            self.event_bus.unsubscribe("MODEL_PREDICTED", self._on_model_predicted)
            self.event_bus.unsubscribe("SIGNAL_GENERATED", self._on_signal_generated)
            self.event_bus.unsubscribe("RISK_CHECKED", self._on_risk_checked)
            self.event_bus.unsubscribe("EXECUTION_COMPLETED", self._on_execution_completed)
            self.event_bus.unsubscribe("VALIDATION_COMPLETED",
                                       self._on_validation_completed)

            return True
        except Exception as e:
            self.logger.error(f"业务服务停止失败: {e}")
            return False

    def _health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "active_workflows": len(self.active_workflows),
            "workflow_configs": len(self.workflow_configs),
            "workflows": len(self.workflow_configs),  # 兼容性字段
            "service_dependencies": len(self.service_dependencies),
            "service_type": "business_service",
            "timestamp": time.time()
        }

    def create_workflow(self, workflow_id: str, config: Dict[str, Any]) -> bool:
        """
        创建业务流程

        Args:
            workflow_id: 工作流ID
            config: 工作流配置

        Returns:
            bool: 创建是否成功
        """
        try:
            self.logger.info(f"创建业务流程: {workflow_id}")

            # 检查工作流ID是否已存在
            if workflow_id in self.workflow_configs:
                self.logger.error(f"工作流ID已存在: {workflow_id}")
                return False

            # 验证配置
            if not self._validate_workflow_config(config):
                self.logger.error(f"工作流配置无效: {workflow_id}")
                return False

            # 保存配置
            self.workflow_configs[workflow_id] = config

            # 初始化指标
            self.workflow_metrics[workflow_id] = {
                "start_time": None,
                "end_time": None,
                "steps_completed": 0,
                "total_steps": len(config.get("steps", [])),
                "status": "created",
                "errors": []
            }

            self.logger.info(f"业务流程创建成功: {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"创建业务流程失败: {workflow_id}, 错误: {e}")
            return False

    def start_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None) -> bool:
        """
        启动业务流程

        Args:
            workflow_id: 工作流ID
            input_data: 输入数据

        Returns:
            bool: 启动是否成功
        """
        try:
            if workflow_id not in self.workflow_configs:
                self.logger.error(f"工作流不存在: {workflow_id}")
                return False

            self.logger.info(f"启动业务流程: {workflow_id}")

            # 创建工作流实例
            workflow_instance = {
                "id": workflow_id,
                "config": self.workflow_configs[workflow_id],
                "input_data": input_data or {},
                "current_step": 0,
                "results": {},
                "start_time": time.time(),
                "status": "running"
            }

            self.active_workflows[workflow_id] = workflow_instance

            # 更新指标
            self.workflow_metrics[workflow_id]["start_time"] = time.time()
            self.workflow_metrics[workflow_id]["status"] = "running"

            # 执行第一个步骤
            self._execute_workflow_step(workflow_id)

            return True

        except Exception as e:
            self.logger.error(f"启动业务流程失败: {workflow_id}, 错误: {e}")
            return False

    def stop_workflow(self, workflow_id: str) -> bool:
        """
        停止业务流程

        Args:
            workflow_id: 工作流ID

        Returns:
            bool: 停止是否成功
        """
        try:
            if workflow_id not in self.active_workflows:
                self.logger.warning(f"工作流不存在或已停止: {workflow_id}")
                return False

            self.logger.info(f"停止业务流程: {workflow_id}")

            # 更新状态
            self.active_workflows[workflow_id]["status"] = "stopped"
            self.workflow_metrics[workflow_id]["status"] = "stopped"
            self.workflow_metrics[workflow_id]["end_time"] = time.time()

            # 移除活跃工作流
            del self.active_workflows[workflow_id]

            return True

        except Exception as e:
            self.logger.error(f"停止业务流程失败: {workflow_id}, 错误: {e}")
            return False

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        获取工作流状态

        Args:
            workflow_id: 工作流ID

        Returns:
            Dict[str, Any]: 工作流状态信息
        """
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "id": workflow_id,
                "status": workflow["status"],
                "current_step": workflow["current_step"],
                "total_steps": len(workflow["config"].get("steps", [])),
                "start_time": workflow["start_time"],
                "results": workflow["results"]
            }
        elif workflow_id in self.workflow_metrics:
            metrics = self.workflow_metrics[workflow_id]
            return {
                "id": workflow_id,
                "status": metrics["status"],
                "steps_completed": metrics["steps_completed"],
                "total_steps": metrics["total_steps"],
                "start_time": metrics["start_time"],
                "end_time": metrics["end_time"]
            }
        else:
            return {"error": "工作流不存在"}

    def list_workflows(self) -> Dict[str, Any]:
        """
        列出所有工作流

        Returns:
            Dict[str, Any]: 工作流列表
        """
        workflows_info = []
        for workflow_id, config in self.workflow_configs.items():
            workflow_info = {
                "id": workflow_id,
                "name": config.get("name", workflow_id),
                "status": "active" if workflow_id in self.active_workflows else "inactive",
                "steps": len(config.get("steps", []))
            }
            workflows_info.append(workflow_info)

        return {
            "workflows": workflows_info,
            "active_count": len(self.active_workflows),
            "total_count": len(self.workflow_configs)
        }

    def _start_default_workflows(self):
        """启动默认业务流程"""
        # 创建标准交易流程
        standard_trading_workflow = {
            "name": "标准交易流程",
            "description": "数据获取 -> 特征提取 -> 模型预测 -> 信号生成 -> 风控检查 -> 交易执行",
            "steps": [
                {
                    "name": "data_validation",
                    "service": "validation_service",
                    "method": "validate_realtime_data",
                    "required": True
                },
                {
                    "name": "feature_extraction",
                    "service": "feature_service",
                    "method": "extract_features",
                    "required": True
                },
                {
                    "name": "model_prediction",
                    "service": "model_service",
                    "method": "predict",
                    "required": True
                },
                {
                    "name": "signal_generation",
                    "service": "trading_service",
                    "method": "generate_signals",
                    "required": True
                },
                {
                    "name": "risk_check",
                    "service": "trading_service",
                    "method": "check_risk",
                    "required": True
                },
                {
                    "name": "execution",
                    "service": "trading_service",
                    "method": "execute_orders",
                    "required": False
                }
            ]
        }

        self.create_workflow("standard_trading", standard_trading_workflow)

    def _validate_workflow_config(self, config: Dict[str, Any]) -> bool:
        """
        验证工作流配置

        Args:
            config: 工作流配置字典

        Returns:
            是否验证通过
        """
        if not self._validate_required_fields(config):
            return False

        if not self._validate_workflow_steps(config.get("steps", [])):
            return False

        return True

    def _validate_required_fields(self, config: Dict[str, Any]) -> bool:
        """
        验证必需字段

        Args:
            config: 配置字典

        Returns:
            是否包含所有必需字段
        """
        required_fields = ["name", "steps"]

        for field in required_fields:
            if field not in config:
                self.logger.error(f"工作流配置缺少必需字段: {field}")
                return False

        return True

    def _validate_workflow_steps(self, steps: List[Dict[str, Any]]) -> bool:
        """
        验证工作流步骤

        Args:
            steps: 步骤配置列表

        Returns:
            是否所有步骤都有效
        """
        if not steps:
            return False  # 空步骤列表无效

        for step in steps:
            if not self._is_step_valid(step):
                self.logger.error(f"工作流步骤配置无效: {step}")
                return False

        return True

    def _is_step_valid(self, step: Dict[str, Any]) -> bool:
        """
        检查步骤是否有效

        Args:
            step: 步骤配置字典

        Returns:
            步骤是否有效
        """
        required_step_fields = ["name", "service", "method"]

        for field in required_step_fields:
            if field not in step:
                return False

        return True

    def _execute_workflow_step(self, workflow_id: str) -> None:
        """
        执行工作流步骤

        Args:
            workflow_id: 工作流ID
        """
        if not self._is_workflow_active(workflow_id):
            return

        workflow = self.active_workflows[workflow_id]
        current_step = self._get_current_step(workflow)

        if current_step is None:
            self._complete_workflow(workflow_id)
            return

        self._execute_single_step(workflow_id, workflow, current_step)

    def _is_workflow_active(self, workflow_id: str) -> bool:
        """
        检查工作流是否活跃

        Args:
            workflow_id: 工作流ID

        Returns:
            是否活跃
        """
        return workflow_id in self.active_workflows

    def _get_current_step(self, workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        获取当前步骤

        Args:
            workflow: 工作流对象

        Returns:
            当前步骤配置，如果工作流已完成则返回None
        """
        config = workflow["config"]
        steps = config.get("steps", [])

        if workflow["current_step"] >= len(steps):
            return None

        return steps[workflow["current_step"]]

    def _execute_single_step(self, workflow_id: str, workflow: Dict[str, Any], step: Dict[str, Any]) -> None:
        """
        执行单个步骤

        Args:
            workflow_id: 工作流ID
            workflow: 工作流对象
            step: 步骤配置
        """
        step_name = step["name"]
        service_name = step["service"]
        method_name = step["method"]

        self.logger.info(f"执行工作流步骤: {workflow_id} -> {step_name}")

        try:
            result = self._invoke_step_method(service_name, method_name, workflow["input_data"])
            self._update_workflow_progress(workflow_id, workflow, step_name, result)
            self._execute_workflow_step(workflow_id)  # 递归执行下一步

        except Exception as e:
            self.logger.error(f"工作流步骤执行失败: {workflow_id} -> {step_name}, 错误: {e}")
            self._handle_workflow_error(workflow_id, step_name, str(e))

    def _invoke_step_method(self, service_name: str, method_name: str, input_data: Any) -> Any:
        """
        调用步骤方法

        Args:
            service_name: 服务名称
            method_name: 方法名称
            input_data: 输入数据

        Returns:
            方法执行结果

        Raises:
            Exception: 服务或方法不存在
        """
        if not self.container.has(service_name):
            raise Exception(f"服务不存在: {service_name}")

        service = self.container.get(service_name)

        if not hasattr(service, method_name):
            raise Exception(f"方法不存在: {service_name}.{method_name}")

        method = getattr(service, method_name)
        return method(input_data)

    def _update_workflow_progress(self, workflow_id: str, workflow: Dict[str, Any],
                                  step_name: str, result: Any) -> None:
        """
        更新工作流进度

        Args:
            workflow_id: 工作流ID
            workflow: 工作流对象
            step_name: 步骤名称
            result: 执行结果
        """
        workflow["results"][step_name] = result
        workflow["current_step"] += 1
        self.workflow_metrics[workflow_id]["steps_completed"] += 1

    def _complete_workflow(self, workflow_id: str):
        """完成工作流"""
        self.logger.info(f"工作流完成: {workflow_id}")

        workflow = self.active_workflows[workflow_id]
        workflow["status"] = "completed"
        workflow["end_time"] = time.time()

        # 更新指标
        self.workflow_metrics[workflow_id]["status"] = "completed"
        self.workflow_metrics[workflow_id]["end_time"] = time.time()

        # 发布完成事件
        completion_event = Event(
            event_type="WORKFLOW_COMPLETED",
            data={
                "workflow_id": workflow_id,
                "results": workflow["results"],
                "duration": workflow["end_time"] - workflow["start_time"],
                "timestamp": time.time(),
                "source": "BusinessService"
            }
        )
        self.event_bus.publish(completion_event)

    def _handle_workflow_error(self, workflow_id: str, step_name: str, error: str):
        """处理工作流错误"""
        self.logger.error(f"工作流错误: {workflow_id} -> {step_name} -> {error}")

        workflow = self.active_workflows[workflow_id]
        workflow["status"] = "error"
        workflow["error"] = {
            "step": step_name,
            "message": error,
            "timestamp": time.time()
        }

        # 更新指标
        self.workflow_metrics[workflow_id]["status"] = "error"
        self.workflow_metrics[workflow_id]["errors"].append({
            "step": step_name,
            "message": error,
            "timestamp": time.time()
        })

        # 发布错误事件
        error_event = Event(
            event_type="WORKFLOW_ERROR",
            data={
                "workflow_id": workflow_id,
                "step": step_name,
                "error": error,
                "timestamp": time.time(),
                "source": "BusinessService"
            }
        )

        self.event_bus.publish(error_event)

        # 事件处理方法

    def _on_data_ready(self, event: Event):
        """处理数据就绪事件"""
        self.logger.info("收到数据就绪事件")
        # 可以在这里触发相关的工作流

    def _on_feature_extracted(self, event: Event):
        """处理特征提取完成事件"""
        self.logger.info("收到特征提取完成事件")

    def _on_model_predicted(self, event: Event):
        """处理模型预测完成事件"""
        self.logger.info("收到模型预测完成事件")

    def _on_signal_generated(self, event: Event):
        """处理信号生成事件"""
        self.logger.info("收到信号生成事件")

    def _on_risk_checked(self, event: Event):
        """处理风控检查完成事件"""
        self.logger.info("收到风控检查完成事件")

    def _on_execution_completed(self, event: Event):
        """处理执行完成事件"""
        self.logger.info("收到执行完成事件")

    def _on_validation_completed(self, event: Event):
        """处理验证完成事件"""
        self.logger.info("收到验证完成事件")

    # ------------------------------------------------------------------
    # 兼容旧接口所需的抽象方法实现
    # ------------------------------------------------------------------
    def _get_status(self) -> Dict[str, Any]:
        """
        BaseService 抽象方法实现：
        返回当前业务服务的健康状态及核心指标。
        """
        status = self._health_check()
        status.update({
            "name": self.name,
            "enabled": True,
            "workflows_total": len(self.workflow_configs),
            "active_workflows": len(self.active_workflows),
        })
        return status

    def _get_info(self) -> Dict[str, Any]:
        """
        BaseService 抽象方法实现：
        返回业务服务的基本信息和配置摘要。
        """
        return {
            "service_name": self.name,
            "service_type": self.__class__.__name__,
            "workflows": len(self.workflow_configs),
            "active_workflows": len(self.active_workflows),
            "dependencies": list(self.service_dependencies.keys()),
            "config": {
                "workflow_templates": list(self.workflow_configs.keys()),
            },
        }


class _NoopEventBus:
    """兼容性事件总线，用于无依赖测试场景"""

    def subscribe(self, *args, **kwargs):
        return None

    def unsubscribe(self, *args, **kwargs):
        return None

    def publish(self, *args, **kwargs):
        return None


class _NoopContainer:
    """兼容性依赖容器，返回空对象"""

    def has(self, _service: str) -> bool:
        return False

    def get(self, _service: str) -> Any:
        raise KeyError(_service)


class TestableBusinessService(BusinessService):
    """
    兼容性业务服务实现

    历史测试依赖 `TestableBusinessService` 提供的默认事件总线和容器。
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        container: Optional[ServiceContainer] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(event_bus=event_bus, container=container, name=name)


try:
    import builtins as _builtins

    if not hasattr(_builtins, "TestableBusinessService"):
        setattr(_builtins, "TestableBusinessService", TestableBusinessService)
except Exception:
    pass


__all__ = [
    "BusinessService",
    "TestableBusinessService",
]

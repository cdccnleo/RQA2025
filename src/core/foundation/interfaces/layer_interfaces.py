#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一接口抽象层定义
基于业务流程驱动的架构各层接口

本模块定义了RQA2025量化平台中各架构层的标准接口规范，
确保层间通信的一致性和可维护性。

架构层级：
1. 数据采集层 - 实时数据采集和处理
2. 特征处理层 - 特征工程和加速计算
3. 模型推理层 - 模型预测和集成学习
4. 策略决策层 - 策略生成和回测
5. 风控合规层 - 风险控制和合规检查
6. 交易执行层 - 订单执行和管理
7. 监控反馈层 - 系统监控和性能优化
8. 基础设施层 - 基础服务和资源管理
9. 核心服务层 - 事件总线和依赖注入

接口设计原则：
- 单一职责：每个接口职责明确
- 依赖倒置：高层依赖抽象接口
- 稳定抽象：接口相对稳定，实现可变
- 清晰命名：接口名称反映其职责
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from packaging import version

from ...integration.interfaces.layer_interface import LayerInterface

logger = logging.getLogger(__name__)


class ILayerInterfaceComponent(ABC):

    """
    层间标准接口基类

    定义了所有架构层组件必须实现的通用方法，
    确保层间通信的标准化和一致性。

    主要职责：
    1. 统一请求处理流程
    2. 标准化状态监控
    3. 规范化健康检查机制
    """

    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理层间请求

        Args:
            request: 请求数据字典，包含：
                    - 'action': 请求动作
                    - 'params': 请求参数
                    - 'context': 请求上下文

        Returns:
            Dict[str, Any]: 响应数据字典，包含：
                           - 'status': 处理状态 ('success'/'error')
                           - 'data': 响应数据
                           - 'message': 状态消息

        Raises:
            NotImplementedError: 子类必须实现此方法
        """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取组件状态信息

        Returns:
            Dict[str, Any]: 状态信息字典，包含：
                           - 'component_name': 组件名称
                           - 'status': 运行状态 ('running'/'stopped'/'error')
                           - 'health': 健康状态 ('healthy'/'unhealthy')
                           - 'uptime': 运行时间(秒)
                           - 'version': 版本信息

        Raises:
            NotImplementedError: 子类必须实现此方法
        """

    @abstractmethod
    def health_check(self) -> bool:
        """
        执行健康检查

        检查组件的各个关键指标是否正常，包括：
        - 内存使用情况
        - 连接状态
        - 依赖服务可用性
        - 业务逻辑完整性

        Returns:
            bool: True表示健康，False表示异常

        Raises:
            NotImplementedError: 子类必须实现此方法
        """

    def get_version(self) -> str:
        """
        获取组件版本信息

        Returns:
            str: 版本字符串，格式为 "major.minor.patch"

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 get_version 方法")

    def check_version_compatibility(self, required_version: str, current_version: str = None) -> Dict[str, Any]:
        """
        检查版本兼容性

        Args:
            required_version: 所需的最低版本
            current_version: 当前版本，如果为None则自动获取

        Returns:
            Dict[str, Any]: 兼容性检查结果，包含：
                           - 'compatible': 是否兼容
                           - 'current_version': 当前版本
                           - 'required_version': 所需版本
                           - 'comparison': 版本比较结果
                           - 'message': 检查结果说明
        """
        try:
            if current_version is None:
                current_version = self.get_version()

            # 使用packaging库进行版本比较
            current_ver = version.parse(current_version)
            required_ver = version.parse(required_version)

            is_compatible = current_ver >= required_ver

            comparison = "compatible" if is_compatible else "incompatible"

            if is_compatible:
                message = f"版本兼容: 当前版本 {current_version} >= 所需版本 {required_version}"
            else:
                message = f"版本不兼容: 当前版本 {current_version} < 所需版本 {required_version}"

            return {
                'compatible': is_compatible,
                'current_version': current_version,
                'required_version': required_version,
                'comparison': comparison,
                'message': message
            }

        except Exception as e:
            logger.error(f"版本兼容性检查失败: {e}")
            return {
                'compatible': False,
                'current_version': current_version or 'unknown',
                'required_version': required_version,
                'comparison': 'error',
                'message': f'版本检查失败: {str(e)}'
            }

# ==================== 数据层接口 ====================


class IDataProviderComponent(ABC):

    """
    数据提供者接口

    定义了数据源的基本访问方法，用于获取各种类型的数据。
    实现类可以是实时数据源、历史数据源、模拟数据源等。

    主要功能：
    1. 实时市场数据获取
    2. 历史数据查询
    3. 数据源状态监控
    """

    @abstractmethod
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        获取实时市场数据

        Args:
            symbols: 股票代码列表，如 ['600519.SH', '000001.SZ']

        Returns:
            Dict[str, Any]: 市场数据字典，包含：
                          - 'timestamp': 数据时间戳
                          - 'data': 各股票的实时数据
                          - 'source': 数据源标识

        Raises:
            ConnectionError: 数据源连接失败
            TimeoutError: 请求超时
            ValueError: 无效的股票代码
        """

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        获取历史数据

        Args:
            symbol: 股票代码，如 '600519.SH'
            start_date: 开始日期，格式 'YYYY - MM - DD'
            end_date: 结束日期，格式 'YYYY - MM - DD'

        Returns:
            Dict[str, Any]: 历史数据字典，包含：
                          - 'symbol': 股票代码
                          - 'data': 历史数据列表
                          - 'frequency': 数据频率
                          - 'period': 时间范围

        Raises:
            ValueError: 日期格式错误或无效的股票代码
            NotFoundError: 未找到相关数据
        """


class DataManagementInterface(ILayerInterfaceComponent):

    """
    数据管理层接口

    负责整个数据层的统筹管理，包括数据采集、存储、质量控制等。
    作为数据层的核心协调器，管理各种数据源和数据处理组件。

    主要职责：
    1. 协调多数据源的数据采集
    2. 统一数据存储和缓存管理
    3. 数据质量监控和异常处理
    4. 数据访问和查询优化
    """

    @abstractmethod
    def collect_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        采集市场数据

        从配置的数据源采集指定股票的实时市场数据，
        支持多数据源 failover 和数据聚合。

        Args:
            symbols: 股票代码列表

        Returns:
            Dict[str, Any]: 采集结果，包含：
                          - 'status': 采集状态
                          - 'data': 采集到的数据
                          - 'sources': 数据源列表
                          - 'timestamp': 采集时间

        Raises:
            DataCollectionError: 数据采集失败
            TimeoutError: 采集超时
        """

    @abstractmethod
    def store_data(self, data: Dict[str, Any]) -> bool:
        """
        存储数据

        将数据存储到配置的存储系统中，支持多种存储后端。

        Args:
            data: 要存储的数据字典

        Returns:
            bool: 存储是否成功

        Raises:
            StorageError: 存储失败
            ValidationError: 数据格式验证失败
        """

    @abstractmethod
    def check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查数据质量

        对数据进行完整性、准确性、一致性等质量检查。

        Args:
            data: 要检查的数据

        Returns:
            Dict[str, Any]: 质量检查报告，包含：
                          - 'score': 质量评分 (0 - 100)
                          - 'issues': 发现的问题列表
                          - 'recommendations': 改进建议

        Raises:
            QualityCheckError: 质量检查过程出错
        """

    @abstractmethod
    def get_data_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        根据股票代码获取数据

        从存储系统中查询指定股票的最新数据。

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 股票数据字典

        Raises:
            NotFoundError: 未找到相关数据
            QueryError: 查询过程出错
        """

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        获取历史数据

        查询指定股票在时间范围内的历史数据。

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, Any]: 历史数据字典

        Raises:
            ValueError: 日期格式错误
            NotFoundError: 未找到数据
            QueryError: 查询失败
        """

# ==================== 特征层接口 ====================


class IFeatureProviderComponent(ABC):

    """特征提供者接口"""

    @abstractmethod
    def extract_features(self, data: Dict) -> Dict:
        """提取特征"""

    @abstractmethod
    def validate_features(self, features: Dict) -> bool:
        """验证特征"""


class FeatureProcessingInterface(LayerInterface):

    """特征处理层接口"""

    @abstractmethod
    def extract_features(self, data: dict) -> dict:
        """特征提取"""

    @abstractmethod
    def process_features(self, features: dict) -> dict:
        """特征处理"""

    @abstractmethod
    def accelerate_with_gpu(self, features: dict) -> dict:
        """GPU加速"""

    @abstractmethod
    def calculate_technical_indicators(self, data: dict) -> dict:
        """计算技术指标"""

    @abstractmethod
    def normalize_features(self, features: dict) -> dict:
        """特征标准化"""

# ==================== 模型层接口 ====================


class IModelProviderComponent(ABC):

    """模型提供者接口"""

    @abstractmethod
    def predict(self, features: Dict) -> Dict:
        """模型预测"""

    @abstractmethod
    def train(self, data: Dict) -> bool:
        """模型训练"""


class ModelInferenceInterface(LayerInterface):

    """模型推理层接口"""

    @abstractmethod
    def train_model(self, features: dict) -> dict:
        """模型训练"""

    @abstractmethod
    def predict(self, features: dict) -> dict:
        """模型预测"""

    @abstractmethod
    def ensemble_predict(self, predictions: dict) -> dict:
        """模型集成预测"""

    @abstractmethod
    def evaluate_model(self, model_id: str, test_data: dict) -> dict:
        """模型评估"""

    @abstractmethod
    def deploy_model(self, model_id: str) -> bool:
        """模型部署"""

# ==================== 策略层接口 ====================


class StrategyDecisionInterface(LayerInterface):

    """策略决策层接口"""

    @abstractmethod
    def make_decision(self, predictions: dict) -> dict:
        """策略决策"""

    @abstractmethod
    def generate_signals(self, decision: dict) -> dict:
        """信号生成"""

    @abstractmethod
    def optimize_parameters(self, parameters: dict) -> dict:
        """参数优化"""

    @abstractmethod
    def backtest_strategy(self, strategy_config: dict) -> dict:
        """策略回测"""

    @abstractmethod
    def calculate_position_size(self, signal: dict, capital: float) -> dict:
        """计算仓位大小"""

# ==================== 风控层接口 ====================


class IRiskProviderComponent(ABC):

    """风控提供者接口"""

    @abstractmethod
    def check_risk(self, order_data: Dict) -> Dict:
        """风控检查"""

    @abstractmethod
    def get_risk_limits(self) -> Dict:
        """获取风控限制"""


class RiskComplianceInterface(LayerInterface):

    """风控合规层接口"""

    @abstractmethod
    def check_risk(self, signals: dict) -> dict:
        """风险检查"""

    @abstractmethod
    def verify_compliance(self, risk_result: dict) -> dict:
        """合规验证"""

    @abstractmethod
    def monitor_realtime(self, metrics: dict) -> dict:
        """实时监控"""

    @abstractmethod
    def calculate_var(self, positions: dict) -> float:
        """计算VaR"""

    @abstractmethod
    def check_exposure_limits(self, positions: dict) -> dict:
        """检查敞口限制"""

# ==================== 交易层接口 ====================


class IExecutionProviderComponent(ABC):

    """执行提供者接口"""

    @abstractmethod
    def execute_orders(self, orders: List[Dict]) -> Dict:
        """执行订单"""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """获取订单状态"""


class TradingExecutionInterface(LayerInterface):

    """交易执行层接口"""

    @abstractmethod
    def generate_orders(self, compliance_result: dict) -> dict:
        """订单生成"""

    @abstractmethod
    def execute_orders(self, orders: dict) -> dict:
        """订单执行"""

    @abstractmethod
    def handle_execution_feedback(self, feedback: dict) -> dict:
        """处理执行反馈"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """获取订单状态"""

# ==================== 监控层接口 ====================


class MonitoringFeedbackInterface(LayerInterface):

    """监控反馈层接口"""

    @abstractmethod
    def update_performance_metrics(self, execution_result: dict) -> dict:
        """更新性能指标"""

    @abstractmethod
    def handle_performance_alert(self, alert: dict) -> dict:
        """处理性能告警"""

    @abstractmethod
    def handle_business_alert(self, alert: dict) -> dict:
        """处理业务告警"""

    @abstractmethod
    def generate_performance_report(self) -> dict:
        """生成性能报告"""

    @abstractmethod
    def send_notification(self, message: str, level: str) -> bool:
        """发送通知"""

# ==================== 基础设施层接口 ====================


class InfrastructureInterface(LayerInterface):

    """基础设施层接口"""

    @abstractmethod
    def get_config(self, key: str) -> Any:
        """获取配置"""

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置"""

    @abstractmethod
    def get_cache(self, key: str) -> Any:
        """获取缓存"""

    @abstractmethod
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存"""

    @abstractmethod
    def log_event(self, event_type: str, data: dict) -> bool:
        """记录事件"""

# ==================== 核心服务层接口 ====================


class CoreServicesInterface(LayerInterface):

    """核心服务层接口"""

    @abstractmethod
    def get_event_bus(self):
        """获取事件总线"""

    @abstractmethod
    def get_dependency_container(self):
        """获取依赖注入容器"""

    @abstractmethod
    def register_service(self, service_name: str, service_instance: Any) -> bool:
        """注册服务"""

    @abstractmethod
    def get_service(self, service_name: str) -> Any:
        """获取服务"""

    @abstractmethod
    def publish_event(self, event_type: str, data: dict) -> bool:
        """发布事件"""

# ==================== 基础实现类 ====================


class BaseLayerImplementation:

    """基础层实现类"""

    def __init__(self, name: str):

        self.name = name
        self.status = "initialized"
        self.health_status = True
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def process_request(self, request: dict) -> dict:
        """处理请求 - 基础实现"""
        self.logger.info(f"处理请求: {request}")
        return {"status": "success", "message": "请求已处理"}

    def get_status(self) -> dict:
        """获取状态 - 基础实现"""
        return {
            "name": self.name,
            "status": self.status,
            "health": self.health_status,
            "timestamp": time.time()
        }

    def health_check(self) -> bool:
        """健康检查 - 基础实现"""
        return self.health_status

    def set_status(self, status: str):
        """设置状态"""
        self.status = status
        self.logger.info(f"状态更新: {status}")

    def set_health_status(self, health: bool):
        """设置健康状态"""
        self.health_status = health
        self.logger.info(f"健康状态更新: {health}")

# ==================== 接口工厂 ====================


class InterfaceFactory:

    """接口工厂类"""

    _interfaces = {}

    @classmethod
    def register_interface(cls, name: str, interface_class: type):
        """注册接口"""
        cls._interfaces[name] = interface_class
        logger.info(f"注册接口: {name}")

    @classmethod
    def get_interface(cls, name: str) -> Optional[type]:
        """获取接口"""
        return cls._interfaces.get(name)

    @classmethod
    def list_interfaces(cls) -> List[str]:
        """列出所有接口"""
        return list(cls._interfaces.keys())


# 避免重复注册的标志
_interfaces_registered = False

def _register_interfaces_once():
    """只执行一次的接口注册"""
    global _interfaces_registered
    if _interfaces_registered:
        return

    # 注册所有接口
    try:
        InterfaceFactory.register_interface("data_provider", IDataProvider)
    except NameError:
        # 如果IDataProvider未定义，注册一个占位符

        class IDataProvider:

            pass
        InterfaceFactory.register_interface("data_provider", IDataProvider)

    # 批量注册接口，处理未定义的接口类
    interfaces_to_register = [
        ("data_management", "DataManagementInterface"),
        ("feature_provider", "IFeatureProvider"),
        ("feature_processing", "FeatureProcessingInterface"),
        ("model_provider", "IModelProvider"),
        ("model_inference", "ModelInferenceInterface"),
    ]

    for interface_name, interface_class_name in interfaces_to_register:
        try:
            # 动态获取接口类
            interface_class = globals().get(interface_class_name)
            if interface_class:
                InterfaceFactory.register_interface(interface_name, interface_class)
            else:
                raise NameError(f"name '{interface_class_name}' is not defined")
        except NameError:
            # 如果接口类未定义，注册一个占位符

            class PlaceholderInterface:

                pass
            InterfaceFactory.register_interface(interface_name, PlaceholderInterface)

    # 批量注册剩余接口
    remaining_interfaces = [
        ("strategy_decision", "StrategyDecisionInterface"),
        ("risk_provider", "IRiskProvider"),
        ("risk_compliance", "RiskComplianceInterface"),
        ("execution_provider", "IExecutionProvider"),
        ("trading_execution", "TradingExecutionInterface"),
    ]

    for interface_name, interface_class_name in remaining_interfaces:
        try:
            # 动态获取接口类
            interface_class = globals().get(interface_class_name)
            if interface_class:
                InterfaceFactory.register_interface(interface_name, interface_class)
            else:
                raise NameError(f"name '{interface_class_name}' is not defined")
        except NameError:
            # 如果接口类未定义，注册一个占位符

            class PlaceholderInterface:

                pass
            InterfaceFactory.register_interface(interface_name, PlaceholderInterface)

    # 批量注册最后几个接口
    final_interfaces = [
        ("monitoring_feedback", "MonitoringFeedbackInterface"),
        ("infrastructure", "InfrastructureInterface"),
        ("core_services", "CoreServicesInterface"),
    ]

    for interface_name, interface_class_name in final_interfaces:
        try:
            # 动态获取接口类
            interface_class = globals().get(interface_class_name)
            if interface_class:
                InterfaceFactory.register_interface(interface_name, interface_class)
            else:
                raise NameError(f"name '{interface_class_name}' is not defined")
        except NameError:
            # 如果接口类未定义，注册一个占位符

            class PlaceholderInterface:

                pass
            InterfaceFactory.register_interface(interface_name, PlaceholderInterface)

    _interfaces_registered = True

# 执行一次性接口注册
_register_interfaces_once()

# 导入时间模块

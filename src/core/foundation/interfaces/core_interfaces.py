#!/usr/bin/env python3
"""
RQA2025 核心服务层统一接口定义

基于标准接口模板定义核心服务层所有组件的标准接口
确保与基础设施层和其他层级的接口一致性

作者: AI Assistant
版本: 2.0.0 (基于统一接口标准)
更新时间: 2025年9月29日
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Protocol
from .standard_interface_template import (
    IStatusProvider, IHealthCheckable, ILifecycleManageable,
    IServiceProvider, StandardComponent, ComponentStatus, ComponentHealth
)


class ICoreComponent(Protocol):
    """核心组件统一接口协议

    所有核心服务层组件都必须实现此接口协议，
    确保统一的组件管理和监控能力。
    """

    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        ...

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...

    def initialize(self) -> bool:
        """初始化"""
        ...

    def start(self) -> bool:
        """启动"""
        ...

    def stop(self) -> bool:
        """停止"""
        ...

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务详细信息"""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件性能指标"""
        ...


class IEventBus(Protocol):
    """事件总线接口协议"""

    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        ...

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...

    def publish(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """发布事件"""
        ...

    def subscribe(self, event_type: str, handler: callable) -> bool:
        """订阅事件"""
        ...

    def unsubscribe(self, event_type: str, handler: callable) -> bool:
        """取消订阅事件"""
        ...


class IDependencyContainer(Protocol):
    """依赖注入容器接口协议"""

    def get_service(self, service_name: str) -> Any:
        """获取服务"""
        ...

    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        ...

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...

    def register(self, name: str, service: Any = None, service_type: Optional[type] = None,
                 factory: Optional[callable] = None, lifecycle: str = "singleton",
                 dependencies: Optional[List[str]] = None, **kwargs) -> bool:
        """注册服务"""
        ...

    def resolve(self, name: str) -> Any:
        """解析服务依赖"""
        ...

    def create_scope(self) -> Any:
        """创建作用域"""
        ...


class IBusinessProcessOrchestrator(Protocol):
    """业务流程编排器接口协议"""

    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        ...

    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...

    def start_process(self, process_name: str, context: Dict[str, Any]) -> str:
        """启动业务流程"""
        ...

    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """获取流程状态"""
        ...

    def stop_process(self, process_id: str) -> bool:
        """停止业务流程"""
        ...


class ILayerInterface(Protocol):
    """层间接口协议"""

    def communicate_up(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向上层通信"""
        ...

    def communicate_down(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向下层通信"""
        ...

    def validate_message(self, message: Dict[str, Any]) -> bool:
        """验证消息格式"""
        ...


class IBusinessProcess(Protocol):
    """业务流程接口"""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行业务流程"""
        ...

    def get_status(self) -> Dict[str, Any]:
        """获取流程状态"""
        ...

    def validate(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        ...


class ITradingEngine(Protocol):
    """交易引擎接口"""

    def submit_order(self, order: Dict[str, Any]) -> str:
        """提交订单"""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        ...

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        ...


class IStrategyManager(Protocol):
    """策略管理器接口"""

    def register_strategy(self, strategy_id: str, strategy: Any) -> bool:
        """注册策略"""
        ...

    def get_strategy(self, strategy_id: str) -> Any:
        """获取策略"""
        ...

    def execute_strategy(self, strategy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行策略"""
        ...


class IPortfolioManager(Protocol):
    """投资组合管理器接口"""

    def get_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """获取投资组合"""
        ...

    def update_position(self, symbol: str, quantity: float) -> bool:
        """更新持仓"""
        ...

    def get_positions(self) -> List[Dict[str, Any]]:
        """获取所有持仓"""
        ...


class IRiskManager(Protocol):
    """风险管理器接口"""

    def check_risk(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """检查风险"""
        ...

    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        ...

    def validate_order(self, order: Dict[str, Any]) -> bool:
        """验证订单"""
        ...


class IMarketDataProvider(Protocol):
    """市场数据提供者接口"""

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """获取行情"""
        ...

    def subscribe(self, symbol: str, callback: callable) -> bool:
        """订阅行情"""
        ...

    def unsubscribe(self, symbol: str) -> bool:
        """取消订阅"""
        ...


class IOrderManager(Protocol):
    """订单管理器接口"""

    def create_order(self, order_data: Dict[str, Any]) -> str:
        """创建订单"""
        ...

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """获取订单"""
        ...

    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """更新订单"""
        ...


class IReportingService(Protocol):
    """报告服务接口"""

    def generate_report(self, report_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        ...

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """获取报告"""
        ...

    def list_reports(self) -> List[Dict[str, Any]]:
        """列出报告"""
        ...


class IAuditService(Protocol):
    """审计服务接口"""

    def log_audit(self, action: str, user: str, details: Dict[str, Any]) -> bool:
        """记录审计日志"""
        ...

    def get_audit_logs(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取审计日志"""
        ...

    def audit_trail(self, entity_id: str) -> List[Dict[str, Any]]:
        """获取审计轨迹"""
        ...


class IMonitoringService(Protocol):
    """监控服务接口"""

    def start_monitoring(self) -> bool:
        """启动监控"""
        ...

    def stop_monitoring(self) -> bool:
        """停止监控"""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        ...

    def set_alert_threshold(self, metric: str, threshold: float) -> bool:
        """设置告警阈值"""
        ...
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标"""
        ...
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        ...
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """获取告警历史"""
        ...


class ICacheManager(Protocol):
    """缓存管理器接口"""

    def get(self, key: str) -> Any:
        """获取缓存"""
        ...

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存"""
        ...

    def delete(self, key: str) -> bool:
        """删除缓存"""
        ...

    def clear(self) -> bool:
        """清空缓存"""
        ...


class IConfigurationManager(Protocol):
    """配置管理器接口"""

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        ...

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置"""
        ...

    def load_config(self, path: str) -> bool:
        """加载配置"""
        ...

    def save_config(self, path: str) -> bool:
        """保存配置"""
        ...


class IDatabaseService(Protocol):
    """数据库服务接口"""

    def connect(self) -> bool:
        """连接数据库"""
        ...

    def disconnect(self) -> bool:
        """断开连接"""
        ...

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行查询"""
        ...

    def execute_command(self, command: str, params: Dict[str, Any] = None) -> bool:
        """执行命令"""
        ...
    
    def begin_transaction(self) -> bool:
        """开始事务"""
        ...
    
    def commit_transaction(self) -> bool:
        """提交事务"""
        ...
    
    def rollback_transaction(self) -> bool:
        """回滚事务"""
        ...
    
    def execute_update(self, command: str, params: Dict[str, Any] = None) -> int:
        """执行更新"""
        ...


class ILoggingService(Protocol):
    """日志服务接口"""

    def log(self, level: str, message: str, context: Dict[str, Any] = None) -> bool:
        """记录日志"""
        ...

    def debug(self, message: str) -> bool:
        """调试日志"""
        ...

    def info(self, message: str) -> bool:
        """信息日志"""
        ...

    def warning(self, message: str) -> bool:
        """警告日志"""
        ...

    def error(self, message: str) -> bool:
        """错误日志"""
        ...

    def set_log_level(self, level: str) -> bool:
        """设置日志级别"""
        ...
    
    def log_debug(self, message: str) -> bool:
        """调试日志（别名）"""
        ...
    
    def log_info(self, message: str) -> bool:
        """信息日志（别名）"""
        ...
    
    def log_warning(self, message: str) -> bool:
        """警告日志（别名）"""
        ...
    
    def log_error(self, message: str) -> bool:
        """错误日志（别名）"""
        ...
    
    def log_critical(self, message: str) -> bool:
        """严重日志"""
        ...
    
    def get_logs(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取日志"""
        ...


class ISecurityService(Protocol):
    """安全服务接口"""

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """认证"""
        ...

    def authorize(self, user: str, resource: str, action: str) -> bool:
        """授权"""
        ...

    def encrypt(self, data: Any) -> Any:
        """加密"""
        ...

    def decrypt(self, data: Any) -> Any:
        """解密"""
        ...


class IPerformanceOptimizer(Protocol):
    """性能优化器接口"""

    def optimize(self, target: str) -> Dict[str, Any]:
        """优化"""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        ...

    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        ...


class ILoadBalancer(Protocol):
    """负载均衡器接口"""

    def add_server(self, server: Any) -> bool:
        """添加服务器"""
        ...

    def get_next_server(self) -> Any:
        """获取下一个服务器"""
        ...

    def remove_server(self, server: Any) -> bool:
        """移除服务器"""
        ...


# =============================================================================
# 标准实现类
# =============================================================================

class CoreComponent(StandardComponent):
    """核心组件标准实现基类"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        super().__init__(name, version, description)

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'component_type': 'core_component',
            'layer': 'core_services',
            **self.get_status_info()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'uptime_seconds': self.get_status_info().get('uptime_seconds', 0),
            'error_count': self.get_status_info().get('error_count', 0),
            'health_checks': 1 if self.get_status_info().get('last_health_check') else 0
        }

    @abstractmethod
    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查的具体逻辑"""


__all__ = [
    # 接口协议
    'ICoreComponent',
    'IEventBus',
    'IDependencyContainer',
    'IBusinessProcessOrchestrator',
    'ILayerInterface',
    'IBusinessProcess',
    'ITradingEngine',
    'IStrategyManager',
    'IPortfolioManager',
    'IRiskManager',
    'IMarketDataProvider',
    'IOrderManager',
    'IReportingService',
    'IAuditService',
    'IMonitoringService',
    'ICacheManager',
    'IConfigurationManager',
    'IDatabaseService',
    'ILoggingService',
    'ISecurityService',
    'IPerformanceOptimizer',
    'ILoadBalancer',

    # 标准实现
    'CoreComponent',

    # 导入的标准类型
    'ComponentStatus',
    'ComponentHealth',
    'StandardComponent'
]

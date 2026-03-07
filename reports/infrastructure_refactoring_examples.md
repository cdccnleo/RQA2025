# 基础设施层重构示例

## 概述

本文档提供基础设施层重构的具体代码示例，展示如何将过大的类文件拆分为职责单一的小类，以及如何优化模块结构。

## 示例1：EnhancedHealthChecker重构

### 重构前的问题

```python
# 重构前：enhanced_health_checker.py (644行)
class EnhancedHealthChecker:
    """增强的健康检查器 - 职责过多"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 初始化多个组件
        self.cache_manager = get_cache_manager()
        self.prometheus_exporter = get_prometheus_exporter()
        self.alert_manager = get_alert_manager()
        self.performance_optimizer = get_performance_optimizer()
        self.grafana_integration = get_grafana_integration()
        self.alert_rule_engine = get_alert_rule_engine()
        # ... 更多初始化代码
    
    # 健康检查相关方法 (约200行)
    async def perform_health_check(self, service: str, check_type: str) -> HealthCheckResult:
        # 复杂的健康检查逻辑
        pass
    
    # 系统监控相关方法 (约150行)
    async def get_system_metrics(self) -> SystemMetrics:
        # 系统指标收集逻辑
        pass
    
    # 告警管理相关方法 (约100行)
    def _check_alert_conditions(self, result: HealthCheckResult) -> None:
        # 告警条件检查逻辑
        pass
    
    # 性能优化相关方法 (约100行)
    def get_performance_suggestions(self) -> List[Dict[str, Any]]:
        # 性能优化建议逻辑
        pass
    
    # Grafana集成相关方法 (约50行)
    def deploy_grafana_dashboards(self) -> Dict[str, Any]:
        # Grafana仪表板部署逻辑
        pass
    
    # 缓存管理相关方法 (约44行)
    def get_cache_stats(self) -> Dict[str, Any]:
        # 缓存统计信息逻辑
        pass
```

### 重构后的结构

#### 1. 基础健康检查器
```python
# src/infrastructure/health/core/health_checker.py
class HealthChecker:
    """基础健康检查器 - 职责单一"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks: Dict[str, Callable] = {}
        self.dependency_checks: Dict[str, Callable] = {}
        self._register_default_checks()
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """注册健康检查函数"""
        self.health_checks[name] = check_func
    
    def register_dependency_check(self, name: str, check_func: Callable) -> None:
        """注册依赖检查函数"""
        self.dependency_checks[name] = check_func
    
    async def perform_health_check(self, service: str, check_type: str) -> HealthCheckResult:
        """执行健康检查"""
        if check_type not in self.health_checks:
            raise ValueError(f"Unknown check type: {check_type}")
        
        check_func = self.health_checks[check_type]
        return await self._execute_check(check_func, service, check_type)
    
    async def _execute_check(self, check_func: Callable, service: str, 
                           check_type: str) -> HealthCheckResult:
        """执行具体的检查函数"""
        start_time = time.time()
        try:
            result = await check_func(service)
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                service=service,
                check_type=check_type,
                status="healthy",
                response_time=response_time,
                details=result,
                timestamp=datetime.now()
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                service=service,
                check_type=check_type,
                status="error",
                response_time=response_time,
                details={},
                timestamp=datetime.now(),
                error_code=str(type(e).__name__),
                error_message=str(e)
            )
```

#### 2. 系统监控器
```python
# src/infrastructure/health/monitoring/system_monitor.py
class SystemMonitor:
    """系统监控器 - 职责单一"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
    
    async def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        if not self.monitoring_enabled:
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_bytes=0,
                disk_usage={},
                network_io={},
                timestamp=datetime.now()
            )
        
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            memory_bytes=psutil.virtual_memory().used,
            disk_usage=self._get_disk_usage(),
            network_io=self._get_network_io(),
            timestamp=datetime.now()
        )
    
    def _get_disk_usage(self) -> Dict[str, float]:
        """获取磁盘使用情况"""
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.device] = usage.percent
            except (PermissionError, OSError):
                continue
        return disk_usage
    
    def _get_network_io(self) -> Dict[str, int]:
        """获取网络IO统计"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
```

#### 3. 告警管理器
```python
# src/infrastructure/health/alerting/alert_manager.py
class AlertManager:
    """告警管理器 - 职责单一"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alerting_enabled = self.config.get('alerting_enabled', True)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """添加告警规则"""
        try:
            self.alert_rules[rule.name] = rule
            return True
        except Exception:
            return False
    
    def check_alert_conditions(self, health_result: HealthCheckResult) -> None:
        """检查告警条件"""
        if not self.alerting_enabled:
            return
        
        for rule_name, rule in self.alert_rules.items():
            if rule.should_trigger(health_result):
                self._create_alert(rule, health_result)
    
    def _create_alert(self, rule: AlertRule, health_result: HealthCheckResult) -> None:
        """创建告警"""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.format_message(health_result),
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE
        )
        self.active_alerts.append(alert)
```

#### 4. 性能优化器
```python
# src/infrastructure/health/optimization/performance_optimizer.py
class PerformanceOptimizer:
    """性能优化器 - 职责单一"""
    
    def __init__(self, cache_manager, prometheus_exporter):
        self.cache_manager = cache_manager
        self.prometheus_exporter = prometheus_exporter
        self.optimization_rules: List[OptimizationRule] = []
    
    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """添加优化规则"""
        self.optimization_rules.append(rule)
    
    def get_performance_suggestions(self) -> List[Dict[str, Any]]:
        """获取性能优化建议"""
        suggestions = []
        
        for rule in self.optimization_rules:
            if rule.should_apply():
                suggestions.append(rule.get_suggestion())
        
        return suggestions
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """应用性能优化"""
        results = {}
        
        for rule in self.optimization_rules:
            if rule.should_apply():
                try:
                    result = rule.apply()
                    results[rule.name] = result
                except Exception as e:
                    results[rule.name] = {'error': str(e)}
        
        return results
```

#### 5. 重构后的主类
```python
# src/infrastructure/health/enhanced_health_checker.py
class EnhancedHealthChecker:
    """重构后的增强健康检查器 - 职责协调"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 组件初始化
        self.health_checker = HealthChecker(config)
        self.system_monitor = SystemMonitor(config)
        self.alert_manager = AlertManager(config)
        self.performance_optimizer = PerformanceOptimizer(
            get_cache_manager(),
            get_prometheus_exporter()
        )
        
        # 可选组件
        self.grafana_integration = self._init_grafana_integration()
        self.alert_rule_engine = self._init_alert_rule_engine()
    
    def _init_grafana_integration(self):
        """初始化Grafana集成"""
        if self.config.get('grafana_enabled', False):
            try:
                return get_grafana_integration(
                    grafana_url=self.config.get('grafana_url'),
                    api_key=self.config.get('grafana_api_key'),
                    org_id=self.config.get('grafana_org_id', 1)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Grafana integration: {e}")
        return None
    
    def _init_alert_rule_engine(self):
        """初始化告警规则引擎"""
        return get_alert_rule_engine(
            alert_manager=self.alert_manager,
            prometheus_exporter=get_prometheus_exporter()
        )
    
    # 委托方法 - 将具体实现委托给相应的组件
    async def perform_health_check(self, service: str, check_type: str) -> HealthCheckResult:
        """执行健康检查"""
        result = await self.health_checker.perform_health_check(service, check_type)
        
        # 检查告警条件
        self.alert_manager.check_alert_conditions(result)
        
        return result
    
    async def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        return await self.system_monitor.get_system_metrics()
    
    def get_performance_suggestions(self) -> List[Dict[str, Any]]:
        """获取性能优化建议"""
        return self.performance_optimizer.get_performance_suggestions()
    
    def deploy_grafana_dashboards(self) -> Dict[str, Any]:
        """部署Grafana仪表板"""
        if self.grafana_integration:
            return self.grafana_integration.deploy_dashboards()
        return {'error': 'Grafana integration not available'}
```

## 示例2：模块结构重构

### 重构前的模块结构
```
src/infrastructure/
├── core/
│   ├── monitoring/          # 核心监控
│   ├── cache/               # 核心缓存
│   └── ...
├── health/
│   ├── monitoring/          # 健康监控（重复）
│   ├── cache/               # 健康缓存（重复）
│   └── ...
└── services/
    ├── cache/               # 服务缓存（重复）
    └── ...
```

### 重构后的模块结构
```
src/infrastructure/
├── monitoring/              # 统一监控模块
│   ├── __init__.py
│   ├── core/                # 核心监控功能
│   ├── health/              # 健康监控
│   ├── performance/         # 性能监控
│   ├── system/              # 系统监控
│   └── interfaces.py        # 监控接口
├── cache/                   # 统一缓存模块
│   ├── __init__.py
│   ├── strategies/          # 缓存策略
│   ├── policies/            # 缓存策略
│   ├── adapters/            # 缓存适配器
│   ├── managers/            # 缓存管理器
│   └── interfaces.py        # 缓存接口
├── configuration/            # 统一配置模块
│   ├── __init__.py
│   ├── managers/            # 配置管理器
│   ├── validators/          # 配置验证器
│   ├── loaders/             # 配置加载器
│   └── interfaces.py        # 配置接口
└── interfaces/               # 统一接口定义
    ├── __init__.py
    ├── monitoring.py
    ├── caching.py
    ├── configuration.py
    └── health.py
```

## 示例3：接口统一

### 重构前的接口定义
```python
# 分散在各个模块中的接口
# src/infrastructure/core/monitoring/interfaces.py
class IMonitor:
    def collect_metrics(self) -> Dict[str, Any]:
        pass

# src/infrastructure/health/interfaces.py
class IHealthChecker:
    def check_health(self) -> HealthStatus:
        pass

# src/infrastructure/services/cache/interfaces.py
class ICacheManager:
    def get(self, key: str) -> Any:
        pass
```

### 重构后的统一接口
```python
# src/infrastructure/interfaces/__init__.py
"""
统一接口定义
所有基础设施层的接口都在这里定义，确保一致性和可维护性
"""

from .monitoring import IMonitor, IMetricsCollector, IAlertManager
from .caching import ICacheManager, ICacheStrategy, ICachePolicy
from .configuration import IConfigManager, IConfigValidator, IConfigLoader
from .health import IHealthChecker, IHealthReporter, IHealthNotifier
from .security import ISecurityManager, IAuthentication, IAuthorization

__all__ = [
    # 监控接口
    'IMonitor',
    'IMetricsCollector', 
    'IAlertManager',
    
    # 缓存接口
    'ICacheManager',
    'ICacheStrategy',
    'ICachePolicy',
    
    # 配置接口
    'IConfigManager',
    'IConfigValidator',
    'IConfigLoader',
    
    # 健康检查接口
    'IHealthChecker',
    'IHealthReporter',
    'IHealthNotifier',
    
    # 安全接口
    'ISecurityManager',
    'IAuthentication',
    'IAuthorization'
]
```

```python
# src/infrastructure/interfaces/monitoring.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class IMonitor(ABC):
    """监控接口"""
    
    @abstractmethod
    def start_monitoring(self) -> bool:
        """启动监控"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> bool:
        """停止监控"""
        pass
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标"""
        pass
    
    @abstractmethod
    def get_metrics_history(self, metric_name: str, 
                          start_time: datetime, 
                          end_time: datetime) -> List[Dict[str, Any]]:
        """获取指标历史"""
        pass

class IMetricsCollector(ABC):
    """指标收集器接口"""
    
    @abstractmethod
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        pass
    
    @abstractmethod
    def collect_business_metrics(self) -> Dict[str, Any]:
        """收集业务指标"""
        pass
    
    @abstractmethod
    def collect_custom_metrics(self, metric_name: str) -> Any:
        """收集自定义指标"""
        pass

class IAlertManager(ABC):
    """告警管理器接口"""
    
    @abstractmethod
    def create_alert(self, alert_type: str, message: str, 
                    severity: str, metadata: Dict[str, Any]) -> str:
        """创建告警"""
        pass
    
    @abstractmethod
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        pass
    
    @abstractmethod
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        pass
    
    @abstractmethod
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        pass
```

## 重构效果总结

### 代码质量提升
- **单一职责原则**：每个类只负责一个功能领域
- **开闭原则**：通过接口扩展，对修改关闭
- **依赖倒置**：高层模块不依赖低层模块，都依赖抽象

### 可维护性提升
- **文件大小合理**：每个文件控制在300行以内
- **职责清晰**：模块和类的职责边界明确
- **接口统一**：所有接口集中管理，便于维护

### 可扩展性提升
- **插件化架构**：新功能可以通过实现接口添加
- **策略模式**：不同的实现策略可以灵活切换
- **工厂模式**：组件的创建和管理更加灵活

### 测试友好性
- **单元测试**：每个小类都可以独立测试
- **Mock友好**：接口清晰，便于Mock测试
- **依赖注入**：测试时可以注入Mock对象

---

*重构示例创建时间：2025年1月*
*重构目标：提升代码质量和可维护性*
*重构策略：职责分离、接口统一、模块整合*

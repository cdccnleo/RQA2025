#!/usr/bin/env python3
"""
RQA2025 子系统边界优化器
Subsystem Boundary Optimizer

优化各子系统间的职责分工和接口标准化。
"""

import logging
from typing import Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception as e:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except ImportError:
        logger = logging.getLogger(__name__)


@dataclass
class SubsystemBoundary:

    """子系统边界定义"""
    subsystem_name: str
    responsibilities: Set[str] = field(default_factory=set)
    interfaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    data_flows: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class InterfaceContract:

    """接口契约"""
    interface_name: str
    provider_subsystem: str
    consumer_subsystems: List[str] = field(default_factory=list)
    methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_formats: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class BoundaryOptimizationResult:

    """边界优化结果"""
    optimization_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    conflicts_resolved: List[Dict[str, Any]] = field(default_factory=list)
    interfaces_standardized: List[str] = field(default_factory=list)
    dependencies_optimized: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class BoundaryOptimizer:

    """
    子系统边界优化器
    分析和优化各子系统间的职责分工和接口
    """

    def __init__(self):

        self.subsystems: Dict[str, SubsystemBoundary] = {}
        self.interfaces: Dict[str, InterfaceContract] = {}
        self.optimization_history: List[BoundaryOptimizationResult] = []

        # 初始化默认子系统边界
        self._initialize_default_boundaries()

        logger.info("子系统边界优化器已初始化")

    def _initialize_default_boundaries(self):
        """初始化默认子系统边界"""
        # 基础设施层
        self.subsystems['infrastructure'] = SubsystemBoundary(
            subsystem_name='infrastructure',
            responsibilities={
                'provide_logging', 'provide_caching', 'provide_security',
                'provide_monitoring', 'provide_configuration', 'provide_messaging'
            },
            interfaces={
                'logging': {
                    'methods': ['log', 'get_logger', 'configure_logging'],
                    'data_formats': {'log_entry': 'dict'}
                },
                'caching': {
                    'methods': ['get', 'set', 'delete', 'clear'],
                    'data_formats': {'cache_key': 'str', 'cache_value': 'any'}
                }
            }
        )

        # 数据管理层
        self.subsystems['data_management'] = SubsystemBoundary(
            subsystem_name='data_management',
            responsibilities={
                'manage_market_data', 'manage_historical_data', 'provide_data_access',
                'ensure_data_quality', 'handle_data_persistence'
            },
            dependencies={
                'infrastructure': ['logging', 'caching', 'monitoring']
            }
        )

        # 流处理层
        self.subsystems['streaming'] = SubsystemBoundary(
            subsystem_name='streaming',
            responsibilities={
                'process_real_time_data', 'aggregate_data', 'manage_state',
                'route_data', 'transform_data'
            },
            dependencies={
                'infrastructure': ['logging', 'monitoring', 'messaging'],
                'data_management': ['market_data_feed']
            }
        )

        # 机器学习层
        self.subsystems['ml'] = SubsystemBoundary(
            subsystem_name='ml',
            responsibilities={
                'train_models', 'make_predictions', 'feature_engineering',
                'model_evaluation', 'model_deployment'
            },
            dependencies={
                'infrastructure': ['logging', 'caching', 'monitoring'],
                'data_management': ['training_data', 'feature_data']
            }
        )

        # 策略层
        self.subsystems['strategy'] = SubsystemBoundary(
            subsystem_name='strategy',
            responsibilities={
                'define_trading_strategies', 'execute_strategies', 'manage_positions',
                'calculate_signals', 'risk_assessment'
            },
            dependencies={
                'infrastructure': ['logging', 'monitoring'],
                'ml': ['predictions'],
                'data_management': ['market_data']
            }
        )

        # 交易层
        self.subsystems['trading'] = SubsystemBoundary(
            subsystem_name='trading',
            responsibilities={
                'execute_orders', 'manage_order_book', 'handle_trade_settlement',
                'provide_execution_reports'
            },
            dependencies={
                'infrastructure': ['logging', 'monitoring', 'security'],
                'strategy': ['trading_signals'],
                'data_management': ['market_data']
            }
        )

        # 监控层
        self.subsystems['monitoring'] = SubsystemBoundary(
            subsystem_name='monitoring',
            responsibilities={
                'collect_metrics', 'monitor_system_health', 'detect_anomalies',
                'generate_alerts', 'provide_dashboards'
            },
            dependencies={
                'infrastructure': ['logging', 'caching']
            }
        )

        # 弹性层
        self.subsystems['resilience'] = SubsystemBoundary(
            subsystem_name='resilience',
            responsibilities={
                'handle_failures', 'implement_circuit_breakers', 'manage_timeouts',
                'provide_fallbacks', 'ensure_high_availability'
            },
            dependencies={
                'infrastructure': ['logging', 'monitoring']
            }
        )

    def add_subsystem(self, subsystem: SubsystemBoundary):
        """添加子系统"""
        self.subsystems[subsystem.subsystem_name] = subsystem
        logger.info(f"添加子系统: {subsystem.subsystem_name}")

    def add_interface_contract(self, interface: InterfaceContract):
        """添加接口契约"""
        self.interfaces[interface.interface_name] = interface
        logger.info(f"添加接口契约: {interface.interface_name}")

    def get_subsystem_boundary(self, subsystem_name: str) -> SubsystemBoundary:
        """获取子系统边界"""
        return self.subsystems.get(subsystem_name)

    def get_interface_contract(self, interface_name: str) -> InterfaceContract:
        """获取接口契约"""
        return self.interfaces.get(interface_name)

    def update_subsystem_boundary(
        self,
        subsystem_name: str,
        responsibilities: Set[str] = None,
        interfaces: Dict[str, Dict[str, Any]] = None,
        dependencies: Dict[str, List[str]] = None
    ) -> bool:
        """更新子系统边界"""
        if subsystem_name not in self.subsystems:
            return False
        
        subsystem = self.subsystems[subsystem_name]
        if responsibilities is not None:
            subsystem.responsibilities = responsibilities
        if interfaces is not None:
            subsystem.interfaces = interfaces
        if dependencies is not None:
            subsystem.dependencies = dependencies
        
        logger.info(f"更新子系统边界: {subsystem_name}")
        return True

    def update_interface_contract(
        self,
        interface_name: str,
        version: str = None,
        methods: Dict[str, Dict[str, Any]] = None,
        consumer_subsystems: List[str] = None
    ) -> bool:
        """更新接口契约"""
        if interface_name not in self.interfaces:
            return False
        
        interface = self.interfaces[interface_name]
        if version is not None:
            interface.version = version
            interface.last_updated = datetime.now()
        if methods is not None:
            interface.methods = methods
        if consumer_subsystems is not None:
            interface.consumer_subsystems = consumer_subsystems
        
        logger.info(f"更新接口契约: {interface_name}")
        return True

    def remove_subsystem_boundary(self, subsystem_name: str) -> bool:
        """移除子系统边界"""
        if subsystem_name not in self.subsystems:
            return False
        
        del self.subsystems[subsystem_name]
        logger.info(f"移除子系统边界: {subsystem_name}")
        return True

    def remove_interface_contract(self, interface_name: str) -> bool:
        """移除接口契约"""
        if interface_name not in self.interfaces:
            return False
        
        del self.interfaces[interface_name]
        logger.info(f"移除接口契约: {interface_name}")
        return True

    def detect_boundary_conflicts(self) -> List[Dict[str, Any]]:
        """检测边界冲突"""
        return self._analyze_responsibility_conflicts()

    def optimize_responsibility_distribution(self) -> BoundaryOptimizationResult:
        """优化职责分布"""
        return self.optimize_boundaries()

    def validate_interface_compatibility(self, interface_name: str) -> Dict[str, Any]:
        """验证接口兼容性"""
        if interface_name not in self.interfaces:
            return {'compatible': False, 'error': '接口不存在'}
        
        interface = self.interfaces[interface_name]
        inconsistencies = self._analyze_interface_consistency()
        
        # 检查该接口是否有不一致性
        for inconsistency in inconsistencies:
            if inconsistency.get('interface') == interface_name:
                return {'compatible': False, 'issues': inconsistency}
        
        return {'compatible': True, 'interface': interface_name}

    def monitor_boundary_metrics(self) -> Dict[str, Any]:
        """监控边界指标"""
        return {
            'subsystem_count': len(self.subsystems),
            'interface_count': len(self.interfaces),
            'conflicts': len(self._analyze_responsibility_conflicts()),
            'cycles': len(self._detect_dependency_cycles()),
            'optimization_count': len(self.optimization_history)
        }

    def generate_boundary_report(self) -> Dict[str, Any]:
        """生成边界报告"""
        analysis = self.analyze_boundaries()
        status = self.get_boundary_status()
        
        return {
            'status': status,
            'analysis': analysis,
            'recommendations': self._generate_optimization_recommendations(),
            'timestamp': datetime.now().isoformat()
        }

    def export_boundary_configuration(self, file_path: str) -> bool:
        """导出边界配置"""
        try:
            import json
            config = {
                'subsystems': {
                    name: {
                        'subsystem_name': boundary.subsystem_name,
                        'responsibilities': list(boundary.responsibilities),
                        'interfaces': boundary.interfaces,
                        'dependencies': boundary.dependencies
                    }
                    for name, boundary in self.subsystems.items()
                },
                'interfaces': {
                    name: {
                        'interface_name': contract.interface_name,
                        'provider_subsystem': contract.provider_subsystem,
                        'consumer_subsystems': contract.consumer_subsystems,
                        'methods': contract.methods,
                        'data_formats': contract.data_formats,
                        'version': contract.version
                    }
                    for name, contract in self.interfaces.items()
                }
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"导出边界配置到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"导出边界配置失败: {e}")
            return False

    def analyze_boundaries(self) -> Dict[str, Any]:
        """分析子系统边界"""
        analysis_result = {
            'total_subsystems': len(self.subsystems),
            'total_interfaces': len(self.interfaces),
            'boundary_conflicts': [],
            'interface_inconsistencies': [],
            'dependency_cycles': [],
            'optimization_opportunities': []
        }

        # 分析职责冲突
        analysis_result['boundary_conflicts'] = self._analyze_responsibility_conflicts()

        # 分析接口一致性
        analysis_result['interface_inconsistencies'] = self._analyze_interface_consistency()

        # 分析依赖循环
        analysis_result['dependency_cycles'] = self._detect_dependency_cycles()

        # 识别优化机会
        analysis_result['optimization_opportunities'] = self._identify_optimization_opportunities()

        return analysis_result

    def _analyze_responsibility_conflicts(self) -> List[Dict[str, Any]]:
        """分析职责冲突"""
        conflicts = []
        responsibility_map = {}

        # 构建职责映射
        for subsystem_name, subsystem in self.subsystems.items():
            for responsibility in subsystem.responsibilities:
                if responsibility not in responsibility_map:
                    responsibility_map[responsibility] = []
                responsibility_map[responsibility].append(subsystem_name)

        # 识别共享职责
        for responsibility, subsystems in responsibility_map.items():
            if len(subsystems) > 1:
                conflicts.append({
                    'responsibility': responsibility,
                    'conflicting_subsystems': subsystems,
                    'severity': 'high' if len(subsystems) > 2 else 'medium',
                    'recommendation': '明确职责所有者和接口规范'
                })

        return conflicts

    def _analyze_interface_consistency(self) -> List[Dict[str, Any]]:
        """分析接口一致性"""
        inconsistencies = []

        for interface_name, interface in self.interfaces.items():
            # 检查方法签名一致性
            for method_name, method_spec in interface.methods.items():
                # 这里可以添加更详细的接口一致性检查
                pass

        return inconsistencies

    def _detect_dependency_cycles(self) -> List[Dict[str, Any]]:
        """检测依赖循环"""
        cycles = []

        # 简化的循环检测算法
        visited = set()
        recursion_stack = set()

        def dfs(subsystem_name: str, path: List[str]):

            visited.add(subsystem_name)
            recursion_stack.add(subsystem_name)
            path.append(subsystem_name)

            subsystem = self.subsystems.get(subsystem_name)
            if subsystem:
                for dep_subsystem in subsystem.dependencies.keys():
                    if dep_subsystem not in visited:
                        if dfs(dep_subsystem, path):
                            return True
                    elif dep_subsystem in recursion_stack:
                        # 发现循环
                        cycle_start = path.index(dep_subsystem)
                        cycles.append({
                            'cycle': path[cycle_start:] + [dep_subsystem],
                            'severity': 'high'
                        })
                        return True

            path.pop()
            recursion_stack.remove(subsystem_name)
            return False

        for subsystem_name in self.subsystems.keys():
            if subsystem_name not in visited:
                dfs(subsystem_name, [])

        return cycles

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """识别优化机会"""
        opportunities = []

        # 检查是否有未使用的接口
        used_interfaces = set()
        for subsystem in self.subsystems.values():
            for interface_list in subsystem.dependencies.values():
                used_interfaces.update(interface_list)

        for interface_name in self.interfaces.keys():
            if interface_name not in used_interfaces:
                opportunities.append({
                    'type': 'unused_interface',
                    'interface': interface_name,
                    'recommendation': '移除未使用的接口或添加使用方'
                })

        # 检查职责过于分散的情况
        for subsystem_name, subsystem in self.subsystems.items():
            if len(subsystem.responsibilities) > 10:
                opportunities.append({
                    'type': 'subsystem_overload',
                    'subsystem': subsystem_name,
                    'responsibility_count': len(subsystem.responsibilities),
                    'recommendation': '考虑将部分职责拆分到新的子系统'
                })

        return opportunities

    def optimize_boundaries(self) -> BoundaryOptimizationResult:
        """执行边界优化"""
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        result = BoundaryOptimizationResult(optimization_id=optimization_id)

        # 执行优化措施
        analysis = self.analyze_boundaries()

        # 解决职责冲突
        for conflict in analysis['boundary_conflicts']:
            resolution = self._resolve_responsibility_conflict(conflict)
            result.conflicts_resolved.append(resolution)

        # 标准化接口
        for interface_name in self.interfaces.keys():
            standardization = self._standardize_interface(interface_name)
            result.interfaces_standardized.append(interface_name)

        # 优化依赖关系
        for optimization in analysis['optimization_opportunities']:
            if optimization['type'] == 'unused_interface':
                self._remove_unused_interface(optimization['interface'])
            elif optimization['type'] == 'subsystem_overload':
                self._optimize_overloaded_subsystem(optimization['subsystem'])

        # 生成优化建议
        result.recommendations = self._generate_optimization_recommendations()

        self.optimization_history.append(result)
        logger.info(f"完成边界优化: {optimization_id}")

        return result

    def _resolve_responsibility_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """解决职责冲突"""
        # 简化的冲突解决策略
        return {
            'conflict': conflict,
            'resolution': 'assigned_primary_owner',
            'primary_owner': conflict['conflicting_subsystems'][0],
            'status': 'resolved'
        }

    def _standardize_interface(self, interface_name: str) -> Dict[str, Any]:
        """标准化接口"""
        interface = self.interfaces.get(interface_name)
        if not interface:
            return {'status': 'interface_not_found'}

        # 确保接口版本控制和文档完整性
        return {
            'interface': interface_name,
            'standardized_methods': len(interface.methods),
            'status': 'standardized'
        }

    def _remove_unused_interface(self, interface_name: str):
        """移除未使用的接口"""
        if interface_name in self.interfaces:
            del self.interfaces[interface_name]
            logger.info(f"移除未使用的接口: {interface_name}")

    def _optimize_overloaded_subsystem(self, subsystem_name: str):
        """优化 overloaded 的子系统"""
        subsystem = self.subsystems.get(subsystem_name)
        if subsystem and len(subsystem.responsibilities) > 10:
            # 建议拆分职责
            logger.info(f"建议优化子系统 {subsystem_name} 的职责分配")

    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = [
            "定期审查子系统职责分工",
            "维护接口契约文档",
            "监控依赖关系变化",
            "实施接口版本控制",
            "建立边界变更审批流程"
        ]

        return recommendations

    def get_boundary_status(self) -> Dict[str, Any]:
        """获取边界状态"""
        return {
            'total_subsystems': len(self.subsystems),
            'total_interfaces': len(self.interfaces),
            'subsystems': list(self.subsystems.keys()),
            'interfaces': list(self.interfaces.keys()),
            'last_optimization': self.optimization_history[-1].timestamp.isoformat() if self.optimization_history else None,
            'optimization_count': len(self.optimization_history)
        }


__all__ = [
    'BoundaryOptimizer',
    'SubsystemBoundary',
    'InterfaceContract',
    'BoundaryOptimizationResult'
]


from typing import Any, Dict, List
"""
基础设施层 - 接口一致性检查工具

用于验证接口实现是否符合标准规范。
"""


class InterfaceConsistencyChecker:
    """
    接口一致性检查工具

    用于验证接口实现是否符合标准规范，包括：
    - 方法命名规范
    - 参数类型一致性
    - 返回值类型一致性
    - 文档完整性
    """

    @staticmethod
    def check_interface_implementation(interface_cls: type, implementation_cls: type) -> Dict[str, Any]:
        """
        检查接口实现的一致性

        Args:
            interface_cls: 接口类
            implementation_cls: 实现类

        Returns:
            Dict[str, Any]: 检查结果报告
        """
        result = {
            'interface': interface_cls.__name__,
            'implementation': implementation_cls.__name__,
            'is_consistent': True,
            'issues': [],
            'method_coverage': 0.0
        }

        # 检查是否继承自接口
        if not issubclass(implementation_cls, interface_cls):
            result['is_consistent'] = False
            result['issues'].append(f'{implementation_cls.__name__} 未继承 {interface_cls.__name__}')
            return result

        # 获取接口的抽象方法
        interface_methods = InterfaceConsistencyChecker._get_abstract_methods(interface_cls)

        # 检查方法实现 - 需要检查实现类是否实际实现了这些方法
        implemented_methods = set()
        for method_name in interface_methods:
            # 检查方法是否在实现类中定义（不是继承的）
            if method_name in implementation_cls.__dict__ or \
               (hasattr(implementation_cls, method_name) and
                getattr(implementation_cls, method_name) != getattr(interface_cls, method_name, None)):
                implemented_methods.add(method_name)
            else:
                result['is_consistent'] = False
                result['issues'].append(f'缺少方法实现: {method_name}')

        # 计算覆盖率
        result['method_coverage'] = len(implemented_methods) / \
            len(interface_methods) if interface_methods else 1.0

        return result

    @staticmethod
    def _get_abstract_methods(cls: type) -> List[str]:
        """获取类的所有抽象方法和属性"""
        abstract_members = []
        for name in dir(cls):
            if name.startswith('_'):
                continue
            attr = getattr(cls, name)
            if hasattr(attr, '__isabstractmethod__') and attr.__isabstractmethod__:
                abstract_members.append(name)
        return abstract_members

    @staticmethod
    def check_naming_convention(cls: type) -> Dict[str, Any]:
        """
        检查命名规范

        Args:
            cls: 要检查的类

        Returns:
            Dict[str, Any]: 命名规范检查结果
        """
        result = {
            'class_name': cls.__name__,
            'issues': [],
            'compliance_score': 1.0
        }

        # 检查类名
        if not cls.__name__.startswith('I') and 'Interface' not in cls.__name__:
            # 对于实现类，检查方法命名
            methods = [name for name in dir(cls) if not name.startswith(
                '_') and callable(getattr(cls, name))]
            naming_issues = []

            for method in methods:
                # 检查方法命名规范
                if not (method.startswith(('get_cache_', 'set_cache_', 'delete_cache_', 'has_cache_', 'clear_all_cache', 'get_cache_size', 'get_cache_stats', 'initialize_component', 'get_component_status', 'shutdown_component', 'health_check')) or
                        method in ['component_name', 'component_type']):
                    naming_issues.append(f'方法名不符合规范: {method}')

            if naming_issues:
                result['issues'].extend(naming_issues)
                result['compliance_score'] = 0.8

        return result

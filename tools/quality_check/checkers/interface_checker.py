"""
接口一致性检查器

检查接口实现是否与接口定义一致。
"""

import ast
import inspect
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import importlib.util

from ..core.base_checker import BaseChecker
from ..core.check_result import IssueSeverity


class InterfaceDefinition:
    """接口定义"""

    def __init__(self, name: str, methods: Dict[str, Dict[str, Any]]):
        self.name = name
        self.methods = methods

    def get_method_signature(self, method_name: str) -> Optional[Dict[str, Any]]:
        """获取方法签名"""
        return self.methods.get(method_name)


class InterfaceConsistencyChecker(BaseChecker):
    """
    接口一致性检查器

    检查类是否正确实现了其声明实现的接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _setup_default_config(self) -> None:
        """设置默认配置"""
        defaults = {
            'check_abstract_methods': True,  # 检查抽象方法实现
            'check_method_signatures': True,  # 检查方法签名
            'check_property_implementations': True,  # 检查属性实现
            'allow_extra_methods': True,  # 允许额外方法
            'strict_mode': False,  # 严格模式
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def checker_name(self) -> str:
        return "interface_consistency_checker"

    @property
    def checker_description(self) -> str:
        return "检查接口实现是否与接口定义一致"

    def check(self, target_path: str) -> 'CheckResult':
        """
        执行接口一致性检查

        Args:
            target_path: 检查目标路径

        Returns:
            CheckResult: 检查结果
        """
        result = self._create_result()

        try:
            # 收集Python文件
            python_files = self._collect_python_files(target_path)

            if not python_files:
                result.metadata['message'] = "未找到Python文件"
                result.set_end_time()
                return result

            # 分析每个文件
            total_classes = 0
            interfaces_found = 0

            for file_path in python_files:
                classes_info = self._analyze_file(file_path)
                total_classes += len(classes_info)

                for class_info in classes_info:
                    self._check_class_interfaces(result, class_info)
                    interfaces_found += len(class_info.get('interfaces', []))

            # 设置元数据
            result.metadata.update({
                'total_files': len(python_files),
                'total_classes': total_classes,
                'interfaces_found': interfaces_found
            })

        except Exception as e:
            self.logger.error(f"接口一致性检查失败: {e}")
            result.add_issue(self._create_issue(
                file_path=target_path,
                message=f"接口一致性检查失败: {e}",
                severity=IssueSeverity.ERROR,
                rule_id="INTERFACE_CHECK_FAILED"
            ))

        result.set_end_time()
        return result

    def _analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        分析Python文件中的类和接口

        Args:
            file_path: 文件路径

        Returns:
            List[Dict[str, Any]]: 类信息列表
        """
        classes_info = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, file_path)
                    if class_info:
                        classes_info.append(class_info)

        except Exception as e:
            self.logger.warning(f"分析文件失败 {file_path}: {e}")

        return classes_info

    def _analyze_class(self, node: ast.ClassDef, file_path: str) -> Optional[Dict[str, Any]]:
        """
        分析类定义

        Args:
            node: 类节点
            file_path: 文件路径

        Returns:
            Optional[Dict[str, Any]]: 类信息
        """
        class_name = node.name
        bases = []

        # 解析基类
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                # 处理 like module.Class
                bases.append(self._get_full_name(base))

        # 查找实现的接口（通过注释或特殊标记）
        interfaces = self._find_implemented_interfaces(node)

        # 分析方法
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_method(item)
                methods[item.name] = method_info

        return {
            'name': class_name,
            'file_path': file_path,
            'line_number': node.lineno,
            'bases': bases,
            'interfaces': interfaces,
            'methods': methods,
            'node': node
        }

    def _get_full_name(self, node: ast.AST) -> str:
        """获取AST节点的完整名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_full_name(node.value)}.{node.attr}"
        return str(node)

    def _find_implemented_interfaces(self, node: ast.ClassDef) -> List[str]:
        """
        查找类实现的接口

        通过docstring或特殊注释识别接口实现。
        """
        interfaces = []

        # 检查docstring
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            docstring = node.body[0].value.s
            # 查找 "implements" 或 "接口" 关键字
            if "implements" in docstring.lower() or "接口" in docstring:
                # 简单的接口提取逻辑
                lines = docstring.split('\n')
                for line in lines:
                    line = line.strip()
                    if "implements" in line.lower() or "接口" in line:
                        # 提取接口名称（简化逻辑）
                        pass

        # 检查基类中的接口
        for base in node.bases:
            base_name = self._get_full_name(base)
            # 简单的启发式：以"I"开头的类可能是接口
            if base_name.startswith('I') and len(base_name) > 1 and base_name[1].isupper():
                interfaces.append(base_name)

        return interfaces

    def _analyze_method(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        分析方法定义

        Args:
            node: 方法节点

        Returns:
            Dict[str, Any]: 方法信息
        """
        # 解析参数
        args = []
        if node.args.args:
            for arg in node.args.args:
                args.append(arg.arg)

        # 检查是否有返回值注解
        has_return_annotation = node.returns is not None

        # 检查是否是抽象方法
        is_abstract = any(
            isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod'
            for decorator in node.decorator_list
        )

        # 检查是否是属性
        is_property = any(
            isinstance(decorator, ast.Name) and decorator.id == 'property'
            for decorator in node.decorator_list
        )

        return {
            'name': node.name,
            'args': args,
            'has_return_annotation': has_return_annotation,
            'is_abstract': is_abstract,
            'is_property': is_property,
            'line_number': node.lineno
        }

    def _check_class_interfaces(self, result: 'CheckResult', class_info: Dict[str, Any]) -> None:
        """
        检查类的接口实现

        Args:
            result: 检查结果
            class_info: 类信息
        """
        interfaces = class_info.get('interfaces', [])
        if not interfaces:
            return

        # 对于每个接口，检查实现
        for interface_name in interfaces:
            try:
                interface_def = self._load_interface_definition(
                    interface_name, class_info['file_path'])
                if interface_def:
                    self._check_single_interface(result, class_info, interface_def)
                else:
                    # 无法加载接口定义
                    result.add_issue(self._create_issue(
                        file_path=class_info['file_path'],
                        message=f"无法加载接口定义: {interface_name}",
                        severity=IssueSeverity.WARNING,
                        rule_id="INTERFACE_NOT_FOUND",
                        line_number=class_info['line_number'],
                        details={'interface': interface_name}
                    ))
            except Exception as e:
                self.logger.warning(f"检查接口失败 {interface_name}: {e}")

    def _load_interface_definition(self, interface_name: str, file_path: str) -> Optional[InterfaceDefinition]:
        """
        加载接口定义

        Args:
            interface_name: 接口名称
            file_path: 当前文件路径

        Returns:
            Optional[InterfaceDefinition]: 接口定义
        """
        try:
            # 首先尝试从当前文件中查找接口定义
            current_file_methods = self._find_interface_in_current_file(interface_name, file_path)
            if current_file_methods:
                return InterfaceDefinition(interface_name, current_file_methods)

            # 尝试从当前文件或相关文件中加载接口
            # 这是一个简化的实现，实际应该使用更复杂的模块加载逻辑

            # 首先尝试从标准库或已知位置加载
            if '.' in interface_name:
                module_name, class_name = interface_name.rsplit('.', 1)
                try:
                    module = importlib.import_module(module_name)
                    interface_class = getattr(module, class_name)

                    # 分析接口类
                    methods = self._analyze_interface_class(interface_class)
                    return InterfaceDefinition(interface_name, methods)
                except (ImportError, AttributeError):
                    pass

            # 如果找不到，尝试从当前项目的接口文件中查找
            interface_methods = self._find_interface_in_project(interface_name, file_path)
            if interface_methods:
                return InterfaceDefinition(interface_name, interface_methods)

        except Exception as e:
            self.logger.warning(f"加载接口定义失败 {interface_name}: {e}")

        return None

    def _find_interface_in_current_file(self, interface_name: str, file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        从当前文件中查找接口定义

        Args:
            interface_name: 接口名称
            file_path: 文件路径

        Returns:
            Optional[Dict[str, Dict[str, Any]]]: 接口方法定义
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            # 查找接口类定义
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == interface_name:
                    # 分析接口类的方法
                    methods = {}
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = self._analyze_method(item)
                            methods[item.name] = method_info
                        elif isinstance(item, ast.AsyncFunctionDef):
                            method_info = self._analyze_method(item)
                            methods[item.name] = method_info

                    return methods

        except Exception as e:
            self.logger.warning(f"从当前文件查找接口失败 {interface_name} in {file_path}: {e}")

        return None

    def _analyze_interface_class(self, interface_class: Type) -> Dict[str, Dict[str, Any]]:
        """
        分析接口类定义

        Args:
            interface_class: 接口类

        Returns:
            Dict[str, Dict[str, Any]]: 方法信息
        """
        methods = {}

        # 获取所有抽象方法和属性
        for name, member in inspect.getmembers(interface_class):
            if not name.startswith('_'):
                if inspect.ismethod(member) or inspect.isfunction(member):
                    # 方法
                    sig = inspect.signature(member)
                    methods[name] = {
                        'type': 'method',
                        'signature': str(sig),
                        'parameters': list(sig.parameters.keys())
                    }
                elif isinstance(member, property):
                    # 属性
                    methods[name] = {
                        'type': 'property'
                    }

        return methods

    def _find_interface_in_project(self, interface_name: str, file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        在项目中查找接口定义

        Args:
            interface_name: 接口名称
            file_path: 当前文件路径

        Returns:
            Optional[Dict[str, Dict[str, Any]]]: 接口方法
        """
        # 简化的实现：查找项目中的接口文件
        # 实际应该使用更复杂的搜索逻辑

        # 假设接口在interfaces目录中
        interfaces_dir = Path(file_path).parent.parent / "interfaces"

        if interfaces_dir.exists():
            for py_file in interfaces_dir.glob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location("temp_module", py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        if hasattr(module, interface_name):
                            interface_class = getattr(module, interface_name)
                            return self._analyze_interface_class(interface_class)
                except Exception:
                    continue

        return None

    def _check_single_interface(self, result: 'CheckResult',
                                class_info: Dict[str, Any],
                                interface_def: InterfaceDefinition) -> None:
        """
        检查单个接口的实现

        Args:
            result: 检查结果
            class_info: 类信息
            interface_def: 接口定义
        """
        class_methods = class_info.get('methods', {})
        interface_name = interface_def.name

        # 检查每个接口方法是否被实现
        for method_name, method_info in interface_def.methods.items():
            if method_name not in class_methods:
                # 方法未实现
                result.add_issue(self._create_issue(
                    file_path=class_info['file_path'],
                    message=f"类 {class_info['name']} 未实现接口 {interface_name} 的方法 {method_name}",
                    severity=IssueSeverity.ERROR,
                    rule_id="MISSING_INTERFACE_METHOD",
                    line_number=class_info['line_number'],
                    details={
                        'class': class_info['name'],
                        'interface': interface_name,
                        'method': method_name,
                        'method_type': method_info.get('type')
                    }
                ))
            else:
                # 方法已实现，检查签名
                if self.config.get('check_method_signatures', True):
                    self._check_method_signature(
                        result, class_info, interface_def, method_name
                    )

    def _check_method_signature(self, result: 'CheckResult',
                                class_info: Dict[str, Any],
                                interface_def: InterfaceDefinition,
                                method_name: str) -> None:
        """
        检查方法签名

        Args:
            result: 检查结果
            class_info: 类信息
            interface_def: 接口定义
            method_name: 方法名
        """
        class_method = class_info['methods'][method_name]
        interface_method = interface_def.get_method_signature(method_name)

        if not interface_method:
            return

        # 检查参数数量（简化检查）
        interface_params = interface_method.get('parameters', [])
        class_params = class_method.get('args', [])

        # 跳过self参数比较
        interface_params = [p for p in interface_params if p != 'self']
        class_params = [p for p in class_params if p != 'self']

        if len(interface_params) != len(class_params):
            result.add_issue(self._create_issue(
                file_path=class_info['file_path'],
                message=f"方法 {method_name} 参数数量不匹配接口定义",
                severity=IssueSeverity.WARNING,
                rule_id="METHOD_SIGNATURE_MISMATCH",
                line_number=class_method['line_number'],
                details={
                    'class': class_info['name'],
                    'method': method_name,
                    'interface_params': len(interface_params),
                    'class_params': len(class_params)
                }
            ))

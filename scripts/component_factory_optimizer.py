#!/usr/bin/env python3
"""
组件工厂优化工具

完善组件工厂设计：
1. 统一组件工厂标准结构
2. 优化接口定义和抽象层次
3. 增强错误处理机制
4. 完善组件生命周期管理
"""

import os
import re
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any


class ComponentFactoryOptimizer:
    """组件工厂优化器"""

    def __init__(self):
        self.standard_interface_template = '''#!/usr/bin/env python3
"""
统一组件接口定义

定义组件的标准化接口，确保一致性和可扩展性
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IComponent(ABC):
    """组件基础接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件

        Args:
            config: 组件配置参数

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """启动组件

        Returns:
            bool: 启动是否成功
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """停止组件

        Returns:
            bool: 停止是否成功
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            Dict[str, Any]: 组件状态信息
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查

        Returns:
            bool: 组件是否健康
        """
        pass
'''

        self.standard_factory_template = '''#!/usr/bin/env python3
"""
统一组件工厂

提供标准化的组件创建和管理机制
"""

from typing import Dict, Any, Optional, Type
from .interfaces import IComponent
import logging


logger = logging.getLogger(__name__)


class ComponentFactoryError(Exception):
    """组件工厂错误"""
    pass


class ComponentConfigError(ComponentFactoryError):
    """组件配置错误"""
    pass


class ComponentCreationError(ComponentFactoryError):
    """组件创建错误"""
    pass


class BaseComponentFactory:
    """基础组件工厂"""

    def __init__(self):
        self._components: Dict[str, IComponent] = {}
        self._component_types: Dict[str, Type[IComponent]] = {}

    def register_component_type(self, component_type: str, component_class: Type[IComponent]):
        """注册组件类型

        Args:
            component_type: 组件类型名称
            component_class: 组件类
        """
        self._component_types[component_type] = component_class
        logger.info(f"注册组件类型: {component_type} -> {component_class.__name__}")

    def create_component(self, component_type: str, component_id: str, config: Dict[str, Any]) -> IComponent:
        """创建组件

        Args:
            component_type: 组件类型
            component_id: 组件唯一标识
            config: 组件配置

        Returns:
            IComponent: 创建的组件实例

        Raises:
            ComponentCreationError: 创建失败时抛出
        """
        try:
            if component_type not in self._component_types:
                raise ComponentCreationError(f"未知的组件类型: {component_type}")

            if component_id in self._components:
                raise ComponentCreationError(f"组件ID已存在: {component_id}")

            # 验证配置
            self._validate_config(component_type, config)

            # 创建组件实例
            component_class = self._component_types[component_type]
            component = component_class()

            # 初始化组件
            if not component.initialize(config):
                raise ComponentCreationError(f"组件初始化失败: {component_id}")

            # 注册组件
            self._components[component_id] = component

            logger.info(f"成功创建组件: {component_id} (类型: {component_type})")
            return component

        except Exception as e:
            logger.error(f"创建组件失败 {component_id}: {e}")
            raise ComponentCreationError(f"组件创建失败: {e}") from e

    def get_component(self, component_id: str) -> Optional[IComponent]:
        """获取组件

        Args:
            component_id: 组件ID

        Returns:
            Optional[IComponent]: 组件实例，如果不存在则返回None
        """
        return self._components.get(component_id)

    def destroy_component(self, component_id: str) -> bool:
        """销毁组件

        Args:
            component_id: 组件ID

        Returns:
            bool: 是否成功销毁
        """
        try:
            if component_id not in self._components:
                logger.warning(f"组件不存在: {component_id}")
                return False

            component = self._components[component_id]

            # 停止组件
            component.stop()

            # 清理资源
            del self._components[component_id]

            logger.info(f"成功销毁组件: {component_id}")
            return True

        except Exception as e:
            logger.error(f"销毁组件失败 {component_id}: {e}")
            return False

    def list_components(self) -> Dict[str, Dict[str, Any]]:
        """列出所有组件

        Returns:
            Dict[str, Dict[str, Any]]: 组件信息字典
        """
        result = {}
        for component_id, component in self._components.items():
            try:
                status = component.get_status()
                result[component_id] = {
                    'type': type(component).__name__,
                    'status': status,
                    'healthy': component.health_check()
                }
            except Exception as e:
                result[component_id] = {
                    'type': type(component).__name__,
                    'status': {'error': str(e)},
                    'healthy': False
                }

        return result

    def health_check_all(self) -> Dict[str, bool]:
        """检查所有组件健康状态

        Returns:
            Dict[str, bool]: 各组件健康状态
        """
        result = {}
        for component_id, component in self._components.items():
            try:
                result[component_id] = component.health_check()
            except Exception as e:
                logger.error(f"健康检查失败 {component_id}: {e}")
                result[component_id] = False

        return result

    def _validate_config(self, component_type: str, config: Dict[str, Any]):
        """验证配置

        Args:
            component_type: 组件类型
            config: 配置字典

        Raises:
            ComponentConfigError: 配置无效时抛出
        """
        if not isinstance(config, dict):
            raise ComponentConfigError("配置必须是字典类型")

        # 检查必需的配置项
        required_keys = self._get_required_config_keys(component_type)
        for key in required_keys:
            if key not in config:
                raise ComponentConfigError(f"缺少必需的配置项: {key}")

        # 验证配置值类型
        for key, value in config.items():
            expected_type = self._get_config_value_type(component_type, key)
            if expected_type and not isinstance(value, expected_type):
                raise ComponentConfigError(
                    f"配置项 {key} 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}")

    def _get_required_config_keys(self, component_type: str) -> List[str]:
        """获取必需的配置键

        Args:
            component_type: 组件类型

        Returns:
            List[str]: 必需的配置键列表
        """
        # 子类可以重写此方法来定义各自的必需配置
        return ['name']

    def _get_config_value_type(self, component_type: str, key: str) -> Optional[type]:
        """获取配置值的期望类型

        Args:
            component_type: 组件类型
            key: 配置键

        Returns:
            Optional[type]: 期望的类型，如果不限制则返回None
        """
        # 子类可以重写此方法来定义配置类型检查
        type_mapping = {
            'name': str,
            'enabled': bool,
            'timeout': (int, float),
            'max_retries': int
        }
        return type_mapping.get(key)


class ComponentFactory(BaseComponentFactory):
    """标准组件工厂"""

    def __init__(self):
        super().__init__()
        self._component_counter = 0

    def create_component_with_auto_id(self, component_type: str, config: Dict[str, Any]) -> tuple[IComponent, str]:
        """创建组件并自动生成ID

        Args:
            component_type: 组件类型
            config: 组件配置

        Returns:
            tuple[IComponent, str]: (组件实例, 组件ID)
        """
        self._component_counter += 1
        component_id = f"{component_type}_{self._component_counter}"

        component = self.create_component(component_type, component_id, config)
        return component, component_id

    def bulk_create_components(self, component_specs: List[Dict[str, Any]]) -> Dict[str, IComponent]:
        """批量创建组件

        Args:
            component_specs: 组件规格列表，每个规格包含 component_type, config, component_id

        Returns:
            Dict[str, IComponent]: 创建的组件字典
        """
        results = {}

        for spec in component_specs:
            try:
                component_type = spec['component_type']
                config = spec.get('config', {})
                component_id = spec.get('component_id', f"{component_type}_auto")

                component = self.create_component(component_type, component_id, config)
                results[component_id] = component

            except Exception as e:
                logger.error(f"批量创建组件失败 {spec}: {e}")

        return results

    def destroy_all_components(self) -> Dict[str, bool]:
        """销毁所有组件

        Returns:
            Dict[str, bool]: 各组件销毁结果
        """
        results = {}
        component_ids = list(self._components.keys())

        for component_id in component_ids:
            results[component_id] = self.destroy_component(component_id)

        return results
'''

        self.standard_component_template = '''#!/usr/bin/env python3
"""
标准组件实现模板

提供标准化的组件实现框架
"""

from typing import Dict, Any
from .interfaces import IComponent
import logging
import time


logger = logging.getLogger(__name__)


class BaseComponent(IComponent):
    """基础组件实现"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._is_running = False
        self._start_time: Optional[float] = None
        self._last_health_check: Optional[float] = None
        self._health_check_interval = 30.0  # 30秒检查一次

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件

        Args:
            config: 组件配置

        Returns:
            bool: 初始化是否成功
        """
        try:
            self._validate_config(config)
            self._config = config.copy()
            self._is_running = False
            self._start_time = None
            self._last_health_check = None

            # 应用配置
            self._health_check_interval = config.get('health_check_interval', 30.0)

            logger.info(f"组件 {self._get_component_name()} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"组件 {self._get_component_name()} 初始化失败: {e}")
            return False

    def start(self) -> bool:
        """启动组件

        Returns:
            bool: 启动是否成功
        """
        try:
            if self._is_running:
                logger.warning(f"组件 {self._get_component_name()} 已在运行")
                return True

            # 执行启动逻辑
            if not self._do_start():
                return False

            self._is_running = True
            self._start_time = time.time()

            logger.info(f"组件 {self._get_component_name()} 启动成功")
            return True

        except Exception as e:
            logger.error(f"组件 {self._get_component_name()} 启动失败: {e}")
            return False

    def stop(self) -> bool:
        """停止组件

        Returns:
            bool: 停止是否成功
        """
        try:
            if not self._is_running:
                logger.warning(f"组件 {self._get_component_name()} 未在运行")
                return True

            # 执行停止逻辑
            if not self._do_stop():
                return False

            self._is_running = False
            self._start_time = None

            logger.info(f"组件 {self._get_component_name()} 停止成功")
            return True

        except Exception as e:
            logger.error(f"组件 {self._get_component_name()} 停止失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            Dict[str, Any]: 组件状态信息
        """
        try:
            status = {
                'component_name': self._get_component_name(),
                'is_running': self._is_running,
                'start_time': self._start_time,
                'uptime_seconds': (time.time() - self._start_time) if self._start_time else 0,
                'config': self._config.copy(),
                'last_health_check': self._last_health_check
            }

            # 获取额外的状态信息
            extra_status = self._get_additional_status()
            status.update(extra_status)

            return status

        except Exception as e:
            return {
                'component_name': self._get_component_name(),
                'error': str(e),
                'is_running': False
            }

    def health_check(self) -> bool:
        """健康检查

        Returns:
            bool: 组件是否健康
        """
        try:
            current_time = time.time()

            # 检查是否需要执行健康检查
            if (self._last_health_check and
                current_time - self._last_health_check < self._health_check_interval):
                return True

            # 执行健康检查
            is_healthy = self._do_health_check()
            self._last_health_check = current_time

            if not is_healthy:
                logger.warning(f"组件 {self._get_component_name()} 健康检查失败")

            return is_healthy

        except Exception as e:
            logger.error(f"组件 {self._get_component_name()} 健康检查异常: {e}")
            return False

    def _validate_config(self, config: Dict[str, Any]):
        """验证配置

        Args:
            config: 配置字典

        Raises:
            ValueError: 配置无效时抛出
        """
        if not isinstance(config, dict):
            raise ValueError("配置必须是字典类型")

        required_keys = self._get_required_config_keys()
        for key in required_keys:
            if key not in config:
                raise ValueError(f"缺少必需的配置项: {key}")

    def _get_component_name(self) -> str:
        """获取组件名称

        Returns:
            str: 组件名称
        """
        return type(self).__name__

    def _get_required_config_keys(self) -> List[str]:
        """获取必需的配置键

        Returns:
            List[str]: 必需的配置键列表
        """
        return ['name']

    def _get_additional_status(self) -> Dict[str, Any]:
        """获取额外状态信息

        Returns:
            Dict[str, Any]: 额外状态信息
        """
        return {}

    def _do_start(self) -> bool:
        """执行启动逻辑

        Returns:
            bool: 启动是否成功
        """
        # 子类重写此方法实现具体的启动逻辑
        return True

    def _do_stop(self) -> bool:
        """执行停止逻辑

        Returns:
            bool: 停止是否成功
        """
        # 子类重写此方法实现具体的停止逻辑
        return True

    def _do_health_check(self) -> bool:
        """执行健康检查逻辑

        Returns:
            bool: 健康检查是否通过
        """
        # 子类重写此方法实现具体的健康检查逻辑
        return self._is_running
'''

    def __init__(self):
        self.layers = {
            'core': 'src/core',
            'infrastructure': 'src/infrastructure',
            'data': 'src/data',
            'features': 'src/features',
            'ml': 'src/ml',
            'backtest': 'src/backtest',
            'risk': 'src/risk',
            'trading': 'src/trading',
            'engine': 'src/engine',
            'gateway': 'src/gateway'
        }

        self.optimized_components = []

    def scan_existing_components(self):
        """扫描现有组件工厂"""
        print("🔍 扫描现有组件工厂...")

        components_found = []

        for layer_name, layer_path in self.layers.items():
            layer_dir = Path(layer_path)
            if not layer_dir.exists():
                continue

            for root, dirs, files in os.walk(layer_dir):
                for file in files:
                    if file.endswith('_components.py'):
                        file_path = Path(root) / file
                        components_found.append({
                            'path': file_path,
                            'layer': layer_name,
                            'filename': file
                        })

        print(f"📋 发现 {len(components_found)} 个组件工厂文件")
        return components_found

    def analyze_component_structure(self, component_file: Dict):
        """分析组件结构"""
        file_path = component_file['path']

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分析组件结构
            structure = {
                'has_interface': 'class I' in content and 'Component(ABC):' in content,
                'has_factory': 'ComponentFactory:' in content,
                'has_create_method': 'create_component' in content,
                'has_import': 'from typing import' in content,
                'has_abc': 'from abc import' in content,
                'has_error_handling': 'try:' in content and 'except' in content,
                'has_logging': 'logger' in content or 'logging' in content,
                'file_size': len(content)
            }

            return structure

        except Exception as e:
            print(f"⚠️ 无法分析文件 {file_path}: {e}")
            return {}

    def generate_standard_interfaces(self):
        """生成标准接口定义"""
        print("📝 生成标准接口定义...")

        interfaces_dir = Path('src/core/interfaces')
        interfaces_dir.mkdir(parents=True, exist_ok=True)

        interface_file = interfaces_dir / 'component_interfaces.py'

        with open(interface_file, 'w', encoding='utf-8') as f:
            f.write(self.standard_interface_template)

        print("✅ 标准接口定义已生成")

    def generate_standard_factory(self):
        """生成标准工厂实现"""
        print("🏭 生成标准工厂实现...")

        factory_dir = Path('src/core/factories')
        factory_dir.mkdir(parents=True, exist_ok=True)

        factory_file = factory_dir / 'component_factory.py'

        with open(factory_file, 'w', encoding='utf-8') as f:
            f.write(self.standard_factory_template)

        print("✅ 标准工厂实现已生成")

    def generate_standard_component_template(self):
        """生成标准组件模板"""
        print("📋 生成标准组件模板...")

        templates_dir = Path('src/core/templates')
        templates_dir.mkdir(parents=True, exist_ok=True)

        template_file = templates_dir / 'standard_component_template.py'

        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(self.standard_component_template)

        print("✅ 标准组件模板已生成")

    def optimize_component_factory(self, component_file: Dict):
        """优化组件工厂"""
        file_path = component_file['path']
        structure = self.analyze_component_structure(component_file)

        print(f"🔧 优化组件工厂: {file_path}")

        # 备份原文件
        backup_path = file_path.with_suffix('.py.factory_backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as original:
                f.write(original.read())

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 应用优化
            optimized_content = self._apply_optimizations(content, structure)

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(optimized_content)

            self.optimized_components.append({
                'original': file_path,
                'backup': backup_path,
                'structure': structure,
                'optimizations_applied': self._get_optimizations_applied(structure)
            })

            print(f"✅ 组件工厂优化完成: {file_path}")

        except Exception as e:
            print(f"❌ 优化失败 {file_path}: {e}")

    def _apply_optimizations(self, content: str, structure: Dict) -> str:
        """应用优化"""
        lines = content.split('\n')
        optimized_lines = []

        # 添加标准导入
        if not structure.get('has_import', False):
            optimized_lines.append("from typing import Dict, Any, Optional, Type")
            optimized_lines.append("")

        if not structure.get('has_abc', False):
            optimized_lines.append("from abc import ABC, abstractmethod")
            optimized_lines.append("")

        # 优化接口定义
        if not structure.get('has_interface', False):
            optimized_lines.extend([
                "class IComponent(ABC):",
                "    \"\"\"组件接口\"\"\"",
                "    @abstractmethod",
                "    def initialize(self, config: Dict[str, Any]) -> bool:",
                "        pass",
                "",
                "    @abstractmethod",
                "    def start(self) -> bool:",
                "        pass",
                "",
                "    @abstractmethod",
                "    def stop(self) -> bool:",
                "        pass",
                "",
                "    @abstractmethod",
                "    def get_status(self) -> Dict[str, Any]:",
                "        pass",
                "",
                "    @abstractmethod",
                "    def health_check(self) -> bool:",
                "        pass",
                ""
            ])

        # 优化工厂类
        if not structure.get('has_factory', False):
            optimized_lines.extend([
                "class ComponentFactory:",
                "    \"\"\"组件工厂\"\"\"",
                "    def __init__(self):",
                "        self._components = {}",
                "",
                "    def create_component(self, component_type: str, config: Dict[str, Any]):",
                "        \"\"\"创建组件\"\"\"",
                "        try:",
                "            # 组件创建逻辑",
                "            component = self._create_component_instance(component_type, config)",
                "            if component and component.initialize(config):",
                "                return component",
                "            return None",
                "        except Exception as e:",
                "            print(f\"创建组件失败: {e}\")",
                "            return None",
                "",
                "    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):",
                "        \"\"\"创建组件实例 - 子类重写此方法\"\"\"",
                "        return None",
                ""
            ])

        # 添加错误处理
        if not structure.get('has_error_handling', False):
            # 为现有的创建方法添加错误处理
            for i, line in enumerate(lines):
                if 'def create_component' in line:
                    # 找到方法体开始
                    method_start = i
                    indent = len(line) - len(line.lstrip())

                    # 找到方法结束
                    method_end = method_start
                    for j in range(method_start + 1, len(lines)):
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent <= indent and lines[j].strip():
                            method_end = j - 1
                            break

                    # 添加try-except块
                    optimized_lines.extend(lines[method_start:method_end+1])
                    optimized_lines.extend([
                        " " * (indent + 4) + "except Exception as e:",
                        " " * (indent + 8) + "print(f\"创建组件失败: {e}\")",
                        " " * (indent + 8) + "return None"
                    ])
                    break
            else:
                optimized_lines.extend(lines)

        # 添加日志支持
        if not structure.get('has_logging', False):
            optimized_lines.insert(0, "import logging")
            optimized_lines.insert(1, "")
            optimized_lines.insert(2, "logger = logging.getLogger(__name__)")
            optimized_lines.insert(3, "")

        return '\n'.join(optimized_lines)

    def _get_optimizations_applied(self, structure: Dict) -> List[str]:
        """获取应用的优化"""
        optimizations = []

        if not structure.get('has_interface', False):
            optimizations.append("添加标准接口定义")

        if not structure.get('has_factory', False):
            optimizations.append("添加标准工厂类")

        if not structure.get('has_error_handling', False):
            optimizations.append("增强错误处理")

        if not structure.get('has_logging', False):
            optimizations.append("添加日志支持")

        if not structure.get('has_import', False):
            optimizations.append("添加类型导入")

        if not structure.get('has_abc', False):
            optimizations.append("添加ABC导入")

        return optimizations

    def generate_optimization_report(self):
        """生成优化报告"""
        report = []

        report.append("# 组件工厂优化报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## 优化概览")
        report.append("")
        report.append(f"- **优化组件数**: {len(self.optimized_components)}")
        report.append("")

        total_optimizations = sum(len(comp['optimizations_applied'])
                                  for comp in self.optimized_components)
        report.append(f"- **总优化项数**: {total_optimizations}")
        report.append("")

        report.append("## 标准组件体系")
        report.append("")
        report.append("### 生成的文件")
        report.append("- ✅ `src/core/interfaces/component_interfaces.py` - 标准接口定义")
        report.append("- ✅ `src/core/factories/component_factory.py` - 标准工厂实现")
        report.append("- ✅ `src/core/templates/standard_component_template.py` - 标准组件模板")
        report.append("")

        report.append("### 标准特性")
        report.append("- 🔧 统一的接口定义 (IComponent)")
        report.append("- 🏭 标准化的工厂模式 (ComponentFactory)")
        report.append("- 📋 完整的生命周期管理")
        report.append("- 🩺 内置健康检查机制")
        report.append("- 🚨 增强的错误处理")
        report.append("- 📊 详细的状态监控")
        report.append("- 📝 完善的日志记录")
        report.append("")

        report.append("## 优化详情")
        report.append("")

        for comp in self.optimized_components:
            report.append(f"### {comp['original']}")
            report.append(f"- **备份文件**: {comp['backup']}")
            report.append("- **优化项**:")
            for opt in comp['optimizations_applied']:
                report.append(f"  - ✅ {opt}")
            report.append("")

        report.append("## 优化效果预期")
        report.append("")
        report.append("### 代码质量提升")
        report.append("- 🎯 组件结构标准化: 统一接口和工厂模式")
        report.append("- 🛡️ 错误处理增强: 完善的异常处理机制")
        report.append("- 📊 可观测性提升: 内置状态监控和健康检查")
        report.append("- 🔧 可维护性改善: 清晰的生命周期管理")
        report.append("")
        report.append("### 开发效率提升")
        report.append("- 🚀 快速组件开发: 标准模板和工厂模式")
        report.append("- 🐛 问题定位加速: 增强的日志和状态监控")
        report.append("- 🧪 测试友好: 标准化的接口和错误处理")
        report.append("- 📚 学习曲线平滑: 统一的组件模式")
        report.append("")

        with open('reports/COMPONENT_FACTORY_OPTIMIZATION_REPORT.md', 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

    def run_optimization(self):
        """运行组件工厂优化"""
        print("🚀 开始组件工厂优化...")
        print("="*60)

        try:
            # 1. 扫描现有组件
            existing_components = self.scan_existing_components()

            if not existing_components:
                print("⚠️ 未发现现有组件工厂文件")
            else:
                print(f"📋 发现 {len(existing_components)} 个组件工厂文件")

            # 2. 生成标准组件体系
            print("\n📝 生成标准组件体系...")
            self.generate_standard_interfaces()
            self.generate_standard_factory()
            self.generate_standard_component_template()

            # 3. 优化现有组件工厂
            if existing_components:
                print("\n🔧 优化现有组件工厂...")
                for component_file in existing_components:
                    self.optimize_component_factory(component_file)

            # 4. 生成报告
            self.generate_optimization_report()

            print("
📋 组件工厂优化报告已生成: "            print(" - src/core/interfaces/component_interfaces.py")
            print("   - src/core/factories/component_factory.py")
            print("   - src/core/templates/standard_component_template.py")
            print("   - reports/COMPONENT_FACTORY_OPTIMIZATION_REPORT.md")
            print("🎉 组件工厂优化完成！"
            return True

        except Exception as e:
            print(f"\n❌ 优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    optimizer=ComponentFactoryOptimizer()
    success=optimizer.run_optimization()

    if success:
        print("\n" + "="*60)
        print("组件工厂优化成功完成！")
        print("✅ 标准组件体系已建立")
        print("✅ 现有组件已优化")
        print("✅ 接口定义已统一")
        print("✅ 错误处理已增强")
        print("="*60)
    else:
        print("\n❌ 组件工厂优化失败！")


if __name__ == "__main__":
    main()

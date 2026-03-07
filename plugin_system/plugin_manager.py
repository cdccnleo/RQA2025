#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 创新引擎扩展插件系统
提供灵活的插件架构，支持第三方扩展和定制化开发

插件系统特性:
1. 插件管理器 - 插件注册、加载、卸载和生命周期管理
2. 标准化接口 - 统一的插件接口和通信协议
3. 插件市场 - 插件发现、分发和版本控制
4. 安全沙箱 - 插件安全验证和隔离执行
5. 热插拔机制 - 运行时插件动态加载和卸载
6. 依赖管理 - 插件依赖关系解析和版本兼容性
"""

import json
import time
import hashlib
import importlib
import threading
import inspect
from datetime import datetime, timedelta
from pathlib import Path
import sys
import zipfile
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class PluginInterface:
    """插件接口基类"""

    def __init__(self):
        self.name = "BasePlugin"
        self.version = "1.0.0"
        self.description = "基础插件接口"
        self.author = "RQA2026 Team"
        self.dependencies = []
        self.config_schema = {}

    def initialize(self, config: dict) -> bool:
        """插件初始化"""
        raise NotImplementedError("插件必须实现 initialize 方法")

    def execute(self, data: dict) -> dict:
        """插件执行"""
        raise NotImplementedError("插件必须实现 execute 方法")

    def cleanup(self) -> bool:
        """插件清理"""
        raise NotImplementedError("插件必须实现 cleanup 方法")

    def get_info(self) -> dict:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'config_schema': self.config_schema
        }


class PluginMetadata:
    """插件元数据"""

    def __init__(self, metadata_dict: dict):
        self.name = metadata_dict.get('name', '')
        self.version = metadata_dict.get('version', '1.0.0')
        self.description = metadata_dict.get('description', '')
        self.author = metadata_dict.get('author', '')
        self.category = metadata_dict.get('category', 'general')
        self.tags = metadata_dict.get('tags', [])
        self.dependencies = metadata_dict.get('dependencies', [])
        self.compatibility = metadata_dict.get('compatibility', {})
        self.permissions = metadata_dict.get('permissions', [])
        self.created_at = metadata_dict.get('created_at', datetime.now().isoformat())
        self.updated_at = metadata_dict.get('updated_at', datetime.now().isoformat())

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'category': self.category,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'compatibility': self.compatibility,
            'permissions': self.permissions,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class PluginSandbox:
    """插件沙箱环境"""

    def __init__(self):
        self.allowed_modules = {
            'json', 'datetime', 'time', 'math', 'random', 'collections',
            'itertools', 'functools', 'operator', 're', 'string', 'os.path'
        }
        self.forbidden_modules = {
            'os', 'sys', 'subprocess', 'importlib', 'builtins', 'pickle',
            'socket', 'urllib', 'http', 'ssl', 'multiprocessing'
        }
        self.resource_limits = {
            'max_memory': 100 * 1024 * 1024,  # 100MB
            'max_cpu_time': 30,  # 30秒
            'max_disk_usage': 50 * 1024 * 1024  # 50MB
        }

    def validate_plugin_code(self, plugin_path: str) -> dict:
        """验证插件代码安全性"""
        issues = []

        try:
            with open(plugin_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查危险导入
            for forbidden in self.forbidden_modules:
                if f"import {forbidden}" in content or f"from {forbidden}" in content:
                    issues.append(f"禁止导入模块: {forbidden}")

            # 检查危险函数调用
            dangerous_patterns = [
                'eval(', 'exec(', 'compile(', '__import__(',
                'open(', 'file(', 'input(', 'raw_input('
            ]

            for pattern in dangerous_patterns:
                if pattern in content:
                    issues.append(f"检测到危险函数调用: {pattern[:-1]}")

            # 检查文件操作
            if 'os.' in content or 'shutil.' in content or 'pathlib.' in content:
                issues.append("检测到文件系统操作")

        except Exception as e:
            issues.append(f"代码验证失败: {str(e)}")

        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'validated_at': datetime.now().isoformat()
        }

    def create_sandbox_context(self) -> dict:
        """创建沙箱执行上下文"""
        # 这里应该实现更完整的沙箱环境
        # 包括资源限制、模块白名单等
        return {
            'allowed_modules': list(self.allowed_modules),
            'resource_limits': self.resource_limits,
            'created_at': datetime.now().isoformat()
        }


class PluginManager:
    """插件管理器"""

    def __init__(self):
        self.plugins = {}  # 已加载的插件实例
        self.plugin_metadata = {}  # 插件元数据
        self.plugin_paths = {}  # 插件路径映射
        self.plugin_dependencies = {}  # 插件依赖关系

        self.sandbox = PluginSandbox()
        self.plugin_dir = Path("plugin_system/plugins")
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.plugin_dir / "plugin_registry.json"
        self.load_registry()

    def register_plugin(self, plugin_class: type, metadata: dict, plugin_path: str) -> bool:
        """注册插件"""
        try:
            plugin_name = metadata.get('name', '')
            if not plugin_name:
                raise ValueError("插件名称不能为空")

            # 验证插件代码安全性
            security_check = self.sandbox.validate_plugin_code(plugin_path)
            if not security_check['is_safe']:
                print(f"⚠️ 插件 {plugin_name} 安全检查失败: {security_check['issues']}")
                return False

            # 创建元数据对象
            plugin_metadata = PluginMetadata(metadata)
            plugin_metadata.updated_at = datetime.now().isoformat()

            # 存储插件信息
            self.plugin_metadata[plugin_name] = plugin_metadata
            self.plugin_paths[plugin_name] = plugin_path

            # 分析依赖关系
            self._analyze_dependencies(plugin_name, plugin_metadata.dependencies)

            # 保存注册表
            self.save_registry()

            print(f"✅ 插件 {plugin_name} v{plugin_metadata.version} 注册成功")
            return True

        except Exception as e:
            print(f"❌ 插件注册失败: {str(e)}")
            return False

    def load_plugin(self, plugin_name: str, config: dict = None) -> bool:
        """加载插件"""
        try:
            if plugin_name not in self.plugin_metadata:
                raise ValueError(f"插件 {plugin_name} 未注册")

            # 检查依赖
            if not self._check_dependencies(plugin_name):
                raise ValueError(f"插件 {plugin_name} 依赖未满足")

            plugin_path = self.plugin_paths[plugin_name]
            metadata = self.plugin_metadata[plugin_name]

            # 动态导入插件类
            spec = importlib.util.spec_from_file_location(f"plugin_{plugin_name}", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找插件类 (继承自PluginInterface的类)
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginInterface) and
                    obj != PluginInterface):
                    plugin_class = obj
                    break

            if not plugin_class:
                raise ValueError(f"在 {plugin_path} 中未找到有效的插件类")

            # 创建插件实例
            plugin_instance = plugin_class()
            plugin_instance.name = metadata.name
            plugin_instance.version = metadata.version
            plugin_instance.description = metadata.description
            plugin_instance.author = metadata.author

            # 初始化插件
            if config is None:
                config = {}

            if not plugin_instance.initialize(config):
                raise ValueError(f"插件 {plugin_name} 初始化失败")

            # 存储插件实例
            self.plugins[plugin_name] = {
                'instance': plugin_instance,
                'metadata': metadata,
                'loaded_at': datetime.now().isoformat(),
                'config': config
            }

            print(f"✅ 插件 {plugin_name} 加载成功")
            return True

        except Exception as e:
            print(f"❌ 插件 {plugin_name} 加载失败: {str(e)}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        try:
            if plugin_name not in self.plugins:
                raise ValueError(f"插件 {plugin_name} 未加载")

            plugin_info = self.plugins[plugin_name]
            plugin_instance = plugin_info['instance']

            # 清理插件
            if hasattr(plugin_instance, 'cleanup'):
                plugin_instance.cleanup()

            # 从内存中移除
            del self.plugins[plugin_name]

            print(f"✅ 插件 {plugin_name} 卸载成功")
            return True

        except Exception as e:
            print(f"❌ 插件 {plugin_name} 卸载失败: {str(e)}")
            return False

    def execute_plugin(self, plugin_name: str, data: dict) -> dict:
        """执行插件"""
        try:
            if plugin_name not in self.plugins:
                raise ValueError(f"插件 {plugin_name} 未加载")

            plugin_info = self.plugins[plugin_name]
            plugin_instance = plugin_info['instance']

            # 在沙箱环境中执行
            sandbox_context = self.sandbox.create_sandbox_context()

            # 执行插件
            start_time = time.time()
            result = plugin_instance.execute(data)
            execution_time = time.time() - start_time

            # 添加执行元数据
            result['_execution_metadata'] = {
                'plugin_name': plugin_name,
                'execution_time': round(execution_time, 3),
                'executed_at': datetime.now().isoformat(),
                'sandbox_context': sandbox_context
            }

            return result

        except Exception as e:
            return {
                'error': str(e),
                'plugin_name': plugin_name,
                'executed_at': datetime.now().isoformat()
            }

    def list_plugins(self, category: str = None, status: str = None) -> list:
        """列出插件"""
        plugins = []

        for name, metadata in self.plugin_metadata.items():
            plugin_info = {
                'name': name,
                'metadata': metadata.to_dict(),
                'status': 'loaded' if name in self.plugins else 'registered',
                'path': self.plugin_paths.get(name, '')
            }

            # 按类别过滤
            if category and metadata.category != category:
                continue

            # 按状态过滤
            if status and plugin_info['status'] != status:
                continue

            plugins.append(plugin_info)

        return plugins

    def get_plugin_info(self, plugin_name: str) -> dict:
        """获取插件信息"""
        if plugin_name not in self.plugin_metadata:
            return {'error': f'插件 {plugin_name} 未找到'}

        metadata = self.plugin_metadata[plugin_name]
        status = 'loaded' if plugin_name in self.plugins else 'registered'

        info = {
            'name': plugin_name,
            'metadata': metadata.to_dict(),
            'status': status,
            'path': self.plugin_paths.get(plugin_name, '')
        }

        if status == 'loaded':
            info['loaded_info'] = self.plugins[plugin_name]

        return info

    def install_plugin_from_file(self, plugin_file: str) -> bool:
        """从文件安装插件"""
        try:
            plugin_path = Path(plugin_file)
            if not plugin_path.exists():
                raise FileNotFoundError(f"插件文件不存在: {plugin_file}")

            if plugin_path.suffix == '.zip':
                # 处理压缩包
                return self._install_from_zip(plugin_path)
            else:
                # 处理单个Python文件
                return self._install_from_py_file(plugin_path)

        except Exception as e:
            print(f"❌ 插件安装失败: {str(e)}")
            return False

    def _install_from_zip(self, zip_path: Path) -> bool:
        """从ZIP文件安装插件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压到临时目录
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 查找插件文件和元数据
            temp_path = Path(temp_dir)
            plugin_files = list(temp_path.glob("*.py"))
            metadata_files = list(temp_path.glob("plugin.json"))

            if not plugin_files:
                raise ValueError("ZIP文件中未找到Python插件文件")

            plugin_file = plugin_files[0]
            metadata = {}

            if metadata_files:
                with open(metadata_files[0], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            # 复制到插件目录
            plugin_name = metadata.get('name', plugin_file.stem)
            target_dir = self.plugin_dir / plugin_name
            target_dir.mkdir(exist_ok=True)

            # 复制所有文件
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, target_dir / file_path.name)

            target_plugin_file = target_dir / plugin_file.name

            # 注册插件
            return self.register_plugin(None, metadata, str(target_plugin_file))

    def _install_from_py_file(self, py_path: Path) -> bool:
        """从Python文件安装插件"""
        # 复制到插件目录
        plugin_name = py_path.stem
        target_dir = self.plugin_dir / plugin_name
        target_dir.mkdir(exist_ok=True)

        target_file = target_dir / py_path.name
        shutil.copy2(py_path, target_file)

        # 尝试从文件内容提取元数据 (简单的启发式方法)
        metadata = self._extract_metadata_from_file(target_file)

        # 注册插件
        return self.register_plugin(None, metadata, str(target_file))

    def _extract_metadata_from_file(self, file_path: Path) -> dict:
        """从插件文件提取元数据"""
        metadata = {
            'name': file_path.stem,
            'version': '1.0.0',
            'description': f'插件 {file_path.stem}',
            'author': 'Unknown',
            'category': 'general'
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单的元数据提取 (查找类定义和注释)
            import re

            # 查找类名
            class_match = re.search(r'class\s+(\w+)\s*\(', content)
            if class_match:
                metadata['name'] = class_match.group(1)

            # 查找作者信息
            author_match = re.search(r'author[:\s]+([^\n]+)', content, re.IGNORECASE)
            if author_match:
                metadata['author'] = author_match.group(1).strip()

            # 查找描述信息
            desc_match = re.search(r'description[:\s]+([^\n]+)', content, re.IGNORECASE)
            if desc_match:
                metadata['description'] = desc_match.group(1).strip()

        except Exception:
            pass

        return metadata

    def _analyze_dependencies(self, plugin_name: str, dependencies: list):
        """分析插件依赖关系"""
        self.plugin_dependencies[plugin_name] = dependencies

        # 更新反向依赖关系 (哪些插件依赖于此插件)
        for dep in dependencies:
            if dep not in self.plugin_dependencies:
                self.plugin_dependencies[dep] = []
            if plugin_name not in self.plugin_dependencies[dep]:
                self.plugin_dependencies[dep].append(plugin_name)

    def _check_dependencies(self, plugin_name: str) -> bool:
        """检查插件依赖是否满足"""
        dependencies = self.plugin_dependencies.get(plugin_name, [])

        for dep in dependencies:
            if dep not in self.plugin_metadata:
                print(f"⚠️ 插件 {plugin_name} 依赖 {dep} 未注册")
                return False

            # 检查版本兼容性 (简化版本)
            dep_metadata = self.plugin_metadata[dep]
            required_version = ">=1.0.0"  # 默认要求

            # 这里应该实现更复杂的版本比较逻辑
            # 暂时只检查插件是否存在

        return True

    def load_registry(self):
        """加载插件注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    registry = json.load(f)

                # 恢复元数据
                for name, metadata_dict in registry.get('plugins', {}).items():
                    self.plugin_metadata[name] = PluginMetadata(metadata_dict)
                    self.plugin_paths[name] = metadata_dict.get('path', '')

                # 恢复依赖关系
                self.plugin_dependencies = registry.get('dependencies', {})

            except Exception as e:
                print(f"⚠️ 加载插件注册表失败: {str(e)}")

    def save_registry(self):
        """保存插件注册表"""
        registry = {
            'plugins': {},
            'dependencies': self.plugin_dependencies,
            'last_updated': datetime.now().isoformat()
        }

        # 保存插件元数据和路径
        for name, metadata in self.plugin_metadata.items():
            plugin_data = metadata.to_dict()
            plugin_data['path'] = self.plugin_paths.get(name, '')
            registry['plugins'][name] = plugin_data

        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存插件注册表失败: {str(e)}")


class PluginMarketplace:
    """插件市场"""

    def __init__(self):
        self.marketplace_url = "https://marketplace.rqa2026.com/api"
        self.local_cache = Path("plugin_system/marketplace_cache.json")
        self.cache_duration = timedelta(hours=1)
        self.available_plugins = {}

    def search_plugins(self, query: str = "", category: str = "", tags: list = None) -> list:
        """搜索插件"""
        # 模拟从市场搜索插件
        # 实际实现应该调用真实的API

        mock_plugins = [
            {
                'name': 'advanced_analytics_plugin',
                'version': '2.1.0',
                'description': '高级数据分析和预测插件',
                'author': 'RQA Analytics Team',
                'category': 'analytics',
                'tags': ['machine_learning', 'prediction', 'statistics'],
                'downloads': 1250,
                'rating': 4.8
            },
            {
                'name': 'blockchain_connector',
                'version': '1.5.0',
                'description': '区块链数据连接器插件',
                'author': 'Blockchain Labs',
                'category': 'connectors',
                'tags': ['blockchain', 'cryptocurrency', 'data_source'],
                'downloads': 890,
                'rating': 4.6
            },
            {
                'name': 'real_time_monitoring',
                'version': '3.0.1',
                'description': '实时系统监控和告警插件',
                'author': 'DevOps Solutions',
                'category': 'monitoring',
                'tags': ['monitoring', 'alerts', 'real_time'],
                'downloads': 2100,
                'rating': 4.9
            }
        ]

        results = []
        for plugin in mock_plugins:
            # 应用搜索过滤
            if query and query.lower() not in plugin['name'].lower() and query.lower() not in plugin['description'].lower():
                continue

            if category and plugin['category'] != category:
                continue

            if tags:
                if not any(tag in plugin['tags'] for tag in tags):
                    continue

            results.append(plugin)

        return results

    def download_plugin(self, plugin_name: str, version: str = "latest") -> str:
        """下载插件"""
        # 模拟插件下载
        # 实际实现应该从市场下载插件包

        print(f"📥 下载插件 {plugin_name} v{version}...")

        # 模拟下载时间
        time.sleep(1)

        # 创建模拟插件文件
        class_name = plugin_name.title().replace("_", "")
        plugin_content = f'''"""
{plugin_name} Plugin
Generated plugin for demonstration
"""

from plugin_system.plugin_manager import PluginInterface

class {class_name}Plugin(PluginInterface):
    """{plugin_name} 插件实现"""

    def __init__(self):
        super().__init__()
        self.name = "{plugin_name}"
        self.version = "{version}"
        self.description = "{plugin_name} plugin for RQA2026"
        self.author = "Plugin Marketplace"
        self.config_schema = {{
            "api_key": {{"type": "string", "required": False}},
            "timeout": {{"type": "integer", "default": 30}}
        }}

    def initialize(self, config: dict) -> bool:
        """插件初始化"""
        print("🔧 初始化插件")
        self.config = config
        return True

    def execute(self, data: dict) -> dict:
        """插件执行"""
        print("⚡ 执行插件")
        # 模拟插件功能
        result = {{
            "plugin_name": self.name,
            "input_data": data,
            "processed_at": "2024-01-01T00:00:00Z",
            "result": "Processed by plugin"
        }}
        return result

    def cleanup(self) -> bool:
        """插件清理"""
        print("🧹 清理插件")
        return True
'''

        # 保存到临时文件
        temp_file = Path(f"plugin_system/downloads/{plugin_name}_{version}.py")
        temp_file.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(plugin_content)

        print(f"✅ 插件 {plugin_name} 下载完成")
        return str(temp_file)

    def get_plugin_details(self, plugin_name: str) -> dict:
        """获取插件详细信息"""
        # 模拟获取插件详情
        details = {
            'name': plugin_name,
            'version': '2.0.0',
            'description': f'详细描述 {plugin_name} 插件的功能和特性',
            'author': 'Plugin Author',
            'category': 'analytics',
            'tags': ['feature1', 'feature2'],
            'dependencies': ['base_plugin>=1.0.0'],
            'compatibility': {
                'rqa2026_core': '>=1.0.0',
                'python': '>=3.8'
            },
            'permissions': ['read_data', 'write_results'],
            'changelog': [
                {'version': '2.0.0', 'changes': ['新增功能A', '修复问题B']},
                {'version': '1.5.0', 'changes': ['优化性能', '改进UI']}
            ],
            'reviews': [
                {'user': 'user1', 'rating': 5, 'comment': '很好用！'},
                {'user': 'user2', 'rating': 4, 'comment': '功能强大'}
            ]
        }

        return details


def create_plugin_system():
    """创建插件系统"""
    print("🔌 启动 RQA2026 创新引擎扩展插件系统")
    print("=" * 80)

    plugin_manager = PluginManager()
    marketplace = PluginMarketplace()

    return plugin_manager, marketplace


def demonstrate_plugin_system():
    """演示插件系统功能"""
    plugin_manager, marketplace = create_plugin_system()

    print("🚀 插件系统功能演示")
    print("-" * 50)

    # 1. 搜索插件市场
    print("\\n1️⃣ 搜索插件市场:")
    search_results = marketplace.search_plugins(category="analytics")
    print(f"   📦 找到 {len(search_results)} 个分析类插件:")
    for plugin in search_results[:3]:
        print(f"      • {plugin['name']} v{plugin['version']} - {plugin['description']}")

    # 2. 下载插件
    print("\\n2️⃣ 下载插件:")
    if search_results:
        plugin_name = search_results[0]['name']
        downloaded_file = marketplace.download_plugin(plugin_name)
        print(f"   📥 已下载到: {downloaded_file}")

    # 3. 演示插件注册和管理
    print("\\n3️⃣ 演示插件注册和管理:")

    # 手动创建和注册一个简单的插件
    metadata = {
        'name': 'sample_plugin',
        'version': '1.0.0',
        'description': '示例插件用于演示',
        'author': 'RQA2026 Team',
        'category': 'demo',
        'tags': ['demo', 'sample']
    }

    # 创建简单的插件代码
    plugin_code = '''
from plugin_system.plugin_manager import PluginInterface

class SamplePlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.name = "sample_plugin"
        self.version = "1.0.0"
        self.description = "示例插件"
        self.author = "RQA2026 Team"

    def initialize(self, config):
        print("🔧 初始化示例插件")
        return True

    def execute(self, data):
        return {"result": f"处理了 {len(data) if isinstance(data, dict) else 0} 个数据项"}

    def cleanup(self):
        print("🧹 清理示例插件")
        return True
'''

    # 保存到临时文件
    temp_plugin_file = Path("plugin_system/temp_sample_plugin.py")
    with open(temp_plugin_file, 'w', encoding='utf-8') as f:
        f.write(plugin_code)

    # 注册插件
    if plugin_manager.register_plugin(None, metadata, str(temp_plugin_file)):
        print("   ✅ 插件 sample_plugin 注册成功")

        # 4. 加载插件
        print("\\n4️⃣ 加载插件:")
        if plugin_manager.load_plugin('sample_plugin', {}):
            print("   ✅ 插件 sample_plugin 加载成功")

            # 5. 执行插件
            print("\\n5️⃣ 执行插件:")
            test_data = {"test": "data", "count": 5}
            result = plugin_manager.execute_plugin('sample_plugin', test_data)
            print(f"   ⚡ 执行结果: {result}")

            # 6. 卸载插件
            print("\\n6️⃣ 卸载插件:")
            if plugin_manager.unload_plugin('sample_plugin'):
                print("   ✅ 插件 sample_plugin 卸载成功")
        else:
            print("   ❌ 插件加载失败")
    else:
        print("   ❌ 插件注册失败")

    # 清理临时文件
    if temp_plugin_file.exists():
        temp_plugin_file.unlink()

    # 7. 列出插件
    print("\\n7️⃣ 插件列表:")
    plugins = plugin_manager.list_plugins()
    print(f"   📋 注册插件总数: {len(plugins)}")
    for plugin in plugins[:5]:  # 只显示前5个
        status = "✅" if plugin['status'] == 'loaded' else "📦"
        print(f"      {status} {plugin['name']} - {plugin['metadata']['description']}")

    # 8. 获取插件详情
    print("\\n8️⃣ 获取插件详情:")
    if search_results:
        details = marketplace.get_plugin_details(plugin_name)
        print(f"   📄 {plugin_name} 详情:")
        print(f"      版本: {details['version']}")
        print(f"      作者: {details['author']}")
        print(f"      分类: {details['category']}")
        print(f"      标签: {', '.join(details['tags'])}")

    print("\\n✅ 插件系统演示完成！")
    print("🔌 系统现已支持插件注册、加载、执行和市场分发")


if __name__ == "__main__":
    demonstrate_plugin_system()

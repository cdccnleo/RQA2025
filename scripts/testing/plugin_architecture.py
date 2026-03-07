#!/usr/bin/env python3
"""
RQA2025 基础插件架构
支持可扩展的插件系统，便于功能扩展
"""

import os
import sys
import json
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import re
from datetime import datetime
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class PluginBase(ABC):
    """插件基类"""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.config = {}

    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件"""

    @abstractmethod
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行插件功能"""

    def cleanup(self):
        """清理插件资源"""

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'class': self.__class__.__name__
        }


class PluginManager:
    """插件管理器"""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)

    def load_plugins(self, config_file: str = None):
        """加载所有插件"""
        logger.info("🔌 开始加载插件...")

        # 加载配置文件
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.plugin_configs = config.get('plugins', {})

        # 扫描插件目录
        plugin_files = list(self.plugin_dir.glob("*.py"))
        logger.info(f"发现 {len(plugin_files)} 个插件文件")

        for plugin_file in plugin_files:
            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                logger.error(f"加载插件失败 {plugin_file}: {e}")

        logger.info(f"✅ 成功加载 {len(self.plugins)} 个插件")

    def _load_plugin(self, plugin_file: Path):
        """加载单个插件"""
        try:
            # 动态导入插件模块
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找插件类
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                        obj != PluginBase):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.warning(f"插件文件 {plugin_file} 中未找到插件类")
                return

            # 实例化插件
            for plugin_class in plugin_classes:
                plugin = plugin_class()

                # 获取插件配置
                plugin_config = self.plugin_configs.get(plugin.name, {})

                # 初始化插件
                if plugin.initialize(plugin_config):
                    self.plugins[plugin.name] = plugin
                    logger.info(f"✅ 加载插件: {plugin.name} v{plugin.version}")
                else:
                    logger.warning(f"⚠️ 插件初始化失败: {plugin.name}")

        except Exception as e:
            logger.error(f"❌ 加载插件异常 {plugin_file}: {e}")

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """获取插件"""
        return self.plugins.get(name)

    def get_all_plugins(self) -> Dict[str, PluginBase]:
        """获取所有插件"""
        return self.plugins.copy()

    def enable_plugin(self, name: str) -> bool:
        """启用插件"""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enabled = True
            logger.info(f"✅ 启用插件: {name}")
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """禁用插件"""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enabled = False
            logger.info(f"✅ 禁用插件: {name}")
            return True
        return False

    def execute_plugin(self, name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行插件"""
        plugin = self.get_plugin(name)
        if not plugin:
            return {'success': False, 'error': f'插件不存在: {name}'}

        if not plugin.enabled:
            return {'success': False, 'error': f'插件已禁用: {name}'}

        try:
            result = plugin.execute(context or {})
            result['success'] = True
            result['plugin'] = name
            return result
        except Exception as e:
            logger.error(f"执行插件失败 {name}: {e}")
            return {'success': False, 'error': str(e), 'plugin': name}

    def execute_all_plugins(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行所有启用的插件"""
        results = {}

        for name, plugin in self.plugins.items():
            if plugin.enabled:
                results[name] = self.execute_plugin(name, context)

        return results

    def register_hook(self, hook_name: str, callback: Callable):
        """注册钩子函数"""
        self.hooks[hook_name].append(callback)
        logger.info(f"✅ 注册钩子: {hook_name}")

    def execute_hooks(self, hook_name: str, data: Any = None) -> List[Any]:
        """执行钩子函数"""
        results = []

        for callback in self.hooks[hook_name]:
            try:
                result = callback(data)
                results.append(result)
            except Exception as e:
                logger.error(f"执行钩子失败 {hook_name}: {e}")

        return results

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """获取插件信息"""
        return [plugin.get_info() for plugin in self.plugins.values()]

    def cleanup_plugins(self):
        """清理所有插件"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"清理插件失败 {plugin.name}: {e}")


class TestGenerationPlugin(PluginBase):
    """测试生成插件"""

    def __init__(self):
        super().__init__("test_generation", "1.0.0")
        self.test_templates = {}

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件"""
        self.config = config or {}

        # 加载测试模板
        self.test_templates = {
            'unit_test': self._load_unit_test_template(),
            'integration_test': self._load_integration_test_template(),
            'performance_test': self._load_performance_test_template()
        }

        logger.info(f"✅ 测试生成插件初始化完成")
        return True

    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行测试生成"""
        context = context or {}

        module_path = context.get('module_path', '')
        test_type = context.get('test_type', 'unit_test')
        template = self.test_templates.get(test_type, '')

        if not template:
            return {'success': False, 'error': f'未找到测试模板: {test_type}'}

        # 生成测试代码
        test_code = self._generate_test_code(module_path, template, context)

        return {
            'success': True,
            'test_code': test_code,
            'test_type': test_type,
            'module_path': module_path
        }

    def _load_unit_test_template(self) -> str:
        """加载单元测试模板"""
        return '''#!/usr/bin/env python3
"""
{module_name} 单元测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class Test{class_name}:
    """{class_name} 测试类"""
    
    def setup_method(self):
        """测试前设置"""
        pass
    
    def test_initialization(self):
        """测试初始化"""
        pass
    
    def test_basic_functionality(self):
        """测试基本功能"""
        pass
    
    def test_error_handling(self):
        """测试错误处理"""
        pass
'''

    def _load_integration_test_template(self) -> str:
        """加载集成测试模板"""
        return '''#!/usr/bin/env python3
"""
{module_name} 集成测试
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class Test{class_name}Integration:
    """{class_name} 集成测试类"""
    
    def setup_method(self):
        """测试前设置"""
        pass
    
    def test_integration_workflow(self):
        """测试集成工作流"""
        pass
    
    def test_external_dependencies(self):
        """测试外部依赖"""
        pass
'''

    def _load_performance_test_template(self) -> str:
        """加载性能测试模板"""
        return '''#!/usr/bin/env python3
"""
{module_name} 性能测试
"""

import pytest
import time
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class Test{class_name}Performance:
    """{class_name} 性能测试类"""
    
    def setup_method(self):
        """测试前设置"""
        pass
    
    def test_performance_baseline(self):
        """测试性能基准"""
        start_time = time.time()
        # 执行测试
        duration = time.time() - start_time
        assert duration < 1.0  # 应该在1秒内完成
    
    def test_memory_usage(self):
        """测试内存使用"""
        pass
'''

    def _generate_test_code(self, module_path: str, template: str, context: Dict[str, Any]) -> str:
        """生成测试代码"""
        # 从模块路径提取信息
        module_name = Path(module_path).stem
        class_name = ''.join(word.capitalize() for word in module_name.split('_'))

        # 替换模板变量
        test_code = template.format(
            module_name=module_name,
            class_name=class_name
        )

        return test_code


class SecurityReviewPlugin(PluginBase):
    """安全审查插件"""

    def __init__(self):
        super().__init__("security_review", "1.0.0")
        self.security_rules = []

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件"""
        self.config = config or {}

        # 加载安全规则
        self.security_rules = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\('
        ]

        logger.info(f"✅ 安全审查插件初始化完成")
        return True

    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行安全审查"""
        context = context or {}

        code = context.get('code', '')
        if not code:
            return {'success': False, 'error': '未提供代码内容'}

        # 执行安全检查
        issues = self._check_security(code)

        return {
            'success': True,
            'issues': issues,
            'issue_count': len(issues),
            'security_score': self._calculate_security_score(issues)
        }

    def _check_security(self, code: str) -> List[Dict[str, Any]]:
        """检查代码安全性"""
        issues = []

        for i, pattern in enumerate(self.security_rules):
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                issues.append({
                    'type': 'security_issue',
                    'pattern': pattern,
                    'line': self._find_line_number(code, match.start()),
                    'severity': 8 - i  # 越靠前的规则越严重
                })

        return issues

    def _find_line_number(self, code: str, position: int) -> int:
        """查找位置对应的行号"""
        try:
            lines = code.split('\n')
            current_pos = 0

            for i, line in enumerate(lines):
                if position >= current_pos and position < current_pos + len(line) + 1:
                    return i + 1
                current_pos += len(line) + 1

            return 1
        except:
            return 1

    def _calculate_security_score(self, issues: List[Dict[str, Any]]) -> int:
        """计算安全评分"""
        score = 100

        for issue in issues:
            severity = issue.get('severity', 5)
            score -= severity

        return max(0, score)


class MetricsCollectionPlugin(PluginBase):
    """指标收集插件"""

    def __init__(self):
        super().__init__("metrics_collection", "1.0.0")
        self.metrics = {}

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化插件"""
        self.config = config or {}
        logger.info(f"✅ 指标收集插件初始化完成")
        return True

    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行指标收集"""
        context = context or {}

        # 收集各种指标
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self._collect_system_metrics(),
            'performance_metrics': self._collect_performance_metrics(),
            'business_metrics': self._collect_business_metrics(context)
        }

        self.metrics.update(metrics)

        return {
            'success': True,
            'metrics': metrics
        }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            import psutil

            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('.').percent
            }
        except ImportError:
            return {'error': 'psutil not available'}

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        return {
            'execution_time': time.time(),
            'memory_usage': self._get_memory_usage()
        }

    def _collect_business_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """收集业务指标"""
        return {
            'plugin_executions': len(self.metrics),
            'context_size': len(context)
        }

    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0.0


def create_plugin_config() -> str:
    """创建插件配置文件"""
    config = {
        "plugins": {
            "test_generation": {
                "enabled": True,
                "templates_dir": "templates",
                "output_dir": "tests"
            },
            "security_review": {
                "enabled": True,
                "strict_mode": True,
                "max_score": 70
            },
            "metrics_collection": {
                "enabled": True,
                "interval": 30,
                "output_file": "metrics.json"
            }
        },
        "hooks": {
            "pre_execution": [],
            "post_execution": [],
            "error_handling": []
        }
    }

    config_file = "config/plugin_config.json"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_file


def main():
    """主函数"""
    # 创建插件配置
    config_file = create_plugin_config()

    # 初始化插件管理器
    plugin_manager = PluginManager()

    # 加载插件
    plugin_manager.load_plugins(config_file)

    # 注册示例插件
    plugin_manager.plugins['test_generation'] = TestGenerationPlugin()
    plugin_manager.plugins['security_review'] = SecurityReviewPlugin()
    plugin_manager.plugins['metrics_collection'] = MetricsCollectionPlugin()

    # 初始化插件
    for plugin in plugin_manager.plugins.values():
        plugin.initialize()

    # 测试插件执行
    context = {
        'module_path': 'src/infrastructure/config_manager.py',
        'test_type': 'unit_test',
        'code': 'import os\ndef test_func():\n    eval("1+1")'
    }

    # 执行测试生成插件
    test_result = plugin_manager.execute_plugin('test_generation', context)
    print(f"测试生成结果: {test_result['success']}")

    # 执行安全审查插件
    security_result = plugin_manager.execute_plugin('security_review', context)
    print(f"安全审查结果: 评分 {security_result['security_score']}")

    # 执行指标收集插件
    metrics_result = plugin_manager.execute_plugin('metrics_collection', context)
    print(f"指标收集结果: {len(metrics_result['metrics'])} 个指标")

    # 显示插件信息
    plugin_info = plugin_manager.get_plugin_info()
    print(f"\n插件信息:")
    for info in plugin_info:
        print(f"  - {info['name']} v{info['version']} ({'启用' if info['enabled'] else '禁用'})")

    # 清理插件
    plugin_manager.cleanup_plugins()


if __name__ == "__main__":
    main()

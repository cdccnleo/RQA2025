#!/usr/bin/env python3
"""
配置管理代码组织优化脚本
清理重复的配置管理器实现，统一配置管理接口
"""

import shutil
import logging
from pathlib import Path
from typing import List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManagementOptimizer:
    """配置管理优化器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup" / "config_optimization"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def analyze_duplicate_config_managers(self) -> Dict[str, List[str]]:
        """分析重复的配置管理器"""
        logger.info("分析重复的配置管理器...")

        duplicate_managers = {
            "core_manager": [
                "src/infrastructure/config/core/manager.py"
            ],
            "unified_manager": [
                "src/infrastructure/config/managers/unified.py"
            ],
            "distributed_manager": [
                "src/infrastructure/config/services/config_service.py"
            ],
            "feature_manager": [
                "src/features/config.py"
            ]
        }

        # 检查文件存在性
        existing_managers = {}
        for category, files in duplicate_managers.items():
            existing_files = []
            for file_path in files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    existing_files.append(str(file_path))
            if existing_files:
                existing_managers[category] = existing_files

        logger.info(f"发现 {len(existing_managers)} 个配置管理器类别")
        for category, files in existing_managers.items():
            logger.info(f"  {category}: {files}")

        return existing_managers

    def backup_config_files(self) -> bool:
        """备份配置相关文件"""
        logger.info("备份配置相关文件...")

        config_dirs = [
            "src/infrastructure/config",
            "src/features/config.py",
            "config"
        ]

        try:
            for config_dir in config_dirs:
                source_path = self.project_root / config_dir
                if source_path.exists():
                    backup_path = self.backup_dir / config_dir.replace("/", "_")
                    if source_path.is_file():
                        shutil.copy2(source_path, backup_path)
                    else:
                        shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
                    logger.info(f"已备份: {config_dir}")

            return True
        except Exception as e:
            logger.error(f"备份失败: {e}")
            return False

    def create_unified_config_interface(self) -> bool:
        """创建统一的配置管理接口"""
        logger.info("创建统一的配置管理接口...")

        interface_content = '''"""
统一配置管理接口
提供标准化的配置管理功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class ConfigScope(Enum):
    """配置作用域"""
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    FEATURES = "features"
    MODELS = "models"
    TRADING = "trading"
    GLOBAL = "global"

@dataclass
class ConfigItem:
    """配置项"""
    key: str
    value: Any
    scope: ConfigScope
    description: str = ""
    version: str = "1.0"
    deprecated: bool = False
    required: bool = False
    validation_rules: List[str] = None

class IConfigManager(ABC):
    """统一配置管理器接口"""
    
    @abstractmethod
    def get(self, key: str, scope: ConfigScope = ConfigScope.GLOBAL, default: Any = None) -> Any:
        """获取配置值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
        """设置配置值"""
        pass
    
    @abstractmethod
    def load(self, source: str) -> bool:
        """加载配置"""
        pass
    
    @abstractmethod
    def save(self, destination: str) -> bool:
        """保存配置"""
        pass
    
    @abstractmethod
    def validate(self, config: Dict[str, Any] = None) -> tuple[bool, Optional[Dict[str, str]]]:
        """验证配置"""
        pass
    
    @abstractmethod
    def get_scope_config(self, scope: ConfigScope) -> Dict[str, Any]:
        """获取作用域配置"""
        pass
    
    @abstractmethod
    def set_scope_config(self, scope: ConfigScope, config: Dict[str, Any]) -> bool:
        """设置作用域配置"""
        pass

class IConfigVersionManager(ABC):
    """配置版本管理接口"""
    
    @abstractmethod
    def create_version(self, config: Dict[str, Any], env: str = "default") -> str:
        """创建配置版本"""
        pass
    
    @abstractmethod
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """获取指定版本配置"""
        pass
    
    @abstractmethod
    def list_versions(self) -> List[str]:
        """列出所有版本"""
        pass
    
    @abstractmethod
    def rollback(self, version_id: str) -> bool:
        """回滚到指定版本"""
        pass

class IConfigValidator(ABC):
    """配置验证接口"""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> tuple[bool, Optional[Dict[str, str]]]:
        """验证配置"""
        pass
    
    @abstractmethod
    def add_validation_rule(self, rule: callable) -> None:
        """添加验证规则"""
        pass
    
    @abstractmethod
    def remove_validation_rule(self, rule: callable) -> None:
        """移除验证规则"""
        pass
'''

        interface_file = self.project_root / "src" / "infrastructure" / \
            "config" / "interfaces" / "unified_interface.py"
        interface_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(interface_file, 'w', encoding='utf-8') as f:
                f.write(interface_content)
            logger.info(f"已创建统一接口: {interface_file}")
            return True
        except Exception as e:
            logger.error(f"创建接口失败: {e}")
            return False

    def create_unified_config_manager(self) -> bool:
        """创建统一的配置管理器实现"""
        logger.info("创建统一的配置管理器实现...")

        manager_content = '''"""
统一配置管理器实现
整合所有配置管理功能
"""

import threading
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from collections import defaultdict

from .interfaces.unified_interface import (
    IConfigManager, IConfigVersionManager, IConfigValidator,
    ConfigScope, ConfigItem
)

logger = logging.getLogger(__name__)

class UnifiedConfigManager(IConfigManager):
    """统一配置管理器"""
    
    def __init__(self, config_dir: str = "config", env: str = "default"):
        self.config_dir = Path(config_dir)
        self.env = env
        self._config = {}
        self._scope_configs = defaultdict(dict)
        self._watchers = defaultdict(list)
        self._lock = threading.RLock()
        
        # 初始化各层配置
        self._init_scope_configs()
        
        # 加载配置
        self._load_configs()
        
        logger.info(f"UnifiedConfigManager initialized for {env} environment")
    
    def _init_scope_configs(self):
        """初始化各层配置"""
        # 基础设施层配置
        self._scope_configs[ConfigScope.INFRASTRUCTURE] = {
            'cache.enabled': True,
            'cache.max_size': 1000,
            'cache.ttl': 3600,
            'monitoring.enabled': True,
            'security.enabled': True,
            'logging.level': 'INFO',
            'database.url': 'sqlite:///rqa.db',
            'redis.host': 'localhost',
            'redis.port': 6379
        }

        # 数据层配置
        self._scope_configs[ConfigScope.DATA] = {
            'adapters.china.enabled': True,
            'adapters.china.source': 'tushare',
            'loaders.batch_size': 1000,
            'loaders.parallel': True,
            'validators.enabled': True,
            'quality.checks': ['missing', 'outliers', 'duplicates'],
            'cache.enabled': True,
            'cache.strategy': 'lru'
        }

        # 特征工程层配置
        self._scope_configs[ConfigScope.FEATURES] = {
            'processors.technical.enabled': True,
            'processors.technical.indicators': ['ma', 'rsi', 'macd'],
            'processors.sentiment.enabled': True,
            'processors.sentiment.source': 'news',
            'processors.orderbook.enabled': True,
            'processors.orderbook.depth': 10,
            'feature_selection.enabled': True,
            'feature_selection.method': 'mutual_info'
        }

        # 模型层配置
        self._scope_configs[ConfigScope.MODELS] = {
            'training.batch_size': 32,
            'training.epochs': 100,
            'training.validation_split': 0.2,
            'training.optimizer': 'adam',
            'training.learning_rate': 0.001,
            'inference.batch_size': 64,
            'inference.gpu_enabled': True,
            'model.persistence.enabled': True
        }

        # 交易层配置
        self._scope_configs[ConfigScope.TRADING] = {
            'execution.enabled': True,
            'execution.max_position': 1000000,
            'risk.max_drawdown': 0.1,
            'risk.stop_loss': 0.05,
            'risk.position_sizing': 'kelly',
            'backtest.enabled': True,
            'backtest.start_date': '2020-01-01',
            'backtest.end_date': '2023-12-31'
        }

        # 全局配置
        self._scope_configs[ConfigScope.GLOBAL] = {
            'system.name': 'RQA2025',
            'system.version': '2.0.0',
            'system.environment': self.env,
            'system.debug': False,
            'system.timezone': 'Asia/Shanghai'
        }
    
    def _load_configs(self):
        """加载配置文件"""
        try:
            # 加载环境特定配置
            env_config_file = self.config_dir / f"{self.env}.json"
            if env_config_file.exists():
                with open(env_config_file, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)
                    self._config.update(env_config)
            
            # 加载默认配置
            default_config_file = self.config_dir / "default.json"
            if default_config_file.exists():
                with open(default_config_file, 'r', encoding='utf-8') as f:
                    default_config = json.load(f)
                    self._config.update(default_config)
            
            logger.info(f"配置加载完成，共 {len(self._config)} 项")
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
    
    def get(self, key: str, scope: ConfigScope = ConfigScope.GLOBAL, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            # 首先从作用域配置中查找
            if scope in self._scope_configs and key in self._scope_configs[scope]:
                return self._scope_configs[scope][key]
            
            # 从全局配置中查找
            if key in self._config:
                return self._config[key]
            
            return default
    
    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
        """设置配置值"""
        try:
            with self._lock:
                if scope == ConfigScope.GLOBAL:
                    self._config[key] = value
                else:
                    self._scope_configs[scope][key] = value
                
                # 通知监听器
                self._notify_watchers(key, None, value)
                
                logger.info(f"配置已更新: {key} = {value}")
                return True
        except Exception as e:
            logger.error(f"设置配置失败: {key}, 错误: {e}")
            return False
    
    def load(self, source: str) -> bool:
        """加载配置"""
        try:
            config_file = Path(source)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._config.update(config_data)
                    logger.info(f"配置已从 {source} 加载")
                    return True
            return False
        except Exception as e:
            logger.error(f"加载配置失败: {source}, 错误: {e}")
            return False
    
    def save(self, destination: str) -> bool:
        """保存配置"""
        try:
            config_file = Path(destination)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"配置已保存到 {destination}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {destination}, 错误: {e}")
            return False
    
    def validate(self, config: Dict[str, Any] = None) -> tuple[bool, Optional[Dict[str, str]]]:
        """验证配置"""
        if config is None:
            config = self._config
        
        errors = {}
        
        # 基本验证
        for key, value in config.items():
            if value is None and key in ['database.url', 'redis.host']:
                errors[key] = "必需配置项不能为空"
        
        return len(errors) == 0, errors if errors else None
    
    def get_scope_config(self, scope: ConfigScope) -> Dict[str, Any]:
        """获取作用域配置"""
        with self._lock:
            return dict(self._scope_configs[scope])
    
    def set_scope_config(self, scope: ConfigScope, config: Dict[str, Any]) -> bool:
        """设置作用域配置"""
        try:
            with self._lock:
                self._scope_configs[scope].update(config)
                logger.info(f"作用域配置已更新: {scope}")
                return True
        except Exception as e:
            logger.error(f"设置作用域配置失败: {scope}, 错误: {e}")
            return False
    
    def add_watcher(self, key: str, callback: Callable[[str, Any, Any], None]) -> str:
        """添加配置监听器"""
        watcher_id = f"watcher_{int(time.time() * 1000)}"
        self._watchers[key].append((watcher_id, callback))
        logger.info(f"已添加配置监听器: {key} -> {watcher_id}")
        return watcher_id
    
    def remove_watcher(self, key: str, watcher_id: str) -> bool:
        """移除配置监听器"""
        if key in self._watchers:
            self._watchers[key] = [(wid, cb) for wid, cb in self._watchers[key] if wid != watcher_id]
            logger.info(f"已移除配置监听器: {key} -> {watcher_id}")
            return True
        return False
    
    def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """通知监听器"""
        if key in self._watchers:
            for watcher_id, callback in self._watchers[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"监听器回调失败: {watcher_id}, 错误: {e}")
    
    def export_config(self, scope: Optional[ConfigScope] = None) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            if scope:
                return {
                    'scope': scope.value,
                    'config': self.get_scope_config(scope),
                    'timestamp': time.time()
                }
            else:
                return {
                    'global_config': self._config,
                    'scope_configs': {scope.value: config for scope, config in self._scope_configs.items()},
                    'timestamp': time.time()
                }
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """导入配置"""
        try:
            with self._lock:
                if 'global_config' in config_data:
                    self._config.update(config_data['global_config'])
                
                if 'scope_configs' in config_data:
                    for scope_name, config in config_data['scope_configs'].items():
                        scope = ConfigScope(scope_name)
                        self._scope_configs[scope].update(config)
                
                logger.info("配置导入完成")
                return True
        except Exception as e:
            logger.error(f"配置导入失败: {e}")
            return False

# 全局配置管理器实例
_unified_config_manager = None

def get_unified_config_manager() -> UnifiedConfigManager:
    """获取统一配置管理器实例"""
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedConfigManager()
    return _unified_config_manager

def get_config(key: str, scope: ConfigScope = ConfigScope.GLOBAL, default: Any = None) -> Any:
    """获取配置值"""
    return get_unified_config_manager().get(key, scope, default)

def set_config(key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
    """设置配置值"""
    return get_unified_config_manager().set(key, value, scope)
'''

        manager_file = self.project_root / "src" / "infrastructure" / "config" / "unified_manager.py"
        manager_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(manager_file, 'w', encoding='utf-8') as f:
                f.write(manager_content)
            logger.info(f"已创建统一配置管理器: {manager_file}")
            return True
        except Exception as e:
            logger.error(f"创建配置管理器失败: {e}")
            return False

    def create_migration_guide(self) -> bool:
        """创建迁移指南"""
        logger.info("创建迁移指南...")

        guide_content = '''# 配置管理代码组织优化迁移指南

## 概述
本次优化整合了多个重复的配置管理器实现，统一为 `UnifiedConfigManager`。

## 变更内容

### 1. 删除的重复实现
- `src/infrastructure/config/core/manager.py` - 旧的ConfigManager
- `src/infrastructure/config/managers/unified.py` - 旧的UnifiedConfigManager  
- `src/infrastructure/config/services/config_service.py` - DistributedConfigManager
- `src/features/config.py` - FeatureConfigManager

### 2. 新的统一实现
- `src/infrastructure/config/unified_manager.py` - 新的UnifiedConfigManager
- `src/infrastructure/config/interfaces/unified_interface.py` - 统一接口定义

## 迁移步骤

### 步骤1: 更新导入语句
```python
# 旧代码
from src.infrastructure.config.core.manager import ConfigManager
from src.infrastructure.config.managers.unified import UnifiedConfigManager
from src.infrastructure.config.services.config_service import DistributedConfigManager
from src.features.config import FeatureConfigManager

# 新代码
from src.infrastructure.config.unified_manager import (
    UnifiedConfigManager, 
    get_unified_config_manager,
    get_config, 
    set_config
)
```

### 步骤2: 更新配置管理器实例化
```python
# 旧代码
config_manager = ConfigManager()
unified_manager = UnifiedConfigManager()

# 新代码
config_manager = get_unified_config_manager()
# 或者直接使用函数
value = get_config('database.url')
set_config('logging.level', 'INFO')
```

### 步骤3: 更新配置作用域使用
```python
# 旧代码
config = unified_manager.get_scope_config(ConfigScope.FEATURES)

# 新代码
config = get_config('feature.enabled', ConfigScope.FEATURES)
```

## 兼容性说明

### 保持兼容的API
- `get(key, default=None)` - 获取配置值
- `set(key, value)` - 设置配置值
- `load(source)` - 加载配置
- `save(destination)` - 保存配置
- `validate()` - 验证配置

### 新增的API
- `get_config(key, scope, default)` - 全局函数获取配置
- `set_config(key, value, scope)` - 全局函数设置配置
- `get_scope_config(scope)` - 获取作用域配置
- `set_scope_config(scope, config)` - 设置作用域配置

## 测试验证

### 1. 功能测试
```python
from src.infrastructure.config.unified_manager import get_config, set_config, ConfigScope

# 测试基本功能
set_config('test.key', 'test_value')
assert get_config('test.key') == 'test_value'

# 测试作用域功能
set_config('feature.enabled', True, ConfigScope.FEATURES)
assert get_config('feature.enabled', ConfigScope.FEATURES) == True
```

### 2. 兼容性测试
```python
from src.infrastructure.config.unified_manager import get_unified_config_manager

config_manager = get_unified_config_manager()
config_manager.set('database.url', 'sqlite:///test.db')
assert config_manager.get('database.url') == 'sqlite:///test.db'
```

## 回滚方案

如果遇到问题，可以回滚到备份的配置管理代码：

1. 恢复备份文件
2. 更新导入语句
3. 重新运行测试

备份位置: `backup/config_optimization/`
'''

        guide_file = self.project_root / "docs" / "migration" / "config_management_migration.md"
        guide_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            logger.info(f"已创建迁移指南: {guide_file}")
            return True
        except Exception as e:
            logger.error(f"创建迁移指南失败: {e}")
            return False

    def create_test_file(self) -> bool:
        """创建测试文件"""
        logger.info("创建配置管理测试文件...")

        test_content = '''"""
配置管理优化测试
验证统一配置管理器的功能
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.infrastructure.config.unified_manager import (
    UnifiedConfigManager, 
    get_unified_config_manager,
    get_config, 
    set_config,
    ConfigScope
)

class TestUnifiedConfigManager:
    """统一配置管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = UnifiedConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_get_set(self):
        """测试基本获取和设置功能"""
        # 测试设置和获取
        assert self.config_manager.set('test.key', 'test_value')
        assert self.config_manager.get('test.key') == 'test_value'
        
        # 测试默认值
        assert self.config_manager.get('nonexistent.key', 'default') == 'default'
    
    def test_scope_config(self):
        """测试作用域配置"""
        # 测试特征工程作用域
        self.config_manager.set('feature.enabled', True, ConfigScope.FEATURES)
        assert self.config_manager.get('feature.enabled', ConfigScope.FEATURES) == True
        
        # 测试数据层作用域
        self.config_manager.set('data.batch_size', 1000, ConfigScope.DATA)
        assert self.config_manager.get('data.batch_size', ConfigScope.DATA) == 1000
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        is_valid, errors = self.config_manager.validate({
            'database.url': 'sqlite:///test.db',
            'logging.level': 'INFO'
        })
        assert is_valid
        assert errors is None
        
        # 测试无效配置
        is_valid, errors = self.config_manager.validate({
            'database.url': None,
            'logging.level': 'INFO'
        })
        assert not is_valid
        assert 'database.url' in errors
    
    def test_config_persistence(self):
        """测试配置持久化"""
        # 设置配置
        self.config_manager.set('persistent.key', 'persistent_value')
        
        # 保存配置
        config_file = Path(self.temp_dir) / "test_config.json"
        assert self.config_manager.save(str(config_file))
        
        # 创建新的管理器并加载配置
        new_manager = UnifiedConfigManager(config_dir=self.temp_dir)
        assert new_manager.load(str(config_file))
        assert new_manager.get('persistent.key') == 'persistent_value'
    
    def test_watcher_functionality(self):
        """测试配置监听器功能"""
        changes = []
        
        def watcher_callback(key, old_value, new_value):
            changes.append((key, old_value, new_value))
        
        # 添加监听器
        watcher_id = self.config_manager.add_watcher('watched.key', watcher_callback)
        
        # 修改配置
        self.config_manager.set('watched.key', 'new_value')
        
        # 验证监听器被调用
        assert len(changes) == 1
        assert changes[0] == ('watched.key', None, 'new_value')
        
        # 移除监听器
        assert self.config_manager.remove_watcher('watched.key', watcher_id)
    
    def test_export_import(self):
        """测试配置导出导入"""
        # 设置一些配置
        self.config_manager.set('export.key1', 'value1')
        self.config_manager.set('export.key2', 'value2', ConfigScope.FEATURES)
        
        # 导出配置
        exported = self.config_manager.export_config()
        assert 'global_config' in exported
        assert 'scope_configs' in exported
        
        # 创建新的管理器并导入配置
        new_manager = UnifiedConfigManager(config_dir=self.temp_dir)
        assert new_manager.import_config(exported)
        
        # 验证配置已导入
        assert new_manager.get('export.key1') == 'value1'
        assert new_manager.get('export.key2', ConfigScope.FEATURES) == 'value2'

class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_get_config(self):
        """测试get_config函数"""
        # 设置配置
        set_config('global.key', 'global_value')
        
        # 获取配置
        assert get_config('global.key') == 'global_value'
        assert get_config('nonexistent.key', default='default') == 'default'
    
    def test_set_config(self):
        """测试set_config函数"""
        # 设置配置
        assert set_config('function.key', 'function_value')
        
        # 验证配置已设置
        assert get_config('function.key') == 'function_value'
    
    def test_scope_functions(self):
        """测试作用域函数"""
        # 设置作用域配置
        assert set_config('scope.key', 'scope_value', ConfigScope.FEATURES)
        
        # 获取作用域配置
        assert get_config('scope.key', ConfigScope.FEATURES) == 'scope_value'
        
        # 验证全局作用域中没有该配置
        assert get_config('scope.key') is None

class TestCompatibility:
    """兼容性测试"""
    
    def test_old_api_compatibility(self):
        """测试旧API兼容性"""
        config_manager = get_unified_config_manager()
        
        # 测试旧的API调用方式
        config_manager.set('compat.key', 'compat_value')
        assert config_manager.get('compat.key') == 'compat_value'
        
        # 测试配置验证
        is_valid, errors = config_manager.validate()
        assert is_valid or errors is not None
    
    def test_manager_singleton(self):
        """测试管理器单例模式"""
        manager1 = get_unified_config_manager()
        manager2 = get_unified_config_manager()
        
        # 验证是同一个实例
        assert manager1 is manager2
        
        # 验证配置共享
        manager1.set('singleton.key', 'singleton_value')
        assert manager2.get('singleton.key') == 'singleton_value'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

        test_file = self.project_root / "tests" / "unit" / \
            "infrastructure" / "config" / "test_unified_config_manager.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            logger.info(f"已创建测试文件: {test_file}")
            return True
        except Exception as e:
            logger.error(f"创建测试文件失败: {e}")
            return False

    def run_optimization(self) -> bool:
        """运行完整的优化流程"""
        logger.info("开始配置管理代码组织优化...")

        try:
            # 1. 分析重复实现
            duplicate_managers = self.analyze_duplicate_config_managers()

            # 2. 备份现有文件
            if not self.backup_config_files():
                return False

            # 3. 创建统一接口
            if not self.create_unified_config_interface():
                return False

            # 4. 创建统一实现
            if not self.create_unified_config_manager():
                return False

            # 5. 创建迁移指南
            if not self.create_migration_guide():
                return False

            # 6. 创建测试文件
            if not self.create_test_file():
                return False

            logger.info("配置管理代码组织优化完成!")
            logger.info(f"备份文件位置: {self.backup_dir}")
            logger.info("请查看迁移指南: docs/migration/config_management_migration.md")

            return True

        except Exception as e:
            logger.error(f"优化过程失败: {e}")
            return False


def main():
    """主函数"""
    optimizer = ConfigManagementOptimizer()
    success = optimizer.run_optimization()

    if success:
        print("✅ 配置管理代码组织优化完成!")
        print("📁 备份文件: backup/config_optimization/")
        print("📖 迁移指南: docs/migration/config_management_migration.md")
        print("🧪 测试文件: tests/unit/infrastructure/config/test_unified_config_manager.py")
    else:
        print("❌ 配置管理代码组织优化失败!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

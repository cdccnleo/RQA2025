# RQA2025 配置管理测试指南

## 📋 文档概述

本文档为RQA2025项目的配置管理模块提供详细的测试策略、测试用例和测试规范，确保配置管理功能的正确性和可靠性。

**版本**: v3.5  
**最后更新**: 2025-01-27  
**维护者**: 配置管理团队  
**状态**: ✅ 活跃维护

---

## 🧪 测试策略

### 1. 测试层次

#### 1.1 单元测试 (Unit Tests)
- **目标**: 测试单个组件和方法的正确性
- **覆盖范围**: 所有公共接口和核心逻辑
- **执行频率**: 每次代码提交
- **工具**: pytest

#### 1.2 集成测试 (Integration Tests)
- **目标**: 测试组件间的交互和协作
- **覆盖范围**: 配置管理完整工作流
- **执行频率**: 每日构建
- **工具**: pytest + subprocess

#### 1.3 性能测试 (Performance Tests)
- **目标**: 验证配置管理性能指标
- **覆盖范围**: 响应时间、吞吐量、资源使用
- **执行频率**: 每周
- **工具**: pytest + time模块

#### 1.4 安全测试 (Security Tests)
- **目标**: 验证配置管理安全特性
- **覆盖范围**: 加密、访问控制、审计日志
- **执行频率**: 每月
- **工具**: pytest + 安全测试工具

### 2. 测试环境

#### 2.1 测试环境配置

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

@pytest.fixture(scope="session")
def test_config_dir():
    """创建测试配置目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_manager(test_config_dir):
    """创建测试用配置管理器"""
    return UnifiedConfigManager(
        config_dir=test_config_dir,
        env="test",
        enable_hot_reload=False,
        enable_distributed_sync=False
    )

@pytest.fixture
def production_config_manager(test_config_dir):
    """创建生产环境配置管理器"""
    return UnifiedConfigManager(
        config_dir=test_config_dir,
        env="production",
        enable_encryption=True,
        enable_hot_reload=True
    )
```

---

## 📝 测试用例

### 1. 统一配置管理器测试

#### 1.1 基础功能测试

```python
# tests/unit/infrastructure/config/test_unified_manager.py
import pytest
import json
import tempfile
from pathlib import Path
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class TestUnifiedConfigManager:
    
    def test_initialization(self, test_config_dir):
        """测试配置管理器初始化"""
        manager = UnifiedConfigManager(
            config_dir=test_config_dir,
            env="test"
        )
        
        assert manager.config_dir == Path(test_config_dir)
        assert manager.env == "test"
        assert not manager.enable_hot_reload
        assert not manager.enable_distributed_sync
    
    def test_basic_get_set(self, config_manager):
        """测试基本的获取和设置功能"""
        # 设置配置
        success = config_manager.set("test.key", "test_value", ConfigScope.GLOBAL)
        assert success
        
        # 获取配置
        value = config_manager.get("test.key", ConfigScope.GLOBAL)
        assert value == "test_value"
        
        # 测试默认值
        default_value = config_manager.get("nonexistent.key", default="default")
        assert default_value == "default"
    
    def test_scope_isolation(self, config_manager):
        """测试配置作用域隔离"""
        # 在不同作用域设置相同键
        config_manager.set("key", "global_value", ConfigScope.GLOBAL)
        config_manager.set("key", "infra_value", ConfigScope.INFRASTRUCTURE)
        config_manager.set("key", "data_value", ConfigScope.DATA)
        
        # 验证作用域隔离
        assert config_manager.get("key", ConfigScope.GLOBAL) == "global_value"
        assert config_manager.get("key", ConfigScope.INFRASTRUCTURE) == "infra_value"
        assert config_manager.get("key", ConfigScope.DATA) == "data_value"
    
    def test_config_validation(self, config_manager):
        """测试配置验证功能"""
        # 设置有效配置
        config_manager.set("valid.key", "valid_value")
        
        # 验证配置
        is_valid, errors = config_manager.validate()
        assert is_valid
        assert errors is None
    
    def test_config_watchers(self, config_manager):
        """测试配置观察者功能"""
        changes = []
        
        def on_change(key, old_value, new_value):
            changes.append((key, old_value, new_value))
        
        # 添加观察者
        watcher_id = config_manager.add_watcher("test.key", on_change)
        
        # 修改配置
        config_manager.set("test.key", "new_value")
        
        # 验证回调被调用
        assert len(changes) == 1
        assert changes[0] == ("test.key", None, "new_value")
        
        # 移除观察者
        success = config_manager.remove_watcher("test.key", watcher_id)
        assert success
    
    def test_config_export_import(self, config_manager):
        """测试配置导出导入功能"""
        # 设置测试配置
        config_manager.set("app.name", "TestApp", ConfigScope.GLOBAL)
        config_manager.set("database.host", "localhost", ConfigScope.INFRASTRUCTURE)
        
        # 导出配置
        exported = config_manager.export_config()
        
        # 验证导出内容
        assert "app.name" in exported
        assert exported["app.name"] == "TestApp"
        
        # 创建新的配置管理器
        new_manager = UnifiedConfigManager(env="test")
        
        # 导入配置
        success = new_manager.import_config(exported)
        assert success
        
        # 验证导入结果
        assert new_manager.get("app.name") == "TestApp"
        assert new_manager.get("database.host") == "localhost"
    
    def test_file_operations(self, config_manager, test_config_dir):
        """测试文件操作功能"""
        config_file = Path(test_config_dir) / "test_config.json"
        
        # 设置配置
        config_manager.set("file.test", "file_value")
        
        # 保存到文件
        success = config_manager.save(str(config_file))
        assert success
        assert config_file.exists()
        
        # 从文件加载
        new_manager = UnifiedConfigManager(config_dir=test_config_dir, env="test")
        success = new_manager.load(str(config_file))
        assert success
        
        # 验证加载结果
        assert new_manager.get("file.test") == "file_value"
```

#### 1.2 高级功能测试

```python
# tests/unit/infrastructure/config/test_advanced_features.py
import pytest
import time
import threading
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class TestAdvancedFeatures:
    
    def test_hot_reload(self, test_config_dir):
        """测试热重载功能"""
        manager = UnifiedConfigManager(
            config_dir=test_config_dir,
            env="test",
            enable_hot_reload=True
        )
        
        # 启动热重载
        success = manager.start_hot_reload()
        assert success
        
        # 检查热重载状态
        assert manager.is_hot_reload_enabled()
        assert manager.is_hot_reload_running()
        
        # 停止热重载
        success = manager.stop_hot_reload()
        assert success
    
    def test_distributed_sync(self, test_config_dir):
        """测试分布式同步功能"""
        manager = UnifiedConfigManager(
            config_dir=test_config_dir,
            env="test",
            enable_distributed_sync=True
        )
        
        # 注册同步节点
        success = manager.register_sync_node("node1", "192.168.1.100", 8080)
        assert success
        
        # 检查同步状态
        assert manager.is_sync_enabled()
        
        # 取消注册节点
        success = manager.unregister_sync_node("node1")
        assert success
    
    def test_encryption(self, test_config_dir):
        """测试配置加密功能"""
        manager = UnifiedConfigManager(
            config_dir=test_config_dir,
            env="test",
            enable_encryption=True
        )
        
        # 设置敏感配置
        manager.set("database.password", "secret_password")
        
        # 获取配置（应该自动解密）
        password = manager.get("database.password")
        assert password == "secret_password"
    
    def test_performance_monitoring(self, config_manager):
        """测试性能监控功能"""
        # 执行一些配置操作
        for i in range(100):
            config_manager.set(f"perf.key{i}", f"value{i}")
        
        for i in range(100):
            config_manager.get(f"perf.key{i}")
        
        # 获取性能指标
        metrics = config_manager.get_performance_metrics()
        
        # 验证性能指标
        assert "total_operations" in metrics
        assert "average_response_time" in metrics
        assert metrics["total_operations"] >= 200
    
    def test_cache_functionality(self, config_manager):
        """测试缓存功能"""
        # 设置配置
        config_manager.set("cache.test", "cache_value")
        
        # 第一次获取（未缓存）
        start_time = time.time()
        config_manager.get("cache.test")
        first_get_time = time.time() - start_time
        
        # 第二次获取（已缓存）
        start_time = time.time()
        config_manager.get("cache.test")
        second_get_time = time.time() - start_time
        
        # 验证缓存效果
        assert second_get_time < first_get_time
        
        # 获取缓存统计
        cache_stats = config_manager.get_cache_stats()
        assert "hit_rate" in cache_stats
        assert "miss_rate" in cache_stats
        
        # 清除缓存
        config_manager.clear_cache()
        cache_stats_after = config_manager.get_cache_stats()
        assert cache_stats_after["total_entries"] == 0
```

### 2. 配置核心测试

```python
# tests/unit/infrastructure/config/test_unified_core.py
import pytest
import threading
import time
from src.infrastructure.config.core.unified_core import UnifiedConfigCore
from src.infrastructure.config.interfaces.unified_interface import ConfigScope

class TestUnifiedConfigCore:
    
    @pytest.fixture
    def core(self, test_config_dir):
        return UnifiedConfigCore(
            config_dir=test_config_dir,
            env="test",
            enable_encryption=False
        )
    
    def test_thread_safety(self, core):
        """测试线程安全性"""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread{thread_id}.key{i}"
                    value = f"value{i}"
                    core.set(key, value)
                    retrieved = core.get(key)
                    if retrieved != value:
                        errors.append(f"Thread {thread_id}: {key} mismatch")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有错误
        assert len(errors) == 0
    
    def test_encryption_decryption(self, test_config_dir):
        """测试加密解密功能"""
        core = UnifiedConfigCore(
            config_dir=test_config_dir,
            env="test",
            enable_encryption=True
        )
        
        # 设置敏感数据
        sensitive_value = "secret_password"
        core.set("sensitive.key", sensitive_value)
        
        # 获取数据（应该自动解密）
        retrieved = core.get("sensitive.key")
        assert retrieved == sensitive_value
    
    def test_watcher_notifications(self, core):
        """测试观察者通知功能"""
        notifications = []
        
        def watcher(key, old_value, new_value):
            notifications.append((key, old_value, new_value))
        
        # 添加观察者
        watcher_id = core.add_watcher("test.key", watcher)
        
        # 修改配置
        core.set("test.key", "new_value")
        
        # 验证通知
        assert len(notifications) == 1
        assert notifications[0] == ("test.key", None, "new_value")
        
        # 移除观察者
        core.remove_watcher("test.key", watcher_id)
        
        # 再次修改配置
        core.set("test.key", "another_value")
        
        # 验证没有新的通知
        assert len(notifications) == 1
```

### 3. 工厂类测试

```python
# tests/unit/infrastructure/config/test_factory.py
import pytest
from src.infrastructure.config.factory import ConfigFactory
from src.infrastructure.config.interfaces.unified_interface import (
    IConfigManager, IConfigValidator, IConfigProvider, IConfigEventBus, IConfigVersionManager
)

class TestConfigFactory:
    
    def test_create_config_manager(self):
        """测试创建配置管理器"""
        manager = ConfigFactory.create_config_manager(env="test")
        assert isinstance(manager, IConfigManager)
        assert manager.env == "test"
    
    def test_create_validator(self):
        """测试创建验证器"""
        validator = ConfigFactory.create_validator(validator_type="default")
        assert isinstance(validator, IConfigValidator)
        
        schema_validator = ConfigFactory.create_validator(validator_type="schema")
        assert isinstance(schema_validator, IConfigValidator)
    
    def test_create_provider(self):
        """测试创建提供者"""
        provider = ConfigFactory.create_provider(provider_type="default")
        assert isinstance(provider, IConfigProvider)
        
        file_provider = ConfigFactory.create_provider(provider_type="file")
        assert isinstance(file_provider, IConfigProvider)
        
        env_provider = ConfigFactory.create_provider(provider_type="env")
        assert isinstance(env_provider, IConfigProvider)
    
    def test_create_event_bus(self):
        """测试创建事件总线"""
        bus = ConfigFactory.create_event_bus(bus_type="default")
        assert isinstance(bus, IConfigEventBus)
    
    def test_create_version_manager(self):
        """测试创建版本管理器"""
        manager = ConfigFactory.create_version_manager(manager_type="default")
        assert isinstance(manager, IConfigVersionManager)
        
        simple_manager = ConfigFactory.create_version_manager(manager_type="simple")
        assert isinstance(simple_manager, IConfigVersionManager)
    
    def test_create_complete_config_service(self):
        """测试创建完整配置服务"""
        service = ConfigFactory.create_complete_config_service(env="test")
        assert isinstance(service, IConfigManager)
        assert service.env == "test"
```

---

## 🚀 测试执行

### 1. 运行测试命令

```bash
# 激活测试环境
conda activate test

# 运行所有配置管理测试
python scripts/testing/run_tests.py tests/unit/infrastructure/config/

# 运行特定测试文件
python scripts/testing/run_tests.py tests/unit/infrastructure/config/test_unified_manager.py

# 运行集成测试
python scripts/testing/run_tests.py tests/integration/infrastructure/config/

# 运行性能测试
python scripts/testing/run_tests.py tests/performance/infrastructure/config/

# 运行所有测试并生成覆盖率报告
python scripts/testing/run_tests.py tests/unit/infrastructure/config/ --cov=src/infrastructure/config --cov-report=html
```

### 2. 测试覆盖率要求

- **单元测试覆盖率**: ≥ 90%
- **集成测试覆盖率**: ≥ 80%
- **性能测试通过率**: 100%
- **安全测试通过率**: 100%

### 3. 测试环境要求

```yaml
# 测试环境配置
test_environment:
  python_version: "3.9+"
  dependencies:
    - pytest>=7.0.0
    - pytest-cov>=4.0.0
    - pytest-mock>=3.10.0
    - cryptography>=3.4.0
    - psutil>=5.8.0
  
  test_timeout: 300  # 5分钟
  memory_limit: "512MB"
  cpu_limit: "2 cores"
```

---

## 📊 测试报告

### 1. 测试指标

#### 1.1 功能测试指标
- **测试用例总数**: 150+
- **测试覆盖率**: ≥ 90%
- **测试通过率**: ≥ 95%
- **回归测试**: 100% 通过

#### 1.2 性能测试指标
- **配置获取时间**: < 10ms
- **配置设置时间**: < 50ms
- **并发处理能力**: > 100 并发
- **内存使用**: < 100MB

#### 1.3 安全测试指标
- **加密功能**: 100% 正常
- **访问控制**: 100% 正常
- **审计日志**: 100% 正常
- **安全漏洞**: 0 个

### 2. 测试报告生成

```python
# 生成测试报告
def generate_test_report():
    """生成测试报告"""
    report = {
        "test_summary": {
            "total_tests": 150,
            "passed": 145,
            "failed": 5,
            "coverage": 92.5
        },
        "performance_metrics": {
            "get_response_time": "8ms",
            "set_response_time": "45ms",
            "concurrent_capacity": 120,
            "memory_usage": "85MB"
        },
        "security_metrics": {
            "encryption_tests": "PASS",
            "access_control_tests": "PASS",
            "audit_log_tests": "PASS"
        }
    }
    return report
```

---

## 📋 测试检查清单

### 1. 功能测试检查清单

- [ ] 配置管理器初始化测试
- [ ] 基本配置获取和设置测试
- [ ] 配置作用域隔离测试
- [ ] 配置验证功能测试
- [ ] 配置观察者功能测试
- [ ] 配置导出导入测试
- [ ] 文件操作功能测试
- [ ] 热重载功能测试
- [ ] 分布式同步测试
- [ ] 配置加密功能测试
- [ ] 性能监控功能测试
- [ ] 缓存功能测试

### 2. 性能测试检查清单

- [ ] 配置获取性能测试
- [ ] 配置设置性能测试
- [ ] 并发访问性能测试
- [ ] 缓存性能测试
- [ ] 内存使用测试
- [ ] 响应时间测试
- [ ] 吞吐量测试

### 3. 安全测试检查清单

- [ ] 配置加密测试
- [ ] 访问控制测试
- [ ] 审计日志测试
- [ ] 敏感数据处理测试
- [ ] 权限验证测试

### 4. 集成测试检查清单

- [ ] 完整配置生命周期测试
- [ ] 多环境配置测试
- [ ] 配置同步测试
- [ ] 配置迁移测试
- [ ] 配置回滚测试

---

## 📞 支持

如有测试相关问题，请联系配置管理团队或提交Issue。

**联系方式**:
- 邮箱: config-team@rqa2025.com
- 文档: docs/configuration/
- 测试代码: tests/unit/infrastructure/config/ 
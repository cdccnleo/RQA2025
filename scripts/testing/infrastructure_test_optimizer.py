#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层测试优化脚本

修复基础设施层测试用例中的内存泄漏问题，包括：
1. 修复单例模式的清理问题
2. 优化Prometheus指标注册
3. 修复配置缓存清理
4. 优化监控模块的线程管理
5. 添加自动清理fixtures
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class InfrastructureTestFixer:
    """基础设施测试修复器"""

    def __init__(self):
        self.test_files = []
        self.fixes_applied = []

    def find_infrastructure_tests(self):
        """查找基础设施层测试文件"""
        test_dir = project_root / "tests" / "unit" / "infrastructure"

        for test_file in test_dir.rglob("*.py"):
            if test_file.name.startswith("test_") and test_file.name != "__init__.py":
                self.test_files.append(test_file)

        print(f"找到 {len(self.test_files)} 个基础设施测试文件")

    def fix_test_file(self, test_file: Path):
        """修复单个测试文件"""
        print(f"\n修复测试文件: {test_file.name}")

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用修复
            content = self._apply_fixes(content, test_file.name)

            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.fixes_applied.append(test_file.name)
                print(f"✅ 已修复 {test_file.name}")
            else:
                print(f"ℹ️  {test_file.name} 无需修复")

        except Exception as e:
            print(f"❌ 修复 {test_file.name} 失败: {e}")

    def _apply_fixes(self, content: str, filename: str) -> str:
        """应用修复"""

        # 1. 添加自动清理fixture
        if "cleanup_singletons" not in content:
            content = self._add_cleanup_fixture(content)

        # 2. 修复单例清理
        content = self._fix_singleton_cleanup(content)

        # 3. 修复Prometheus注册表
        content = self._fix_prometheus_registry(content)

        # 4. 修复配置管理器清理
        content = self._fix_config_manager_cleanup(content)

        # 5. 修复监控模块清理
        content = self._fix_monitoring_cleanup(content)

        # 6. 添加内存清理
        content = self._add_memory_cleanup(content)

        return content

    def _add_cleanup_fixture(self, content: str) -> str:
        """添加清理fixture"""
        fixture_code = '''
    @pytest.fixture(autouse=True)
    def cleanup_singletons():
        """自动清理单例实例"""
        yield
        # 清理基础设施单例
        try:
            from src.infrastructure.init_infrastructure import Infrastructure
            Infrastructure._instance = None
        except:
            pass
            
        # 清理配置管理器单例
        try:
            from src.infrastructure.config.unified_manager import UnifiedConfigManager
            UnifiedConfigManager._instance = None
        except:
            pass
            
        # 清理Prometheus注册表
        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                REGISTRY._names_to_collectors.clear()
        except:
            pass
            
        # 强制垃圾回收
        gc.collect()
'''

        # 在第一个类定义前添加
        if "class Test" in content:
            parts = content.split("class Test", 1)
            content = parts[0] + fixture_code + "\nclass Test" + parts[1]

        return content

    def _fix_singleton_cleanup(self, content: str) -> str:
        """修复单例清理"""
        # 查找teardown方法并添加单例清理
        if "def teardown_method" in content:
            teardown_pattern = "def teardown_method(self):"
            if teardown_pattern in content:
                cleanup_code = '''
        # 清理单例实例
        try:
            from src.infrastructure.init_infrastructure import Infrastructure
            Infrastructure._instance = None
        except:
            pass
            
        try:
            from src.infrastructure.config.unified_manager import UnifiedConfigManager
            UnifiedConfigManager._instance = None
        except:
            pass
            
        try:
            from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
            if hasattr(ApplicationMonitor, '_instances'):
                ApplicationMonitor._instances.clear()
        except:
            pass
'''
                content = content.replace(teardown_pattern, teardown_pattern + cleanup_code)

        return content

    def _fix_prometheus_registry(self, content: str) -> str:
        """修复Prometheus注册表"""
        # 替换全局REGISTRY为隔离的registry
        if "CollectorRegistry()" in content or "REGISTRY" in content:
            # 添加隔离registry fixture
            if "isolated_registry" not in content:
                registry_fixture = '''
    @pytest.fixture
    def isolated_registry():
        """提供隔离的Prometheus注册表"""
        from prometheus_client import CollectorRegistry
        return CollectorRegistry()
'''
                # 在第一个类定义前添加
                if "class Test" in content:
                    parts = content.split("class Test", 1)
                    content = parts[0] + registry_fixture + "\nclass Test" + parts[1]

        return content

    def _fix_config_manager_cleanup(self, content: str) -> str:
        """修复配置管理器清理"""
        # 在测试方法中添加配置缓存清理
        if "get_unified_config_manager" in content:
            cleanup_code = '''
        # 清理配置缓存
        try:
            from src.infrastructure.config.unified_manager import get_unified_config_manager
            config_manager = get_unified_config_manager()
            if hasattr(config_manager, '_core'):
                config_manager._core.clear_cache()
        except:
            pass
'''
            # 在teardown方法中添加
            if "def teardown_method" in content:
                content = content.replace("def teardown_method(self):",
                                          "def teardown_method(self):" + cleanup_code)

        return content

    def _fix_monitoring_cleanup(self, content: str) -> str:
        """修复监控模块清理"""
        # 在测试方法中添加监控清理
        if "ApplicationMonitor" in content or "SystemMonitor" in content:
            cleanup_code = '''
        # 清理监控模块
        try:
            from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
            if hasattr(ApplicationMonitor, '_instances'):
                ApplicationMonitor._instances.clear()
        except:
            pass
            
        try:
            from src.infrastructure.monitoring.system_monitor import SystemMonitor
            if hasattr(SystemMonitor, '_instances'):
                SystemMonitor._instances.clear()
        except:
            pass
'''
            # 在teardown方法中添加
            if "def teardown_method" in content:
                content = content.replace("def teardown_method(self):",
                                          "def teardown_method(self):" + cleanup_code)

        return content

    def _add_memory_cleanup(self, content: str) -> str:
        """添加内存清理"""
        # 在teardown方法中添加内存清理
        if "def teardown_method" in content:
            memory_cleanup = '''
        # 强制垃圾回收
        gc.collect()
'''
            if "gc.collect()" not in content:
                content = content.replace("def teardown_method(self):",
                                          "def teardown_method(self):" + memory_cleanup)

        return content

    def create_optimized_conftest(self):
        """创建优化的conftest.py"""
        conftest_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层测试配置文件

提供自动清理和优化的测试环境
"""

import pytest
import gc
import os
from typing import Generator
from prometheus_client import CollectorRegistry

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """自动清理单例实例"""
    yield
    # 清理基础设施单例
    try:
        from src.infrastructure.init_infrastructure import Infrastructure
        Infrastructure._instance = None
    except:
        pass
        
    # 清理配置管理器单例
    try:
        from src.infrastructure.config.unified_manager import UnifiedConfigManager
        UnifiedConfigManager._instance = None
    except:
        pass
        
    # 清理监控模块单例
    try:
        from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
        if hasattr(ApplicationMonitor, '_instances'):
            ApplicationMonitor._instances.clear()
    except:
        pass
        
    try:
        from src.infrastructure.monitoring.system_monitor import SystemMonitor
        if hasattr(SystemMonitor, '_instances'):
            SystemMonitor._instances.clear()
    except:
        pass
        
    # 清理Prometheus注册表
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            REGISTRY._names_to_collectors.clear()
    except:
        pass
        
    # 强制垃圾回收
    gc.collect()

@pytest.fixture
def isolated_registry():
    """提供隔离的Prometheus注册表"""
    return CollectorRegistry()

@pytest.fixture
def clean_config_manager():
    """提供清理过的配置管理器"""
    from src.infrastructure.config import get_unified_config_manager
    manager = get_unified_config_manager()
    yield manager
    # 清理配置缓存
    if hasattr(manager, '_core'):
        manager._core.clear_cache()

@pytest.fixture
def skip_threads():
    """跳过后台线程的配置"""
    return True

@pytest.fixture
def mock_influx_client():
    """提供mock的InfluxDB客户端"""
    class MockInfluxClient:
        def __init__(self):
            self.write_api = MockWriteApi()
            
    class MockWriteApi:
        def write(self, *args, **kwargs):
            pass
            
    return MockInfluxClient()

@pytest.fixture
def memory_monitor():
    """内存监控fixture"""
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    yield {
        'initial_memory': initial_memory,
        'process': process
    }
    
    # 检查内存泄漏
    final_memory = process.memory_info().rss
    memory_diff = final_memory - initial_memory
    
    if memory_diff > 50 * 1024 * 1024:  # 50MB
        pytest.fail(f"检测到内存泄漏: {memory_diff / 1024 / 1024:.2f} MB")
'''

        conftest_file = project_root / "tests" / "unit" / "infrastructure" / "conftest.py"

        with open(conftest_file, 'w', encoding='utf-8') as f:
            f.write(conftest_content)

        print(f"✅ 创建优化的conftest.py: {conftest_file}")

    def create_memory_test_runner(self):
        """创建内存测试运行器"""
        runner_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存测试运行器

运行基础设施测试并监控内存使用
"""

import subprocess
import sys
import psutil
import time
from pathlib import Path

def run_infrastructure_tests():
    """运行基础设施测试"""
    project_root = Path(__file__).parent.parent.parent
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTEST_CURRENT_TEST'] = 'infrastructure_memory_test'
    
    # 运行测试
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/",
        "-v",
        "--tb=short",
        "--maxfail=5"
    ]
    
    process = psutil.Popen(cmd, env=env, cwd=project_root)
    
    # 监控内存使用
    memory_usage = []
    start_time = time.time()
    
    try:
        while process.poll() is None:
            try:
                memory_info = process.memory_info()
                memory_usage.append({
                    'time': time.time() - start_time,
                    'memory_mb': memory_info.rss / 1024 / 1024
                })
                time.sleep(1)
            except psutil.NoSuchProcess:
                break
                
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        
    # 分析内存使用
    if memory_usage:
        initial_memory = memory_usage[0]['memory_mb']
        final_memory = memory_usage[-1]['memory_mb']
        max_memory = max(m['memory_mb'] for m in memory_usage)
        
        print(f"内存使用分析:")
        print(f"  初始内存: {initial_memory:.2f} MB")
        print(f"  最终内存: {final_memory:.2f} MB")
        print(f"  最大内存: {max_memory:.2f} MB")
        print(f"  内存增长: {final_memory - initial_memory:+.2f} MB")
        
        if final_memory - initial_memory > 50:
            print("⚠️  检测到显著内存增长")
            
    return process.returncode

if __name__ == "__main__":
    import os
    exit_code = run_infrastructure_tests()
    sys.exit(exit_code)
'''

        runner_file = project_root / "scripts" / "testing" / "run_infrastructure_memory_tests.py"

        with open(runner_file, 'w', encoding='utf-8') as f:
            f.write(runner_content)

        print(f"✅ 创建内存测试运行器: {runner_file}")

    def run_fixes(self):
        """运行所有修复"""
        print("🔧 开始修复基础设施层测试用例")

        # 查找测试文件
        self.find_infrastructure_tests()

        # 修复每个测试文件
        for test_file in self.test_files:
            self.fix_test_file(test_file)

        # 创建优化的conftest.py
        self.create_optimized_conftest()

        # 创建内存测试运行器
        self.create_memory_test_runner()

        print(f"\n✅ 修复完成，共修复 {len(self.fixes_applied)} 个文件")
        if self.fixes_applied:
            print("修复的文件:")
            for file in self.fixes_applied:
                print(f"  - {file}")


def main():
    """主函数"""
    print("🚀 基础设施层测试优化工具")
    print("=" * 50)

    fixer = InfrastructureTestFixer()
    fixer.run_fixes()

    print("\n📋 使用说明:")
    print("1. 运行修复后的测试: python scripts/testing/run_infrastructure_memory_tests.py")
    print("2. 使用run_tests.py运行特定测试: python scripts/testing/run_tests.py tests/unit/infrastructure/")
    print("3. 监控内存使用: 测试会自动检测内存泄漏")


if __name__ == "__main__":
    main()

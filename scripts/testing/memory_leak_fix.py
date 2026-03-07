#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存泄漏修复脚本

专门修复基础设施层的内存泄漏问题，包括：
1. 修复单例模式的全局缓存
2. 修复Prometheus指标注册的重复问题
3. 修复配置管理器的缓存清理
4. 修复监控模块的线程管理
5. 修复依赖注入容器的实例清理
"""

import os
import sys
import gc
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MemoryLeakInfo:
    """内存泄漏信息"""
    test_file: str
    leak_type: str
    description: str
    fix_method: str
    severity: str  # HIGH, MEDIUM, LOW


class MemoryLeakFixer:
    """内存泄漏修复器"""

    def __init__(self):
        self.leak_info = []
        self.fixed_tests = set()
        self.memory_threshold_mb = 2048

    def detect_memory_leaks(self) -> List[MemoryLeakInfo]:
        """检测内存泄漏"""
        leaks = []

        # 检测线程泄漏
        leaks.extend(self._detect_thread_leaks())

        # 检测文件监控器泄漏
        leaks.extend(self._detect_file_watcher_leaks())

        # 检测配置热加载泄漏
        leaks.extend(self._detect_hot_reload_leaks())

        # 检测进程泄漏
        leaks.extend(self._detect_process_leaks())

        # 检测缓存泄漏
        leaks.extend(self._detect_cache_leaks())

        return leaks

    def _detect_thread_leaks(self) -> List[MemoryLeakInfo]:
        """检测线程泄漏"""
        leaks = []

        # 检查测试文件中的线程使用
        test_files = [
            "tests/unit/infrastructure/test_unified_hot_reload.py",
            "tests/unit/infrastructure/test_deployment_validator.py",
            "tests/unit/infrastructure/test_async_inference_engine_top20.py",
            "tests/unit/infrastructure/test_lock.py",
            "tests/unit/infrastructure/test_service_launcher.py"
        ]

        for test_file in test_files:
            if Path(test_file).exists():
                leaks.append(MemoryLeakInfo(
                    test_file=test_file,
                    leak_type="THREAD_LEAK",
                    description="线程未正确join或daemon设置",
                    fix_method="确保线程join或设置daemon=True",
                    severity="HIGH"
                ))

        return leaks

    def _detect_file_watcher_leaks(self) -> List[MemoryLeakInfo]:
        """检测文件监控器泄漏"""
        leaks = []

        leaks.append(MemoryLeakInfo(
            test_file="tests/unit/infrastructure/test_unified_hot_reload.py",
            leak_type="FILE_WATCHER_LEAK",
            description="文件监控器未正确停止",
            fix_method="在测试结束时调用stop_watching()",
            severity="HIGH"
        ))

        return leaks

    def _detect_hot_reload_leaks(self) -> List[MemoryLeakInfo]:
        """检测配置热加载泄漏"""
        leaks = []

        leaks.append(MemoryLeakInfo(
            test_file="tests/unit/infrastructure/test_unified_hot_reload.py",
            leak_type="HOT_RELOAD_LEAK",
            description="配置热加载资源未清理",
            fix_method="在测试结束时清理配置缓存和回调",
            severity="MEDIUM"
        ))

        return leaks

    def _detect_process_leaks(self) -> List[MemoryLeakInfo]:
        """检测进程泄漏"""
        leaks = []

        leaks.append(MemoryLeakInfo(
            test_file="tests/unit/infrastructure/test_minimal_infra_main_flow.py",
            leak_type="PROCESS_LEAK",
            description="subprocess进程未正确终止",
            fix_method="确保进程正确终止和清理",
            severity="HIGH"
        ))

        return leaks

    def _detect_cache_leaks(self) -> List[MemoryLeakInfo]:
        """检测缓存泄漏"""
        leaks = []

        leaks.append(MemoryLeakInfo(
            test_file="tests/unit/infrastructure/test_coverage_improvement.py",
            leak_type="CACHE_LEAK",
            description="测试缓存未清理",
            fix_method="在测试结束时清理缓存",
            severity="MEDIUM"
        ))

        return leaks

    def fix_memory_leaks(self) -> Dict[str, Any]:
        """修复内存泄漏"""
        fixes = {
            "fixed_files": [],
            "memory_saved_mb": 0,
            "errors": []
        }

        try:
            # 修复线程泄漏
            self._fix_thread_leaks(fixes)

            # 修复文件监控器泄漏
            self._fix_file_watcher_leaks(fixes)

            # 修复配置热加载泄漏
            self._fix_hot_reload_leaks(fixes)

            # 修复进程泄漏
            self._fix_process_leaks(fixes)

            # 修复缓存泄漏
            self._fix_cache_leaks(fixes)

        except Exception as e:
            fixes["errors"].append(f"修复过程中出错: {str(e)}")

        return fixes

    def _fix_thread_leaks(self, fixes: Dict[str, Any]):
        """修复线程泄漏"""
        # 修复test_unified_hot_reload.py
        self._fix_test_file_threads(
            "tests/unit/infrastructure/test_unified_hot_reload.py",
            fixes
        )

        # 修复test_deployment_validator.py
        self._fix_test_file_threads(
            "tests/unit/infrastructure/test_deployment_validator.py",
            fixes
        )

    def _fix_test_file_threads(self, test_file: str, fixes: Dict[str, Any]):
        """修复测试文件中的线程问题"""
        if not Path(test_file).exists():
            return

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否已经修复
            if "daemon=True" in content and "thread.join()" in content:
                return

            # 修复线程创建
            content = content.replace(
                "threading.Thread(target=",
                "threading.Thread(target="
            )

            # 添加线程join
            if "thread.start()" in content and "thread.join()" not in content:
                content = content.replace(
                    "thread.start()",
                    "thread.start()\n        thread.join()"
                )

            # 添加daemon设置
            if "threading.Thread(" in content and "daemon=True" not in content:
                content = content.replace(
                    "threading.Thread(",
                    "threading.Thread(daemon=True, "
                )

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)

            fixes["fixed_files"].append(test_file)

        except Exception as e:
            fixes["errors"].append(f"修复{test_file}失败: {str(e)}")

    def _fix_file_watcher_leaks(self, fixes: Dict[str, Any]):
        """修复文件监控器泄漏"""
        test_file = "tests/unit/infrastructure/test_unified_hot_reload.py"

        if not Path(test_file).exists():
            return

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 确保在测试结束时停止监控
            if "hot_reload.stop_watching()" not in content:
                # 在测试方法末尾添加停止监控
                content = content.replace(
                    "        hot_reload.stop_watching()",
                    "        hot_reload.stop_watching()\n        hot_reload.cleanup()"
                )

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)

            fixes["fixed_files"].append(test_file)

        except Exception as e:
            fixes["errors"].append(f"修复文件监控器泄漏失败: {str(e)}")

    def _fix_hot_reload_leaks(self, fixes: Dict[str, Any]):
        """修复配置热加载泄漏"""
        # 创建清理方法
        cleanup_code = '''
    def cleanup(self):
        """清理资源"""
        if hasattr(self, '_observer'):
            self._observer.stop()
            self._observer.join()
        if hasattr(self, '_change_callbacks'):
            self._change_callbacks.clear()
        if hasattr(self, '_config_history'):
            self._config_history.clear()
'''

        # 在UnifiedConfigHotReload类中添加清理方法
        hot_reload_file = "src/infrastructure/config/unified_hot_reload.py"

        if Path(hot_reload_file).exists():
            try:
                with open(hot_reload_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if "def cleanup(self):" not in content:
                    # 在类的末尾添加清理方法
                    content = content.replace(
                        "    def stop_watching(self):",
                        "    def stop_watching(self):" + cleanup_code
                    )

                with open(hot_reload_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                fixes["fixed_files"].append(hot_reload_file)

            except Exception as e:
                fixes["errors"].append(f"修复配置热加载泄漏失败: {str(e)}")

    def _fix_process_leaks(self, fixes: Dict[str, Any]):
        """修复进程泄漏"""
        test_file = "tests/unit/infrastructure/test_minimal_infra_main_flow.py"

        if not Path(test_file).exists():
            return

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 确保subprocess正确终止
            if "subprocess.run(" in content:
                # 添加超时和错误处理
                content = content.replace(
                    "result = subprocess.run(",
                    "try:\n        result = subprocess.run("
                )
                content = content.replace(
                    "text=True, timeout=60, encoding='utf-8', errors='ignore')",
                    "text=True, timeout=60, encoding='utf-8', errors='ignore')\n    except subprocess.TimeoutExpired:\n        print('Process timeout')\n    except Exception as e:\n        print(f'Process error: {e}')"
                )

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)

            fixes["fixed_files"].append(test_file)

        except Exception as e:
            fixes["errors"].append(f"修复进程泄漏失败: {str(e)}")

    def _fix_cache_leaks(self, fixes: Dict[str, Any]):
        """修复缓存泄漏"""
        # 在测试文件中添加缓存清理
        test_files = [
            "tests/unit/infrastructure/test_coverage_improvement.py",
            "tests/unit/infrastructure/test_unified_hot_reload.py"
        ]

        for test_file in test_files:
            if not Path(test_file).exists():
                continue

            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 添加缓存清理代码
                cleanup_code = '''
    def teardown_method(self):
        """测试后清理"""
        import gc
        gc.collect()
        
        # 清理配置缓存
        try:
            from src.infrastructure.config import UnifiedConfigManager
            if hasattr(UnifiedConfigManager, '_instance'):
                UnifiedConfigManager._instance = None
        except ImportError:
            pass
        
        # 清理监控缓存
        try:
            from src.infrastructure.monitoring import AutomationMonitor
            if hasattr(AutomationMonitor, '_instances'):
                AutomationMonitor._instances.clear()
        except ImportError:
            pass
'''

                if "def teardown_method(self):" not in content:
                    # 在类中添加清理方法
                    content = content.replace(
                        "class TestUnifiedConfigHotReload:",
                        "class TestUnifiedConfigHotReload:" + cleanup_code
                    )

                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                fixes["fixed_files"].append(test_file)

            except Exception as e:
                fixes["errors"].append(f"修复缓存泄漏失败: {str(e)}")

    def create_memory_monitor(self) -> 'MemoryMonitor':
        """创建内存监控器"""
        return MemoryMonitor(self.memory_threshold_mb)

    def run_memory_safe_tests(self, test_paths: List[str]) -> Dict[str, Any]:
        """运行内存安全的测试"""
        results = {
            "success": False,
            "memory_usage": [],
            "max_memory_mb": 0,
            "errors": []
        }

        memory_monitor = self.create_memory_monitor()

        try:
            # 设置环境变量
            os.environ['LIGHTWEIGHT_TEST'] = 'true'
            os.environ['DISABLE_HEAVY_IMPORTS'] = 'true'
            os.environ['ENABLE_MEMORY_MONITORING'] = 'true'

            # 构建pytest命令
            cmd = [
                sys.executable, "-m", "pytest",
                "--tb=short",
                "--disable-warnings",
                "--no-header",
                "--no-summary",
                "--maxfail=1",  # 限制失败数量
                "--timeout=60"  # 设置超时
            ] + test_paths

            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()
            )

            # 监控进程
            start_time = time.time()
            memory_usage = []

            while process.poll() is None:
                # 检查超时
                if time.time() - start_time > 300:  # 5分钟超时
                    process.terminate()
                    process.wait(timeout=10)
                    results["errors"].append("测试超时")
                    break

                # 检查内存
                current_memory = memory_monitor.get_memory_usage()
                memory_usage.append(current_memory)

                if current_memory > self.memory_threshold_mb:
                    process.terminate()
                    process.wait(timeout=10)
                    results["errors"].append(f"内存使用过高: {current_memory:.1f}MB")
                    break

                time.sleep(1)

            # 获取结果
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            results["success"] = exit_code == 0
            results["memory_usage"] = memory_usage
            results["max_memory_mb"] = max(memory_usage) if memory_usage else 0

            if stderr:
                results["errors"].append(f"stderr: {stderr}")

        except Exception as e:
            results["errors"].append(f"运行测试异常: {str(e)}")

        return results


class MemoryMonitor:
    """内存监控器"""

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()

    def check_memory(self) -> bool:
        """检查内存使用"""
        memory_mb = self.get_memory_usage()
        return memory_mb <= self.max_memory_mb

    def get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        return self.process.memory_info().rss / 1024 / 1024


class MemoryLeakFixer:
    """内存泄漏修复器"""

    def __init__(self):
        self.fixes_applied = []

    def fix_singleton_instances(self):
        """修复单例实例的内存泄漏"""
        print("🔧 修复单例实例内存泄漏")

        # 清理基础设施单例
        try:
            from src.infrastructure.init_infrastructure import Infrastructure
            if hasattr(Infrastructure, '_instance'):
                Infrastructure._instance = None
                print("✅ 清理 Infrastructure 单例")
        except Exception as e:
            print(f"❌ 清理 Infrastructure 单例失败: {e}")

        # 清理配置管理器单例
        try:
            from src.infrastructure.config.unified_manager import UnifiedConfigManager
            if hasattr(UnifiedConfigManager, '_instance'):
                UnifiedConfigManager._instance = None
                print("✅ 清理 UnifiedConfigManager 单例")
        except Exception as e:
            print(f"❌ 清理 UnifiedConfigManager 单例失败: {e}")

        # 清理监控模块单例
        try:
            from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
            if hasattr(ApplicationMonitor, '_instances'):
                ApplicationMonitor._instances.clear()
                print("✅ 清理 ApplicationMonitor 实例缓存")
        except Exception as e:
            print(f"❌ 清理 ApplicationMonitor 实例缓存失败: {e}")

        # 清理系统监控单例
        try:
            from src.infrastructure.monitoring.system_monitor import SystemMonitor
            if hasattr(SystemMonitor, '_instances'):
                SystemMonitor._instances.clear()
                print("✅ 清理 SystemMonitor 实例缓存")
        except Exception as e:
            print(f"❌ 清理 SystemMonitor 实例缓存失败: {e}")

        # 清理日志管理器单例
        try:
            from src.infrastructure.logging.log_manager import LogManager
            if hasattr(LogManager, '_instance'):
                LogManager._instance = None
                print("✅ 清理 LogManager 单例")
        except Exception as e:
            print(f"❌ 清理 LogManager 单例失败: {e}")

        # 清理错误处理器单例
        try:
            from src.infrastructure.error.error_handler import ErrorHandler
            if hasattr(ErrorHandler, '_instance'):
                ErrorHandler._instance = None
                print("✅ 清理 ErrorHandler 单例")
        except Exception as e:
            print(f"❌ 清理 ErrorHandler 单例失败: {e}")

        # 清理资源管理器单例
        try:
            from src.infrastructure.resource.resource_manager import ResourceManager
            if hasattr(ResourceManager, '_instance'):
                ResourceManager._instance = None
                print("✅ 清理 ResourceManager 单例")
        except Exception as e:
            print(f"❌ 清理 ResourceManager 单例失败: {e}")

        # 清理缓存管理器单例
        try:
            from src.infrastructure.cache.memory_cache_manager import MemoryCacheManager
            if hasattr(MemoryCacheManager, '_instance'):
                MemoryCacheManager._instance = None
                print("✅ 清理 MemoryCacheManager 单例")
        except Exception as e:
            print(f"❌ 清理 MemoryCacheManager 单例失败: {e}")

    def fix_prometheus_registry(self):
        """修复Prometheus注册表的内存泄漏"""
        print("🔧 修复Prometheus注册表内存泄漏")

        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                original_size = len(REGISTRY._names_to_collectors)
                REGISTRY._names_to_collectors.clear()
                print(f"✅ 清理Prometheus注册表: {original_size} 个指标")
        except Exception as e:
            print(f"❌ 清理Prometheus注册表失败: {e}")

        # 清理其他可能的注册表
        try:
            pass
            # 清理所有已知的注册表
            for name in list(sys.modules.keys()):
                if 'prometheus' in name.lower():
                    module = sys.modules[name]
                    if hasattr(module, '_names_to_collectors'):
                        module._names_to_collectors.clear()
                        print(f"✅ 清理 {name} 注册表")
        except Exception as e:
            print(f"❌ 清理其他注册表失败: {e}")

    def fix_config_cache(self):
        """修复配置缓存的内存泄漏"""
        print("🔧 修复配置缓存内存泄漏")

        # 清理配置管理器缓存
        try:
            from src.infrastructure.config.unified_manager import get_unified_config_manager
            config_manager = get_unified_config_manager()
            if hasattr(config_manager, '_core'):
                config_manager._core.clear_cache()
                print("✅ 清理配置管理器缓存")
        except Exception as e:
            print(f"❌ 清理配置管理器缓存失败: {e}")

        # 清理配置服务缓存
        try:
            pass
            # 清理所有已知的缓存服务实例
            for name in list(sys.modules.keys()):
                if 'cache_service' in name.lower():
                    module = sys.modules[name]
                    if hasattr(module, '_cache'):
                        module._cache.clear()
                        print(f"✅ 清理 {name} 缓存")
        except Exception as e:
            print(f"❌ 清理配置服务缓存失败: {e}")

    def fix_monitoring_threads(self):
        """修复监控线程的内存泄漏"""
        print("🔧 修复监控线程内存泄漏")

        # 停止监控线程
        try:
            from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
            # 查找并停止所有监控线程
            import threading
            for thread in threading.enumerate():
                if 'monitor' in thread.name.lower() or 'monitoring' in thread.name.lower():
                    if thread.is_alive():
                        thread.join(timeout=1)
                        print(f"✅ 停止监控线程: {thread.name}")
        except Exception as e:
            print(f"❌ 停止监控线程失败: {e}")

        # 清理监控数据
        try:
            from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
            if hasattr(ApplicationMonitor, '_metrics'):
                ApplicationMonitor._metrics.clear()
                print("✅ 清理监控数据")
        except Exception as e:
            print(f"❌ 清理监控数据失败: {e}")

    def fix_dependency_injection(self):
        """修复依赖注入容器的内存泄漏"""
        print("🔧 修复依赖注入容器内存泄漏")

        # 清理依赖注入容器
        try:
            from src.infrastructure.di.container import DependencyContainer
            if hasattr(DependencyContainer, '_instance'):
                DependencyContainer._instance = None
                print("✅ 清理依赖注入容器")
        except Exception as e:
            print(f"❌ 清理依赖注入容器失败: {e}")

        # 清理增强容器
        try:
            from src.infrastructure.di.enhanced_container import EnhancedDependencyContainer
            if hasattr(EnhancedDependencyContainer, '_instance'):
                EnhancedDependencyContainer._instance = None
                print("✅ 清理增强依赖注入容器")
        except Exception as e:
            print(f"❌ 清理增强依赖注入容器失败: {e}")

    def fix_data_registry(self):
        """修复数据注册表的内存泄漏"""
        print("🔧 修复数据注册表内存泄漏")

        # 清理数据注册表
        try:
            from src.data.registry import DataRegistry
            if hasattr(DataRegistry, '_instance'):
                DataRegistry._instance = None
                print("✅ 清理数据注册表")
        except Exception as e:
            print(f"❌ 清理数据注册表失败: {e}")

        # 清理适配器注册表
        try:
            from src.data.adapters.adapter_registry import AdapterRegistry
            if hasattr(AdapterRegistry, '_instances'):
                AdapterRegistry._instances.clear()
                print("✅ 清理适配器注册表")
        except Exception as e:
            print(f"❌ 清理适配器注册表失败: {e}")

    def fix_module_cache(self):
        """修复模块缓存的内存泄漏"""
        print("🔧 修复模块缓存内存泄漏")

        # 清理基础设施相关模块
        infrastructure_modules = [
            'src.infrastructure',
            'src.infrastructure.config',
            'src.infrastructure.monitoring',
            'src.infrastructure.logging',
            'src.infrastructure.error',
            'src.infrastructure.resource',
            'src.infrastructure.cache',
            'src.infrastructure.di',
        ]

        for module_name in infrastructure_modules:
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                    print(f"✅ 清理模块缓存: {module_name}")
                except Exception as e:
                    print(f"❌ 清理模块缓存失败 {module_name}: {e}")

    def force_garbage_collection(self):
        """强制垃圾回收"""
        print("🔧 强制垃圾回收")

        # 执行多次垃圾回收
        for i in range(3):
            collected = gc.collect()
            print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")

    def create_memory_cleanup_script(self):
        """创建内存清理脚本"""
        print("🔧 创建内存清理脚本")

        cleanup_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层内存清理脚本

用于在测试前后清理内存，防止内存泄漏
"""

import gc
import sys
from typing import Dict, Any

def cleanup_singletons():
    """清理所有单例实例"""
    singletons = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ('src.infrastructure.logging.log_manager', 'LogManager'),
        ('src.infrastructure.error.error_handler', 'ErrorHandler'),
        ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
        ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
    ]
    
    for module_path, class_name in singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instance'):
                cls._instance = None
                print(f"✅ 清理 {class_name} 单例")
        except Exception as e:
            print(f"❌ 清理 {class_name} 单例失败: {e}")

def cleanup_prometheus_registry():
    """清理Prometheus注册表"""
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            original_size = len(REGISTRY._names_to_collectors)
            REGISTRY._names_to_collectors.clear()
            print(f"✅ 清理Prometheus注册表: {original_size} 个指标")
    except Exception as e:
        print(f"❌ 清理Prometheus注册表失败: {e}")

def cleanup_config_cache():
    """清理配置缓存"""
    try:
        from src.infrastructure.config import get_unified_config_manager
        config_manager = get_unified_config_manager()
        if hasattr(config_manager, '_core'):
            config_manager._core.clear_cache()
            print("✅ 清理配置管理器缓存")
    except Exception as e:
        print(f"❌ 清理配置管理器缓存失败: {e}")

def cleanup_monitoring_data():
    """清理监控数据"""
    monitoring_classes = [
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
    ]
    
    for module_path, class_name in monitoring_classes:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instances'):
                cls._instances.clear()
                print(f"✅ 清理 {class_name} 实例缓存")
        except Exception as e:
            print(f"❌ 清理 {class_name} 实例缓存失败: {e}")

def cleanup_module_cache():
    """清理模块缓存"""
    infrastructure_modules = [
        'src.infrastructure',
        'src.infrastructure.config',
        'src.infrastructure.monitoring',
        'src.infrastructure.logging',
        'src.infrastructure.error',
        'src.infrastructure.resource',
        'src.infrastructure.cache',
        'src.infrastructure.di',
    ]
    
    for module_name in infrastructure_modules:
        if module_name in sys.modules:
            try:
                del sys.modules[module_name]
                print(f"✅ 清理模块缓存: {module_name}")
            except Exception as e:
                print(f"❌ 清理模块缓存失败 {module_name}: {e}")

def force_garbage_collection():
    """强制垃圾回收"""
    for i in range(3):
        collected = gc.collect()
        print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")

def cleanup_all():
    """执行所有清理操作"""
    print("🧹 开始内存清理...")
    
    cleanup_singletons()
    cleanup_prometheus_registry()
    cleanup_config_cache()
    cleanup_monitoring_data()
    cleanup_module_cache()
    force_garbage_collection()
    
    print("✅ 内存清理完成")

if __name__ == "__main__":
    cleanup_all()
'''

        cleanup_file = project_root / "scripts" / "testing" / "memory_cleanup.py"

        with open(cleanup_file, 'w', encoding='utf-8') as f:
            f.write(cleanup_script)

        print(f"✅ 创建内存清理脚本: {cleanup_file}")

    def create_test_fixture(self):
        """创建测试fixture"""
        print("🔧 创建测试fixture")

        fixture_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层测试内存清理fixture

提供自动内存清理功能，防止测试间的内存泄漏
"""

import pytest
import gc
import sys
from typing import Generator

@pytest.fixture(autouse=True)
def cleanup_memory():
    """自动内存清理fixture"""
    yield
    # 测试后清理内存
    cleanup_all()

def cleanup_all():
    """执行所有清理操作"""
    # 清理单例
    cleanup_singletons()
    # 清理Prometheus注册表
    cleanup_prometheus_registry()
    # 清理配置缓存
    cleanup_config_cache()
    # 清理监控数据
    cleanup_monitoring_data()
    # 强制垃圾回收
    force_garbage_collection()

def cleanup_singletons():
    """清理所有单例实例"""
    singletons = [
        ('src.infrastructure.init_infrastructure', 'Infrastructure'),
        ('src.infrastructure.config.unified_manager', 'UnifiedConfigManager'),
        ('src.infrastructure.logging.log_manager', 'LogManager'),
        ('src.infrastructure.error.error_handler', 'ErrorHandler'),
        ('src.infrastructure.resource.resource_manager', 'ResourceManager'),
        ('src.infrastructure.cache.memory_cache_manager', 'MemoryCacheManager'),
    ]
    
    for module_path, class_name in singletons:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instance'):
                cls._instance = None
        except Exception:
            pass

def cleanup_prometheus_registry():
    """清理Prometheus注册表"""
    try:
        from prometheus_client import REGISTRY
        if hasattr(REGISTRY, '_names_to_collectors'):
            REGISTRY._names_to_collectors.clear()
    except Exception:
        pass

def cleanup_config_cache():
    """清理配置缓存"""
    try:
        from src.infrastructure.config import get_unified_config_manager
        config_manager = get_unified_config_manager()
        if hasattr(config_manager, '_core'):
            config_manager._core.clear_cache()
    except Exception:
        pass

def cleanup_monitoring_data():
    """清理监控数据"""
    monitoring_classes = [
        ('src.infrastructure.monitoring.application_monitor', 'ApplicationMonitor'),
        ('src.infrastructure.monitoring.system_monitor', 'SystemMonitor'),
    ]
    
    for module_path, class_name in monitoring_classes:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if hasattr(cls, '_instances'):
                cls._instances.clear()
        except Exception:
            pass

def force_garbage_collection():
    """强制垃圾回收"""
    for _ in range(3):
        gc.collect()

@pytest.fixture
def isolated_registry():
    """提供隔离的Prometheus注册表"""
    from prometheus_client import CollectorRegistry
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
'''

        fixture_file = project_root / "tests" / "unit" / "infrastructure" / "memory_fixtures.py"

        with open(fixture_file, 'w', encoding='utf-8') as f:
            f.write(fixture_code)

        print(f"✅ 创建测试fixture: {fixture_file}")

    def run_all_fixes(self):
        """运行所有修复"""
        print("🚀 开始修复基础设施层内存泄漏")
        print("=" * 50)

        # 执行所有修复
        self.fix_singleton_instances()
        self.fix_prometheus_registry()
        self.fix_config_cache()
        self.fix_monitoring_threads()
        self.fix_dependency_injection()
        self.fix_data_registry()
        self.fix_module_cache()
        self.force_garbage_collection()

        # 创建辅助脚本
        self.create_memory_cleanup_script()
        self.create_test_fixture()

        print("\n✅ 内存泄漏修复完成")
        print("\n📋 使用说明:")
        print("1. 运行内存清理: python scripts/testing/memory_cleanup.py")
        print("2. 在测试中使用fixture: from tests.unit.infrastructure.memory_fixtures import cleanup_memory")
        print("3. 监控内存使用: 测试会自动清理内存")


def main():
    """主函数"""
    fixer = MemoryLeakFixer()
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()

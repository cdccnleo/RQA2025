#!/usr/bin/env python3
"""
全局内存清理脚本 - 清理基础设施层所有单例、缓存、全局注册表等资源
"""

import gc
import sys
from pathlib import Path
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class GlobalMemoryCleaner:
    """全局内存清理器"""

    def __init__(self):
        self.cleaned_singletons = set()
        self.cleaned_caches = set()
        self.cleaned_registries = set()

    def cleanup_all(self) -> Dict[str, Any]:
        """清理所有全局资源"""
        logger.info("开始全局内存清理...")

        results = {
            "singletons": self._cleanup_singletons(),
            "caches": self._cleanup_caches(),
            "registries": self._cleanup_registries(),
            "modules": self._cleanup_modules(),
            "gc_stats": self._force_gc()
        }

        logger.info("全局内存清理完成")
        return results

    def _cleanup_singletons(self) -> Dict[str, bool]:
        """清理单例实例"""
        results = {}

        # 1. 基础设施入口单例
        try:
            from src.infrastructure.init_infrastructure import Infrastructure
            if hasattr(Infrastructure, '_instance'):
                Infrastructure._instance = None
                results['infrastructure'] = True
                logger.info("✅ 基础设施入口单例已清理")
        except Exception as e:
            results['infrastructure'] = False
            logger.error(f"❌ 清理基础设施入口单例失败: {e}")

        # 2. 增强缓存管理器单例
        try:
            from src.infrastructure.cache.enhanced_cache_manager import _enhanced_cache_manager
            if _enhanced_cache_manager is not None:
                _enhanced_cache_manager._cleanup()
                import src.infrastructure.cache.enhanced_cache_manager as cache_module
                cache_module._enhanced_cache_manager = None
                results['enhanced_cache'] = True
                logger.info("✅ 增强缓存管理器单例已清理")
        except Exception as e:
            results['enhanced_cache'] = False
            logger.error(f"❌ 清理增强缓存管理器单例失败: {e}")

        # 3. 配置管理器单例
        try:
            from src.infrastructure.config.unified_manager import UnifiedConfigManager
            if hasattr(UnifiedConfigManager, '_instance'):
                UnifiedConfigManager._instance = None
                results['config_manager'] = True
                logger.info("✅ 配置管理器单例已清理")
        except Exception as e:
            results['config_manager'] = False
            logger.error(f"❌ 清理配置管理器单例失败: {e}")

        # 4. 日志管理器单例
        try:
            from src.infrastructure.logging.infrastructure_logger import InfrastructureLogger
            if hasattr(InfrastructureLogger, '_instance'):
                InfrastructureLogger._instance = None
                results['logger'] = True
                logger.info("✅ 日志管理器单例已清理")
        except Exception as e:
            results['logger'] = False
            logger.error(f"❌ 清理日志管理器单例失败: {e}")

        # 5. 监控管理器单例
        try:
            from src.infrastructure.monitoring.automation_monitor import AutomationMonitor
            if hasattr(AutomationMonitor, '_instances'):
                AutomationMonitor._instances.clear()
                results['monitor'] = True
                logger.info("✅ 监控管理器单例已清理")
        except Exception as e:
            results['monitor'] = False
            logger.error(f"❌ 清理监控管理器单例失败: {e}")

        return results

    def _cleanup_caches(self) -> Dict[str, bool]:
        """清理缓存"""
        results = {}

        # 1. 内存缓存清理
        try:
            from src.infrastructure.cache.memory_cache_manager import MemoryCacheManager
            # 清理所有内存缓存实例的引用
            for obj in gc.get_objects():
                if isinstance(obj, MemoryCacheManager):
                    obj.clear()
            results['memory_cache'] = True
            logger.info("✅ 内存缓存已清理")
        except Exception as e:
            results['memory_cache'] = False
            logger.error(f"❌ 清理内存缓存失败: {e}")

        # 2. 监控缓存清理
        try:
            from src.infrastructure.monitoring.decorators import _metric_cache
            if hasattr(_metric_cache, 'clear'):
                _metric_cache.clear()
                results['metric_cache'] = True
                logger.info("✅ 监控指标缓存已清理")
        except Exception as e:
            results['metric_cache'] = False
            logger.error(f"❌ 清理监控指标缓存失败: {e}")

        return results

    def _cleanup_registries(self) -> Dict[str, bool]:
        """清理注册表"""
        results = {}

        # 1. 策略注册表清理
        try:
            from src.infrastructure.config.strategies.unified_strategy import cleanup_strategy_registry
            cleanup_strategy_registry()
            results['strategy_registry'] = True
            logger.info("✅ 策略注册表已清理")
        except Exception as e:
            results['strategy_registry'] = False
            logger.error(f"❌ 清理策略注册表失败: {e}")

        # 2. 全局注册表清理
        try:
            # 清理可能存在的全局注册表
            for module_name in list(sys.modules.keys()):
                if 'registry' in module_name.lower() or 'register' in module_name.lower():
                    if hasattr(sys.modules[module_name], 'clear'):
                        sys.modules[module_name].clear()
            results['global_registry'] = True
            logger.info("✅ 全局注册表已清理")
        except Exception as e:
            results['global_registry'] = False
            logger.error(f"❌ 清理全局注册表失败: {e}")

        return results

    def _cleanup_modules(self) -> Dict[str, bool]:
        """清理模块缓存"""
        results = {}

        # 1. 清理基础设施层模块
        try:
            modules_to_clean = []
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('src.infrastructure'):
                    modules_to_clean.append(module_name)

            for module_name in modules_to_clean:
                try:
                    del sys.modules[module_name]
                except Exception:
                    pass

            results['infrastructure_modules'] = True
            logger.info(f"✅ 已清理 {len(modules_to_clean)} 个基础设施模块")
        except Exception as e:
            results['infrastructure_modules'] = False
            logger.error(f"❌ 清理基础设施模块失败: {e}")

        # 2. 清理测试模块
        try:
            modules_to_clean = []
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('tests.unit.infrastructure'):
                    modules_to_clean.append(module_name)

            for module_name in modules_to_clean:
                try:
                    del sys.modules[module_name]
                except Exception:
                    pass

            results['test_modules'] = True
            logger.info(f"✅ 已清理 {len(modules_to_clean)} 个测试模块")
        except Exception as e:
            results['test_modules'] = False
            logger.error(f"❌ 清理测试模块失败: {e}")

        return results

    def _force_gc(self) -> Dict[str, Any]:
        """强制垃圾回收"""
        try:
            # 获取GC统计信息
            before_stats = gc.get_stats()

            # 强制垃圾回收
            collected = gc.collect()

            # 再次强制回收
            collected2 = gc.collect()

            # 获取GC统计信息
            after_stats = gc.get_stats()

            results = {
                'collected_objects': collected + collected2,
                'before_stats': before_stats,
                'after_stats': after_stats
            }

            logger.info(f"✅ 垃圾回收完成，回收对象数: {collected + collected2}")
            return results
        except Exception as e:
            logger.error(f"❌ 垃圾回收失败: {e}")
            return {'error': str(e)}


def cleanup_global_memory() -> Dict[str, Any]:
    """清理全局内存的便捷函数"""
    cleaner = GlobalMemoryCleaner()
    return cleaner.cleanup_all()


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 执行清理
    results = cleanup_global_memory()

    # 输出结果
    print("\n=== 全局内存清理结果 ===")
    for category, items in results.items():
        print(f"\n{category.upper()}:")
        if isinstance(items, dict):
            for item, status in items.items():
                print(f"  {item}: {'✅' if status else '❌'}")
        else:
            print(f"  {items}")

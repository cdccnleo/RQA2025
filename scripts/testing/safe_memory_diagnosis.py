#!/usr/bin/env python3
"""
安全的内存诊断脚本 - 逐步检查基础设施层内存使用情况
避免一次性加载所有组件导致的内存泄漏
"""

import gc
import sys
import psutil
import tracemalloc
from pathlib import Path
import logging
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class SafeMemoryDiagnosis:
    """安全的内存诊断器"""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"初始内存使用: {self.baseline_memory:.1f}MB")

    def check_memory_usage(self, stage: str) -> float:
        """检查当前内存使用情况"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.baseline_memory
        logger.info(f"{stage} - 当前内存: {current_memory:.1f}MB, 增长: {memory_increase:.1f}MB")
        return current_memory

    def test_import_safety(self, module_name: str) -> Dict[str, Any]:
        """安全测试模块导入"""
        logger.info(f"测试导入: {module_name}")

        # 开始内存跟踪
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        try:
            # 清理模块缓存
            if module_name in sys.modules:
                del sys.modules[module_name]

            # 强制垃圾回收
            gc.collect()

            # 记录导入前内存
            before_memory = self.check_memory_usage(f"导入前 {module_name}")

            # 尝试导入
            module = __import__(module_name, fromlist=['*'])

            # 记录导入后内存
            after_memory = self.check_memory_usage(f"导入后 {module_name}")

            # 获取内存快照
            snapshot2 = tracemalloc.take_snapshot()

            # 分析内存差异
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')

            # 停止内存跟踪
            tracemalloc.stop()

            memory_increase = after_memory - before_memory

            result = {
                'success': True,
                'memory_increase_mb': memory_increase,
                'top_allocations': top_stats[:5] if top_stats else [],
                'module_loaded': module is not None
            }

            if memory_increase > 100:  # 超过100MB增长
                logger.warning(f"⚠️ {module_name} 内存增长过大: {memory_increase:.1f}MB")
                result['warning'] = f"内存增长过大: {memory_increase:.1f}MB"

            return result

        except Exception as e:
            tracemalloc.stop()
            logger.error(f"❌ 导入 {module_name} 失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'memory_increase_mb': 0
            }

    def test_config_manager_safety(self) -> Dict[str, Any]:
        """安全测试配置管理器"""
        logger.info("=== 测试配置管理器 ===")

        try:
            # 清理可能的现有实例
            if 'src.infrastructure.config.unified_manager' in sys.modules:
                del sys.modules['src.infrastructure.config.unified_manager']

            gc.collect()
            before_memory = self.check_memory_usage("配置管理器导入前")

            # 尝试导入配置管理器
            from src.infrastructure.config.unified_manager import UnifiedConfigManager

            after_memory = self.check_memory_usage("配置管理器导入后")

            # 尝试创建实例
            try:
                config_manager = UnifiedConfigManager()
                instance_memory = self.check_memory_usage("配置管理器实例化后")

                # 检查是否有单例模式
                if hasattr(UnifiedConfigManager, '_instance'):
                    logger.info("✅ 配置管理器使用单例模式")

                # 尝试清理
                if hasattr(config_manager, 'cleanup'):
                    config_manager.cleanup()
                    cleanup_memory = self.check_memory_usage("配置管理器清理后")
                else:
                    cleanup_memory = instance_memory

                return {
                    'success': True,
                    'import_memory_increase': after_memory - before_memory,
                    'instance_memory_increase': instance_memory - after_memory,
                    'cleanup_effectiveness': instance_memory - cleanup_memory,
                    'has_cleanup_method': hasattr(config_manager, 'cleanup'),
                    'is_singleton': hasattr(UnifiedConfigManager, '_instance')
                }

            except Exception as e:
                logger.error(f"❌ 配置管理器实例化失败: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'import_memory_increase': after_memory - before_memory
                }

        except Exception as e:
            logger.error(f"❌ 配置管理器导入失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def test_logging_safety(self) -> Dict[str, Any]:
        """安全测试日志系统"""
        logger.info("=== 测试日志系统 ===")

        try:
            # 清理可能的现有实例
            if 'src.infrastructure.logging.infrastructure_logger' in sys.modules:
                del sys.modules['src.infrastructure.logging.infrastructure_logger']

            gc.collect()
            before_memory = self.check_memory_usage("日志系统导入前")

            # 尝试导入日志系统
            from src.infrastructure.logging.infrastructure_logger import InfrastructureLogger

            after_memory = self.check_memory_usage("日志系统导入后")

            # 尝试创建实例
            try:
                logger_instance = InfrastructureLogger('test_logger')
                instance_memory = self.check_memory_usage("日志系统实例化后")

                # 测试日志记录
                logger_instance.info("测试日志记录")
                log_memory = self.check_memory_usage("日志记录后")

                # 尝试清理
                if hasattr(logger_instance, 'cleanup'):
                    logger_instance.cleanup()
                    cleanup_memory = self.check_memory_usage("日志系统清理后")
                else:
                    cleanup_memory = log_memory

                return {
                    'success': True,
                    'import_memory_increase': after_memory - before_memory,
                    'instance_memory_increase': instance_memory - after_memory,
                    'log_memory_increase': log_memory - instance_memory,
                    'cleanup_effectiveness': log_memory - cleanup_memory,
                    'has_cleanup_method': hasattr(logger_instance, 'cleanup')
                }

            except Exception as e:
                logger.error(f"❌ 日志系统实例化失败: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'import_memory_increase': after_memory - before_memory
                }

        except Exception as e:
            logger.error(f"❌ 日志系统导入失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def test_cache_safety(self) -> Dict[str, Any]:
        """安全测试缓存系统"""
        logger.info("=== 测试缓存系统 ===")

        try:
            # 清理可能的现有实例
            if 'src.infrastructure.cache.enhanced_cache_manager' in sys.modules:
                del sys.modules['src.infrastructure.cache.enhanced_cache_manager']

            gc.collect()
            before_memory = self.check_memory_usage("缓存系统导入前")

            # 尝试导入缓存系统
            from src.infrastructure.cache.enhanced_cache_manager import EnhancedCacheManager

            after_memory = self.check_memory_usage("缓存系统导入后")

            # 尝试创建实例
            try:
                cache_manager = EnhancedCacheManager()
                instance_memory = self.check_memory_usage("缓存系统实例化后")

                # 测试缓存操作
                cache_manager.set('test_key', 'test_value')
                cache_memory = self.check_memory_usage("缓存操作后")

                # 尝试清理
                if hasattr(cache_manager, 'cleanup'):
                    cache_manager.cleanup()
                    cleanup_memory = self.check_memory_usage("缓存系统清理后")
                else:
                    cleanup_memory = cache_memory

                return {
                    'success': True,
                    'import_memory_increase': after_memory - before_memory,
                    'instance_memory_increase': instance_memory - after_memory,
                    'cache_memory_increase': cache_memory - instance_memory,
                    'cleanup_effectiveness': cache_memory - cleanup_memory,
                    'has_cleanup_method': hasattr(cache_manager, 'cleanup')
                }

            except Exception as e:
                logger.error(f"❌ 缓存系统实例化失败: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'import_memory_increase': after_memory - before_memory
                }

        except Exception as e:
            logger.error(f"❌ 缓存系统导入失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """运行综合诊断"""
        logger.info("开始安全内存诊断...")

        results = {
            'config_manager': self.test_config_manager_safety(),
            'logging_system': self.test_logging_safety(),
            'cache_system': self.test_cache_safety(),
            'final_memory': self.check_memory_usage("诊断完成")
        }

        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收完成，回收对象数: {collected}")

        final_memory = self.check_memory_usage("垃圾回收后")
        results['gc_collected'] = collected
        results['final_memory_after_gc'] = final_memory

        return results


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建诊断器
    diagnosis = SafeMemoryDiagnosis()

    try:
        # 运行诊断
        results = diagnosis.run_comprehensive_diagnosis()

        # 输出结果
        print("\n=== 安全内存诊断结果 ===")
        for component, result in results.items():
            if component in ['final_memory', 'gc_collected', 'final_memory_after_gc']:
                continue

            print(f"\n{component.upper()}:")
            if result.get('success'):
                print(f"  ✅ 成功")
                print(f"  导入内存增长: {result.get('import_memory_increase', 0):.1f}MB")
                print(f"  实例化内存增长: {result.get('instance_memory_increase', 0):.1f}MB")
                if 'cache_memory_increase' in result:
                    print(f"  缓存操作内存增长: {result.get('cache_memory_increase', 0):.1f}MB")
                if 'log_memory_increase' in result:
                    print(f"  日志操作内存增长: {result.get('log_memory_increase', 0):.1f}MB")
                print(f"  清理效果: {result.get('cleanup_effectiveness', 0):.1f}MB")
                print(f"  有清理方法: {'是' if result.get('has_cleanup_method') else '否'}")
                if result.get('warning'):
                    print(f"  ⚠️ {result['warning']}")
            else:
                print(f"  ❌ 失败: {result.get('error', '未知错误')}")

        print(f"\n最终内存使用: {results['final_memory']:.1f}MB")
        print(f"垃圾回收对象数: {results['gc_collected']}")
        print(f"垃圾回收后内存: {results['final_memory_after_gc']:.1f}MB")

    except Exception as e:
        logger.error(f"诊断过程中发生错误: {e}")
        print(f"❌ 诊断失败: {e}")


if __name__ == "__main__":
    main()

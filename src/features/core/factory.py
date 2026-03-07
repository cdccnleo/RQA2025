#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征模块统一工厂

整合所有特征处理器，解决代码重复问题，提供统一的创建接口。
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from .feature_config import FeatureConfig, FeatureType
from ..processors.base_processor import BaseFeatureProcessor
from ..processors.feature_processor import FeatureProcessor
from ..processors.distributed_processor import DistributedFeatureProcessor
from .parallel_feature_processor import ParallelFeatureProcessor, ParallelConfig
from .optimized_feature_manager import OptimizedFeatureManager
# 使用绝对导入避免命名冲突
from .feature_manager import FeatureManager as MainFeatureManager

logger = logging.getLogger(__name__)


class ProcessorType(Enum):

    """特征处理器类型枚举"""
    BASE = "base"
    FEATURE = "feature"
    DISTRIBUTED = "distributed"
    PARALLEL = "parallel"
    OPTIMIZED = "optimized"
    MANAGER = "manager"


class FeatureProcessorFactory:

    """特征处理器工厂"""

    def __init__(self):
        """初始化工厂"""
        self._processors = {
            'base': {
                'type': 'base',
                'class_name': 'BaseFeatureProcessor',
                'module': 'src.features.processors.base_processor',
                'class': BaseFeatureProcessor
            },
            'feature': {
                'type': 'feature',
                'class_name': 'FeatureProcessor',
                'module': 'src.features.processors.feature_processor',
                'class': FeatureProcessor
            },
            'distributed': {
                'type': 'distributed',
                'class_name': 'DistributedFeatureProcessor',
                'module': 'src.features.processors.distributed_processor',
                'class': DistributedFeatureProcessor
            },
            'parallel': {
                'type': 'parallel',
                'class_name': 'ParallelFeatureProcessor',
                'module': 'src.features.parallel_feature_processor',
                'class': ParallelFeatureProcessor
            },
            'optimized': {
                'type': 'optimized',
                'class_name': 'OptimizedFeatureManager',
                'module': 'src.features.optimized_feature_manager',
                'class': OptimizedFeatureManager
            },
            'manager': {
                'type': 'manager',
                'class_name': 'FeatureManager',
                'module': 'src.features.feature_manager',
                'class': MainFeatureManager
            }
        }

    def list_available_processors(self) -> List[str]:
        """列出可用的处理器类型"""
        return list(self._processors.keys())

    def get_processor_info(self, processor_type: str) -> Dict[str, str]:
        """获取处理器信息"""
        if processor_type in self._processors:
            info = self._processors[processor_type].copy()
            # 移除class对象，只返回字符串信息
            info.pop('class', None)
            return info
        return {}

    def register_processor(self, processor_type: str, processor_class: Any) -> None:
        """注册新的处理器类型"""
        try:
            # 安全地获取类名，处理Mock对象
            if hasattr(processor_class, '__name__'):

                class_name = processor_class.__name__
            elif hasattr(processor_class, '_extract_mock_name'):

                class_name = processor_class._extract_mock_name()
            else:

                class_name = str(processor_class)

            # 获取模块名，处理Mock对象
            if hasattr(processor_class, '__module__'):
                module = processor_class.__module__
            else:
                module = 'unknown'

            self._processors[processor_type] = {
                'type': processor_type,
                'class_name': class_name,
                'module': module,
                'class': processor_class
            }
            logger.info(f"注册处理器类型: {processor_type} -> {class_name}")
        except Exception as e:
            logger.error(f"注册处理器失败: {e}")
            raise

    def create_processor(self, processor_type: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:

        if processor_type not in self._processors:
            raise ValueError(f"不支持的处理器类型: {processor_type}")

        try:
            processor_info = self._processors[processor_type]
            processor_class = processor_info['class']

            # 根据处理器类型创建适当的配置对象
            if processor_type == 'parallel':
                # 并行处理器需要ParallelConfig
                if config is None:
                    config = {}
                parallel_config = ParallelConfig(
                    n_jobs=config.get('n_jobs', 4),
                    chunk_size=config.get('chunk_size', 500),
                    **kwargs
                )
                return processor_class(parallel_config)

            elif processor_type == 'manager':
                # 特征管理器需要FeatureConfig
                if config is None:
                    config = {}
                # 过滤掉FeatureManager不支持的参数
                manager_config = {k: v for k, v in config.items()
                                  if k not in ['enable_cache', 'max_workers']}
                feature_config = FeatureConfig(
                    feature_types=manager_config.get('feature_types', [FeatureType.TECHNICAL]),
                    technical_indicators=manager_config.get('technical_indicators', ["sma", "rsi"]),
                    enable_feature_selection=manager_config.get('enable_feature_selection', False),
                    enable_standardization=manager_config.get('enable_standardization', True),
                    **{k: v for k, v in manager_config.items()
                       if k not in ['feature_types', 'technical_indicators', 'enable_feature_selection', 'enable_standardization']}
                )
                return processor_class(feature_config)

            elif processor_type == 'feature':
                # 特征处理器需要ProcessorConfig
                from ..processors.base_processor import ProcessorConfig
                if config is None:
                    config = {}
                processor_config = ProcessorConfig(
                    processor_type=processor_type,
                    **config
                )
                return processor_class(processor_config)

            elif processor_type == 'optimized':
                return processor_class(config)

            else:
                # 其他处理器直接使用配置字典
                return processor_class(config or {}, **kwargs)

        except Exception as e:
            logger.error(f"创建处理器 {processor_type} 失败: {e}")
            raise


class UnifiedFeatureManager:

    """统一特征管理器"""

    def __init__(self, config: Dict[str, Any]):
        """初始化管理器"""
        self.config = config
        self.factory = FeatureProcessorFactory()
        self._active_processors = {}

        # 初始化默认处理器
        self._init_default_processors()

    def _init_default_processors(self):
        """初始化默认处理器"""
        try:
            # 初始化管理器处理器
            if 'manager' in self.config:
                self._active_processors['manager'] = self.factory.create_processor(
                    'manager', self.config['manager'])

            # 初始化并行处理器
            if 'parallel' in self.config:
                self._active_processors['parallel'] = self.factory.create_processor(
                    'parallel', self.config['parallel'])

            if 'optimized' in self.config:
                optimized_proc = self.factory.create_processor('optimized', self.config['optimized'])
                if optimized_proc:
                    self._active_processors['optimized'] = optimized_proc

        except Exception as e:
            logger.error(f"初始化默认处理器失败: {e}")

    def get_processor(self, processor_type: str):
        """获取处理器"""
        return self._active_processors.get(processor_type)

    def create_processor(self, processor_type: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        """创建新处理器"""
        try:
            processor = self.factory.create_processor(processor_type, config, **kwargs)
            self._active_processors[processor_type] = processor
            return processor
        except Exception as e:
            logger.error(f"创建处理器 {processor_type} 失败: {e}")
            return None

    def process_features(self, data: Any, feature_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理特征，使用可用的处理器"""
        results = {}

        for config in feature_configs:
            processor_type = config.get('processor_type', 'default')
            processor = self.get_processor(processor_type)

            if processor is None:
                logger.warning(f"未找到处理器: {processor_type}")
                continue

            try:
                # 检查处理器是否支持process_features方法
                if hasattr(processor, 'process_features'):
                    result = processor.process_features(data, config)
                    results[processor_type] = result
                elif hasattr(processor, 'process_features_parallel'):
                    # 检查方法是否可调用且不是None
                    method = getattr(processor, 'process_features_parallel')
                    if callable(method):
                        result = method(data, config)
                        results[processor_type] = result
                    else:
                        logger.warning(f"处理器 {processor_type} 的 process_features_parallel 方法不可调用")
                        results[processor_type] = {'error': '方法不可调用'}
                else:
                    logger.warning(f"处理器 {processor_type} 不支持特征处理方法")
                    results[processor_type] = {'error': '不支持的特征处理方法'}

            except Exception as e:
                logger.error(f"处理器 {processor_type} 处理特征失败: {e}")
                results[processor_type] = {'error': str(e)}

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {}
        for name, processor in self._active_processors.items():
            if hasattr(processor, 'get_stats'):
                stats[name] = processor.get_stats()
            else:
                stats[name] = {'status': 'active', 'type': type(processor).__name__}
        return stats

    def close(self):
        """关闭管理器"""
        for processor in self._active_processors.values():
            if hasattr(processor, 'close'):
                try:
                    processor.close()
                except Exception as e:
                    logger.warning(f"关闭处理器失败: {e}")
        self._active_processors.clear()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局工厂实例
feature_processor_factory = FeatureProcessorFactory()

# 便捷函数


def create_feature_processor(processor_type: str, config: Optional[Dict[str, Any]] = None, **kwargs):
    """便捷创建特征处理器"""
    return feature_processor_factory.create_processor(processor_type, config, **kwargs)


def get_unified_feature_manager(config: Dict[str, Any]) -> UnifiedFeatureManager:
    """获取统一特征管理器"""
    return UnifiedFeatureManager(config)

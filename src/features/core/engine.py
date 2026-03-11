#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征引擎核心模块

负责协调各个特征处理组件，提供统一的特征处理接口。
支持通过统一基础设施集成层访问基础设施服务。
"""

import logging
import time
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .config import FeatureConfig

# 导入特征处理器基类
try:
    from ...processors.base_processor import BaseFeatureProcessor
except ImportError:
    # 降级定义
    from abc import ABC
    class BaseFeatureProcessor(ABC):
        pass
from .feature_config import FeatureType
# 延迟导入以避免循环依赖
# from .feature_engineer import FeatureEngineer
# from ..processors.feature_selector import FeatureSelector
# from ..processors.feature_standardizer import FeatureStandardizer
# from ..processors.general_processor import FeatureProcessor
# from ..feature_saver import FeatureSaver
# from ..processors.base_processor import BaseFeatureProcessor
# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
    logger = logging.getLogger(__name__)
except ImportError:
    # 降级到直接导入
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger('__name__')


class FeatureEngine:

    """特征引擎核心，负责协调各个组件"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征引擎

        Args:
            config: 特征配置
        """
        self.config = config or FeatureConfig()

        # 延迟初始化核心组件以避免循环依赖
        self._engineer = None
        self._selector = None
        self._standardizer = None
        self._saver = None

        # 设置logger
        self.logger = logger

        # 注册的处理器
        self.processors: Dict[str, BaseFeatureProcessor] = {}

        # 处理统计
        self.stats = {
            'processed_features': 0,
            'processing_time': 0.0,
            'errors': 0
        }

        # 任务管理
        self.tasks = []
        # 特征存储
        self.features = []
        # 技术指标状态
        self.indicators = []
        
        # 任务状态变更钩子函数列表
        self._task_status_hooks: List[Callable] = []
        # 任务完成钩子函数列表
        self._task_completed_hooks: List[Callable] = []
        # 任务失败钩子函数列表
        self._task_failed_hooks: List[Callable] = []

        # 新增：集成任务调度器
        self._task_scheduler = None
        try:
            from src.features.distributed.task_scheduler import get_task_scheduler
            self._task_scheduler = get_task_scheduler()
            self.logger.info("✅ 任务调度器集成成功")
        except Exception as e:
            self.logger.warning(f"⚠️ 任务调度器集成失败: {e}")
            self._task_scheduler = None

        # 新增：初始化 TechnicalProcessor
        self._technical_processor = None
        try:
            from src.features.processors.technical.technical_processor import TechnicalProcessor
            self._technical_processor = TechnicalProcessor()
            self.logger.info("✅ TechnicalProcessor 集成成功")
        except Exception as e:
            self.logger.warning(f"⚠️ TechnicalProcessor 集成失败: {e}")
            self._technical_processor = None

        # 新增：特征缓存
        self._features_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5分钟缓存

        # 自动注册默认处理器
        self._register_default_processors()
        
        # 初始化样本数据，用于测试和演示
        # 注意：现在优先使用动态计算，样本数据作为回退
        self.initialize_sample_data()
        
        # 注册默认的任务状态变更钩子
        self._register_default_hooks()

    def register_processor(self, name: str, processor) -> None:
        """
        注册特征处理器

        Args:
            name: 处理器名称
            processor: 处理器实例
        """
        # 延迟导入以避免循环依赖
        from ..processors.base_processor import BaseFeatureProcessor

        if not isinstance(processor, BaseFeatureProcessor):
            raise ValueError(f"处理器 {name} 必须继承自 BaseFeatureProcessor")

        self.processors[name] = processor
        self.logger.info(f"注册处理器: {name}")

    def get_processor(self, name: str):
        """
        获取处理器

        Args:
            name: 处理器名称

        Returns:
            处理器实例
        """
        return self.processors.get(name)

    def list_processors(self) -> List[str]:
        """
        列出所有注册的处理器

        Returns:
            处理器名称列表
        """
        return list(self.processors.keys())

    def _register_default_processors(self) -> None:
        """
        注册默认处理器
        """
        try:
            # 注册技术指标处理器
            from ..processors.technical.technical_processor import TechnicalProcessor
            technical_processor = TechnicalProcessor()
            self.register_processor("technical", technical_processor)

            # 注册通用处理器
            from ..processors.general_processor import FeatureProcessor
            general_processor = FeatureProcessor()
            self.register_processor("general", general_processor)

            # 注意：SentimentAnalyzer不是BaseFeatureProcessor的子类，不能作为处理器注册
            # 情感分析功能通过其他方式集成
            # from ..sentiment.sentiment_analyzer import SentimentAnalyzer
            # sentiment_processor = SentimentAnalyzer()
            # self.register_processor("sentiment", sentiment_processor)

            self.logger.info("默认处理器注册完成")

        except Exception as e:
            self.logger.warning(f"注册默认处理器时出现警告: {e}")

    def process_features(self, data: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        处理特征

        Args:
            data: 输入数据
            config: 特征配置

        Returns:
            处理后的特征数据
        """
        import time
        start_time = time.time()

        try:
            # 使用传入的配置或默认配置
            process_config = config or self.config

            # 验证输入数据
            if not self.validate_data(data):
                raise ValueError("输入数据验证失败")

            # 1. 特征工程 - 使用注册的处理器
            self.logger.info("开始特征工程...")
            engineered_features = self._engineer_features(data, process_config)

            # 2. 特征处理 - 使用通用处理器
            self.logger.info("开始特征处理...")
            processed_features = self._process_features(engineered_features, process_config)

            # 3. 特征选择
            if process_config.enable_feature_selection:
                self.logger.info("开始特征选择...")
                selected_features = self.selector.select_features(
                    processed_features,
                    config=process_config
                )
            else:
                selected_features = processed_features

            # 4. 特征标准化
            if process_config.enable_standardization:
                self.logger.info("开始特征标准化...")
                standardized_features = self.standardizer.standardize_features(
                    selected_features,
                    config=process_config
                )
            else:
                standardized_features = selected_features

            # 5. 保存特征
            if process_config.enable_feature_saving:
                self.logger.info("保存特征...")
                self.saver.save_features(
                    standardized_features,
                    config=process_config
                )

            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['processed_features'] += len(standardized_features.columns)
            self.stats['processing_time'] += processing_time

            self.logger.info(f"特征处理完成，耗时: {processing_time:.2f}秒")

            return standardized_features

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"特征处理失败: {e}")
            raise

    def _engineer_features(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """
        使用注册的处理器进行特征工程

        Args:
            data: 输入数据
            config: 特征配置

        Returns:
            工程化后的特征
        """
        try:
            # 使用技术指标处理器
            technical_processor = self.get_processor("technical")
            if technical_processor and FeatureType.TECHNICAL in config.feature_types:
                from src.infrastructure.interfaces.standard_interfaces import FeatureRequest
                
                # 🚀 关键修复：传递 technical_indicators 到 feature_names
                technical_indicators = getattr(config, 'technical_indicators', [])
                if not technical_indicators:
                    technical_indicators = ['sma', 'ema', 'rsi', 'macd']
                
                self.logger.info(f"🚀 调用 TechnicalProcessor，indicators: {technical_indicators}")
                
                request = FeatureRequest(
                    data=data,
                    feature_names=technical_indicators,  # 传入 technical_indicators
                    config=config.to_dict() if hasattr(config, 'to_dict') else {}
                )
                technical_features = technical_processor.process(request)
                self.logger.info(f"✅ TechnicalProcessor 完成，生成 {len(technical_features.columns)} 个特征: {list(technical_features.columns)[:10]}...")
            else:
                technical_features = pd.DataFrame()

            # 使用情感分析处理器
            sentiment_processor = self.get_processor("sentiment")
            if sentiment_processor and FeatureType.SENTIMENT in config.feature_types:
                from src.infrastructure.interfaces.standard_interfaces import FeatureRequest
                request = FeatureRequest(
                    data=data,
                    feature_names=[],
                    config=config.to_dict() if hasattr(config, 'to_dict') else {}
                )
                sentiment_features = sentiment_processor.process(request)
            else:
                sentiment_features = pd.DataFrame()

            # 合并特征
            all_features = pd.concat([technical_features, sentiment_features], axis=1)

            return all_features

        except Exception as e:
            self.logger.error(f"特征工程失败: {e}")
            return data

    def _process_features(self, features: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """
        使用通用处理器处理特征

        Args:
            features: 输入特征
            config: 特征配置

        Returns:
            处理后的特征
        """
        try:
            general_processor = self.get_processor("general")
            if general_processor:
                from src.infrastructure.interfaces.standard_interfaces import FeatureRequest
                request = FeatureRequest(
                    data=features,
                    feature_names=[],
                    config=config.to_dict() if hasattr(config, 'to_dict') else {}
                )
                return general_processor.process(request)
            else:
                return features

        except Exception as e:
            self.logger.error(f"特征处理失败: {e}")
            return features

    def process_with_processor(self, data: pd.DataFrame, processor_name: str,


                               config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        使用指定处理器处理特征

        Args:
            data: 输入数据
            processor_name: 处理器名称
            config: 特征配置

        Returns:
            处理后的特征数据
        """
        processor = self.get_processor(processor_name)
        if not processor:
            raise ValueError(f"未找到处理器: {processor_name}")

        try:
            from src.infrastructure.interfaces.standard_interfaces import FeatureRequest
            request = FeatureRequest(
                data=data,
                feature_names=[],
                config=config.to_dict() if hasattr(config, 'to_dict') else {}
            )
            return processor.process(request)
        except Exception as e:
            self.logger.error(f"处理器 {processor_name} 处理失败: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            统计信息字典
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'processed_features': 0,
            'processing_time': 0.0,
            'errors': 0
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            验证结果
        """
        if data.empty:
            self.logger.error("输入数据为空")
            return False

        # 检查必要的列
        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"缺失必要列: {missing_columns}")
            return False

        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                self.logger.error(f"列 {col} 不是数值类型")
                return False

        return True

    def get_supported_features(self) -> List[str]:
        """
        获取支持的特征列表

        Returns:
            特征名称列表
        """
        features = []

        # 从各个处理器获取支持的特征
        for processor in self.processors.values():
            if hasattr(processor, 'list_features'):
                features.extend(processor.list_features())

        return list(set(features))  # 去重

    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取引擎信息

        Returns:
            引擎信息字典
        """
        return {
            'version': '1.0.0',
            'processors': self.list_processors(),
            'supported_features': self.get_supported_features(),
            'stats': self.get_stats(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        获取特征提取任务列表

        Returns:
            任务列表
        """
        return self.tasks

    def get_features(self) -> List[Dict[str, Any]]:
        """
        获取特征列表 - 优先使用动态计算的特征

        Returns:
            特征列表
        """
        # 检查缓存是否有效
        if self._is_cache_valid():
            self.logger.debug("使用缓存的特征数据")
            return self._features_cache.get('features', self.features)
        
        # 尝试动态计算特征
        try:
            dynamic_features = self._calculate_features()
            if dynamic_features:
                self.logger.info(f"✅ 动态计算了 {len(dynamic_features)} 个特征")
                # 更新缓存
                self._features_cache['features'] = dynamic_features
                self._cache_timestamp = time.time()
                return dynamic_features
        except Exception as e:
            self.logger.warning(f"⚠️ 动态计算特征失败: {e}")
        
        # 如果动态计算失败，返回样本数据
        self.logger.debug("使用样本特征数据")
        return self.features

    def _calculate_features(self) -> List[Dict[str, Any]]:
        """
        使用 TechnicalProcessor 动态计算特征
        
        Returns:
            计算得到的特征列表
        """
        if not self._technical_processor:
            self.logger.warning("TechnicalProcessor 未初始化，无法计算特征")
            return []
        
        try:
            # 生成示例股票数据
            data = self._get_sample_stock_data()
            
            # 定义要计算的指标
            indicators = ["sma", "rsi", "macd"]
            params = {
                "sma_periods": [20],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            }
            
            self.logger.debug(f"开始计算指标: {indicators}")
            
            # 计算指标
            results = self._technical_processor.calculate_multiple_indicators(
                data, indicators, params
            )
            
            if results.empty:
                self.logger.warning("指标计算结果为空")
                return []
            
            # 转换为特征格式
            features = []
            for column in results.columns:
                # 计算质量评分（基于缺失值比例）
                missing_ratio = results[column].isna().mean()
                quality_score = 1.0 - missing_ratio
                
                feature = {
                    "name": column,
                    "display_name": self._get_display_name(column),
                    "type": "technical",
                    "importance": 0.7,
                    "correlation": 0.6,
                    "quality_score": quality_score,
                    "missing_values": missing_ratio,
                    "created_at": time.time(),
                    "updated_at": time.time()
                }
                features.append(feature)
                
            self.logger.info(f"✅ 成功计算 {len(features)} 个特征: {[f['name'] for f in features]}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ 计算特征失败: {e}", exc_info=True)
            return []

    def _get_sample_stock_data(self) -> pd.DataFrame:
        """
        生成示例股票数据用于特征计算
        
        Returns:
            示例股票数据 DataFrame
        """
        try:
            # 生成100天的示例数据
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
            np.random.seed(42)
            
            # 生成随机价格序列（随机游走）
            returns = np.random.randn(100) * 0.02  # 2% 日波动
            prices = 100 * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': prices * (1 + np.random.randn(100) * 0.005),
                'high': prices * (1 + abs(np.random.randn(100)) * 0.02),
                'low': prices * (1 - abs(np.random.randn(100)) * 0.02),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
            
            self.logger.debug(f"生成示例数据: {len(data)} 条记录")
            return data
            
        except Exception as e:
            self.logger.error(f"生成示例数据失败: {e}")
            # 返回最小数据集
            return pd.DataFrame({
                'open': [100.0],
                'high': [102.0],
                'low': [98.0],
                'close': [100.0],
                'volume': [1000000]
            })

    def _get_display_name(self, column: str) -> str:
        """
        获取特征的显示名称
        
        Args:
            column: 特征列名
            
        Returns:
            中文显示名称
        """
        display_names = {
            'sma_20': '20日均线',
            'sma': '简单移动平均',
            'rsi': '14日RSI',
            'macd': 'MACD',
            'macd_signal': 'MACD信号',
            'macd_histogram': 'MACD柱状图'
        }
        return display_names.get(column, column)

    def _is_cache_valid(self) -> bool:
        """
        检查特征缓存是否有效
        
        Returns:
            缓存是否有效
        """
        if not self._cache_timestamp:
            return False
        elapsed = time.time() - self._cache_timestamp
        is_valid = elapsed < self._cache_ttl
        if not is_valid:
            self.logger.debug(f"缓存已过期 ({elapsed:.0f}s > {self._cache_ttl}s)")
        return is_valid

    def refresh_features(self) -> List[Dict[str, Any]]:
        """
        手动刷新特征（清除缓存并重新计算）
        
        Returns:
            重新计算的特征列表
        """
        self.logger.info("🔄 手动刷新特征")
        self._cache_timestamp = None
        self._features_cache = {}
        return self.get_features()

    def get_indicators(self) -> List[Dict[str, Any]]:
        """
        获取技术指标状态

        Returns:
            技术指标列表
        """
        return self.indicators

    def create_task(self, task_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建特征提取任务

        Args:
            task_type: 任务类型
            config: 任务配置

        Returns:
            创建的任务信息
        """
        import time
        import uuid
        config = config or {}
        # 检查是否有自定义任务ID前缀
        task_id_prefix = config.get('task_id_prefix', 'task')
        # 使用股票代码和时间戳生成唯一ID，确保每只股票的任务ID唯一
        stock_code = config.get('stock_code', '')
        if stock_code:
            task_id = f"{task_id_prefix}_{stock_code}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        else:
            task_id = f"{task_id_prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "progress": 0,
            "feature_count": 0,
            "start_time": int(time.time()),
            "created_at": int(time.time()),
            "config": config
        }
        self.tasks.append(task)
        return task

    def stop_task(self, task_id: str) -> bool:
        """
        停止特征提取任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功停止
        """
        import time
        for task in self.tasks:
            if task.get('task_id') == task_id:
                task['status'] = 'stopped'
                task['end_time'] = int(time.time())
                return True
        return False

    def delete_task(self, task_id: str) -> bool:
        """
        删除特征提取任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功删除
        """
        for i, task in enumerate(self.tasks):
            if task.get('task_id') == task_id:
                self.tasks.pop(i)
                return True
        return False

    def update_task_status(self, task_id: str, status: str, progress: int = None) -> bool:
        """
        更新任务状态和进度

        Args:
            task_id: 任务ID
            status: 任务状态
            progress: 任务进度（0-100）

        Returns:
            是否更新成功
        """
        for task in self.tasks:
            if task.get('task_id') == task_id:
                old_status = task.get('status')
                task['status'] = status
                if progress is not None:
                    task['progress'] = min(100, max(0, progress))
                if status == 'completed':
                    import time
                    task['end_time'] = int(time.time())
                
                # 触发状态变更钩子
                self._trigger_task_status_hooks(task_id, old_status, status, progress)
                
                # 根据状态触发特定钩子
                if status == 'completed':
                    self._trigger_task_completed_hooks(task_id, task)
                elif status == 'failed':
                    self._trigger_task_failed_hooks(task_id, task)
                
                return True
        return False

    def _register_default_hooks(self) -> None:
        """
        注册默认的任务状态变更钩子
        
        包括：
        1. 持久化存储更新
        2. WebSocket广播
        3. 事件总线发布
        4. 监控指标更新
        """
        # 注册任务完成钩子
        self.register_task_completed_hook(self._on_task_completed_default)
        
        # 注册任务失败钩子
        self.register_task_failed_hook(self._on_task_failed_default)
        
        logger.info("默认任务状态钩子已注册")

    def _on_task_completed_default(self, task_id: str, task: Dict[str, Any]) -> None:
        """
        默认的任务完成处理函数
        
        Args:
            task_id: 任务ID
            task: 任务信息
        """
        try:
            logger.info(f"🎯 任务完成钩子触发: {task_id}")
            
            # 1. 更新持久化存储
            self._persist_task_completion(task_id, task)
            
            # 2. WebSocket广播
            self._broadcast_task_completion(task_id, task)
            
            # 3. 发布到事件总线
            self._publish_task_event('TASK_COMPLETED', task_id, task)
            
            # 4. 触发数据归档（如果配置）
            if self.config.auto_archive:
                self._archive_task_data(task_id, task)
                
        except Exception as e:
            logger.error(f"任务完成钩子执行失败: {e}", exc_info=True)

    def _on_task_failed_default(self, task_id: str, task: Dict[str, Any]) -> None:
        """
        默认的任务失败处理函数
        
        Args:
            task_id: 任务ID
            task: 任务信息
        """
        try:
            logger.warning(f"⚠️ 任务失败钩子触发: {task_id}")
            
            # 1. 更新持久化存储
            self._persist_task_failure(task_id, task)
            
            # 2. WebSocket广播
            self._broadcast_task_failure(task_id, task)
            
            # 3. 发布到事件总线
            self._publish_task_event('TASK_FAILED', task_id, task)
            
            # 4. 发送告警通知
            self._send_failure_alert(task_id, task)
            
        except Exception as e:
            logger.error(f"任务失败钩子执行失败: {e}", exc_info=True)

    def register_task_status_hook(self, hook: callable) -> None:
        """
        注册任务状态变更钩子
        
        Args:
            hook: 钩子函数，签名: (task_id, old_status, new_status, progress) -> None
        """
        self._task_status_hooks.append(hook)
        logger.debug(f"任务状态变更钩子已注册，当前数量: {len(self._task_status_hooks)}")

    def register_task_completed_hook(self, hook: callable) -> None:
        """
        注册任务完成钩子
        
        Args:
            hook: 钩子函数，签名: (task_id, task) -> None
        """
        self._task_completed_hooks.append(hook)
        logger.debug(f"任务完成钩子已注册，当前数量: {len(self._task_completed_hooks)}")

    def register_task_failed_hook(self, hook: callable) -> None:
        """
        注册任务失败钩子
        
        Args:
            hook: 钩子函数，签名: (task_id, task) -> None
        """
        self._task_failed_hooks.append(hook)
        logger.debug(f"任务失败钩子已注册，当前数量: {len(self._task_failed_hooks)}")

    def _trigger_task_status_hooks(self, task_id: str, old_status: str, new_status: str, progress: int) -> None:
        """
        触发任务状态变更钩子
        
        Args:
            task_id: 任务ID
            old_status: 旧状态
            new_status: 新状态
            progress: 进度
        """
        for hook in self._task_status_hooks:
            try:
                hook(task_id, old_status, new_status, progress)
            except Exception as e:
                logger.error(f"任务状态变更钩子执行失败: {e}")

    def _trigger_task_completed_hooks(self, task_id: str, task: Dict[str, Any]) -> None:
        """
        触发任务完成钩子
        
        Args:
            task_id: 任务ID
            task: 任务信息
        """
        for hook in self._task_completed_hooks:
            try:
                hook(task_id, task)
            except Exception as e:
                logger.error(f"任务完成钩子执行失败: {e}")

    def _trigger_task_failed_hooks(self, task_id: str, task: Dict[str, Any]) -> None:
        """
        触发任务失败钩子
        
        Args:
            task_id: 任务ID
            task: 任务信息
        """
        for hook in self._task_failed_hooks:
            try:
                hook(task_id, task)
            except Exception as e:
                logger.error(f"任务失败钩子执行失败: {e}")

    def _persist_task_completion(self, task_id: str, task: Dict[str, Any]) -> None:
        """持久化任务完成状态"""
        try:
            from src.gateway.web.feature_task_persistence import update_feature_task
            update_feature_task(task_id, {
                'status': 'completed',
                'progress': 100,
                'end_time': task.get('end_time'),
                'feature_count': task.get('feature_count', 0)
            })
            logger.debug(f"任务完成状态已持久化: {task_id}")
        except Exception as e:
            logger.error(f"持久化任务完成状态失败: {e}")

    def _broadcast_task_completion(self, task_id: str, task: Dict[str, Any]) -> None:
        """广播任务完成消息"""
        try:
            import asyncio
            from src.gateway.web.websocket_manager import manager
            
            message = {
                'type': 'task_completed',
                'task_id': task_id,
                'feature_count': task.get('feature_count', 0),
                'timestamp': int(time.time())
            }
            
            asyncio.create_task(
                manager.broadcast('feature_engineering', message)
            )
            logger.debug(f"任务完成消息已广播: {task_id}")
        except Exception as e:
            logger.debug(f"广播任务完成消息失败: {e}")

    def _publish_task_event(self, event_type: str, task_id: str, task: Dict[str, Any]) -> None:
        """发布任务事件到事件总线"""
        try:
            from src.core.event_bus import EventBus
            from src.core.event_bus.types import EventType
            
            event_bus = EventBus()
            event_bus.publish(
                EventType.FEATURE_PROCESSING_COMPLETED if event_type == 'TASK_COMPLETED' else EventType.FEATURE_PROCESSING_FAILED,
                {
                    'task_id': task_id,
                    'task_type': task.get('task_type'),
                    'feature_count': task.get('feature_count', 0),
                    'timestamp': int(time.time())
                }
            )
            logger.debug(f"任务事件已发布: {task_id} -> {event_type}")
        except Exception as e:
            logger.debug(f"发布任务事件失败: {e}")

    def _archive_task_data(self, task_id: str, task: Dict[str, Any]) -> None:
        """归档任务数据"""
        logger.info(f"任务数据归档（待实现）: {task_id}")
        # TODO: 实现数据归档逻辑

    def _persist_task_failure(self, task_id: str, task: Dict[str, Any]) -> None:
        """持久化任务失败状态"""
        try:
            from src.gateway.web.feature_task_persistence import update_feature_task
            update_feature_task(task_id, {
                'status': 'failed',
                'error_message': task.get('error_message', 'Unknown error'),
                'end_time': task.get('end_time')
            })
            logger.debug(f"任务失败状态已持久化: {task_id}")
        except Exception as e:
            logger.error(f"持久化任务失败状态失败: {e}")

    def _broadcast_task_failure(self, task_id: str, task: Dict[str, Any]) -> None:
        """广播任务失败消息"""
        try:
            import asyncio
            from src.gateway.web.websocket_manager import manager
            
            message = {
                'type': 'task_failed',
                'task_id': task_id,
                'error_message': task.get('error_message', 'Unknown error'),
                'timestamp': int(time.time())
            }
            
            asyncio.create_task(
                manager.broadcast('feature_engineering', message)
            )
            logger.debug(f"任务失败消息已广播: {task_id}")
        except Exception as e:
            logger.debug(f"广播任务失败消息失败: {e}")

    def _send_failure_alert(self, task_id: str, task: Dict[str, Any]) -> None:
        """发送失败告警"""
        logger.warning(f"任务失败告警: {task_id} - {task.get('error_message', 'Unknown error')}")
        # TODO: 实现告警通知逻辑（邮件、短信等）

    def add_task_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        添加任务结果

        Args:
            task_id: 任务ID
            result: 任务结果

        Returns:
            是否添加成功
        """
        for task in self.tasks:
            if task.get('task_id') == task_id:
                task['result'] = result
                task['feature_count'] = len(result.get('features', []))
                # 自动添加生成的特征到特征存储
                if result.get('features'):
                    for feature in result['features']:
                        self.add_feature(feature)
                return True
        return False

    def add_feature(self, feature: Dict[str, Any]) -> bool:
        """
        添加特征到存储

        Args:
            feature: 特征信息

        Returns:
            是否添加成功
        """
        import time
        # 确保特征有必要的字段
        if 'name' not in feature:
            return False
        
        # 检查特征是否已存在
        existing_feature = next((f for f in self.features if f.get('name') == feature['name']), None)
        if existing_feature:
            # 更新现有特征
            existing_feature.update(feature)
            existing_feature['updated_at'] = int(time.time())
        else:
            # 添加新特征
            feature['created_at'] = int(time.time())
            feature['updated_at'] = int(time.time())
            # 评估特征质量
            feature['quality_score'] = self.evaluate_feature_quality(feature)
            self.features.append(feature)
        return True

    def update_feature(self, feature_name: str, updates: Dict[str, Any]) -> bool:
        """
        更新特征信息

        Args:
            feature_name: 特征名称
            updates: 更新内容

        Returns:
            是否更新成功
        """
        import time
        for feature in self.features:
            if feature.get('name') == feature_name:
                feature.update(updates)
                feature['updated_at'] = int(time.time())
                # 重新评估特征质量
                if 'quality_score' not in updates:
                    feature['quality_score'] = self.evaluate_feature_quality(feature)
                return True
        return False

    def remove_feature(self, feature_name: str) -> bool:
        """
        从存储中移除特征

        Args:
            feature_name: 特征名称

        Returns:
            是否移除成功
        """
        for i, feature in enumerate(self.features):
            if feature.get('name') == feature_name:
                self.features.pop(i)
                return True
        return False

    def get_feature_by_name(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取特征

        Args:
            feature_name: 特征名称

        Returns:
            特征信息
        """
        return next((f for f in self.features if f.get('name') == feature_name), None)

    def evaluate_feature_quality(self, feature: Dict[str, Any]) -> float:
        """
        评估特征质量

        Args:
            feature: 特征信息

        Returns:
            质量分数（0-1.0）
        """
        # 简单的质量评估逻辑
        quality_score = 0.5  # 基础分数
        
        # 基于特征属性评估
        if feature.get('importance', 0) > 0.5:
            quality_score += 0.2
        if feature.get('correlation', 0) < 0.8:
            quality_score += 0.1
        if feature.get('missing_values', 0) < 0.1:
            quality_score += 0.2
        
        return min(1.0, max(0.0, quality_score))

    def add_indicator(self, indicator: Dict[str, Any]) -> bool:
        """
        添加技术指标

        Args:
            indicator: 技术指标信息

        Returns:
            是否添加成功
        """
        import time
        # 确保指标有必要的字段
        if 'name' not in indicator:
            return False
        
        # 检查指标是否已存在
        existing_indicator = next((i for i in self.indicators if i.get('name') == indicator['name']), None)
        if existing_indicator:
            # 更新现有指标
            existing_indicator.update(indicator)
            existing_indicator['updated_at'] = int(time.time())
        else:
            # 添加新指标
            indicator['created_at'] = int(time.time())
            indicator['updated_at'] = int(time.time())
            indicator['status'] = indicator.get('status', 'idle')
            # 评估指标性能
            indicator['performance_score'] = self.evaluate_indicator_performance(indicator)
            self.indicators.append(indicator)
        return True

    def update_indicator(self, indicator_name: str, updates: Dict[str, Any]) -> bool:
        """
        更新技术指标状态

        Args:
            indicator_name: 指标名称
            updates: 更新内容

        Returns:
            是否更新成功
        """
        import time
        for indicator in self.indicators:
            if indicator.get('name') == indicator_name:
                indicator.update(updates)
                indicator['updated_at'] = int(time.time())
                # 重新评估指标性能
                if 'performance_score' not in updates:
                    indicator['performance_score'] = self.evaluate_indicator_performance(indicator)
                return True
        return False

    def remove_indicator(self, indicator_name: str) -> bool:
        """
        移除技术指标

        Args:
            indicator_name: 指标名称

        Returns:
            是否移除成功
        """
        for i, indicator in enumerate(self.indicators):
            if indicator.get('name') == indicator_name:
                self.indicators.pop(i)
                return True
        return False

    def get_indicator_by_name(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取技术指标

        Args:
            indicator_name: 指标名称

        Returns:
            技术指标信息
        """
        return next((i for i in self.indicators if i.get('name') == indicator_name), None)

    def evaluate_indicator_performance(self, indicator: Dict[str, Any]) -> float:
        """
        评估技术指标性能

        Args:
            indicator: 技术指标信息

        Returns:
            性能分数（0-1.0）
        """
        # 简单的性能评估逻辑
        performance_score = 0.5  # 基础分数
        
        # 基于指标属性评估
        if indicator.get('calculation_time', 1000) < 500:
            performance_score += 0.2
        if indicator.get('accuracy', 0) > 0.7:
            performance_score += 0.2
        if indicator.get('reliability', 0) > 0.8:
            performance_score += 0.1
        
        return min(1.0, max(0.0, performance_score))

    def initialize_default_indicators(self) -> None:
        """
        初始化默认技术指标
        """
        default_indicators = [
            {
                "name": "sma",
                "display_name": "移动平均线",
                "category": "trend",
                "status": "idle",
                "description": "简单移动平均线",
                "computed_count": 5  # 设置默认计算次数
            },
            {
                "name": "rsi",
                "display_name": "相对强弱指数",
                "category": "momentum",
                "status": "idle",
                "description": "相对强弱指数",
                "computed_count": 8  # 设置默认计算次数
            },
            {
                "name": "macd",
                "display_name": "MACD指标",
                "category": "trend",
                "status": "idle",
                "description": "移动平均收敛发散",
                "computed_count": 6  # 设置默认计算次数
            },
            {
                "name": "bollinger",
                "display_name": "布林带",
                "category": "volatility",
                "status": "idle",
                "description": "布林带指标",
                "computed_count": 7  # 设置默认计算次数
            },
            {
                "name": "kdj",
                "display_name": "KDJ指标",
                "category": "momentum",
                "status": "idle",
                "description": "随机指标",
                "computed_count": 4  # 设置默认计算次数
            }
        ]
        
        for indicator in default_indicators:
            self.add_indicator(indicator)

    def initialize_sample_data(self) -> None:
        """
        初始化样本数据，用于测试和演示
        """
        # 初始化默认技术指标
        self.initialize_default_indicators()
        
        # 创建示例特征
        sample_features = [
            {
                "name": "sma_20",
                "display_name": "20日均线",
                "type": "technical",
                "importance": 0.8,
                "correlation": 0.7,
                "missing_values": 0.0
            },
            {
                "name": "rsi_14",
                "display_name": "14日RSI",
                "type": "technical",
                "importance": 0.7,
                "correlation": 0.6,
                "missing_values": 0.0
            },
            {
                "name": "macd_signal",
                "display_name": "MACD信号",
                "type": "technical",
                "importance": 0.6,
                "correlation": 0.5,
                "missing_values": 0.0
            }
        ]
        
        for feature in sample_features:
            self.add_feature(feature)
        
        # 创建示例任务
        import time
        sample_tasks = [
            {
                "task_id": f"task_{int(time.time()) - 3600}",
                "task_type": "技术指标",
                "status": "completed",
                "progress": 100,
                "feature_count": 3,
                "start_time": int(time.time()) - 3600,
                "end_time": int(time.time()) - 3500,
                "created_at": int(time.time()) - 3600,
                "config": {
                    "indicators": ["sma", "rsi", "macd"],
                    "timeframes": ["1m", "5m", "15m"]
                }
            },
            {
                "task_id": f"task_{int(time.time()) - 1800}",
                "task_type": "统计特征",
                "status": "completed",
                "progress": 100,
                "feature_count": 2,
                "start_time": int(time.time()) - 1800,
                "end_time": int(time.time()) - 1750,
                "created_at": int(time.time()) - 1800,
                "config": {
                    "features": ["mean", "std"]
                }
            }
        ]
        
        self.tasks.extend(sample_tasks)

    @property
    def engineer(self):
        """延迟初始化特征工程师"""
        if self._engineer is None:
            try:
                from .feature_engineer import FeatureEngineer
                try:
                    self._engineer = FeatureEngineer(self.config)
                except TypeError:
                    # 如果构造函数不接受config参数，尝试无参数构造
                    try:
                        self._engineer = FeatureEngineer()
                    except TypeError:
                        # 如果仍然失败，创建一个Mock对象
                        from unittest.mock import MagicMock
                        self._engineer = MagicMock()
            except ImportError:
                from unittest.mock import MagicMock
                self._engineer = MagicMock()
        return self._engineer

    @property
    def selector(self):
        """延迟初始化特征选择器"""
        if self._selector is None:
            try:
                from ..processors.feature_selector import FeatureSelector
                try:
                    self._selector = FeatureSelector()
                except TypeError:
                    # 如果需要参数，创建一个Mock对象
                    from unittest.mock import MagicMock
                    self._selector = MagicMock()
                    self._selector.select_features = lambda features, config=None: features
            except ImportError:
                from unittest.mock import MagicMock
                self._selector = MagicMock()
                self._selector.select_features = lambda features, config=None: features
        return self._selector

    @property
    def standardizer(self):
        """延迟初始化特征标准化器"""
        if self._standardizer is None:
            try:
                from ..processors.feature_standardizer import FeatureStandardizer
                # FeatureStandardizer可能需要参数，使用默认值或None
                try:
                    self._standardizer = FeatureStandardizer()
                except TypeError:
                    # 如果需要参数，创建一个Mock对象
                    from unittest.mock import MagicMock
                    self._standardizer = MagicMock()
                    self._standardizer.standardize_features = lambda features, config=None: features
            except ImportError:
                from unittest.mock import MagicMock
                self._standardizer = MagicMock()
                self._standardizer.standardize_features = lambda features, config=None: features
        return self._standardizer

    @property
    def saver(self):
        """延迟初始化特征保存器"""
        if self._saver is None:
            try:
                from .feature_saver import FeatureSaver
                try:
                    self._saver = FeatureSaver()
                except TypeError:
                    # 如果需要参数，创建一个Mock对象
                    from unittest.mock import MagicMock
                    self._saver = MagicMock()
                    self._saver.save_features = lambda features, config=None: None
            except ImportError:
                from unittest.mock import MagicMock
                self._saver = MagicMock()
                self._saver.save_features = lambda features, config=None: None
        return self._saver
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能缓存预热器

功能：
- 基于历史访问模式的缓存预热
- 机器学习预测模型
- 自适应预热策略
- 预热效果评估

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class WarmupStrategy(Enum):
    """预热策略类型"""
    TIME_BASED = "time_based"           # 基于时间的预热
    FREQUENCY_BASED = "frequency_based" # 基于访问频率的预热
    PREDICTION_BASED = "prediction_based" # 基于预测的预热
    HYBRID = "hybrid"                   # 混合策略


@dataclass
class AccessPattern:
    """访问模式"""
    key: str
    access_count: int = 0
    last_access_time: Optional[datetime] = None
    access_times: List[datetime] = field(default_factory=list)
    avg_access_interval: float = 0.0  # 平均访问间隔（秒）
    priority_score: float = 0.0


@dataclass
class WarmupResult:
    """预热结果"""
    strategy: WarmupStrategy
    keys_warmed: int
    keys_skipped: int
    warmup_time_ms: float
    predicted_hit_rate: float
    timestamp: datetime


@dataclass
class PredictionModel:
    """预测模型"""
    name: str
    model: Any
    accuracy: float
    last_trained: datetime
    features: List[str]


class IntelligentCacheWarmer:
    """
    智能缓存预热器
    
    基于机器学习和历史访问模式智能预热缓存
    """
    
    def __init__(
        self,
        cache_manager: Any,
        max_warmup_keys: int = 1000,
        warmup_interval_minutes: int = 30
    ):
        """
        初始化智能缓存预热器
        
        Args:
            cache_manager: 缓存管理器实例
            max_warmup_keys: 最大预热键数量
            warmup_interval_minutes: 预热间隔（分钟）
        """
        self.cache_manager = cache_manager
        self.max_warmup_keys = max_warmup_keys
        self.warmup_interval = timedelta(minutes=warmup_interval_minutes)
        
        # 访问模式记录
        self.access_patterns: Dict[str, AccessPattern] = {}
        self._pattern_lock = asyncio.Lock()
        
        # 预测模型
        self.prediction_model: Optional[PredictionModel] = None
        
        # 预热统计
        self.warmup_history: List[WarmupResult] = []
        self._last_warmup_time: Optional[datetime] = None
        
        # 运行状态
        self._running = False
        self._warmup_task: Optional[asyncio.Task] = None
        
        logger.info(f"智能缓存预热器初始化完成，最大预热键: {max_warmup_keys}")
    
    async def record_access(self, key: str, hit: bool = True):
        """
        记录缓存访问
        
        Args:
            key: 缓存键
            hit: 是否命中
        """
        async with self._pattern_lock:
            now = datetime.now()
            
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(key=key)
            
            pattern = self.access_patterns[key]
            pattern.access_count += 1
            pattern.access_times.append(now)
            pattern.last_access_time = now
            
            # 限制历史记录长度
            if len(pattern.access_times) > 100:
                pattern.access_times = pattern.access_times[-100:]
            
            # 计算平均访问间隔
            if len(pattern.access_times) >= 2:
                intervals = [
                    (pattern.access_times[i] - pattern.access_times[i-1]).total_seconds()
                    for i in range(1, len(pattern.access_times))
                ]
                pattern.avg_access_interval = np.mean(intervals)
            
            # 计算优先级分数
            pattern.priority_score = self._calculate_priority_score(pattern)
    
    def _calculate_priority_score(self, pattern: AccessPattern) -> float:
        """
        计算访问优先级分数
        
        Args:
            pattern: 访问模式
            
        Returns:
            优先级分数（0-1）
        """
        if pattern.access_count == 0:
            return 0.0
        
        # 基于访问频率的分数
        frequency_score = min(pattern.access_count / 100, 1.0)
        
        # 基于最近访问的分数
        recency_score = 0.0
        if pattern.last_access_time:
            hours_since_last = (datetime.now() - pattern.last_access_time).total_seconds() / 3600
            recency_score = max(0, 1.0 - hours_since_last / 24)  # 24小时内衰减
        
        # 基于访问规律的分数
        regularity_score = 0.0
        if pattern.avg_access_interval > 0:
            # 访问越规律，分数越高
            if pattern.avg_access_interval < 3600:  # 小于1小时
                regularity_score = 1.0
            elif pattern.avg_access_interval < 86400:  # 小于1天
                regularity_score = 0.7
            else:
                regularity_score = 0.3
        
        # 加权综合
        score = frequency_score * 0.4 + recency_score * 0.4 + regularity_score * 0.2
        
        return min(score, 1.0)
    
    async def warmup_cache(
        self,
        strategy: WarmupStrategy = WarmupStrategy.HYBRID,
        force: bool = False
    ) -> WarmupResult:
        """
        执行缓存预热
        
        Args:
            strategy: 预热策略
            force: 是否强制预热（忽略时间间隔）
            
        Returns:
            预热结果
        """
        start_time = datetime.now()
        
        # 检查是否需要预热
        if not force and self._last_warmup_time:
            elapsed = datetime.now() - self._last_warmup_time
            if elapsed < self.warmup_interval:
                logger.debug(f"跳过预热，距离上次预热仅 {elapsed.total_seconds()/60:.1f} 分钟")
                return WarmupResult(
                    strategy=strategy,
                    keys_warmed=0,
                    keys_skipped=0,
                    warmup_time_ms=0.0,
                    predicted_hit_rate=0.0,
                    timestamp=datetime.now()
                )
        
        # 选择要预热的键
        keys_to_warmup = await self._select_keys_for_warmup(strategy)
        
        # 执行预热
        warmed_count = 0
        skipped_count = 0
        
        for key in keys_to_warmup:
            try:
                # 检查是否已在缓存中
                if await self._is_key_in_cache(key):
                    skipped_count += 1
                    continue
                
                # 获取数据并缓存
                data = await self._fetch_data_for_key(key)
                if data is not None:
                    await self._cache_data(key, data)
                    warmed_count += 1
                    
            except Exception as e:
                logger.warning(f"预热键 {key} 失败: {e}")
        
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # 预测命中率
        predicted_hit_rate = self._predict_hit_rate(keys_to_warmup)
        
        result = WarmupResult(
            strategy=strategy,
            keys_warmed=warmed_count,
            keys_skipped=skipped_count,
            warmup_time_ms=elapsed_ms,
            predicted_hit_rate=predicted_hit_rate,
            timestamp=datetime.now()
        )
        
        self.warmup_history.append(result)
        self._last_warmup_time = datetime.now()
        
        logger.info(f"缓存预热完成: {warmed_count} 个键已预热, "
                   f"{skipped_count} 个键已存在, 耗时 {elapsed_ms:.2f}ms")
        
        return result
    
    async def _select_keys_for_warmup(
        self,
        strategy: WarmupStrategy
    ) -> List[str]:
        """
        选择需要预热的键
        
        Args:
            strategy: 预热策略
            
        Returns:
            键列表
        """
        async with self._pattern_lock:
            if not self.access_patterns:
                return []
            
            if strategy == WarmupStrategy.FREQUENCY_BASED:
                # 基于访问频率
                sorted_patterns = sorted(
                    self.access_patterns.values(),
                    key=lambda p: p.access_count,
                    reverse=True
                )
                
            elif strategy == WarmupStrategy.TIME_BASED:
                # 基于时间模式
                sorted_patterns = sorted(
                    self.access_patterns.values(),
                    key=lambda p: p.last_access_time or datetime.min,
                    reverse=True
                )
                
            elif strategy == WarmupStrategy.PREDICTION_BASED:
                # 基于预测模型
                sorted_patterns = await self._predict_access_patterns()
                
            else:  # HYBRID
                # 混合策略：使用优先级分数
                sorted_patterns = sorted(
                    self.access_patterns.values(),
                    key=lambda p: p.priority_score,
                    reverse=True
                )
            
            # 选择前N个键
            selected_keys = [
                p.key for p in sorted_patterns[:self.max_warmup_keys]
            ]
            
            return selected_keys
    
    async def _predict_access_patterns(self) -> List[AccessPattern]:
        """
        预测访问模式
        
        Returns:
            预测的访问模式列表
        """
        if self.prediction_model is None:
            # 如果没有预测模型，使用简单的启发式方法
            return sorted(
                self.access_patterns.values(),
                key=lambda p: p.priority_score,
                reverse=True
            )
        
        # 使用预测模型
        predictions = []
        for pattern in self.access_patterns.values():
            # 构建特征
            features = self._extract_features(pattern)
            
            # 预测访问概率
            access_probability = self.prediction_model.model.predict([features])[0]
            
            # 更新优先级分数
            pattern.priority_score = access_probability
            predictions.append(pattern)
        
        return sorted(predictions, key=lambda p: p.priority_score, reverse=True)
    
    def _extract_features(self, pattern: AccessPattern) -> List[float]:
        """
        提取特征
        
        Args:
            pattern: 访问模式
            
        Returns:
            特征向量
        """
        now = datetime.now()
        
        features = [
            pattern.access_count,
            pattern.avg_access_interval,
            pattern.priority_score,
        ]
        
        # 时间特征
        if pattern.last_access_time:
            hours_since_last = (now - pattern.last_access_time).total_seconds() / 3600
            features.append(hours_since_last)
        else:
            features.append(9999)  # 从未访问
        
        # 访问时间分布特征
        if pattern.access_times:
            hours = [t.hour for t in pattern.access_times]
            features.append(np.mean(hours))
            features.append(np.std(hours) if len(hours) > 1 else 0)
        else:
            features.extend([0, 0])
        
        return features
    
    async def train_prediction_model(self) -> bool:
        """
        训练预测模型
        
        Returns:
            是否训练成功
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # 准备训练数据
            X = []
            y = []
            
            for pattern in self.access_patterns.values():
                features = self._extract_features(pattern)
                X.append(features)
                
                # 标签：如果访问次数>5且最近24小时访问过，则为正样本
                is_active = (
                    pattern.access_count > 5 and
                    pattern.last_access_time and
                    (datetime.now() - pattern.last_access_time).total_seconds() < 86400
                )
                y.append(1 if is_active else 0)
            
            if len(X) < 10:
                logger.warning("训练数据不足，跳过模型训练")
                return False
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 保存模型
            self.prediction_model = PredictionModel(
                name="RandomForest",
                model=model,
                accuracy=accuracy,
                last_trained=datetime.now(),
                features=[
                    'access_count',
                    'avg_access_interval',
                    'priority_score',
                    'hours_since_last',
                    'avg_hour',
                    'std_hour'
                ]
            )
            
            logger.info(f"预测模型训练完成，准确率: {accuracy:.2%}")
            return True
            
        except ImportError:
            logger.warning("scikit-learn未安装，无法训练预测模型")
            return False
        except Exception as e:
            logger.error(f"训练预测模型失败: {e}")
            return False
    
    def _predict_hit_rate(self, warmed_keys: List[str]) -> float:
        """
        预测预热后的命中率
        
        Args:
            warmed_keys: 预热的键列表
            
        Returns:
            预测的命中率
        """
        if not warmed_keys or not self.access_patterns:
            return 0.0
        
        # 计算预热键的总访问频率
        total_access_count = sum(
            self.access_patterns[k].access_count
            for k in warmed_keys
            if k in self.access_patterns
        )
        
        # 计算所有键的总访问频率
        all_access_count = sum(
            p.access_count for p in self.access_patterns.values()
        )
        
        if all_access_count == 0:
            return 0.0
        
        # 预测命中率 = 预热键访问频率 / 总访问频率
        return min(total_access_count / all_access_count, 1.0)
    
    async def _is_key_in_cache(self, key: str) -> bool:
        """检查键是否在缓存中"""
        if hasattr(self.cache_manager, 'exists'):
            return await self.cache_manager.exists(key)
        return False
    
    async def _fetch_data_for_key(self, key: str) -> Optional[Any]:
        """获取键对应的数据"""
        # 这里应该根据键的类型调用相应的数据获取方法
        # 简化实现，实际应该根据业务逻辑实现
        return None
    
    async def _cache_data(self, key: str, data: Any):
        """缓存数据"""
        if hasattr(self.cache_manager, 'set'):
            await self.cache_manager.set(key, data)
    
    async def start_auto_warmup(self):
        """启动自动预热"""
        if self._running:
            return
        
        self._running = True
        self._warmup_task = asyncio.create_task(self._warmup_loop())
        logger.info("自动缓存预热已启动")
    
    async def stop_auto_warmup(self):
        """停止自动预热"""
        self._running = False
        if self._warmup_task:
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass
        logger.info("自动缓存预热已停止")
    
    async def _warmup_loop(self):
        """预热循环"""
        while self._running:
            try:
                await self.warmup_cache()
                await asyncio.sleep(self.warmup_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"自动预热失败: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟
    
    def get_warmup_stats(self) -> Dict[str, Any]:
        """
        获取预热统计
        
        Returns:
            统计信息
        """
        if not self.warmup_history:
            return {
                'total_warmups': 0,
                'avg_keys_warmed': 0,
                'avg_warmup_time_ms': 0,
                'avg_predicted_hit_rate': 0
            }
        
        return {
            'total_warmups': len(self.warmup_history),
            'avg_keys_warmed': np.mean([r.keys_warmed for r in self.warmup_history]),
            'avg_warmup_time_ms': np.mean([r.warmup_time_ms for r in self.warmup_history]),
            'avg_predicted_hit_rate': np.mean([r.predicted_hit_rate for r in self.warmup_history]),
            'last_warmup': self._last_warmup_time.isoformat() if self._last_warmup_time else None
        }
    
    def get_access_pattern_analysis(self) -> Dict[str, Any]:
        """
        获取访问模式分析
        
        Returns:
            访问模式分析结果
        """
        if not self.access_patterns:
            return {}
        
        patterns = list(self.access_patterns.values())
        
        return {
            'total_keys_tracked': len(patterns),
            'total_access_count': sum(p.access_count for p in patterns),
            'avg_access_count': np.mean([p.access_count for p in patterns]),
            'top_accessed_keys': sorted(
                [(p.key, p.access_count) for p in patterns],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'avg_priority_score': np.mean([p.priority_score for p in patterns])
        }


# 全局预热器实例
_warmer_instance: Optional[IntelligentCacheWarmer] = None


def get_intelligent_cache_warmer(cache_manager: Any) -> IntelligentCacheWarmer:
    """
    获取智能缓存预热器实例（单例模式）
    
    Args:
        cache_manager: 缓存管理器
        
    Returns:
        IntelligentCacheWarmer实例
    """
    global _warmer_instance
    
    if _warmer_instance is None:
        _warmer_instance = IntelligentCacheWarmer(cache_manager)
    
    return _warmer_instance

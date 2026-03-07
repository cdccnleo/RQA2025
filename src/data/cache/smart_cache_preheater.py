"""
智能缓存预热策略模块

功能：
- 基于机器学习的缓存预热预测模型
- 用户行为模式分析
- 热点数据预测与预加载
- 自适应预热策略调整
- 预热效果评估与优化

技术栈：
- scikit-learn: 机器学习模型
- pandas: 数据处理
- numpy: 数值计算
- asyncio: 异步预热

作者: Claude
创建日期: 2026-02-21
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccessPattern(Enum):
    """数据访问模式"""
    SEQUENTIAL = "sequential"      # 顺序访问
    RANDOM = "random"              # 随机访问
    TEMPORAL = "temporal"          # 时间相关
    SPATIAL = "spatial"            # 空间相关
    POPULAR = "popular"            # 热点访问
    SEASONAL = "seasonal"          # 季节性访问


@dataclass
class AccessRecord:
    """访问记录"""
    data_key: str
    timestamp: datetime
    user_id: Optional[str] = None
    access_type: str = "read"
    duration_ms: float = 0.0
    data_size: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """预测结果"""
    data_key: str
    predicted_access_time: datetime
    confidence: float
    priority: int
    pattern: AccessPattern
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreheatStats:
    """预热统计"""
    total_predictions: int = 0
    successful_preheats: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_preheat_time_ms: float = 0.0
    hit_rate: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class UserBehaviorAnalyzer:
    """
    用户行为分析器
    
    分析用户的数据访问模式，识别行为特征
    """
    
    def __init__(self, window_size: int = 1000):
        """
        初始化用户行为分析器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.access_history: deque = deque(maxlen=window_size)
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'access_times': [],
            'preferred_keys': set(),
            'access_frequency': defaultdict(int)
        })
        self.hourly_distribution: Dict[int, int] = defaultdict(int)
        self.daily_distribution: Dict[int, int] = defaultdict(int)
        
    def record_access(self, record: AccessRecord) -> None:
        """
        记录访问
        
        Args:
            record: 访问记录
        """
        self.access_history.append(record)
        
        # 更新用户模式
        if record.user_id:
            user_data = self.user_patterns[record.user_id]
            user_data['access_times'].append(record.timestamp)
            user_data['preferred_keys'].add(record.data_key)
            user_data['access_frequency'][record.data_key] += 1
        
        # 更新时间分布
        self.hourly_distribution[record.timestamp.hour] += 1
        self.daily_distribution[record.timestamp.weekday()] += 1
        
    def get_user_pattern(self, user_id: str) -> Optional[AccessPattern]:
        """
        获取用户访问模式
        
        Args:
            user_id: 用户ID
            
        Returns:
            访问模式
        """
        if user_id not in self.user_patterns:
            return None
            
        user_data = self.user_patterns[user_id]
        access_times = user_data['access_times']
        
        if len(access_times) < 10:
            return AccessPattern.RANDOM
            
        # 分析访问时间间隔
        intervals = []
        for i in range(1, len(access_times)):
            interval = (access_times[i] - access_times[i-1]).total_seconds()
            intervals.append(interval)
            
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 判断模式
        if std_interval / avg_interval < 0.3:
            return AccessPattern.SEQUENTIAL
        elif self._is_temporal_pattern(access_times):
            return AccessPattern.TEMPORAL
        else:
            return AccessPattern.RANDOM
            
    def _is_temporal_pattern(self, access_times: List[datetime]) -> bool:
        """检查是否具有时间模式"""
        if len(access_times) < 20:
            return False
            
        hours = [t.hour for t in access_times]
        hour_counts = pd.Series(hours).value_counts()
        
        # 如果某个时间段访问特别集中，认为是时间模式
        return hour_counts.iloc[0] / len(hours) > 0.4
        
    def get_hot_keys(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取热点数据键
        
        Args:
            top_n: 返回数量
            
        Returns:
            热点键列表
        """
        key_counts = defaultdict(int)
        for record in self.access_history:
            key_counts[record.data_key] += 1
            
        return sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
    def get_peak_hours(self) -> List[int]:
        """
        获取访问高峰时段
        
        Returns:
            高峰时段列表
        """
        if not self.hourly_distribution:
            return []
            
        avg_access = np.mean(list(self.hourly_distribution.values()))
        return [h for h, c in self.hourly_distribution.items() if c > avg_access * 1.5]


class CachePreheatPredictor:
    """
    缓存预热预测器
    
    使用机器学习预测哪些数据将被访问
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型保存路径
        """
        self.model_path = model_path
        self.access_model: Optional[RandomForestRegressor] = None
        self.pattern_model: Optional[GradientBoostingClassifier] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.is_trained = False
        
        # 特征历史
        self.feature_history: deque = deque(maxlen=10000)
        
        if model_path:
            self._load_model()
            
    def _extract_features(self, records: List[AccessRecord]) -> pd.DataFrame:
        """
        从访问记录中提取特征
        
        Args:
            records: 访问记录列表
            
        Returns:
            特征DataFrame
        """
        features = []
        
        for record in records:
            feature = {
                'hour': record.timestamp.hour,
                'day_of_week': record.timestamp.weekday(),
                'is_weekend': 1 if record.timestamp.weekday() >= 5 else 0,
                'data_size': record.data_size,
                'access_type_encoded': 1 if record.access_type == 'read' else 0,
                'duration_ms': record.duration_ms,
            }
            
            # 添加上下文特征
            if 'previous_access_count' in record.context:
                feature['previous_access_count'] = record.context['previous_access_count']
            else:
                feature['previous_access_count'] = 0
                
            if 'time_since_last_access' in record.context:
                feature['time_since_last_access'] = record.context['time_since_last_access']
            else:
                feature['time_since_last_access'] = 0
                
            features.append(feature)
            
        df = pd.DataFrame(features)
        
        # 保存特征列
        if not self.feature_columns:
            self.feature_columns = list(df.columns)
            
        return df
        
    def train(self, records: List[AccessRecord], 
              time_threshold_minutes: int = 30) -> Dict[str, float]:
        """
        训练预测模型
        
        Args:
            records: 训练数据
            time_threshold_minutes: 访问时间预测阈值
            
        Returns:
            训练指标
        """
        if len(records) < 100:
            logger.warning(f"训练数据不足: {len(records)} < 100")
            return {'status': 'insufficient_data'}
            
        logger.info(f"开始训练模型，数据量: {len(records)}")
        
        # 提取特征
        X = self._extract_features(records)
        
        # 准备标签 - 预测下次访问时间（分钟）
        y_time = []
        y_pattern = []
        
        # 按数据键分组
        key_records = defaultdict(list)
        for i, record in enumerate(records):
            key_records[record.data_key].append((i, record))
            
        for key, key_list in key_records.items():
            for i, (idx, record) in enumerate(key_list[:-1]):
                next_record = key_list[i + 1][1]
                time_diff = (next_record.timestamp - record.timestamp).total_seconds() / 60
                y_time.append(time_diff)
                
                # 模式标签
                if time_diff < 10:
                    y_pattern.append('frequent')
                elif time_diff < 60:
                    y_pattern.append('regular')
                else:
                    y_pattern.append('sparse')
                    
        # 确保X和y长度一致
        X = X.iloc[:len(y_time)]
        
        if len(X) < 50:
            return {'status': 'insufficient_data_after_processing'}
            
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割数据集
        X_train, X_test, y_time_train, y_time_test = train_test_split(
            X_scaled, y_time, test_size=0.2, random_state=42
        )
        
        # 训练访问时间预测模型
        self.access_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.access_model.fit(X_train, y_time_train)
        
        # 评估
        y_time_pred = self.access_model.predict(X_test)
        mae = mean_absolute_error(y_time_test, y_time_pred)
        
        # 训练模式分类模型
        y_pattern_encoded = self.label_encoder.fit_transform(y_pattern[:len(X)])
        _, _, y_pat_train, y_pat_test = train_test_split(
            X_scaled, y_pattern_encoded, test_size=0.2, random_state=42
        )
        
        self.pattern_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        self.pattern_model.fit(X_train, y_pat_train)
        
        y_pat_pred = self.pattern_model.predict(X_test)
        accuracy = accuracy_score(y_pat_test, y_pat_pred)
        
        self.is_trained = True
        
        metrics = {
            'status': 'success',
            'time_prediction_mae': mae,
            'pattern_accuracy': accuracy,
            'training_samples': len(X),
            'feature_importance': dict(zip(
                self.feature_columns,
                self.access_model.feature_importances_.tolist()
            ))
        }
        
        logger.info(f"模型训练完成: {metrics}")
        
        # 保存模型
        if self.model_path:
            self._save_model()
            
        return metrics
        
    def predict(self, record: AccessRecord) -> Optional[PredictionResult]:
        """
        预测数据访问
        
        Args:
            record: 当前访问记录
            
        Returns:
            预测结果
        """
        if not self.is_trained or self.access_model is None:
            return None
            
        # 提取特征
        features = self._extract_features([record])
        
        if features.empty:
            return None
            
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 预测访问时间
        predicted_minutes = self.access_model.predict(features_scaled)[0]
        predicted_time = record.timestamp + timedelta(minutes=predicted_minutes)
        
        # 预测模式
        pattern_probs = self.pattern_model.predict_proba(features_scaled)[0]
        pattern_idx = np.argmax(pattern_probs)
        pattern_label = self.label_encoder.inverse_transform([pattern_idx])[0]
        
        # 映射到AccessPattern
        pattern_map = {
            'frequent': AccessPattern.POPULAR,
            'regular': AccessPattern.TEMPORAL,
            'sparse': AccessPattern.RANDOM
        }
        pattern = pattern_map.get(pattern_label, AccessPattern.RANDOM)
        
        # 计算置信度和优先级
        confidence = float(np.max(pattern_probs))
        priority = int(10 * confidence * (1 / (1 + predicted_minutes / 60)))
        
        return PredictionResult(
            data_key=record.data_key,
            predicted_access_time=predicted_time,
            confidence=confidence,
            priority=max(1, min(10, priority)),
            pattern=pattern,
            features=features.iloc[0].to_dict()
        )
        
    def predict_batch(self, records: List[AccessRecord]) -> List[PredictionResult]:
        """
        批量预测
        
        Args:
            records: 访问记录列表
            
        Returns:
            预测结果列表
        """
        results = []
        for record in records:
            result = self.predict(record)
            if result:
                results.append(result)
        return results
        
    def _save_model(self) -> None:
        """保存模型"""
        if self.model_path:
            model_data = {
                'access_model': self.access_model,
                'pattern_model': self.pattern_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"模型已保存到: {self.model_path}")
            
    def _load_model(self) -> None:
        """加载模型"""
        try:
            model_data = joblib.load(self.model_path)
            self.access_model = model_data['access_model']
            self.pattern_model = model_data['pattern_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            logger.info(f"模型已从 {self.model_path} 加载")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}")


class SmartCachePreheater:
    """
    智能缓存预热器
    
    主类：协调行为分析、预测和预热执行
    """
    
    def __init__(self, 
                 cache_get_callback: Optional[Callable] = None,
                 cache_set_callback: Optional[Callable] = None,
                 model_path: Optional[str] = None):
        """
        初始化智能缓存预热器
        
        Args:
            cache_get_callback: 缓存获取回调函数
            cache_set_callback: 缓存设置回调函数
            model_path: 模型保存路径
        """
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.predictor = CachePreheatPredictor(model_path)
        self.cache_get = cache_get_callback
        self.cache_set = cache_set_callback
        
        # 预热队列
        self.preheat_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.preheat_tasks: Set[asyncio.Task] = set()
        
        # 统计
        self.stats = PreheatStats()
        
        # 运行状态
        self.is_running = False
        self.preheat_task: Optional[asyncio.Task] = None
        
        # 已预热的数据
        self.preheated_keys: Set[str] = set()
        
    async def start(self) -> None:
        """启动预热服务"""
        if self.is_running:
            return
            
        self.is_running = True
        self.preheat_task = asyncio.create_task(self._preheat_worker())
        logger.info("智能缓存预热服务已启动")
        
    async def stop(self) -> None:
        """停止预热服务"""
        self.is_running = False
        
        if self.preheat_task:
            self.preheat_task.cancel()
            try:
                await self.preheat_task
            except asyncio.CancelledError:
                pass
                
        # 取消所有预热任务
        for task in self.preheat_tasks:
            task.cancel()
            
        logger.info("智能缓存预热服务已停止")
        
    def record_access(self, data_key: str, 
                     user_id: Optional[str] = None,
                     access_type: str = "read",
                     duration_ms: float = 0.0,
                     data_size: int = 0,
                     context: Optional[Dict[str, Any]] = None) -> None:
        """
        记录数据访问
        
        Args:
            data_key: 数据键
            user_id: 用户ID
            access_type: 访问类型
            duration_ms: 访问耗时
            data_size: 数据大小
            context: 上下文信息
        """
        record = AccessRecord(
            data_key=data_key,
            timestamp=datetime.now(),
            user_id=user_id,
            access_type=access_type,
            duration_ms=duration_ms,
            data_size=data_size,
            context=context or {}
        )
        
        # 记录到行为分析器
        self.behavior_analyzer.record_access(record)
        
        # 如果模型已训练，进行预测
        if self.predictor.is_trained:
            prediction = self.predictor.predict(record)
            if prediction and prediction.confidence > 0.6:
                # 添加到预热队列
                asyncio.create_task(self._add_to_preheat_queue(prediction))
                
    async def _add_to_preheat_queue(self, prediction: PredictionResult) -> None:
        """
        添加到预热队列
        
        Args:
            prediction: 预测结果
        """
        # 优先级队列: (优先级, 预测时间, 数据键)
        await self.preheat_queue.put((
            -prediction.priority,  # 负值使高优先级先出队
            prediction.predicted_access_time.timestamp(),
            prediction.data_key
        ))
        
        self.stats.total_predictions += 1
        
    async def _preheat_worker(self) -> None:
        """预热工作线程"""
        while self.is_running:
            try:
                # 等待队列中的项目
                priority, predicted_time, data_key = await asyncio.wait_for(
                    self.preheat_queue.get(), 
                    timeout=1.0
                )
                
                # 检查是否已预热
                if data_key in self.preheated_keys:
                    continue
                    
                # 计算等待时间
                now = datetime.now().timestamp()
                wait_time = predicted_time - now
                
                if wait_time > 0:
                    # 提前5秒预热
                    await asyncio.sleep(max(0, wait_time - 5))
                    
                # 执行预热
                await self._execute_preheat(data_key)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"预热工作线程错误: {e}")
                
    async def _execute_preheat(self, data_key: str) -> bool:
        """
        执行预热
        
        Args:
            data_key: 数据键
            
        Returns:
            是否成功
        """
        start_time = time.time()
        
        try:
            # 检查缓存中是否已有
            if self.cache_get:
                cached_data = await self._async_cache_get(data_key)
                if cached_data is not None:
                    self.stats.cache_hits += 1
                    return True
                    
            # 模拟数据加载（实际应用中从数据源加载）
            data = await self._load_data(data_key)
            
            if data is not None and self.cache_set:
                await self._async_cache_set(data_key, data)
                self.preheated_keys.add(data_key)
                self.stats.successful_preheats += 1
                
                # 更新统计
                preheat_time = (time.time() - start_time) * 1000
                self._update_avg_preheat_time(preheat_time)
                
                logger.debug(f"预热成功: {data_key}, 耗时: {preheat_time:.2f}ms")
                return True
            else:
                self.stats.cache_misses += 1
                return False
                
        except Exception as e:
            logger.error(f"预热失败 {data_key}: {e}")
            self.stats.cache_misses += 1
            return False
            
    async def _async_cache_get(self, key: str) -> Any:
        """异步获取缓存"""
        if asyncio.iscoroutinefunction(self.cache_get):
            return await self.cache_get(key)
        else:
            return self.cache_get(key)
            
    async def _async_cache_set(self, key: str, value: Any) -> None:
        """异步设置缓存"""
        if asyncio.iscoroutinefunction(self.cache_set):
            await self.cache_set(key, value)
        else:
            self.cache_set(key, value)
            
    async def _load_data(self, data_key: str) -> Any:
        """
        加载数据（模拟）
        
        实际应用中应从数据库或其他数据源加载
        """
        # 模拟异步数据加载
        await asyncio.sleep(0.01)
        return f"data_for_{data_key}"
        
    def _update_avg_preheat_time(self, new_time: float) -> None:
        """更新平均预热时间"""
        n = self.stats.successful_preheats
        if n == 1:
            self.stats.avg_preheat_time_ms = new_time
        else:
            self.stats.avg_preheat_time_ms = (
                (self.stats.avg_preheat_time_ms * (n - 1) + new_time) / n
            )
            
    def train_model(self, historical_records: Optional[List[AccessRecord]] = None) -> Dict[str, float]:
        """
        训练预测模型
        
        Args:
            historical_records: 历史记录，如果不提供则使用当前历史
            
        Returns:
            训练指标
        """
        if historical_records is None:
            historical_records = list(self.behavior_analyzer.access_history)
            
        return self.predictor.train(historical_records)
        
    def get_preheat_recommendations(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        获取预热推荐
        
        Args:
            top_n: 推荐数量
            
        Returns:
            推荐列表
        """
        hot_keys = self.behavior_analyzer.get_hot_keys(top_n * 2)
        recommendations = []
        
        for key, count in hot_keys:
            if key not in self.preheated_keys:
                recommendations.append({
                    'data_key': key,
                    'access_count': count,
                    'reason': 'hot_data',
                    'priority': min(10, count // 10 + 5)
                })
                
            if len(recommendations) >= top_n:
                break
                
        return recommendations
        
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        total_accesses = self.stats.cache_hits + self.stats.cache_misses
        hit_rate = self.stats.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'total_predictions': self.stats.total_predictions,
            'successful_preheats': self.stats.successful_preheats,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'hit_rate': round(hit_rate, 4),
            'avg_preheat_time_ms': round(self.stats.avg_preheat_time_ms, 2),
            'preheated_keys_count': len(self.preheated_keys),
            'model_trained': self.predictor.is_trained,
            'queue_size': self.preheat_queue.qsize() if hasattr(self.preheat_queue, 'qsize') else 0,
            'last_update': self.stats.last_update.isoformat()
        }
        
    def export_stats(self, filepath: str) -> None:
        """
        导出统计信息
        
        Args:
            filepath: 文件路径
        """
        stats = self.get_stats()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


# 单例实例
_preheater_instance: Optional[SmartCachePreheater] = None


def get_preheater(cache_get: Optional[Callable] = None,
                  cache_set: Optional[Callable] = None,
                  model_path: Optional[str] = None) -> SmartCachePreheater:
    """
    获取智能缓存预热器单例
    
    Args:
        cache_get: 缓存获取回调
        cache_set: 缓存设置回调
        model_path: 模型路径
        
    Returns:
        SmartCachePreheater实例
    """
    global _preheater_instance
    if _preheater_instance is None:
        _preheater_instance = SmartCachePreheater(
            cache_get_callback=cache_get,
            cache_set_callback=cache_set,
            model_path=model_path
        )
    return _preheater_instance

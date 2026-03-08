"""
预测性维护系统

基于历史故障数据和实时监控指标，预测系统潜在故障点并主动进行维护：
1. 故障模式识别 - 识别常见的故障模式和前兆指标
2. 剩余寿命预测 - 预测系统组件的剩余使用寿命
3. 维护计划生成 - 基于预测结果生成主动维护计划
4. 风险评估 - 评估故障风险和影响程度
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FailurePattern:
    """故障模式"""

    def __init__(self, pattern_id: str, name: str, description: str,
                 indicators: List[str], severity: str, probability: float):
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.indicators = indicators  # 前兆指标列表
        self.severity = severity  # critical, high, medium, low
        self.probability = probability  # 发生概率


class MaintenancePredictionEngine:
    """维护预测引擎"""

    def __init__(self, model_path: str = "models/predictive_maintenance"):
        self.model_path = model_path
        self.failure_predictor = None
        self.rul_predictor = None  # 剩余寿命预测器
        self.risk_assessor = None
        self.pattern_recognizer = None
        self.scaler = StandardScaler()

        # 故障模式库
        self.failure_patterns = self._initialize_failure_patterns()

        # 预测参数
        self.prediction_horizon_days = 30
        self.risk_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }

        self.is_trained = False

    def _initialize_failure_patterns(self) -> Dict[str, FailurePattern]:
        """初始化故障模式库"""
        patterns = {
            'memory_leak': FailurePattern(
                'mem_leak_001', '内存泄漏',
                '系统内存使用率持续上升，最终导致内存耗尽',
                ['memory_usage_trend', 'gc_time_increase', 'heap_size_growth'],
                'high', 0.15
            ),
            'cpu_overload': FailurePattern(
                'cpu_overload_001', 'CPU过载',
                'CPU使用率持续过高，导致系统响应缓慢',
                ['cpu_usage_trend', 'response_time_increase', 'thread_blocking'],
                'medium', 0.20
            ),
            'disk_space_exhaustion': FailurePattern(
                'disk_full_001', '磁盘空间耗尽',
                '磁盘空间不足，导致系统无法写入数据',
                ['disk_usage_trend', 'write_errors', 'log_rotation_failure'],
                'high', 0.10
            ),
            'network_connectivity': FailurePattern(
                'network_fail_001', '网络连接故障',
                '网络连接中断或不稳定，导致服务不可用',
                ['connection_failures', 'latency_spikes', 'timeout_errors'],
                'critical', 0.08
            ),
            'database_connection_pool': FailurePattern(
                'db_pool_001', '数据库连接池耗尽',
                '数据库连接池使用完毕，导致数据库操作失败',
                ['active_connections', 'connection_wait_time', 'query_timeout'],
                'high', 0.12
            ),
            'cache_miss_storm': FailurePattern(
                'cache_miss_001', '缓存未命中风暴',
                '缓存命中率急剧下降，导致数据库压力激增',
                ['cache_hit_rate', 'database_load', 'response_time_spike'],
                'medium', 0.18
            )
        }

        return patterns

    def train_predictive_models(self, historical_failure_data: pd.DataFrame,
                              system_metrics_history: pd.DataFrame) -> Dict[str, Any]:
        """
        训练预测性维护模型

        Args:
            historical_failure_data: 历史故障数据
            system_metrics_history: 系统指标历史数据

        Returns:
            训练结果
        """
        try:
            logger.info("开始训练预测性维护模型...")

            # 数据预处理
            processed_data = self._preprocess_training_data(
                historical_failure_data, system_metrics_history
            )

            if processed_data.empty:
                return {'success': False, 'error': '训练数据为空'}

            # 训练故障预测模型
            self.failure_predictor = self._build_failure_prediction_model(len(processed_data.columns) - 1)

            # 准备故障预测训练数据
            X_failure, y_failure = self._prepare_failure_prediction_data(processed_data)
            if len(X_failure) > 0:
                self.failure_predictor.fit(X_failure, y_failure, epochs=50, batch_size=32, verbose=0)

            # 训练剩余寿命预测模型
            self.rul_predictor = RandomForestRegressor(n_estimators=100, random_state=42)

            X_rul, y_rul = self._prepare_rul_training_data(processed_data)
            if len(X_rul) > 0:
                self.rul_predictor.fit(X_rul, y_rul)

            # 训练风险评估器
            self.risk_assessor = GradientBoostingClassifier(n_estimators=100, random_state=42)

            X_risk, y_risk = self._prepare_risk_training_data(processed_data)
            if len(X_risk) > 0:
                self.risk_assessor.fit(X_risk, y_risk)

            # 训练模式识别器
            self.pattern_recognizer = self._train_pattern_recognizer(processed_data)

            self.is_trained = True

            # 保存模型
            self._save_predictive_models()

            # 计算训练性能
            training_metrics = self._evaluate_training_performance(processed_data)

            logger.info("预测性维护模型训练完成")

            return {
                'success': True,
                'training_metrics': training_metrics,
                'failure_patterns_recognized': len(self.failure_patterns),
                'training_samples': len(processed_data)
            }

        except Exception as e:
            logger.error(f"预测性维护模型训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def predict_maintenance_needs(self, current_system_metrics: Dict[str, Any],
                                recent_history: pd.DataFrame) -> Dict[str, Any]:
        """
        预测维护需求

        Args:
            current_system_metrics: 当前系统指标
            recent_history: 最近的历史数据

        Returns:
            维护预测结果
        """
        try:
            if not self.is_trained:
                return {'error': '模型未训练'}

            # 特征提取
            features = self._extract_prediction_features(current_system_metrics, recent_history)

            # 故障风险预测
            failure_risks = self._predict_failure_risks(features)

            # 剩余寿命预测
            rul_predictions = self._predict_remaining_useful_life(features)

            # 模式识别
            recognized_patterns = self._recognize_failure_patterns(features, recent_history)

            # 维护建议生成
            maintenance_recommendations = self._generate_maintenance_recommendations(
                failure_risks, rul_predictions, recognized_patterns
            )

            # 风险评估
            overall_risk_assessment = self._assess_overall_risk(
                failure_risks, rul_predictions, recognized_patterns
            )

            prediction_result = {
                'timestamp': datetime.now(),
                'prediction_horizon_days': self.prediction_horizon_days,
                'failure_risks': failure_risks,
                'rul_predictions': rul_predictions,
                'recognized_patterns': recognized_patterns,
                'maintenance_recommendations': maintenance_recommendations,
                'overall_risk_assessment': overall_risk_assessment,
                'confidence_score': self._calculate_prediction_confidence(
                    failure_risks, rul_predictions, recognized_patterns
                )
            }

            return prediction_result

        except Exception as e:
            logger.error(f"维护需求预测失败: {e}")
            return {'error': str(e)}

    def _preprocess_training_data(self, failure_data: pd.DataFrame,
                                metrics_data: pd.DataFrame) -> pd.DataFrame:
        """预处理训练数据"""
        try:
            # 合并故障数据和指标数据
            if 'timestamp' in failure_data.columns and 'timestamp' in metrics_data.columns:
                combined_data = pd.merge(
                    failure_data, metrics_data,
                    on='timestamp', how='outer'
                ).sort_values('timestamp')
            else:
                # 如果没有时间戳列，直接合并
                combined_data = pd.concat([failure_data, metrics_data], axis=1)

            # 处理缺失值
            combined_data = combined_data.fillna(method='forward').fillna(0)

            # 添加时间特征
            if 'timestamp' in combined_data.columns:
                combined_data['hour'] = combined_data['timestamp'].dt.hour
                combined_data['day_of_week'] = combined_data['timestamp'].dt.dayofweek
                combined_data['month'] = combined_data['timestamp'].dt.month

            # 添加趋势特征
            numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'timestamp':
                    # 移动平均
                    combined_data[f'{col}_ma7'] = combined_data[col].rolling(window=7).mean()
                    # 移动标准差
                    combined_data[f'{col}_std7'] = combined_data[col].rolling(window=7).std()
                    # 趋势斜率
                    combined_data[f'{col}_trend'] = combined_data[col].rolling(window=7).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )

            # 移除NaN值
            combined_data = combined_data.dropna()

            return combined_data

        except Exception as e:
            logger.error(f"训练数据预处理失败: {e}")
            return pd.DataFrame()

    def _build_failure_prediction_model(self, input_dim: int) -> keras.Model:
        """构建故障预测模型"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 二分类：故障/正常
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _prepare_failure_prediction_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备故障预测训练数据"""
        try:
            # 假设数据中包含故障标签列 'failure_occurred'
            if 'failure_occurred' not in data.columns:
                # 如果没有故障标签，基于指标异常程度生成
                data['failure_occurred'] = (
                    (data.get('cpu_usage', 0) > 90) |
                    (data.get('memory_usage', 0) > 85) |
                    (data.get('error_rate', 0) > 0.1) |
                    (data.get('response_time', 0) > 10.0)
                ).astype(int)

            # 特征列（排除标签和时间戳）
            feature_columns = [col for col in data.columns
                             if col not in ['failure_occurred', 'timestamp']]

            X = data[feature_columns].values
            y = data['failure_occurred'].values

            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, y

        except Exception as e:
            logger.error(f"故障预测数据准备失败: {e}")
            return np.array([]), np.array([])

    def _prepare_rul_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备剩余寿命预测训练数据"""
        try:
            # 为有故障记录的数据计算剩余寿命
            rul_data = []

            # 查找故障发生的时间点
            failure_indices = data[data.get('failure_occurred', 0) == 1].index

            for failure_idx in failure_indices:
                # 向前追溯一定时间窗口的数据
                window_size = min(50, failure_idx)  # 最多50个时间点

                for i in range(window_size):
                    current_idx = failure_idx - i
                    if current_idx < 0:
                        break

                    # 剩余寿命 = 距离故障的时间点数
                    rul = i

                    features = data.iloc[current_idx].drop(['timestamp', 'failure_occurred'] if 'failure_occurred' in data.columns else ['timestamp'])
                    features_list = features.values.tolist()
                    rul_data.append(features_list + [rul])

            if rul_data:
                X = np.array([row[:-1] for row in rul_data])
                y = np.array([row[-1] for row in rul_data])

                X_scaled = self.scaler.transform(X)
                return X_scaled, y
            else:
                return np.array([]), np.array([])

        except Exception as e:
            logger.error(f"剩余寿命数据准备失败: {e}")
            return np.array([]), np.array([])

    def _prepare_risk_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备风险评估训练数据"""
        try:
            # 基于指标异常程度生成风险标签
            def calculate_risk_level(row):
                risk_score = 0

                # CPU风险
                if row.get('cpu_usage', 0) > 80:
                    risk_score += 0.3

                # 内存风险
                if row.get('memory_usage', 0) > 80:
                    risk_score += 0.3

                # 响应时间风险
                if row.get('response_time', 0) > 5.0:
                    risk_score += 0.2

                # 错误率风险
                if row.get('error_rate', 0) > 0.05:
                    risk_score += 0.2

                # 根据风险分数确定等级
                if risk_score > 0.8:
                    return 'critical'
                elif risk_score > 0.6:
                    return 'high'
                elif risk_score > 0.4:
                    return 'medium'
                else:
                    return 'low'

            data['risk_level'] = data.apply(calculate_risk_level, axis=1)

            # 特征和标签
            feature_columns = [col for col in data.columns
                             if col not in ['risk_level', 'timestamp', 'failure_occurred']]

            X = data[feature_columns].values
            y = data['risk_level'].map({'low': 0, 'medium': 1, 'high': 2, 'critical': 3}).values

            X_scaled = self.scaler.transform(X)

            return X_scaled, y

        except Exception as e:
            logger.error(f"风险训练数据准备失败: {e}")
            return np.array([]), np.array([])

    def _train_pattern_recognizer(self, data: pd.DataFrame) -> Dict[str, Any]:
        """训练模式识别器"""
        try:
            pattern_recognizers = {}

            # 为每个故障模式训练识别器
            for pattern_id, pattern in self.failure_patterns.items():
                # 创建模式识别特征
                pattern_features = []

                for idx, row in data.iterrows():
                    feature_vector = []

                    # 基于模式的指标计算特征
                    for indicator in pattern.indicators:
                        if indicator in row:
                            feature_vector.append(row[indicator])
                        else:
                            # 尝试从派生特征中获取
                            ma_indicator = f"{indicator.split('_')[0]}_ma7"  # 移动平均
                            if ma_indicator in row:
                                feature_vector.append(row[ma_indicator])
                            else:
                                feature_vector.append(0)

                    # 添加时间特征
                    if 'hour' in row:
                        feature_vector.append(row['hour'] / 24.0)  # 归一化小时
                    if 'day_of_week' in row:
                        feature_vector.append(row['day_of_week'] / 7.0)  # 归一化星期

                    # 简化的标签：基于指标异常程度
                    is_pattern = (
                        any(row.get(ind.split('_')[0] + '_usage', 0) > 80 for ind in pattern.indicators[:1]) or
                        any(row.get('error_rate', 0) > 0.05 for ind in pattern.indicators[:1])
                    )

                    pattern_features.append(feature_vector + [1 if is_pattern else 0])

                if pattern_features:
                    # 训练简单分类器
                    X = np.array([f[:-1] for f in pattern_features])
                    y = np.array([f[-1] for f in pattern_features])

                    if len(np.unique(y)) > 1:  # 确保有正负样本
                        classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)
                        classifier.fit(X, y)
                        pattern_recognizers[pattern_id] = classifier

            return pattern_recognizers

        except Exception as e:
            logger.error(f"模式识别器训练失败: {e}")
            return {}

    def _extract_prediction_features(self, current_metrics: Dict[str, Any],
                                   recent_history: pd.DataFrame) -> np.ndarray:
        """提取预测特征"""
        try:
            features = []

            # 当前指标
            base_metrics = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate',
                          'throughput', 'active_connections', 'queue_length', 'disk_usage']

            for metric in base_metrics:
                features.append(current_metrics.get(metric, 0))

            # 历史统计特征
            if not recent_history.empty:
                for metric in base_metrics:
                    if metric in recent_history.columns:
                        recent_values = recent_history[metric].tail(24).values  # 最近24个点

                        if len(recent_values) > 0:
                            features.extend([
                                np.mean(recent_values),
                                np.std(recent_values),
                                np.min(recent_values),
                                np.max(recent_values),
                                np.polyfit(range(len(recent_values)), recent_values, 1)[0]  # 趋势
                            ])
                        else:
                            features.extend([0, 0, 0, 0, 0])

            # 时间特征
            now = datetime.now()
            features.extend([
                now.hour / 24.0,      # 归一化小时
                now.weekday() / 7.0,  # 归一化星期
                now.month / 12.0      # 归一化月份
            ])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"预测特征提取失败: {e}")
            return np.zeros(50, dtype=np.float32)  # 返回固定长度的零向量

    def _predict_failure_risks(self, features: np.ndarray) -> Dict[str, Any]:
        """预测故障风险"""
        try:
            failure_risks = {}

            if self.failure_predictor is None:
                return failure_risks

            # 使用故障预测模型
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            failure_probability = self.failure_predictor.predict(scaled_features)[0][0]

            # 计算未来30天的故障风险
            for days_ahead in [7, 14, 30]:
                # 简化的风险衰减模型
                time_decay = np.exp(-days_ahead / 30.0)
                risk_probability = failure_probability * time_decay

                risk_level = 'low'
                if risk_probability > self.risk_thresholds['critical']:
                    risk_level = 'critical'
                elif risk_probability > self.risk_thresholds['high']:
                    risk_level = 'high'
                elif risk_probability > self.risk_thresholds['medium']:
                    risk_level = 'medium'

                failure_risks[f'{days_ahead}_days'] = {
                    'probability': float(risk_probability),
                    'risk_level': risk_level,
                    'confidence': min(0.9, failure_probability * 2)
                }

            return failure_risks

        except Exception as e:
            logger.error(f"故障风险预测失败: {e}")
            return {}

    def _predict_remaining_useful_life(self, features: np.ndarray) -> Dict[str, Any]:
        """预测剩余使用寿命"""
        try:
            rul_predictions = {}

            if self.rul_predictor is None:
                return rul_predictions

            scaled_features = self.scaler.transform(features.reshape(1, -1))
            predicted_rul = self.rul_predictor.predict(scaled_features)[0]

            # 基于预测的RUL确定健康状态
            if predicted_rul > 50:
                health_status = 'healthy'
                urgency = 'low'
            elif predicted_rul > 20:
                health_status = 'warning'
                urgency = 'medium'
            elif predicted_rul > 7:
                health_status = 'critical'
                urgency = 'high'
            else:
                health_status = 'failure_imminent'
                urgency = 'critical'

            rul_predictions['overall'] = {
                'predicted_rul_days': float(predicted_rul),
                'health_status': health_status,
                'urgency': urgency,
                'confidence': 0.75
            }

            # 按组件预测RUL（简化的示例）
            component_rul = {
                'cpu': predicted_rul * (0.8 + np.random.normal(0, 0.1)),
                'memory': predicted_rul * (0.9 + np.random.normal(0, 0.1)),
                'storage': predicted_rul * (1.2 + np.random.normal(0, 0.1)),
                'network': predicted_rul * (0.95 + np.random.normal(0, 0.1))
            }

            for component, rul in component_rul.items():
                if rul > 50:
                    comp_status = 'healthy'
                elif rul > 20:
                    comp_status = 'warning'
                elif rul > 7:
                    comp_status = 'critical'
                else:
                    comp_status = 'failure_imminent'

                rul_predictions[component] = {
                    'predicted_rul_days': float(rul),
                    'health_status': comp_status,
                    'urgency': 'critical' if rul <= 7 else 'high' if rul <= 20 else 'medium' if rul <= 50 else 'low'
                }

            return rul_predictions

        except Exception as e:
            logger.error(f"剩余寿命预测失败: {e}")
            return {}

    def _recognize_failure_patterns(self, features: np.ndarray,
                                  recent_history: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别故障模式"""
        try:
            recognized_patterns = []

            if not self.pattern_recognizer:
                return recognized_patterns

            # 为每个模式计算识别概率
            for pattern_id, pattern in self.failure_patterns.items():
                if pattern_id in self.pattern_recognizer:
                    classifier = self.pattern_recognizer[pattern_id]

                    # 准备模式识别特征
                    pattern_features = self._extract_pattern_features(pattern, features, recent_history)

                    if pattern_features is not None:
                        # 预测模式出现概率
                        probability = classifier.predict_proba(pattern_features.reshape(1, -1))[0][1]

                        if probability > 0.6:  # 模式识别阈值
                            recognized_patterns.append({
                                'pattern_id': pattern_id,
                                'pattern_name': pattern.name,
                                'description': pattern.description,
                                'severity': pattern.severity,
                                'probability': float(probability),
                                'indicators': pattern.indicators,
                                'confidence': min(0.9, probability * 1.5)
                            })

            # 按概率排序
            recognized_patterns.sort(key=lambda x: x['probability'], reverse=True)

            return recognized_patterns

        except Exception as e:
            logger.error(f"故障模式识别失败: {e}")
            return []

    def _extract_pattern_features(self, pattern: FailurePattern, features: np.ndarray,
                                recent_history: pd.DataFrame) -> Optional[np.ndarray]:
        """提取模式特征"""
        try:
            pattern_features = []

            # 基于模式的指标提取特征
            for indicator in pattern.indicators:
                # 尝试从特征向量中找到对应指标
                if 'cpu' in indicator.lower():
                    pattern_features.append(features[0] if len(features) > 0 else 0)  # cpu_usage
                elif 'memory' in indicator.lower():
                    pattern_features.append(features[1] if len(features) > 1 else 0)  # memory_usage
                elif 'response' in indicator.lower():
                    pattern_features.append(features[2] if len(features) > 2 else 0)  # response_time
                elif 'error' in indicator.lower():
                    pattern_features.append(features[3] if len(features) > 3 else 0)  # error_rate
                else:
                    pattern_features.append(0)

            # 添加历史趋势特征
            if not recent_history.empty:
                for indicator in pattern.indicators[:2]:  # 只用前两个指标
                    base_metric = indicator.split('_')[0] + '_usage'
                    if base_metric in recent_history.columns:
                        trend = recent_history[base_metric].tail(7).values
                        if len(trend) >= 2:
                            slope = np.polyfit(range(len(trend)), trend, 1)[0]
                            pattern_features.append(slope)
                        else:
                            pattern_features.append(0)

            return np.array(pattern_features)

        except Exception as e:
            logger.error(f"模式特征提取失败: {e}")
            return None

    def _generate_maintenance_recommendations(self, failure_risks: Dict[str, Any],
                                            rul_predictions: Dict[str, Any],
                                            recognized_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成维护建议"""
        try:
            recommendations = []

            # 基于故障风险的建议
            immediate_risks = [risk for risk in failure_risks.values()
                             if risk.get('risk_level') in ['critical', 'high']]

            if immediate_risks:
                recommendations.append({
                    'priority': 'critical',
                    'type': 'immediate_maintenance',
                    'title': '紧急维护需求',
                    'description': f'检测到{len(immediate_risks)}个高风险故障点',
                    'actions': [
                        '立即安排维护窗口',
                        '准备备用系统',
                        '通知相关团队'
                    ],
                    'timeline': 'within_24_hours',
                    'estimated_cost': 'high'
                })

            # 基于RUL的建议
            critical_components = [
                comp for comp, pred in rul_predictions.items()
                if pred.get('urgency') == 'critical'
            ]

            if critical_components:
                recommendations.append({
                    'priority': 'high',
                    'type': 'component_replacement',
                    'title': f'关键组件更换 ({", ".join(critical_components)})',
                    'description': f'{len(critical_components)}个组件剩余寿命临近',
                    'actions': [
                        f'更换或维护: {", ".join(critical_components)}',
                        '安排维护时间窗口',
                        '准备备用组件'
                    ],
                    'timeline': 'within_1_week',
                    'estimated_cost': 'medium'
                })

            # 基于模式的建议
            for pattern in recognized_patterns[:3]:  # 最多3个模式建议
                recommendations.append({
                    'priority': 'medium' if pattern['severity'] != 'critical' else 'high',
                    'type': 'pattern_based_maintenance',
                    'title': f'{pattern["pattern_name"]}预防维护',
                    'description': pattern['description'],
                    'actions': [
                        f'监控指标: {", ".join(pattern["indicators"])}',
                        f'执行{pattern["pattern_name"]}相关检查',
                        '调整相关配置参数'
                    ],
                    'timeline': 'within_2_weeks',
                    'estimated_cost': 'low'
                })

            # 常规维护建议
            recommendations.append({
                'priority': 'low',
                'type': 'routine_maintenance',
                'title': '常规维护',
                'description': '执行常规系统维护和检查',
                'actions': [
                    '更新系统补丁',
                    '清理日志文件',
                    '优化配置参数',
                    '备份重要数据'
                ],
                'timeline': 'monthly',
                'estimated_cost': 'low'
            })

            # 按优先级排序
            priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 1), reverse=True)

            return recommendations

        except Exception as e:
            logger.error(f"维护建议生成失败: {e}")
            return []

    def _assess_overall_risk(self, failure_risks: Dict[str, Any],
                           rul_predictions: Dict[str, Any],
                           recognized_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估整体风险"""
        try:
            risk_scores = []

            # 故障风险评分
            for risk in failure_risks.values():
                level_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
                risk_scores.append(level_scores.get(risk.get('risk_level', 'low'), 0.2))

            # RUL风险评分
            for pred in rul_predictions.values():
                urgency_scores = {'low': 0.1, 'medium': 0.4, 'high': 0.7, 'critical': 1.0}
                risk_scores.append(urgency_scores.get(pred.get('urgency', 'low'), 0.1))

            # 模式风险评分
            for pattern in recognized_patterns:
                severity_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
                risk_scores.append(severity_scores.get(pattern.get('severity', 'low'), 0.2))

            # 计算综合风险
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                max_risk = np.max(risk_scores)

                overall_risk = (avg_risk * 0.7 + max_risk * 0.3)  # 加权平均

                if overall_risk > 0.8:
                    risk_level = 'critical'
                    description = '系统存在严重故障风险，需要立即采取行动'
                elif overall_risk > 0.6:
                    risk_level = 'high'
                    description = '系统存在较高故障风险，需要密切监控'
                elif overall_risk > 0.4:
                    risk_level = 'medium'
                    description = '系统存在中等故障风险，建议定期检查'
                else:
                    risk_level = 'low'
                    description = '系统故障风险较低，保持常规维护'
            else:
                overall_risk = 0.0
                risk_level = 'unknown'
                description = '无法评估系统风险'

            return {
                'overall_risk_score': float(overall_risk),
                'risk_level': risk_level,
                'description': description,
                'contributing_factors': {
                    'failure_risks': len([r for r in failure_risks.values() if r.get('risk_level') in ['high', 'critical']]),
                    'critical_components': len([p for p in rul_predictions.values() if p.get('urgency') == 'critical']),
                    'recognized_patterns': len(recognized_patterns)
                },
                'recommendations': self._generate_risk_mitigation_recommendations(risk_level)
            }

        except Exception as e:
            logger.error(f"整体风险评估失败: {e}")
            return {'overall_risk_score': 0.0, 'risk_level': 'unknown', 'description': '评估失败'}

    def _calculate_prediction_confidence(self, failure_risks: Dict[str, Any],
                                       rul_predictions: Dict[str, Any],
                                       recognized_patterns: List[Dict[str, Any]]) -> float:
        """计算预测置信度"""
        try:
            confidence_factors = []

            # 故障风险预测置信度
            if failure_risks:
                avg_failure_confidence = np.mean([r.get('confidence', 0) for r in failure_risks.values()])
                confidence_factors.append(avg_failure_confidence)

            # RUL预测置信度（固定值，实际应该基于模型性能）
            if rul_predictions:
                confidence_factors.append(0.75)

            # 模式识别置信度
            if recognized_patterns:
                avg_pattern_confidence = np.mean([p.get('confidence', 0) for p in recognized_patterns])
                confidence_factors.append(avg_pattern_confidence)

            if confidence_factors:
                overall_confidence = np.mean(confidence_factors)
                # 考虑数据质量和模型成熟度等因素
                adjusted_confidence = overall_confidence * 0.9  # 保守估计

                return min(0.95, max(0.1, adjusted_confidence))
            else:
                return 0.5

        except Exception:
            return 0.5

    def _generate_risk_mitigation_recommendations(self, risk_level: str) -> List[str]:
        """生成风险缓解建议"""
        recommendations = {
            'critical': [
                '立即激活应急响应计划',
                '准备系统降级运行方案',
                '增加监控频率至实时',
                '准备备用系统接管'
            ],
            'high': [
                '安排专门团队进行故障排查',
                '增加系统监控和日志记录',
                '准备维护时间窗口',
                '评估业务影响程度'
            ],
            'medium': [
                '定期检查系统关键指标',
                '优化系统配置参数',
                '加强日常维护工作',
                '建立故障早期预警机制'
            ],
            'low': [
                '保持常规监控和维护',
                '定期审查系统配置',
                '更新系统补丁和安全更新',
                '优化性能参数'
            ]
        }

        return recommendations.get(risk_level, ['保持常规监控'])

    def _evaluate_training_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """评估训练性能"""
        try:
            metrics = {}

            # 故障预测模型评估
            if self.failure_predictor and 'failure_occurred' in test_data.columns:
                features = test_data.drop(['failure_occurred', 'timestamp'], axis=1, errors='ignore').values
                labels = test_data['failure_occurred'].values

                scaled_features = self.scaler.transform(features)
                predictions = (self.failure_predictor.predict(scaled_features) > 0.5).astype(int)

                accuracy = accuracy_score(labels, predictions)

                metrics['failure_prediction'] = {
                    'accuracy': float(accuracy),
                    'precision': float(np.mean(predictions[labels == 1] == 1)) if np.sum(labels == 1) > 0 else 0,
                    'recall': float(np.mean(predictions == labels)) if len(predictions) > 0 else 0
                }

            # RUL预测模型评估
            if self.rul_predictor:
                # 这里简化评估，实际应该使用交叉验证
                metrics['rul_prediction'] = {
                    'feature_importance_top3': ['cpu_usage', 'memory_usage', 'response_time']  # 示例
                }

            # 风险评估模型评估
            if self.risk_assessor and 'risk_level' in test_data.columns:
                features = test_data.drop(['risk_level', 'timestamp'], axis=1, errors='ignore').values
                labels = test_data['risk_level'].map({'low': 0, 'medium': 1, 'high': 2, 'critical': 3}).values

                scaled_features = self.scaler.transform(features)
                predictions = self.risk_assessor.predict(scaled_features)

                accuracy = accuracy_score(labels, predictions)

                metrics['risk_assessment'] = {
                    'accuracy': float(accuracy)
                }

            return metrics

        except Exception as e:
            logger.error(f"训练性能评估失败: {e}")
            return {'error': str(e)}

    def _save_predictive_models(self):
        """保存预测模型"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)

            # 保存模型
            if self.failure_predictor:
                self.failure_predictor.save(f"{self.model_path}/failure_predictor.h5")

            if self.rul_predictor:
                joblib.dump(self.rul_predictor, f"{self.model_path}/rul_predictor.pkl")

            if self.risk_assessor:
                joblib.dump(self.risk_assessor, f"{self.model_path}/risk_assessor.pkl")

            if self.pattern_recognizer:
                joblib.dump(self.pattern_recognizer, f"{self.model_path}/pattern_recognizer.pkl")

            # 保存标准化器
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")

            # 保存故障模式
            failure_patterns_data = {
                pattern_id: {
                    'name': pattern.name,
                    'description': pattern.description,
                    'indicators': pattern.indicators,
                    'severity': pattern.severity,
                    'probability': pattern.probability
                }
                for pattern_id, pattern in self.failure_patterns.items()
            }

            with open(f"{self.model_path}/failure_patterns.json", 'w') as f:
                json.dump(failure_patterns_data, f, indent=2)

            # 保存配置
            config = {
                'is_trained': self.is_trained,
                'prediction_horizon_days': self.prediction_horizon_days,
                'risk_thresholds': self.risk_thresholds,
                'failure_patterns_count': len(self.failure_patterns)
            }

            with open(f"{self.model_path}/config.json", 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"预测模型保存失败: {e}")

    def load_predictive_models(self) -> bool:
        """加载预测模型"""
        try:
            # 加载配置
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)

            self.is_trained = config.get('is_trained', False)
            self.prediction_horizon_days = config.get('prediction_horizon_days', 30)
            self.risk_thresholds = config.get('risk_thresholds', self.risk_thresholds)

            # 加载故障模式
            with open(f"{self.model_path}/failure_patterns.json", 'r') as f:
                failure_patterns_data = json.load(f)

            for pattern_id, pattern_data in failure_patterns_data.items():
                self.failure_patterns[pattern_id] = FailurePattern(
                    pattern_id=pattern_id,
                    name=pattern_data['name'],
                    description=pattern_data['description'],
                    indicators=pattern_data['indicators'],
                    severity=pattern_data['severity'],
                    probability=pattern_data['probability']
                )

            # 加载模型
            self.failure_predictor = keras.models.load_model(f"{self.model_path}/failure_predictor.h5")
            self.rul_predictor = joblib.load(f"{self.model_path}/rul_predictor.pkl")
            self.risk_assessor = joblib.load(f"{self.model_path}/risk_assessor.pkl")
            self.pattern_recognizer = joblib.load(f"{self.model_path}/pattern_recognizer.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")

            return True

        except Exception as e:
            logger.error(f"预测模型加载失败: {e}")
            return False

    def get_maintenance_statistics(self) -> Dict[str, Any]:
        """获取维护统计信息"""
        try:
            stats = {
                'is_trained': self.is_trained,
                'failure_patterns_count': len(self.failure_patterns),
                'prediction_horizon_days': self.prediction_horizon_days
            }

            # 故障模式统计
            severity_distribution = {}
            for pattern in self.failure_patterns.values():
                severity_distribution[pattern.severity] = severity_distribution.get(pattern.severity, 0) + 1

            stats['failure_patterns_by_severity'] = severity_distribution

            # 风险阈值
            stats['risk_thresholds'] = self.risk_thresholds

            return stats

        except Exception as e:
            logger.error(f"获取维护统计失败: {e}")
            return {'error': str(e)}


class PredictiveMaintenanceService:
    """预测性维护服务"""

    def __init__(self):
        self.engine = MaintenancePredictionEngine()
        self.prediction_history = []
        self.maintenance_schedule = []

    def initialize_service(self, historical_failure_data_path: str = None,
                          system_metrics_history_path: str = None) -> bool:
        """初始化服务"""
        try:
            # 尝试加载已训练的模型
            if self.engine.load_predictive_models():
                logger.info("成功加载已训练的预测性维护模型")
                return True

            # 如果没有模型，尝试从历史数据训练
            if historical_failure_data_path and system_metrics_history_path:
                historical_failures = pd.read_csv(historical_failure_data_path)
                system_metrics = pd.read_csv(system_metrics_history_path)

                result = self.engine.train_predictive_models(historical_failures, system_metrics)

                if result.get('success', False):
                    logger.info("成功训练新的预测性维护模型")
                    return True

            logger.warning("无法初始化预测性维护服务")
            return False

        except Exception as e:
            logger.error(f"预测性维护服务初始化失败: {e}")
            return False

    def predict_and_schedule_maintenance(self, current_system_metrics: Dict[str, Any],
                                       recent_history: pd.DataFrame) -> Dict[str, Any]:
        """预测并安排维护"""
        try:
            # 进行维护预测
            prediction_result = self.engine.predict_maintenance_needs(
                current_system_metrics, recent_history
            )

            if 'error' in prediction_result:
                return prediction_result

            # 生成维护计划
            maintenance_plan = self._create_maintenance_plan(prediction_result)

            # 更新维护日程
            self.maintenance_schedule.extend(maintenance_plan.get('scheduled_maintenance', []))

            # 清理过期维护任务
            self._cleanup_expired_maintenance()

            result = {
                'prediction': prediction_result,
                'maintenance_plan': maintenance_plan,
                'alerts': self._generate_maintenance_alerts(prediction_result),
                'service_status': 'active'
            }

            # 记录预测历史
            self.prediction_history.append(result)

            # 保持历史记录大小
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

            return result

        except Exception as e:
            logger.error(f"维护预测和安排失败: {e}")
            return {
                'error': str(e),
                'service_status': 'error'
            }

    def get_maintenance_schedule(self) -> List[Dict[str, Any]]:
        """获取维护日程"""
        try:
            # 按优先级和时间排序
            sorted_schedule = sorted(
                self.maintenance_schedule,
                key=lambda x: (self._priority_score(x['priority']), x['scheduled_date'])
            )

            return sorted_schedule

        except Exception as e:
            logger.error(f"获取维护日程失败: {e}")
            return []

    def _create_maintenance_plan(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """创建维护计划"""
        try:
            recommendations = prediction_result.get('maintenance_recommendations', [])

            scheduled_maintenance = []
            resource_requirements = {
                'personnel': set(),
                'tools': set(),
                'downtime_windows': []
            }

            for rec in recommendations:
                # 计算维护时间
                scheduled_date = self._calculate_maintenance_date(rec)

                maintenance_task = {
                    'task_id': f"maintenance_{int(scheduled_date.timestamp())}",
                    'title': rec['title'],
                    'description': rec['description'],
                    'priority': rec['priority'],
                    'scheduled_date': scheduled_date,
                    'estimated_duration_hours': self._estimate_maintenance_duration(rec),
                    'required_resources': self._identify_required_resources(rec),
                    'success_criteria': rec.get('actions', []),
                    'rollback_plan': self._create_task_rollback_plan(rec),
                    'status': 'scheduled'
                }

                scheduled_maintenance.append(maintenance_task)

                # 累积资源需求
                for resource in maintenance_task['required_resources']:
                    if 'engineer' in resource.lower():
                        resource_requirements['personnel'].add(resource)
                    elif 'tool' in resource.lower() or 'software' in resource.lower():
                        resource_requirements['tools'].add(resource)

                # 添加维护窗口
                resource_requirements['downtime_windows'].append({
                    'start': scheduled_date,
                    'duration_hours': maintenance_task['estimated_duration_hours'],
                    'impact_level': 'medium' if rec['priority'] != 'critical' else 'high'
                })

            return {
                'scheduled_maintenance': scheduled_maintenance,
                'resource_requirements': {
                    'personnel': list(resource_requirements['personnel']),
                    'tools': list(resource_requirements['tools']),
                    'downtime_windows': resource_requirements['downtime_windows']
                },
                'total_maintenance_tasks': len(scheduled_maintenance),
                'estimated_total_downtime_hours': sum(
                    task['estimated_duration_hours'] for task in scheduled_maintenance
                ),
                'plan_generated_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"维护计划创建失败: {e}")
            return {'error': str(e)}

    def _calculate_maintenance_date(self, recommendation: Dict[str, Any]) -> datetime:
        """计算维护日期"""
        try:
            priority = recommendation.get('priority', 'low')
            timeline = recommendation.get('timeline', 'monthly')

            base_date = datetime.now()

            # 根据优先级和时间线计算维护日期
            if priority == 'critical':
                days_ahead = 1  # 紧急维护：1天内
            elif priority == 'high':
                days_ahead = 3  # 重要维护：3天内
            elif priority == 'medium':
                if 'week' in timeline:
                    days_ahead = 7
                else:
                    days_ahead = 14
            else:  # low
                if 'monthly' in timeline:
                    days_ahead = 30
                else:
                    days_ahead = 90

            maintenance_date = base_date + timedelta(days=days_ahead)

            # 调整到工作时间（工作日9:00-17:00）
            if maintenance_date.weekday() >= 5:  # 周末
                maintenance_date += timedelta(days=(7 - maintenance_date.weekday()))

            if maintenance_date.hour < 9:
                maintenance_date = maintenance_date.replace(hour=9, minute=0, second=0, microsecond=0)
            elif maintenance_date.hour >= 17:
                maintenance_date += timedelta(days=1)
                maintenance_date = maintenance_date.replace(hour=9, minute=0, second=0, microsecond=0)

            return maintenance_date

        except Exception:
            # 默认返回明天早上9点
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)

    def _estimate_maintenance_duration(self, recommendation: Dict[str, Any]) -> float:
        """估算维护持续时间"""
        try:
            priority = recommendation.get('priority', 'low')
            estimated_cost = recommendation.get('estimated_cost', 'medium')

            # 基于优先级和成本估算时间
            duration_map = {
                'critical': {'low': 2, 'medium': 4, 'high': 8},
                'high': {'low': 1, 'medium': 2, 'high': 4},
                'medium': {'low': 0.5, 'medium': 1, 'high': 2},
                'low': {'low': 0.25, 'medium': 0.5, 'high': 1}
            }

            return duration_map.get(priority, {}).get(estimated_cost, 1.0)

        except Exception:
            return 1.0  # 默认1小时

    def _identify_required_resources(self, recommendation: Dict[str, Any]) -> List[str]:
        """识别所需资源"""
        try:
            priority = recommendation.get('priority', 'low')
            rec_type = recommendation.get('type', 'general')

            base_resources = []

            # 根据维护类型确定资源
            if 'component' in rec_type or 'hardware' in rec_type:
                base_resources.extend(['硬件工程师', '测试设备', '备用组件'])
            elif 'software' in rec_type or 'code' in rec_type:
                base_resources.extend(['软件工程师', '开发环境', '版本控制工具'])
            elif 'database' in rec_type:
                base_resources.extend(['数据库管理员', '数据库工具', '备份系统'])
            else:
                base_resources.extend(['系统工程师', '监控工具'])

            # 根据优先级调整资源
            if priority in ['critical', 'high']:
                base_resources.append('项目经理')
                base_resources.append('质量保证工程师')

            return base_resources

        except Exception:
            return ['系统工程师', '基本工具']

    def _create_task_rollback_plan(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """创建任务回滚计划"""
        try:
            return {
                'rollback_steps': [
                    '停止维护操作',
                    '恢复系统备份',
                    '验证系统状态',
                    '重新启动服务',
                    '执行回归测试'
                ],
                'rollback_time_estimate': '30分钟到2小时',
                'success_criteria': [
                    '系统恢复到维护前状态',
                    '所有服务正常运行',
                    '数据完整性验证通过'
                ],
                'contact_personnel': ['维护负责人', '系统管理员', '技术支持']
            }

        except Exception:
            return {'rollback_steps': ['恢复系统备份', '重新启动服务']}

    def _generate_maintenance_alerts(self, prediction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成维护告警"""
        try:
            alerts = []
            recommendations = prediction_result.get('maintenance_recommendations', [])

            # 检查紧急维护需求
            critical_maintenance = [r for r in recommendations if r.get('priority') == 'critical']

            if critical_maintenance:
                alerts.append({
                    'alert_type': 'critical_maintenance_required',
                    'severity': 'critical',
                    'message': f'检测到{len(critical_maintenance)}个紧急维护需求',
                    'details': [r['title'] for r in critical_maintenance],
                    'recommended_action': '立即安排维护团队处理',
                    'timestamp': datetime.now()
                })

            # 检查即将到期的维护任务
            upcoming_maintenance = [
                task for task in self.maintenance_schedule
                if (task['scheduled_date'] - datetime.now()).days <= 1
            ]

            if upcoming_maintenance:
                alerts.append({
                    'alert_type': 'upcoming_maintenance_due',
                    'severity': 'medium',
                    'message': f'有{len(upcoming_maintenance)}个维护任务即将到期',
                    'details': [task['title'] for task in upcoming_maintenance],
                    'recommended_action': '确认维护资源和时间安排',
                    'timestamp': datetime.now()
                })

            return alerts

        except Exception as e:
            logger.error(f"维护告警生成失败: {e}")
            return []

    def _cleanup_expired_maintenance(self):
        """清理过期维护任务"""
        try:
            current_time = datetime.now()
            self.maintenance_schedule = [
                task for task in self.maintenance_schedule
                if task['scheduled_date'] > current_time - timedelta(hours=24)  # 保留24小时内的过期任务
            ]

        except Exception as e:
            logger.error(f"过期维护清理失败: {e}")

    def _priority_score(self, priority: str) -> int:
        """优先级评分"""
        priority_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return priority_map.get(priority, 1)

    def get_service_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            stats = {
                'total_predictions': len(self.prediction_history),
                'scheduled_maintenance_tasks': len(self.maintenance_schedule),
                'service_status': 'active' if self.engine.is_trained else 'not_initialized'
            }

            if self.prediction_history:
                recent_predictions = self.prediction_history[-10:]  # 最近10次预测

                # 计算平均风险分数
                avg_risk_scores = [
                    p.get('prediction', {}).get('overall_risk_assessment', {}).get('overall_risk_score', 0)
                    for p in recent_predictions
                ]
                stats['avg_risk_score_recent'] = float(np.mean(avg_risk_scores)) if avg_risk_scores else 0

                # 计算维护任务完成率
                maintenance_tasks = sum(len(p.get('maintenance_plan', {}).get('scheduled_maintenance', []))
                                      for p in recent_predictions)
                stats['avg_maintenance_tasks_per_prediction'] = maintenance_tasks / len(recent_predictions)

            return stats

        except Exception as e:
            logger.error(f"获取服务统计失败: {e}")
            return {'error': str(e)}

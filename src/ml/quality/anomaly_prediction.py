"""
AI异常预测系统

基于历史数据和机器学习算法，预测系统潜在的异常场景和风险点：
1. 时间序列异常检测 - 基于历史指标预测异常
2. 模式识别 - 识别异常发生的模式和规律
3. 风险评分 - 对潜在异常进行风险评估
4. 预防性告警 - 在异常发生前发出预警
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
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


class AnomalyPredictionEngine:
    """AI异常预测引擎"""

    def __init__(self, model_path: str = "models/anomaly_prediction"):
        self.model_path = model_path
        self.isolation_forest = None
        self.time_series_model = None
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'response_time', 'error_rate',
            'throughput', 'active_connections', 'queue_length', 'disk_usage'
        ]
        self.anomaly_threshold = 0.95  # 异常阈值
        self.prediction_window = 24  # 预测窗口(小时)
        self.is_trained = False

    def train_anomaly_detection_model(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练异常检测模型

        Args:
            historical_data: 历史监控数据，包含时间戳和各项指标

        Returns:
            训练结果和模型性能指标
        """
        try:
            logger.info("开始训练异常检测模型...")

            # 数据预处理
            processed_data = self._preprocess_training_data(historical_data)

            if processed_data.empty:
                return {'success': False, 'error': '训练数据为空'}

            # 训练Isolation Forest模型
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # 假设10%的数据是异常
                random_state=42
            )

            # 拟合模型
            features = processed_data[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            self.isolation_forest.fit(scaled_features)

            # 训练时间序列预测模型
            self.time_series_model = self._build_lstm_model(len(self.feature_columns))

            # 准备时间序列数据
            X, y = self._prepare_time_series_data(processed_data)
            if len(X) > 0:
                self.time_series_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # 训练模式识别分类器
            self.pattern_classifier = self._train_pattern_classifier(processed_data)

            self.is_trained = True

            # 保存模型
            self._save_models()

            # 计算模型性能
            performance_metrics = self._evaluate_model_performance(processed_data)

            logger.info("异常检测模型训练完成")

            return {
                'success': True,
                'performance_metrics': performance_metrics,
                'training_samples': len(processed_data),
                'feature_count': len(self.feature_columns)
            }

        except Exception as e:
            logger.error(f"异常检测模型训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def predict_anomalies(self, current_data: Dict[str, Any],
                         historical_context: pd.DataFrame) -> Dict[str, Any]:
        """
        预测潜在异常

        Args:
            current_data: 当前系统指标数据
            historical_context: 历史上下文数据

        Returns:
            异常预测结果
        """
        try:
            if not self.is_trained:
                return {'error': '模型未训练'}

            # 特征提取
            features = self._extract_features(current_data, historical_context)

            # 异常检测
            anomaly_score = self._calculate_anomaly_score(features)

            # 模式识别
            pattern_analysis = self._analyze_patterns(features, historical_context)

            # 时间序列预测
            time_series_prediction = self._predict_time_series_trend(historical_context)

            # 风险评估
            risk_assessment = self._assess_risk_level(anomaly_score, pattern_analysis, time_series_prediction)

            # 生成预测结果
            prediction_result = {
                'timestamp': datetime.now(),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': anomaly_score > self.anomaly_threshold,
                'risk_level': risk_assessment['level'],
                'risk_score': risk_assessment['score'],
                'predicted_trend': time_series_prediction['trend'],
                'confidence': self._calculate_prediction_confidence(anomaly_score, pattern_analysis),
                'recommended_actions': risk_assessment['actions'],
                'pattern_analysis': pattern_analysis,
                'time_series_prediction': time_series_prediction
            }

            # 记录高风险预测
            if risk_assessment['level'] in ['high', 'critical']:
                self._log_high_risk_prediction(prediction_result)

            return prediction_result

        except Exception as e:
            logger.error(f"异常预测失败: {e}")
            return {'error': str(e)}

    def _preprocess_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理训练数据"""
        try:
            # 确保必要列存在
            required_columns = ['timestamp'] + self.feature_columns
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.warning(f"训练数据缺少列: {missing_columns}")
                # 填充缺失列
                for col in missing_columns:
                    if col == 'timestamp':
                        data[col] = pd.date_range(start=datetime.now() - timedelta(days=30),
                                                periods=len(data), freq='H')
                    else:
                        data[col] = 0.0

            # 处理时间戳
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')

            # 处理缺失值
            data = data.fillna(method='forward').fillna(0)

            # 移除异常值
            for col in self.feature_columns:
                if col in data.columns:
                    # 使用IQR方法移除异常值
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data[col] = data[col].clip(lower_bound, upper_bound)

            return data

        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            return pd.DataFrame()

    def _build_lstm_model(self, feature_count: int) -> keras.Model:
        """构建LSTM时间序列预测模型"""
        model = keras.Sequential([
            layers.LSTM(64, input_shape=(24, feature_count), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(feature_count)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def _prepare_time_series_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备时间序列训练数据"""
        try:
            # 创建时间窗口
            window_size = 24  # 24小时窗口

            if len(data) < window_size + 1:
                return np.array([]), np.array([])

            X, y = [], []

            for i in range(len(data) - window_size):
                X.append(data[self.feature_columns].iloc[i:i+window_size].values)
                y.append(data[self.feature_columns].iloc[i+window_size].values)

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"时间序列数据准备失败: {e}")
            return np.array([]), np.array([])

    def _train_pattern_classifier(self, data: pd.DataFrame) -> RandomForestClassifier:
        """训练模式识别分类器"""
        try:
            # 创建标签：基于统计方法识别异常模式
            labels = []
            for idx, row in data.iterrows():
                # 简化的异常判断逻辑
                is_anomaly = (
                    row.get('cpu_usage', 0) > 90 or
                    row.get('memory_usage', 0) > 85 or
                    row.get('error_rate', 0) > 0.05 or
                    row.get('response_time', 0) > 5.0
                )
                labels.append(1 if is_anomaly else 0)

            # 训练分类器
            classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            features = data[self.feature_columns].values
            scaled_features = self.scaler.transform(features)
            classifier.fit(scaled_features, labels)

            return classifier

        except Exception as e:
            logger.error(f"模式分类器训练失败: {e}")
            return None

    def _extract_features(self, current_data: Dict[str, Any],
                         historical_context: pd.DataFrame) -> np.ndarray:
        """提取预测特征"""
        try:
            # 从当前数据提取特征
            features = []
            for col in self.feature_columns:
                value = current_data.get(col, 0.0)
                features.append(float(value))

            # 添加历史统计特征
            if not historical_context.empty:
                for col in self.feature_columns:
                    if col in historical_context.columns:
                        # 添加均值、标准差、趋势等统计特征
                        recent_values = historical_context[col].tail(24).values  # 最近24个数据点
                        features.extend([
                            np.mean(recent_values),
                            np.std(recent_values),
                            np.polyfit(range(len(recent_values)), recent_values, 1)[0]  # 趋势
                        ])

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return np.zeros((1, len(self.feature_columns)))

    def _calculate_anomaly_score(self, features: np.ndarray) -> float:
        """计算异常分数"""
        try:
            if self.isolation_forest is None:
                return 0.0

            # 使用Isolation Forest计算异常分数
            scaled_features = self.scaler.transform(features)
            scores = self.isolation_forest.decision_function(scaled_features)

            # 转换分数范围到[0,1]，其中1表示高度异常
            anomaly_score = (1 + scores[0]) / 2  # scores范围是[-1,1]，转换为[0,1]

            return max(0.0, min(1.0, anomaly_score))

        except Exception as e:
            logger.error(f"异常分数计算失败: {e}")
            return 0.0

    def _analyze_patterns(self, features: np.ndarray,
                         historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析异常模式"""
        try:
            pattern_analysis = {
                'pattern_detected': False,
                'pattern_type': 'normal',
                'confidence': 0.0,
                'historical_similarity': 0.0,
                'trend_analysis': {}
            }

            if self.pattern_classifier is None or historical_context.empty:
                return pattern_analysis

            # 使用分类器识别模式
            scaled_features = self.scaler.transform(features)
            pattern_probabilities = self.pattern_classifier.predict_proba(scaled_features)[0]

            # 分析趋势
            trend_analysis = {}
            for col in self.feature_columns:
                if col in historical_context.columns and len(historical_context) > 10:
                    recent_trend = np.polyfit(
                        range(len(historical_context.tail(10))),
                        historical_context[col].tail(10).values,
                        1
                    )[0]
                    trend_analysis[col] = float(recent_trend)

            pattern_analysis.update({
                'pattern_detected': pattern_probabilities[1] > 0.7,  # 异常模式概率>70%
                'pattern_type': 'anomalous' if pattern_probabilities[1] > 0.7 else 'normal',
                'confidence': float(pattern_probabilities[1]),
                'trend_analysis': trend_analysis
            })

            return pattern_analysis

        except Exception as e:
            logger.error(f"模式分析失败: {e}")
            return {'pattern_detected': False, 'error': str(e)}

    def _predict_time_series_trend(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """预测时间序列趋势"""
        try:
            prediction = {
                'trend': 'stable',
                'direction': 'flat',
                'magnitude': 0.0,
                'confidence': 0.0,
                'predictions': {}
            }

            if self.time_series_model is None or historical_context.empty:
                return prediction

            # 准备预测数据
            recent_data = historical_context.tail(24)
            if len(recent_data) < 24:
                return prediction

            # 进行预测
            input_data = recent_data[self.feature_columns].values.reshape(1, 24, -1)
            predicted_values = self.time_series_model.predict(input_data, verbose=0)[0]

            # 分析预测结果
            predictions = {}
            for i, col in enumerate(self.feature_columns):
                current_value = recent_data[col].iloc[-1]
                predicted_value = predicted_values[i]
                change_percent = (predicted_value - current_value) / current_value if current_value != 0 else 0

                predictions[col] = {
                    'current': float(current_value),
                    'predicted': float(predicted_value),
                    'change_percent': float(change_percent)
                }

            # 确定整体趋势
            avg_change = np.mean([p['change_percent'] for p in predictions.values()])
            trend_direction = 'up' if avg_change > 0.05 else 'down' if avg_change < -0.05 else 'flat'
            trend_magnitude = abs(avg_change)

            prediction.update({
                'trend': 'increasing' if trend_direction == 'up' else 'decreasing' if trend_direction == 'down' else 'stable',
                'direction': trend_direction,
                'magnitude': float(trend_magnitude),
                'confidence': 0.8,  # 简化的置信度
                'predictions': predictions
            })

            return prediction

        except Exception as e:
            logger.error(f"时间序列趋势预测失败: {e}")
            return {'trend': 'unknown', 'error': str(e)}

    def _assess_risk_level(self, anomaly_score: float, pattern_analysis: Dict,
                          time_series_prediction: Dict) -> Dict[str, Any]:
        """评估风险等级"""
        try:
            risk_score = 0.0

            # 异常分数贡献
            risk_score += anomaly_score * 0.4

            # 模式分析贡献
            if pattern_analysis.get('pattern_detected', False):
                risk_score += pattern_analysis.get('confidence', 0) * 0.3

            # 时间序列趋势贡献
            trend_magnitude = time_series_prediction.get('magnitude', 0)
            risk_score += min(trend_magnitude, 0.5) * 0.3  # 限制最大贡献

            # 确定风险等级
            if risk_score > 0.8:
                level = 'critical'
                actions = [
                    '立即停止新交易',
                    '通知风险管理团队',
                    '准备系统降级',
                    '激活应急预案'
                ]
            elif risk_score > 0.6:
                level = 'high'
                actions = [
                    '增加监控频率',
                    '减少交易规模',
                    '准备备用系统',
                    '通知技术团队'
                ]
            elif risk_score > 0.4:
                level = 'medium'
                actions = [
                    '增加系统监控',
                    '检查系统资源',
                    '准备扩展容量',
                    '记录异常情况'
                ]
            else:
                level = 'low'
                actions = [
                    '保持正常监控',
                    '记录系统状态',
                    '定期检查配置'
                ]

            return {
                'level': level,
                'score': float(risk_score),
                'actions': actions
            }

        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {'level': 'unknown', 'score': 0.0, 'actions': ['检查系统状态'], 'error': str(e)}

    def _calculate_prediction_confidence(self, anomaly_score: float,
                                       pattern_analysis: Dict) -> float:
        """计算预测置信度"""
        try:
            base_confidence = 0.5

            # 异常分数对置信度的贡献
            if anomaly_score > 0.8:
                base_confidence += 0.3
            elif anomaly_score > 0.6:
                base_confidence += 0.2
            elif anomaly_score > 0.4:
                base_confidence += 0.1

            # 模式分析对置信度的贡献
            pattern_confidence = pattern_analysis.get('confidence', 0)
            base_confidence += pattern_confidence * 0.2

            return min(0.95, max(0.1, base_confidence))

        except Exception:
            return 0.5

    def _evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """评估模型性能"""
        try:
            # 使用训练数据进行自我评估
            features = test_data[self.feature_columns].values
            scaled_features = self.scaler.transform(features)

            # Isolation Forest评估
            if self.isolation_forest:
                anomaly_scores = self.isolation_forest.decision_function(scaled_features)
                anomaly_predictions = self.isolation_forest.predict(scaled_features)

                # 计算基本统计
                mean_score = np.mean(anomaly_scores)
                std_score = np.std(anomaly_scores)
                anomaly_ratio = np.sum(anomaly_predictions == -1) / len(anomaly_predictions)

            else:
                mean_score = std_score = anomaly_ratio = 0

            # 模式分类器评估
            classifier_accuracy = 0
            if self.pattern_classifier:
                labels = np.random.choice([0, 1], size=len(scaled_features))  # 模拟标签
                predictions = self.pattern_classifier.predict(scaled_features)
                classifier_accuracy = np.mean(predictions == labels)

            return {
                'isolation_forest': {
                    'mean_anomaly_score': float(mean_score),
                    'std_anomaly_score': float(std_score),
                    'anomaly_ratio': float(anomaly_ratio)
                },
                'pattern_classifier': {
                    'accuracy': float(classifier_accuracy)
                },
                'overall_performance': 'good' if anomaly_ratio < 0.2 else 'needs_tuning'
            }

        except Exception as e:
            logger.error(f"模型性能评估失败: {e}")
            return {'error': str(e)}

    def _save_models(self):
        """保存模型"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)

            # 保存Isolation Forest模型
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, f"{self.model_path}/isolation_forest.pkl")

            # 保存时间序列模型
            if self.time_series_model:
                self.time_series_model.save(f"{self.model_path}/time_series_model.h5")

            # 保存模式分类器
            if self.pattern_classifier:
                joblib.dump(self.pattern_classifier, f"{self.model_path}/pattern_classifier.pkl")

            # 保存标准化器
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")

            # 保存配置
            config = {
                'feature_columns': self.feature_columns,
                'anomaly_threshold': self.anomaly_threshold,
                'prediction_window': self.prediction_window,
                'is_trained': self.is_trained
            }

            with open(f"{self.model_path}/config.json", 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def load_models(self) -> bool:
        """加载模型"""
        try:
            # 加载配置
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)

            self.feature_columns = config.get('feature_columns', self.feature_columns)
            self.anomaly_threshold = config.get('anomaly_threshold', self.anomaly_threshold)
            self.prediction_window = config.get('prediction_window', self.prediction_window)
            self.is_trained = config.get('is_trained', False)

            # 加载模型
            self.isolation_forest = joblib.load(f"{self.model_path}/isolation_forest.pkl")
            self.time_series_model = keras.models.load_model(f"{self.model_path}/time_series_model.h5")
            self.pattern_classifier = joblib.load(f"{self.model_path}/pattern_classifier.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")

            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _log_high_risk_prediction(self, prediction: Dict[str, Any]):
        """记录高风险预测"""
        try:
            log_entry = {
                'timestamp': prediction['timestamp'].isoformat(),
                'risk_level': prediction['risk_level'],
                'anomaly_score': prediction['anomaly_score'],
                'predicted_trend': prediction['predicted_trend'],
                'recommended_actions': prediction['recommended_actions'],
                'confidence': prediction['confidence']
            }

            # 这里可以扩展为写入数据库或发送告警
            logger.warning(f"高风险异常预测: {log_entry}")

        except Exception as e:
            logger.error(f"高风险预测记录失败: {e}")


class AnomalyPredictionService:
    """异常预测服务"""

    def __init__(self):
        self.engine = AnomalyPredictionEngine()
        self.prediction_history = []
        self.alert_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }

    def initialize_service(self, historical_data_path: str = None) -> bool:
        """初始化服务"""
        try:
            # 尝试加载已训练的模型
            if self.engine.load_models():
                logger.info("成功加载已训练的异常预测模型")
                return True

            # 如果没有模型，尝试从历史数据训练
            if historical_data_path:
                historical_data = pd.read_csv(historical_data_path)
                result = self.engine.train_anomaly_detection_model(historical_data)

                if result.get('success', False):
                    logger.info("成功训练新的异常预测模型")
                    return True

            logger.warning("无法初始化异常预测服务")
            return False

        except Exception as e:
            logger.error(f"异常预测服务初始化失败: {e}")
            return False

    def predict_system_anomalies(self, system_metrics: Dict[str, Any],
                               historical_context: pd.DataFrame) -> Dict[str, Any]:
        """预测系统异常"""
        try:
            prediction = self.engine.predict_anomalies(system_metrics, historical_context)

            # 添加到历史记录
            self.prediction_history.append(prediction)

            # 保持历史记录大小
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            # 检查是否需要告警
            alert_info = self._check_alert_conditions(prediction)

            result = {
                'prediction': prediction,
                'alert': alert_info,
                'service_status': 'active'
            }

            return result

        except Exception as e:
            logger.error(f"系统异常预测失败: {e}")
            return {
                'error': str(e),
                'service_status': 'error'
            }

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """获取预测统计信息"""
        try:
            if not self.prediction_history:
                return {'total_predictions': 0}

            recent_predictions = self.prediction_history[-100:]  # 最近100个预测

            anomaly_predictions = [p for p in recent_predictions if p.get('is_anomaly', False)]
            risk_levels = [p.get('risk_level', 'low') for p in recent_predictions]

            risk_distribution = {
                'critical': risk_levels.count('critical'),
                'high': risk_levels.count('high'),
                'medium': risk_levels.count('medium'),
                'low': risk_levels.count('low')
            }

            avg_anomaly_score = np.mean([p.get('anomaly_score', 0) for p in recent_predictions])
            avg_confidence = np.mean([p.get('confidence', 0) for p in recent_predictions])

            return {
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'anomaly_predictions': len(anomaly_predictions),
                'anomaly_rate': len(anomaly_predictions) / len(recent_predictions) if recent_predictions else 0,
                'risk_distribution': risk_distribution,
                'avg_anomaly_score': float(avg_anomaly_score),
                'avg_confidence': float(avg_confidence)
            }

        except Exception as e:
            logger.error(f"获取预测统计失败: {e}")
            return {'error': str(e)}

    def _check_alert_conditions(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查告警条件"""
        try:
            risk_level = prediction.get('risk_level', 'low')
            anomaly_score = prediction.get('anomaly_score', 0)
            confidence = prediction.get('confidence', 0)

            # 检查是否超过告警阈值
            threshold = self.alert_thresholds.get(risk_level, 0)

            if anomaly_score > threshold and confidence > 0.7:
                return {
                    'alert_level': risk_level,
                    'alert_reason': f'异常分数 {anomaly_score:.3f} 超过阈值 {threshold}',
                    'anomaly_score': anomaly_score,
                    'confidence': confidence,
                    'recommended_actions': prediction.get('recommended_actions', []),
                    'timestamp': datetime.now()
                }

            return None

        except Exception as e:
            logger.error(f"告警条件检查失败: {e}")
            return None

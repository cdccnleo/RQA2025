"""
质量趋势分析系统

基于机器学习算法分析质量指标趋势，预测质量风险并提供改进建议：
1. 趋势预测 - 预测质量指标的未来走势
2. 风险识别 - 识别质量下降的风险点
3. 模式分析 - 分析质量问题的发生模式
4. 改进建议 - 基于趋势分析提供持续改进建议
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
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


class QualityTrendAnalyzer:
    """质量趋势分析器"""

    def __init__(self, model_path: str = "models/quality_trend"):
        self.model_path = model_path
        self.trend_model = None
        self.risk_predictor = None
        self.pattern_analyzer = None
        self.quality_predictor = None
        self.scaler = StandardScaler()
        self.quality_metrics = [
            'test_coverage', 'test_success_rate', 'code_quality_score',
            'performance_score', 'error_rate', 'response_time',
            'deployment_frequency', 'lead_time', 'change_failure_rate',
            'mean_time_to_recovery', 'availability'
        ]
        self.risk_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        self.is_trained = False

    def train_quality_trend_models(self, historical_quality_data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练质量趋势分析模型

        Args:
            historical_quality_data: 历史质量数据，包含时间戳和各项质量指标

        Returns:
            训练结果和模型性能指标
        """
        try:
            logger.info("开始训练质量趋势分析模型...")

            # 数据预处理
            processed_data = self._preprocess_quality_data(historical_quality_data)

            if processed_data.empty:
                return {'success': False, 'error': '训练数据为空'}

            # 训练趋势预测模型
            self.trend_model = self._build_trend_prediction_model(len(self.quality_metrics))

            # 准备时间序列数据
            X, y = self._prepare_trend_training_data(processed_data)
            if len(X) > 0:
                self.trend_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # 训练风险预测器
            self.risk_predictor = self._train_risk_predictor(processed_data)

            # 训练模式分析器
            self.pattern_analyzer = self._train_pattern_analyzer(processed_data)

            # 训练质量预测器
            self.quality_predictor = self._train_quality_predictor(processed_data)

            self.is_trained = True

            # 保存模型
            self._save_trend_models()

            # 计算模型性能
            performance_metrics = self._evaluate_trend_models(processed_data)

            logger.info("质量趋势分析模型训练完成")

            return {
                'success': True,
                'performance_metrics': performance_metrics,
                'training_samples': len(processed_data),
                'metrics_count': len(self.quality_metrics)
            }

        except Exception as e:
            logger.error(f"质量趋势分析模型训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_quality_trends(self, current_quality_metrics: Dict[str, Any],
                             historical_context: pd.DataFrame) -> Dict[str, Any]:
        """
        分析质量趋势

        Args:
            current_quality_metrics: 当前质量指标
            historical_context: 历史质量上下文

        Returns:
            质量趋势分析结果
        """
        try:
            if not self.is_trained:
                return {'error': '模型未训练'}

            # 趋势预测
            trend_prediction = self._predict_quality_trends(historical_context)

            # 风险评估
            risk_assessment = self._assess_quality_risks(current_quality_metrics, historical_context)

            # 模式分析
            pattern_analysis = self._analyze_quality_patterns(historical_context)

            # 改进建议
            improvement_suggestions = self._generate_improvement_suggestions(
                trend_prediction, risk_assessment, pattern_analysis
            )

            # 生成分析结果
            analysis_result = {
                'timestamp': datetime.now(),
                'current_metrics': current_quality_metrics,
                'trend_prediction': trend_prediction,
                'risk_assessment': risk_assessment,
                'pattern_analysis': pattern_analysis,
                'improvement_suggestions': improvement_suggestions,
                'overall_quality_score': self._calculate_overall_quality_score(
                    current_quality_metrics, trend_prediction
                ),
                'quality_trend_direction': self._determine_trend_direction(trend_prediction)
            }

            return analysis_result

        except Exception as e:
            logger.error(f"质量趋势分析失败: {e}")
            return {'error': str(e)}

    def predict_quality_degradation_risks(self, analysis_result: Dict[str, Any],
                                        prediction_horizon: int = 30) -> List[Dict[str, Any]]:
        """
        预测质量下降风险

        Args:
            analysis_result: 质量分析结果
            prediction_horizon: 预测时间范围(天)

        Returns:
            质量风险预测列表
        """
        try:
            risks = []

            trend_prediction = analysis_result.get('trend_prediction', {})
            risk_assessment = analysis_result.get('risk_assessment', {})

            # 基于趋势预测的质量风险
            for metric, prediction in trend_prediction.get('predictions', {}).items():
                if prediction.get('trend', 'stable') == 'decreasing':
                    decline_rate = abs(prediction.get('slope', 0))

                    if decline_rate > 0.01:  # 日下降率>1%
                        risk_level = 'high' if decline_rate > 0.05 else 'medium'

                        risks.append({
                            'risk_type': 'quality_degradation',
                            'metric': metric,
                            'risk_level': risk_level,
                            'predicted_decline': decline_rate * prediction_horizon,
                            'time_horizon_days': prediction_horizon,
                            'description': f"{metric}预计在{prediction_horizon}天内下降{decline_rate * prediction_horizon:.1%}",
                            'mitigation_actions': self._get_metric_specific_actions(metric)
                        })

            # 基于当前风险水平的质量风险
            for risk_area, risk_score in risk_assessment.items():
                if risk_score > self.risk_thresholds['medium']:
                    risks.append({
                        'risk_type': 'current_risk_area',
                        'metric': risk_area,
                        'risk_level': self._score_to_risk_level(risk_score),
                        'current_risk_score': risk_score,
                        'description': f"{risk_area}当前风险分数为{risk_score:.2f}",
                        'mitigation_actions': [f"关注{risk_area}指标的改善"]
                    })

            # 按风险等级排序
            risks.sort(key=lambda x: self._risk_level_priority(x['risk_level']), reverse=True)

            return risks

        except Exception as e:
            logger.error(f"质量下降风险预测失败: {e}")
            return []

    def _preprocess_quality_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理质量数据"""
        try:
            # 确保必要列存在
            required_columns = ['timestamp'] + self.quality_metrics
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.warning(f"质量数据缺少列: {missing_columns}")
                # 填充缺失列
                for col in missing_columns:
                    if col == 'timestamp':
                        data[col] = pd.date_range(start=datetime.now() - timedelta(days=90),
                                                periods=len(data), freq='D')
                    else:
                        # 根据指标类型设置合理的默认值
                        if 'rate' in col or 'score' in col:
                            data[col] = 0.5  # 0-1之间的分数
                        elif 'time' in col:
                            data[col] = 1.0  # 时间相关的指标
                        else:
                            data[col] = 0.0

            # 处理时间戳
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')

            # 处理异常值
            for col in self.quality_metrics:
                if col in data.columns:
                    # 使用IQR方法检测异常值
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data[col] = data[col].clip(lower_bound, upper_bound)

            # 添加趋势特征
            data = self._add_trend_features(data)

            return data

        except Exception as e:
            logger.error(f"质量数据预处理失败: {e}")
            return pd.DataFrame()

    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加趋势特征"""
        try:
            for col in self.quality_metrics:
                if col in data.columns and len(data) > 7:
                    # 7天移动平均
                    data[f'{col}_ma7'] = data[col].rolling(window=7).mean()

                    # 7天标准差
                    data[f'{col}_std7'] = data[col].rolling(window=7).std()

                    # 7天趋势斜率
                    data[f'{col}_trend'] = data[col].rolling(window=7).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )

            return data

        except Exception as e:
            logger.error(f"添加趋势特征失败: {e}")
            return data

    def _build_trend_prediction_model(self, feature_count: int) -> keras.Model:
        """构建趋势预测模型"""
        model = keras.Sequential([
            layers.LSTM(64, input_shape=(30, feature_count), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(feature_count)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _prepare_trend_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备趋势训练数据"""
        try:
            window_size = 30  # 30天窗口

            if len(data) < window_size + 7:
                return np.array([]), np.array([])

            X, y = [], []

            for i in range(len(data) - window_size - 7):
                # 输入：过去30天的指标
                features = data[self.quality_metrics].iloc[i:i+window_size].values
                X.append(features)

                # 目标：未来7天的平均值
                future_values = data[self.quality_metrics].iloc[i+window_size:i+window_size+7].mean().values
                y.append(future_values)

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"准备趋势训练数据失败: {e}")
            return np.array([]), np.array([])

    def _train_risk_predictor(self, data: pd.DataFrame) -> RandomForestRegressor:
        """训练风险预测器"""
        try:
            # 创建风险标签（简化的逻辑）
            risk_scores = []

            for idx, row in data.iterrows():
                # 计算综合风险分数
                risk_score = 0

                # 基于各个指标计算风险
                if row.get('error_rate', 0) > 0.05:
                    risk_score += 0.3
                if row.get('test_success_rate', 1) < 0.95:
                    risk_score += 0.2
                if row.get('performance_score', 100) < 70:
                    risk_score += 0.2
                if row.get('change_failure_rate', 0) > 0.15:
                    risk_score += 0.3

                risk_scores.append(min(1.0, risk_score))

            # 训练回归模型
            predictor = RandomForestRegressor(n_estimators=50, random_state=42)

            features = data[self.quality_metrics].values
            scaled_features = self.scaler.fit_transform(features)

            predictor.fit(scaled_features, risk_scores)

            return predictor

        except Exception as e:
            logger.error(f"风险预测器训练失败: {e}")
            return None

    def _train_pattern_analyzer(self, data: pd.DataFrame) -> IsolationForest:
        """训练模式分析器"""
        try:
            analyzer = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

            features = data[self.quality_metrics].values
            scaled_features = self.scaler.fit_transform(features)

            analyzer.fit(scaled_features)

            return analyzer

        except Exception as e:
            logger.error(f"模式分析器训练失败: {e}")
            return None

    def _train_quality_predictor(self, data: pd.DataFrame) -> ARIMA:
        """训练质量预测器"""
        try:
            # 使用ARIMA模型预测整体质量分数
            quality_scores = data[self.quality_metrics].mean(axis=1).values

            # 检查时间序列的平稳性
            try:
                adf_result = adfuller(quality_scores)
                if adf_result[1] > 0.05:  # 不平稳
                    quality_scores = np.diff(quality_scores)  # 差分
            except:
                pass

            # 训练ARIMA模型
            model = ARIMA(quality_scores, order=(1, 1, 1))
            fitted_model = model.fit()

            return fitted_model

        except Exception as e:
            logger.error(f"质量预测器训练失败: {e}")
            return None

    def _predict_quality_trends(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """预测质量趋势"""
        try:
            predictions = {}

            if historical_context.empty:
                return {'predictions': predictions, 'overall_trend': 'unknown'}

            # 对每个质量指标进行趋势预测
            for metric in self.quality_metrics:
                if metric in historical_context.columns and len(historical_context) > 10:
                    values = historical_context[metric].values[-30:]  # 最近30个数据点

                    # 线性回归趋势
                    X = np.arange(len(values)).reshape(-1, 1)
                    y = values

                    reg = LinearRegression()
                    reg.fit(X, y)

                    slope = reg.coef_[0]
                    intercept = reg.intercept_

                    # 预测未来7天
                    future_X = np.arange(len(values), len(values) + 7).reshape(-1, 1)
                    future_predictions = reg.predict(future_X)

                    # 确定趋势方向
                    if slope > 0.001:
                        trend = 'increasing'
                    elif slope < -0.001:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'

                    predictions[metric] = {
                        'current_value': float(values[-1]),
                        'predicted_values': [float(p) for p in future_predictions],
                        'slope': float(slope),
                        'trend': trend,
                        'confidence': min(0.9, max(0.1, abs(slope) * 100))  # 简化的置信度
                    }

            # 计算整体趋势
            improving_metrics = sum(1 for p in predictions.values() if p['trend'] == 'increasing')
            declining_metrics = sum(1 for p in predictions.values() if p['trend'] == 'decreasing')
            stable_metrics = sum(1 for p in predictions.values() if p['trend'] == 'stable')

            if improving_metrics > declining_metrics and improving_metrics > stable_metrics:
                overall_trend = 'improving'
            elif declining_metrics > improving_metrics and declining_metrics > stable_metrics:
                overall_trend = 'declining'
            else:
                overall_trend = 'stable'

            return {
                'predictions': predictions,
                'overall_trend': overall_trend,
                'metrics_summary': {
                    'improving': improving_metrics,
                    'declining': declining_metrics,
                    'stable': stable_metrics,
                    'total': len(predictions)
                }
            }

        except Exception as e:
            logger.error(f"质量趋势预测失败: {e}")
            return {'predictions': {}, 'overall_trend': 'error'}

    def _assess_quality_risks(self, current_metrics: Dict[str, Any],
                            historical_context: pd.DataFrame) -> Dict[str, float]:
        """评估质量风险"""
        try:
            risk_scores = {}

            if self.risk_predictor is None:
                # 基于规则的风险评估
                for metric in self.quality_metrics:
                    current_value = current_metrics.get(metric, 0)

                    # 根据指标类型设置风险阈值
                    if 'error_rate' in metric or 'failure' in metric:
                        risk_scores[metric] = min(1.0, current_value / 0.1)  # 10%错误率作为高风险
                    elif 'success' in metric or 'coverage' in metric or 'score' in metric:
                        risk_scores[metric] = max(0.0, (1.0 - current_value) * 2)  # 成功率越低风险越高
                    elif 'time' in metric:
                        risk_scores[metric] = min(1.0, current_value / 10.0)  # 10秒作为高风险阈值
                    else:
                        risk_scores[metric] = 0.0
            else:
                # 使用训练的模型
                features = np.array([[current_metrics.get(m, 0) for m in self.quality_metrics]])
                scaled_features = self.scaler.transform(features)
                predicted_risks = self.risk_predictor.predict(scaled_features)

                for i, metric in enumerate(self.quality_metrics):
                    risk_scores[metric] = float(predicted_risks[i % len(predicted_risks)])

            return risk_scores

        except Exception as e:
            logger.error(f"质量风险评估失败: {e}")
            return {}

    def _analyze_quality_patterns(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析质量模式"""
        try:
            pattern_analysis = {
                'anomalies_detected': 0,
                'pattern_types': [],
                'seasonal_patterns': {},
                'correlation_patterns': {}
            }

            if self.pattern_analyzer is None or historical_context.empty:
                return pattern_analysis

            # 使用Isolation Forest检测异常模式
            features = historical_context[self.quality_metrics].values
            scaled_features = self.scaler.transform(features)
            anomaly_scores = self.pattern_analyzer.decision_function(scaled_features)
            anomaly_predictions = self.pattern_analyzer.predict(scaled_features)

            pattern_analysis['anomalies_detected'] = int(np.sum(anomaly_predictions == -1))

            # 分析季节性模式
            if len(historical_context) > 14:  # 至少两周数据
                for metric in self.quality_metrics:
                    if metric in historical_context.columns:
                        values = historical_context[metric].values

                        # 检查周模式（简化的分析）
                        weekly_pattern = self._detect_weekly_pattern(values)
                        if weekly_pattern:
                            pattern_analysis['seasonal_patterns'][metric] = weekly_pattern

            # 分析相关性模式
            if len(self.quality_metrics) > 1:
                corr_matrix = historical_context[self.quality_metrics].corr()

                # 找出强相关指标对
                strong_correlations = []
                for i in range(len(self.quality_metrics)):
                    for j in range(i+1, len(self.quality_metrics)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append({
                                'metric1': self.quality_metrics[i],
                                'metric2': self.quality_metrics[j],
                                'correlation': float(corr_value),
                                'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                            })

                pattern_analysis['correlation_patterns'] = strong_correlations

            return pattern_analysis

        except Exception as e:
            logger.error(f"质量模式分析失败: {e}")
            return {'anomalies_detected': 0, 'pattern_types': [], 'error': str(e)}

    def _detect_weekly_pattern(self, values: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测周模式"""
        try:
            if len(values) < 14:
                return None

            # 计算工作日vs周末的平均值差异
            weekdays = []
            weekends = []

            for i, value in enumerate(values):
                day_of_week = (i % 7)
                if day_of_week < 5:  # 周一到周五
                    weekdays.append(value)
                else:  # 周六周日
                    weekends.append(value)

            if weekdays and weekends:
                weekday_avg = np.mean(weekdays)
                weekend_avg = np.mean(weekends)
                difference = abs(weekday_avg - weekend_avg)

                if difference > np.std(values) * 0.5:  # 差异显著
                    return {
                        'weekday_avg': float(weekday_avg),
                        'weekend_avg': float(weekend_avg),
                        'difference': float(difference),
                        'pattern': 'weekday_vs_weekend'
                    }

            return None

        except Exception:
            return None

    def _generate_improvement_suggestions(self, trend_prediction: Dict[str, Any],
                                        risk_assessment: Dict[str, float],
                                        pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        try:
            suggestions = []

            # 基于趋势的建议
            overall_trend = trend_prediction.get('overall_trend', 'stable')
            if overall_trend == 'declining':
                suggestions.append({
                    'category': 'trend_improvement',
                    'priority': 'high',
                    'title': '质量指标整体下降',
                    'description': '多个质量指标显示下降趋势，需要立即关注',
                    'actions': [
                        '进行全面的质量评估',
                        '识别下降指标的根本原因',
                        '制定改进计划并跟踪执行'
                    ]
                })

            # 基于风险的建议
            high_risk_metrics = [metric for metric, score in risk_assessment.items()
                               if score > self.risk_thresholds['high']]

            if high_risk_metrics:
                suggestions.append({
                    'category': 'risk_mitigation',
                    'priority': 'high',
                    'title': f'高风险指标: {", ".join(high_risk_metrics)}',
                    'description': f'{len(high_risk_metrics)}个指标存在高风险',
                    'actions': [
                        f'优先改善: {", ".join(high_risk_metrics)}',
                        '建立监控和预警机制',
                        '制定具体改进措施'
                    ]
                })

            # 基于模式的建议
            if pattern_analysis.get('anomalies_detected', 0) > 0:
                suggestions.append({
                    'category': 'anomaly_investigation',
                    'priority': 'medium',
                    'title': '检测到质量异常模式',
                    'description': f'发现{pattern_analysis["anomalies_detected"]}个异常模式',
                    'actions': [
                        '分析异常发生的时间和条件',
                        '识别异常的根本原因',
                        '建立预防措施'
                    ]
                })

            # 基于相关性的建议
            correlations = pattern_analysis.get('correlation_patterns', [])
            if correlations:
                strong_corrs = [c for c in correlations if c['strength'] == 'strong']
                if strong_corrs:
                    suggestions.append({
                        'category': 'correlation_optimization',
                        'priority': 'medium',
                        'title': '强相关指标优化机会',
                        'description': f'发现{len(strong_corrs)}对强相关指标',
                        'actions': [
                            '分析指标间的因果关系',
                            '优化相关指标的同时改进策略',
                            '建立联动改进机制'
                        ]
                    })

            # 按优先级排序
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            suggestions.sort(key=lambda x: priority_order.get(x['priority'], 1), reverse=True)

            return suggestions

        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            return []

    def _calculate_overall_quality_score(self, current_metrics: Dict[str, Any],
                                       trend_prediction: Dict[str, Any]) -> float:
        """计算整体质量分数"""
        try:
            # 基于当前指标计算基础分数
            current_scores = []
            for metric in self.quality_metrics:
                value = current_metrics.get(metric, 0)

                # 标准化到0-1范围
                if 'rate' in metric or 'score' in metric:
                    normalized_score = min(1.0, max(0.0, value))
                elif 'error' in metric or 'failure' in metric:
                    normalized_score = max(0.0, 1.0 - value * 10)  # 错误率越高分数越低
                elif 'time' in metric:
                    normalized_score = max(0.0, 1.0 - value / 10.0)  # 时间越长分数越低
                else:
                    normalized_score = min(1.0, max(0.0, value / 100.0))  # 假设0-100范围

                current_scores.append(normalized_score)

            base_score = np.mean(current_scores) * 100

            # 根据趋势调整分数
            overall_trend = trend_prediction.get('overall_trend', 'stable')
            trend_adjustment = {
                'improving': 10,
                'stable': 0,
                'declining': -10,
                'unknown': 0
            }

            adjusted_score = base_score + trend_adjustment.get(overall_trend, 0)

            return max(0.0, min(100.0, adjusted_score))

        except Exception:
            return 50.0  # 默认中等质量

    def _determine_trend_direction(self, trend_prediction: Dict[str, Any]) -> str:
        """确定趋势方向"""
        overall_trend = trend_prediction.get('overall_trend', 'stable')

        # 转换为更详细的方向描述
        trend_mapping = {
            'improving': '上升趋势',
            'declining': '下降趋势',
            'stable': '稳定趋势',
            'unknown': '未知趋势'
        }

        return trend_mapping.get(overall_trend, '未知趋势')

    def _score_to_risk_level(self, score: float) -> str:
        """将分数转换为风险等级"""
        if score >= self.risk_thresholds['critical']:
            return 'critical'
        elif score >= self.risk_thresholds['high']:
            return 'high'
        elif score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    def _risk_level_priority(self, risk_level: str) -> int:
        """风险等级优先级"""
        priorities = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return priorities.get(risk_level, 1)

    def _get_metric_specific_actions(self, metric: str) -> List[str]:
        """获取指标特定的改进措施"""
        metric_actions = {
            'test_coverage': [
                '增加单元测试覆盖率',
                '完善集成测试场景',
                '添加端到端测试用例'
            ],
            'test_success_rate': [
                '修复测试用例失败问题',
                '优化测试环境稳定性',
                '改进测试数据质量'
            ],
            'error_rate': [
                '进行错误根因分析',
                '优化错误处理机制',
                '加强系统监控'
            ],
            'performance_score': [
                '进行性能瓶颈分析',
                '优化系统架构',
                '改进资源配置'
            ],
            'response_time': [
                '优化数据库查询',
                '改进缓存策略',
                '优化网络通信'
            ]
        }

        return metric_actions.get(metric, ['制定具体改进计划', '建立监控机制', '定期评估效果'])

    def _evaluate_trend_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """评估趋势模型"""
        try:
            metrics = {}

            # 评估风险预测器
            if self.risk_predictor:
                features = test_data[self.quality_metrics].values
                scaled_features = self.scaler.transform(features)

                # 生成模拟标签用于评估
                predictions = self.risk_predictor.predict(scaled_features)
                mse = mean_squared_error(np.random.uniform(0, 1, len(predictions)), predictions)

                metrics['risk_predictor'] = {
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse))
                }

            # 评估模式分析器
            if self.pattern_analyzer:
                features = test_data[self.quality_metrics].values
                scaled_features = self.scaler.transform(features)

                scores = self.pattern_analyzer.decision_function(scaled_features)
                anomaly_ratio = np.sum(self.pattern_analyzer.predict(scaled_features) == -1) / len(features)

                metrics['pattern_analyzer'] = {
                    'anomaly_ratio': float(anomaly_ratio),
                    'avg_anomaly_score': float(np.mean(scores))
                }

            return metrics

        except Exception as e:
            logger.error(f"趋势模型评估失败: {e}")
            return {'error': str(e)}

    def _save_trend_models(self):
        """保存趋势模型"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)

            # 保存模型
            if self.trend_model:
                self.trend_model.save(f"{self.model_path}/trend_model.h5")

            if self.risk_predictor:
                joblib.dump(self.risk_predictor, f"{self.model_path}/risk_predictor.pkl")

            if self.pattern_analyzer:
                joblib.dump(self.pattern_analyzer, f"{self.model_path}/pattern_analyzer.pkl")

            if self.quality_predictor:
                joblib.dump(self.quality_predictor, f"{self.model_path}/quality_predictor.pkl")

            # 保存标准化器
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")

            # 保存配置
            config = {
                'quality_metrics': self.quality_metrics,
                'risk_thresholds': self.risk_thresholds,
                'is_trained': self.is_trained
            }

            with open(f"{self.model_path}/config.json", 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"趋势模型保存失败: {e}")

    def load_trend_models(self) -> bool:
        """加载趋势模型"""
        try:
            # 加载配置
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)

            self.quality_metrics = config.get('quality_metrics', self.quality_metrics)
            self.risk_thresholds = config.get('risk_thresholds', self.risk_thresholds)
            self.is_trained = config.get('is_trained', False)

            # 加载模型
            self.trend_model = keras.models.load_model(f"{self.model_path}/trend_model.h5")
            self.risk_predictor = joblib.load(f"{self.model_path}/risk_predictor.pkl")
            self.pattern_analyzer = joblib.load(f"{self.model_path}/pattern_analyzer.pkl")
            self.quality_predictor = joblib.load(f"{self.model_path}/quality_predictor.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")

            return True

        except Exception as e:
            logger.error(f"趋势模型加载失败: {e}")
            return False


class QualityTrendAnalysisService:
    """质量趋势分析服务"""

    def __init__(self):
        self.analyzer = QualityTrendAnalyzer()
        self.analysis_history = []
        self.quality_alerts = []

    def initialize_service(self, historical_data_path: str = None) -> bool:
        """初始化服务"""
        try:
            # 尝试加载已训练的模型
            if self.analyzer.load_trend_models():
                logger.info("成功加载已训练的质量趋势分析模型")
                return True

            # 如果没有模型，尝试从历史数据训练
            if historical_data_path:
                historical_data = pd.read_csv(historical_data_path)
                result = self.analyzer.train_quality_trend_models(historical_data)

                if result.get('success', False):
                    logger.info("成功训练新的质量趋势分析模型")
                    return True

            logger.warning("无法初始化质量趋势分析服务")
            return False

        except Exception as e:
            logger.error(f"质量趋势分析服务初始化失败: {e}")
            return False

    def analyze_quality_trends_and_predict_risks(self, current_quality_metrics: Dict[str, Any],
                                               historical_context: pd.DataFrame,
                                               prediction_horizon: int = 30) -> Dict[str, Any]:
        """分析质量趋势并预测风险"""
        try:
            # 分析质量趋势
            trend_analysis = self.analyzer.analyze_quality_trends(current_quality_metrics, historical_context)

            if 'error' in trend_analysis:
                return trend_analysis

            # 预测质量下降风险
            quality_risks = self.analyzer.predict_quality_degradation_risks(trend_analysis, prediction_horizon)

            # 检查质量告警
            alerts = self._check_quality_alerts(trend_analysis, quality_risks)

            result = {
                'trend_analysis': trend_analysis,
                'quality_risks': quality_risks,
                'alerts': alerts,
                'risk_summary': self._summarize_quality_risks(quality_risks),
                'service_status': 'active'
            }

            # 记录分析历史
            self.analysis_history.append(result)

            # 保持历史记录大小
            if len(self.analysis_history) > 50:
                self.analysis_history = self.analysis_history[-50:]

            return result

        except Exception as e:
            logger.error(f"质量趋势分析和风险预测失败: {e}")
            return {
                'error': str(e),
                'service_status': 'error'
            }

    def get_trend_analysis_statistics(self) -> Dict[str, Any]:
        """获取趋势分析统计信息"""
        try:
            if not self.analysis_history:
                return {'total_analyses': 0}

            recent_analyses = self.analysis_history[-20:]  # 最近20次分析

            # 计算统计指标
            avg_quality_score = np.mean([a.get('trend_analysis', {}).get('overall_quality_score', 50)
                                       for a in recent_analyses])
            total_risks = sum(len(a.get('quality_risks', [])) for a in recent_analyses)
            total_alerts = sum(len(a.get('alerts', [])) for a in recent_analyses)

            # 趋势方向分布
            trend_directions = [a.get('trend_analysis', {}).get('quality_trend_direction', '未知')
                              for a in recent_analyses]

            trend_distribution = {
                '上升趋势': trend_directions.count('上升趋势'),
                '下降趋势': trend_directions.count('下降趋势'),
                '稳定趋势': trend_directions.count('稳定趋势'),
                '未知趋势': trend_directions.count('未知趋势')
            }

            # 风险等级分布
            risk_levels = []
            for analysis in recent_analyses:
                for risk in analysis.get('quality_risks', []):
                    risk_levels.append(risk.get('risk_level', 'low'))

            risk_distribution = {
                'critical': risk_levels.count('critical'),
                'high': risk_levels.count('high'),
                'medium': risk_levels.count('medium'),
                'low': risk_levels.count('low')
            }

            return {
                'total_analyses': len(self.analysis_history),
                'recent_analyses': len(recent_analyses),
                'avg_quality_score': float(avg_quality_score),
                'total_risks_identified': total_risks,
                'total_alerts_generated': total_alerts,
                'trend_distribution': trend_distribution,
                'risk_distribution': risk_distribution
            }

        except Exception as e:
            logger.error(f"获取趋势分析统计失败: {e}")
            return {'error': str(e)}

    def _check_quality_alerts(self, trend_analysis: Dict[str, Any],
                            quality_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查质量告警"""
        try:
            alerts = []

            # 检查质量分数告警
            overall_score = trend_analysis.get('overall_quality_score', 100)
            if overall_score < 60:
                alerts.append({
                    'alert_type': 'quality_score_low',
                    'severity': 'high' if overall_score < 40 else 'medium',
                    'message': f'整体质量分数过低: {overall_score:.1f}',
                    'current_score': overall_score,
                    'threshold': 60,
                    'recommendation': '立即进行质量评估和改进'
                })

            # 检查趋势告警
            trend_direction = trend_analysis.get('quality_trend_direction', '稳定趋势')
            if trend_direction == '下降趋势':
                alerts.append({
                    'alert_type': 'quality_trend_declining',
                    'severity': 'high',
                    'message': '质量指标显示下降趋势',
                    'trend_direction': trend_direction,
                    'recommendation': '分析下降原因并制定改进计划'
                })

            # 检查高风险告警
            critical_risks = [r for r in quality_risks if r.get('risk_level') == 'critical']
            high_risks = [r for r in quality_risks if r.get('risk_level') == 'high']

            if critical_risks:
                alerts.append({
                    'alert_type': 'critical_quality_risks',
                    'severity': 'critical',
                    'message': f'发现{len(critical_risks)}个严重质量风险',
                    'risk_count': len(critical_risks),
                    'recommendation': '立即处理严重质量风险'
                })

            if high_risks:
                alerts.append({
                    'alert_type': 'high_quality_risks',
                    'severity': 'high',
                    'message': f'发现{len(high_risks)}个高风险质量问题',
                    'risk_count': len(high_risks),
                    'recommendation': '优先处理高风险质量问题'
                })

            return alerts

        except Exception as e:
            logger.error(f"质量告警检查失败: {e}")
            return []

    def _summarize_quality_risks(self, quality_risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总质量风险"""
        try:
            if not quality_risks:
                return {'total_risks': 0, 'risk_summary': '无风险'}

            risk_counts = {
                'critical': len([r for r in quality_risks if r.get('risk_level') == 'critical']),
                'high': len([r for r in quality_risks if r.get('risk_level') == 'high']),
                'medium': len([r for r in quality_risks if r.get('risk_level') == 'medium']),
                'low': len([r for r in quality_risks if r.get('risk_level') == 'low'])
            }

            total_risks = sum(risk_counts.values())
            highest_risk_level = max(risk_counts.keys(),
                                   key=lambda k: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[k]
                                   if risk_counts[k] > 0 else 0)

            risk_summary = f"共{total_risks}个风险，最高风险等级: {highest_risk_level}"

            return {
                'total_risks': total_risks,
                'risk_counts': risk_counts,
                'highest_risk_level': highest_risk_level,
                'risk_summary': risk_summary
            }

        except Exception as e:
            logger.error(f"质量风险汇总失败: {e}")
            return {'total_risks': 0, 'risk_summary': '汇总失败'}

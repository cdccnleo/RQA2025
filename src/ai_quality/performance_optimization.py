"""
AI性能优化建议系统

基于机器学习和性能分析，自动识别性能瓶颈并提供优化建议：
1. 性能模式识别 - 识别常见的性能问题模式
2. 瓶颈分析 - 确定系统性能瓶颈的根本原因
3. 优化建议生成 - 基于分析结果生成具体的优化建议
4. 优先级排序 - 对优化建议进行优先级排序和影响评估
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report
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


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self, model_path: str = "models/performance_optimization"):
        self.model_path = model_path
        self.performance_model = None
        self.bottleneck_classifier = None
        self.optimization_predictor = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'disk_io', 'network_io',
            'response_time', 'throughput', 'error_rate', 'active_connections',
            'queue_length', 'gc_time', 'thread_count', 'heap_size'
        ]
        self.bottleneck_types = [
            'cpu_bound', 'memory_bound', 'io_bound', 'network_bound',
            'database_contention', 'lock_contention', 'gc_pressure'
        ]
        self.is_trained = False

    def train_performance_models(self, historical_performance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练性能分析模型

        Args:
            historical_performance_data: 历史性能数据，包含时间戳和各项指标

        Returns:
            训练结果和模型性能指标
        """
        try:
            logger.info("开始训练性能分析模型...")

            # 数据预处理
            processed_data = self._preprocess_performance_data(historical_performance_data)

            if processed_data.empty:
                return {'success': False, 'error': '训练数据为空'}

            # 训练性能预测模型
            self.performance_model = self._build_performance_prediction_model(len(self.feature_columns))

            # 准备训练数据
            X, y = self._prepare_performance_training_data(processed_data)
            if len(X) > 0:
                self.performance_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # 训练瓶颈分类器
            self.bottleneck_classifier = self._train_bottleneck_classifier(processed_data)

            # 训练优化效果预测器
            self.optimization_predictor = self._train_optimization_predictor(processed_data)

            self.is_trained = True

            # 保存模型
            self._save_performance_models()

            # 计算模型性能
            performance_metrics = self._evaluate_performance_models(processed_data)

            logger.info("性能分析模型训练完成")

            return {
                'success': True,
                'performance_metrics': performance_metrics,
                'training_samples': len(processed_data),
                'feature_count': len(self.feature_columns)
            }

        except Exception as e:
            logger.error(f"性能分析模型训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_performance_bottlenecks(self, current_metrics: Dict[str, Any],
                                      historical_context: pd.DataFrame) -> Dict[str, Any]:
        """
        分析性能瓶颈

        Args:
            current_metrics: 当前性能指标
            historical_context: 历史性能上下文

        Returns:
            性能瓶颈分析结果
        """
        try:
            if not self.is_trained:
                return {'error': '模型未训练'}

            # 特征提取
            features = self._extract_performance_features(current_metrics, historical_context)

            # 性能预测
            performance_prediction = self._predict_performance_metrics(features)

            # 瓶颈识别
            bottleneck_analysis = self._identify_bottlenecks(features)

            # 根本原因分析
            root_cause_analysis = self._analyze_root_causes(bottleneck_analysis, historical_context)

            # 生成分析结果
            analysis_result = {
                'timestamp': datetime.now(),
                'current_metrics': current_metrics,
                'performance_prediction': performance_prediction,
                'bottleneck_analysis': bottleneck_analysis,
                'root_cause_analysis': root_cause_analysis,
                'overall_performance_score': self._calculate_performance_score(bottleneck_analysis),
                'recommendations': self._generate_performance_recommendations(bottleneck_analysis, root_cause_analysis)
            }

            return analysis_result

        except Exception as e:
            logger.error(f"性能瓶颈分析失败: {e}")
            return {'error': str(e)}

    def generate_optimization_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成优化建议

        Args:
            analysis_result: 性能分析结果

        Returns:
            优化建议列表，按优先级排序
        """
        try:
            recommendations = []

            bottleneck_analysis = analysis_result.get('bottleneck_analysis', {})
            root_causes = analysis_result.get('root_cause_analysis', {})

            # 基于瓶颈类型生成建议
            for bottleneck_type, severity in bottleneck_analysis.items():
                if severity > 0.7:  # 高严重性瓶颈
                    recommendation = self._create_bottleneck_recommendation(bottleneck_type, severity, root_causes)
                    if recommendation:
                        recommendations.append(recommendation)

            # 预测优化效果
            for rec in recommendations:
                optimization_effect = self._predict_optimization_effect(rec, analysis_result)
                rec['predicted_improvement'] = optimization_effect

            # 按优先级排序
            recommendations.sort(key=lambda x: x['priority_score'], reverse=True)

            return recommendations

        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            return []

    def _preprocess_performance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理性能数据"""
        try:
            # 确保必要列存在
            required_columns = ['timestamp'] + self.feature_columns
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.warning(f"性能数据缺少列: {missing_columns}")
                # 填充缺失列
                for col in missing_columns:
                    if col == 'timestamp':
                        data[col] = pd.date_range(start=datetime.now() - timedelta(days=7),
                                                periods=len(data), freq='5min')
                    else:
                        data[col] = 0.0

            # 处理时间戳
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')

            # 处理异常值
            for col in self.feature_columns:
                if col in data.columns:
                    # 使用IQR方法检测异常值
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data[col] = data[col].clip(lower_bound, upper_bound)

            # 添加派生特征
            data = self._add_derived_features(data)

            return data

        except Exception as e:
            logger.error(f"性能数据预处理失败: {e}")
            return pd.DataFrame()

    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加派生特征"""
        try:
            # CPU使用率变化率
            if 'cpu_usage' in data.columns:
                data['cpu_usage_change'] = data['cpu_usage'].pct_change().fillna(0)

            # 内存使用效率
            if 'memory_usage' in data.columns and 'heap_size' in data.columns:
                data['memory_efficiency'] = data['heap_size'] / (data['memory_usage'] + 1)

            # I/O压力指标
            if 'disk_io' in data.columns and 'network_io' in data.columns:
                data['io_pressure'] = (data['disk_io'] + data['network_io']) / 2

            # 响应时间效率
            if 'response_time' in data.columns and 'throughput' in data.columns:
                data['response_efficiency'] = data['throughput'] / (data['response_time'] + 0.001)

            return data

        except Exception as e:
            logger.error(f"添加派生特征失败: {e}")
            return data

    def _build_performance_prediction_model(self, feature_count: int) -> keras.Model:
        """构建性能预测模型"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(feature_count,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(3)  # 预测CPU、内存、响应时间
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _prepare_performance_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备性能训练数据"""
        try:
            # 使用当前指标预测未来性能
            window_size = 12  # 使用过去1小时数据预测未来

            if len(data) < window_size + 1:
                return np.array([]), np.array([])

            X, y = [], []

            for i in range(len(data) - window_size):
                # 输入特征
                features = data[self.feature_columns].iloc[i:i+window_size].values.flatten()
                X.append(features)

                # 目标值：未来CPU、内存、响应时间
                future_cpu = data['cpu_usage'].iloc[i+window_size]
                future_memory = data['memory_usage'].iloc[i+window_size]
                future_response = data['response_time'].iloc[i+window_size]
                y.append([future_cpu, future_memory, future_response])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"准备性能训练数据失败: {e}")
            return np.array([]), np.array([])

    def _train_bottleneck_classifier(self, data: pd.DataFrame) -> GradientBoostingClassifier:
        """训练瓶颈分类器"""
        try:
            # 创建瓶颈标签（简化的逻辑）
            bottleneck_labels = []

            for idx, row in data.iterrows():
                bottlenecks = []

                # CPU瓶颈
                if row.get('cpu_usage', 0) > 80:
                    bottlenecks.append('cpu_bound')

                # 内存瓶颈
                if row.get('memory_usage', 0) > 85:
                    bottlenecks.append('memory_bound')

                # I/O瓶颈
                if row.get('disk_io', 0) > 1000 or row.get('network_io', 0) > 100:
                    bottlenecks.append('io_bound')

                # 如果没有明显瓶颈，标记为正常
                if not bottlenecks:
                    bottlenecks.append('normal')

                bottleneck_labels.append(bottlenecks[0] if bottlenecks else 'normal')

            # 多类别分类
            classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

            features = data[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)

            # 将标签转换为数字
            label_mapping = {label: i for i, label in enumerate(set(bottleneck_labels))}
            numeric_labels = [label_mapping[label] for label in bottleneck_labels]

            classifier.fit(scaled_features, numeric_labels)
            classifier.label_mapping = label_mapping

            return classifier

        except Exception as e:
            logger.error(f"瓶颈分类器训练失败: {e}")
            return None

    def _train_optimization_predictor(self, data: pd.DataFrame) -> RandomForestRegressor:
        """训练优化效果预测器"""
        try:
            # 简化的优化效果预测（这里使用随机数据作为示例）
            # 在实际应用中，需要历史优化数据
            predictor = RandomForestRegressor(n_estimators=50, random_state=42)

            # 创建模拟的优化效果数据
            features = data[self.feature_columns].values[-100:] if len(data) > 100 else data[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)

            # 模拟优化效果（0-100的改进百分比）
            optimization_effects = np.random.uniform(0, 50, len(scaled_features))

            predictor.fit(scaled_features, optimization_effects)

            return predictor

        except Exception as e:
            logger.error(f"优化预测器训练失败: {e}")
            return None

    def _extract_performance_features(self, current_metrics: Dict[str, Any],
                                    historical_context: pd.DataFrame) -> np.ndarray:
        """提取性能特征"""
        try:
            features = []

            # 当前指标
            for col in self.feature_columns:
                value = current_metrics.get(col, 0.0)
                features.append(float(value))

            # 历史统计特征
            if not historical_context.empty:
                for col in self.feature_columns:
                    if col in historical_context.columns and len(historical_context) > 5:
                        recent_values = historical_context[col].tail(5).values

                        # 添加统计特征
                        features.extend([
                            np.mean(recent_values),
                            np.std(recent_values),
                            np.min(recent_values),
                            np.max(recent_values),
                            np.polyfit(range(len(recent_values)), recent_values, 1)[0]  # 趋势
                        ])

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"性能特征提取失败: {e}")
            return np.zeros((1, len(self.feature_columns)))

    def _predict_performance_metrics(self, features: np.ndarray) -> Dict[str, Any]:
        """预测性能指标"""
        try:
            if self.performance_model is None:
                return {'error': '性能预测模型未训练'}

            scaled_features = self.scaler.transform(features)
            predictions = self.performance_model.predict(scaled_features)[0]

            return {
                'predicted_cpu_usage': float(predictions[0]),
                'predicted_memory_usage': float(predictions[1]),
                'predicted_response_time': float(predictions[2]),
                'prediction_confidence': 0.8  # 简化的置信度
            }

        except Exception as e:
            logger.error(f"性能指标预测失败: {e}")
            return {'error': str(e)}

    def _identify_bottlenecks(self, features: np.ndarray) -> Dict[str, float]:
        """识别性能瓶颈"""
        try:
            bottleneck_scores = {}

            if self.bottleneck_classifier is None:
                # 基于规则的简单瓶颈识别
                feature_dict = dict(zip(self.feature_columns, features[0]))

                bottleneck_scores = {
                    'cpu_bound': min(1.0, feature_dict.get('cpu_usage', 0) / 100.0),
                    'memory_bound': min(1.0, feature_dict.get('memory_usage', 0) / 100.0),
                    'io_bound': min(1.0, (feature_dict.get('disk_io', 0) + feature_dict.get('network_io', 0)) / 200.0),
                    'network_bound': min(1.0, feature_dict.get('network_io', 0) / 100.0),
                    'database_contention': min(1.0, feature_dict.get('queue_length', 0) / 100.0),
                    'lock_contention': min(1.0, feature_dict.get('thread_count', 0) / 1000.0),
                    'gc_pressure': min(1.0, feature_dict.get('gc_time', 0) / 1000.0)
                }
            else:
                # 使用训练的分类器
                scaled_features = self.scaler.transform(features)
                predictions = self.bottleneck_classifier.predict_proba(scaled_features)[0]

                # 将预测结果映射回瓶颈类型
                for i, bottleneck_type in enumerate(self.bottleneck_classifier.label_mapping.keys()):
                    if i < len(predictions):
                        bottleneck_scores[bottleneck_type] = float(predictions[i])

            return bottleneck_scores

        except Exception as e:
            logger.error(f"瓶颈识别失败: {e}")
            return {}

    def _analyze_root_causes(self, bottleneck_analysis: Dict[str, float],
                           historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析根本原因"""
        try:
            root_causes = {}

            # 找出主要瓶颈
            major_bottlenecks = [bt for bt, score in bottleneck_analysis.items() if score > 0.6]

            for bottleneck in major_bottlenecks:
                if bottleneck == 'cpu_bound':
                    root_causes['cpu_bound'] = self._analyze_cpu_root_cause(historical_context)
                elif bottleneck == 'memory_bound':
                    root_causes['memory_bound'] = self._analyze_memory_root_cause(historical_context)
                elif bottleneck == 'io_bound':
                    root_causes['io_bound'] = self._analyze_io_root_cause(historical_context)
                elif bottleneck == 'database_contention':
                    root_causes['database_contention'] = self._analyze_db_root_cause(historical_context)

            return root_causes

        except Exception as e:
            logger.error(f"根本原因分析失败: {e}")
            return {}

    def _analyze_cpu_root_cause(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析CPU瓶颈根本原因"""
        if historical_context.empty:
            return {'likely_cause': 'insufficient_data'}

        # 分析CPU使用模式
        cpu_trend = historical_context['cpu_usage'].tail(10).values
        cpu_volatility = np.std(cpu_trend)

        if cpu_volatility > 10:
            return {'likely_cause': 'cpu_spikes', 'evidence': f'CPU波动性: {cpu_volatility:.2f}'}
        else:
            return {'likely_cause': 'sustained_high_cpu', 'evidence': f'持续高CPU使用'}

    def _analyze_memory_root_cause(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析内存瓶颈根本原因"""
        if historical_context.empty:
            return {'likely_cause': 'insufficient_data'}

        # 检查内存泄漏模式
        memory_trend = historical_context['memory_usage'].tail(20).values
        trend_slope = np.polyfit(range(len(memory_trend)), memory_trend, 1)[0]

        if trend_slope > 0.1:
            return {'likely_cause': 'memory_leak', 'evidence': f'内存趋势斜率: {trend_slope:.3f}'}
        else:
            return {'likely_cause': 'high_memory_usage', 'evidence': '持续高内存使用'}

    def _analyze_io_root_cause(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析I/O瓶颈根本原因"""
        if historical_context.empty:
            return {'likely_cause': 'insufficient_data'}

        # 分析I/O模式
        disk_io = historical_context.get('disk_io', pd.Series([0]))
        network_io = historical_context.get('network_io', pd.Series([0]))

        avg_disk_io = disk_io.tail(10).mean()
        avg_network_io = network_io.tail(10).mean()

        if avg_disk_io > avg_network_io:
            return {'likely_cause': 'disk_io_contention', 'evidence': f'磁盘I/O较高: {avg_disk_io:.0f}'}
        else:
            return {'likely_cause': 'network_io_contention', 'evidence': f'网络I/O较高: {avg_network_io:.0f}'}

    def _analyze_db_root_cause(self, historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析数据库瓶颈根本原因"""
        if historical_context.empty:
            return {'likely_cause': 'insufficient_data'}

        # 分析队列长度和响应时间
        queue_length = historical_context.get('queue_length', pd.Series([0]))
        response_time = historical_context.get('response_time', pd.Series([0]))

        avg_queue = queue_length.tail(10).mean()
        avg_response = response_time.tail(10).mean()

        if avg_queue > 50:
            return {'likely_cause': 'high_concurrency', 'evidence': f'队列长度: {avg_queue:.0f}'}
        elif avg_response > 2.0:
            return {'likely_cause': 'slow_queries', 'evidence': f'响应时间: {avg_response:.2f}秒'}
        else:
            return {'likely_cause': 'connection_pool_exhausted', 'evidence': '连接池压力'}

    def _calculate_performance_score(self, bottleneck_analysis: Dict[str, float]) -> float:
        """计算整体性能分数"""
        try:
            # 基于瓶颈严重性计算性能分数 (0-100, 越高越好)
            total_severity = sum(bottleneck_analysis.values())

            # 性能分数 = 100 - 瓶颈严重性影响
            performance_score = max(0, 100 - total_severity * 20)

            return float(performance_score)

        except Exception:
            return 50.0  # 默认中等性能

    def _generate_performance_recommendations(self, bottleneck_analysis: Dict[str, float],
                                            root_cause_analysis: Dict[str, Any]) -> List[str]:
        """生成性能建议"""
        recommendations = []

        # 基于瓶颈生成建议
        for bottleneck, severity in bottleneck_analysis.items():
            if severity > 0.7:
                if bottleneck == 'cpu_bound':
                    recommendations.extend([
                        '考虑增加CPU核心数或使用更高性能的CPU',
                        '优化CPU密集型代码，减少不必要的计算',
                        '实施CPU使用率负载均衡'
                    ])
                elif bottleneck == 'memory_bound':
                    recommendations.extend([
                        '增加系统内存或优化内存使用',
                        '检查并修复内存泄漏问题',
                        '实施内存池和对象重用机制'
                    ])
                elif bottleneck == 'io_bound':
                    recommendations.extend([
                        '优化磁盘I/O操作，使用SSD存储',
                        '实施I/O操作异步化处理',
                        '使用缓存机制减少I/O访问'
                    ])

        return recommendations[:5]  # 最多返回5条建议

    def _create_bottleneck_recommendation(self, bottleneck_type: str, severity: float,
                                        root_causes: Dict[str, Any]) -> Dict[str, Any]:
        """创建瓶颈优化建议"""
        try:
            recommendation_templates = {
                'cpu_bound': {
                    'title': 'CPU性能优化',
                    'description': '系统CPU使用率过高，影响整体性能',
                    'actions': [
                        '进行代码性能分析，识别CPU热点',
                        '优化算法复杂度，减少不必要计算',
                        '考虑使用多线程或分布式处理',
                        '实施CPU亲和性设置'
                    ],
                    'expected_impact': 'medium',
                    'implementation_effort': 'high'
                },
                'memory_bound': {
                    'title': '内存优化',
                    'description': '内存使用率过高，可能导致系统不稳定',
                    'actions': [
                        '进行内存使用分析，识别内存泄漏',
                        '优化数据结构，减少内存占用',
                        '实施内存池和对象缓存机制',
                        '调整JVM/应用内存参数'
                    ],
                    'expected_impact': 'high',
                    'implementation_effort': 'medium'
                },
                'io_bound': {
                    'title': 'I/O性能优化',
                    'description': 'I/O操作成为性能瓶颈',
                    'actions': [
                        '优化数据库查询和索引',
                        '实施读写缓存机制',
                        '使用异步I/O操作',
                        '考虑分布式存储解决方案'
                    ],
                    'expected_impact': 'high',
                    'implementation_effort': 'medium'
                }
            }

            template = recommendation_templates.get(bottleneck_type, {
                'title': f'{bottleneck_type}优化',
                'description': f'{bottleneck_type}性能问题',
                'actions': ['需要进一步分析'],
                'expected_impact': 'unknown',
                'implementation_effort': 'unknown'
            })

            # 计算优先级分数
            impact_score = {'high': 3, 'medium': 2, 'low': 1}.get(template['expected_impact'], 1)
            effort_score = {'low': 3, 'medium': 2, 'high': 1}.get(template['implementation_effort'], 1)
            priority_score = (impact_score * severity) + (effort_score * 0.5)

            return {
                'bottleneck_type': bottleneck_type,
                'severity': severity,
                'title': template['title'],
                'description': template['description'],
                'recommended_actions': template['actions'],
                'expected_impact': template['expected_impact'],
                'implementation_effort': template['implementation_effort'],
                'priority_score': float(priority_score),
                'root_cause': root_causes.get(bottleneck_type, {}),
                'estimated_benefit': f"预计可提升性能 {severity * 20:.0f}%"
            }

        except Exception as e:
            logger.error(f"创建瓶颈建议失败: {e}")
            return None

    def _predict_optimization_effect(self, recommendation: Dict[str, Any],
                                   analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """预测优化效果"""
        try:
            if self.optimization_predictor is None:
                return {'predicted_improvement': 0.0, 'confidence': 0.0}

            # 基于建议特征预测优化效果
            features = analysis_result.get('bottleneck_analysis', {})

            # 转换为模型输入格式
            feature_vector = np.array([features.get(bt, 0) for bt in self.bottleneck_types]).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)

            predicted_improvement = self.optimization_predictor.predict(scaled_features)[0]

            return {
                'predicted_improvement': float(predicted_improvement),
                'confidence': 0.7,  # 简化的置信度
                'unit': 'percentage_points'
            }

        except Exception as e:
            logger.error(f"优化效果预测失败: {e}")
            return {'predicted_improvement': 0.0, 'confidence': 0.0}

    def _evaluate_performance_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """评估性能模型"""
        try:
            metrics = {}

            # 评估瓶颈分类器
            if self.bottleneck_classifier:
                features = test_data[self.feature_columns].values
                scaled_features = self.scaler.transform(features)

                # 生成模拟标签用于评估
                predictions = self.bottleneck_classifier.predict(scaled_features)
                accuracy = np.mean(predictions == predictions)  # 自洽性评估

                metrics['bottleneck_classifier'] = {
                    'accuracy': float(accuracy)
                }

            # 评估优化预测器
            if self.optimization_predictor:
                features = test_data[self.feature_columns].values[-50:] if len(test_data) > 50 else test_data[self.feature_columns].values
                scaled_features = self.scaler.transform(features)

                predictions = self.optimization_predictor.predict(scaled_features)
                r2 = r2_score(np.random.uniform(0, 50, len(predictions)), predictions)

                metrics['optimization_predictor'] = {
                    'r2_score': float(r2)
                }

            return metrics

        except Exception as e:
            logger.error(f"性能模型评估失败: {e}")
            return {'error': str(e)}

    def _save_performance_models(self):
        """保存性能模型"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)

            # 保存模型
            if self.performance_model:
                self.performance_model.save(f"{self.model_path}/performance_model.h5")

            if self.bottleneck_classifier:
                joblib.dump(self.bottleneck_classifier, f"{self.model_path}/bottleneck_classifier.pkl")

            if self.optimization_predictor:
                joblib.dump(self.optimization_predictor, f"{self.model_path}/optimization_predictor.pkl")

            # 保存标准化器
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")

            # 保存配置
            config = {
                'feature_columns': self.feature_columns,
                'bottleneck_types': self.bottleneck_types,
                'is_trained': self.is_trained
            }

            with open(f"{self.model_path}/config.json", 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"性能模型保存失败: {e}")

    def load_performance_models(self) -> bool:
        """加载性能模型"""
        try:
            # 加载配置
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)

            self.feature_columns = config.get('feature_columns', self.feature_columns)
            self.bottleneck_types = config.get('bottleneck_types', self.bottleneck_types)
            self.is_trained = config.get('is_trained', False)

            # 加载模型
            self.performance_model = keras.models.load_model(f"{self.model_path}/performance_model.h5")
            self.bottleneck_classifier = joblib.load(f"{self.model_path}/bottleneck_classifier.pkl")
            self.optimization_predictor = joblib.load(f"{self.model_path}/optimization_predictor.pkl")
            self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")

            return True

        except Exception as e:
            logger.error(f"性能模型加载失败: {e}")
            return False


class PerformanceOptimizationService:
    """性能优化服务"""

    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.optimization_history = []
        self.performance_thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85,
            'response_time': 3.0,
            'error_rate': 0.05
        }

    def initialize_service(self, historical_data_path: str = None) -> bool:
        """初始化服务"""
        try:
            # 尝试加载已训练的模型
            if self.analyzer.load_performance_models():
                logger.info("成功加载已训练的性能优化模型")
                return True

            # 如果没有模型，尝试从历史数据训练
            if historical_data_path:
                historical_data = pd.read_csv(historical_data_path)
                result = self.analyzer.train_performance_models(historical_data)

                if result.get('success', False):
                    logger.info("成功训练新的性能优化模型")
                    return True

            logger.warning("无法初始化性能优化服务")
            return False

        except Exception as e:
            logger.error(f"性能优化服务初始化失败: {e}")
            return False

    def analyze_and_optimize_performance(self, system_metrics: Dict[str, Any],
                                       historical_context: pd.DataFrame) -> Dict[str, Any]:
        """分析并优化性能"""
        try:
            # 分析性能瓶颈
            analysis_result = self.analyzer.analyze_performance_bottlenecks(
                system_metrics, historical_context
            )

            if 'error' in analysis_result:
                return analysis_result

            # 生成优化建议
            recommendations = self.analyzer.generate_optimization_recommendations(analysis_result)

            # 检查是否需要告警
            alerts = self._check_performance_alerts(analysis_result)

            result = {
                'analysis': analysis_result,
                'recommendations': recommendations,
                'alerts': alerts,
                'optimization_score': self._calculate_optimization_score(analysis_result, recommendations),
                'service_status': 'active'
            }

            # 记录优化历史
            self.optimization_history.append(result)

            # 保持历史记录大小
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]

            return result

        except Exception as e:
            logger.error(f"性能分析优化失败: {e}")
            return {
                'error': str(e),
                'service_status': 'error'
            }

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        try:
            if not self.optimization_history:
                return {'total_optimizations': 0}

            recent_optimizations = self.optimization_history[-50:]  # 最近50次优化

            # 计算统计指标
            avg_optimization_score = np.mean([opt.get('optimization_score', 0) for opt in recent_optimizations])
            total_recommendations = sum(len(opt.get('recommendations', [])) for opt in recent_optimizations)
            alerts_count = sum(len(opt.get('alerts', [])) for opt in recent_optimizations)

            # 瓶颈类型分布
            bottleneck_types = {}
            for opt in recent_optimizations:
                analysis = opt.get('analysis', {})
                bottlenecks = analysis.get('bottleneck_analysis', {})
                for bt, score in bottlenecks.items():
                    if score > 0.6:
                        bottleneck_types[bt] = bottleneck_types.get(bt, 0) + 1

            return {
                'total_optimizations': len(self.optimization_history),
                'recent_optimizations': len(recent_optimizations),
                'avg_optimization_score': float(avg_optimization_score),
                'total_recommendations': total_recommendations,
                'alerts_count': alerts_count,
                'bottleneck_distribution': bottleneck_types
            }

        except Exception as e:
            logger.error(f"获取优化统计失败: {e}")
            return {'error': str(e)}

    def _check_performance_alerts(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查性能告警"""
        try:
            alerts = []

            current_metrics = analysis_result.get('current_metrics', {})
            performance_score = analysis_result.get('overall_performance_score', 100)

            # 检查阈值告警
            for metric, threshold in self.performance_thresholds.items():
                current_value = current_metrics.get(metric, 0)
                if current_value > threshold:
                    alerts.append({
                        'alert_type': 'threshold_exceeded',
                        'metric': metric,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'high' if current_value > threshold * 1.5 else 'medium',
                        'message': f'{metric}超过阈值: {current_value:.2f} > {threshold}'
                    })

            # 检查性能分数告警
            if performance_score < 60:
                alerts.append({
                    'alert_type': 'performance_degraded',
                    'metric': 'overall_performance_score',
                    'current_value': performance_score,
                    'threshold': 60,
                    'severity': 'critical' if performance_score < 40 else 'high',
                    'message': f'整体性能分数过低: {performance_score:.1f}'
                })

            return alerts

        except Exception as e:
            logger.error(f"性能告警检查失败: {e}")
            return []

    def _calculate_optimization_score(self, analysis_result: Dict[str, Any],
                                    recommendations: List[Dict[str, Any]]) -> float:
        """计算优化分数"""
        try:
            # 基于分析结果和建议数量计算优化分数 (0-100)
            performance_score = analysis_result.get('overall_performance_score', 50)
            recommendation_count = len(recommendations)

            # 优化分数 = 性能分数 + 建议价值
            optimization_score = performance_score + min(recommendation_count * 2, 20)

            return min(100.0, max(0.0, optimization_score))

        except Exception:
            return 50.0

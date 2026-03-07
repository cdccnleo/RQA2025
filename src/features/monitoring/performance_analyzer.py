import logging
"""
特征层性能分析器

提供性能趋势分析、异常检测、瓶颈识别等功能，支持相关性分析和性能预测。
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import json
from datetime import datetime
# 使用统一的sklearn导入工具
from ..utils.sklearn_imports import LinearRegression


logger = logging.getLogger(__name__)


class AnalysisType(Enum):

    """分析类型枚举"""
    TREND = "trend"
    ANOMALY = "anomaly"
    BOTTLENECK = "bottleneck"
    CORRELATION = "correlation"
    PREDICTION = "prediction"


@dataclass
class PerformanceAnalysis:

    """性能分析结果"""
    analysis_type: AnalysisType
    metric_name: str
    timestamp: float
    result: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


class PerformanceAnalyzer:

    """
    性能分析器

    提供性能趋势分析、异常检测、瓶颈识别等功能，支持：
    - 趋势分析
    - 异常检测
    - 瓶颈识别
    - 相关性分析
    - 性能预测
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化性能分析器

        Args:
            config: 分析器配置
        """
        self.config = config or {}

        # 分析配置
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # 标准差倍数
        self.trend_window = self.config.get('trend_window', 60)  # 趋势分析窗口
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)  # 相关性阈值
        self.prediction_horizon = self.config.get('prediction_horizon', 10)  # 预测步数

        # 分析结果缓存
        self.analysis_cache: Dict[str, PerformanceAnalysis] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟缓存

        logger.info("性能分析器初始化完成")

    def analyze_performance(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析性能指标

        Args:
            metrics_history: 指标历史数据

        Returns:
            分析结果字典
        """
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'trends': {},
            'anomalies': {},
            'bottlenecks': {},
            'correlations': {},
            'predictions': {}
        }

        try:
            # 趋势分析
            analysis_results['trends'] = self._analyze_trends(metrics_history)

            # 异常检测
            analysis_results['anomalies'] = self._detect_anomalies(metrics_history)

            # 瓶颈识别
            analysis_results['bottlenecks'] = self._identify_bottlenecks(metrics_history)

            # 相关性分析
            analysis_results['correlations'] = self._analyze_correlations(metrics_history)

            # 性能预测
            analysis_results['predictions'] = self._predict_performance(metrics_history)

        except Exception as e:
            logger.error(f"性能分析失败: {str(e)}")

        return analysis_results

    def _analyze_trends(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析趋势

        Args:
            metrics_history: 指标历史数据

        Returns:
            趋势分析结果
        """
        trends = {}

        for metric_name, metric_data in metrics_history.items():
            if not metric_data or len(metric_data) < 3:
                continue

            try:
                # 提取时间序列数据
                timestamps = [m.timestamp for m in metric_data]
                values = [m.value for m in metric_data]

                # 计算趋势
                x = np.array(timestamps).reshape(-1, 1)
                y = np.array(values)

                # 线性回归
                model = LinearRegression()
                model.fit(x, y)

                # 计算趋势指标
                slope = model.coef_[0]
                r_squared = model.score(x, y)

                # 判断趋势方向
                if abs(slope) < 0.001:
                    trend_direction = "stable"
                elif slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"

                # 计算变化率
                if len(values) > 1:
                    change_rate = (values[-1] - values[0]) / values[0] * 100
                else:
                    change_rate = 0

                trends[metric_name] = {
                    'slope': slope,
                    'r_squared': r_squared,
                    'trend_direction': trend_direction,
                    'change_rate': change_rate,
                    'confidence': min(r_squared, 1.0)
                }

            except Exception as e:
                logger.warning(f"趋势分析失败 {metric_name}: {str(e)}")

        return trends

    def _detect_anomalies(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测异常

        Args:
            metrics_history: 指标历史数据

        Returns:
            异常检测结果
        """
        anomalies = {}

        for metric_name, metric_data in metrics_history.items():
            if not metric_data or len(metric_data) < 5:
                continue

            try:
                values = [m.value for m in metric_data]
                timestamps = [m.timestamp for m in metric_data]

                # 计算统计指标
                mean_val = np.mean(values)
                std_val = np.std(values)

                if std_val == 0:
                    continue

                # 检测异常点
                anomaly_indices = []
                anomaly_values = []

                for i, value in enumerate(values):
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > self.anomaly_threshold:
                        anomaly_indices.append(i)
                        anomaly_values.append(value)

                if anomaly_indices:
                    anomalies[metric_name] = {
                        'anomaly_count': len(anomaly_indices),
                        'anomaly_indices': anomaly_indices,
                        'anomaly_values': anomaly_values,
                        'anomaly_timestamps': [timestamps[i] for i in anomaly_indices],
                        'mean': mean_val,
                        'std': std_val,
                        'threshold': self.anomaly_threshold
                    }

            except Exception as e:
                logger.warning(f"异常检测失败 {metric_name}: {str(e)}")

        return anomalies

    def _identify_bottlenecks(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        识别瓶颈

        Args:
            metrics_history: 指标历史数据

        Returns:
            瓶颈识别结果
        """
        bottlenecks = {}

        # 定义瓶颈指标
        bottleneck_indicators = {
            'response_time': {'threshold': 1.0, 'direction': 'high'},
            'cpu_usage': {'threshold': 80.0, 'direction': 'high'},
            'memory_usage': {'threshold': 80.0, 'direction': 'high'},
            'error_rate': {'threshold': 0.05, 'direction': 'high'},
            'throughput': {'threshold': 100, 'direction': 'low'}
        }

        for metric_name, metric_data in metrics_history.items():
            if not metric_data:
                continue

            # 检查是否是瓶颈指标
            indicator = None
            for indicator_name, config in bottleneck_indicators.items():
                if indicator_name in metric_name.lower():
                    indicator = config
                    break

            if not indicator:
                continue

            try:
                values = [m.value for m in metric_data]
                recent_values = values[-10:] if len(values) >= 10 else values

                # 计算瓶颈程度
                threshold = indicator['threshold']
                direction = indicator['direction']

                if direction == 'high':
                    bottleneck_score = sum(1 for v in recent_values if v >
                                           threshold) / len(recent_values)
                else:  # low
                    bottleneck_score = sum(1 for v in recent_values if v <
                                           threshold) / len(recent_values)

                if bottleneck_score > 0.5:  # 超过50 % 的时间处于瓶颈状态
                    bottlenecks[metric_name] = {
                        'bottleneck_score': bottleneck_score,
                        'threshold': threshold,
                        'direction': direction,
                        'recent_values': recent_values,
                        'severity': 'high' if bottleneck_score > 0.8 else 'medium' if bottleneck_score > 0.6 else 'low'
                    }

            except Exception as e:
                logger.warning(f"瓶颈识别失败 {metric_name}: {str(e)}")

        return bottlenecks

    def _analyze_correlations(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析相关性

        Args:
            metrics_history: 指标历史数据

        Returns:
            相关性分析结果
        """
        correlations = {}

        # 准备数据矩阵
        metric_names = list(metrics_history.keys())
        if len(metric_names) < 2:
            return correlations

        # 对齐时间序列
        aligned_data = self._align_time_series(metrics_history)
        if aligned_data.empty:
            return correlations

        try:
            # 计算相关性矩阵
            correlation_matrix = aligned_data.corr()

            # 找出强相关性
            strong_correlations = []

            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    metric1 = metric_names[i]
                    metric2 = metric_names[j]
                    corr_value = correlation_matrix.iloc[i, j]

                    if abs(corr_value) > self.correlation_threshold:
                        strong_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })

            correlations['strong_correlations'] = strong_correlations
            correlations['correlation_matrix'] = correlation_matrix.to_dict()

        except Exception as e:
            logger.warning(f"相关性分析失败: {str(e)}")

        return correlations

    def _predict_performance(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测性能

        Args:
            metrics_history: 指标历史数据

        Returns:
            性能预测结果
        """
        predictions = {}

        for metric_name, metric_data in metrics_history.items():
            if not metric_data or len(metric_data) < 10:
                continue

            try:
                # 准备时间序列数据
                timestamps = [m.timestamp for m in metric_data]
                values = [m.value for m in metric_data]

                # 创建时间特征
                time_diffs = np.diff(timestamps)
                if len(time_diffs) == 0 or np.std(time_diffs) == 0:
                    continue

                # 使用简单线性回归进行预测
                x = np.array(range(len(values))).reshape(-1, 1)
                y = np.array(values)

                model = LinearRegression()
                model.fit(x, y)

                # 预测未来值
                future_x = np.array(range(len(values), len(values) +
                                    self.prediction_horizon)).reshape(-1, 1)
                future_predictions = model.predict(future_x)

                # 计算预测置信度
                y_pred = model.predict(x)
                mse = np.mean((y - y_pred) ** 2)
                confidence = max(0, 1 - mse / (np.var(y) + 1e-8))

                predictions[metric_name] = {
                    'predictions': future_predictions.tolist(),
                    'confidence': confidence,
                    'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                    'current_value': values[-1],
                    'predicted_change': (future_predictions[-1] - values[-1]) / values[-1] * 100 if values[-1] != 0 else 0
                }

            except Exception as e:
                logger.warning(f"性能预测失败 {metric_name}: {str(e)}")

        return predictions

    def _align_time_series(self, metrics_history: Dict[str, Any]) -> pd.DataFrame:
        """
        对齐时间序列数据

        Args:
            metrics_history: 指标历史数据

        Returns:
            对齐后的数据框
        """
        try:
            # 收集所有时间戳
            all_timestamps = set()
            for metric_data in metrics_history.values():
                if metric_data:
                    all_timestamps.update(m.timestamp for m in metric_data)

            if not all_timestamps:
                return pd.DataFrame()

            # 创建时间索引
            timestamps = sorted(all_timestamps)

            # 创建数据框
            data = {}
            for metric_name, metric_data in metrics_history.items():
                if not metric_data:
                    continue

                # 创建时间序列映射
                metric_dict = {m.timestamp: m.value for m in metric_data}

                # 填充缺失值
                values = []
                for ts in timestamps:
                    if ts in metric_dict:
                        values.append(metric_dict[ts])
                    else:
                        # 使用前向填充
                        prev_values = [v for t, v in metric_dict.items() if t < ts]
                        if prev_values:
                            values.append(prev_values[-1])
                        else:
                            values.append(np.nan)

                data[metric_name] = values

            return pd.DataFrame(data, index=timestamps)

        except Exception as e:
            logger.warning(f"时间序列对齐失败: {str(e)}")
            return pd.DataFrame()

    def analyze_component_performance(self, component_name: str,


                                      metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析组件性能

        Args:
            component_name: 组件名称
            metrics: 组件指标

        Returns:
            组件性能分析结果
        """
        analysis = {
            'component': component_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'issues': [],
            'recommendations': []
        }

        try:
            # 分析响应时间
            if 'response_time' in metrics:
                response_times = metrics['response_time']
                if response_times:
                    avg_response_time = np.mean(response_times)
                    max_response_time = np.max(response_times)

                    analysis['summary']['response_time'] = {
                        'average': avg_response_time,
                        'maximum': max_response_time,
                        'status': 'good' if avg_response_time < 1.0 else 'warning' if avg_response_time < 2.0 else 'critical'
                    }

                    if avg_response_time > 1.0:
                        analysis['issues'].append(f"平均响应时间过高: {avg_response_time:.3f}s")
                        analysis['recommendations'].append("优化算法实现或增加缓存")

            # 分析错误率
            if 'error_rate' in metrics:
                error_rates = metrics['error_rate']
                if error_rates:
                    avg_error_rate = np.mean(error_rates)

                    analysis['summary']['error_rate'] = {
                        'average': avg_error_rate,
                        'status': 'good' if avg_error_rate < 0.01 else 'warning' if avg_error_rate < 0.05 else 'critical'
                    }

                    if avg_error_rate > 0.01:
                        analysis['issues'].append(f"错误率过高: {avg_error_rate:.3%}")
                        analysis['recommendations'].append("检查错误处理逻辑和异常情况")

            # 分析吞吐量
            if 'throughput' in metrics:
                throughputs = metrics['throughput']
                if throughputs:
                    avg_throughput = np.mean(throughputs)

                    analysis['summary']['throughput'] = {
                        'average': avg_throughput,
                        'status': 'good' if avg_throughput > 100 else 'warning' if avg_throughput > 50 else 'critical'
                    }

                    if avg_throughput < 100:
                        analysis['issues'].append(f"吞吐量过低: {avg_throughput:.1f}")
                        analysis['recommendations'].append("优化并发处理或增加资源")

        except Exception as e:
            logger.error(f"组件性能分析失败 {component_name}: {str(e)}")

        return analysis

    def generate_performance_report(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成性能报告

        Args:
            metrics_history: 指标历史数据

        Returns:
            性能报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'trends': {},
            'anomalies': {},
            'bottlenecks': {},
            'recommendations': []
        }

        try:
            # 执行全面分析
            analysis_results = self.analyze_performance(metrics_history)

            # 汇总趋势
            trends = analysis_results.get('trends', {})
            increasing_trends = [name for name, trend in trends.items(
            ) if trend['trend_direction'] == 'increasing']
            decreasing_trends = [name for name, trend in trends.items(
            ) if trend['trend_direction'] == 'decreasing']

            report['trends'] = {
                'increasing': increasing_trends,
                'decreasing': decreasing_trends,
                'stable': [name for name, trend in trends.items() if trend['trend_direction'] == 'stable']
            }

            # 汇总异常
            anomalies = analysis_results.get('anomalies', {})
            total_anomalies = sum(len(anomaly['anomaly_indices']) for anomaly in anomalies.values())

            report['anomalies'] = {
                'total_count': total_anomalies,
                'affected_metrics': list(anomalies.keys())
            }

            # 汇总瓶颈
            bottlenecks = analysis_results.get('bottlenecks', {})
            critical_bottlenecks = [
                name for name, bottleneck in bottlenecks.items() if bottleneck['severity'] == 'high']

            report['bottlenecks'] = {
                'total_count': len(bottlenecks),
                'critical_count': len(critical_bottlenecks),
                'critical_metrics': critical_bottlenecks
            }

            # 生成建议
            if critical_bottlenecks:
                report['recommendations'].append("发现关键性能瓶颈，建议优先优化")

            if total_anomalies > 10:
                report['recommendations'].append("异常点较多，建议检查系统稳定性")

            if len(decreasing_trends) > len(increasing_trends):
                report['recommendations'].append("性能趋势下降，建议进行性能优化")

        except Exception as e:
            logger.error(f"生成性能报告失败: {str(e)}")

        return report

    def export_analysis(self, file_path: str, analysis_results: Dict[str, Any]) -> None:
        """
        导出分析结果

        Args:
            file_path: 导出文件路径
            analysis_results: 分析结果
        """
        try:
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)

            logger.info(f"性能分析结果已导出到: {file_path}")

        except Exception as e:
            logger.error(f"导出分析结果失败: {str(e)}")


# 全局性能分析器实例
_global_analyzer: Optional[PerformanceAnalyzer] = None


def get_analyzer(config: Optional[Dict] = None) -> PerformanceAnalyzer:
    """
    获取全局性能分析器实例

    Args:
        config: 分析器配置

    Returns:
        性能分析器实例
    """
    global _global_analyzer

    if _global_analyzer is None:
        _global_analyzer = PerformanceAnalyzer(config)

    return _global_analyzer


def analyze_performance(metrics_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析性能的便捷函数

    Args:
        metrics_history: 指标历史数据

    Returns:
        分析结果
    """
    analyzer = get_analyzer()
    return analyzer.analyze_performance(metrics_history)

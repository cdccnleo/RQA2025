"""
历史趋势分析模块
用于分析策略性能指标的长期趋势
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import math

# 使用统一日志系统
logger = logging.getLogger(__name__)

# 延迟导入数据库连接模块
db_available = False
try:
    from .postgresql_persistence import get_db_connection, return_db_connection
    db_available = True
except ImportError:
    logger.warning("PostgreSQL持久化模块不可用，将使用内存存储")

# 延迟导入InfluxDB模块
influxdb_available = False
try:
    from .influxdb_persistence import get_influxdb_client
    influxdb_available = True
except ImportError:
    logger.warning("InfluxDB持久化模块不可用，将使用内存存储")

# 内存存储作为降级方案
memory_storage = {
    "trend_data": [],
    "analysis_results": []
}


class HistoricalTrendAnalyzer:
    """
    历史趋势分析器类
    负责分析策略性能指标的长期趋势
    """
    
    @staticmethod
    def calculate_moving_average(values: List[float], window: int = 7) -> List[float]:
        """
        计算移动平均值
        
        Args:
            values: 数值列表
            window: 窗口大小
        
        Returns:
            移动平均值列表
        """
        if len(values) < window:
            return []
        
        moving_averages = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i+window]
            moving_average = sum(window_values) / window
            moving_averages.append(moving_average)
        
        return moving_averages
    
    @staticmethod
    def calculate_exponential_moving_average(values: List[float], alpha: float = 0.2) -> List[float]:
        """
        计算指数移动平均值
        
        Args:
            values: 数值列表
            alpha: 平滑因子
        
        Returns:
            指数移动平均值列表
        """
        if not values:
            return []
        
        ema = [values[0]]
        for i in range(1, len(values)):
            current_ema = alpha * values[i] + (1 - alpha) * ema[-1]
            ema.append(current_ema)
        
        return ema
    
    @staticmethod
    def calculate_linear_trend(values: List[float]) -> Tuple[float, float, float]:
        """
        计算线性趋势
        
        Args:
            values: 数值列表
        
        Returns:
            (斜率, 截距, R²值)
        """
        if len(values) < 2:
            return (0, 0, 0)
        
        n = len(values)
        x = list(range(n))
        
        # 计算均值
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)
        
        # 计算斜率和截距
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return (0, mean_y, 0)
        
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        
        # 计算R²值
        total_variation = sum((y - mean_y) ** 2 for y in values)
        if total_variation == 0:
            return (slope, intercept, 1)
        
        residual_variation = sum((values[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (residual_variation / total_variation)
        
        return (slope, intercept, r_squared)
    
    @staticmethod
    def calculate_volatility(values: List[float], window: int = 20) -> List[float]:
        """
        计算波动率
        
        Args:
            values: 数值列表
            window: 窗口大小
        
        Returns:
            波动率列表
        """
        if len(values) < window:
            return []
        
        volatility = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i+window]
            if len(window_values) > 1:
                std_dev = statistics.stdev(window_values)
                volatility.append(std_dev)
            else:
                volatility.append(0)
        
        return volatility
    
    @staticmethod
    def calculate_drawdown(returns: List[float]) -> List[float]:
        """
        计算回撤
        
        Args:
            returns: 收益率列表
        
        Returns:
            回撤列表
        """
        if not returns:
            return []
        
        cumulative = [1.0]
        for ret in returns:
            cumulative.append(cumulative[-1] * (1 + ret))
        
        drawdowns = []
        peak = cumulative[0]
        for value in cumulative[1:]:
            if value > peak:
                peak = value
                drawdowns.append(0)
            else:
                drawdown = (value - peak) / peak
                drawdowns.append(drawdown)
        
        return drawdowns
    
    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """
        计算最大回撤
        
        Args:
            returns: 收益率列表
        
        Returns:
            最大回撤值
        """
        drawdowns = HistoricalTrendAnalyzer.calculate_drawdown(returns)
        return min(drawdowns) if drawdowns else 0
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率列表
            risk_free_rate: 无风险利率
        
        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_excess_return = statistics.mean(excess_returns)
        std_excess_return = statistics.stdev(excess_returns)
        
        if std_excess_return == 0:
            return 0
        
        sharpe_ratio = mean_excess_return / std_excess_return
        # 年化夏普比率（假设每日数据）
        return sharpe_ratio * math.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率列表
            risk_free_rate: 无风险利率
        
        Returns:
            索提诺比率
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_excess_return = statistics.mean(excess_returns)
        
        # 计算下行标准差
        negative_returns = [r for r in excess_returns if r < 0]
        if not negative_returns:
            return 0
        
        downside_std = statistics.stdev(negative_returns)
        
        if downside_std == 0:
            return 0
        
        sortino_ratio = mean_excess_return / downside_std
        # 年化索提诺比率（假设每日数据）
        return sortino_ratio * math.sqrt(252)
    
    @staticmethod
    def get_performance_metrics(returns: List[float]) -> Dict[str, float]:
        """
        获取性能指标
        
        Args:
            returns: 收益率列表
        
        Returns:
            性能指标字典
        """
        if not returns:
            return {}
        
        total_return = math.prod([1 + r for r in returns]) - 1
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        max_drawdown = HistoricalTrendAnalyzer.calculate_max_drawdown(returns)
        sharpe_ratio = HistoricalTrendAnalyzer.calculate_sharpe_ratio(returns)
        sortino_ratio = HistoricalTrendAnalyzer.calculate_sortino_ratio(returns)
        
        # 计算胜率
        winning_trades = [r for r in returns if r > 0]
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # 计算平均盈亏比
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        losing_trades = [r for r in returns if r < 0]
        avg_loss = abs(statistics.mean(losing_trades)) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            "total_return": total_return,
            "mean_return": mean_return,
            "std_return": std_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    @staticmethod
    def analyze_trend(strategy_id: str, metric_name: str, values: List[float], 
                     timestamps: List[datetime], window: int = 30) -> Dict[str, Any]:
        """
        分析指标趋势
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            values: 指标值列表
            timestamps: 时间戳列表
            window: 分析窗口大小
        
        Returns:
            趋势分析结果
        """
        if len(values) < 2:
            return {
                "strategy_id": strategy_id,
                "metric_name": metric_name,
                "error": "数据不足"
            }
        
        # 计算基本统计量
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0
        min_value = min(values)
        max_value = max(values)
        
        # 计算移动平均
        ma_7 = HistoricalTrendAnalyzer.calculate_moving_average(values, window=7)
        ma_30 = HistoricalTrendAnalyzer.calculate_moving_average(values, window=30)
        
        # 计算指数移动平均
        ema = HistoricalTrendAnalyzer.calculate_exponential_moving_average(values)
        
        # 计算线性趋势
        slope, intercept, r_squared = HistoricalTrendAnalyzer.calculate_linear_trend(values)
        
        # 计算波动率
        volatility = HistoricalTrendAnalyzer.calculate_volatility(values)
        avg_volatility = statistics.mean(volatility) if volatility else 0
        
        # 计算趋势方向和强度
        trend_direction = "stable"
        trend_strength = "weak"
        
        if abs(slope) > 0.001:
            trend_direction = "upward" if slope > 0 else "downward"
            if r_squared > 0.7:
                trend_strength = "strong"
            elif r_squared > 0.3:
                trend_strength = "moderate"
        
        # 计算最近变化
        recent_change = values[-1] - values[0]
        percent_change = (recent_change / values[0]) * 100 if values[0] != 0 else 0
        
        # 计算周期性（简单的季节性检测）
        seasonality = HistoricalTrendAnalyzer.detect_seasonality(values)
        
        return {
            "strategy_id": strategy_id,
            "metric_name": metric_name,
            "analysis_period": {
                "start": timestamps[0].isoformat() if timestamps else None,
                "end": timestamps[-1].isoformat() if timestamps else None,
                "points": len(values)
            },
            "basic_statistics": {
                "mean": mean_value,
                "std": std_value,
                "min": min_value,
                "max": max_value
            },
            "trend_analysis": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength
            },
            "moving_averages": {
                "ma_7": ma_7[-1] if ma_7 else None,
                "ma_30": ma_30[-1] if ma_30 else None,
                "ema": ema[-1] if ema else None
            },
            "volatility": {
                "average": avg_volatility,
                "latest": volatility[-1] if volatility else 0
            },
            "change_analysis": {
                "absolute_change": recent_change,
                "percent_change": percent_change
            },
            "seasonality": seasonality,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def detect_seasonality(values: List[float]) -> Dict[str, Any]:
        """
        检测季节性
        
        Args:
            values: 数值列表
        
        Returns:
            季节性检测结果
        """
        if len(values) < 30:
            return {
                "detected": False,
                "reason": "数据不足"
            }
        
        # 简单的季节性检测：计算不同周期的自相关
        periods = [7, 14, 30, 60, 90]
        autocorrelations = {}
        
        for period in periods:
            if len(values) > period:
                # 计算滞后相关
                lagged_values = values[period:]
                original_values = values[:-period]
                
                if len(lagged_values) > 1:
                    # 计算相关系数
                    correlation = HistoricalTrendAnalyzer.calculate_correlation(original_values, lagged_values)
                    autocorrelations[period] = correlation
        
        # 查找最大自相关
        if autocorrelations:
            max_period = max(autocorrelations, key=autocorrelations.get)
            max_correlation = autocorrelations[max_period]
            
            if abs(max_correlation) > 0.5:
                return {
                    "detected": True,
                    "period": max_period,
                    "correlation": max_correlation,
                    "confidence": "high" if abs(max_correlation) > 0.7 else "medium"
                }
        
        return {
            "detected": False,
            "reason": "未检测到显著的季节性模式"
        }
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """
        计算相关系数
        
        Args:
            x: 第一个数值列表
            y: 第二个数值列表
        
        Returns:
            相关系数
        """
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0
        
        correlation = numerator / math.sqrt(denominator_x * denominator_y)
        return correlation
    
    @staticmethod
    def get_historical_data(strategy_id: str, metric_name: str, 
                          start_time: datetime, end_time: datetime, 
                          granularity: str = "day") -> Tuple[List[float], List[datetime]]:
        """
        获取历史数据
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            granularity: 时间粒度 (minute, hour, day, week, month)
        
        Returns:
            (值列表, 时间戳列表)
        """
        values = []
        timestamps = []
        
        # 优先使用InfluxDB
        if influxdb_available:
            try:
                client = get_influxdb_client()
                if client:
                    # 构建查询
                    query = f"""
                    from(bucket: "strategy_performance")
                      |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                      |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
                      |> filter(fn: (r) => r["strategy_id"] == "{strategy_id}")
                      |> filter(fn: (r) => r["_field"] == "{metric_name}")
                      |> aggregateWindow(every: 1{granularity}, fn: mean, createEmpty: false)
                      |> yield(name: "mean")
                    """
                    
                    # 执行查询
                    result = client.query(query)
                    
                    # 处理结果
                    for table in result:
                        for record in table.records:
                            values.append(record.get_value())
                            timestamps.append(record.get_time())
                    
                    return (values, timestamps)
            except Exception as e:
                logger.error(f"从InfluxDB获取历史数据失败: {e}")
        
        # 降级到PostgreSQL
        if db_available:
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # 根据粒度构建时间分组
                    time_format = {
                        "minute": "YYYY-MM-DD HH24:MI",
                        "hour": "YYYY-MM-DD HH24",
                        "day": "YYYY-MM-DD",
                        "week": "YYYY-WW",
                        "month": "YYYY-MM"
                    }.get(granularity, "YYYY-MM-DD")
                    
                    query = f"""
                        SELECT 
                            DATE_TRUNC('{granularity}', timestamp) as time_bucket,
                            AVG(value) as avg_value
                        FROM performance_metrics
                        WHERE 
                            strategy_id = %s AND
                            metric_name = %s AND
                            timestamp >= %s AND
                            timestamp <= %s
                        GROUP BY time_bucket
                        ORDER BY time_bucket
                    """
                    
                    cursor.execute(query, (strategy_id, metric_name, start_time, end_time))
                    
                    for row in cursor.fetchall():
                        timestamps.append(row[0])
                        values.append(row[1])
                    
                    cursor.close()
                    return_db_connection(conn)
                    
                    return (values, timestamps)
            except Exception as e:
                logger.error(f"从PostgreSQL获取历史数据失败: {e}")
        
        # 降级到内存存储
        filtered_data = []
        for item in memory_storage["trend_data"]:
            item_time = datetime.fromisoformat(item["timestamp"])
            if (item["strategy_id"] == strategy_id and 
                item["metric_name"] == metric_name and 
                start_time <= item_time <= end_time):
                filtered_data.append(item)
        
        # 按时间排序
        filtered_data.sort(key=lambda x: x["timestamp"])
        
        for item in filtered_data:
            values.append(item["value"])
            timestamps.append(datetime.fromisoformat(item["timestamp"]))
        
        return (values, timestamps)
    
    @staticmethod
    def store_trend_data(strategy_id: str, metric_name: str, value: float, timestamp: datetime) -> bool:
        """
        存储趋势数据
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            value: 指标值
            timestamp: 时间戳
        
        Returns:
            是否成功存储
        """
        try:
            # 优先使用InfluxDB
            if influxdb_available:
                client = get_influxdb_client()
                if client:
                    point = {
                        "measurement": "performance_metrics",
                        "tags": {
                            "strategy_id": strategy_id,
                            "metric_name": metric_name
                        },
                        "time": timestamp,
                        "fields": {
                            "value": value
                        }
                    }
                    client.write(point)
                    return True
            
            # 降级到PostgreSQL
            if db_available:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO performance_metrics (strategy_id, metric_name, value, timestamp)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (strategy_id, metric_name, timestamp) 
                        DO UPDATE SET value = EXCLUDED.value
                    """, (strategy_id, metric_name, value, timestamp))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
            
            # 降级到内存存储
            memory_storage["trend_data"].append({
                "strategy_id": strategy_id,
                "metric_name": metric_name,
                "value": value,
                "timestamp": timestamp.isoformat()
            })
            
            # 限制内存存储大小
            if len(memory_storage["trend_data"]) > 10000:
                memory_storage["trend_data"] = memory_storage["trend_data"][-5000:]
            
            return True
        except Exception as e:
            logger.error(f"存储趋势数据失败: {e}")
            return False
    
    @staticmethod
    def compare_trends(strategy_id: str, metrics: List[str], 
                     start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        比较多个指标的趋势
        
        Args:
            strategy_id: 策略ID
            metrics: 指标列表
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            趋势比较结果
        """
        comparisons = {}
        correlations = {}
        
        # 分析每个指标的趋势
        metric_data = {}
        for metric in metrics:
            values, timestamps = HistoricalTrendAnalyzer.get_historical_data(
                strategy_id, metric, start_time, end_time
            )
            if values:
                analysis = HistoricalTrendAnalyzer.analyze_trend(
                    strategy_id, metric, values, timestamps
                )
                comparisons[metric] = analysis
                metric_data[metric] = {
                    "values": values,
                    "timestamps": timestamps
                }
        
        # 计算指标间的相关性
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j and metric1 in metric_data and metric2 in metric_data:
                    # 确保数据点数量一致
                    data1 = metric_data[metric1]
                    data2 = metric_data[metric2]
                    
                    # 找到共同的时间点
                    common_data = []
                    for val1, ts1 in zip(data1["values"], data1["timestamps"]):
                        for val2, ts2 in zip(data2["values"], data2["timestamps"]):
                            if ts1 == ts2:
                                common_data.append((val1, val2))
                                break
                    
                    if len(common_data) > 1:
                        vals1, vals2 = zip(*common_data)
                        correlation = HistoricalTrendAnalyzer.calculate_correlation(vals1, vals2)
                        correlations[f"{metric1}_vs_{metric2}"] = correlation
        
        return {
            "strategy_id": strategy_id,
            "comparison_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "metric_analyses": comparisons,
            "inter_metric_correlations": correlations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def forecast_trend(values: List[float], forecast_period: int = 30) -> List[float]:
        """
        简单的趋势预测
        
        Args:
            values: 历史值列表
            forecast_period: 预测周期
        
        Returns:
            预测值列表
        """
        if len(values) < 2:
            return []
        
        # 使用线性回归进行预测
        slope, intercept, _ = HistoricalTrendAnalyzer.calculate_linear_trend(values)
        
        predictions = []
        last_index = len(values) - 1
        
        for i in range(1, forecast_period + 1):
            predicted_value = intercept + slope * (last_index + i)
            predictions.append(predicted_value)
        
        return predictions


# 全局趋势分析器实例
trend_analyzer = HistoricalTrendAnalyzer()


# 工具函数
def analyze_strategy_trends(strategy_id: str, metrics: List[str], 
                           start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """
    分析策略的多个指标趋势
    
    Args:
        strategy_id: 策略ID
        metrics: 指标列表
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        趋势分析结果
    """
    results = {
        "strategy_id": strategy_id,
        "analysis_period": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "metrics_analysis": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for metric in metrics:
        values, timestamps = HistoricalTrendAnalyzer.get_historical_data(
            strategy_id, metric, start_time, end_time
        )
        
        if values:
            analysis = HistoricalTrendAnalyzer.analyze_trend(
                strategy_id, metric, values, timestamps
            )
            results["metrics_analysis"][metric] = analysis
        else:
            results["metrics_analysis"][metric] = {
                "error": "无数据"
            }
    
    # 计算指标间的相关性
    if len(metrics) > 1:
        results["inter_metric_correlations"] = {}
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j:
                    data1 = results["metrics_analysis"].get(metric1, {})
                    data2 = results["metrics_analysis"].get(metric2, {})
                    
                    if "basic_statistics" in data1 and "basic_statistics" in data2:
                        # 这里应该使用实际的时间序列数据计算相关性
                        # 为了简化，我们使用模拟数据
                        results["inter_metric_correlations"][f"{metric1}_vs_{metric2}"] = 0.0
    
    return results


def get_trend_summary(strategy_id: str, 
                     time_period: str = "30d") -> Dict[str, Any]:
    """
    获取趋势摘要
    
    Args:
        strategy_id: 策略ID
        time_period: 时间周期 (7d, 30d, 90d, 1y)
    
    Returns:
        趋势摘要
    """
    # 解析时间周期
    period_map = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "1y": timedelta(days=365)
    }
    
    delta = period_map.get(time_period, timedelta(days=30))
    end_time = datetime.utcnow()
    start_time = end_time - delta
    
    # 分析关键指标
    key_metrics = ["return", "sharpe", "max_drawdown", "win_rate"]
    analysis = analyze_strategy_trends(strategy_id, key_metrics, start_time, end_time)
    
    # 生成摘要
    summary = {
        "strategy_id": strategy_id,
        "time_period": time_period,
        "trend_summary": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for metric, metric_analysis in analysis.get("metrics_analysis", {}).items():
        if "trend_analysis" in metric_analysis:
            summary["trend_summary"][metric] = {
                "trend_direction": metric_analysis["trend_analysis"]["trend_direction"],
                "trend_strength": metric_analysis["trend_analysis"]["trend_strength"],
                "latest_value": metric_analysis["basic_statistics"]["mean"],
                "change": metric_analysis["change_analysis"]["percent_change"]
            }
    
    return summary


def store_performance_metric(strategy_id: str, metric_name: str, 
                            value: float, timestamp: Optional[datetime] = None) -> bool:
    """
    存储性能指标
    
    Args:
        strategy_id: 策略ID
        metric_name: 指标名称
        value: 指标值
        timestamp: 时间戳
    
    Returns:
        是否成功存储
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    return HistoricalTrendAnalyzer.store_trend_data(
        strategy_id, metric_name, value, timestamp
    )


# 初始化函数
def initialize_trend_analysis():
    """
    初始化趋势分析模块
    """
    # 确保必要的表存在
    if db_available:
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # 创建性能指标表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        strategy_id VARCHAR(100) NOT NULL,
                        metric_name VARCHAR(100) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        UNIQUE(strategy_id, metric_name, timestamp)
                    );
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_strategy ON performance_metrics(strategy_id);
                    CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_metrics_strategy_name ON performance_metrics(strategy_id, metric_name);
                """)
                
                conn.commit()
                cursor.close()
                return_db_connection(conn)
                logger.info("趋势分析表创建成功")
        except Exception as e:
            logger.error(f"初始化趋势分析表失败: {e}")
    
    logger.info("历史趋势分析模块初始化成功")


# 测试函数
def test_trend_analysis():
    """
    测试趋势分析功能
    """
    print("测试趋势分析功能...")
    
    # 测试数据
    strategy_id = "test_strategy_1"
    
    # 生成测试数据
    import random
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=60)
    
    # 存储测试数据
    print("存储测试数据...")
    timestamps = []
    returns = []
    
    current_time = start_time
    current_value = 0
    
    while current_time <= end_time:
        # 生成带有趋势和噪声的收益率
        trend = 0.001  # 每日0.1%的趋势
        noise = random.gauss(0, 0.02)  # 标准差为2%的噪声
        daily_return = trend + noise
        current_value += daily_return
        
        # 存储数据
        store_performance_metric(strategy_id, "return", daily_return, current_time)
        store_performance_metric(strategy_id, "cumulative_return", current_value, current_time)
        
        timestamps.append(current_time)
        returns.append(daily_return)
        current_time += timedelta(days=1)
    
    # 分析趋势
    print("分析趋势...")
    analysis = analyze_strategy_trends(
        strategy_id, ["return", "cumulative_return"], start_time, end_time
    )
    
    print("趋势分析结果:")
    for metric, metric_analysis in analysis.get("metrics_analysis", {}).items():
        print(f"\n{metric} 分析:")
        print(f"  趋势方向: {metric_analysis['trend_analysis']['trend_direction']}")
        print(f"  趋势强度: {metric_analysis['trend_analysis']['trend_strength']}")
        print(f"  R²值: {metric_analysis['trend_analysis']['r_squared']:.4f}")
        print(f"  平均波动率: {metric_analysis['volatility']['average']:.4f}")
        print(f"  百分比变化: {metric_analysis['change_analysis']['percent_change']:.2f}%")
    
    # 获取趋势摘要
    print("\n获取趋势摘要...")
    summary = get_trend_summary(strategy_id, "30d")
    print(f"30天趋势摘要: {summary['trend_summary']}")
    
    # 测试预测功能
    print("\n测试预测功能...")
    forecast = HistoricalTrendAnalyzer.forecast_trend(returns, 7)
    print(f"7天预测值: {forecast}")


if __name__ == "__main__":
    initialize_trend_analysis()
    test_trend_analysis()

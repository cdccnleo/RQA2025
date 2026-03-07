"""
性能指标异常检测模块
用于检测策略性能指标中的异常值和异常模式
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

# 内存存储作为降级方案
memory_storage = {
    "anomalies": [],
    "detection_history": []
}


class AnomalyDetector:
    """
    异常检测器类
    负责检测和分析性能指标异常
    """
    
    @staticmethod
    def ensure_anomaly_tables() -> bool:
        """
        确保异常检测表存在
        
        Returns:
            是否成功创建表
        """
        if not db_available:
            return False
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 创建异常记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_anomalies (
                    id SERIAL PRIMARY KEY,
                    strategy_id VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    expected_value DOUBLE PRECISION,
                    deviation DOUBLE PRECISION,
                    severity VARCHAR(20) NOT NULL,  -- low, medium, high
                    detection_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    status VARCHAR(20) DEFAULT 'unresolved'  -- unresolved, resolved
                );
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomalies_strategy ON performance_anomalies(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_anomalies_metric ON performance_anomalies(metric_name);
                CREATE INDEX IF NOT EXISTS idx_anomalies_detection_time ON performance_anomalies(detection_time);
                CREATE INDEX IF NOT EXISTS idx_anomalies_status ON performance_anomalies(status);
            """)
            
            conn.commit()
            cursor.close()
            
            logger.info("异常检测表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建异常检测表失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    @staticmethod
    def detect_z_score_anomaly(values: List[float], new_value: float, threshold: float = 2.0) -> Dict[str, Any]:
        """
        使用Z-score方法检测异常
        
        Args:
            values: 历史值列表
            new_value: 新值
            threshold: Z-score阈值
        
        Returns:
            异常检测结果
        """
        if len(values) < 3:
            return {
                "is_anomaly": False,
                "reason": "样本数量不足"
            }
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return {
                "is_anomaly": False,
                "reason": "标准差为0"
            }
        
        z_score = abs((new_value - mean) / std_dev)
        is_anomaly = z_score > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "mean": mean,
            "std_dev": std_dev,
            "threshold": threshold,
            "deviation": abs(new_value - mean)
        }
    
    @staticmethod
    def detect_iqr_anomaly(values: List[float], new_value: float, multiplier: float = 1.5) -> Dict[str, Any]:
        """
        使用IQR方法检测异常
        
        Args:
            values: 历史值列表
            new_value: 新值
            multiplier: IQR乘数
        
        Returns:
            异常检测结果
        """
        if len(values) < 4:
            return {
                "is_anomaly": False,
                "reason": "样本数量不足"
            }
        
        sorted_values = sorted(values)
        q1 = sorted_values[int(len(sorted_values) * 0.25)]
        q3 = sorted_values[int(len(sorted_values) * 0.75)]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        is_anomaly = new_value < lower_bound or new_value > upper_bound
        
        return {
            "is_anomaly": is_anomaly,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "iqr": iqr,
            "deviation": min(abs(new_value - lower_bound), abs(new_value - upper_bound)) if is_anomaly else 0
        }
    
    @staticmethod
    def detect_moving_average_anomaly(values: List[float], new_value: float, window: int = 5, threshold: float = 0.3) -> Dict[str, Any]:
        """
        使用移动平均方法检测异常
        
        Args:
            values: 历史值列表
            new_value: 新值
            window: 移动窗口大小
            threshold: 偏差阈值（相对于移动平均的比例）
        
        Returns:
            异常检测结果
        """
        if len(values) < window:
            return {
                "is_anomaly": False,
                "reason": "样本数量不足"
            }
        
        moving_avg = sum(values[-window:]) / window
        deviation = abs(new_value - moving_avg) / moving_avg if moving_avg != 0 else 0
        is_anomaly = deviation > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "moving_avg": moving_avg,
            "deviation": deviation,
            "threshold": threshold
        }
    
    @staticmethod
    def detect_anomaly(strategy_id: str, metric_name: str, metric_value: float, 
                      historical_values: List[float]) -> Dict[str, Any]:
        """
        综合检测异常
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            metric_value: 指标值
            historical_values: 历史值列表
        
        Returns:
            异常检测结果
        """
        # 根据指标类型选择合适的检测方法
        detection_methods = {
            "return": [
                ("z_score", AnomalyDetector.detect_z_score_anomaly),
                ("moving_average", AnomalyDetector.detect_moving_average_anomaly)
            ],
            "drawdown": [
                ("iqr", AnomalyDetector.detect_iqr_anomaly),
                ("z_score", AnomalyDetector.detect_z_score_anomaly)
            ],
            "sharpe": [
                ("z_score", AnomalyDetector.detect_z_score_anomaly),
                ("moving_average", AnomalyDetector.detect_moving_average_anomaly)
            ],
            "max_drawdown": [
                ("iqr", AnomalyDetector.detect_iqr_anomaly)
            ],
            "win_rate": [
                ("z_score", AnomalyDetector.detect_z_score_anomaly)
            ],
            "avg_win": [
                ("z_score", AnomalyDetector.detect_z_score_anomaly)
            ],
            "avg_loss": [
                ("z_score", AnomalyDetector.detect_z_score_anomaly)
            ],
            "trades_per_day": [
                ("iqr", AnomalyDetector.detect_iqr_anomaly)
            ]
        }
        
        # 默认使用z-score方法
        default_methods = [("z_score", AnomalyDetector.detect_z_score_anomaly)]
        
        # 选择检测方法
        methods = detection_methods.get(metric_name.lower(), default_methods)
        
        # 执行检测
        results = {}
        anomaly_count = 0
        
        for method_name, method_func in methods:
            result = method_func(historical_values, metric_value)
            results[method_name] = result
            if result.get("is_anomaly", False):
                anomaly_count += 1
        
        # 综合判断：如果超过一半的方法检测到异常，则认为是异常
        is_anomaly = anomaly_count > len(methods) / 2
        
        # 确定严重程度
        severity = "low"
        if is_anomaly:
            # 计算异常得分
            anomaly_score = 0
            if "z_score" in results:
                z_result = results["z_score"]
                if z_result.get("is_anomaly", False):
                    anomaly_score += z_result.get("z_score", 0)
            
            if "iqr" in results:
                iqr_result = results["iqr"]
                if iqr_result.get("is_anomaly", False):
                    anomaly_score += iqr_result.get("deviation", 0) / iqr_result.get("iqr", 1)
            
            if "moving_average" in results:
                ma_result = results["moving_average"]
                if ma_result.get("is_anomaly", False):
                    anomaly_score += ma_result.get("deviation", 0) * 10
            
            # 根据得分确定严重程度
            if anomaly_score > 5:
                severity = "high"
            elif anomaly_score > 2:
                severity = "medium"
        
        # 计算预期值
        expected_value = None
        if historical_values:
            expected_value = statistics.mean(historical_values)
        
        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "detection_results": results,
            "expected_value": expected_value,
            "deviation": abs(metric_value - expected_value) if expected_value is not None else 0
        }
    
    @staticmethod
    def record_anomaly(strategy_id: str, metric_name: str, metric_value: float, 
                      expected_value: float, deviation: float, 
                      severity: str, description: str = None) -> bool:
        """
        记录异常
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            metric_value: 指标值
            expected_value: 预期值
            deviation: 偏差
            severity: 严重程度
            description: 描述
        
        Returns:
            是否成功记录
        """
        try:
            timestamp = datetime.utcnow()
            
            if db_available:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO performance_anomalies 
                        (strategy_id, metric_name, metric_value, expected_value, deviation, severity, description, detection_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (strategy_id, metric_name, metric_value, expected_value, deviation, severity, description, timestamp))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
            
            # 降级到内存存储
            memory_storage["anomalies"].append({
                "strategy_id": strategy_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "expected_value": expected_value,
                "deviation": deviation,
                "severity": severity,
                "description": description,
                "detection_time": timestamp.isoformat(),
                "status": "unresolved"
            })
            
            # 限制内存存储大小
            if len(memory_storage["anomalies"]) > 1000:
                memory_storage["anomalies"] = memory_storage["anomalies"][-500:]
            
            return True
            
        except Exception as e:
            logger.error(f"记录异常失败: {e}")
            return False
    
    @staticmethod
    def get_anomalies(strategy_id: str = None, metric_name: str = None, 
                     start_time: Optional[datetime] = None, 
                     end_time: Optional[datetime] = None, 
                     severity: str = None, status: str = None) -> List[Dict[str, Any]]:
        """
        获取异常记录
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            severity: 严重程度
            status: 状态
        
        Returns:
            异常记录列表
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT id, strategy_id, metric_name, metric_value, expected_value, deviation, 
                               severity, detection_time, description, status
                        FROM performance_anomalies
                    """
                    
                    where_clause = []
                    params = []
                    
                    if strategy_id:
                        where_clause.append("strategy_id = %s")
                        params.append(strategy_id)
                    if metric_name:
                        where_clause.append("metric_name = %s")
                        params.append(metric_name)
                    if start_time:
                        where_clause.append("detection_time >= %s")
                        params.append(start_time)
                    if end_time:
                        where_clause.append("detection_time <= %s")
                        params.append(end_time)
                    if severity:
                        where_clause.append("severity = %s")
                        params.append(severity)
                    if status:
                        where_clause.append("status = %s")
                        params.append(status)
                    
                    if where_clause:
                        query += " WHERE " + " AND ".join(where_clause)
                    
                    query += " ORDER BY detection_time DESC LIMIT 100"
                    
                    cursor.execute(query, params)
                    
                    results = []
                    for row in cursor.fetchall():
                        results.append({
                            "id": row[0],
                            "strategy_id": row[1],
                            "metric_name": row[2],
                            "metric_value": row[3],
                            "expected_value": row[4],
                            "deviation": row[5],
                            "severity": row[6],
                            "detection_time": row[7].isoformat(),
                            "description": row[8],
                            "status": row[9]
                        })
                    
                    cursor.close()
                    return_db_connection(conn)
                    return results
                    
                except Exception as e:
                    logger.error(f"获取异常记录失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        filtered_anomalies = memory_storage["anomalies"]
        
        if strategy_id:
            filtered_anomalies = [a for a in filtered_anomalies if a["strategy_id"] == strategy_id]
        if metric_name:
            filtered_anomalies = [a for a in filtered_anomalies if a["metric_name"] == metric_name]
        if start_time:
            filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["detection_time"]) >= start_time]
        if end_time:
            filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["detection_time"]) <= end_time]
        if severity:
            filtered_anomalies = [a for a in filtered_anomalies if a["severity"] == severity]
        if status:
            filtered_anomalies = [a for a in filtered_anomalies if a["status"] == status]
        
        # 按检测时间排序
        filtered_anomalies.sort(key=lambda x: x["detection_time"], reverse=True)
        
        return filtered_anomalies[:100]
    
    @staticmethod
    def resolve_anomaly(anomaly_id: int) -> bool:
        """
        解决异常
        
        Args:
            anomaly_id: 异常ID
        
        Returns:
            是否成功解决
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE performance_anomalies
                        SET status = 'resolved'
                        WHERE id = %s
                    """, (anomaly_id,))
                    conn.commit()
                    cursor.close()
                    return_db_connection(conn)
                    return True
                except Exception as e:
                    logger.error(f"解决异常失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        for anomaly in memory_storage["anomalies"]:
            if anomaly.get("id") == anomaly_id:
                anomaly["status"] = "resolved"
                return True
        
        return False
    
    @staticmethod
    def get_anomaly_summary(start_time: Optional[datetime] = None, 
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取异常摘要
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            异常摘要
        """
        if db_available:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT 
                            severity,
                            COUNT(*) as anomaly_count
                        FROM performance_anomalies
                    """
                    
                    where_clause = []
                    params = []
                    
                    if start_time:
                        where_clause.append("detection_time >= %s")
                        params.append(start_time)
                    if end_time:
                        where_clause.append("detection_time <= %s")
                        params.append(end_time)
                    
                    if where_clause:
                        query += " WHERE " + " AND ".join(where_clause)
                    
                    query += " GROUP BY severity"
                    
                    cursor.execute(query, params)
                    
                    severity_counts = {}
                    for row in cursor.fetchall():
                        severity_counts[row[0]] = row[1]
                    
                    # 获取按指标分组的异常数
                    cursor.execute("""
                        SELECT 
                            metric_name,
                            COUNT(*) as anomaly_count
                        FROM performance_anomalies
                    """
                    + (" WHERE " + " AND ".join(where_clause) if where_clause else "")
                    + " GROUP BY metric_name ORDER BY anomaly_count DESC LIMIT 10")
                    
                    metric_counts = []
                    for row in cursor.fetchall():
                        metric_counts.append({
                            "metric_name": row[0],
                            "anomaly_count": row[1]
                        })
                    
                    # 获取按策略分组的异常数
                    cursor.execute("""
                        SELECT 
                            strategy_id,
                            COUNT(*) as anomaly_count
                        FROM performance_anomalies
                    """
                    + (" WHERE " + " AND ".join(where_clause) if where_clause else "")
                    + " GROUP BY strategy_id ORDER BY anomaly_count DESC LIMIT 10")
                    
                    strategy_counts = []
                    for row in cursor.fetchall():
                        strategy_counts.append({
                            "strategy_id": row[0],
                            "anomaly_count": row[1]
                        })
                    
                    cursor.close()
                    return_db_connection(conn)
                    
                    return {
                        "severity_counts": severity_counts,
                        "metric_counts": metric_counts,
                        "strategy_counts": strategy_counts,
                        "total_anomalies": sum(severity_counts.values())
                    }
                    
                except Exception as e:
                    logger.error(f"获取异常摘要失败: {e}")
                    return_db_connection(conn)
        
        # 降级到内存存储
        filtered_anomalies = memory_storage["anomalies"]
        if start_time:
            filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["detection_time"]) >= start_time]
        if end_time:
            filtered_anomalies = [a for a in filtered_anomalies if datetime.fromisoformat(a["detection_time"]) <= end_time]
        
        # 计算严重程度分布
        severity_counts = {}
        for anomaly in filtered_anomalies:
            severity = anomaly["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 计算按指标分布
        metric_counts = {}
        for anomaly in filtered_anomalies:
            metric = anomaly["metric_name"]
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        # 计算按策略分布
        strategy_counts = {}
        for anomaly in filtered_anomalies:
            strategy = anomaly["strategy_id"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 转换为列表格式
        metric_counts_list = [
            {"metric_name": k, "anomaly_count": v}
            for k, v in sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        strategy_counts_list = [
            {"strategy_id": k, "anomaly_count": v}
            for k, v in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return {
            "severity_counts": severity_counts,
            "metric_counts": metric_counts_list,
            "strategy_counts": strategy_counts_list,
            "total_anomalies": len(filtered_anomalies)
        }


# 全局异常检测器实例
anomaly_detector = AnomalyDetector()


# 工具函数
def detect_strategy_anomalies(strategy_id: str, performance_metrics: Dict[str, float], 
                             historical_metrics: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    """
    检测策略性能指标异常
    
    Args:
        strategy_id: 策略ID
        performance_metrics: 性能指标字典
        historical_metrics: 历史指标字典
    
    Returns:
        检测到的异常列表
    """
    detected_anomalies = []
    
    for metric_name, metric_value in performance_metrics.items():
        # 获取历史值
        historical_values = historical_metrics.get(metric_name, [])
        
        # 检测异常
        detection_result = AnomalyDetector.detect_anomaly(
            strategy_id, metric_name, metric_value, historical_values
        )
        
        if detection_result["is_anomaly"]:
            # 记录异常
            description = f"{metric_name} 值 {metric_value} 异常，预期值约为 {detection_result['expected_value']:.2f}"
            AnomalyDetector.record_anomaly(
                strategy_id=strategy_id,
                metric_name=metric_name,
                metric_value=metric_value,
                expected_value=detection_result['expected_value'],
                deviation=detection_result['deviation'],
                severity=detection_result['severity'],
                description=description
            )
            
            detected_anomalies.append({
                "metric_name": metric_name,
                "metric_value": metric_value,
                "severity": detection_result['severity'],
                "description": description,
                "detection_result": detection_result
            })
    
    return detected_anomalies


def get_anomaly_alerts(severity: str = "medium", 
                      time_window: int = 24) -> List[Dict[str, Any]]:
    """
    获取异常警报
    
    Args:
        severity: 最小严重程度
        time_window: 时间窗口（小时）
    
    Returns:
        异常警报列表
    """
    start_time = datetime.utcnow() - timedelta(hours=time_window)
    
    # 获取未解决的异常
    anomalies = AnomalyDetector.get_anomalies(
        start_time=start_time,
        status="unresolved"
    )
    
    # 过滤严重程度
    severity_levels = {"low": 1, "medium": 2, "high": 3}
    min_severity_level = severity_levels.get(severity, 2)
    
    alerts = []
    for anomaly in anomalies:
        anomaly_severity_level = severity_levels.get(anomaly["severity"], 1)
        if anomaly_severity_level >= min_severity_level:
            alerts.append(anomaly)
    
    return alerts


# 初始化函数
def initialize_anomaly_detection():
    """
    初始化异常检测模块
    """
    # 确保异常检测表存在
    AnomalyDetector.ensure_anomaly_tables()
    logger.info("性能指标异常检测模块初始化成功")


# 测试函数
def test_anomaly_detection():
    """
    测试异常检测功能
    """
    print("测试异常检测功能...")
    
    # 测试数据
    strategy_id = "test_strategy_1"
    
    # 正常历史数据
    historical_returns = [0.01, 0.005, -0.002, 0.008, 0.003, 0.012, 0.007]
    historical_drawdown = [0.05, 0.03, 0.06, 0.04, 0.02, 0.05, 0.03]
    
    # 异常值
    abnormal_return = 0.1  # 异常高的收益
    abnormal_drawdown = 0.2  # 异常高的回撤
    
    # 检测异常
    performance_metrics = {
        "return": abnormal_return,
        "max_drawdown": abnormal_drawdown,
        "sharpe": 3.5
    }
    
    historical_metrics = {
        "return": historical_returns,
        "max_drawdown": historical_drawdown,
        "sharpe": [1.2, 1.5, 1.3, 1.4, 1.6, 1.5, 1.4]
    }
    
    print("检测异常...")
    detected_anomalies = detect_strategy_anomalies(
        strategy_id, performance_metrics, historical_metrics
    )
    
    print(f"检测到 {len(detected_anomalies)} 个异常:")
    for anomaly in detected_anomalies:
        print(f"- {anomaly['metric_name']}: {anomaly['metric_value']} ({anomaly['severity']})")
        print(f"  描述: {anomaly['description']}")
    
    # 获取异常摘要
    print("\n获取异常摘要...")
    summary = AnomalyDetector.get_anomaly_summary()
    print(f"严重程度分布: {summary['severity_counts']}")
    print(f"按指标分布: {summary['metric_counts']}")
    print(f"按策略分布: {summary['strategy_counts']}")
    print(f"总异常数: {summary['total_anomalies']}")
    
    # 获取异常警报
    print("\n获取异常警报...")
    alerts = get_anomaly_alerts(severity="medium")
    print(f"中高严重程度警报数: {len(alerts)}")
    for alert in alerts:
        print(f"- 策略: {alert['strategy_id']}, 指标: {alert['metric_name']}, 严重程度: {alert['severity']}")


if __name__ == "__main__":
    initialize_anomaly_detection()
    test_anomaly_detection()

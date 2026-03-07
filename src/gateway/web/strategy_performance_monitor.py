"""
策略性能监控模块
提供策略性能指标采集、趋势分析和报告生成功能
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: float
    metric_type: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    snapshot_id: str
    strategy_id: str
    timestamp: float
    metrics: Dict[str, float]
    period: str  # daily, weekly, monthly
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyPerformanceMonitor:
    """策略性能监控器"""
    
    def __init__(self, monitor_dir: str = "data/performance_monitor"):
        self.monitor_dir = monitor_dir
        self._ensure_directory()
        
        # 指标定义
        self.metric_definitions = {
            'total_return': {'label': '总收益率', 'unit': '%', 'format': '.2f'},
            'annual_return': {'label': '年化收益率', 'unit': '%', 'format': '.2f'},
            'sharpe_ratio': {'label': '夏普比率', 'unit': '', 'format': '.2f'},
            'max_drawdown': {'label': '最大回撤', 'unit': '%', 'format': '.2f'},
            'volatility': {'label': '波动率', 'unit': '%', 'format': '.2f'},
            'win_rate': {'label': '胜率', 'unit': '%', 'format': '.1f'},
            'profit_factor': {'label': '盈亏比', 'unit': '', 'format': '.2f'},
            'total_trades': {'label': '交易次数', 'unit': '', 'format': 'd'},
            'avg_trade_return': {'label': '平均交易收益', 'unit': '%', 'format': '.2f'},
            'calmar_ratio': {'label': '卡玛比率', 'unit': '', 'format': '.2f'}
        }
    
    def _ensure_directory(self):
        """确保监控目录存在"""
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)
            logger.info(f"创建性能监控目录: {self.monitor_dir}")
    
    def _get_strategy_monitor_dir(self, strategy_id: str) -> str:
        """获取策略监控目录"""
        strategy_dir = os.path.join(self.monitor_dir, strategy_id)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
        return strategy_dir
    
    def _generate_snapshot_id(self) -> str:
        """生成快照ID"""
        return f"snap_{int(time.time())}_{hash(str(time.time())) % 10000}"
    
    def record_performance(self, strategy_id: str, metrics: Dict[str, float],
                          period: str = "daily", metadata: Dict = None) -> PerformanceSnapshot:
        """记录性能快照"""
        try:
            snapshot = PerformanceSnapshot(
                snapshot_id=self._generate_snapshot_id(),
                strategy_id=strategy_id,
                timestamp=time.time(),
                metrics=metrics,
                period=period,
                metadata=metadata or {}
            )
            
            # 保存快照
            self._save_snapshot(snapshot)
            
            logger.info(f"记录性能快照: {strategy_id} - {period}")
            return snapshot
            
        except Exception as e:
            logger.error(f"记录性能快照失败: {e}")
            raise
    
    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """保存性能快照"""
        try:
            strategy_dir = self._get_strategy_monitor_dir(snapshot.strategy_id)
            
            # 按日期组织文件
            date_str = datetime.fromtimestamp(snapshot.timestamp).strftime('%Y%m%d')
            filepath = os.path.join(strategy_dir, f"{snapshot.period}_{date_str}.jsonl")
            
            # 追加写入
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'snapshot_id': snapshot.snapshot_id,
                    'timestamp': snapshot.timestamp,
                    'metrics': snapshot.metrics,
                    'metadata': snapshot.metadata
                }, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"保存性能快照失败: {e}")
            raise
    
    def get_performance_history(self, strategy_id: str, 
                               metric_name: str = None,
                               start_time: float = None,
                               end_time: float = None,
                               period: str = "daily") -> List[Dict]:
        """获取性能历史数据"""
        try:
            strategy_dir = self._get_strategy_monitor_dir(strategy_id)
            history = []
            
            # 读取所有相关文件
            for filename in os.listdir(strategy_dir):
                if filename.startswith(period) and filename.endswith('.jsonl'):
                    filepath = os.path.join(strategy_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                data = json.loads(line.strip())
                                timestamp = data['timestamp']
                                
                                # 时间范围筛选
                                if start_time and timestamp < start_time:
                                    continue
                                if end_time and timestamp > end_time:
                                    continue
                                
                                if metric_name:
                                    # 只返回特定指标
                                    if metric_name in data['metrics']:
                                        history.append({
                                            'timestamp': timestamp,
                                            'value': data['metrics'][metric_name]
                                        })
                                else:
                                    # 返回所有指标
                                    history.append({
                                        'timestamp': timestamp,
                                        'metrics': data['metrics']
                                    })
                    except Exception as e:
                        logger.warning(f"读取性能文件失败 {filename}: {e}")
            
            # 按时间排序
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"获取性能历史失败: {e}")
            return []
    
    def analyze_trend(self, strategy_id: str, metric_name: str,
                     period: str = "daily", days: int = 30) -> Dict[str, Any]:
        """分析指标趋势"""
        try:
            end_time = time.time()
            start_time = end_time - (days * 24 * 3600)
            
            history = self.get_performance_history(
                strategy_id, metric_name, start_time, end_time, period
            )
            
            if len(history) < 2:
                return {
                    'metric': metric_name,
                    'data_points': len(history),
                    'trend': 'insufficient_data',
                    'message': '数据点不足，无法分析趋势'
                }
            
            values = [h['value'] for h in history]
            timestamps = [h['timestamp'] for h in history]
            
            # 计算趋势
            first_value = values[0]
            last_value = values[-1]
            change = last_value - first_value
            change_percent = (change / abs(first_value) * 100) if first_value != 0 else 0
            
            # 线性回归计算趋势斜率
            x = np.array(range(len(values)))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0] if len(values) > 1 else 0
            
            # 计算统计指标
            avg_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            # 确定趋势方向
            if abs(slope) < std_value * 0.1:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            return {
                'metric': metric_name,
                'data_points': len(history),
                'trend': trend_direction,
                'slope': float(slope),
                'change': float(change),
                'change_percent': float(change_percent),
                'statistics': {
                    'average': float(avg_value),
                    'std': float(std_value),
                    'min': float(min_value),
                    'max': float(max_value),
                    'first': float(first_value),
                    'last': float(last_value)
                },
                'analysis_period_days': days
            }
            
        except Exception as e:
            logger.error(f"分析趋势失败: {e}")
            return {'error': str(e)}
    
    def compare_periods(self, strategy_id: str, metric_name: str,
                       period1_start: float, period1_end: float,
                       period2_start: float, period2_end: float) -> Dict[str, Any]:
        """对比两个时期的性能"""
        try:
            # 获取两个时期的数据
            period1_data = self.get_performance_history(
                strategy_id, metric_name, period1_start, period1_end
            )
            period2_data = self.get_performance_history(
                strategy_id, metric_name, period2_start, period2_end
            )
            
            if not period1_data or not period2_data:
                return {
                    'error': '数据不足，无法对比',
                    'period1_points': len(period1_data),
                    'period2_points': len(period2_data)
                }
            
            period1_values = [d['value'] for d in period1_data]
            period2_values = [d['value'] for d in period2_data]
            
            period1_avg = np.mean(period1_values)
            period2_avg = np.mean(period2_values)
            
            change = period2_avg - period1_avg
            change_percent = (change / abs(period1_avg) * 100) if period1_avg != 0 else 0
            
            return {
                'metric': metric_name,
                'period1': {
                    'start': period1_start,
                    'end': period1_end,
                    'average': float(period1_avg),
                    'std': float(np.std(period1_values)),
                    'data_points': len(period1_values)
                },
                'period2': {
                    'start': period2_start,
                    'end': period2_end,
                    'average': float(period2_avg),
                    'std': float(np.std(period2_values)),
                    'data_points': len(period2_values)
                },
                'comparison': {
                    'change': float(change),
                    'change_percent': float(change_percent),
                    'improved': change > 0
                }
            }
            
        except Exception as e:
            logger.error(f"对比时期失败: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self, strategy_id: str,
                                   start_time: float = None,
                                   end_time: float = None) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            if not end_time:
                end_time = time.time()
            if not start_time:
                start_time = end_time - (30 * 24 * 3600)  # 默认30天
            
            report = {
                'strategy_id': strategy_id,
                'report_period': {
                    'start': start_time,
                    'end': end_time,
                    'start_date': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d'),
                    'end_date': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')
                },
                'summary': {},
                'trends': {},
                'alerts': []
            }
            
            # 获取所有指标的历史数据
            for metric_name in self.metric_definitions.keys():
                history = self.get_performance_history(
                    strategy_id, metric_name, start_time, end_time
                )
                
                if history:
                    values = [h['value'] for h in history]
                    
                    # 汇总统计
                    report['summary'][metric_name] = {
                        'latest': values[-1],
                        'average': float(np.mean(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'std': float(np.std(values)),
                        'data_points': len(values)
                    }
                    
                    # 趋势分析
                    trend = self.analyze_trend(strategy_id, metric_name, 
                                              start_time=start_time, end_time=end_time)
                    if 'error' not in trend:
                        report['trends'][metric_name] = trend
            
            # 生成警报
            report['alerts'] = self._generate_report_alerts(report['summary'])
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}
    
    def _generate_report_alerts(self, summary: Dict) -> List[Dict]:
        """根据汇总数据生成警报"""
        alerts = []
        
        # 夏普比率警报
        if 'sharpe_ratio' in summary:
            sharpe = summary['sharpe_ratio']['latest']
            if sharpe < 0.5:
                alerts.append({
                    'level': 'warning',
                    'metric': 'sharpe_ratio',
                    'message': f'夏普比率 {sharpe:.2f} 偏低，建议优化策略',
                    'value': sharpe
                })
        
        # 最大回撤警报
        if 'max_drawdown' in summary:
            drawdown = summary['max_drawdown']['latest']
            if drawdown > 0.2:
                alerts.append({
                    'level': 'critical' if drawdown > 0.3 else 'warning',
                    'metric': 'max_drawdown',
                    'message': f'最大回撤 {drawdown*100:.1f}% 过高，注意风险控制',
                    'value': drawdown
                })
        
        # 胜率警报
        if 'win_rate' in summary:
            win_rate = summary['win_rate']['latest']
            if win_rate < 0.4:
                alerts.append({
                    'level': 'warning',
                    'metric': 'win_rate',
                    'message': f'胜率 {win_rate*100:.1f}% 偏低，建议优化入场条件',
                    'value': win_rate
                })
        
        return alerts
    
    def get_latest_metrics(self, strategy_id: str) -> Dict[str, float]:
        """获取最新指标"""
        try:
            latest_metrics = {}
            
            for metric_name in self.metric_definitions.keys():
                history = self.get_performance_history(strategy_id, metric_name)
                if history:
                    latest_metrics[metric_name] = history[-1]['value']
            
            return latest_metrics
            
        except Exception as e:
            logger.error(f"获取最新指标失败: {e}")
            return {}
    
    def calculate_performance_score(self, strategy_id: str) -> Dict[str, Any]:
        """计算综合性能评分"""
        try:
            latest = self.get_latest_metrics(strategy_id)
            
            if not latest:
                return {'error': '无性能数据'}
            
            # 各项权重
            weights = {
                'sharpe_ratio': 0.3,
                'total_return': 0.25,
                'max_drawdown': 0.2,
                'win_rate': 0.15,
                'profit_factor': 0.1
            }
            
            scores = {}
            total_score = 0
            total_weight = 0
            
            # 夏普比率评分 (0-2 映射到 0-100)
            if 'sharpe_ratio' in latest:
                sharpe_score = min(100, max(0, latest['sharpe_ratio'] * 50))
                scores['sharpe_ratio'] = round(sharpe_score, 1)
                total_score += sharpe_score * weights['sharpe_ratio']
                total_weight += weights['sharpe_ratio']
            
            # 收益率评分 (年化20%为满分)
            if 'annual_return' in latest:
                return_score = min(100, max(0, latest['annual_return'] * 5))
                scores['total_return'] = round(return_score, 1)
                total_score += return_score * weights['total_return']
                total_weight += weights['total_return']
            
            # 回撤评分 (越小越好，20%为0分)
            if 'max_drawdown' in latest:
                drawdown_score = max(0, 100 - latest['max_drawdown'] * 500)
                scores['max_drawdown'] = round(drawdown_score, 1)
                total_score += drawdown_score * weights['max_drawdown']
                total_weight += weights['max_drawdown']
            
            # 胜率评分 (60%为满分)
            if 'win_rate' in latest:
                winrate_score = min(100, latest['win_rate'] * 166.67)
                scores['win_rate'] = round(winrate_score, 1)
                total_score += winrate_score * weights['win_rate']
                total_weight += weights['win_rate']
            
            # 盈亏比评分 (2.0为满分)
            if 'profit_factor' in latest:
                pf_score = min(100, latest['profit_factor'] * 50)
                scores['profit_factor'] = round(pf_score, 1)
                total_score += pf_score * weights['profit_factor']
                total_weight += weights['profit_factor']
            
            final_score = total_score / total_weight if total_weight > 0 else 0
            
            # 评级
            if final_score >= 80:
                grade = 'A'
                grade_desc = '优秀'
            elif final_score >= 60:
                grade = 'B'
                grade_desc = '良好'
            elif final_score >= 40:
                grade = 'C'
                grade_desc = '一般'
            else:
                grade = 'D'
                grade_desc = '需改进'
            
            return {
                'strategy_id': strategy_id,
                'overall_score': round(final_score, 1),
                'grade': grade,
                'grade_description': grade_desc,
                'component_scores': scores,
                'weights': weights,
                'calculated_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"计算性能评分失败: {e}")
            return {'error': str(e)}


# 全局性能监控器实例
performance_monitor = StrategyPerformanceMonitor()


# 便捷的API函数
def record_strategy_performance(strategy_id: str, metrics: Dict[str, float],
                               period: str = "daily", metadata: Dict = None):
    """记录策略性能"""
    return performance_monitor.record_performance(strategy_id, metrics, period, metadata)


def get_strategy_performance_history(strategy_id: str, metric_name: str = None,
                                    days: int = 30) -> List[Dict]:
    """获取策略性能历史"""
    end_time = time.time()
    start_time = end_time - (days * 24 * 3600)
    return performance_monitor.get_performance_history(strategy_id, metric_name, 
                                                       start_time, end_time)


def generate_strategy_performance_report(strategy_id: str, days: int = 30) -> Dict[str, Any]:
    """生成策略性能报告"""
    end_time = time.time()
    start_time = end_time - (days * 24 * 3600)
    return performance_monitor.generate_performance_report(strategy_id, start_time, end_time)


def calculate_strategy_score(strategy_id: str) -> Dict[str, Any]:
    """计算策略评分"""
    return performance_monitor.calculate_performance_score(strategy_id)

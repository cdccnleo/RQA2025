#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量指标模块

提供数据质量监控指标和集成到监控系统的功能
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class QualityMetricsService:
    """
    数据质量指标服务
    
    提供数据质量监控和指标收集功能
    """
    
    def __init__(self):
        """
        初始化数据质量指标服务
        """
        self.metrics_cache = {}
        self.metrics_history = {}
        self.cache_timeout = 300  # 5分钟缓存
        
        logger.info("✅ 数据质量指标服务初始化完成")
    
    def collect_quality_metrics(self, data_source: str, data_type: str, 
                              record_count: int, invalid_count: int, 
                              quality_score: float) -> Dict[str, Any]:
        """
        收集数据质量指标
        
        Args:
            data_source: 数据源
            data_type: 数据类型
            record_count: 记录数
            invalid_count: 无效记录数
            quality_score: 质量评分
            
        Returns:
            质量指标字典
        """
        timestamp = time.time()
        
        metrics = {
            "timestamp": timestamp,
            "data_source": data_source,
            "data_type": data_type,
            "record_count": record_count,
            "invalid_count": invalid_count,
            "valid_count": record_count - invalid_count,
            "invalid_ratio": invalid_count / record_count if record_count > 0 else 0,
            "quality_score": quality_score,
            "quality_level": self._calculate_quality_level(quality_score),
            "processing_time": time.time() - timestamp
        }
        
        # 缓存指标
        cache_key = f"{data_source}:{data_type}"
        self.metrics_cache[cache_key] = metrics
        
        # 记录历史指标
        if cache_key not in self.metrics_history:
            self.metrics_history[cache_key] = []
        self.metrics_history[cache_key].append(metrics)
        
        # 限制历史记录长度
        if len(self.metrics_history[cache_key]) > 100:
            self.metrics_history[cache_key] = self.metrics_history[cache_key][-100:]
        
        logger.info(f"📊 收集数据质量指标: {data_source}/{data_type}, 评分: {quality_score:.2f}")
        return metrics
    
    def _calculate_quality_level(self, score: float) -> str:
        """
        计算质量等级
        
        Args:
            score: 质量评分
            
        Returns:
            质量等级
        """
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        elif score >= 0.3:
            return "poor"
        else:
            return "critical"
    
    def get_latest_metrics(self, data_source: Optional[str] = None, 
                          data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取最新的质量指标
        
        Args:
            data_source: 数据源
            data_type: 数据类型
            
        Returns:
            质量指标列表
        """
        result = []
        
        for cache_key, metrics in self.metrics_cache.items():
            source, type_ = cache_key.split(":")
            
            if data_source and source != data_source:
                continue
            if data_type and type_ != data_type:
                continue
            
            result.append(metrics)
        
        # 按时间戳排序
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result
    
    def get_metrics_history(self, data_source: str, data_type: str, 
                           hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取历史质量指标
        
        Args:
            data_source: 数据源
            data_type: 数据类型
            hours: 时间范围（小时）
            
        Returns:
            历史质量指标列表
        """
        cache_key = f"{data_source}:{data_type}"
        history = self.metrics_history.get(cache_key, [])
        
        # 过滤时间范围
        cutoff_time = time.time() - (hours * 3600)
        filtered_history = [m for m in history if m["timestamp"] >= cutoff_time]
        
        return filtered_history
    
    def generate_quality_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        生成质量报告
        
        Args:
            hours: 时间范围（小时）
            
        Returns:
            质量报告
        """
        report = {
            "timestamp": time.time(),
            "time_range": hours,
            "summary": {},
            "detailed": {}
        }
        
        # 计算摘要
        total_records = 0
        total_invalid = 0
        total_score = 0
        metric_count = 0
        
        for cache_key, history in self.metrics_history.items():
            source, type_ = cache_key.split(":")
            
            # 过滤时间范围
            cutoff_time = time.time() - (hours * 3600)
            filtered_history = [m for m in history if m["timestamp"] >= cutoff_time]
            
            if not filtered_history:
                continue
            
            # 计算该数据源的统计信息
            source_records = sum(m["record_count"] for m in filtered_history)
            source_invalid = sum(m["invalid_count"] for m in filtered_history)
            source_score = sum(m["quality_score"] for m in filtered_history) / len(filtered_history)
            
            # 添加到报告
            if source not in report["detailed"]:
                report["detailed"][source] = {}
            report["detailed"][source][type_] = {
                "record_count": source_records,
                "invalid_count": source_invalid,
                "invalid_ratio": source_invalid / source_records if source_records > 0 else 0,
                "avg_quality_score": source_score,
                "quality_level": self._calculate_quality_level(source_score),
                "sample_count": len(filtered_history)
            }
            
            # 累计总统计
            total_records += source_records
            total_invalid += source_invalid
            total_score += source_score * len(filtered_history)
            metric_count += len(filtered_history)
        
        # 计算总摘要
        if metric_count > 0:
            avg_score = total_score / metric_count
            report["summary"] = {
                "total_records": total_records,
                "total_invalid": total_invalid,
                "total_valid": total_records - total_invalid,
                "overall_invalid_ratio": total_invalid / total_records if total_records > 0 else 0,
                "overall_quality_score": avg_score,
                "overall_quality_level": self._calculate_quality_level(avg_score),
                "metric_count": metric_count,
                "data_sources": list(report["detailed"].keys())
            }
        
        logger.info(f"📋 生成质量报告: 时间范围 {hours} 小时, 数据源 {len(report['detailed'])} 个")
        return report
    
    def check_quality_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查质量阈值
        
        Args:
            metrics: 质量指标
            
        Returns:
            告警列表
        """
        alerts = []
        
        # 检查质量评分阈值
        if metrics["quality_score"] < 0.7:
            alerts.append({
                "level": "warning",
                "type": "quality_score_low",
                "message": f"数据质量评分过低: {metrics['quality_score']:.2f}",
                "details": metrics
            })
        
        if metrics["quality_score"] < 0.5:
            alerts.append({
                "level": "critical",
                "type": "quality_score_critical",
                "message": f"数据质量评分严重过低: {metrics['quality_score']:.2f}",
                "details": metrics
            })
        
        # 检查无效记录比例
        if metrics["invalid_ratio"] > 0.1:
            alerts.append({
                "level": "warning",
                "type": "invalid_ratio_high",
                "message": f"无效记录比例过高: {metrics['invalid_ratio']:.2f}",
                "details": metrics
            })
        
        if metrics["invalid_ratio"] > 0.3:
            alerts.append({
                "level": "critical",
                "type": "invalid_ratio_critical",
                "message": f"无效记录比例严重过高: {metrics['invalid_ratio']:.2f}",
                "details": metrics
            })
        
        return alerts
    
    def cleanup_old_metrics(self):
        """
        清理旧的指标数据
        """
        current_time = time.time()
        
        # 清理过期缓存
        expired_keys = []
        for cache_key, metrics in self.metrics_cache.items():
            if current_time - metrics["timestamp"] > self.cache_timeout:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.metrics_cache[key]
        
        # 清理过期历史数据
        for cache_key, history in self.metrics_history.items():
            cutoff_time = current_time - (7 * 24 * 3600)  # 保留7天
            self.metrics_history[cache_key] = [
                m for m in history if m["timestamp"] >= cutoff_time
            ]
        
        logger.info(f"🧹 清理过期指标数据: 移除 {len(expired_keys)} 个缓存项")
    
    def get_prometheus_metrics(self) -> str:
        """
        获取Prometheus格式的指标
        
        Returns:
            Prometheus格式的指标字符串
        """
        metrics_lines = []
        
        # 添加指标描述
        metrics_lines.append("# HELP rqa2025_data_quality_score Data quality score")
        metrics_lines.append("# TYPE rqa2025_data_quality_score gauge")
        metrics_lines.append("# HELP rqa2025_data_invalid_ratio Invalid data ratio")
        metrics_lines.append("# TYPE rqa2025_data_invalid_ratio gauge")
        metrics_lines.append("# HELP rqa2025_data_record_count Total record count")
        metrics_lines.append("# TYPE rqa2025_data_record_count gauge")
        
        # 添加指标数据
        for cache_key, metrics in self.metrics_cache.items():
            source, type_ = cache_key.split(":")
            
            # 质量评分
            metrics_lines.append(
                f'rqa2025_data_quality_score{{data_source="{source}", data_type="{type}"}} {metrics["quality_score"]}'
            )
            
            # 无效记录比例
            metrics_lines.append(
                f'rqa2025_data_invalid_ratio{{data_source="{source}", data_type="{type}"}} {metrics["invalid_ratio"]}'
            )
            
            # 记录数
            metrics_lines.append(
                f'rqa2025_data_record_count{{data_source="{source}", data_type="{type}"}} {metrics["record_count"]}'
            )
        
        return "\n".join(metrics_lines)


# 全局质量指标服务实例
_quality_metrics_service_instance = None


def get_quality_metrics_service() -> QualityMetricsService:
    """
    获取数据质量指标服务实例
    
    Returns:
        QualityMetricsService实例
    """
    global _quality_metrics_service_instance
    if _quality_metrics_service_instance is None:
        _quality_metrics_service_instance = QualityMetricsService()
    return _quality_metrics_service_instance

def reset_quality_metrics_service():
    """
    重置数据质量指标服务实例
    """
    global _quality_metrics_service_instance
    _quality_metrics_service_instance = None
    logger.info("🔄 数据质量指标服务实例已重置")

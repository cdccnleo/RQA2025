#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择历史记录系统 - PostgreSQL + 文件系统双存储方案

记录和管理特征选择的历史记录，包括：
- 特征选择操作记录
- 选择结果保存
- 历史查询和回放
- 选择策略评估

存储方案：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统 (JSON)
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionRecord:
    """特征选择记录"""
    selection_id: str
    task_id: str
    timestamp: float
    datetime: str
    
    # 输入特征
    input_features: List[str] = field(default_factory=list)
    input_feature_count: int = 0
    
    # 选择参数
    selection_method: str = ""  # 选择方法: correlation, importance, etc.
    selection_params: Dict[str, Any] = field(default_factory=dict)
    
    # 选择结果
    selected_features: List[str] = field(default_factory=list)
    selected_feature_count: int = 0
    selection_ratio: float = 0.0  # 选择比例
    
    # 评估指标
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 性能指标
    processing_time: float = 0.0
    
    # 备注
    notes: str = ""


class FeatureSelectorHistoryManager:
    """
    特征选择历史管理器 - 支持 PostgreSQL + 文件系统双存储
    
    管理特征选择的历史记录，支持持久化和查询
    优先使用 PostgreSQL，失败时降级到文件系统
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        初始化历史管理器
        
        Args:
            max_history_size: 最大历史记录数量
        """
        self.max_history_size = max_history_size
        self._history: List[FeatureSelectionRecord] = []
        self._lock = threading.Lock()
        
        # 持久化文件路径（降级存储）
        self._history_file = "data/feature_selection_history.json"
        
        # PostgreSQL 配置
        self._pg_config = self._get_postgresql_config()
        
        # 加载历史记录（优先从 PostgreSQL，失败则从文件系统）
        self._load_history()
        
        logger.info(f"特征选择历史管理器已初始化，当前历史记录: {len(self._history)} 条")
    
    def _get_postgresql_config(self) -> Dict[str, str]:
        """获取 PostgreSQL 配置"""
        return {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
            "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "SecurePass123!")
        }
    
    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self._pg_config["host"],
                port=self._pg_config["port"],
                database=self._pg_config["database"],
                user=self._pg_config["user"],
                password=self._pg_config["password"]
            )
            return conn
        except Exception as e:
            logger.debug(f"PostgreSQL 连接失败: {e}")
            return None
    
    def _save_to_postgresql(self, record: FeatureSelectionRecord) -> bool:
        """保存记录到 PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_selection_history (
                        selection_id, task_id, timestamp, input_features,
                        input_feature_count, selection_method, selection_params,
                        selected_features, selected_feature_count, selection_ratio,
                        evaluation_metrics, processing_time, notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (selection_id) DO UPDATE SET
                        task_id = EXCLUDED.task_id,
                        timestamp = EXCLUDED.timestamp,
                        input_features = EXCLUDED.input_features,
                        input_feature_count = EXCLUDED.input_feature_count,
                        selection_method = EXCLUDED.selection_method,
                        selection_params = EXCLUDED.selection_params,
                        selected_features = EXCLUDED.selected_features,
                        selected_feature_count = EXCLUDED.selected_feature_count,
                        selection_ratio = EXCLUDED.selection_ratio,
                        evaluation_metrics = EXCLUDED.evaluation_metrics,
                        processing_time = EXCLUDED.processing_time,
                        notes = EXCLUDED.notes
                """, (
                    record.selection_id,
                    record.task_id,
                    datetime.fromtimestamp(record.timestamp),
                    json.dumps(record.input_features),
                    record.input_feature_count,
                    record.selection_method,
                    json.dumps(record.selection_params),
                    json.dumps(record.selected_features),
                    record.selected_feature_count,
                    record.selection_ratio,
                    json.dumps(record.evaluation_metrics),
                    record.processing_time,
                    record.notes
                ))
            conn.commit()
            logger.debug(f"记录已保存到 PostgreSQL: {record.selection_id}")
            return True
            
        except Exception as e:
            logger.debug(f"保存到 PostgreSQL 失败: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def _load_from_postgresql(self) -> List[FeatureSelectionRecord]:
        """从 PostgreSQL 加载历史记录"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT selection_id, task_id, timestamp, input_features,
                           input_feature_count, selection_method, selection_params,
                           selected_features, selected_feature_count, selection_ratio,
                           evaluation_metrics, processing_time, notes
                    FROM feature_selection_history
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (self.max_history_size,))
                
                records = []
                for row in cur.fetchall():
                    record = FeatureSelectionRecord(
                        selection_id=row[0],
                        task_id=row[1],
                        timestamp=row[2].timestamp(),
                        datetime=row[2].isoformat(),
                        input_features=json.loads(row[3]) if row[3] else [],
                        input_feature_count=row[4],
                        selection_method=row[5] or "",
                        selection_params=json.loads(row[6]) if row[6] else {},
                        selected_features=json.loads(row[7]) if row[7] else [],
                        selected_feature_count=row[8],
                        selection_ratio=row[9] or 0.0,
                        evaluation_metrics=json.loads(row[10]) if row[10] else {},
                        processing_time=row[11] or 0.0,
                        notes=row[12] or ""
                    )
                    records.append(record)
                
                logger.info(f"从 PostgreSQL 加载了 {len(records)} 条历史记录")
                return records
                
        except Exception as e:
            logger.debug(f"从 PostgreSQL 加载失败: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _save_to_filesystem(self):
        """保存历史记录到文件系统（降级存储）"""
        try:
            os.makedirs(os.path.dirname(self._history_file), exist_ok=True)
            
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [asdict(r) for r in self._history],
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            logger.debug(f"记录已保存到文件系统: {self._history_file}")
                
        except Exception as e:
            logger.error(f"保存到文件系统失败: {e}")
    
    def _load_from_filesystem(self) -> List[FeatureSelectionRecord]:
        """从文件系统加载历史记录（降级加载）"""
        try:
            if not os.path.exists(self._history_file):
                return []
            
            with open(self._history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = [FeatureSelectionRecord(**r) for r in data]
            logger.info(f"从文件系统加载了 {len(records)} 条历史记录")
            return records
            
        except Exception as e:
            logger.error(f"从文件系统加载失败: {e}")
            return []
    
    def _sync_to_postgresql(self):
        """将文件系统的数据同步到 PostgreSQL"""
        try:
            for record in self._history:
                self._save_to_postgresql(record)
            logger.info(f"已将 {len(self._history)} 条记录同步到 PostgreSQL")
        except Exception as e:
            logger.debug(f"同步到 PostgreSQL 失败: {e}")
    
    def _save_history(self):
        """保存历史记录（双存储）"""
        # 保存到 PostgreSQL（主存储）
        if self._history:
            self._save_to_postgresql(self._history[-1])
        
        # 保存到文件系统（降级存储）
        self._save_to_filesystem()
    
    def _load_history(self):
        """加载历史记录（优先 PostgreSQL，降级文件系统）"""
        # 优先从 PostgreSQL 加载
        records = self._load_from_postgresql()
        
        if records:
            self._history = records
            # 同步到文件系统作为备份
            self._save_to_filesystem()
        else:
            # PostgreSQL 不可用，从文件系统加载
            logger.warning("PostgreSQL 不可用，从文件系统加载历史记录")
            self._history = self._load_from_filesystem()
    
    def record_selection(
        self,
        task_id: str,
        input_features: List[str],
        selected_features: List[str],
        selection_method: str = "",
        selection_params: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        processing_time: float = 0.0,
        notes: str = ""
    ) -> FeatureSelectionRecord:
        """
        记录特征选择操作
        
        Args:
            task_id: 任务ID
            input_features: 输入特征列表
            selected_features: 选择的特征列表
            selection_method: 选择方法
            selection_params: 选择参数
            evaluation_metrics: 评估指标
            processing_time: 处理时间
            notes: 备注
            
        Returns:
            选择记录
        """
        try:
            with self._lock:
                # 生成选择ID
                selection_id = f"sel_{int(datetime.now().timestamp() * 1000)}"
                
                # 创建记录
                record = FeatureSelectionRecord(
                    selection_id=selection_id,
                    task_id=task_id,
                    timestamp=datetime.now().timestamp(),
                    datetime=datetime.now().isoformat(),
                    input_features=input_features,
                    input_feature_count=len(input_features),
                    selection_method=selection_method,
                    selection_params=selection_params or {},
                    selected_features=selected_features,
                    selected_feature_count=len(selected_features),
                    selection_ratio=len(selected_features) / len(input_features) if input_features else 0.0,
                    evaluation_metrics=evaluation_metrics or {},
                    processing_time=processing_time,
                    notes=notes
                )
                
                # 添加到历史
                self._history.append(record)
                
                # 限制历史大小
                if len(self._history) > self.max_history_size:
                    self._history = self._history[-self.max_history_size:]
                
                # 持久化（双存储）
                self._save_history()
                
                logger.info(f"特征选择记录已保存: {selection_id}, 任务: {task_id}, "
                           f"选择 {record.selected_feature_count}/{record.input_feature_count} 个特征")
                
                return record
                
        except Exception as e:
            logger.error(f"保存特征选择记录失败: {e}")
            raise
    
    def get_selection_history(
        self,
        task_id: Optional[str] = None,
        selection_method: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        获取选择历史
        
        Args:
            task_id: 任务ID过滤
            selection_method: 选择方法过滤
            limit: 返回记录数量限制
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            选择历史记录列表
        """
        try:
            # 优先从 PostgreSQL 查询
            conn = self._get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        query = """
                            SELECT selection_id, task_id, timestamp, input_features,
                                   input_feature_count, selection_method, selection_params,
                                   selected_features, selected_feature_count, selection_ratio,
                                   evaluation_metrics, processing_time, notes
                            FROM feature_selection_history
                            WHERE 1=1
                        """
                        params = []
                        
                        if task_id:
                            query += " AND task_id = %s"
                            params.append(task_id)
                        
                        if selection_method:
                            query += " AND selection_method = %s"
                            params.append(selection_method)
                        
                        if start_time:
                            query += " AND timestamp >= %s"
                            params.append(datetime.fromtimestamp(start_time))
                        
                        if end_time:
                            query += " AND timestamp <= %s"
                            params.append(datetime.fromtimestamp(end_time))
                        
                        query += " ORDER BY timestamp DESC LIMIT %s"
                        params.append(limit)
                        
                        cur.execute(query, params)
                        
                        records = []
                        for row in cur.fetchall():
                            records.append({
                                "selection_id": row[0],
                                "task_id": row[1],
                                "timestamp": row[2].timestamp(),
                                "datetime": row[2].isoformat(),
                                "input_features": json.loads(row[3]) if row[3] else [],
                                "input_feature_count": row[4],
                                "selection_method": row[5] or "",
                                "selection_params": json.loads(row[6]) if row[6] else {},
                                "selected_features": json.loads(row[7]) if row[7] else [],
                                "selected_feature_count": row[8],
                                "selection_ratio": row[9] or 0.0,
                                "evaluation_metrics": json.loads(row[10]) if row[10] else {},
                                "processing_time": row[11] or 0.0,
                                "notes": row[12] or ""
                            })
                        
                        logger.debug(f"从 PostgreSQL 查询到 {len(records)} 条记录")
                        return records
                        
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.debug(f"从 PostgreSQL 查询失败，降级到内存查询: {e}")
        
        # PostgreSQL 不可用，从内存查询
        try:
            with self._lock:
                records = self._history
                
                # 应用过滤条件
                if task_id:
                    records = [r for r in records if r.task_id == task_id]
                
                if selection_method:
                    records = [r for r in records if r.selection_method == selection_method]
                
                if start_time:
                    records = [r for r in records if r.timestamp >= start_time]
                
                if end_time:
                    records = [r for r in records if r.timestamp <= end_time]
                
                # 按时间倒序
                records = sorted(records, key=lambda x: x.timestamp, reverse=True)
                
                # 限制数量
                records = records[:limit]
                
                # 转换为字典
                return [asdict(r) for r in records]
                
        except Exception as e:
            logger.error(f"获取选择历史失败: {e}")
            return []
    
    def get_selection_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        获取选择统计信息
        
        Args:
            days: 统计天数
            
        Returns:
            统计信息
        """
        try:
            with self._lock:
                cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
                recent_records = [r for r in self._history if r.timestamp >= cutoff_time]
                
                if not recent_records:
                    return {
                        "period_days": days,
                        "total_selections": 0,
                        "message": "该时间段内无选择记录"
                    }
                
                # 统计信息
                stats = {
                    "period_days": days,
                    "total_selections": len(recent_records),
                    "avg_selection_ratio": sum(r.selection_ratio for r in recent_records) / len(recent_records),
                    "avg_processing_time": sum(r.processing_time for r in recent_records) / len(recent_records),
                    "method_distribution": {},
                    "feature_count_trend": []
                }
                
                # 方法分布
                for record in recent_records:
                    method = record.selection_method or "unknown"
                    stats["method_distribution"][method] = stats["method_distribution"].get(method, 0) + 1
                
                # 特征数量趋势（最近10次）
                for record in recent_records[:10]:
                    stats["feature_count_trend"].append({
                        "timestamp": record.timestamp,
                        "datetime": record.datetime,
                        "input_count": record.input_feature_count,
                        "selected_count": record.selected_feature_count
                    })
                
                return stats
                
        except Exception as e:
            logger.error(f"获取选择统计失败: {e}")
            return {"error": str(e)}
    
    def get_feature_importance_ranking(
        self,
        feature_names: Optional[List[str]] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取特征重要性排名
        
        Args:
            feature_names: 特征名称列表过滤
            days: 统计天数
            
        Returns:
            特征重要性排名
        """
        try:
            with self._lock:
                cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
                recent_records = [r for r in self._history if r.timestamp >= cutoff_time]
                
                # 统计特征被选择的次数
                feature_scores = {}
                
                for record in recent_records:
                    for feature in record.selected_features:
                        if feature_names and feature not in feature_names:
                            continue
                        
                        if feature not in feature_scores:
                            feature_scores[feature] = {
                                "feature_name": feature,
                                "selected_count": 0,
                                "selection_records": []
                            }
                        
                        feature_scores[feature]["selected_count"] += 1
                        feature_scores[feature]["selection_records"].append({
                            "selection_id": record.selection_id,
                            "task_id": record.task_id,
                            "timestamp": record.timestamp,
                            "method": record.selection_method
                        })
                
                # 排序
                ranked_features = sorted(
                    feature_scores.values(),
                    key=lambda x: x["selected_count"],
                    reverse=True
                )
                
                return ranked_features
                
        except Exception as e:
            logger.error(f"获取特征重要性排名失败: {e}")
            return []
    
    def clear_history(self, days: Optional[int] = None) -> int:
        """
        清理历史记录
        
        Args:
            days: 清理多少天前的记录，None表示清理所有
            
        Returns:
            清理的记录数量
        """
        try:
            with self._lock:
                if days is None:
                    count = len(self._history)
                    self._history = []
                    
                    # 清理 PostgreSQL
                    conn = self._get_db_connection()
                    if conn:
                        try:
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM feature_selection_history")
                            conn.commit()
                        finally:
                            conn.close()
                else:
                    cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
                    old_count = len(self._history)
                    self._history = [r for r in self._history if r.timestamp >= cutoff_time]
                    count = old_count - len(self._history)
                    
                    # 清理 PostgreSQL
                    conn = self._get_db_connection()
                    if conn:
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "DELETE FROM feature_selection_history WHERE timestamp < %s",
                                    (datetime.fromtimestamp(cutoff_time),)
                                )
                            conn.commit()
                        finally:
                            conn.close()
                
                # 保存到文件系统
                self._save_to_filesystem()
                
                logger.info(f"清理了 {count} 条历史记录")
                return count
                
        except Exception as e:
            logger.error(f"清理历史记录失败: {e}")
            return 0


# 全局历史管理器实例
_history_manager: Optional[FeatureSelectorHistoryManager] = None


def get_feature_selector_history_manager() -> FeatureSelectorHistoryManager:
    """
    获取全局特征选择历史管理器实例
    
    Returns:
        特征选择历史管理器实例
    """
    global _history_manager
    if _history_manager is None:
        _history_manager = FeatureSelectorHistoryManager()
    return _history_manager

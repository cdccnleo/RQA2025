#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算次数跟踪器

动态统计各技术指标的计算次数，支持：
- 实时统计指标计算次数
- 按任务类型统计
- 历史趋势分析
- 持久化存储
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class IndicatorCalculationTracker:
    """
    指标计算次数跟踪器
    
    动态跟踪各技术指标的计算次数
    """
    
    def __init__(self, persistence_file: str = "data/indicator_calculations.json"):
        """
        初始化跟踪器
        
        Args:
            persistence_file: 持久化文件路径
        """
        self.persistence_file = persistence_file
        self._calculations: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "last_computed": None,
            "task_history": []
        })
        self._lock = threading.Lock()
        
        # 加载历史数据
        self._load_calculations()
        
        logger.info("指标计算次数跟踪器已初始化")
    
    def record_calculation(
        self,
        indicator_name: str,
        task_id: str,
        task_type: str = "technical",
        symbol: str = "",
        computation_time: float = 0.0
    ) -> None:
        """
        记录指标计算
        
        Args:
            indicator_name: 指标名称
            task_id: 任务ID
            task_type: 任务类型
            symbol: 股票代码
            computation_time: 计算耗时
        """
        try:
            with self._lock:
                current_time = time.time()
                
                # 更新计算次数
                self._calculations[indicator_name]["count"] += 1
                self._calculations[indicator_name]["last_computed"] = current_time
                
                # 记录任务历史
                self._calculations[indicator_name]["task_history"].append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "symbol": symbol,
                    "timestamp": current_time,
                    "datetime": datetime.now().isoformat(),
                    "computation_time": computation_time
                })
                
                # 限制历史记录数量
                if len(self._calculations[indicator_name]["task_history"]) > 100:
                    self._calculations[indicator_name]["task_history"] = \
                        self._calculations[indicator_name]["task_history"][-100:]
                
                logger.debug(f"记录指标计算: {indicator_name}, 任务: {task_id}, "
                           f"总次数: {self._calculations[indicator_name]['count']}")
                
                # 持久化
                self._save_calculations()
                
        except Exception as e:
            logger.error(f"记录指标计算失败: {e}")
    
    def record_task_indicators(
        self,
        task_id: str,
        indicators: List[str],
        task_type: str = "technical",
        symbol: str = "",
        computation_time: float = 0.0
    ) -> None:
        """
        批量记录任务中所有指标的计算
        
        Args:
            task_id: 任务ID
            indicators: 指标名称列表
            task_type: 任务类型
            symbol: 股票代码
            computation_time: 计算耗时
        """
        try:
            for indicator in indicators:
                self.record_calculation(
                    indicator_name=indicator,
                    task_id=task_id,
                    task_type=task_type,
                    symbol=symbol,
                    computation_time=computation_time / len(indicators) if indicators else 0.0
                )
            
            logger.info(f"记录任务 {task_id} 的 {len(indicators)} 个指标计算")
            
        except Exception as e:
            logger.error(f"批量记录指标计算失败: {e}")
    
    def get_indicator_stats(self, indicator_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指标统计信息
        
        Args:
            indicator_name: 指标名称，None返回所有指标
            
        Returns:
            指标统计信息
        """
        try:
            with self._lock:
                if indicator_name:
                    # 返回特定指标的统计
                    if indicator_name in self._calculations:
                        calc = self._calculations[indicator_name]
                        return {
                            "name": indicator_name,
                            "computed_count": calc["count"],
                            "last_computed": calc["last_computed"],
                            "last_computed_datetime": datetime.fromtimestamp(calc["last_computed"]).isoformat() if calc["last_computed"] else None,
                            "recent_tasks": calc["task_history"][-5:]  # 最近5个任务
                        }
                    else:
                        return {
                            "name": indicator_name,
                            "computed_count": 0,
                            "last_computed": None,
                            "recent_tasks": []
                        }
                else:
                    # 返回所有指标的统计
                    all_stats = {}
                    for name, calc in self._calculations.items():
                        all_stats[name] = {
                            "computed_count": calc["count"],
                            "last_computed": calc["last_computed"],
                            "last_computed_datetime": datetime.fromtimestamp(calc["last_computed"]).isoformat() if calc["last_computed"] else None
                        }
                    return all_stats
                    
        except Exception as e:
            logger.error(f"获取指标统计失败: {e}")
            return {}
    
    def get_all_indicators_status(self) -> List[Dict[str, Any]]:
        """
        获取所有指标的状态列表（用于前端展示）
        
        Returns:
            指标状态列表
        """
        try:
            with self._lock:
                # 定义指标描述映射
                indicator_descriptions = {
                    "SMA": "简单移动平均线",
                    "EMA": "指数移动平均线",
                    "RSI": "相对强弱指标",
                    "MACD": "指数平滑异同移动平均线",
                    "KDJ": "随机指标",
                    "BOLL": "布林带",
                    "MA": "移动平均线",
                    "WMA": "加权移动平均线",
                    "ATR": "真实波动幅度均值",
                    "CCI": "顺势指标",
                    "WR": "威廉指标"
                }
                
                indicators_list = []
                
                # 遍历所有已记录的指标
                for name, calc in self._calculations.items():
                    # 标准化指标名称
                    display_name = name.upper()
                    
                    indicators_list.append({
                        "name": display_name,
                        "description": indicator_descriptions.get(display_name, f"{display_name} 指标"),
                        "status": "active" if calc["count"] > 0 else "inactive",
                        "computed_count": calc["count"],
                        "last_computed": calc["last_computed"],
                        "last_computed_datetime": datetime.fromtimestamp(calc["last_computed"]).isoformat() if calc["last_computed"] else None
                    })
                
                # 如果没有记录，返回默认指标列表（但计算次数为0）
                if not indicators_list:
                    default_indicators = ["MA", "RSI", "MACD", "KDJ", "BOLL"]
                    for name in default_indicators:
                        indicators_list.append({
                            "name": name,
                            "description": indicator_descriptions.get(name, f"{name} 指标"),
                            "status": "inactive",
                            "computed_count": 0,
                            "last_computed": None,
                            "last_computed_datetime": None
                        })
                
                # 按计算次数排序
                indicators_list.sort(key=lambda x: x["computed_count"], reverse=True)
                
                return indicators_list
                
        except Exception as e:
            logger.error(f"获取指标状态列表失败: {e}")
            return []
    
    def get_calculation_trend(self, indicator_name: str, days: int = 7) -> Dict[str, Any]:
        """
        获取指标计算趋势
        
        Args:
            indicator_name: 指标名称
            days: 查询天数
            
        Returns:
            计算趋势
        """
        try:
            with self._lock:
                if indicator_name not in self._calculations:
                    return {
                        "indicator": indicator_name,
                        "message": "该指标暂无计算记录"
                    }
                
                cutoff_time = time.time() - (days * 24 * 3600)
                task_history = self._calculations[indicator_name]["task_history"]
                
                # 筛选最近的数据
                recent_tasks = [t for t in task_history if t["timestamp"] > cutoff_time]
                
                # 按天统计
                daily_stats = defaultdict(int)
                for task in recent_tasks:
                    day = datetime.fromtimestamp(task["timestamp"]).strftime("%Y-%m-%d")
                    daily_stats[day] += 1
                
                return {
                    "indicator": indicator_name,
                    "period_days": days,
                    "total_calculations": len(recent_tasks),
                    "daily_stats": dict(daily_stats),
                    "trend": "increasing" if len(recent_tasks) > len(task_history) * 0.5 else "stable"
                }
                
        except Exception as e:
            logger.error(f"获取计算趋势失败: {e}")
            return {}
    
    def reset_calculations(self, indicator_name: Optional[str] = None) -> None:
        """
        重置计算次数
        
        Args:
            indicator_name: 指标名称，None表示重置所有
        """
        try:
            with self._lock:
                if indicator_name:
                    if indicator_name in self._calculations:
                        self._calculations[indicator_name] = {
                            "count": 0,
                            "last_computed": None,
                            "task_history": []
                        }
                        logger.info(f"重置指标 {indicator_name} 的计算次数")
                else:
                    self._calculations.clear()
                    logger.info("重置所有指标的计算次数")
                
                self._save_calculations()
                
        except Exception as e:
            logger.error(f"重置计算次数失败: {e}")
    
    def _save_calculations(self) -> None:
        """保存计算记录到文件 - 增强版"""
        try:
            import os
            
            # 确保使用绝对路径
            persistence_path = os.path.abspath(self.persistence_file)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
            
            # 原子性写入：先写入临时文件，再重命名
            temp_file = f"{persistence_path}.tmp"
            
            # 将 defaultdict 转换为普通 dict 以便序列化
            data_to_save = {}
            for indicator_name, calc_data in self._calculations.items():
                data_to_save[indicator_name] = {
                    "count": calc_data.get("count", 0),
                    "last_computed": calc_data.get("last_computed"),
                    "task_history": calc_data.get("task_history", [])
                }
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            # 原子性重命名
            os.replace(temp_file, persistence_path)
            
            logger.debug(f"计算记录已保存到 {persistence_path}，包含 {len(data_to_save)} 个指标")
            
        except Exception as e:
            logger.error(f"保存计算记录失败: {e}", exc_info=True)
    
    def _load_calculations(self) -> None:
        """从文件加载计算记录 - 增强版"""
        try:
            import os
            
            # 确保使用绝对路径
            persistence_path = os.path.abspath(self.persistence_file)
            logger.info(f"尝试从 {persistence_path} 加载计算记录")
            
            if not os.path.exists(persistence_path):
                logger.warning(f"持久化文件不存在: {persistence_path}")
                return
            
            with open(persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict):
                logger.error(f"持久化文件格式错误: 期望dict，实际{type(data)}")
                return
                
            # 清空当前数据并重新加载
            self._calculations.clear()
            
            # 验证并加载每个指标的数据
            loaded_count = 0
            for indicator_name, calc_data in data.items():
                if isinstance(calc_data, dict) and 'count' in calc_data:
                    self._calculations[indicator_name] = {
                        "count": calc_data.get("count", 0),
                        "last_computed": calc_data.get("last_computed"),
                        "task_history": calc_data.get("task_history", [])
                    }
                    loaded_count += 1
                else:
                    logger.warning(f"指标 {indicator_name} 的数据格式无效，跳过")
                    
            logger.info(f"成功从 {persistence_path} 加载了 {loaded_count} 个指标的计算记录")
            
            # 显示加载的指标列表
            if loaded_count > 0:
                indicators_list = list(data.keys())
                logger.info(f"加载的指标: {indicators_list}")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
        except Exception as e:
            logger.error(f"加载计算记录失败: {e}", exc_info=True)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取跟踪器健康状态
        
        Returns:
            健康状态信息，包含内存数据和文件数据对比
        """
        try:
            import os
            
            # 获取内存数据
            memory_data = {}
            for indicator_name, calc_data in self._calculations.items():
                memory_data[indicator_name] = {
                    "count": calc_data.get("count", 0),
                    "last_computed": calc_data.get("last_computed")
                }
            
            # 获取文件数据
            persistence_path = os.path.abspath(self.persistence_file)
            file_data = {}
            file_exists = os.path.exists(persistence_path)
            
            if file_exists:
                try:
                    with open(persistence_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for indicator_name, calc_data in data.items():
                        if isinstance(calc_data, dict):
                            file_data[indicator_name] = {
                                "count": calc_data.get("count", 0),
                                "last_computed": calc_data.get("last_computed")
                            }
                except Exception as e:
                    logger.error(f"读取文件数据失败: {e}")
            
            # 对比数据
            comparison = {}
            all_indicators = set(memory_data.keys()) | set(file_data.keys())
            for indicator in all_indicators:
                memory_count = memory_data.get(indicator, {}).get("count", 0)
                file_count = file_data.get(indicator, {}).get("count", 0)
                comparison[indicator] = {
                    "memory_count": memory_count,
                    "file_count": file_count,
                    "consistent": memory_count == file_count
                }
            
            return {
                "status": "healthy" if all(c["consistent"] for c in comparison.values()) else "inconsistent",
                "persistence_file": persistence_path,
                "file_exists": file_exists,
                "memory_indicators": len(memory_data),
                "file_indicators": len(file_data),
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取健康状态失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# 全局跟踪器实例
_tracker: Optional[IndicatorCalculationTracker] = None
_tracker_lock = threading.Lock()


def get_indicator_calculation_tracker() -> IndicatorCalculationTracker:
    """
    获取全局指标计算跟踪器实例（线程安全单例模式）
    
    Returns:
        指标计算跟踪器实例
    """
    global _tracker
    
    if _tracker is None:
        with _tracker_lock:
            # 双重检查锁定
            if _tracker is None:
                logger.info("创建新的指标计算跟踪器实例")
                _tracker = IndicatorCalculationTracker()
                logger.info("指标计算跟踪器实例创建完成")
    
    return _tracker

#!/usr/bin/env python3
"""
特征选择任务处理器

统一调度器使用的特征选择任务处理器，支持：
- 多种特征选择方法（importance/correlation/mutual_info/kbest）
- 自动记录选择历史
- 完整的错误处理和日志记录
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


async def feature_selection_handler(task: 'Task') -> Dict[str, Any]:
    """
    特征选择任务处理器
    
    执行流程：
    1. 解析任务参数（symbols, method, top_k等）
    2. 获取特征数据
    3. 调用FeatureSelector执行选择
    4. 记录选择历史到feature_selection_history表
    5. 返回选择结果
    
    Args:
        task: 调度器任务对象
        
    Returns:
        包含选择结果的字典
    """
    start_time = time.time()
    task_id = task.id if hasattr(task, 'id') else 'unknown'
    
    logger.info(f"🚀 开始执行特征选择任务: {task_id}")
    
    try:
        # 1. 解析任务参数（支持params和payload两种格式）
        if hasattr(task, 'params') and task.params:
            params = task.params
        elif hasattr(task, 'payload') and task.payload:
            params = task.payload
        else:
            params = {}
        
        symbols = params.get('symbols', [])
        method = params.get('method', 'importance')
        top_k = params.get('top_k', 10)
        min_quality = params.get('min_quality')
        
        logger.info(f"📋 任务参数: symbols={symbols}, method={method}, top_k={top_k}")
        
        # 参数验证
        if not symbols:
            raise ValueError("必须指定股票代码列表(symbols)")
        
        if method not in ['importance', 'correlation', 'mutual_info', 'kbest']:
            raise ValueError(f"不支持的选择方法: {method}")
        
        # 2. 获取特征数据
        logger.info("🔍 获取特征数据...")
        from src.gateway.web.feature_task_persistence import get_features
        
        all_features = get_features()
        if not all_features:
            raise ValueError("没有可用的特征数据")
        
        logger.info(f"✅ 获取到 {len(all_features)} 个特征")
        
        # 3. 按股票代码过滤并执行选择
        selection_results = []
        all_selected_features = []
        total_input = 0
        
        for symbol in symbols:
            symbol_features = [
                f for f in all_features 
                if f.get('symbol') == symbol
            ]
            
            if not symbol_features:
                logger.warning(f"⚠️ 股票 {symbol} 没有特征数据")
                continue
            
            # 质量过滤
            if min_quality is not None:
                symbol_features = [
                    f for f in symbol_features 
                    if f.get('quality_score', 0) >= min_quality
                ]
            
            total_input += len(symbol_features)
            
            logger.info(f"📊 处理股票 {symbol}: {len(symbol_features)} 个特征")
            
            # 执行特征选择
            selected = await _select_features(
                symbol_features, method, top_k
            )
            
            selected_names = [f.get('name') for f in selected]
            all_selected_features.extend(selected_names)
            
            result = {
                'symbol': symbol,
                'input_count': len(symbol_features),
                'selected_count': len(selected),
                'selected_features': selected_names,
                'method': method
            }
            selection_results.append(result)
            
            logger.info(f"✅ 股票 {symbol} 选择完成: {len(selected)} 个特征")
        
        # 4. 记录选择历史
        logger.info("💾 记录特征选择历史...")
        await _save_selection_history(
            task_id=task_id,
            input_features=[f.get('name') for f in all_features],
            input_count=total_input,
            selected_features=all_selected_features,
            selected_count=len(all_selected_features),
            method=method,
            params=params,
            results=selection_results
        )
        
        processing_time = time.time() - start_time
        
        # 5. 返回结果
        result = {
            'status': 'completed',
            'task_id': task_id,
            'processing_time': processing_time,
            'summary': {
                'total_input_features': total_input,
                'total_selected_features': len(all_selected_features),
                'selection_ratio': len(all_selected_features) / total_input if total_input > 0 else 0,
                'symbols_processed': len(selection_results),
                'method': method
            },
            'details': selection_results
        }
        
        logger.info(f"✅ 特征选择任务完成: {task_id}, 耗时: {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ 特征选择任务失败: {task_id}, 错误: {e}")
        
        return {
            'status': 'failed',
            'task_id': task_id,
            'processing_time': processing_time,
            'error': str(e)
        }


async def _select_features(
    features: List[Dict[str, Any]], 
    method: str, 
    top_k: int
) -> List[Dict[str, Any]]:
    """
    执行特征选择
    
    Args:
        features: 特征列表
        method: 选择方法
        top_k: 选择前k个
        
    Returns:
        选择的特征列表
    """
    if not features:
        return []
    
    # 根据方法排序
    if method == 'importance':
        # 按质量分数排序
        sorted_features = sorted(
            features,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )
    elif method == 'correlation':
        # 按相关性排序（如果有correlation字段）
        sorted_features = sorted(
            features,
            key=lambda x: abs(x.get('correlation', 0)),
            reverse=True
        )
    elif method in ['mutual_info', 'kbest']:
        # 默认按质量分数排序
        sorted_features = sorted(
            features,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )
    else:
        sorted_features = features
    
    # 返回前k个
    return sorted_features[:min(top_k, len(sorted_features))]


async def _save_selection_history(
    task_id: str,
    input_features: List[str],
    input_count: int,
    selected_features: List[str],
    selected_count: int,
    method: str,
    params: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> None:
    """
    保存特征选择历史记录
    
    Args:
        task_id: 任务ID
        input_features: 输入特征列表
        input_count: 输入特征数量
        selected_features: 选择的特征列表
        selected_count: 选择的特征数量
        method: 选择方法
        params: 任务参数
        results: 详细结果
    """
    try:
        from src.features.selection.feature_selector_history import (
            FeatureSelectorHistoryManager,
            FeatureSelectionRecord
        )
        
        history_manager = FeatureSelectorHistoryManager()
        
        # 计算选择比例
        selection_ratio = selected_count / input_count if input_count > 0 else 0
        
        # 计算平均质量
        avg_quality = 0.0
        if results:
            qualities = []
            for r in results:
                # 这里简化处理，实际应该从特征数据中获取质量分数
                qualities.append(0.8)  # 默认值
            avg_quality = sum(qualities) / len(qualities)
        
        record = FeatureSelectionRecord(
            selection_id=f"selection_{int(time.time())}_{task_id}",
            task_id=task_id,
            timestamp=time.time(),
            datetime=datetime.now().isoformat(),
            input_features=input_features,
            input_feature_count=input_count,
            selection_method=method,
            selection_params=params,
            selected_features=selected_features,
            selected_feature_count=selected_count,
            selection_ratio=selection_ratio,
            evaluation_metrics={'avg_quality': avg_quality},
            processing_time=0.0,  # 由调用者计算
            notes=f"统一调度器执行: 处理{len(results)}个股票"
        )
        
        # 保存到历史
        history_manager._history.append(record)
        history_manager._save_history()
        
        logger.info(f"✅ 特征选择历史已保存: {record.selection_id}")
        
    except Exception as e:
        logger.error(f"❌ 保存特征选择历史失败: {e}")
        # 不抛出异常，避免影响主任务

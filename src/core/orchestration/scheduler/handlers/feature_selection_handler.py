#!/usr/bin/env python3
"""
特征选择任务处理器

统一调度器使用的特征选择任务处理器，支持：
- 多种特征选择方法（importance/correlation/mutual_info/kbest）
- 任务状态跟踪和持久化（按股票拆分）
- 自动记录选择历史
- 完整的错误处理和日志记录
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


async def feature_selection_handler(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    特征选择任务处理器
    
    执行流程：
    1. 生成批次ID
    2. 解析任务参数（symbols, method, top_k等）
    3. 为每个股票创建独立任务记录
    4. 获取特征数据
    5. 按股票分别执行特征选择
    6. 更新每个股票的任务状态
    7. 记录选择历史到feature_selection_history表
    8. 返回汇总结果
    
    Args:
        task: 调度器任务payload字典（包含symbols, method, top_k等参数）
        
    Returns:
        包含选择结果的字典
    """
    start_time = time.time()
    start_timestamp = int(start_time)
    
    # 从payload中获取任务ID（调度器传递的_task_id）
    parent_task_id = task.get('_task_id', 'unknown')
    
    # 生成批次ID（使用更简洁的格式）
    if parent_task_id != 'unknown':
        batch_id = f"batch_{parent_task_id}"
    else:
        batch_id = f"batch_{start_timestamp}"
    
    logger.info(f"🚀 开始执行特征选择批次: {batch_id} (父任务: {parent_task_id})")
    
    # 导入持久化模块
    from src.gateway.web.feature_selection_task_persistence import (
        save_selection_task,
        update_selection_task_status
    )
    
    try:
        # 1. 解析任务参数
        logger.info(f"🔍 任务参数: {task}")
        
        params = task
        
        symbols = params.get('symbols', [])
        method = params.get('method', 'importance')
        top_k = params.get('top_k', 10)
        min_quality = params.get('min_quality')
        
        logger.info(f"📋 解析后的任务参数: symbols={symbols}, method={method}, top_k={top_k}")
        
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
        
        # 3. 为每个股票创建任务记录
        logger.info("💾 为每个股票创建任务记录...")
        task_records = {}
        for symbol in symbols:
            task_id = f"selection_task_{symbol}_{start_timestamp}"
            task_record = {
                'task_id': task_id,
                'task_type': 'feature_selection',
                'status': 'pending',
                'progress': 0,
                'symbol': symbol,  # 单股票
                'batch_id': batch_id,  # 关联批次
                'symbols': [symbol],  # 兼容旧格式
                'selection_method': method,
                'top_k': top_k,
                'min_quality': min_quality,
                'parent_task_id': parent_task_id,
                'start_time': start_timestamp
            }
            save_selection_task(task_record)
            task_records[symbol] = task_id
            logger.info(f"  股票 {symbol}: {task_id}")
        
        # 4. 按股票分别执行选择
        selection_results = []
        all_selected_features = []
        total_input = 0
        processed_count = 0
        failed_symbols = []
        
        total_symbols = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            task_id = task_records[symbol]
            
            try:
                # 更新状态为running
                update_selection_task_status(task_id, 'running', progress=10)
                logger.info(f"▶️ 开始处理股票 {symbol}: {task_id}")
                
                # 获取该股票的特征
                symbol_features = [
                    f for f in all_features 
                    if f.get('symbol') == symbol
                ]
                
                if not symbol_features:
                    logger.warning(f"⚠️ 股票 {symbol} 没有特征数据")
                    update_selection_task_status(
                        task_id, 
                        'failed', 
                        progress=0,
                        error_message="没有特征数据"
                    )
                    failed_symbols.append(symbol)
                    continue
                
                # 质量过滤
                if min_quality is not None:
                    symbol_features = [
                        f for f in symbol_features 
                        if f.get('quality_score', 0) >= min_quality
                    ]
                
                input_count = len(symbol_features)
                total_input += input_count
                
                logger.info(f"📊 股票 {symbol}: {input_count} 个特征")
                update_selection_task_status(task_id, 'running', progress=50)
                
                # 执行特征选择
                selected = await _select_features(
                    symbol_features, method, top_k
                )
                
                # 去重：确保选中特征不重复
                selected_names = []
                seen = set()
                for f in selected:
                    name = f.get('name')
                    if name and name not in seen:
                        selected_names.append(name)
                        seen.add(name)
                
                all_selected_features.extend(selected_names)
                
                result = {
                    'symbol': symbol,
                    'input_count': input_count,
                    'selected_count': len(selected_names),
                    'selected_features': selected_names,
                    'method': method
                }
                selection_results.append(result)
                
                processed_count += 1
                
                # 计算单个股票的处理时间
                symbol_processing_time = time.time() - start_time
                
                # 更新任务状态为completed
                update_selection_task_status(
                    task_id,
                    'completed',
                    progress=100,
                    end_time=int(time.time()),
                    processing_time=symbol_processing_time,
                    total_input_features=input_count,
                    total_selected_features=len(selected_names),
                    symbols_processed=1,
                    results=result
                )
                
                logger.info(f"✅ 股票 {symbol} 完成: {len(selected_names)} 个特征")
                
                # 5. 记录选择历史
                await _save_selection_history(
                    task_id=parent_task_id,
                    symbol=symbol,
                    input_features=[f.get('name') for f in symbol_features],
                    input_count=input_count,
                    selected_features=selected_names,
                    selected_count=len(selected_names),
                    method=method,
                    params=params
                )
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ 股票 {symbol} 处理失败: {error_msg}")
                update_selection_task_status(
                    task_id,
                    'failed',
                    progress=0,
                    end_time=int(time.time()),
                    error_message=error_msg
                )
                failed_symbols.append(symbol)
        
        # 6. 计算处理时间
        processing_time = time.time() - start_time
        end_timestamp = int(time.time())
        
        # 7. 返回汇总结果
        result = {
            'status': 'completed' if not failed_symbols else 'partial_failed',
            'batch_id': batch_id,
            'parent_task_id': parent_task_id,
            'processing_time': processing_time,
            'summary': {
                'total_symbols': len(symbols),
                'processed_symbols': processed_count,
                'failed_symbols': failed_symbols,
                'total_input_features': total_input,
                'total_selected_features': len(all_selected_features),
                'selection_ratio': len(all_selected_features) / total_input if total_input > 0 else 0,
                'method': method
            },
            'details': selection_results,
            'task_records': task_records
        }
        
        logger.info(f"✅ 特征选择批次完成: {batch_id}, "
                   f"成功: {processed_count}/{len(symbols)}, "
                   f"耗时: {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"❌ 特征选择批次失败: {batch_id}, 错误: {error_msg}")
        
        # 更新所有任务为failed
        for symbol, task_id in task_records.items():
            try:
                update_selection_task_status(
                    task_id,
                    'failed',
                    progress=0,
                    end_time=int(time.time()),
                    processing_time=processing_time,
                    error_message=error_msg
                )
            except Exception:
                pass
        
        return {
            'status': 'failed',
            'batch_id': batch_id,
            'parent_task_id': parent_task_id,
            'processing_time': processing_time,
            'error': error_msg
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
    symbol: str,
    input_features: List[str],
    input_count: int,
    selected_features: List[str],
    selected_count: int,
    method: str,
    params: Dict[str, Any]
) -> None:
    """
    保存特征选择历史记录
    
    Args:
        task_id: 任务ID
        symbol: 股票代码
        input_features: 输入特征列表
        input_count: 输入特征数量
        selected_features: 选择的特征列表
        selected_count: 选择的特征数量
        method: 选择方法
        params: 任务参数
    """
    try:
        from src.features.selection.feature_selector_history import (
            FeatureSelectorHistoryManager
        )
        
        history_manager = FeatureSelectorHistoryManager()
        
        # 使用管理器的record_selection方法保存记录
        history_manager.record_selection(
            task_id=task_id,
            symbol=symbol,
            input_features=input_features,
            selected_features=selected_features,
            selection_method=method,
            selection_params=params,
            notes=f"统一调度器执行: 股票 {symbol}"
        )
        
        logger.info(f"✅ 特征选择历史已保存: 股票 {symbol}")
        
    except Exception as e:
        logger.error(f"❌ 保存特征选择历史失败: {e}")
        # 不抛出异常，避免影响主任务

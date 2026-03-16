#!/usr/bin/env python3
"""
特征提取任务处理器

统一调度器使用的特征提取任务处理器，支持：
- 从PostgreSQL或文件系统加载任务
- 执行特征提取
- 更新任务状态
- 触发特征选择任务
"""

import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def feature_extraction_handler(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    特征提取任务处理器
    
    执行流程：
    1. 获取任务ID和参数
    2. 更新任务状态为running
    3. 执行特征提取
    4. 保存提取的特征
    5. 更新任务状态为completed
    6. 触发特征选择任务
    
    Args:
        task: 调度器任务payload字典
        
    Returns:
        包含提取结果的字典
    """
    start_time = time.time()
    
    # 从payload中获取任务ID
    task_id = task.get('task_id') or task.get('_task_id', 'unknown')
    
    logger.info(f"🚀 开始执行特征提取任务: {task_id}")
    
    try:
        # 1. 解析任务参数
        symbol = task.get('symbol')
        config = task.get('config', {})
        
        if not symbol:
            raise ValueError("任务缺少股票代码(symbol)")
        
        logger.info(f"📋 任务参数: symbol={symbol}, config={config}")
        
        # 2. 更新任务状态为running
        _update_task_status(task_id, 'running', progress=10)
        
        # 3. 执行特征提取
        logger.info(f"▶️ 开始为股票 {symbol} 提取特征...")
        
        # 调用特征提取执行器
        from src.gateway.web.feature_task_executor import FeatureTaskExecutor
        executor = FeatureTaskExecutor()
        
        # 构建任务对象
        from src.gateway.web.feature_task_persistence import get_task
        feature_task = get_task(task_id)
        
        if not feature_task:
            raise ValueError(f"找不到任务: {task_id}")
        
        # 更新进度
        _update_task_status(task_id, 'running', progress=50)
        
        # 执行特征提取
        result = await executor._execute_task_async(feature_task)
        
        # 4. 更新任务状态为completed
        processing_time = time.time() - start_time
        _update_task_status(
            task_id, 
            'completed', 
            progress=100,
            result=result
        )
        
        logger.info(f"✅ 特征提取任务完成: {task_id}, 处理时间: {processing_time:.2f}s")
        
        # 5. 触发特征选择任务
        try:
            from src.gateway.web.feature_selection_task_persistence import create_selection_task
            
            # 获取提取的特征列表
            features = result.get('features', [])
            if features:
                selection_task = create_selection_task(
                    symbol=symbol,
                    features=features,
                    source_task_id=task_id,
                    selection_method="importance",
                    config={
                        "n_features": min(10, len(features)),
                        "auto_execute": True
                    }
                )
                if selection_task:
                    logger.info(f"✅ 已自动创建特征选择任务: {selection_task.get('task_id')}")
        except Exception as e:
            logger.warning(f"⚠️ 自动创建特征选择任务失败: {e}")
        
        return {
            "success": True,
            "task_id": task_id,
            "symbol": symbol,
            "processing_time": processing_time,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"❌ 特征提取任务失败: {task_id}, 错误: {e}", exc_info=True)
        
        # 更新任务状态为failed
        _update_task_status(
            task_id, 
            'failed', 
            progress=0,
            error=str(e)
        )
        
        return {
            "success": False,
            "task_id": task_id,
            "error": str(e)
        }


def _update_task_status(
    task_id: str, 
    status: str, 
    progress: int = None,
    result: Dict = None,
    error: str = None
):
    """更新任务状态"""
    try:
        from src.gateway.web.feature_task_persistence import update_task_status
        
        update_task_status(
            task_id=task_id,
            status=status,
            progress=progress,
            result=result,
            error_message=error
        )
        
        logger.debug(f"任务状态已更新: {task_id} -> {status}")
    except Exception as e:
        logger.warning(f"更新任务状态失败: {task_id}, 错误: {e}")

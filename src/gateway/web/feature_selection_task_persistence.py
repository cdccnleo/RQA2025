

def create_selection_task(
    symbol: str,
    features: List[str],
    source_task_id: str,
    selection_method: str = "importance",
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    创建特征选择任务
    
    Args:
        symbol: 股票代码
        features: 特征列表
        source_task_id: 源任务ID（特征提取任务ID）
        selection_method: 选择方法，默认"importance"
        config: 配置参数，包括n_features（选择特征数）、auto_execute（是否自动执行）等
    
    Returns:
        创建的任务信息字典，失败返回None
    """
    try:
        import uuid
        
        task_id = f"selection_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        n_features = config.get("n_features", 10) if config else 10
        auto_execute = config.get("auto_execute", True) if config else True
        
        task = {
            "task_id": task_id,
            "task_type": "feature_selection",
            "status": "pending",
            "progress": 0,
            "symbol": symbol,
            "source_task_id": source_task_id,
            "selection_method": selection_method,
            "n_features": n_features,
            "auto_execute": auto_execute,
            "input_features": features,
            "total_input_features": len(features),
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # 保存任务
        if save_selection_task(task):
            logger.info(f"✅ 特征选择任务创建成功: {task_id}, 股票: {symbol}, 输入特征: {len(features)}")
            
            # 如果配置了自动执行，提交到调度器
            if auto_execute:
                try:
                    from src.core.orchestration.scheduler import get_unified_scheduler
                    scheduler = get_unified_scheduler()
                    if scheduler:
                        import asyncio
                        payload = {
                            "symbol": symbol,
                            "features": features,
                            "selection_method": selection_method,
                            "n_features": n_features,
                            "task_id": task_id
                        }
                        scheduler_task_id = asyncio.run(scheduler.submit_task(
                            task_type="feature_selection",
                            payload=payload,
                            priority=5
                        ))
                        logger.info(f"✅ 特征选择任务已提交到调度器: {scheduler_task_id}")
                        task["scheduler_task_id"] = scheduler_task_id
                except Exception as e:
                    logger.warning(f"⚠️ 提交特征选择任务到调度器失败: {e}")
            
            return task
        else:
            logger.error(f"❌ 保存特征选择任务失败: {task_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 创建特征选择任务失败: {e}", exc_info=True)
        return None

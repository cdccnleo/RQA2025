#!/usr/bin/env python3
"""注册数据采集器工作节点"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def register_data_collector():
    """注册数据采集器工作节点"""
    try:
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        registry = get_unified_worker_registry()
        
        # 注册数据采集器
        worker_id = "data_collector_001"
        capabilities = ["akshare", "baostock", "tushare"]
        
        registry.register_worker(
            worker_type=WorkerType.DATA_COLLECTOR,
            worker_id=worker_id,
            capabilities=capabilities,
            metadata={
                "version": "1.0.0",
                "max_concurrent_tasks": 5
            }
        )
        
        logger.info(f"✅ 数据采集器工作节点已注册: {worker_id}")
        
        # 验证注册
        workers = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
        logger.info(f"当前数据采集器数量: {len(workers)}")
        
        for worker in workers:
            logger.info(f"  - {worker.worker_id}: {worker.status.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(register_data_collector())
    exit(0 if result else 1)

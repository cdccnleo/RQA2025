"""
测试模型训练任务调度和执行流程
"""

import sys
import os
import asyncio
import time
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_training_job_scheduler():
    """测试训练任务调度和执行流程"""
    print("=" * 60)
    print("测试模型训练任务调度和执行流程")
    print("=" * 60)
    
    try:
        # 1. 测试任务创建和调度器集成
        print("\n[1] 测试任务创建和调度器集成...")
        from src.gateway.web.model_training_routes import create_training_job
        
        # 创建测试任务
        test_request = {
            "model_type": "LSTM",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10  # 使用较小的epoch数以便快速测试
            }
        }
        
        result = await create_training_job(test_request)
        job_id = result.get("job_id")
        
        if job_id:
            print(f"✓ 任务创建成功: {job_id}")
        else:
            print("✗ 任务创建失败")
            return False
        
        # 2. 测试任务执行器启动
        print("\n[2] 测试任务执行器启动...")
        from src.gateway.web.training_job_executor import (
            start_training_job_executor,
            get_training_job_executor
        )
        
        executor = await start_training_job_executor()
        if executor and executor.running:
            print("✓ 任务执行器已启动")
        else:
            print("✗ 任务执行器启动失败")
            return False
        
        # 等待一段时间让任务被调度和执行
        print("\n[3] 等待任务调度和执行（10秒）...")
        await asyncio.sleep(10)
        
        # 4. 检查任务状态
        print("\n[4] 检查任务状态...")
        from src.gateway.web.training_job_persistence import load_training_job
        
        job = load_training_job(job_id)
        if job:
            status = job.get("status")
            progress = job.get("progress", 0)
            print(f"✓ 任务状态: {status}, 进度: {progress}%")
            
            if status == "running":
                print("✓ 任务已进入运行状态")
            elif status == "completed":
                print("✓ 任务已完成")
                accuracy = job.get("accuracy")
                loss = job.get("loss")
                if accuracy:
                    print(f"✓ 最终准确率: {accuracy:.4f}")
                if loss:
                    print(f"✓ 最终损失值: {loss:.4f}")
            elif status == "pending":
                print("⚠ 任务仍处于等待状态（可能需要更多时间）")
            else:
                print(f"⚠ 任务状态: {status}")
        else:
            print("✗ 无法加载任务")
            return False
        
        # 5. 测试任务列表
        print("\n[5] 测试任务列表...")
        from src.gateway.web.model_training_service import get_training_jobs
        
        jobs = get_training_jobs()
        print(f"✓ 当前任务数量: {len(jobs)}")
        
        if jobs:
            print("✓ 任务列表:")
            for job in jobs[:5]:  # 显示前5个任务
                print(f"  - {job.get('job_id')}: {job.get('status')} ({job.get('progress', 0)}%)")
        
        # 6. 测试WebSocket广播数据
        print("\n[6] 测试WebSocket广播数据...")
        from src.gateway.web.websocket_manager import ConnectionManager
        
        manager = ConnectionManager()
        
        # 模拟广播（不实际连接WebSocket）
        try:
            from src.gateway.web.model_training_service import (
                get_training_jobs_stats, get_training_jobs
            )
            stats = get_training_jobs_stats()
            jobs = get_training_jobs()
            
            broadcast_data = {
                "type": "model_training",
                "data": {
                    "stats": stats,
                    "job_list": jobs[:10]
                }
            }
            
            print(f"✓ 广播数据包含统计信息: {bool(broadcast_data['data'].get('stats'))}")
            print(f"✓ 广播数据包含任务列表: {len(broadcast_data['data'].get('job_list', []))} 个任务")
            
            if broadcast_data['data'].get('job_list'):
                print("✓ WebSocket广播数据格式正确")
            else:
                print("⚠ WebSocket广播数据中任务列表为空")
        except Exception as e:
            print(f"✗ WebSocket广播测试失败: {e}")
            return False
        
        # 7. 停止执行器
        print("\n[7] 停止任务执行器...")
        from src.gateway.web.training_job_executor import stop_training_job_executor
        
        await stop_training_job_executor()
        print("✓ 任务执行器已停止")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        print(f"\n✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_training_job_scheduler())
    sys.exit(0 if success else 1)


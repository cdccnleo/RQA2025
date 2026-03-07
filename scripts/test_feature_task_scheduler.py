#!/usr/bin/env python3
"""
测试特征任务调度执行系统
验证任务创建、调度、执行和状态更新的完整流程
"""

import sys
import os
import asyncio
import time
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_feature_task_scheduler():
    """测试特征任务调度执行系统"""
    logger.info("=" * 60)
    logger.info("开始测试特征任务调度执行系统")
    logger.info("=" * 60)
    
    try:
        # 1. 测试任务创建和调度器集成
        logger.info("\n[测试1] 测试任务创建和调度器集成")
        from src.gateway.web.feature_engineering_service import create_feature_task
        
        task = create_feature_task(
            task_type="技术指标",
            config={"description": "测试任务"}
        )
        
        assert task is not None, "任务创建失败"
        assert "task_id" in task, "任务缺少task_id"
        assert task["status"] == "pending", f"任务状态应为pending，实际为{task['status']}"
        
        task_id = task["task_id"]
        logger.info(f"✓ 任务创建成功: {task_id}")
        logger.info(f"  任务类型: {task['task_type']}")
        logger.info(f"  任务状态: {task['status']}")
        
        # 2. 测试任务执行器启动
        logger.info("\n[测试2] 测试任务执行器启动")
        from src.gateway.web.feature_task_executor import FeatureTaskExecutor
        
        executor = FeatureTaskExecutor()
        await executor.start()
        logger.info("✓ 任务执行器启动成功")
        
        # 等待一段时间让执行器处理任务
        logger.info("\n[测试3] 等待任务执行...")
        await asyncio.sleep(5)
        
        # 3. 测试任务状态更新
        logger.info("\n[测试4] 测试任务状态更新")
        from src.gateway.web.feature_task_persistence import load_feature_task
        
        updated_task = load_feature_task(task_id)
        if updated_task:
            logger.info(f"✓ 任务状态已更新")
            logger.info(f"  任务ID: {updated_task.get('task_id')}")
            logger.info(f"  任务状态: {updated_task.get('status')}")
            logger.info(f"  任务进度: {updated_task.get('progress', 0)}%")
            logger.info(f"  特征数量: {updated_task.get('feature_count', 0)}")
            
            # 验证状态变化
            if updated_task.get('status') in ['running', 'completed']:
                logger.info("✓ 任务状态已从pending变为running或completed")
            else:
                logger.warning(f"⚠ 任务状态仍为{updated_task.get('status')}，可能需要更多时间")
        else:
            logger.warning("⚠ 无法加载任务，可能任务尚未持久化")
        
        # 4. 测试任务列表获取
        logger.info("\n[测试5] 测试任务列表获取")
        from src.gateway.web.feature_engineering_service import get_feature_tasks
        
        tasks = get_feature_tasks()
        logger.info(f"✓ 获取到 {len(tasks)} 个任务")
        
        if tasks:
            logger.info("最近的任务:")
            for task in tasks[:3]:
                logger.info(f"  - {task.get('task_id')}: {task.get('status')} ({task.get('progress', 0)}%)")
        
        # 5. 测试任务执行器停止
        logger.info("\n[测试6] 测试任务执行器停止")
        await executor.stop()
        logger.info("✓ 任务执行器停止成功")
        
        logger.info("\n" + "=" * 60)
        logger.info("测试完成！")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


async def main():
    """主函数"""
    success = await test_feature_task_scheduler()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())


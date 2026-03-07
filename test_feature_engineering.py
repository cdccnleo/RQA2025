#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程任务调度测试脚本

用于验证特征工程任务的调度和执行功能，包括：
1. 启动调度器
2. 创建特征提取任务
3. 验证任务执行状态
4. 检查工作节点状态
"""

import time
import logging
from src.gateway.web.feature_engineering_service import (
    start_scheduler,
    create_feature_task,
    get_feature_tasks,
    get_scheduler_status
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_engineering_workflow():
    """
    测试特征工程工作流
    """
    logger.info("开始测试特征工程任务调度和执行")
    
    # 1. 启动调度器
    logger.info("步骤 1: 启动特征任务调度器")
    scheduler_started = start_scheduler()
    if scheduler_started:
        logger.info("✅ 调度器启动成功")
    else:
        logger.error("❌ 调度器启动失败")
        return False
    
    # 等待调度器完全启动
    time.sleep(3)
    
    # 2. 检查调度器状态
    logger.info("步骤 2: 检查调度器状态")
    scheduler_status = get_scheduler_status()
    logger.info(f"调度器状态: {scheduler_status}")
    
    if scheduler_status.get("is_running"):
        logger.info("✅ 调度器正在运行")
    else:
        logger.error("❌ 调度器未运行")
        return False
    
    # 3. 创建特征提取任务
    logger.info("步骤 3: 创建特征提取任务")
    task_types = ["技术指标", "统计特征", "情感特征", "自定义特征"]
    created_tasks = []
    
    for task_type in task_types:
        logger.info(f"创建 {task_type} 任务")
        task = create_feature_task(
            task_type=task_type,
            config={"param1": "value1", "param2": "value2"}
        )
        if task:
            logger.info(f"✅ 任务创建成功: {task['task_id']}")
            created_tasks.append(task)
        else:
            logger.error(f"❌ 任务创建失败: {task_type}")
        
        # 等待一段时间再创建下一个任务
        time.sleep(1)
    
    if not created_tasks:
        logger.error("❌ 没有成功创建任何任务")
        return False
    
    # 4. 等待任务执行
    logger.info("步骤 4: 等待任务执行完成")
    logger.info(f"创建了 {len(created_tasks)} 个任务，等待执行...")
    
    # 等待10秒让任务有时间执行
    time.sleep(10)
    
    # 5. 检查任务状态
    logger.info("步骤 5: 检查任务执行状态")
    tasks = get_feature_tasks()
    logger.info(f"获取到 {len(tasks)} 个任务")
    
    completed_count = 0
    running_count = 0
    failed_count = 0
    pending_count = 0
    
    for task in tasks:
        task_id = task.get('task_id')
        status = task.get('status')
        logger.info(f"任务 {task_id} 状态: {status}")
        
        if status == 'completed':
            completed_count += 1
        elif status == 'running':
            running_count += 1
        elif status == 'failed':
            failed_count += 1
        elif status == 'pending':
            pending_count += 1
    
    logger.info(f"任务状态统计: 完成={completed_count}, 运行中={running_count}, 失败={failed_count}, 等待中={pending_count}")
    
    # 6. 验证结果
    logger.info("步骤 6: 验证测试结果")
    
    if completed_count > 0:
        logger.info("✅ 测试成功: 有任务执行完成")
        return True
    elif running_count > 0:
        logger.info("⚠️  测试结果: 任务正在执行中")
        return True
    elif failed_count > 0:
        logger.warning("⚠️  测试结果: 任务执行失败")
        return False
    else:
        logger.error("❌ 测试失败: 没有任务执行")
        return False

if __name__ == "__main__":
    logger.info("=== 特征工程任务调度测试开始 ===")
    success = test_feature_engineering_workflow()
    logger.info("=== 特征工程任务调度测试结束 ===")
    
    if success:
        logger.info("🎉 测试通过: 特征工程任务调度和执行功能正常")
    else:
        logger.error("💥 测试失败: 特征工程任务调度和执行功能异常")

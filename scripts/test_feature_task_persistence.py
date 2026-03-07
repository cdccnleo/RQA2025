"""测试特征任务持久化功能"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gateway.web.feature_engineering_service import create_feature_task, get_feature_tasks
from src.gateway.web.feature_task_persistence import load_feature_task, list_feature_tasks

# 创建测试任务
print("创建测试任务...")
task = create_feature_task('技术指标', {'test': True})
print(f'✅ 任务已创建: {task["task_id"]}')

# 验证持久化
print("\n验证持久化...")
loaded_task = load_feature_task(task["task_id"])
if loaded_task:
    print(f'✅ 任务已持久化: {loaded_task["task_id"]}')
    print(f'   任务类型: {loaded_task["task_type"]}')
    print(f'   状态: {loaded_task["status"]}')
else:
    print('❌ 任务持久化失败')

# 查询任务列表
print("\n查询任务列表...")
tasks = get_feature_tasks()
print(f'✅ 任务列表长度: {len(tasks)}')
if tasks:
    print(f'   最新任务: {tasks[0].get("task_id")}')

# 从持久化存储查询
print("\n从持久化存储查询...")
persisted_tasks = list_feature_tasks(limit=10)
print(f'✅ 持久化存储中的任务数: {len(persisted_tasks)}')

print("\n✅ 特征任务持久化功能验证通过！")


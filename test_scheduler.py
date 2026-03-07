"""
调度器模块测试脚本
"""
import time
from datetime import datetime, timedelta
from src.pipeline.scheduler import (
    UnifiedScheduler, ScheduleJob, JobTrigger,
    JobStatus, TriggerType, CRON_PRESETS
)

def test_scheduler():
    """测试调度器功能"""
    print('=== 测试调度器功能 ===')

    # 1. 测试 JobTrigger
    print('\n1. 测试 JobTrigger 创建:')
    cron_trigger = JobTrigger.cron('0 9 * * 1-5')
    print(f'   Cron触发器: {cron_trigger.to_dict()}')

    interval_trigger = JobTrigger.interval(3600)
    print(f'   间隔触发器: {interval_trigger.to_dict()}')

    event_trigger = JobTrigger.event('data_ready', {'source': 'market'})
    print(f'   事件触发器: {event_trigger.to_dict()}')

    # 2. 测试 ScheduleJob
    print('\n2. 测试 ScheduleJob 创建:')
    job = ScheduleJob(
        job_id='test-job-001',
        name='测试任务',
        description='这是一个测试任务',
        trigger=cron_trigger,
        pipeline_config={'name': 'test_pipeline'}
    )
    print(f'   任务ID: {job.job_id}')
    print(f'   任务名称: {job.name}')
    print(f'   任务状态: {job.status.name}')
    print(f'   是否可执行: {job.can_execute()}')

    # 3. 测试 UnifiedScheduler
    print('\n3. 测试 UnifiedScheduler:')
    scheduler = UnifiedScheduler(max_workers=2)
    print(f'   调度器创建成功')
    print(f'   统计信息: {scheduler.get_statistics()}')

    # 4. 测试创建任务
    print('\n4. 测试创建任务:')
    new_job = scheduler.create_job(
        name='定时训练任务',
        trigger=JobTrigger.cron('0 2 * * *'),
        pipeline_config={'name': 'ml_pipeline'},
        description='每天凌晨2点执行ML训练'
    )
    print(f'   创建任务成功: {new_job.job_id}')
    print(f'   下次执行时间: {new_job.next_run_time}')

    # 5. 测试事件任务
    print('\n5. 测试事件任务:')
    event_job = scheduler.create_job(
        name='数据处理任务',
        trigger=JobTrigger.event('data_ready'),
        pipeline_config={'name': 'data_pipeline'}
    )
    print(f'   创建事件任务成功: {event_job.job_id}')

    # 6. 测试任务管理
    print('\n6. 测试任务管理:')
    print(f'   所有任务数: {len(scheduler.get_all_jobs())}')
    print(f'   暂停任务...')
    scheduler.pause_job(new_job.job_id)
    print(f'   任务状态: {scheduler.get_job_status(new_job.job_id).name}')
    print(f'   恢复任务...')
    scheduler.resume_job(new_job.job_id)
    print(f'   任务状态: {scheduler.get_job_status(new_job.job_id).name}')

    # 7. 测试统计信息
    print('\n7. 测试统计信息:')
    stats = scheduler.get_statistics()
    print(f'   总任务数: {stats["total_jobs"]}')
    print(f'   运行中任务: {stats["running_jobs"]}')
    print(f'   状态统计: {stats["status_counts"]}')

    # 8. 测试状态持久化
    print('\n8. 测试状态持久化:')
    scheduler.save_state('data/scheduler_state.json')
    print('   状态已保存到 data/scheduler_state.json')

    # 清理
    scheduler.delete_job(new_job.job_id)
    scheduler.delete_job(event_job.job_id)
    print('\n=== 所有测试通过！===')


if __name__ == '__main__':
    test_scheduler()

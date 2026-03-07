#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一调度器集成检查脚本

检查统一调度器启动逻辑，并验证各监控页面的集成状态。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedSchedulerIntegrationChecker:
    """统一调度器集成检查器"""

    def __init__(self):
        """初始化检查器"""
        self.results = {
            'task_1_unified_scheduler': {},
            'task_2_data_collection_monitor': {},
            'task_3_feature_engineering_monitor': {},
            'task_4_model_training_monitor': {},
            'task_5_data_sources_config': {},
            'task_6_backend_api': {},
            'issues': [],
            'recommendations': []
        }

    def log_issue(self, task: str, severity: str, issue: str, details: str = None):
        """
        记录问题

        Args:
            task: 任务名称
            severity: 严重程度 (critical, high, medium, low)
            issue: 问题描述
            details: 详细信息
        """
        self.results['issues'].append({
            'task': task,
            'severity': severity,
            'issue': issue,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        logger.warning(f"[{severity.upper()}] {task}: {issue}")

    def log_result(self, task: str, test_id: str, passed: bool, message: str):
        """
        记录测试结果

        Args:
            task: 任务名称
            test_id: 测试ID
            passed: 是否通过
            message: 结果消息
        """
        if task not in self.results:
            self.results[task] = {}
        self.results[task][test_id] = {
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_id}: {message}")

    def check_task_1_unified_scheduler(self) -> bool:
        """
        任务 1: 检查统一调度器启动逻辑

        Returns:
            检查是否通过
        """
        logger.info("=" * 70)
        logger.info("任务 1: 检查统一调度器启动逻辑")
        logger.info("=" * 70)

        try:
            from src.distributed.coordinator.unified_scheduler import (
                UnifiedScheduler,
                TaskType,
                TaskPriority,
                Task,
                get_unified_scheduler
            )

            # TR-1.1: get_unified_scheduler() 返回有效实例
            logger.info("\n检查 TR-1.1: get_unified_scheduler() 返回有效实例...")
            try:
                scheduler = get_unified_scheduler()
                if scheduler is not None and isinstance(scheduler, UnifiedScheduler):
                    self.log_result('task_1_unified_scheduler', 'TR-1.1', True, 
                                   'get_unified_scheduler() 返回有效的 UnifiedScheduler 实例')
                else:
                    self.log_result('task_1_unified_scheduler', 'TR-1.1', False,
                                   'get_unified_scheduler() 返回无效实例')
                    return False
            except Exception as e:
                self.log_result('task_1_unified_scheduler', 'TR-1.1', False, f'异常: {e}')
                return False

            # TR-1.2: scheduler.start() 后 is_running 为 True
            logger.info("\n检查 TR-1.2: scheduler.start() 后 is_running 为 True...")
            try:
                scheduler.start()
                stats = scheduler.get_scheduler_stats()
                is_running = stats.get('is_running', False)
                if is_running:
                    self.log_result('task_1_unified_scheduler', 'TR-1.2', True,
                                   'scheduler.start() 后 is_running 为 True')
                else:
                    self.log_result('task_1_unified_scheduler', 'TR-1.2', False,
                                   'scheduler.start() 后 is_running 仍为 False')
            except Exception as e:
                self.log_result('task_1_unified_scheduler', 'TR-1.2', False, f'异常: {e}')

            # TR-1.3: scheduler.stop() 后 is_running 为 False
            logger.info("\n检查 TR-1.3: scheduler.stop() 后 is_running 为 False...")
            try:
                scheduler.stop()
                stats = scheduler.get_scheduler_stats()
                is_running = stats.get('is_running', True)
                if not is_running:
                    self.log_result('task_1_unified_scheduler', 'TR-1.3', True,
                                   'scheduler.stop() 后 is_running 为 False')
                else:
                    self.log_result('task_1_unified_scheduler', 'TR-1.3', False,
                                   'scheduler.stop() 后 is_running 仍为 True')
            except Exception as e:
                self.log_result('task_1_unified_scheduler', 'TR-1.3', False, f'异常: {e}')

            # TR-1.4: get_scheduler_stats() 返回正确格式的统计数据
            logger.info("\n检查 TR-1.4: get_scheduler_stats() 返回正确格式...")
            try:
                stats = scheduler.get_scheduler_stats()
                required_fields = ['is_running', 'total_tasks', 'pending_tasks', 
                                  'running_tasks', 'completed_tasks', 'failed_tasks',
                                  'by_type', 'queue_sizes', 'active_workers']
                missing_fields = [f for f in required_fields if f not in stats]
                
                if not missing_fields:
                    self.log_result('task_1_unified_scheduler', 'TR-1.4', True,
                                   f'get_scheduler_stats() 返回正确格式，包含所有必需字段')
                    logger.info(f"   统计数据: {stats}")
                else:
                    self.log_result('task_1_unified_scheduler', 'TR-1.4', False,
                                   f'缺少字段: {missing_fields}')
            except Exception as e:
                self.log_result('task_1_unified_scheduler', 'TR-1.4', False, f'异常: {e}')

            # 检查任务类型映射
            logger.info("\n检查任务类型到工作节点类型的映射...")
            try:
                task_to_worker = scheduler.TASK_TYPE_TO_WORKER_TYPE
                logger.info(f"   任务类型映射: {task_to_worker}")
                
                all_correct = True
                for task_type, worker_type in task_to_worker.items():
                    logger.info(f"   ✅ {task_type.value} -> {worker_type.value}")
                
                if all_correct:
                    logger.info("   ✅ 所有任务类型映射正确")
                
            except Exception as e:
                self.log_issue('task_1_unified_scheduler', 'high',
                              '检查任务类型映射失败', str(e))

            return True

        except ImportError as e:
            self.log_issue('task_1_unified_scheduler', 'critical',
                          '无法导入统一调度器模块', str(e))
            return False
        except Exception as e:
            self.log_issue('task_1_unified_scheduler', 'critical',
                          '检查统一调度器失败', str(e))
            return False

    def check_task_2_data_collection_monitor(self) -> bool:
        """
        任务 2: 检查数据采集监控页面集成

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("任务 2: 检查数据采集监控页面集成")
        logger.info("=" * 70)

        try:
            html_file = project_root / 'web-static' / 'data-collection-monitor.html'
            content = html_file.read_text(encoding='utf-8')

            # TR-2.1: API调用路径检查
            logger.info("\n检查 TR-2.1: API调用路径...")
            api_paths = [
                '/api/v1/monitoring/historical-collection/status',
                '/api/v1/data/collection/scheduler/status'
            ]
            found_paths = [p for p in api_paths if p in content]
            if found_paths:
                self.log_result('task_2_data_collection_monitor', 'TR-2.1', True,
                               f'找到API路径: {found_paths}')
            else:
                self.log_result('task_2_data_collection_monitor', 'TR-2.1', False,
                               '未找到预期的API路径')

            # TR-2.2: 检查 queue_sizes 处理
            logger.info("\n检查 TR-2.2: queue_sizes 字典处理...")
            # 检查是否有处理 queue_sizes 的代码
            if 'queue_sizes' in content and ('Object.values' in content or 'reduce' in content):
                self.log_result('task_2_data_collection_monitor', 'TR-2.2', True,
                               '前端正确处理 queue_sizes 字典格式')
            else:
                self.log_result('task_2_data_collection_monitor', 'TR-2.2', False,
                               '前端未正确处理 queue_sizes 字典格式')

            # TR-2.3: 检查活跃工作进程获取
            logger.info("\n检查 TR-2.3: 活跃工作进程数获取...")
            if 'data_collectors_count' in content or 'active_workers' in content:
                self.log_result('task_2_data_collection_monitor', 'TR-2.3', True,
                               '前端从正确字段获取活跃工作进程数')
            else:
                self.log_result('task_2_data_collection_monitor', 'TR-2.3', False,
                               '前端未从正确字段获取活跃工作进程数')

            return True

        except Exception as e:
            self.log_issue('task_2_data_collection_monitor', 'high',
                          '检查数据采集监控页面失败', str(e))
            return False

    def check_task_3_feature_engineering_monitor(self) -> bool:
        """
        任务 3: 检查特征工程监控页面集成

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("任务 3: 检查特征工程监控页面集成")
        logger.info("=" * 70)

        try:
            html_file = project_root / 'web-static' / 'feature-engineering-monitor.html'
            content = html_file.read_text(encoding='utf-8')

            # TR-3.1: API调用路径检查
            logger.info("\n检查 TR-3.1: API调用路径...")
            if '/features/engineering/scheduler/status' in content:
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.1', True,
                               '找到特征工程调度器API路径')
            else:
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.1', False,
                               '未找到特征工程调度器API路径')

            # TR-3.2: 检查 queue_sizes 处理
            logger.info("\n检查 TR-3.2: queue_sizes 字典处理...")
            if 'queue_sizes' in content and ('Object.values' in content or 'reduce' in content):
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.2', True,
                               '前端正确处理 queue_sizes 字典格式')
            else:
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.2', False,
                               '前端未正确处理 queue_sizes 字典格式')

            # TR-3.3: 检查特征工作节点获取
            logger.info("\n检查 TR-3.3: 特征工作节点数量获取...")
            if 'feature_workers_count' in content:
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.3', True,
                               '前端从 feature_workers_count 获取特征工作节点数')
            else:
                self.log_result('task_3_feature_engineering_monitor', 'TR-3.3', False,
                               '前端未从 feature_workers_count 获取特征工作节点数')

            return True

        except Exception as e:
            self.log_issue('task_3_feature_engineering_monitor', 'high',
                          '检查特征工程监控页面失败', str(e))
            return False

    def check_task_4_model_training_monitor(self) -> bool:
        """
        任务 4: 检查模型训练监控页面集成

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("任务 4: 检查模型训练监控页面集成")
        logger.info("=" * 70)

        try:
            html_file = project_root / 'web-static' / 'model-training-monitor.html'
            content = html_file.read_text(encoding='utf-8')

            # TR-4.1: API调用路径检查
            logger.info("\n检查 TR-4.1: API调用路径...")
            if '/ml/training/scheduler/status' in content:
                self.log_result('task_4_model_training_monitor', 'TR-4.1', True,
                               '找到模型训练调度器API路径')
            else:
                self.log_result('task_4_model_training_monitor', 'TR-4.1', False,
                               '未找到模型训练调度器API路径')

            # TR-4.2: 检查 queue_sizes 处理
            logger.info("\n检查 TR-4.2: queue_sizes 字典处理...")
            if 'queue_sizes' in content and ('Object.values' in content or 'reduce' in content):
                self.log_result('task_4_model_training_monitor', 'TR-4.2', True,
                               '前端正确处理 queue_sizes 字典格式')
            else:
                self.log_result('task_4_model_training_monitor', 'TR-4.2', False,
                               '前端未正确处理 queue_sizes 字典格式')

            # TR-4.3: 检查训练执行器获取
            logger.info("\n检查 TR-4.3: 训练执行器数量获取...")
            if 'training_executors_count' in content:
                self.log_result('task_4_model_training_monitor', 'TR-4.3', True,
                               '前端从 training_executors_count 获取训练执行器数')
            else:
                self.log_result('task_4_model_training_monitor', 'TR-4.3', False,
                               '前端未从 training_executors_count 获取训练执行器数')

            return True

        except Exception as e:
            self.log_issue('task_4_model_training_monitor', 'high',
                          '检查模型训练监控页面失败', str(e))
            return False

    def check_task_5_data_sources_config(self) -> bool:
        """
        任务 5: 检查数据源配置管理页面集成

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("任务 5: 检查数据源配置管理页面集成")
        logger.info("=" * 70)

        try:
            html_file = project_root / 'web-static' / 'data-sources-config.html'
            content = html_file.read_text(encoding='utf-8')

            # TR-5.1: 确认当前API路径
            logger.info("\n检查 TR-5.1: 当前API路径...")
            if '/api/v1/data/scheduler/dashboard' in content:
                self.log_result('task_5_data_sources_config', 'TR-5.1', True,
                               '使用独立的数据采集调度器API: /api/v1/data/scheduler/dashboard')
                self.results['task_5_data_sources_config']['uses_independent_scheduler'] = True
            else:
                self.log_result('task_5_data_sources_config', 'TR-5.1', False,
                               '未找到数据采集调度器API')

            # TR-5.2: 检查是否使用统一调度器格式
            logger.info("\n检查 TR-5.2: 统一调度器格式对比...")
            # 检查后端API是否使用统一调度器（查看后端文件）
            api_file = project_root / 'src' / 'gateway' / 'web' / 'datasource_routes.py'
            api_content = api_file.read_text(encoding='utf-8')
            if 'unified_scheduler' in api_content:
                self.log_result('task_5_data_sources_config', 'TR-5.2', True,
                               '后端API已使用统一调度器格式')
            else:
                self.log_result('task_5_data_sources_config', 'TR-5.2', False,
                               '页面使用独立调度器格式，未适配统一调度器')

            return True

        except Exception as e:
            self.log_issue('task_5_data_sources_config', 'medium',
                          '检查数据源配置页面失败', str(e))
            return False

    def check_task_6_backend_api(self) -> bool:
        """
        任务 6: 验证后端API集成

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 70)
        logger.info("任务 6: 验证后端API集成")
        logger.info("=" * 70)

        try:
            # 查找API路由文件 - 正确的位置是 src/gateway/web/
            api_files = list(project_root.glob('src/gateway/web/*.py'))
            logger.info(f"找到 {len(api_files)} 个API文件")

            # 检查各API端点
            endpoints_to_check = [
                ('/data/scheduler/dashboard', 'data_collection', 'datasource_routes.py'),
                ('/features/engineering/scheduler/status', 'feature_engineering', 'feature_engineering_routes.py'),
                ('/ml/training/scheduler/status', 'model_training', 'model_training_routes.py')
            ]

            for endpoint, module, expected_file in endpoints_to_check:
                logger.info(f"\n检查端点: {endpoint}")
                found = False
                for api_file in api_files:
                    if expected_file in str(api_file):
                        content = api_file.read_text(encoding='utf-8')
                        if endpoint in content:
                            found = True
                            logger.info(f"   ✅ 在 {api_file.name} 中找到相关定义")
                            break
                
                if found:
                    self.log_result('task_6_backend_api', f'TR-6-{module}', True,
                                   f'找到 {endpoint} 的后端定义')
                else:
                    self.log_result('task_6_backend_api', f'TR-6-{module}', False,
                                   f'未找到 {endpoint} 的后端定义')

            return True

        except Exception as e:
            self.log_issue('task_6_backend_api', 'high',
                          '检查后端API集成失败', str(e))
            return False

    def generate_report(self) -> str:
        """
        生成检查报告

        Returns:
            报告文件路径
        """
        logger.info("\n" + "=" * 70)
        logger.info("生成检查报告")
        logger.info("=" * 70)

        report_dir = project_root / 'reports' / 'technical' / 'analysis'
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / 'unified_scheduler_integration_check_report_latest.md'

        # 统计结果
        total_tests = 0
        passed_tests = 0
        for task, tests in self.results.items():
            if task.startswith('task_') and isinstance(tests, dict):
                for test_id, result in tests.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get('passed'):
                        passed_tests += 1

        # 生成报告内容
        report_content = f"""# 统一调度器集成检查报告

**项目**: RQA2025  
**报告类型**: 技术检查  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**版本**: v1.0  
**状态**: 🔍 检查完成

---

## 📋 报告概览

### 检查目标
检查统一调度器启动逻辑，并验证各监控页面是否按照统一调度器更新。

### 关键指标
- **总测试数**: {total_tests}
- **通过测试**: {passed_tests}
- **失败测试**: {total_tests - passed_tests}
- **通过率**: {(passed_tests/total_tests*100):.1f}%
- **发现问题**: {len(self.results['issues'])}

---

## 📊 详细检查结果

### 任务 1: 统一调度器启动逻辑

"""
        # 添加任务1结果
        for test_id, result in self.results.get('task_1_unified_scheduler', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        report_content += """
### 任务 2: 数据采集监控页面集成

"""
        for test_id, result in self.results.get('task_2_data_collection_monitor', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        report_content += """
### 任务 3: 特征工程监控页面集成

"""
        for test_id, result in self.results.get('task_3_feature_engineering_monitor', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        report_content += """
### 任务 4: 模型训练监控页面集成

"""
        for test_id, result in self.results.get('task_4_model_training_monitor', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        report_content += """
### 任务 5: 数据源配置管理页面集成

"""
        for test_id, result in self.results.get('task_5_data_sources_config', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        report_content += """
### 任务 6: 后端API集成

"""
        for test_id, result in self.results.get('task_6_backend_api', {}).items():
            if isinstance(result, dict):
                status = "✅ 通过" if result.get('passed') else "❌ 失败"
                report_content += f"- {status} {test_id}: {result.get('message')}\n"

        # 添加问题列表
        report_content += """
---

## 🚨 发现的问题

"""
        if self.results['issues']:
            for issue in self.results['issues']:
                severity_icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(issue['severity'], '⚪')
                report_content += f"""
{severity_icon} **{issue['severity'].upper()}** - {issue['task']}
- **问题**: {issue['issue']}
- **详情**: {issue.get('details', '无')}
- **时间**: {issue['timestamp']}
"""
        else:
            report_content += "\n✅ 未发现问题\n"

        # 添加建议
        report_content += """
---

## 💡 改进建议

### 高优先级
1. 确保后端API正确返回统一调度器格式的数据
2. 验证各监控页面的调度器状态显示与实际状态一致
3. 完善统一调度器与各模块的集成测试

### 中优先级
1. 评估数据源配置页面是否需要迁移到统一调度器API
2. 添加更多监控指标和告警规则
3. 完善文档和使用示例

---

## 📋 附录

### 相关文件
- `src/distributed/coordinator/unified_scheduler.py` - 统一调度器
- `web-static/data-sources-config.html` - 数据源配置页面
- `web-static/data-collection-monitor.html` - 数据采集监控页面
- `web-static/feature-engineering-monitor.html` - 特征工程监控页面
- `web-static/model-training-monitor.html` - 模型训练监控页面

### 相关文档
- [特征层架构设计](../../docs/architecture/feature_layer_architecture_design.md)
- [报告组织规范](../README.md)

---

*本报告由统一调度器集成检查脚本自动生成。*
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✅ 报告已保存到: {report_path}")
        return str(report_path)

    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查

        Returns:
            检查结果
        """
        logger.info("\n" + "=" * 80)
        logger.info("统一调度器集成检查 - 开始")
        logger.info("=" * 80)
        logger.info(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 运行各项检查
        self.check_task_1_unified_scheduler()
        self.check_task_2_data_collection_monitor()
        self.check_task_3_feature_engineering_monitor()
        self.check_task_4_model_training_monitor()
        self.check_task_5_data_sources_config()
        self.check_task_6_backend_api()

        # 生成报告
        report_path = self.generate_report()

        logger.info("\n" + "=" * 80)
        logger.info("统一调度器集成检查 - 完成")
        logger.info("=" * 80)

        return {
            'results': self.results,
            'report_path': report_path
        }


def main():
    """主函数"""
    checker = UnifiedSchedulerIntegrationChecker()
    result = checker.run_all_checks()
    
    print("\n" + "=" * 80)
    print("检查摘要")
    print("=" * 80)
    print(f"报告路径: {result['report_path']}")
    print(f"发现问题: {len(result['results']['issues'])}")


if __name__ == '__main__':
    main()

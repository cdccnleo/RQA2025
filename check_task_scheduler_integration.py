#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取任务提交问题检查分析脚本

检查分析特征提取任务创建提交后未提交至特征任务调度器的问题。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureTaskSchedulerIntegrationChecker:
    """特征任务调度器集成检查器"""

    def __init__(self):
        self.issues = []
        self.analysis_results = {}

    def log_issue(self, severity: str, issue: str, details: str = None, location: str = None):
        """记录问题"""
        self.issues.append({
            'severity': severity,
            'issue': issue,
            'details': details,
            'location': location,
            'timestamp': datetime.now().isoformat()
        })

    def check_feature_engine_integration(self) -> bool:
        """检查特征引擎与任务调度器的集成"""
        logger.info("=" * 70)
        logger.info("检查 1: 特征引擎(FeatureEngine)与任务调度器的集成")
        logger.info("=" * 70)

        try:
            from src.features.core.engine import FeatureEngine
            from src.features.core.config import FeatureConfig

            engine = FeatureEngine(FeatureConfig())

            # 检查 1.1: create_task()方法是否调用任务调度器
            logger.info("\n检查 1.1: create_task()方法...")
            import inspect
            create_task_source = inspect.getsource(engine.create_task)
            
            if 'submit_task' not in create_task_source and 'get_task_scheduler' not in create_task_source:
                self.log_issue(
                    severity='critical',
                    issue='FeatureEngine.create_task() 未集成任务调度器',
                    details='create_task()方法仅在内部维护self.tasks列表，没有调用FeatureTaskScheduler提交任务',
                    location='src/features/core/engine.py:446-470'
                )
                logger.error("❌ create_task() 未集成任务调度器")
            else:
                logger.info("✅ create_task() 已集成任务调度器")

            # 检查 1.2: 是否有任务调度器引用
            logger.info("\n检查 1.2: 任务调度器引用...")
            if not hasattr(engine, '_task_scheduler') and not hasattr(engine, 'task_scheduler'):
                self.log_issue(
                    severity='high',
                    issue='FeatureEngine 缺少任务调度器引用',
                    details='FeatureEngine初始化时没有获取或存储FeatureTaskScheduler实例',
                    location='src/features/core/engine.py:49-98'
                )
                logger.error("❌ 缺少任务调度器引用")
            else:
                logger.info("✅ 存在任务调度器引用")

            # 检查 1.3: 初始化默认钩子
            logger.info("\n检查 1.3: 默认钩子注册...")
            if hasattr(engine, '_register_default_hooks'):
                logger.info("✅ _register_default_hooks() 方法存在")
            else:
                self.log_issue(
                    severity='medium',
                    issue='_register_default_hooks() 方法缺失',
                    details='FeatureEngine缺少_register_default_hooks()方法来注册默认任务状态钩子',
                    location='src/features/core/engine.py'
                )

            # 检查 1.4: 验证任务创建流程
            logger.info("\n检查 1.4: 任务创建流程验证...")
            test_task = engine.create_task("技术指标", {'indicators': ['sma', 'rsi']})
            logger.info(f"   创建的测试任务: {test_task['task_id']}")
            logger.info(f"   任务状态: {test_task['status']}")
            logger.info(f"   引擎内部任务数: {len(engine.tasks)}")

            return True

        except Exception as e:
            self.log_issue(
                severity='critical',
                issue='特征引擎集成检查失败',
                details=str(e),
                location='src/features/core/engine.py'
            )
            logger.error(f"   检查异常: {e}")
            return False

    def check_task_scheduler_availability(self) -> bool:
        """检查任务调度器的可用性"""
        logger.info("\n" + "=" * 70)
        logger.info("检查 2: 特征任务调度器(FeatureTaskScheduler)可用性")
        logger.info("=" * 70)

        try:
            from src.features.distributed.task_scheduler import (
                FeatureTaskScheduler,
                get_task_scheduler,
                submit_task
            )

            # 检查 2.1: 全局调度器实例
            logger.info("\n检查 2.1: 全局调度器实例...")
            scheduler = get_task_scheduler()
            if scheduler:
                logger.info("✅ 全局任务调度器实例可用")
                
                # 获取调度器统计
                stats = scheduler.get_scheduler_stats()
                logger.info(f"   调度器统计: {stats}")
            else:
                self.log_issue(
                    severity='high',
                    issue='全局任务调度器实例获取失败',
                    details='get_task_scheduler()返回None或无效实例',
                    location='src/features/distributed/task_scheduler.py:578-580'
                )

            # 检查 2.2: 便捷提交函数
            logger.info("\n检查 2.2: 便捷提交函数...")
            if callable(submit_task):
                logger.info("✅ submit_task() 便捷函数可用")
            else:
                self.log_issue(
                    severity='medium',
                    issue='submit_task() 便捷函数不可用',
                    details='submit_task便捷函数可能不存在或无法调用',
                    location='src/features/distributed/task_scheduler.py:583-590'
                )

            return True

        except Exception as e:
            self.log_issue(
                severity='critical',
                issue='任务调度器可用性检查失败',
                details=str(e),
                location='src/features/distributed/task_scheduler.py'
            )
            logger.error(f"   检查异常: {e}")
            return False

    def check_event_listeners_integration(self) -> bool:
        """检查事件监听器的集成"""
        logger.info("\n" + "=" * 70)
        logger.info("检查 3: 事件监听器(FeatureEventListeners)集成")
        logger.info("=" * 70)

        try:
            from src.features.core.event_listeners import (
                FeatureEventListeners,
                get_feature_event_listeners
            )

            # 检查 3.1: 事件监听器的任务创建
            logger.info("\n检查 3.1: _create_feature_task() 方法...")
            import inspect
            source = inspect.getsource(FeatureEventListeners._create_feature_task)
            
            if 'submit_task' in source or 'scheduler.submit_task' in source:
                logger.info("✅ _create_feature_task() 使用了任务调度器")
            else:
                self.log_issue(
                    severity='high',
                    issue='FeatureEventListeners._create_feature_task() 任务提交不完整',
                    details='_create_feature_task()尝试使用调度器提交任务，但FeatureEngine.create_task()没有同步',
                    location='src/features/core/event_listeners.py:130-181'
                )
                logger.warning("⚠️ _create_feature_task() 任务提交需要验证")

            return True

        except Exception as e:
            self.log_issue(
                severity='medium',
                issue='事件监听器集成检查异常',
                details=str(e),
                location='src/features/core/event_listeners.py'
            )
            return False

    def simulate_task_creation(self) -> dict:
        """模拟任务创建并分析问题"""
        logger.info("\n" + "=" * 70)
        logger.info("模拟分析: 任务创建流程")
        logger.info("=" * 70)

        analysis = {
            'feature_engine_tasks': [],
            'scheduler_tasks': [],
            'discrepancy': False
        }

        try:
            from src.features.core.engine import FeatureEngine
            from src.features.core.config import FeatureConfig
            from src.features.distributed.task_scheduler import get_task_scheduler

            engine = FeatureEngine(FeatureConfig())
            scheduler = get_task_scheduler()

            # 获取初始状态
            initial_engine_tasks = len(engine.tasks)
            initial_scheduler_stats = scheduler.get_scheduler_stats()
            initial_scheduler_pending = initial_scheduler_stats.get('pending_tasks', 0)

            logger.info(f"\n初始状态:")
            logger.info(f"  FeatureEngine任务数: {initial_engine_tasks}")
            logger.info(f"  调度器待处理任务: {initial_scheduler_pending}")

            # 创建测试任务
            test_task_id = f"task_test_{int(datetime.now().timestamp())}"
            logger.info(f"\n创建测试任务: {test_task_id}")

            # 通过FeatureEngine创建任务
            task = engine.create_task(
                "技术指标",
                {'indicators': ['sma', 'rsi', 'macd'], 'test_task_id': test_task_id}
            )
            logger.info(f"  FeatureEngine创建任务: {task['task_id']}")

            # 检查状态
            final_engine_tasks = len(engine.tasks)
            final_scheduler_stats = scheduler.get_scheduler_stats()
            final_scheduler_pending = final_scheduler_stats.get('pending_tasks', 0)

            logger.info(f"\n最终状态:")
            logger.info(f"  FeatureEngine任务数: {final_engine_tasks} (变化: +{final_engine_tasks - initial_engine_tasks})")
            logger.info(f"  调度器待处理任务: {final_scheduler_pending} (变化: {final_scheduler_pending - initial_scheduler_pending})")

            analysis['feature_engine_tasks'] = engine.tasks
            analysis['scheduler_tasks'] = scheduler.get_task_history(limit=10)
            analysis['discrepancy'] = (final_scheduler_pending == initial_scheduler_pending)

            if analysis['discrepancy']:
                self.log_issue(
                    severity='critical',
                    issue='任务创建不一致: FeatureEngine有任务但调度器没有',
                    details=f'FeatureEngine创建了任务但调度器待处理任务数没有变化 ({initial_scheduler_pending} → {final_scheduler_pending})',
                    location='模拟分析'
                )
                logger.error("❌ 发现问题: 任务没有提交到调度器")
            else:
                logger.info("✅ 任务已正确提交到调度器")

        except Exception as e:
            logger.error(f"模拟分析异常: {e}")
            self.log_issue(
                severity='high',
                issue='模拟分析失败',
                details=str(e),
                location='模拟分析'
            )

        return analysis

    def run_all_checks(self) -> dict:
        """运行所有检查"""
        logger.info("\n" + "=" * 80)
        logger.info("特征提取任务提交问题检查分析 - 开始")
        logger.info("=" * 80)
        logger.info(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 运行各项检查
        check1 = self.check_feature_engine_integration()
        check2 = self.check_task_scheduler_availability()
        check3 = self.check_event_listeners_integration()
        simulation = self.simulate_task_creation()

        # 生成报告
        report = self.generate_report()
        return report

    def generate_report(self) -> dict:
        """生成检查报告"""
        total_issues = len(self.issues)
        critical_issues = len([i for i in self.issues if i['severity'] == 'critical'])
        high_issues = len([i for i in self.issues if i['severity'] == 'high'])
        medium_issues = len([i for i in self.issues if i['severity'] == 'medium'])

        report = {
            'check_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': total_issues,
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'medium_issues': medium_issues
            },
            'issues': self.issues,
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> list:
        """生成修复建议"""
        recommendations = [
            {
                'priority': 'high',
                'description': '在FeatureEngine中集成FeatureTaskScheduler',
                'details': '在FeatureEngine.__init__()中获取并存储任务调度器引用'
            },
            {
                'priority': 'high',
                'description': '修改FeatureEngine.create_task()方法',
                'details': '在create_task()中调用scheduler.submit_task()将任务提交到调度器'
            },
            {
                'priority': 'medium',
                'description': '确保任务状态同步',
                'details': '建立FeatureEngine内部任务列表与调度器任务状态的双向同步机制'
            },
            {
                'priority': 'medium',
                'description': '添加任务完成后的监控更新钩子',
                'details': '在任务完成钩子中集成监控系统，自动更新仪表盘数据'
            }
        ]
        return recommendations

    def save_report(self, report_path: str = './reports/technical/analysis/feature_task_scheduler_integration_issue_report_latest.md'):
        """保存检查报告"""
        report_dir = Path(report_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        report_content = f"""# 特征提取任务提交至调度器问题分析报告

**项目**: RQA2025  
**报告类型**: 技术分析  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**版本**: v1.0  
**状态**: 🔍 分析完成

---

## 📋 报告概览

### 问题描述
特征提取任务(如 task_1771199808)在FeatureEngine中创建后，没有正确提交至FeatureTaskScheduler进行调度执行。

### 关键指标
- **总问题数**: {report['summary']['total_issues']}
- **严重问题**: {report['summary']['critical_issues']}
- **高优先级**: {report['summary']['high_issues']}
- **中优先级**: {report['summary']['medium_issues']}

---

## 📊 详细分析

### 发现的问题
"""

        for issue in report['issues']:
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(issue['severity'], '⚪')
            
            report_content += f"""
{severity_icon} **{issue['severity'].upper()}** - {issue['issue']}
- **时间**: {issue['timestamp']}
- **位置**: {issue.get('location', '未知')}
- **详情**: {issue.get('details', '无')}
"""

        report_content += f"""
---

## 🔧 修复建议

### 高优先级修复
"""

        for rec in [r for r in report['recommendations'] if r['priority'] == 'high']:
            report_content += f"""
1. **{rec['description']}**
   - {rec['details']}
"""

        report_content += f"""
### 中优先级修复
"""

        for rec in [r for r in report['recommendations'] if r['priority'] == 'medium']:
            report_content += f"""
1. **{rec['description']}**
   - {rec['details']}
"""

        report_content += f"""
---

## 📝 技术细节

### 问题根因分析

1. **FeatureEngine.create_task() 仅维护内部任务列表**
   - 位置: `src/features/core/engine.py:446-470`
   - 问题: 仅将任务添加到 `self.tasks` 列表，不调用调度器

2. **缺少任务调度器集成**
   - 位置: `src/features/core/engine.py:49-98`
   - 问题: FeatureEngine 初始化时没有获取或存储 FeatureTaskScheduler 实例

3. **任务状态管理分散**
   - 问题: FeatureEngine 和 FeatureTaskScheduler 各自维护独立的任务状态，缺乏同步

### 建议的修复方案

```python
# 在 FeatureEngine.__init__() 中添加
try:
    from src.features.distributed.task_scheduler import get_task_scheduler
    self._task_scheduler = get_task_scheduler()
    logger.info("任务调度器集成成功")
except Exception as e:
    logger.warning(f"任务调度器集成失败: {e}")
    self._task_scheduler = None

# 在 FeatureEngine.create_task() 中添加
if self._task_scheduler:
    try:
        scheduler_task_id = self._task_scheduler.submit_task(
            task_type=task_type,
            data=config or {},
            metadata={'engine_task_id': task_id}
        )
        task['scheduler_task_id'] = scheduler_task_id
        logger.info(f"任务已提交至调度器: {scheduler_task_id}")
    except Exception as e:
        logger.error(f"提交任务至调度器失败: {e}")
```

---

## 📋 附录

### 相关文档
- [特征层架构设计](../../docs/architecture/feature_layer_architecture_design.md)
- [报告组织规范](../README.md)
- [报告索引](../INDEX.md)

### 相关文件
- `src/features/core/engine.py` - 特征引擎核心
- `src/features/distributed/task_scheduler.py` - 特征任务调度器
- `src/features/core/event_listeners.py` - 事件监听器

---

*本报告由特征提取任务提交问题检查分析脚本自动生成。*
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"\n✅ 分析报告已保存到: {report_path}")
        return report_path


def main():
    """主函数"""
    checker = FeatureTaskSchedulerIntegrationChecker()
    report = checker.run_all_checks()
    report_path = checker.save_report()
    
    print("\n" + "=" * 80)
    print("检查摘要")
    print("=" * 80)
    print(f"总问题数: {report['summary']['total_issues']}")
    print(f"严重问题: {report['summary']['critical_issues']}")
    print(f"高优先级: {report['summary']['high_issues']}")
    print(f"中优先级: {report['summary']['medium_issues']}")
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == '__main__':
    main()

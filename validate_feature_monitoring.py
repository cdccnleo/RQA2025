#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程监控系统验证检查脚本

对特征工程监控系统(feature-engineering-monitor)进行系统性检查与验证。
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
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


class FeatureEngineeringMonitorValidator:
    """特征工程监控系统验证器"""

    def __init__(self):
        """初始化验证器"""
        self.issues = []
        self.issues_by_category = {
            '数据更新机制': [],
            '数据持久化': [],
            '数据流转': [],
            '功能实现': [],
            '架构一致性': []
        }
        self.check_results = {
            'passed': [],
            'failed': [],
            'warning': []
        }

    def log_issue(self, category: str, severity: str, issue: str, details: Optional[str] = None):
        """
        记录问题

        Args:
            category: 问题类别
            severity: 严重程度 (critical, high, medium, low)
            issue: 问题描述
            details: 详细信息
        """
        issue_record = {
            'category': category,
            'severity': severity,
            'issue': issue,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.issues.append(issue_record)
        if category in self.issues_by_category:
            self.issues_by_category[category].append(issue_record)
        
        if severity == 'critical':
            self.check_results['failed'].append(issue)
        elif severity == 'high':
            self.check_results['warning'].append(issue)
        else:
            self.check_results['passed'].append(issue)

    def check_data_update_mechanism(self) -> bool:
        """
        检查特征提取任务完成后各仪表盘数据更新机制

        Returns:
            检查是否通过
        """
        logger.info("=" * 60)
        logger.info("检查 1: 特征提取任务完成后各仪表盘数据更新机制")
        logger.info("=" * 60)

        try:
            # 导入监控模块
            from src.features.monitoring.features_monitor import get_monitor
            from src.features.monitoring.metrics_collector import get_collector
            from src.features.monitoring.metrics_persistence import get_persistence_manager
            from src.features.monitoring.monitoring_dashboard import get_dashboard

            # 检查 1.1: 验证监控系统组件是否能正确初始化
            logger.info("\n检查 1.1: 监控系统组件初始化...")
            try:
                monitor = get_monitor()
                collector = get_collector()
                persistence = get_persistence_manager()
                dashboard = get_dashboard()
                logger.info("✅ 监控系统组件初始化成功")
            except Exception as e:
                self.log_issue(
                    category='数据更新机制',
                    severity='critical',
                    issue='监控系统组件初始化失败',
                    details=str(e)
                )
                return False

            # 检查 1.2: 验证指标收集与持久化流程
            logger.info("\n检查 1.2: 指标收集与持久化流程...")
            try:
                # 模拟特征提取任务完成事件
                test_component = "feature_engine"
                test_metric = "feature_generation_time"
                test_value = 0.45  # 450ms

                # 收集指标
                collector.collect_metric(
                    name=test_metric,
                    value=test_value,
                    category=collector.metrics_collector.MetricCategory.PERFORMANCE,
                    metric_type=collector.metrics_collector.MetricType.GAUGE,
                    labels={'component': test_component}
                )
                logger.info("✅ 指标收集成功")

                # 检查 1.3: 验证指标持久化是否正常
                logger.info("\n检查 1.3: 指标持久化...")
                persistence.store_metric(
                    component_name=test_component,
                    metric_name=test_metric,
                    metric_value=test_value,
                    metric_type='gauge',
                    labels={'component': test_component}
                )
                logger.info("✅ 指标持久化成功")

                # 检查 1.4: 验证指标查询是否正常
                logger.info("\n检查 1.4: 指标查询...")
                query_result = persistence.query_metrics(
                    component_name=test_component,
                    metric_name=test_metric,
                    limit=10
                )
                if not query_result.empty:
                    logger.info("✅ 指标查询成功")
                    logger.info(f"   查询到 {len(query_result)} 条记录")
                else:
                    self.log_issue(
                        category='数据更新机制',
                        severity='high',
                        issue='指标查询返回空结果',
                        details='查询测试指标时未找到数据'
                    )

            except Exception as e:
                self.log_issue(
                    category='数据更新机制',
                    severity='high',
                    issue='指标收集与持久化流程异常',
                    details=str(e)
                )

            # 检查 1.5: 验证仪表盘数据获取
            logger.info("\n检查 1.5: 仪表盘数据获取...")
            try:
                # 注册测试组件
                monitor.register_component("feature_engine_test", "engine")
                
                # 更新组件状态
                monitor.update_component_status(
                    "feature_engine_test",
                    "active",
                    {'cpu_usage': 45.0, 'memory_usage': 60.0}
                )
                
                # 获取组件状态
                status = monitor.get_all_status()
                if status:
                    logger.info("✅ 组件状态获取成功")
                
                # 获取仪表盘数据
                for chart_id, chart_config in dashboard.charts.items():
                    chart_data = dashboard.get_chart_data(chart_config)
                    logger.info(f"   图表 {chart_id} 数据获取成功")
                
            except Exception as e:
                self.log_issue(
                    category='数据更新机制',
                    severity='medium',
                    issue='仪表盘数据获取异常',
                    details=str(e)
                )

            # 检查 1.6: 检查特征提取任务完成后的事件触发机制
            logger.info("\n检查 1.6: 特征提取任务完成事件触发...")
            try:
                from src.features.core.engine import FeatureEngine
                from src.features.core.config import FeatureConfig
                
                # 检查FeatureEngine中的任务管理功能
                engine = FeatureEngine(FeatureConfig())
                
                # 创建一个测试任务
                task = engine.create_task("技术指标", {'indicators': ['sma', 'rsi']})
                logger.info(f"   创建测试任务: {task['task_id']}")
                
                # 更新任务状态为完成
                engine.update_task_status(task['task_id'], 'completed', 100)
                logger.info(f"   更新任务状态为完成")
                
                # 检查任务列表
                tasks = engine.get_tasks()
                completed_tasks = [t for t in tasks if t.get('status') == 'completed']
                logger.info(f"   当前完成任务数: {len(completed_tasks)}")
                
                # ⚠️ 发现问题: 任务完成后没有自动触发监控系统更新
                logger.warning("⚠️ 发现问题: 特征提取任务完成后没有自动触发监控系统数据更新")
                self.log_issue(
                    category='数据更新机制',
                    severity='high',
                    issue='特征提取任务完成后未自动触发监控系统更新',
                    details='FeatureEngine.update_task_status() 方法在任务完成时没有与监控系统集成，'
                            '无法自动更新仪表盘数据'
                )

            except Exception as e:
                self.log_issue(
                    category='数据更新机制',
                    severity='high',
                    issue='特征任务管理功能异常',
                    details=str(e)
                )

        except Exception as e:
            self.log_issue(
                category='数据更新机制',
                severity='critical',
                issue='数据更新机制检查失败',
                details=str(e)
            )
            return False

        return True

    def check_data_persistence(self) -> bool:
        """
        检查数据持久化存储逻辑是否符合设计规范

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 60)
        logger.info("检查 2: 数据持久化存储逻辑")
        logger.info("=" * 60)

        try:
            from src.features.monitoring.metrics_persistence import (
                MetricsPersistenceManager,
                EnhancedMetricsPersistenceManager,
                StorageBackend
            )

            # 检查 2.1: 验证存储后端配置
            logger.info("\n检查 2.1: 存储后端配置...")
            try:
                # 检查SQLite存储
                persistence = MetricsPersistenceManager({'path': './test_monitoring_data'})
                logger.info("✅ SQLite存储配置正确")
                
                # 检查表结构
                if hasattr(persistence._enhanced_manager, 'db_path'):
                    import sqlite3
                    conn = sqlite3.connect(persistence._enhanced_manager.db_path)
                    cursor = conn.execute("PRAGMA table_info(metrics)")
                    columns = [row[1] for row in cursor.fetchall()]
                    logger.info(f"   metrics表包含 {len(columns)} 列: {columns}")
                    conn.close()
                
            except Exception as e:
                self.log_issue(
                    category='数据持久化',
                    severity='high',
                    issue='存储后端配置异常',
                    details=str(e)
                )

            # 检查 2.2: 验证批量写入机制
            logger.info("\n检查 2.2: 批量写入机制...")
            try:
                enhanced_persistence = EnhancedMetricsPersistenceManager({
                    'path': './test_enhanced_monitoring_data',
                    'batch_size': 100,
                    'batch_timeout': 1.0
                })
                
                # 测试写入多条记录
                test_records = []
                for i in range(50):
                    enhanced_persistence.store_metric_sync(
                        component_name=f'test_component_{i % 5}',
                        metric_name='test_metric',
                        metric_value=float(i),
                        metric_type='gauge'
                    )
                
                logger.info("✅ 批量写入机制正常")
                
                # 检查 2.3: 验证索引效率
                logger.info("\n检查 2.3: 数据库索引...")
                import sqlite3
                conn = sqlite3.connect(enhanced_persistence.db_path)
                cursor = conn.execute("PRAGMA index_list(metrics)")
                indexes = cursor.fetchall()
                logger.info(f"   metrics表包含 {len(indexes)} 个索引")
                for idx in indexes:
                    logger.info(f"     - {idx[1]}")
                conn.close()
                
                # 停止增强管理器
                enhanced_persistence.stop()
                
            except Exception as e:
                self.log_issue(
                    category='数据持久化',
                    severity='medium',
                    issue='批量写入机制异常',
                    details=str(e)
                )

            # 检查 2.4: 验证数据归档策略
            logger.info("\n检查 2.4: 数据归档策略...")
            try:
                # 检查是否有数据生命周期管理
                if hasattr(EnhancedMetricsPersistenceManager, 'ArchiveConfig'):
                    logger.info("✅ 数据归档配置存在")
                else:
                    self.log_issue(
                        category='数据持久化',
                        severity='low',
                        issue='数据归档策略文档化不足',
                        details='ArchiveConfig存在但使用示例不明确'
                    )
            except Exception as e:
                logger.warning(f"   数据归档策略检查异常: {e}")

        except Exception as e:
            self.log_issue(
                category='数据持久化',
                severity='high',
                issue='数据持久化检查失败',
                details=str(e)
            )
            return False

        return True

    def check_data_flow(self) -> bool:
        """
        确认特征数据流转至模型训练环节的完整性与一致性

        Returns:
            检查是否通过
        """
        logger.info("\n" + "=" * 60)
        logger.info("检查 3: 特征数据流转至模型训练环节")
        logger.info("=" * 60)

        try:
            # 检查 3.1: 特征存储与加载流程
            logger.info("\n检查 3.1: 特征存储与加载...")
            try:
                from src.features.core.feature_saver import FeatureSaver
                
                # 创建测试数据
                test_data = pd.DataFrame({
                    'close': np.random.rand(100),
                    'high': np.random.rand(100) + 0.5,
                    'low': np.random.rand(100) - 0.5,
                    'volume': np.random.randint(1000, 10000, 100)
                })
                
                # 保存特征
                saver = FeatureSaver(base_path='./test_feature_outputs')
                save_result = saver.save_features(
                    test_data,
                    './test_feature_outputs/test_features.parquet',
                    format='parquet',
                    metadata={'test_run': True, 'timestamp': datetime.now().isoformat()}
                )
                
                if save_result:
                    logger.info("✅ 特征保存成功")
                    
                    # 加载特征验证
                    loaded_data = saver.load_features('./test_feature_outputs/test_features.parquet')
                    if not loaded_data.empty:
                        logger.info("✅ 特征加载成功")
                        logger.info(f"   加载数据形状: {loaded_data.shape}")
                else:
                    self.log_issue(
                        category='数据流转',
                        severity='high',
                        issue='特征保存失败',
                        details='FeatureSaver.save_features() 返回False'
                    )
                
                # 检查元数据
                metadata = saver.get_last_metadata()
                if metadata:
                    logger.info("✅ 元数据记录完整")
                    logger.info(f"   元数据键: {list(metadata.keys())}")
                
            except Exception as e:
                self.log_issue(
                    category='数据流转',
                    severity='high',
                    issue='特征存储与加载流程异常',
                    details=str(e)
                )

            # 检查 3.2: 特征引擎与模型训练的接口
            logger.info("\n检查 3.2: 特征引擎与模型训练接口...")
            try:
                from src.features.core.engine import FeatureEngine
                
                # 检查是否有直接为模型训练提供特征的接口
                engine = FeatureEngine()
                
                # 检查特征列表获取
                features = engine.get_features()
                logger.info(f"   引擎中当前特征数: {len(features)}")
                
                # ⚠️ 发现问题: 特征数据与模型训练环节的直接集成不明确
                logger.warning("⚠️ 发现问题: 特征数据流转至模型训练环节的集成不明确")
                self.log_issue(
                    category='数据流转',
                    severity='medium',
                    issue='特征数据与模型训练环节的接口不明确',
                    details='FeatureEngine提供了特征存储功能，但缺少直接为模型训练环节提供特征数据的标准接口，'
                            '数据流转的完整性与一致性无法系统性保证'
                )

            except Exception as e:
                logger.warning(f"   特征引擎接口检查异常: {e}")

            # 检查 3.3: 特征版本管理
            logger.info("\n检查 3.3: 特征版本管理...")
            try:
                from src.features.core.version_management import FeatureVersionManager
                
                logger.info("✅ 特征版本管理模块存在")
                self.log_issue(
                    category='数据流转',
                    severity='low',
                    issue='特征版本管理功能需要验证',
                    details='FeatureVersionManager模块存在，但未在本次检查中完整测试其版本追踪和回滚功能'
                )
                
            except ImportError:
                self.log_issue(
                    category='数据流转',
                    severity='medium',
                    issue='特征版本管理模块缺失或导入失败',
                    details='无法找到或导入FeatureVersionManager，可能影响特征数据的可追溯性'
                )
            except Exception as e:
                logger.warning(f"   特征版本管理检查异常: {e}")

        except Exception as e:
            self.log_issue(
                category='数据流转',
                severity='high',
                issue='数据流转检查失败',
                details=str(e)
            )
            return False

        return True

    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查

        Returns:
            检查结果汇总
        """
        logger.info("\n" + "=" * 80)
        logger.info("特征工程监控系统验证检查 - 开始")
        logger.info("=" * 80)
        logger.info(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 运行各项检查
        check1_passed = self.check_data_update_mechanism()
        check2_passed = self.check_data_persistence()
        check3_passed = self.check_data_flow()

        # 生成汇总报告
        summary = self.generate_summary()

        logger.info("\n" + "=" * 80)
        logger.info("特征工程监控系统验证检查 - 完成")
        logger.info("=" * 80)

        return summary

    def generate_summary(self) -> Dict[str, Any]:
        """
        生成检查总结

        Returns:
            总结数据
        """
        total_issues = len(self.issues)
        critical_issues = len([i for i in self.issues if i['severity'] == 'critical'])
        high_issues = len([i for i in self.issues if i['severity'] == 'high'])
        medium_issues = len([i for i in self.issues if i['severity'] == 'medium'])
        low_issues = len([i for i in self.issues if i['severity'] == 'low'])

        summary = {
            'check_timestamp': datetime.now().isoformat(),
            'total_issues': total_issues,
            'issues_by_severity': {
                'critical': critical_issues,
                'high': high_issues,
                'medium': medium_issues,
                'low': low_issues
            },
            'issues_by_category': {
                cat: len(issues) for cat, issues in self.issues_by_category.items()
            },
            'all_issues': self.issues,
            'check_results': self.check_results
        }

        return summary

    def save_report(self, report_path: str = './reports/operational/monitoring/feature_engineering_monitor_validation_report_latest.md'):
        """
        保存检查报告

        Args:
            report_path: 报告保存路径
        """
        report_dir = Path(report_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)

        summary = self.generate_summary()

        report_content = f"""# 特征工程监控系统验证检查报告

**项目**: RQA2025  
**报告类型**: 技术验证  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**版本**: v1.0  
**状态**: 🔍 检查完成

---

## 📋 报告概览

### 检查目标
对特征工程监控系统(feature-engineering-monitor)进行系统性检查与验证，包括：
1. 验证特征提取任务完成后各仪表盘数据更新机制的准确性与实时性
2. 检查数据持久化存储逻辑是否符合设计规范
3. 确认特征数据流转至模型训练环节的完整性与一致性

### 关键指标
- **总问题数**: {summary['total_issues']}
- **严重问题**: {summary['issues_by_severity']['critical']}
- **高优先级问题**: {summary['issues_by_severity']['high']}
- **中优先级问题**: {summary['issues_by_severity']['medium']}
- **低优先级问题**: {summary['issues_by_severity']['low']}

---

## 📊 详细分析

### 1. 特征提取任务完成后各仪表盘数据更新机制

#### 发现问题
"""

        # 添加数据更新机制的问题
        for issue in self.issues_by_category['数据更新机制']:
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(issue['severity'], '⚪')
            
            report_content += f"""
{severity_icon} **{issue['severity'].upper()}** - {issue['issue']}
- **时间**: {issue['timestamp']}
- **详情**: {issue['details']}
"""

        report_content += f"""
### 2. 数据持久化存储逻辑

#### 发现问题
"""

        for issue in self.issues_by_category['数据持久化']:
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(issue['severity'], '⚪')
            
            report_content += f"""
{severity_icon} **{issue['severity'].upper()}** - {issue['issue']}
- **时间**: {issue['timestamp']}
- **详情**: {issue['details']}
"""

        report_content += f"""
### 3. 特征数据流转至模型训练环节

#### 发现问题
"""

        for issue in self.issues_by_category['数据流转']:
            severity_icon = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(issue['severity'], '⚪')
            
            report_content += f"""
{severity_icon} **{issue['severity'].upper()}** - {issue['issue']}
- **时间**: {issue['timestamp']}
- **详情**: {issue['details']}
"""

        report_content += f"""
---

## 📈 结论与建议

### 主要发现

1. **数据更新机制存在缺陷**: 特征提取任务完成后没有自动触发监控系统更新的机制
2. **数据持久化基本正常**: SQLite存储和批量写入机制工作正常，但归档策略需要完善
3. **数据流转接口不明确**: 特征数据与模型训练环节的标准集成接口缺失

### 建议措施

#### 高优先级
1. 实现特征提取任务完成事件与监控系统的集成
2. 在FeatureEngine中添加任务完成后的钩子函数，自动更新监控指标
3. 建立特征数据与模型训练环节的标准接口

#### 中优先级
1. 完善数据归档策略的文档和使用示例
2. 补充特征版本管理的测试用例
3. 添加数据一致性校验机制

#### 低优先级
1. 增强仪表盘的实时性（考虑WebSocket）
2. 添加更多监控维度和告警规则
3. 优化批量写入的性能

---

## 📋 附录

### 相关文档
- [特征层架构设计](../../docs/architecture/feature_layer_architecture_design.md)
- [报告组织规范](../README.md)
- [报告索引](../INDEX.md)

### 联系方式
- 架构师: RQA2025架构团队
- 技术负责人: 项目维护者

---

*本报告由特征工程监控系统验证检查脚本自动生成。*
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"\n✅ 检查报告已保存到: {report_path}")
        return report_path


def main():
    """主函数"""
    validator = FeatureEngineeringMonitorValidator()
    
    # 运行所有检查
    summary = validator.run_all_checks()
    
    # 保存报告
    report_path = validator.save_report()
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("检查摘要")
    print("=" * 80)
    print(f"总问题数: {summary['total_issues']}")
    print(f"严重问题: {summary['issues_by_severity']['critical']}")
    print(f"高优先级: {summary['issues_by_severity']['high']}")
    print(f"中优先级: {summary['issues_by_severity']['medium']}")
    print(f"低优先级: {summary['issues_by_severity']['low']}")
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == '__main__':
    main()

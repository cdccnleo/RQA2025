#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进进度跟踪脚本

自动收集系统指标，生成进度报告
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovementProgressTracker:
    """改进进度跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.report_file = "docs/improvement_metrics.json"
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "system": {},
            "scheduler": {},
            "tasks": {},
            "features": {}
        }
        
        try:
            # 收集调度器指标
            from src.features.distributed.task_scheduler import get_task_scheduler
            from src.features.distributed.worker_manager import get_worker_manager
            
            scheduler = get_task_scheduler()
            worker_manager = get_worker_manager()
            
            metrics["scheduler"] = {
                "is_running": scheduler._running,
                "worker_count": len(worker_manager.get_all_workers()),
                "stats": scheduler.get_scheduler_stats() if scheduler._running else {}
            }
            
        except Exception as e:
            logger.error(f"收集调度器指标失败: {e}")
            metrics["scheduler"]["error"] = str(e)
        
        try:
            # 收集任务指标
            from src.gateway.web.feature_task_persistence import list_feature_tasks
            
            tasks = list_feature_tasks()
            total_tasks = len(tasks)
            completed_tasks = sum(1 for t in tasks if t.get("status") == "completed")
            failed_tasks = sum(1 for t in tasks if t.get("status") == "failed")
            pending_tasks = sum(1 for t in tasks if t.get("status") == "pending")
            
            metrics["tasks"] = {
                "total": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "pending": pending_tasks,
                "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"收集任务指标失败: {e}")
            metrics["tasks"]["error"] = str(e)
        
        try:
            # 收集特征指标
            from src.gateway.web.feature_engineering_service import get_features
            
            features = get_features()
            metrics["features"] = {
                "total_count": len(features),
                "categories": {}
            }
            
            # 统计特征类别
            for feature in features:
                category = feature.get("category", "unknown")
                metrics["features"]["categories"][category] = \
                    metrics["features"]["categories"].get(category, 0) + 1
                    
        except Exception as e:
            logger.error(f"收集特征指标失败: {e}")
            metrics["features"]["error"] = str(e)
        
        return metrics
    
    def calculate_kpis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算关键绩效指标"""
        kpis = {
            "system_availability": 0,
            "task_success_rate": 0,
            "feature_quality_score": 0,
            "overall_health": "unknown"
        }
        
        # 系统可用性
        scheduler = metrics.get("scheduler", {})
        if scheduler.get("is_running"):
            kpis["system_availability"] = 100
        else:
            kpis["system_availability"] = 0
        
        # 任务成功率
        tasks = metrics.get("tasks", {})
        kpis["task_success_rate"] = tasks.get("success_rate", 0)
        
        # 整体健康度
        if kpis["system_availability"] == 100 and kpis["task_success_rate"] >= 95:
            kpis["overall_health"] = "healthy"
        elif kpis["system_availability"] == 100 and kpis["task_success_rate"] >= 80:
            kpis["overall_health"] = "warning"
        else:
            kpis["overall_health"] = "critical"
        
        return kpis
    
    def generate_report(self) -> Dict[str, Any]:
        """生成进度报告"""
        logger.info("开始生成改进进度报告...")
        
        # 收集指标
        metrics = self.collect_system_metrics()
        
        # 计算KPI
        kpis = self.calculate_kpis(metrics)
        
        # 生成报告
        report = {
            "report_time": datetime.now().isoformat(),
            "version": "1.0",
            "metrics": metrics,
            "kpis": kpis,
            "phase_progress": self._get_phase_progress(),
            "recommendations": self._generate_recommendations(metrics, kpis)
        }
        
        # 保存报告
        self._save_report(report)
        
        logger.info("改进进度报告生成完成")
        return report
    
    def _get_phase_progress(self) -> Dict[str, Any]:
        """获取各阶段进度"""
        return {
            "phase1": {
                "name": "核心功能完善",
                "progress": 65,
                "status": "in_progress",
                "tasks_completed": 2,
                "tasks_total": 4
            },
            "phase2": {
                "name": "质量评估系统",
                "progress": 0,
                "status": "not_started",
                "tasks_completed": 0,
                "tasks_total": 4
            },
            "phase3": {
                "name": "智能优化",
                "progress": 0,
                "status": "not_started",
                "tasks_completed": 0,
                "tasks_total": 4
            }
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any], kpis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于系统可用性
        if kpis["system_availability"] < 100:
            recommendations.append("调度器未运行，建议检查调度器启动逻辑")
        
        # 基于任务成功率
        if kpis["task_success_rate"] < 95:
            recommendations.append("任务成功率较低，建议检查失败任务原因")
        
        # 基于任务队列
        tasks = metrics.get("tasks", {})
        if tasks.get("pending", 0) > 10:
            recommendations.append("待处理任务较多，建议增加工作节点")
        
        # 基于特征数量
        features = metrics.get("features", {})
        if features.get("total_count", 0) == 0:
            recommendations.append("特征数量为空，建议检查特征生成逻辑")
        
        if not recommendations:
            recommendations.append("系统运行正常，继续保持")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """保存报告到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
            
            # 读取历史报告
            history = []
            if os.path.exists(self.report_file):
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加新报告
            if not isinstance(history, list):
                history = []
            history.append(report)
            
            # 只保留最近30天的报告
            history = history[-30:]
            
            # 保存
            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"报告已保存到: {self.report_file}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印报告摘要"""
        print("\n" + "="*60)
        print("  改进进度报告摘要")
        print("="*60)
        print(f"报告时间: {report['report_time']}")
        print(f"整体健康度: {report['kpis']['overall_health'].upper()}")
        print(f"系统可用性: {report['kpis']['system_availability']:.1f}%")
        print(f"任务成功率: {report['kpis']['task_success_rate']:.1f}%")
        
        print("\n阶段进度:")
        for phase_id, phase in report['phase_progress'].items():
            status_icon = "🟢" if phase['status'] == 'completed' else "🟡" if phase['status'] == 'in_progress' else "⚪"
            print(f"  {status_icon} {phase['name']}: {phase['progress']}% ({phase['tasks_completed']}/{phase['tasks_total']} 任务)")
        
        print("\n改进建议:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*60 + "\n")


def main():
    """主函数"""
    tracker = ImprovementProgressTracker()
    report = tracker.generate_report()
    tracker.print_summary(report)


if __name__ == "__main__":
    main()

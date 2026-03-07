"""
风险报告服务层
封装实际的风险报告生成组件，为API提供统一接口
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# 导入风险控制层组件（尝试多个可能的路径）
REPORT_GENERATOR_AVAILABLE = False
REPORT_MANAGER_AVAILABLE = False
RiskReportGenerator = None
ReportManager = None

# 尝试从 risk.reporting 导入
try:
    from src.risk.reporting.report_generator import RiskReportGenerator
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    # 尝试从其他可能的位置导入
    try:
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator as RiskReportGenerator
        REPORT_GENERATOR_AVAILABLE = True
    except ImportError:
        try:
            from src.infrastructure.security.audit.audit_reporting import AuditReportGenerator as RiskReportGenerator
            REPORT_GENERATOR_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"无法导入风险报告生成器: {e}")
            REPORT_GENERATOR_AVAILABLE = False

try:
    from src.risk.reporting.report_manager import ReportManager
    REPORT_MANAGER_AVAILABLE = True
except ImportError:
    # 尝试从其他可能的位置导入
    try:
        # 如果没有专门的 ReportManager，使用占位类
        class ReportManager:
            """报告管理器占位类"""
            def __init__(self):
                pass
            def get_templates(self):
                return []
            def list_templates(self):
                return []
            def get_generation_tasks(self):
                return []
            def list_tasks(self):
                return []
            def get_report_history(self):
                return []
            def list_reports(self):
                return []
            def get_all_reports(self):
                return []
        REPORT_MANAGER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"无法导入报告管理器: {e}")
        REPORT_MANAGER_AVAILABLE = False


# 单例实例
_report_generator: Optional[Any] = None
_report_manager: Optional[Any] = None


def get_report_generator() -> Optional[Any]:
    """获取报告生成器实例"""
    global _report_generator
    if _report_generator is None and REPORT_GENERATOR_AVAILABLE:
        try:
            _report_generator = RiskReportGenerator()
            logger.info("风险报告生成器初始化成功")
        except Exception as e:
            logger.error(f"初始化报告生成器失败: {e}")
    return _report_generator


def get_report_manager() -> Optional[Any]:
    """获取报告管理器实例"""
    global _report_manager
    if _report_manager is None and REPORT_MANAGER_AVAILABLE:
        try:
            _report_manager = ReportManager()
            logger.info("报告管理器初始化成功")
        except Exception as e:
            logger.error(f"初始化报告管理器失败: {e}")
    return _report_manager


# ==================== 报告模板服务 ====================

def get_report_templates() -> List[Dict[str, Any]]:
    """获取报告模板列表 - 从真实报告管理器获取，不使用模拟数据"""
    report_manager = get_report_manager()
    report_generator = get_report_generator()
    
    templates = []
    
    # 尝试从报告管理器获取模板
    if report_manager:
        try:
            if hasattr(report_manager, 'get_templates'):
                templates = report_manager.get_templates()
            elif hasattr(report_manager, 'list_templates'):
                templates = report_manager.list_templates()
        except Exception as e:
            logger.debug(f"从报告管理器获取模板失败: {e}")
    
    # 尝试从报告生成器获取模板
    if not templates and report_generator:
        try:
            if hasattr(report_generator, 'get_available_templates'):
                templates = report_generator.get_available_templates()
        except Exception as e:
            logger.debug(f"从报告生成器获取模板失败: {e}")
    
    # 格式化模板数据
    if templates:
        formatted_templates = []
        for template in templates:
            if not isinstance(template, dict):
                if hasattr(template, '__dict__'):
                    template_dict = template.__dict__
                elif hasattr(template, 'to_dict'):
                    template_dict = template.to_dict()
                else:
                    continue
            else:
                template_dict = template
            
            formatted_templates.append({
                "id": template_dict.get('id', ''),
                "name": template_dict.get('name', ''),
                "report_type": template_dict.get('report_type', ''),
                "frequency": template_dict.get('frequency', ''),
                "status": template_dict.get('status', 'active'),
                "last_generated": template_dict.get('last_generated', 0)
            })
        
        return formatted_templates
    
    # 量化交易系统要求：不使用模拟数据，返回空列表
    return []


# ==================== 报告生成任务服务 ====================

def get_generation_tasks() -> List[Dict[str, Any]]:
    """获取报告生成任务列表 - 从真实报告管理器获取，不使用模拟数据"""
    report_manager = get_report_manager()
    report_generator = get_report_generator()
    
    tasks = []
    
    # 尝试从报告管理器获取任务
    if report_manager:
        try:
            if hasattr(report_manager, 'get_generation_tasks'):
                tasks = report_manager.get_generation_tasks()
            elif hasattr(report_manager, 'list_tasks'):
                tasks = report_manager.list_tasks()
        except Exception as e:
            logger.debug(f"从报告管理器获取生成任务失败: {e}")
    
    # 尝试从报告生成器获取任务
    if not tasks and report_generator:
        try:
            if hasattr(report_generator, 'get_active_tasks'):
                tasks = report_generator.get_active_tasks()
        except Exception as e:
            logger.debug(f"从报告生成器获取生成任务失败: {e}")
    
    # 格式化任务数据
    if tasks:
        formatted_tasks = []
        for task in tasks:
            if not isinstance(task, dict):
                if hasattr(task, '__dict__'):
                    task_dict = task.__dict__
                elif hasattr(task, 'to_dict'):
                    task_dict = task.to_dict()
                else:
                    continue
            else:
                task_dict = task
            
            formatted_tasks.append({
                "task_id": task_dict.get('id', task_dict.get('task_id', '')),
                "template_name": task_dict.get('template_name', ''),
                "status": task_dict.get('status', 'unknown'),
                "progress": task_dict.get('progress', 0),
                "start_time": task_dict.get('start_time', int(datetime.now().timestamp()))
            })
        
        return formatted_tasks
    
    # 量化交易系统要求：不使用模拟数据，返回空列表
    return []


# ==================== 报告历史服务 ====================

def get_report_history() -> List[Dict[str, Any]]:
    """获取报告历史列表 - 从真实报告管理器获取，不使用模拟数据"""
    report_manager = get_report_manager()
    report_generator = get_report_generator()
    
    history = []
    
    # 尝试从报告管理器获取历史
    if report_manager:
        try:
            if hasattr(report_manager, 'get_report_history'):
                history = report_manager.get_report_history()
            elif hasattr(report_manager, 'list_reports'):
                history = report_manager.list_reports()
            elif hasattr(report_manager, 'get_all_reports'):
                history = report_manager.get_all_reports()
        except Exception as e:
            logger.debug(f"从报告管理器获取报告历史失败: {e}")
    
    # 尝试从报告生成器获取历史
    if not history and report_generator:
        try:
            if hasattr(report_generator, 'get_generated_reports'):
                history = report_generator.get_generated_reports()
        except Exception as e:
            logger.debug(f"从报告生成器获取报告历史失败: {e}")
    
    # 格式化历史数据
    if history:
        formatted_history = []
        for report in history:
            if not isinstance(report, dict):
                if hasattr(report, '__dict__'):
                    report_dict = report.__dict__
                elif hasattr(report, 'to_dict'):
                    report_dict = report.to_dict()
                else:
                    continue
            else:
                report_dict = report
            
            formatted_history.append({
                "id": report_dict.get('id', ''),
                "name": report_dict.get('name', ''),
                "report_type": report_dict.get('report_type', ''),
                "generated_at": report_dict.get('generated_at', report_dict.get('timestamp', 0)),
                "size": report_dict.get('size', 0),
                "generation_time": report_dict.get('generation_time', 0)
            })
        
        return formatted_history
    
    # 量化交易系统要求：不使用模拟数据，返回空列表
    return []


# ==================== 报告统计服务 ====================

def get_reporting_stats() -> Dict[str, Any]:
    """获取报告统计"""
    templates = get_report_templates()
    tasks = get_generation_tasks()
    history = get_report_history()
    
    generating_tasks = [t for t in tasks if t.get('status') == 'generating']
    
    generation_times = [h.get('generation_time', 0) for h in history if h.get('generation_time')]
    avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0.0
    
    return {
        "template_count": len(templates),
        "generating_tasks": len(generating_tasks),
        "history_count": len(history),
        "avg_generation_time": int(avg_generation_time)
    }


# ==================== 降级方案 ====================

def _get_mock_templates() -> List[Dict[str, Any]]:
    """获取模拟报告模板"""
    import random
    return [
        {
            "id": f"template_{i}",
            "name": random.choice(["每日风险报告", "周度风险报告", "月度风险报告", "实时风险报告"]),
            "report_type": random.choice(["daily", "weekly", "monthly", "realtime"]),
            "frequency": random.choice(["每日", "每周", "每月", "实时"]),
            "status": random.choice(["active", "inactive", "paused"]),
            "last_generated": int((datetime.now() - timedelta(days=random.randint(0, 7))).timestamp())
        }
        for i in range(1, 6)
    ]


def _get_mock_generation_tasks() -> List[Dict[str, Any]]:
    """获取模拟生成任务"""
    import random
    return [
        {
            "task_id": f"task_{i}",
            "template_name": random.choice(["每日风险报告", "周度风险报告", "月度风险报告"]),
            "status": random.choice(["generating", "completed", "failed", "pending"]),
            "progress": random.randint(0, 100),
            "start_time": int((datetime.now() - timedelta(minutes=random.randint(0, 60))).timestamp())
        }
        for i in range(1, 4)
    ]


def _get_mock_report_history() -> List[Dict[str, Any]]:
    """获取模拟报告历史"""
    import random
    return [
        {
            "id": f"report_{i}",
            "name": f"风险报告_{datetime.now().strftime('%Y%m%d')}_{i}",
            "report_type": random.choice(["daily", "weekly", "monthly"]),
            "generated_at": int((datetime.now() - timedelta(days=random.randint(0, 30))).timestamp()),
            "size": random.randint(100 * 1024, 5 * 1024 * 1024),  # 100KB to 5MB
            "generation_time": random.randint(1, 10)
        }
        for i in range(1, 21)
    ]


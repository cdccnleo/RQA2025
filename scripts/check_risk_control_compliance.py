#!/usr/bin/env python3
"""
风险控制流程架构符合性检查脚本

全面检查风险控制流程监控仪表盘的功能实现、持久化实现、架构设计符合性
以及与风险控制层的集成情况（6个业务流程步骤）
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查结果
check_results = {
    "timestamp": datetime.now().isoformat(),
    "frontend_modules": {},
    "backend_apis": {},
    "service_layer": {},
    "persistence": {},
    "architecture_compliance": {},
    "risk_layer_integration": {},
    "websocket_integration": {},
    "business_orchestration": {},
    "workflow_steps": {},
    "summary": {
        "total_items": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "not_implemented": 0
    }
}


def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(file_path)


def check_code_pattern(file_path: str, patterns: List[str], description: str, required_count: int = None) -> Dict[str, Any]:
    """检查代码中是否包含指定模式"""
    if not check_file_exists(file_path):
        return {
            "status": "failed",
            "message": f"文件不存在: {file_path}",
            "file": file_path
        }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found_patterns = []
        missing_patterns = []
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                found_patterns.append((pattern, len(matches)))
            else:
                missing_patterns.append(pattern)
        
        found_count = sum(count for _, count in found_patterns)
        total_count = len(patterns)
        
        if required_count is not None:
            if found_count >= required_count:
                return {
                    "status": "passed",
                    "message": f"{description}: 找到 {found_count}/{required_count} 个必需模式",
                    "found_patterns": found_patterns,
                    "found_count": found_count,
                    "required_count": required_count,
                    "file": file_path
                }
            else:
                return {
                    "status": "warning",
                    "message": f"{description}: 仅找到 {found_count}/{required_count} 个必需模式",
                    "found_patterns": found_patterns,
                    "missing_patterns": missing_patterns,
                    "found_count": found_count,
                    "required_count": required_count,
                    "file": file_path
                }
        
        if len(found_patterns) == len(patterns):
            return {
                "status": "passed",
                "message": f"{description}: 所有模式都找到",
                "found_patterns": found_patterns,
                "file": file_path
            }
        elif len(found_patterns) > 0:
            return {
                "status": "warning",
                "message": f"{description}: 部分模式找到 ({len(found_patterns)}/{len(patterns)})",
                "found_patterns": found_patterns,
                "missing_patterns": missing_patterns,
                "file": file_path
            }
        else:
            return {
                "status": "failed",
                "message": f"{description}: 未找到任何模式",
                "missing_patterns": missing_patterns,
                "file": file_path
            }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"读取文件失败: {str(e)}",
            "file": file_path
        }


def check_frontend_modules():
    """检查前端功能模块"""
    print("\n" + "="*80)
    print("1. 前端功能模块检查")
    print("="*80)
    
    frontend_checks = {}
    
    # 1.1 风险控制流程监控仪表盘
    print("\n1.1 风险控制流程监控仪表盘")
    dashboard_file = "web-static/risk-control-monitor.html"
    frontend_checks["dashboard_exists"] = {
        "file": dashboard_file,
        "exists": check_file_exists(dashboard_file),
        "status": "passed" if check_file_exists(dashboard_file) else "failed"
    }
    
    if check_file_exists(dashboard_file):
        # 检查统计卡片模块
        frontend_checks["statistics_cards"] = check_code_pattern(
            dashboard_file,
            [r"monitoring-coverage|avg-monitoring-latency|active-risk-alerts|current-var"],
            "统计卡片模块（实时监测覆盖、平均监测延迟、活跃风险告警、当前VaR）",
            required_count=4
        )
        
        # 检查6个业务流程步骤展示
        frontend_checks["workflow_steps"] = check_code_pattern(
            dashboard_file,
            [r"实时监测|风险评估|风险拦截|合规检查|风险报告|告警通知"],
            "6个业务流程步骤展示（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）",
            required_count=6
        )
        
        # 检查API集成
        frontend_checks["api_integration"] = check_code_pattern(
            dashboard_file,
            [r"/risk/control/overview|/risk/control/heatmap|/risk/control/timeline|/risk/control/alerts|/risk/control/stages",
             r"fetch\(|getApiBaseUrl"],
            "API集成（/api/v1/risk/control/*）",
            required_count=2
        )
        
        # 检查图表和可视化渲染
        frontend_checks["chart_rendering"] = check_code_pattern(
            dashboard_file,
            [r"varTrendChart|riskDistributionChart|Chart\.js|new Chart|risk-heatmap|risk-timeline"],
            "图表和可视化渲染（VaR趋势图、风险分布图、风险热力图、风险时间线）",
            required_count=4
        )
        
        # 检查流程步骤状态显示
        frontend_checks["step_status_display"] = check_code_pattern(
            dashboard_file,
            [r"status-indicator|showStageDetails|monitoring|assessment|interception|compliance|report|notification"],
            "流程步骤状态显示（6个步骤的状态和性能指标）",
            required_count=6
        )
    
    # 更新统计
    for check_name, check_result in frontend_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["frontend_modules"] = frontend_checks
    return frontend_checks


def check_backend_apis():
    """检查后端API端点"""
    print("\n" + "="*80)
    print("2. 后端API端点检查")
    print("="*80)
    
    api_checks = {}
    
    # 2.1 风险控制API路由
    print("\n2.1 风险控制API路由")
    routes_file = "src/gateway/web/risk_control_routes.py"
    routes_exists = check_file_exists(routes_file)
    api_checks["routes_file_exists"] = {
        "file": routes_file,
        "exists": routes_exists,
        "status": "passed" if routes_exists else "failed"
    }
    
    if routes_exists:
        # 检查API端点
        api_checks["api_endpoints"] = check_code_pattern(
            routes_file,
            [r"@router\.get\(.*/risk/control/overview|@router\.get\(.*/risk/control/heatmap|@router\.get\(.*/risk/control/timeline|@router\.get\(.*/risk/control/alerts|@router\.get\(.*/risk/control/stages"],
            "API端点实现（GET /api/v1/risk/control/overview, /heatmap, /timeline, /alerts, /stages/{stageId}）",
            required_count=3
        )
        
        # 检查服务层使用
        api_checks["service_layer_usage"] = check_code_pattern(
            routes_file,
            [r"from \.risk_control_service import|get_risk_control_data|risk_control_persistence"],
            "服务层封装使用",
            required_count=1
        )
        
        # 检查统一日志系统
        api_checks["unified_logger"] = check_code_pattern(
            routes_file,
            [r"get_unified_logger|统一日志"],
            "统一日志系统使用",
            required_count=1
        )
        
        # 检查事件总线集成
        api_checks["event_bus"] = check_code_pattern(
            routes_file,
            [r"EventBus|event_bus|\.publish\(|EventType\.RISK_CHECK_COMPLETED"],
            "事件总线集成（发布风险控制事件）",
            required_count=2
        )
        
        # 检查业务流程编排器
        api_checks["business_orchestrator"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|start_process|RISK_CONTROL"],
            "BusinessProcessOrchestrator业务流程编排（start_process, RISK_CONTROL）",
            required_count=2
        )
        
        # 检查WebSocket广播
        api_checks["websocket_broadcast"] = check_code_pattern(
            routes_file,
            [r"websocket_manager|_get_websocket_manager|manager\.broadcast|broadcast.*risk_control"],
            "WebSocket实时广播（manager.broadcast）",
            required_count=1
        )
        
        # 检查服务容器集成
        api_checks["service_container"] = check_code_pattern(
            routes_file,
            [r"ServiceContainer|DependencyContainer|container\.resolve|_get_container"],
            "ServiceContainer依赖注入",
            required_count=1
        )
    else:
        # 文件不存在，标记为失败
        api_checks["api_endpoints"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在，需要创建",
            "file": routes_file
        }
        check_results["summary"]["total_items"] += 1
        check_results["summary"]["failed"] += 1
    
    # 更新统计
    for check_name, check_result in api_checks.items():
        if isinstance(check_result, dict) and check_name != "routes_file_exists":
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["backend_apis"] = api_checks
    return api_checks


def check_service_layer():
    """检查服务层实现"""
    print("\n" + "="*80)
    print("3. 服务层实现检查")
    print("="*80)
    
    service_checks = {}
    
    service_file = "src/gateway/web/risk_control_service.py"
    service_exists = check_file_exists(service_file)
    service_checks["service_file_exists"] = {
        "file": service_file,
        "exists": service_exists,
        "status": "passed" if service_exists else "failed"
    }
    
    if service_exists:
        # 检查统一日志系统
        service_checks["unified_logger"] = check_code_pattern(
            service_file,
            [r"get_unified_logger|统一日志"],
            "统一日志系统使用",
            required_count=1
        )
        
        # 检查统一适配器工厂使用
        service_checks["adapter_factory_usage"] = check_code_pattern(
            service_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.RISK|统一适配器工厂"],
            "统一适配器工厂使用（风险控制层）",
            required_count=2
        )
        
        # 检查风险控制层适配器
        service_checks["risk_adapter"] = check_code_pattern(
            service_file,
            [r"_get_risk_adapter|risk_adapter|风险控制层适配器"],
            "风险控制层适配器获取（通过统一适配器工厂）",
            required_count=1
        )
        
        # 检查降级机制
        service_checks["fallback_mechanism"] = check_code_pattern(
            service_file,
            [r"降级方案|fallback|except.*ImportError|直接实例化|最终降级方案"],
            "降级服务机制（当风险控制层适配器不可用时的降级处理）",
            required_count=2
        )
        
        # 检查6个业务流程步骤数据收集
        service_checks["workflow_steps_collection"] = check_code_pattern(
            service_file,
            [r"realtime_monitoring|risk_assessment|risk_intercept|compliance_check|risk_report|alert_notify"],
            "6个业务流程步骤数据收集（实时监测、风险评估、风险拦截、合规检查、风险报告、告警通知）",
            required_count=6
        )
        
        # 检查流程状态映射
        service_checks["process_state_mapping"] = check_code_pattern(
            service_file,
            [r"step_state_mapping|MONITORING|RISK_ASSESSING|RISK_INTERCEPTING|COMPLIANCE_CHECKING|REPORT_GENERATING|ALERT_NOTIFYING"],
            "流程状态映射（6个步骤与BusinessProcessState的映射关系）",
            required_count=3
        )
    else:
        # 文件不存在，标记为失败
        service_checks["unified_logger"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在，需要创建",
            "file": service_file
        }
        check_results["summary"]["total_items"] += 1
        check_results["summary"]["failed"] += 1
    
    # 更新统计
    for check_name, check_result in service_checks.items():
        if isinstance(check_result, dict) and check_name != "service_file_exists":
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["service_layer"] = service_checks
    return service_checks


def check_persistence():
    """检查持久化实现"""
    print("\n" + "="*80)
    print("4. 持久化实现检查")
    print("="*80)
    
    persistence_checks = {}
    
    # 4.1 风险控制记录持久化
    print("\n4.1 风险控制记录持久化")
    persistence_file = "src/gateway/web/risk_control_persistence.py"
    persistence_exists = check_file_exists(persistence_file)
    persistence_checks["persistence_file_exists"] = {
        "file": persistence_file,
        "exists": persistence_exists,
        "status": "passed" if persistence_exists else "failed"
    }
    
    if persistence_exists:
        # 检查文件系统持久化
        persistence_checks["file_persistence"] = check_code_pattern(
            persistence_file,
            [r"save_risk_control_record|json\.dump|文件系统|RISK_CONTROL_DIR"],
            "文件系统持久化（JSON格式）",
            required_count=3
        )
        
        # 检查PostgreSQL持久化
        persistence_checks["postgresql_persistence"] = check_code_pattern(
            persistence_file,
            [r"_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*risk_control_records"],
            "PostgreSQL持久化",
            required_count=2
        )
        
        # 检查6个步骤数据字段
        persistence_checks["workflow_steps_fields"] = check_code_pattern(
            persistence_file,
            [r"realtime_monitoring|risk_assessment|risk_intercept|compliance_check|risk_report|alert_notify"],
            "6个步骤数据字段（realtime_monitoring, risk_assessment, risk_intercept, compliance_check, risk_report, alert_notify）",
            required_count=6
        )
        
        # 检查统一日志系统
        persistence_checks["unified_logger"] = check_code_pattern(
            persistence_file,
            [r"get_unified_logger|统一日志"],
            "统一日志系统使用",
            required_count=1
        )
    else:
        # 文件不存在，标记为失败
        persistence_checks["file_persistence"] = {
            "status": "failed",
            "message": "风险控制持久化文件不存在，需要创建",
            "file": persistence_file
        }
        check_results["summary"]["total_items"] += 1
        check_results["summary"]["failed"] += 1
    
    # 更新统计
    for check_name, check_result in persistence_checks.items():
        if isinstance(check_result, dict) and check_name != "persistence_file_exists":
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["persistence"] = persistence_checks
    return persistence_checks


def check_architecture_compliance():
    """检查架构设计符合性"""
    print("\n" + "="*80)
    print("5. 架构设计符合性检查")
    print("="*80)
    
    compliance_checks = {}
    
    routes_file = "src/gateway/web/risk_control_routes.py"
    service_file = "src/gateway/web/risk_control_service.py"
    
    # 5.1 基础设施层符合性
    print("\n5.1 基础设施层符合性")
    
    # 检查统一日志系统使用（如果文件存在）
    if check_file_exists(routes_file):
        compliance_checks["infrastructure_logger"] = check_code_pattern(
            routes_file,
            [r"get_unified_logger"],
            "基础设施层统一日志系统集成",
            required_count=1
        )
    else:
        compliance_checks["infrastructure_logger"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
    
    # 5.2 核心服务层符合性
    print("\n5.2 核心服务层符合性")
    
    if check_file_exists(routes_file):
        # 检查EventBus使用
        compliance_checks["event_bus_usage"] = check_code_pattern(
            routes_file,
            [r"EventBus|event_bus|\.publish\(|EventType\."],
            "EventBus事件驱动通信",
            required_count=2
        )
        
        # 检查ServiceContainer使用
        compliance_checks["service_container"] = check_code_pattern(
            routes_file,
            [r"ServiceContainer|DependencyContainer|container\.resolve|_get_container"],
            "ServiceContainer依赖注入",
            required_count=1
        )
        
        # 检查BusinessProcessOrchestrator使用
        compliance_checks["business_orchestrator"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator"],
            "BusinessProcessOrchestrator业务流程编排",
            required_count=1
        )
    else:
        compliance_checks["event_bus_usage"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
        compliance_checks["service_container"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
        compliance_checks["business_orchestrator"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
    
    # 5.3 风险控制层符合性
    print("\n5.3 风险控制层符合性")
    
    if check_file_exists(service_file):
        # 检查统一适配器工厂使用
        compliance_checks["adapter_factory_usage"] = check_code_pattern(
            service_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.RISK|统一适配器工厂"],
            "统一适配器工厂使用（风险控制层）",
            required_count=2
        )
        
        # 检查风险控制层组件访问
        compliance_checks["risk_layer_access"] = check_code_pattern(
            service_file,
            [r"风险控制层适配器|risk_adapter|get_risk_manager|get_risk_monitor|get_risk_calculator|get_alert_system"],
            "风险控制层组件访问（通过适配器）",
            required_count=2
        )
    else:
        compliance_checks["adapter_factory_usage"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
        compliance_checks["risk_layer_access"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
    
    # 更新统计
    for check_name, check_result in compliance_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["architecture_compliance"] = compliance_checks
    return compliance_checks


def check_risk_layer_integration():
    """检查风险控制层集成"""
    print("\n" + "="*80)
    print("6. 风险控制层集成检查")
    print("="*80)
    
    risk_checks = {}
    
    service_file = "src/gateway/web/risk_control_service.py"
    
    if check_file_exists(service_file):
        # 检查统一适配器工厂使用
        risk_checks["adapter_factory_usage"] = check_code_pattern(
            service_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.RISK|统一适配器工厂"],
            "通过统一适配器工厂访问风险控制层",
            required_count=1
        )
        
        # 检查风险控制层适配器获取
        risk_checks["risk_adapter_access"] = check_code_pattern(
            service_file,
            [r"_get_risk_adapter|risk_adapter|风险控制层适配器"],
            "风险控制层适配器获取（通过统一适配器工厂）",
            required_count=1
        )
        
        # 检查风险控制层组件使用
        risk_checks["risk_components_usage"] = check_code_pattern(
            service_file,
            [r"adapter\.get_risk_manager|adapter\.get_risk_monitor|adapter\.get_risk_calculator|adapter\.get_alert_system"],
            "风险控制层组件使用（RiskManager, RealTimeRiskMonitor, RiskCalculationEngine, AlertSystem）",
            required_count=2
        )
    else:
        risk_checks["adapter_factory_usage"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
        risk_checks["risk_adapter_access"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
        risk_checks["risk_components_usage"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
    
    # 更新统计
    for check_name, check_result in risk_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["risk_layer_integration"] = risk_checks
    return risk_checks


def check_websocket():
    """检查WebSocket实时更新"""
    print("\n" + "="*80)
    print("7. WebSocket实时更新检查")
    print("="*80)
    
    websocket_checks = {}
    
    # 7.1 WebSocket端点
    print("\n7.1 WebSocket端点实现")
    websocket_routes = "src/gateway/web/websocket_routes.py"
    if check_file_exists(websocket_routes):
        websocket_checks["websocket_endpoint"] = check_code_pattern(
            websocket_routes,
            [r"/ws/risk-control|websocket_risk_control"],
            "WebSocket端点注册（/ws/risk-control）",
            required_count=1
        )
    
    # 7.2 WebSocket管理器
    print("\n7.2 WebSocket管理器")
    websocket_manager_file = "src/gateway/web/websocket_manager.py"
    if check_file_exists(websocket_manager_file):
        websocket_checks["websocket_manager"] = check_code_pattern(
            websocket_manager_file,
            [r"_broadcast_risk_control|risk_control|get_risk_control_data"],
            "风险控制WebSocket广播实现",
            required_count=2
        )
    
    # 7.3 前端WebSocket处理
    print("\n7.3 前端WebSocket处理")
    dashboard_file = "web-static/risk-control-monitor.html"
    if check_file_exists(dashboard_file):
        websocket_checks["frontend_websocket"] = check_code_pattern(
            dashboard_file,
            [r"/ws/risk-control|connectRiskWebSocket|riskWebSocket|onmessage|risk_control_event"],
            "前端WebSocket消息处理（/ws/risk-control）",
            required_count=2
        )
    
    # 更新统计
    for check_name, check_result in websocket_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["websocket_integration"] = websocket_checks
    return websocket_checks


def check_6_workflow_steps():
    """检查6个业务流程步骤的完整实现"""
    print("\n" + "="*80)
    print("8. 6个业务流程步骤检查")
    print("="*80)
    
    workflow_checks = {}
    
    service_file = "src/gateway/web/risk_control_service.py"
    routes_file = "src/gateway/web/risk_control_routes.py"
    
    if check_file_exists(service_file):
        # 检查步骤1: 实时监测 (Real-time Monitoring)
        workflow_checks["step1_realtime_monitoring"] = check_code_pattern(
            service_file,
            [r"realtime_monitoring|get_risk_monitor|实时监测|MONITORING"],
            "步骤1: 实时监测（Real-time Monitoring）",
            required_count=2
        )
        
        # 检查步骤2: 风险评估 (Risk Assessment)
        workflow_checks["step2_risk_assessment"] = check_code_pattern(
            service_file,
            [r"risk_assessment|RISK_ASSESSMENT_COMPLETED|EventType\.RISK_ASSESSMENT_COMPLETED|get_risk_calculator|风险评估"],
            "步骤2: 风险评估（Risk Assessment）",
            required_count=2
        )
        
        # 检查步骤3: 风险拦截 (Risk Intercept)
        workflow_checks["step3_risk_intercept"] = check_code_pattern(
            service_file,
            [r"risk_intercept|RISK_INTERCEPTED|EventType\.RISK_INTERCEPTED|风险拦截"],
            "步骤3: 风险拦截（Risk Intercept）",
            required_count=2
        )
        
        # 检查步骤4: 合规检查 (Compliance Check)
        workflow_checks["step4_compliance_check"] = check_code_pattern(
            service_file,
            [r"compliance_check|COMPLIANCE_CHECK_COMPLETED|EventType\.COMPLIANCE_CHECK_COMPLETED|合规检查"],
            "步骤4: 合规检查（Compliance Check）",
            required_count=2
        )
        
        # 检查步骤5: 风险报告 (Risk Report)
        workflow_checks["step5_risk_report"] = check_code_pattern(
            service_file,
            [r"risk_report|RISK_REPORT_GENERATED|EventType\.RISK_REPORT_GENERATED|风险报告"],
            "步骤5: 风险报告（Risk Report）",
            required_count=2
        )
        
        # 检查步骤6: 告警通知 (Alert Notify)
        workflow_checks["step6_alert_notify"] = check_code_pattern(
            service_file,
            [r"alert_notify|ALERT_TRIGGERED|EventType\.ALERT_TRIGGERED|get_alert_system|告警通知"],
            "步骤6: 告警通知（Alert Notify）",
            required_count=3
        )
        
        # 检查6个步骤的状态映射
        workflow_checks["step_state_mapping"] = check_code_pattern(
            service_file,
            [r"step_state_mapping|MONITORING|RISK_ASSESSING|RISK_INTERCEPTING|COMPLIANCE_CHECKING|REPORT_GENERATING|ALERT_NOTIFYING"],
            "6个步骤与流程状态的映射关系",
            required_count=4
        )
    else:
        # 如果服务文件不存在，所有步骤检查都失败
        for i in range(1, 7):
            step_name = f"step{i}_"
            if i == 1:
                step_name += "realtime_monitoring"
            elif i == 2:
                step_name += "risk_assessment"
            elif i == 3:
                step_name += "risk_intercept"
            elif i == 4:
                step_name += "compliance_check"
            elif i == 5:
                step_name += "risk_report"
            elif i == 6:
                step_name += "alert_notify"
            
            workflow_checks[step_name] = {
                "status": "failed",
                "message": "风险控制服务层文件不存在",
                "file": service_file
            }
            check_results["summary"]["total_items"] += 1
            check_results["summary"]["failed"] += 1
    
    # 更新统计
    for check_name, check_result in workflow_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            # 跳过在else分支中已经统计过的失败项
            if status != "failed" or check_name == "step_state_mapping":
                check_results["summary"]["total_items"] += 1
                if status == "passed":
                    check_results["summary"]["passed"] += 1
                elif status == "failed":
                    check_results["summary"]["failed"] += 1
                elif status == "warning":
                    check_results["summary"]["warnings"] += 1
    
    check_results["workflow_steps"] = workflow_checks
    return workflow_checks


def check_business_orchestration():
    """检查业务流程编排"""
    print("\n" + "="*80)
    print("9. 业务流程编排检查")
    print("="*80)
    
    orchestration_checks = {}
    
    routes_file = "src/gateway/web/risk_control_routes.py"
    service_file = "src/gateway/web/risk_control_service.py"
    
    # 检查BusinessProcessOrchestrator使用（符合架构设计：业务流程编排器用于管理风险控制流程）
    if check_file_exists(routes_file):
        orchestration_checks["orchestrator_usage"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器"],
            "BusinessProcessOrchestrator使用",
            required_count=2
        )
        
        # 检查流程状态管理（业务流程编排器用于管理风险控制流程）
        orchestration_checks["process_management"] = check_code_pattern(
            routes_file,
            [r"start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|RISK_CONTROL"],
            "流程状态管理（风险控制流程状态管理）",
            required_count=2
        )
    else:
        orchestration_checks["orchestrator_usage"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
        orchestration_checks["process_management"] = {
            "status": "failed",
            "message": "风险控制API路由文件不存在",
            "file": routes_file
        }
    
    # 检查6个步骤的事件发布
    if check_file_exists(service_file):
        orchestration_checks["event_publishing"] = check_code_pattern(
            service_file,
            [r"EventBus\.publish|event_bus\.publish|RISK_ASSESSMENT_COMPLETED|RISK_INTERCEPTED|COMPLIANCE_CHECK_COMPLETED|RISK_REPORT_GENERATED|ALERT_TRIGGERED|ALERT_RESOLVED"],
            "风险控制流程事件发布（6个步骤的事件）",
            required_count=4
        )
        
        # 检查流程状态机集成
        orchestration_checks["state_machine_integration"] = check_code_pattern(
            service_file,
            [r"get_current_state|state_machine|process_state|state_history|流程状态机"],
            "流程状态机集成（获取当前状态、状态历史）",
            required_count=2
        )
    else:
        orchestration_checks["event_publishing"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
        orchestration_checks["state_machine_integration"] = {
            "status": "failed",
            "message": "风险控制服务层文件不存在",
            "file": service_file
        }
    
    # 更新统计
    for check_name, check_result in orchestration_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["business_orchestration"] = orchestration_checks
    return orchestration_checks


def generate_report():
    """生成检查报告"""
    print("\n" + "="*80)
    print("生成检查报告")
    print("="*80)
    
    total = check_results["summary"]["total_items"]
    passed = check_results["summary"]["passed"]
    failed = check_results["summary"]["failed"]
    warnings = check_results["summary"]["warnings"]
    not_implemented = check_results["summary"]["not_implemented"]
    
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    report_file = project_root / "docs" / f"risk_control_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 风险控制流程架构符合性检查报告\n\n")
        f.write(f"**检查时间**: {check_results['timestamp']}\n\n")
        f.write("## 检查摘要\n\n")
        f.write(f"- **总检查项**: {total}\n")
        f.write(f"- **通过**: {passed} ✅\n")
        f.write(f"- **失败**: {failed} ❌\n")
        f.write(f"- **警告**: {warnings} ⚠️\n")
        f.write(f"- **未实现**: {not_implemented} 📋\n")
        f.write(f"- **通过率**: {pass_rate:.2f}%\n\n")
        
        # 各分类详细报告
        categories = [
            ("frontend_modules", "1. 前端功能模块检查"),
            ("backend_apis", "2. 后端API端点检查"),
            ("service_layer", "3. 服务层实现检查"),
            ("persistence", "4. 持久化实现检查"),
            ("architecture_compliance", "5. 架构设计符合性检查"),
            ("risk_layer_integration", "6. 风险控制层集成检查"),
            ("websocket_integration", "7. WebSocket实时更新检查"),
            ("workflow_steps", "8. 6个业务流程步骤检查"),
            ("business_orchestration", "9. 业务流程编排检查")
        ]
        
        for category_key, category_title in categories:
            f.write(f"## {category_title}\n\n")
            category_data = check_results.get(category_key, {})
            
            for check_name, check_result in category_data.items():
                status_icon = "✅" if isinstance(check_result, dict) and check_result.get("status") == "passed" else \
                             "❌" if isinstance(check_result, dict) and check_result.get("status") == "failed" else \
                             "⚠️" if isinstance(check_result, dict) and check_result.get("status") == "warning" else \
                             "📋" if isinstance(check_result, dict) and check_result.get("status") == "not_implemented" else "❓"
                
                f.write(f"### {check_name} {status_icon}\n\n")
                if isinstance(check_result, dict):
                    f.write(f"- **文件**: {check_result.get('file', 'N/A')}\n")
                    f.write(f"- **状态**: {check_result.get('status', 'unknown')}\n")
                    if 'message' in check_result:
                        f.write(f"- **消息**: {check_result['message']}\n")
                    if 'found_count' in check_result and 'required_count' in check_result:
                        f.write(f"- **匹配情况**: {check_result['found_count']}/{check_result['required_count']}\n")
                f.write("\n")
        
        f.write("## 详细检查结果\n\n")
        f.write("```json\n")
        f.write(json.dumps(check_results, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    
    print(f"\n检查报告已生成: {report_file}")
    return report_file


def main():
    """主函数"""
    print("="*80)
    print("风险控制流程架构符合性检查")
    print("="*80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_service_layer()
    check_persistence()
    check_architecture_compliance()
    check_risk_layer_integration()
    check_websocket()
    check_6_workflow_steps()
    check_business_orchestration()
    
    # 生成报告
    report_file = generate_report()
    
    # 打印摘要
    print("\n" + "="*80)
    print("检查摘要")
    print("="*80)
    print(f"总检查项: {check_results['summary']['total_items']}")
    print(f"通过: {check_results['summary']['passed']} ✅")
    print(f"失败: {check_results['summary']['failed']} ❌")
    print(f"警告: {check_results['summary']['warnings']} ⚠️")
    print(f"未实现: {check_results['summary']['not_implemented']} 📋")
    pass_rate = (check_results['summary']['passed'] / check_results['summary']['total_items'] * 100) if check_results['summary']['total_items'] > 0 else 0
    print(f"通过率: {pass_rate:.2f}%")
    print(f"\n详细报告: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
策略执行监控仪表盘架构符合性检查脚本

全面检查策略执行监控仪表盘的功能实现、持久化实现、架构设计符合性
以及与策略层和交易层的集成情况
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
    "strategy_layer_integration": {},
    "trading_layer_integration": {},
    "websocket_integration": {},
    "business_orchestration": {},
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
    
    # 1.1 策略执行监控仪表盘
    print("\n1.1 策略执行监控仪表盘")
    dashboard_file = "web-static/strategy-execution-monitor.html"
    frontend_checks["dashboard_exists"] = {
        "file": dashboard_file,
        "exists": check_file_exists(dashboard_file),
        "status": "passed" if check_file_exists(dashboard_file) else "failed"
    }
    
    if check_file_exists(dashboard_file):
        # 检查统计卡片模块
        frontend_checks["statistics_cards"] = check_code_pattern(
            dashboard_file,
            [r"running-strategies|avg-latency|today-signals|total-trades"],
            "统计卡片模块（运行中策略、平均延迟、今日信号数、总交易数）",
            required_count=4
        )
        
        # 检查API集成
        frontend_checks["api_integration"] = check_code_pattern(
            dashboard_file,
            [r"/strategy/execution/status|/strategy/execution/metrics|/strategy/realtime/signals",
             r"fetch\(|getApiBaseUrl"],
            "API集成（/strategy/execution/status, /strategy/execution/metrics, /strategy/realtime/signals）",
            required_count=2
        )
        
        # 检查WebSocket实时更新集成
        frontend_checks["websocket_integration"] = check_code_pattern(
            dashboard_file,
            [r"/ws/execution-status|connectWebSocket|ws\.onmessage"],
            "WebSocket实时更新集成（/ws/execution-status）",
            required_count=2
        )
        
        # 检查图表和可视化渲染
        frontend_checks["chart_rendering"] = check_code_pattern(
            dashboard_file,
            [r"latencyChart|throughputChart|Chart\.js|new Chart"],
            "图表和可视化渲染（延迟趋势图、吞吐量趋势图）",
            required_count=3
        )
        
        # 检查功能模块完整性
        frontend_checks["function_modules"] = check_code_pattern(
            dashboard_file,
            [r"策略执行列表|最近信号|toggleStrategy|viewStrategyDetails"],
            "功能模块完整性（策略执行状态表格、实时交易信号列表、策略操作功能）",
            required_count=4
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
    
    # 2.1 策略执行API路由
    print("\n2.1 策略执行API路由")
    routes_file = "src/gateway/web/strategy_execution_routes.py"
    if check_file_exists(routes_file):
        # 检查API端点
        api_checks["api_endpoints"] = check_code_pattern(
            routes_file,
            [r"@router\.get\(.*/strategy/execution/status|@router\.get\(.*/strategy/execution/metrics",
             r"@router\.post\(.*/strategy/execution/.*start|@router\.post\(.*/strategy/execution/.*pause"],
            "API端点实现（GET /strategy/execution/status, GET /strategy/execution/metrics, POST /strategy/execution/{strategy_id}/start, POST /strategy/execution/{strategy_id}/pause）",
            required_count=2
        )
        
        # 检查服务层使用
        api_checks["service_layer_usage"] = check_code_pattern(
            routes_file,
            [r"from \.strategy_execution_service import|get_strategy_execution_status|get_execution_metrics|start_strategy|pause_strategy"],
            "服务层封装使用",
            required_count=2
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
            [r"EventBus|event_bus|\.publish\(|EventType\.EXECUTION_STARTED|EventType\.EXECUTION_COMPLETED"],
            "事件总线集成（发布EXECUTION_STARTED, EXECUTION_COMPLETED事件）",
            required_count=2
        )
        
        # 检查业务流程编排器
        api_checks["business_orchestrator"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|start_process|update_process_state"],
            "BusinessProcessOrchestrator业务流程编排（start_process, update_process_state）",
            required_count=2
        )
        
        # 检查WebSocket广播
        api_checks["websocket_broadcast"] = check_code_pattern(
            routes_file,
            [r"websocket_manager|_get_websocket_manager|manager\.broadcast|broadcast.*execution"],
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
    
    # 更新统计
    for check_name, check_result in api_checks.items():
        if isinstance(check_result, dict):
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
    
    service_file = "src/gateway/web/strategy_execution_service.py"
    if check_file_exists(service_file):
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
            [r"get_unified_adapter_factory|BusinessLayerType\.STRATEGY|BusinessLayerType\.TRADING"],
            "统一适配器工厂使用（策略层和交易层）",
            required_count=2
        )
        
        # 检查策略层适配器
        service_checks["strategy_adapter"] = check_code_pattern(
            service_file,
            [r"_get_strategy_adapter|strategy_adapter|策略层适配器"],
            "策略层适配器获取",
            required_count=1
        )
        
        # 检查交易层适配器
        service_checks["trading_adapter"] = check_code_pattern(
            service_file,
            [r"_get_trading_adapter|trading_adapter|交易层适配器"],
            "交易层适配器获取",
            required_count=1
        )
        
        # 检查降级机制
        service_checks["fallback_mechanism"] = check_code_pattern(
            service_file,
            [r"降级方案|fallback|except.*ImportError|直接实例化|RealTimeStrategyEngine"],
            "降级服务机制（当策略层适配器不可用时的降级处理）",
            required_count=2
        )
        
        # 检查实时策略引擎封装
        service_checks["realtime_engine"] = check_code_pattern(
            service_file,
            [r"get_realtime_engine|RealTimeStrategyEngine|实时策略引擎"],
            "实时策略引擎封装",
            required_count=2
        )
        
        # 检查持久化集成
        service_checks["persistence_integration"] = check_code_pattern(
            service_file,
            [r"execution_persistence|save_execution_state|list_execution_states|持久化存储"],
            "持久化集成",
            required_count=2
        )
    
    # 更新统计
    for check_name, check_result in service_checks.items():
        if isinstance(check_result, dict):
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
    
    # 4.1 执行状态持久化
    print("\n4.1 执行状态持久化")
    persistence_file = "src/gateway/web/execution_persistence.py"
    if check_file_exists(persistence_file):
        # 检查文件系统持久化
        persistence_checks["file_persistence"] = check_code_pattern(
            persistence_file,
            [r"save_execution_state|json\.dump|文件系统|EXECUTION_STATES_DIR"],
            "文件系统持久化（JSON格式）",
            required_count=3
        )
        
        # 检查PostgreSQL持久化
        persistence_checks["postgresql_persistence"] = check_code_pattern(
            persistence_file,
            [r"_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*strategy_execution_states"],
            "PostgreSQL持久化",
            required_count=2
        )
        
        # 检查双重存储机制
        persistence_checks["dual_storage"] = check_code_pattern(
            persistence_file,
            [r"优先.*PostgreSQL|如果.*PostgreSQL|故障转移|fallback|return None|文件系统"],
            "双重存储机制（PostgreSQL优先，文件系统降级）",
            required_count=2
        )
        
        # 检查执行状态CRUD操作
        persistence_checks["crud_operations"] = check_code_pattern(
            persistence_file,
            [r"save_execution_state|load_execution_state|update_execution_state|delete_execution_state|list_execution_states"],
            "执行状态CRUD操作",
            required_count=4
        )
        
        # 检查统一日志系统
        persistence_checks["unified_logger"] = check_code_pattern(
            persistence_file,
            [r"get_unified_logger|统一日志"],
            "统一日志系统使用",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in persistence_checks.items():
        if isinstance(check_result, dict):
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
    
    routes_file = "src/gateway/web/strategy_execution_routes.py"
    service_file = "src/gateway/web/strategy_execution_service.py"
    
    # 5.1 基础设施层符合性
    print("\n5.1 基础设施层符合性")
    
    # 检查统一日志系统使用（所有文件）
    compliance_checks["infrastructure_logger"] = check_code_pattern(
        routes_file,
        [r"get_unified_logger"],
        "基础设施层统一日志系统集成",
        required_count=1
    )
    
    # 5.2 核心服务层符合性
    print("\n5.2 核心服务层符合性")
    
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
    
    # 5.3 策略层和交易层符合性
    print("\n5.3 策略层和交易层符合性")
    
    # 检查统一适配器工厂使用
    compliance_checks["adapter_factory_usage"] = check_code_pattern(
        service_file,
        [r"get_unified_adapter_factory|BusinessLayerType\.STRATEGY|BusinessLayerType\.TRADING"],
        "统一适配器工厂使用（策略层和交易层）",
        required_count=2
    )
    
    # 检查策略层组件访问
    compliance_checks["strategy_layer_access"] = check_code_pattern(
        service_file,
        [r"策略层适配器|strategy_adapter|RealTimeStrategyEngine|策略执行服务"],
        "策略层组件访问（通过适配器）",
        required_count=1
    )
    
    # 检查交易层组件访问
    compliance_checks["trading_layer_access"] = check_code_pattern(
        service_file,
        [r"交易层适配器|trading_adapter|交易信号服务|交易层服务"],
        "交易层组件访问（通过适配器）",
        required_count=1
    )
    
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


def check_strategy_layer_integration():
    """检查策略层集成"""
    print("\n" + "="*80)
    print("6. 策略层集成检查")
    print("="*80)
    
    strategy_checks = {}
    
    service_file = "src/gateway/web/strategy_execution_service.py"
    
    if check_file_exists(service_file):
        # 检查统一适配器工厂使用
        strategy_checks["adapter_factory_usage"] = check_code_pattern(
            service_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.STRATEGY|统一适配器工厂"],
            "通过统一适配器工厂访问策略层",
            required_count=1
        )
        
        # 检查策略层适配器获取
        strategy_checks["strategy_adapter_access"] = check_code_pattern(
            service_file,
            [r"_get_strategy_adapter|strategy_adapter|策略层适配器"],
            "策略层适配器获取",
            required_count=1
        )
        
        # 检查实时策略引擎使用
        strategy_checks["realtime_engine_usage"] = check_code_pattern(
            service_file,
            [r"RealTimeStrategyEngine|get_realtime_engine|实时策略引擎"],
            "实时策略引擎使用（通过适配器或降级方案）",
            required_count=1
        )
        
        # 检查策略执行状态集成
        strategy_checks["execution_state_integration"] = check_code_pattern(
            service_file,
            [r"get_strategy_execution_status|策略执行状态|执行状态集成"],
            "策略执行状态集成（从实时引擎获取策略执行状态）",
            required_count=1
        )
        
        # 检查策略性能指标集成
        strategy_checks["performance_metrics_integration"] = check_code_pattern(
            service_file,
            [r"get_execution_metrics|延迟|吞吐量|信号数|性能指标"],
            "策略性能指标集成（延迟、吞吐量、信号数等）",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in strategy_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["strategy_layer_integration"] = strategy_checks
    return strategy_checks


def check_trading_layer_integration():
    """检查交易层集成"""
    print("\n" + "="*80)
    print("7. 交易层集成检查")
    print("="*80)
    
    trading_checks = {}
    
    routes_file = "src/gateway/web/strategy_execution_routes.py"
    service_file = "src/gateway/web/strategy_execution_service.py"
    
    if check_file_exists(service_file):
        # 检查统一适配器工厂使用
        trading_checks["adapter_factory_usage"] = check_code_pattern(
            service_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.TRADING|统一适配器工厂"],
            "通过统一适配器工厂访问交易层",
            required_count=1
        )
        
        # 检查交易层适配器获取
        trading_checks["trading_adapter_access"] = check_code_pattern(
            service_file,
            [r"_get_trading_adapter|trading_adapter|交易层适配器"],
            "交易层适配器获取",
            required_count=1
        )
    
    # 检查实时交易信号集成（在routes中）
    if check_file_exists(routes_file):
        trading_checks["realtime_signals_integration"] = check_code_pattern(
            routes_file,
            [r"get_realtime_signals|trading_signal_service|交易信号|最近信号"],
            "实时交易信号集成（从交易信号服务获取最近信号）",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in trading_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["trading_layer_integration"] = trading_checks
    return trading_checks


def check_websocket():
    """检查WebSocket实时更新"""
    print("\n" + "="*80)
    print("8. WebSocket实时更新检查")
    print("="*80)
    
    websocket_checks = {}
    
    # 8.1 WebSocket端点
    print("\n8.1 WebSocket端点实现")
    websocket_routes = "src/gateway/web/websocket_routes.py"
    if check_file_exists(websocket_routes):
        websocket_checks["websocket_endpoint"] = check_code_pattern(
            websocket_routes,
            [r"/ws/execution-status|websocket_execution_status"],
            "WebSocket端点注册（/ws/execution-status）",
            required_count=1
        )
    
    # 8.2 WebSocket管理器
    print("\n8.2 WebSocket管理器")
    websocket_manager_file = "src/gateway/web/websocket_manager.py"
    if check_file_exists(websocket_manager_file):
        websocket_checks["websocket_manager"] = check_code_pattern(
            websocket_manager_file,
            [r"_broadcast_execution_status|execution_status|get_strategy_execution_status"],
            "执行状态WebSocket广播实现",
            required_count=2
        )
    
    # 8.3 前端WebSocket处理
    print("\n8.3 前端WebSocket处理")
    dashboard_file = "web-static/strategy-execution-monitor.html"
    if check_file_exists(dashboard_file):
        websocket_checks["frontend_websocket"] = check_code_pattern(
            dashboard_file,
            [r"/ws/execution-status|connectWebSocket|onmessage|execution_status"],
            "前端WebSocket消息处理",
            required_count=3
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


def check_business_orchestration():
    """检查业务流程编排"""
    print("\n" + "="*80)
    print("9. 业务流程编排检查")
    print("="*80)
    
    orchestration_checks = {}
    
    routes_file = "src/gateway/web/strategy_execution_routes.py"
    
    # 检查BusinessProcessOrchestrator使用（符合架构设计：业务流程编排器用于管理策略执行流程）
    orchestration_checks["orchestrator_usage"] = check_code_pattern(
        routes_file,
        [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器"],
        "BusinessProcessOrchestrator使用",
        required_count=2
    )
    
    # 检查流程状态管理（业务流程编排器用于管理策略执行流程）
    orchestration_checks["process_management"] = check_code_pattern(
        routes_file,
        [r"start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|STRATEGY_EXECUTION"],
        "流程状态管理（策略执行流程状态管理）",
        required_count=2
    )
    
    # 检查执行流程事件发布
    orchestration_checks["event_publishing"] = check_code_pattern(
        routes_file,
        [r"EventBus\.publish|event_bus\.publish|EXECUTION_STARTED|EXECUTION_COMPLETED|SIGNAL_GENERATED|执行.*事件"],
        "执行流程事件发布（EXECUTION_STARTED, EXECUTION_COMPLETED, SIGNAL_GENERATED）",
        required_count=2
    )
    
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
    
    report_file = project_root / "docs" / f"strategy_execution_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 策略执行监控仪表盘架构符合性检查报告\n\n")
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
            ("strategy_layer_integration", "6. 策略层集成检查"),
            ("trading_layer_integration", "7. 交易层集成检查"),
            ("websocket_integration", "8. WebSocket实时更新检查"),
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
    import sys
    
    print("="*80)
    print("策略执行监控仪表盘架构符合性检查")
    print("="*80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_service_layer()
    check_persistence()
    check_architecture_compliance()
    check_strategy_layer_integration()
    check_trading_layer_integration()
    check_websocket()
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
#!/usr/bin/env python3
"""
交易执行全流程架构符合性检查脚本

全面检查交易执行全流程监控仪表盘的功能实现、持久化实现、架构设计符合性
以及与交易层的集成情况（8个业务流程步骤）
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
    "trading_layer_integration": {},
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
    
    # 1.1 交易执行全流程监控仪表盘
    print("\n1.1 交易执行全流程监控仪表盘")
    dashboard_file = "web-static/trading-execution.html"
    frontend_checks["dashboard_exists"] = {
        "file": dashboard_file,
        "exists": check_file_exists(dashboard_file),
        "status": "passed" if check_file_exists(dashboard_file) else "failed"
    }
    
    if check_file_exists(dashboard_file):
        # 检查统计卡片模块
        frontend_checks["statistics_cards"] = check_code_pattern(
            dashboard_file,
            [r"today-signals|pending-orders|today-trades|portfolio-value"],
            "统计卡片模块（今日信号、待处理订单、今日交易、投资组合价值）",
            required_count=4
        )
        
        # 检查8个业务流程步骤展示
        frontend_checks["workflow_steps"] = check_code_pattern(
            dashboard_file,
            [r"市场监控|信号生成|风险检查|订单生成|智能路由|成交执行|结果反馈|持仓管理"],
            "8个业务流程步骤展示（市场监控、信号生成、风险检查、订单生成、智能路由、成交执行、结果反馈、持仓管理）",
            required_count=8
        )
        
        # 检查API集成
        frontend_checks["api_integration"] = check_code_pattern(
            dashboard_file,
            [r"/trading/execution/flow|/trading/overview",
             r"fetch\(|getApiBaseUrl"],
            "API集成（/api/v1/trading/execution/flow, /api/v1/trading/overview）",
            required_count=2
        )
        
        # 检查WebSocket实时更新集成
        frontend_checks["websocket_integration"] = check_code_pattern(
            dashboard_file,
            [r"/ws/trading-execution|connectExecutionWebSocket|executionWebSocket"],
            "WebSocket实时更新集成（/ws/trading-execution）",
            required_count=2
        )
        
        # 检查图表和可视化渲染
        frontend_checks["chart_rendering"] = check_code_pattern(
            dashboard_file,
            [r"executionPerformanceChart|orderFlowChart|Chart\.js|new Chart"],
            "图表和可视化渲染（执行性能图表、订单流图表）",
            required_count=3
        )
        
        # 检查流程步骤状态显示
        frontend_checks["step_status_display"] = check_code_pattern(
            dashboard_file,
            [r"market-data-status|signal-frequency|risk-check-latency|order-generation-rate|execution-success-rate|position-changes"],
            "流程步骤状态显示（8个步骤的状态和性能指标）",
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
    
    # 2.1 交易执行API路由
    print("\n2.1 交易执行API路由")
    routes_file = "src/gateway/web/trading_execution_routes.py"
    if check_file_exists(routes_file):
        # 检查API端点
        api_checks["api_endpoints"] = check_code_pattern(
            routes_file,
            [r"@router\.get\(.*/trading/execution/flow|@router\.get\(.*/trading/overview"],
            "API端点实现（GET /api/v1/trading/execution/flow, GET /api/v1/trading/overview）",
            required_count=2
        )
        
        # 检查服务层使用
        api_checks["service_layer_usage"] = check_code_pattern(
            routes_file,
            [r"from \.trading_execution_service import|get_execution_flow_data|trading_execution_persistence"],
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
            [r"EventBus|event_bus|\.publish\(|EventType\.EXECUTION_STARTED"],
            "事件总线集成（发布EXECUTION_STARTED事件）",
            required_count=2
        )
        
        # 检查业务流程编排器
        api_checks["business_orchestrator"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|start_process|TRADING_EXECUTION"],
            "BusinessProcessOrchestrator业务流程编排（start_process, TRADING_EXECUTION）",
            required_count=2
        )
        
        # 检查WebSocket广播
        api_checks["websocket_broadcast"] = check_code_pattern(
            routes_file,
            [r"websocket_manager|_get_websocket_manager|manager\.broadcast|broadcast.*trading_execution"],
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
    
    service_file = "src/gateway/web/trading_execution_service.py"
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
            [r"get_unified_adapter_factory|BusinessLayerType\.TRADING|统一适配器工厂"],
            "统一适配器工厂使用（交易层）",
            required_count=2
        )
        
        # 检查交易层适配器
        service_checks["trading_adapter"] = check_code_pattern(
            service_file,
            [r"_get_trading_adapter|trading_adapter|交易层适配器"],
            "交易层适配器获取（通过统一适配器工厂）",
            required_count=1
        )
        
        # 检查降级机制
        service_checks["fallback_mechanism"] = check_code_pattern(
            service_file,
            [r"降级方案|fallback|except.*ImportError|直接实例化|最终降级方案"],
            "降级服务机制（当交易层适配器不可用时的降级处理）",
            required_count=2
        )
        
        # 检查8个业务流程步骤数据收集
        service_checks["workflow_steps_collection"] = check_code_pattern(
            service_file,
            [r"market_monitoring|signal_generation|risk_check|order_generation|order_routing|execution|position_management|result_feedback"],
            "8个业务流程步骤数据收集（市场监控、信号生成、风险检查、订单生成、智能路由、成交执行、结果反馈、持仓管理）",
            required_count=8
        )
        
        # 检查流程状态映射
        service_checks["process_state_mapping"] = check_code_pattern(
            service_file,
            [r"step_state_mapping|MONITORING|SIGNAL_GENERATING|RISK_CHECKING|ORDER_GENERATING|ORDER_ROUTING|EXECUTING"],
            "流程状态映射（8个步骤与BusinessProcessState的映射关系）",
            required_count=3
        )
        
        # 注意：服务层不直接调用持久化（符合架构设计：职责分离）
        # 持久化集成在路由层（trading_execution_routes.py）中实现
    
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
    
    # 4.1 执行记录持久化
    print("\n4.1 执行记录持久化")
    persistence_file = "src/gateway/web/trading_execution_persistence.py"
    if check_file_exists(persistence_file):
        # 检查文件系统持久化
        persistence_checks["file_persistence"] = check_code_pattern(
            persistence_file,
            [r"save_execution_record|json\.dump|文件系统|TRADING_EXECUTION_DIR"],
            "文件系统持久化（JSON格式）",
            required_count=3
        )
        
        # 检查PostgreSQL持久化
        persistence_checks["postgresql_persistence"] = check_code_pattern(
            persistence_file,
            [r"_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*trading_execution_records"],
            "PostgreSQL持久化",
            required_count=2
        )
        
        # 检查8个步骤数据字段
        persistence_checks["workflow_steps_fields"] = check_code_pattern(
            persistence_file,
            [r"market_monitoring|signal_generation|risk_check|order_generation|order_routing|execution|position_management|result_feedback"],
            "8个步骤数据字段（market_monitoring, signal_generation, risk_check, order_generation, order_routing, execution, position_management, result_feedback）",
            required_count=8
        )
        
        # 检查双重存储机制
        persistence_checks["dual_storage"] = check_code_pattern(
            persistence_file,
            [r"优先.*PostgreSQL|如果.*PostgreSQL|故障转移|fallback|return None|文件系统"],
            "双重存储机制（PostgreSQL优先，文件系统降级）",
            required_count=2
        )
        
        # 检查执行记录CRUD操作
        persistence_checks["crud_operations"] = check_code_pattern(
            persistence_file,
            [r"save_execution_record|load_execution_record|get_latest_execution_record|list_execution_records"],
            "执行记录CRUD操作",
            required_count=3
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
    
    routes_file = "src/gateway/web/trading_execution_routes.py"
    service_file = "src/gateway/web/trading_execution_service.py"
    
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
    
    # 5.3 交易层符合性
    print("\n5.3 交易层符合性")
    
    # 检查统一适配器工厂使用
    compliance_checks["adapter_factory_usage"] = check_code_pattern(
        service_file,
        [r"get_unified_adapter_factory|BusinessLayerType\.TRADING|统一适配器工厂"],
        "统一适配器工厂使用（交易层）",
        required_count=2
    )
    
    # 检查交易层组件访问
    compliance_checks["trading_layer_access"] = check_code_pattern(
        service_file,
        [r"交易层适配器|trading_adapter|get_order_manager|get_execution_engine|get_portfolio_manager|get_monitoring_system"],
        "交易层组件访问（通过适配器）",
        required_count=2
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


def check_trading_layer_integration():
    """检查交易层集成"""
    print("\n" + "="*80)
    print("6. 交易层集成检查")
    print("="*80)
    
    trading_checks = {}
    
    service_file = "src/gateway/web/trading_execution_service.py"
    
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
            "交易层适配器获取（通过统一适配器工厂）",
            required_count=1
        )
        
        # 检查交易层组件使用
        trading_checks["trading_components_usage"] = check_code_pattern(
            service_file,
            [r"adapter\.get_order_manager|adapter\.get_execution_engine|adapter\.get_portfolio_manager|adapter\.get_monitoring_system"],
            "交易层组件使用（OrderManager, ExecutionEngine, PositionManager, MonitoringSystem）",
            required_count=2
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
    print("7. WebSocket实时更新检查")
    print("="*80)
    
    websocket_checks = {}
    
    # 7.1 WebSocket端点
    print("\n7.1 WebSocket端点实现")
    websocket_routes = "src/gateway/web/websocket_routes.py"
    if check_file_exists(websocket_routes):
        websocket_checks["websocket_endpoint"] = check_code_pattern(
            websocket_routes,
            [r"/ws/trading-execution|websocket_trading_execution"],
            "WebSocket端点注册（/ws/trading-execution）",
            required_count=1
        )
    
    # 7.2 WebSocket管理器
    print("\n7.2 WebSocket管理器")
    websocket_manager_file = "src/gateway/web/websocket_manager.py"
    if check_file_exists(websocket_manager_file):
        websocket_checks["websocket_manager"] = check_code_pattern(
            websocket_manager_file,
            [r"_broadcast_execution|trading_execution|get_execution_flow_data"],
            "交易执行WebSocket广播实现",
            required_count=2
        )
    
    # 7.3 前端WebSocket处理
    print("\n7.3 前端WebSocket处理")
    dashboard_file = "web-static/trading-execution.html"
    if check_file_exists(dashboard_file):
        websocket_checks["frontend_websocket"] = check_code_pattern(
            dashboard_file,
            [r"/ws/trading-execution|connectExecutionWebSocket|executionWebSocket|onmessage|execution_event"],
            "前端WebSocket消息处理（/ws/trading-execution）",
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


def check_8_workflow_steps():
    """检查8个业务流程步骤的完整实现"""
    print("\n" + "="*80)
    print("8. 8个业务流程步骤检查")
    print("="*80)
    
    workflow_checks = {}
    
    service_file = "src/gateway/web/trading_execution_service.py"
    routes_file = "src/gateway/web/trading_execution_routes.py"
    
    if check_file_exists(service_file):
        # 检查步骤1: 市场监控 (Market Monitoring)
        workflow_checks["step1_market_monitoring"] = check_code_pattern(
            service_file,
            [r"market_monitoring|get_monitoring_system|市场监控|MONITORING"],
            "步骤1: 市场监控（Market Monitoring）",
            required_count=2
        )
        
        # 检查步骤2: 信号生成 (Signal Generation)
        workflow_checks["step2_signal_generation"] = check_code_pattern(
            service_file,
            [r"signal_generation|SIGNALS_GENERATED|EventType\.SIGNALS_GENERATED|信号生成"],
            "步骤2: 信号生成（Signal Generation）",
            required_count=2
        )
        
        # 检查步骤3: 风险检查 (Risk Check)
        workflow_checks["step3_risk_check"] = check_code_pattern(
            service_file,
            [r"risk_check|RISK_CHECK_COMPLETED|EventType\.RISK_CHECK_COMPLETED|风险检查"],
            "步骤3: 风险检查（Risk Check）",
            required_count=2
        )
        
        # 检查步骤4: 订单生成 (Order Generation)
        workflow_checks["step4_order_generation"] = check_code_pattern(
            service_file,
            [r"order_generation|ORDERS_GENERATED|EventType\.ORDERS_GENERATED|get_order_manager|订单生成"],
            "步骤4: 订单生成（Order Generation）",
            required_count=2
        )
        
        # 检查步骤5: 智能路由 (Smart Routing)
        workflow_checks["step5_order_routing"] = check_code_pattern(
            service_file,
            [r"order_routing|ORDER_ROUTING|get_routing_stats|智能路由"],
            "步骤5: 智能路由（Smart Routing）",
            required_count=2
        )
        
        # 检查步骤6: 成交执行 (Execution)
        workflow_checks["step6_execution"] = check_code_pattern(
            service_file,
            [r"\"execution\"|EXECUTION_STARTED|EXECUTION_COMPLETED|EventType\.EXECUTION|get_execution_engine|成交执行"],
            "步骤6: 成交执行（Execution）",
            required_count=3
        )
        
        # 检查步骤7: 结果反馈 (Result Feedback)
        workflow_checks["step7_result_feedback"] = check_code_pattern(
            service_file,
            [r"result_feedback|结果反馈|反馈延迟|反馈质量"],
            "步骤7: 结果反馈（Result Feedback）",
            required_count=2
        )
        
        # 检查步骤8: 持仓管理 (Position Management)
        workflow_checks["step8_position_management"] = check_code_pattern(
            service_file,
            [r"position_management|POSITION_UPDATED|EventType\.POSITION_UPDATED|get_portfolio_manager|持仓管理"],
            "步骤8: 持仓管理（Position Management）",
            required_count=3
        )
        
        # 检查8个步骤的状态映射
        workflow_checks["step_state_mapping"] = check_code_pattern(
            service_file,
            [r"step_state_mapping|MONITORING|SIGNAL_GENERATING|RISK_CHECKING|ORDER_GENERATING|ORDER_ROUTING|EXECUTING"],
            "8个步骤与流程状态的映射关系",
            required_count=4
        )
    
    # 更新统计
    for check_name, check_result in workflow_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
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
    
    routes_file = "src/gateway/web/trading_execution_routes.py"
    service_file = "src/gateway/web/trading_execution_service.py"
    
    # 检查BusinessProcessOrchestrator使用（符合架构设计：业务流程编排器用于管理交易执行流程）
    orchestration_checks["orchestrator_usage"] = check_code_pattern(
        routes_file,
        [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器"],
        "BusinessProcessOrchestrator使用",
        required_count=2
    )
    
    # 检查流程状态管理（业务流程编排器用于管理交易执行流程）
    orchestration_checks["process_management"] = check_code_pattern(
        routes_file,
        [r"start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|TRADING_EXECUTION"],
        "流程状态管理（交易执行流程状态管理）",
        required_count=2
    )
    
    # 检查8个步骤的事件发布
    orchestration_checks["event_publishing"] = check_code_pattern(
        service_file,
        [r"EventBus\.publish|event_bus\.publish|EXECUTION_STARTED|EXECUTION_COMPLETED|SIGNALS_GENERATED|ORDERS_GENERATED|RISK_CHECK_COMPLETED|POSITION_UPDATED"],
        "交易执行流程事件发布（8个步骤的事件）",
        required_count=4
    )
    
    # 检查流程状态机集成
    orchestration_checks["state_machine_integration"] = check_code_pattern(
        service_file,
        [r"get_current_state|state_machine|process_state|state_history|流程状态机"],
        "流程状态机集成（获取当前状态、状态历史）",
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
    
    report_file = project_root / "docs" / f"trading_execution_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 交易执行全流程架构符合性检查报告\n\n")
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
            ("trading_layer_integration", "6. 交易层集成检查"),
            ("websocket_integration", "7. WebSocket实时更新检查"),
            ("workflow_steps", "8. 8个业务流程步骤检查"),
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
    print("交易执行全流程架构符合性检查")
    print("="*80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_service_layer()
    check_persistence()
    check_architecture_compliance()
    check_trading_layer_integration()
    check_websocket()
    check_8_workflow_steps()
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
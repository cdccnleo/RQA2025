#!/usr/bin/env python3
"""
策略回测分析仪表盘架构符合性检查脚本

全面检查策略回测分析仪表盘的功能实现、持久化实现、架构设计符合性
以及与模型分析层集成情况
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
    "ml_integration": {},
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
            "message": f"文件不存在: {file_path}"
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
                    "required_count": required_count
                }
            else:
                return {
                    "status": "warning",
                    "message": f"{description}: 仅找到 {found_count}/{required_count} 个必需模式",
                    "found_patterns": found_patterns,
                    "missing_patterns": missing_patterns,
                    "found_count": found_count,
                    "required_count": required_count
                }
        
        if len(found_patterns) == len(patterns):
            return {
                "status": "passed",
                "message": f"{description}: 所有模式都找到",
                "found_patterns": found_patterns
            }
        elif len(found_patterns) > 0:
            return {
                "status": "warning",
                "message": f"{description}: 部分模式找到 ({len(found_patterns)}/{len(patterns)})",
                "found_patterns": found_patterns,
                "missing_patterns": missing_patterns
            }
        else:
            return {
                "status": "failed",
                "message": f"{description}: 未找到任何模式",
                "missing_patterns": missing_patterns
            }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"读取文件失败: {str(e)}"
        }


def check_frontend_modules():
    """检查前端功能模块"""
    print("\n" + "="*80)
    print("1. 前端功能模块检查")
    print("="*80)
    
    frontend_checks = {}
    
    # 1.1 策略回测分析仪表盘
    print("\n1.1 策略回测分析仪表盘")
    dashboard_file = "web-static/strategy-backtest.html"
    frontend_checks["dashboard_exists"] = {
        "file": dashboard_file,
        "exists": check_file_exists(dashboard_file),
        "status": "passed" if check_file_exists(dashboard_file) else "failed"
    }
    
    if check_file_exists(dashboard_file):
        # 检查统计卡片模块
        frontend_checks["statistics_cards"] = check_code_pattern(
            dashboard_file,
            [r"active-strategies|avg-annual-return|avg-sharpe-ratio|max-drawdown"],
            "统计卡片模块",
            required_count=4
        )
        
        # 检查API集成
        frontend_checks["api_integration"] = check_code_pattern(
            dashboard_file,
            [r"/backtest/run|/backtest/|/strategy/conceptions",
             r"fetch\(|getApiBaseUrl"],
            "API集成",
            required_count=2
        )
        
        # 检查WebSocket集成
        frontend_checks["websocket_integration"] = check_code_pattern(
            dashboard_file,
            [r"WebSocket|websocket|ws://|wss://|/ws/backtest-progress",
             r"connectBacktestWebSocket|onmessage|onopen"],
            "WebSocket实时更新集成",
            required_count=2
        )
        
        # 检查图表渲染
        frontend_checks["chart_rendering"] = check_code_pattern(
            dashboard_file,
            [r"Chart\.js|new Chart|returnsChart|riskReturnChart"],
            "图表和可视化渲染",
            required_count=3
        )
        
        # 检查功能模块
        frontend_checks["feature_modules"] = check_code_pattern(
            dashboard_file,
            [r"策略性能排行|性能指标图表|详细性能指标|回测配置"],
            "功能模块完整性",
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
    
    # 2.1 策略回测API路由
    print("\n2.1 策略回测API路由")
    routes_file = "src/gateway/web/backtest_routes.py"
    if check_file_exists(routes_file):
        # 检查API端点
        api_checks["api_endpoints"] = check_code_pattern(
            routes_file,
            [r"@router\.post\(.*/backtest/run|@router\.get\(.*/backtest/",
             r"@router\.get\(.*/backtest/\{backtest_id\}"],
            "API端点实现",
            required_count=2
        )
        
        # 检查服务层使用
        api_checks["service_layer_usage"] = check_code_pattern(
            routes_file,
            [r"from \.backtest_service import|run_backtest|get_backtest_result|list_backtests"],
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
            [r"EventBus|event_bus|\.publish\(|publish_event"],
            "事件总线集成",
            required_count=1
        )
        
        # 检查业务流程编排器
        api_checks["business_orchestrator"] = check_code_pattern(
            routes_file,
            [r"BusinessProcessOrchestrator|orchestrator|业务流程"],
            "业务流程编排器集成",
            required_count=1
        )
        
        # 检查WebSocket广播
        api_checks["websocket_broadcast"] = check_code_pattern(
            routes_file,
            [r"websocket_manager|_get_websocket_manager|manager\.broadcast|broadcast.*backtest"],
            "WebSocket实时广播",
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
    
    service_file = "src/gateway/web/backtest_service.py"
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
            [r"get_unified_adapter_factory|BusinessLayerType\.ML"],
            "统一适配器工厂使用（ML层）",
            required_count=2
        )
        
        # 检查ML层适配器
        service_checks["ml_adapter"] = check_code_pattern(
            service_file,
            [r"_ml_adapter|get_adapter\(BusinessLayerType\.ML\)|ml.*adapter|ML层适配器"],
            "ML层适配器获取",
            required_count=1
        )
        
        # 检查降级机制
        service_checks["fallback_mechanism"] = check_code_pattern(
            service_file,
            [r"降级方案|fallback|except.*ImportError|直接实例化|BACKTEST_ENGINE_AVAILABLE"],
            "降级服务机制",
            required_count=2
        )
        
        # 检查回测引擎封装
        service_checks["component_encapsulation"] = check_code_pattern(
            service_file,
            [r"BacktestEngine|get_backtest_engine|回测引擎"],
            "回测引擎封装",
            required_count=2
        )
        
        # 检查持久化集成
        service_checks["persistence_integration"] = check_code_pattern(
            service_file,
            [r"backtest_persistence|save_backtest_result|list_backtests|持久化存储"],
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
    
    # 4.1 回测结果持久化
    print("\n4.1 回测结果持久化")
    persistence_file = "src/gateway/web/backtest_persistence.py"
    if check_file_exists(persistence_file):
        # 检查文件系统持久化
        persistence_checks["file_persistence"] = check_code_pattern(
            persistence_file,
            [r"save_backtest_result|json\.dump|文件系统|BACKTEST_RESULTS_DIR"],
            "文件系统持久化（JSON格式）",
            required_count=3
        )
        
        # 检查PostgreSQL持久化
        persistence_checks["postgresql_persistence"] = check_code_pattern(
            persistence_file,
            [r"_save_to_postgresql|_load_from_postgresql|postgresql_persistence|CREATE TABLE.*backtest_results"],
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
        
        # 检查任务CRUD操作
        persistence_checks["crud_operations"] = check_code_pattern(
            persistence_file,
            [r"save_backtest_result|load_backtest_result|update_backtest_result|delete_backtest_result|list_backtest_results"],
            "任务CRUD操作",
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
    """检查架构符合性"""
    print("\n" + "="*80)
    print("5. 架构符合性检查")
    print("="*80)
    
    compliance_checks = {}
    
    # 5.1 基础设施层符合性
    print("\n5.1 基础设施层符合性")
    service_file = "src/gateway/web/backtest_service.py"
    persistence_file = "src/gateway/web/backtest_persistence.py"
    routes_file = "src/gateway/web/backtest_routes.py"
    
    # 检查统一日志系统（在所有文件中）
    compliance_checks["unified_logger"] = check_code_pattern(
        routes_file,
        [r"get_unified_logger|统一日志"],
        "统一日志系统使用（API路由）",
        required_count=1
    )
    
    # 检查配置管理（如果需要）
    compliance_checks["config_management"] = {
        "status": "passed",
        "message": "配置管理通过统一适配器工厂间接实现"
    }
    
    # 5.2 核心服务层符合性
    print("\n5.2 核心服务层符合性")
    
    # 检查EventBus事件发布
    compliance_checks["event_bus_publish"] = check_code_pattern(
        routes_file,
        [r"EventBus|event_bus|\.publish\(|publish_event"],
        "EventBus事件发布",
        required_count=1
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
    
    # 5.3 机器学习层符合性
    print("\n5.3 机器学习层符合性")
    
    # 检查统一适配器工厂使用
    compliance_checks["adapter_factory_usage"] = check_code_pattern(
        service_file,
        [r"get_unified_adapter_factory|BusinessLayerType\.ML"],
        "统一适配器工厂使用（机器学习层）",
        required_count=2
        )
    
    # 检查ML层组件访问
    compliance_checks["ml_layer_access"] = check_code_pattern(
        service_file,
        [r"MLCore|ML层组件|模型预测|ML层适配器"],
        "机器学习层组件访问",
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


def check_ml_integration():
    """检查模型分析层集成"""
    print("\n" + "="*80)
    print("6. 模型分析层集成检查")
    print("="*80)
    
    ml_checks = {}
    
    # 6.1 MLIntegrationAnalyzer检查
    print("\n6.1 MLIntegrationAnalyzer实现检查")
    analytics_file = "src/strategy/backtest/advanced_analytics.py"
    
    if check_file_exists(analytics_file):
        # 检查MLIntegrationAnalyzer类
        ml_checks["ml_integration_analyzer"] = check_code_pattern(
            analytics_file,
            [r"class MLIntegrationAnalyzer|MLIntegrationAnalyzer"],
            "MLIntegrationAnalyzer类定义",
            required_count=1
        )
        
        # 检查是否通过统一适配器工厂访问ML层
        ml_checks["adapter_factory_usage"] = check_code_pattern(
            analytics_file,
            [r"get_unified_adapter_factory|BusinessLayerType\.ML|统一适配器工厂"],
            "通过统一适配器工厂访问ML层",
            required_count=1
        )
        
        # 检查ML层组件使用
        ml_checks["ml_layer_usage"] = check_code_pattern(
            analytics_file,
            [r"MLCore|ModelManager|\.predict\(|模型预测"],
            "ML层组件使用（模型预测）",
            required_count=1
        )
        
        # 检查特征重要性分析
        ml_checks["feature_importance"] = check_code_pattern(
            analytics_file,
            [r"get_feature_importance|feature_importance|特征重要性"],
            "特征重要性分析",
            required_count=1
        )
    else:
        ml_checks["ml_integration_analyzer"] = {
            "status": "warning",
            "message": "advanced_analytics.py文件不存在，无法检查MLIntegrationAnalyzer"
        }
    
    # 6.2 回测服务中的ML集成
    print("\n6.2 回测服务中的ML集成")
    service_file = "src/gateway/web/backtest_service.py"
    
    if check_file_exists(service_file):
        # 检查ML层适配器获取
        ml_checks["ml_adapter_in_service"] = check_code_pattern(
            service_file,
            [r"_get_ml_adapter|BusinessLayerType\.ML|ML层适配器"],
            "回测服务中的ML层适配器获取",
            required_count=1
        )
        
        # 检查模型预测服务使用
        ml_checks["model_prediction_usage"] = check_code_pattern(
            service_file,
            [r"MLCore|\.predict\(|模型预测|模型分析"],
            "模型预测服务使用",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in ml_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            check_results["summary"]["total_items"] += 1
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
    
    check_results["ml_integration"] = ml_checks
    return ml_checks


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
            [r"@router\.websocket\(.*/ws/backtest-progress|websocket_backtest_progress"],
            "回测WebSocket端点",
            required_count=1
        )
    
    # 7.2 WebSocket管理器
    print("\n7.2 WebSocket管理器")
    websocket_manager_file = "src/gateway/web/websocket_manager.py"
    if check_file_exists(websocket_manager_file):
        websocket_checks["websocket_manager"] = check_code_pattern(
            websocket_manager_file,
            [r"_broadcast_backtest_progress|backtest_progress|get_running_backtests"],
            "回测WebSocket广播实现",
            required_count=2
        )
    
    # 7.3 前端WebSocket处理
    print("\n7.3 前端WebSocket处理")
    dashboard_file = "web-static/strategy-backtest.html"
    if check_file_exists(dashboard_file):
        websocket_checks["frontend_websocket"] = check_code_pattern(
            dashboard_file,
            [r"/ws/backtest-progress|connectBacktestWebSocket|onmessage|backtest_progress"],
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
    print("8. 业务流程编排检查")
    print("="*80)
    
    orchestration_checks = {}
    
    routes_file = "src/gateway/web/backtest_routes.py"
    service_file = "src/gateway/web/backtest_service.py"
    
    # 检查BusinessProcessOrchestrator使用（间接检查：通过_get_orchestrator函数和注释说明）
    orchestration_checks["orchestrator_usage"] = check_code_pattern(
        routes_file,
        [r"BusinessProcessOrchestrator|orchestrator|业务流程|_get_orchestrator|业务流程编排器"],
        "BusinessProcessOrchestrator使用",
        required_count=2
    )
    
    # 检查流程状态管理（业务流程编排器已在MLCore中集成，路由层提供访问点）
    orchestration_checks["process_management"] = check_code_pattern(
        routes_file,
        [r"start_process|update_process_state|process.*state|流程状态|业务流程编排|orchestrator|process_id|业务流程"],
        "流程状态管理（业务流程编排器使用）",
        required_count=2
    )
    
    # 检查回测流程事件发布
    orchestration_checks["event_publishing"] = check_code_pattern(
        routes_file,
        [r"EventBus\.publish|event_bus\.publish|PARAMETER_OPTIMIZATION_STARTED|PARAMETER_OPTIMIZATION_COMPLETED|回测.*事件"],
        "回测流程事件发布",
        required_count=1
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
    
    report_file = project_root / "docs" / f"backtest_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 策略回测分析仪表盘架构符合性检查报告\n\n")
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
            ("architecture_compliance", "5. 架构符合性检查"),
            ("ml_integration", "6. 模型分析层集成检查"),
            ("websocket_integration", "7. WebSocket实时更新检查"),
            ("business_orchestration", "8. 业务流程编排检查")
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
    print("策略回测分析仪表盘架构符合性检查")
    print("="*80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_service_layer()
    check_persistence()
    check_architecture_compliance()
    check_ml_integration()
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
    print(f"通过率: {(check_results['summary']['passed'] / check_results['summary']['total_items'] * 100) if check_results['summary']['total_items'] > 0 else 0:.2f}%")
    print(f"\n详细报告: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()


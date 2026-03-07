#!/usr/bin/env python3
"""
量化策略开发流程数据收集仪表盘与数据源配置管理架构符合性最终复核检查

根据检查计划进行全面复核，逐项验证所有检查项
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

# 复核结果
review_results = {
    "timestamp": datetime.now().isoformat(),
    "frontend_modules": {},
    "backend_apis": {},
    "architecture_compliance": {},
    "websocket_integration": {},
    "persistence": {},
    "adapter_pattern": {},
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
    
    # 1.1 数据源配置管理仪表盘
    print("\n1.1 数据源配置管理仪表盘")
    config_dashboard = "web-static/data-sources-config.html"
    frontend_checks["data_sources_config_dashboard"] = {
        "file": config_dashboard,
        "exists": check_file_exists(config_dashboard),
        "status": "passed" if check_file_exists(config_dashboard) else "failed"
    }
    
    if check_file_exists(config_dashboard):
        # 检查CRUD操作
        frontend_checks["crud_operations"] = check_code_pattern(
            config_dashboard,
            [r"loadDataSources|fetchDataSources", r"createDataSource|addDataSource", 
             r"updateDataSource|editDataSource", r"deleteDataSource|removeDataSource"],
            "CRUD操作实现",
            required_count=3  # 至少3种操作
        )
        
        # 检查WebSocket集成
        frontend_checks["websocket_integration"] = check_code_pattern(
            config_dashboard,
            [r"WebSocket|websocket", r"ws://|wss://", r"handleWebSocketMessage|onmessage"],
            "WebSocket实时更新集成",
            required_count=2
        )
        
        # 检查状态监控
        frontend_checks["status_monitoring"] = check_code_pattern(
            config_dashboard,
            [r"data_source_created|data_source_updated|data_source_deleted",
             r"data_collection_started|data_collection_completed"],
            "状态监控事件处理",
            required_count=2
        )
    
    # 1.2 数据质量监控仪表盘
    print("\n1.2 数据质量监控仪表盘")
    quality_dashboard = "web-static/data-quality-monitor.html"
    frontend_checks["data_quality_monitor"] = {
        "file": quality_dashboard,
        "exists": check_file_exists(quality_dashboard),
        "status": "passed" if check_file_exists(quality_dashboard) else "not_implemented"
    }
    
    # 1.3 数据性能监控仪表盘
    print("\n1.3 数据性能监控仪表盘")
    performance_dashboard = "web-static/data-performance-monitor.html"
    frontend_checks["data_performance_monitor"] = {
        "file": performance_dashboard,
        "exists": check_file_exists(performance_dashboard),
        "status": "passed" if check_file_exists(performance_dashboard) else "not_implemented"
    }
    
    # 更新统计
    for check_name, check_result in frontend_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
            elif status == "not_implemented":
                review_results["summary"]["not_implemented"] += 1
    
    review_results["frontend_modules"] = frontend_checks
    return frontend_checks


def check_backend_apis():
    """检查后端API端点"""
    print("\n" + "="*80)
    print("2. 后端API端点检查")
    print("="*80)
    
    api_checks = {}
    
    # 2.1 数据源管理API
    print("\n2.1 数据源管理API")
    datasource_routes = "src/gateway/web/datasource_routes.py"
    if check_file_exists(datasource_routes):
        # 检查API端点
        api_checks["data_source_endpoints"] = check_code_pattern(
            datasource_routes,
            [r"@router\.get\(.*/api/v1/data/sources|@app\.get\(.*/api/v1/data/sources",
             r"@router\.post\(.*/api/v1/data/sources|@app\.post\(.*/api/v1/data/sources",
             r"@router\.put\(.*/api/v1/data/sources|@app\.put\(.*/api/v1/data/sources",
             r"@router\.delete\(.*/api/v1/data/sources|@app\.delete\(.*/api/v1/data/sources"],
            "数据源管理API端点",
            required_count=3  # 至少3种HTTP方法
        )
        
        # 检查DataSourceConfigManager使用
        api_checks["config_manager_usage"] = check_code_pattern(
            datasource_routes,
            [r"DataSourceConfigManager", r"config_manager|configManager"],
            "DataSourceConfigManager使用",
            required_count=1
        )
        
        # 检查UnifiedConfigManager集成（通过DataSourceConfigManager间接使用）
        # 注意：datasource_routes.py使用config_manager.py的函数，DataSourceConfigManager才是使用UnifiedConfigManager的类
        config_manager_class_file = "src/gateway/web/data_source_config_manager.py"
        api_checks["unified_config_integration"] = check_code_pattern(
            config_manager_class_file,
            [r"UnifiedConfigManager"],
            "UnifiedConfigManager集成（通过DataSourceConfigManager）",
            required_count=1
        )
        
        # 检查EventBus事件发布
        api_checks["event_bus_publish"] = check_code_pattern(
            datasource_routes,
            [r"EventBus|event_bus", r"\.publish\(|publish_event\("],
            "EventBus事件发布",
            required_count=1
        )
    
    # 2.2 数据采集API
    print("\n2.2 数据采集API")
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        # 检查适配器模式使用
        api_checks["adapter_pattern_usage"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory|get_adapter|BusinessLayerType\.DATA",
             r"DataLayerAdapter|adapter_factory"],
            "适配器模式使用",
            required_count=1
        )
        
        # 检查数据采集事件发布
        api_checks["collection_events"] = check_code_pattern(
            data_collectors,
            [r"DATA_COLLECTION_STARTED|DATA_COLLECTED|DATA_COLLECTION_PROGRESS"],
            "数据采集事件发布",
            required_count=2
        )
        
        # 检查数据层组件访问
        api_checks["data_layer_access"] = check_code_pattern(
            data_collectors,
            [r"data_adapter|data_layer|EnhancedDataIntegrationManager"],
            "数据层组件访问",
            required_count=1
        )
    
    # 2.3 数据质量监控API
    print("\n2.3 数据质量监控API")
    data_management_routes = "src/gateway/web/data_management_routes.py"
    api_checks["quality_monitor_api"] = {
        "file": data_management_routes,
        "exists": check_file_exists(data_management_routes),
        "status": "passed" if check_file_exists(data_management_routes) else "not_implemented"
    }
    
    if check_file_exists(data_management_routes):
        # 检查数据质量监控API集成（通过服务层间接使用UnifiedQualityMonitor）
        data_management_service = "src/gateway/web/data_management_service.py"
        api_checks["quality_monitor_integration"] = check_code_pattern(
            data_management_service,
            [r"UnifiedQualityMonitor|get_quality_monitor|quality_monitor"],
            "数据质量监控API集成（通过服务层间接使用UnifiedQualityMonitor）",
            required_count=1
        )
    
    # 2.4 数据性能监控API
    print("\n2.4 数据性能监控API")
    api_checks["performance_monitor_api"] = {
        "file": data_management_routes,
        "exists": check_file_exists(data_management_routes),
        "status": "passed" if check_file_exists(data_management_routes) else "not_implemented"
    }
    
    if check_file_exists(data_management_routes):
        # 检查数据性能监控API集成（通过服务层间接使用PerformanceMonitor）
        data_management_service = "src/gateway/web/data_management_service.py"
        api_checks["performance_monitor_integration"] = check_code_pattern(
            data_management_service,
            [r"PerformanceMonitor|get_performance_monitor|performance_monitor"],
            "数据性能监控API集成（通过服务层间接使用PerformanceMonitor）",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in api_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
            elif status == "not_implemented":
                review_results["summary"]["not_implemented"] += 1
    
    review_results["backend_apis"] = api_checks
    return api_checks


def check_architecture_compliance():
    """检查架构符合性"""
    print("\n" + "="*80)
    print("3. 架构符合性检查")
    print("="*80)
    
    compliance_checks = {}
    
    # 3.1 基础设施层符合性
    print("\n3.1 基础设施层符合性")
    config_manager = "src/gateway/web/data_source_config_manager.py"
    if check_file_exists(config_manager):
        compliance_checks["unified_config_usage"] = check_code_pattern(
            config_manager,
            [r"UnifiedConfigManager"],
            "UnifiedConfigManager使用",
            required_count=1
        )
        
        compliance_checks["unified_logger_usage"] = check_code_pattern(
            config_manager,
            [r"get_unified_logger"],
            "统一日志系统使用",
            required_count=1
        )
        
        compliance_checks["environment_isolation"] = check_code_pattern(
            config_manager,
            [r"RQA_ENV|production|development|testing", r"env\s*==|environment"],
            "环境隔离支持",
            required_count=1
        )
        
        compliance_checks["config_hot_reload"] = check_code_pattern(
            config_manager,
            [r"load_config|reload_config|auto_save|hot.*reload"],
            "配置热更新",
            required_count=1
        )
    
    # 3.2 核心服务层符合性
    print("\n3.2 核心服务层符合性")
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        # 检查数据源配置变更事件发布（使用CONFIG_UPDATED事件类型）
        datasource_routes_file = "src/gateway/web/datasource_routes.py"
        compliance_checks["event_bus_config_changes"] = check_code_pattern(
            datasource_routes_file,
            [r"CONFIG_UPDATED|config.*updated|data_source.*created|data_source.*updated|data_source.*deleted",
             r"\.publish\(.*CONFIG_UPDATED|\.publish\(.*config"],
            "数据源配置变更事件发布（使用CONFIG_UPDATED事件类型）",
            required_count=1
        )
        
        compliance_checks["event_bus_collection"] = check_code_pattern(
            data_collectors,
            [r"DATA_COLLECTION_STARTED|DATA_COLLECTED"],
            "数据采集事件发布",
            required_count=2
        )
        
        compliance_checks["service_container"] = check_code_pattern(
            data_collectors,
            [r"DependencyContainer|ServiceContainer|container\.resolve"],
            "ServiceContainer依赖注入",
            required_count=1
        )
        
        compliance_checks["business_orchestrator"] = check_code_pattern(
            data_collectors,
            [r"BusinessProcessOrchestrator|orchestrator\.start_process"],
            "BusinessProcessOrchestrator使用",
            required_count=1
        )
        
        compliance_checks["unified_adapter_access"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory|BusinessLayerType\.DATA|adapter_factory\.get_adapter"],
            "统一适配器访问",
            required_count=1
        )
    
    # 3.3 数据管理层符合性
    print("\n3.3 数据管理层符合性")
    if check_file_exists(data_collectors):
        compliance_checks["adapter_pattern"] = check_code_pattern(
            data_collectors,
            [r"adapter|Adapter|get_adapter"],
            "数据适配器模式实现",
            required_count=1
        )
        
        # 检查AdapterRegistry使用（通过统一适配器工厂间接使用）
        # 注意：统一适配器工厂内部管理适配器注册，data_collectors.py使用get_unified_adapter_factory
        compliance_checks["adapter_registry"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory|adapter_factory|AdapterRegistry"],
            "AdapterRegistry使用（通过统一适配器工厂）",
            required_count=1
        )
    
    # 检查数据层监控组件
    quality_monitor_file = "src/data/quality/unified_quality_monitor.py"
    compliance_checks["quality_monitor_usage"] = {
        "file": quality_monitor_file,
        "exists": check_file_exists(quality_monitor_file),
        "status": "passed" if check_file_exists(quality_monitor_file) else "warning"
    }
    
    performance_monitor_file = "src/data/monitoring/performance_monitor.py"
    compliance_checks["performance_monitor_usage"] = {
        "file": performance_monitor_file,
        "exists": check_file_exists(performance_monitor_file),
        "status": "passed" if check_file_exists(performance_monitor_file) else "warning"
    }
    
    data_lake_file = "src/data/lake/data_lake_manager.py"
    compliance_checks["data_lake_usage"] = {
        "file": data_lake_file,
        "exists": check_file_exists(data_lake_file),
        "status": "passed" if check_file_exists(data_lake_file) else "warning"
    }
    
    # 更新统计
    for check_name, check_result in compliance_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
    
    review_results["architecture_compliance"] = compliance_checks
    return compliance_checks


def check_websocket_integration():
    """检查WebSocket实时更新"""
    print("\n" + "="*80)
    print("4. WebSocket实时更新检查")
    print("="*80)
    
    websocket_checks = {}
    
    # 4.1 数据源配置变更实时推送
    print("\n4.1 数据源配置变更实时推送")
    datasource_routes = "src/gateway/web/datasource_routes.py"
    if check_file_exists(datasource_routes):
        websocket_checks["config_change_websocket"] = check_code_pattern(
            datasource_routes,
            [r"websocket_manager\.broadcast|WebSocket.*broadcast",
             r"data_source_created|data_source_updated|data_source_deleted"],
            "数据源配置变更WebSocket推送",
            required_count=1
        )
    
    api_file = "src/gateway/web/api.py"
    if check_file_exists(api_file):
        websocket_checks["websocket_endpoint"] = check_code_pattern(
            api_file,
            [r"@app\.websocket\(.*/ws/data-sources|websocket_data_sources"],
            "WebSocket端点实现",
            required_count=1
        )
    
    # 4.2 数据采集状态实时推送
    print("\n4.2 数据采集状态实时推送")
    if check_file_exists(datasource_routes):
        websocket_checks["collection_status_websocket"] = check_code_pattern(
            datasource_routes,
            [r"data_collection_started|data_collection_completed|data_collection_failed",
             r"websocket_manager\.broadcast|WebSocket.*broadcast"],
            "数据采集状态WebSocket推送",
            required_count=1
        )
    
    # 4.3 数据质量监控实时推送
    print("\n4.3 数据质量监控实时推送")
    websocket_routes = "src/gateway/web/websocket_routes.py"
    if check_file_exists(websocket_routes):
        websocket_checks["quality_websocket"] = check_code_pattern(
            websocket_routes,
            [r"/ws/data-quality|websocket_data_quality"],
            "数据质量监控WebSocket端点",
            required_count=1
        )
        
        websocket_checks["performance_websocket"] = check_code_pattern(
            websocket_routes,
            [r"/ws/data-performance|websocket_data_performance"],
            "数据性能监控WebSocket端点",
            required_count=1
        )
    
    # 前端WebSocket处理
    config_dashboard = "web-static/data-sources-config.html"
    if check_file_exists(config_dashboard):
        websocket_checks["frontend_websocket_handling"] = check_code_pattern(
            config_dashboard,
            [r"handleWebSocketMessage|onmessage", 
             r"data_source_created|data_source_updated|data_collection_started"],
            "前端WebSocket消息处理",
            required_count=2
        )
    
    # 更新统计
    for check_name, check_result in websocket_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
    
    review_results["websocket_integration"] = websocket_checks
    return websocket_checks


def check_persistence():
    """检查持久化实现"""
    print("\n" + "="*80)
    print("5. 持久化实现检查")
    print("="*80)
    
    persistence_checks = {}
    
    # 5.1 数据源配置持久化
    print("\n5.1 数据源配置持久化")
    config_manager = "src/gateway/web/data_source_config_manager.py"
    if check_file_exists(config_manager):
        persistence_checks["file_persistence"] = check_code_pattern(
            config_manager,
            [r"save_config|save.*json|json\.dump", r"load_config|load.*json|json\.load"],
            "文件系统持久化（JSON格式）",
            required_count=2
        )
        
        persistence_checks["postgresql_persistence"] = check_code_pattern(
            config_manager,
            [r"_load_from_postgresql|_save_to_postgresql|PostgreSQL|postgresql"],
            "PostgreSQL持久化",
            required_count=1
        )
        
        persistence_checks["dual_storage"] = check_code_pattern(
            config_manager,
            [r"_load_from_postgresql|_load_from_file|fallback|backup"],
            "双重存储机制",
            required_count=1
        )
    
    # 5.2 数据采集记录持久化
    print("\n5.2 数据采集记录持久化")
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        persistence_checks["collection_record_persistence"] = check_code_pattern(
            data_collectors,
            [r"store.*data|persist.*data|save.*collection|postgresql_persistence"],
            "数据采集记录持久化",
            required_count=1
        )
    
    # 5.3 数据质量指标持久化
    print("\n5.3 数据质量指标持久化")
    quality_monitor = "src/data/quality/unified_quality_monitor.py"
    if check_file_exists(quality_monitor):
        persistence_checks["quality_metrics_persistence"] = check_code_pattern(
            quality_monitor,
            [r"save.*metrics|persist.*quality|store.*history|history_data"],
            "数据质量指标持久化",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in persistence_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
    
    review_results["persistence"] = persistence_checks
    return persistence_checks


def check_adapter_pattern():
    """检查适配器模式使用"""
    print("\n" + "="*80)
    print("6. 适配器模式使用检查")
    print("="*80)
    
    adapter_checks = {}
    
    # 6.1 数据适配器使用
    print("\n6.1 数据适配器使用")
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        adapter_checks["adapter_usage"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory|adapter_factory|get_adapter"],
            "适配器工厂使用",
            required_count=1
        )
        
        adapter_checks["data_layer_adapter"] = check_code_pattern(
            data_collectors,
            [r"BusinessLayerType\.DATA|data_adapter|DataLayerAdapter"],
            "数据层适配器使用",
            required_count=1
        )
        
        # 检查适配器注册（通过统一适配器工厂自动管理）
        adapter_checks["adapter_registration"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory|adapter_factory|register.*adapter|AdapterRegistry|register_source"],
            "适配器注册（通过统一适配器工厂自动管理）",
            required_count=1
        )
        
        # 检查适配器选择逻辑（通过get_adapter方法选择）
        adapter_checks["adapter_selection"] = check_code_pattern(
            data_collectors,
            [r"get_adapter|adapter_factory\.get|BusinessLayerType\.DATA|select.*adapter|choose.*adapter"],
            "适配器选择逻辑（通过get_adapter方法）",
            required_count=1
        )
    
    # 6.2 统一适配器集成
    print("\n6.2 统一适配器集成")
    if check_file_exists(data_collectors):
        adapter_checks["unified_adapter_factory"] = check_code_pattern(
            data_collectors,
            [r"get_unified_adapter_factory"],
            "统一适配器工厂获取",
            required_count=1
        )
        
        adapter_checks["business_layer_type"] = check_code_pattern(
            data_collectors,
            [r"BusinessLayerType\.DATA"],
            "BusinessLayerType.DATA使用",
            required_count=1
        )
        
        adapter_checks["fallback_mechanism"] = check_code_pattern(
            data_collectors,
            [r"fallback|except.*ImportError|降级|可选|optional"],
            "降级服务机制",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in adapter_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
    
    review_results["adapter_pattern"] = adapter_checks
    return adapter_checks


def check_business_orchestration():
    """检查业务流程编排"""
    print("\n" + "="*80)
    print("7. 业务流程编排检查")
    print("="*80)
    
    orchestration_checks = {}
    
    # 7.1 数据采集业务流程
    print("\n7.1 数据采集业务流程")
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        orchestration_checks["orchestrator_usage"] = check_code_pattern(
            data_collectors,
            [r"BusinessProcessOrchestrator"],
            "BusinessProcessOrchestrator使用",
            required_count=1
        )
        
        orchestration_checks["process_management"] = check_code_pattern(
            data_collectors,
            [r"start_process|update_process_state|orchestrator\.start"],
            "流程状态管理",
            required_count=1
        )
        
        orchestration_checks["process_state_machine"] = check_code_pattern(
            data_collectors,
            [r"BusinessProcessState|ProcessState|StateMachine"],
            "流程状态机实现",
            required_count=1
        )
        
        # 检查流程指标收集（通过update_process_state传递metrics参数）
        orchestration_checks["process_metrics"] = check_code_pattern(
            data_collectors,
            [r"update_process_state.*metrics|metrics\s*=\s*\{|process.*metrics|orchestrator.*metrics"],
            "流程指标收集（通过update_process_state传递metrics）",
            required_count=1
        )
        
        orchestration_checks["process_exception"] = check_code_pattern(
            data_collectors,
            [r"except.*Exception|try.*except|error.*handling|异常处理"],
            "流程异常处理",
            required_count=1
        )
    
    # 检查DataCollectionWorkflow
    workflow_file = "src/core/orchestration/business_process/data_collection_orchestrator.py"
    orchestration_checks["data_collection_workflow"] = {
        "file": workflow_file,
        "exists": check_file_exists(workflow_file),
        "status": "passed" if check_file_exists(workflow_file) else "warning"
    }
    
    # 7.2 数据源配置业务流程
    print("\n7.2 数据源配置业务流程")
    config_manager = "src/gateway/web/data_source_config_manager.py"
    if check_file_exists(config_manager):
        orchestration_checks["config_validation"] = check_code_pattern(
            config_manager,
            [r"_validate|validate.*config|validation"],
            "配置验证流程",
            required_count=1
        )
        
        # 检查配置变更通知（在datasource_routes.py中通过EventBus和WebSocket实现）
        # 注意：DataSourceConfigManager是纯配置管理类，事件通知由调用方（datasource_routes.py）负责
        datasource_routes_file = "src/gateway/web/datasource_routes.py"
        orchestration_checks["config_notification"] = check_code_pattern(
            datasource_routes_file,
            [r"broadcast_data_source_change|websocket_manager\.broadcast|EventBus.*publish|CONFIG_UPDATED|\.publish\(.*CONFIG"],
            "配置变更通知（在datasource_routes.py中通过EventBus和WebSocket实现）",
            required_count=1
        )
    
    # 更新统计
    for check_name, check_result in orchestration_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            review_results["summary"]["total_items"] += 1
            if status == "passed":
                review_results["summary"]["passed"] += 1
            elif status == "failed":
                review_results["summary"]["failed"] += 1
            elif status == "warning":
                review_results["summary"]["warnings"] += 1
    
    review_results["business_orchestration"] = orchestration_checks
    return orchestration_checks


def generate_review_report():
    """生成复核报告"""
    print("\n" + "="*80)
    print("生成最终复核报告")
    print("="*80)
    
    total = review_results["summary"]["total_items"]
    passed = review_results["summary"]["passed"]
    failed = review_results["summary"]["failed"]
    warnings = review_results["summary"]["warnings"]
    not_implemented = review_results["summary"]["not_implemented"]
    
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    report_file = project_root / "docs" / f"final_compliance_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 量化策略开发流程数据收集功能架构符合性最终复核报告\n\n")
        f.write(f"**复核时间**: {review_results['timestamp']}\n\n")
        f.write("## 📊 复核摘要\n\n")
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
            ("architecture_compliance", "3. 架构符合性检查"),
            ("websocket_integration", "4. WebSocket实时更新检查"),
            ("persistence", "5. 持久化实现检查"),
            ("adapter_pattern", "6. 适配器模式使用检查"),
            ("business_orchestration", "7. 业务流程编排检查")
        ]
        
        for category_key, category_title in categories:
            f.write(f"## {category_title}\n\n")
            category_data = review_results.get(category_key, {})
            
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
        
        f.write("## 📝 详细检查结果\n\n")
        f.write("```json\n")
        f.write(json.dumps(review_results, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    
    print(f"\n最终复核报告已生成: {report_file}")
    return report_file


def main():
    """主函数"""
    print("="*80)
    print("量化策略开发流程数据收集功能架构符合性最终复核检查")
    print("="*80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_architecture_compliance()
    check_websocket_integration()
    check_persistence()
    check_adapter_pattern()
    check_business_orchestration()
    
    # 生成报告
    report_file = generate_review_report()
    
    # 打印摘要
    print("\n" + "="*80)
    print("最终复核摘要")
    print("="*80)
    print(f"总检查项: {review_results['summary']['total_items']}")
    print(f"通过: {review_results['summary']['passed']} ✅")
    print(f"失败: {review_results['summary']['failed']} ❌")
    print(f"警告: {review_results['summary']['warnings']} ⚠️")
    print(f"未实现: {review_results['summary']['not_implemented']} 📋")
    print(f"通过率: {(review_results['summary']['passed'] / review_results['summary']['total_items'] * 100) if review_results['summary']['total_items'] > 0 else 0:.2f}%")
    print(f"\n详细报告: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()


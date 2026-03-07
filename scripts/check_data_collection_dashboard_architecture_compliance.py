#!/usr/bin/env python3
"""
数据收集仪表盘与数据源配置管理架构符合性检查脚本

检查范围：
1. 前端功能模块（data-sources-config.html, data-quality-monitor.html, data-performance-monitor.html）
2. 后端API端点（datasource_routes.py, data_source_config_manager.py, data_collectors.py, data_management_routes.py）
3. 基础设施层符合性（UnifiedConfigManager, 统一日志系统）
4. 核心服务层符合性（EventBus, ServiceContainer, BusinessProcessOrchestrator）
5. 数据管理层符合性（数据适配器模式, 质量监控, 性能监控, 数据湖管理器）
6. WebSocket实时更新
7. 持久化实现
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查结果存储
check_results = {
    "timestamp": datetime.now().isoformat(),
    "frontend_checks": {},
    "backend_checks": {},
    "infrastructure_compliance": {},
    "core_services_compliance": {},
    "data_layer_compliance": {},
    "websocket_integration": {},
    "persistence_implementation": {},
    "adapter_pattern": {},
    "summary": {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
}


def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(file_path)


def check_code_pattern(file_path: str, patterns: List[str], description: str) -> Dict[str, Any]:
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
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                found_patterns.append(pattern)
            else:
                missing_patterns.append(pattern)
        
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
    print("\n=== 检查前端功能模块 ===")
    
    frontend_files = {
        "data_sources_config": "web-static/data-sources-config.html",
        "data_quality_monitor": "web-static/data-quality-monitor.html",
        "data_performance_monitor": "web-static/data-performance-monitor.html"
    }
    
    for module_name, file_path in frontend_files.items():
        print(f"\n检查 {module_name} ({file_path})...")
        
        # 检查文件是否存在
        if not check_file_exists(file_path):
            check_results["frontend_checks"][module_name] = {
                "status": "failed",
                "message": f"文件不存在: {file_path}"
            }
            continue
        
        # 检查API调用
        api_patterns = [
            r"fetch\s*\([^)]*getApiBaseUrl|fetch\s*\([^)]*['\"]/api/v1",
            r"POST|GET|PUT|DELETE",
            r"WebSocket|ws://|new WebSocket"
        ]
        
        api_check = check_code_pattern(
            file_path,
            api_patterns,
            f"{module_name} API调用"
        )
        
        check_results["frontend_checks"][module_name] = {
            "file_exists": True,
            "api_calls": api_check
        }
        
        print(f"  API调用检查: {api_check['status']}")


def check_backend_apis():
    """检查后端API端点"""
    print("\n=== 检查后端API端点 ===")
    
    backend_files = {
        "datasource_routes": "src/gateway/web/datasource_routes.py",
        "data_source_config_manager": "src/gateway/web/data_source_config_manager.py",
        "data_collectors": "src/gateway/web/data_collectors.py",
        "data_management_routes": "src/gateway/web/data_management_routes.py"
    }
    
    for module_name, file_path in backend_files.items():
        print(f"\n检查 {module_name} ({file_path})...")
        
        if not check_file_exists(file_path):
            check_results["backend_checks"][module_name] = {
                "status": "failed",
                "message": f"文件不存在: {file_path}"
            }
            continue
        
        # 检查API路由定义
        route_patterns = [
            r"@router\.(get|post|put|delete)\s*\(",
            r"async def\s+\w+\s*\("
        ]
        
        route_check = check_code_pattern(
            file_path,
            route_patterns,
            f"{module_name} API路由"
        )
        
        check_results["backend_checks"][module_name] = {
            "file_exists": True,
            "api_routes": route_check
        }
        
        print(f"  API路由检查: {route_check['status']}")


def check_infrastructure_compliance():
    """检查基础设施层符合性"""
    print("\n=== 检查基础设施层符合性 ===")
    
    # 检查数据源配置管理器是否使用UnifiedConfigManager
    config_manager_file = "src/gateway/web/data_source_config_manager.py"
    
    if check_file_exists(config_manager_file):
        print(f"\n检查 {config_manager_file}...")
        
        # 检查UnifiedConfigManager使用
        unified_config_patterns = [
            r"from src\.infrastructure\.config\.core\.unified_manager_enhanced import UnifiedConfigManager",
            r"UnifiedConfigManager\s*\(",
            r"self\.config_manager\s*="
        ]
        
        unified_config_check = check_code_pattern(
            config_manager_file,
            unified_config_patterns,
            "UnifiedConfigManager使用"
        )
        
        # 检查统一日志系统使用
        logger_patterns = [
            r"from src\.infrastructure\.logging\.core\.unified_logger import get_unified_logger",
            r"get_unified_logger\s*\("
        ]
        
        logger_check = check_code_pattern(
            config_manager_file,
            logger_patterns,
            "统一日志系统使用"
        )
        
        check_results["infrastructure_compliance"]["data_source_config_manager"] = {
            "unified_config_manager": unified_config_check,
            "unified_logger": logger_check
        }
        
        print(f"  UnifiedConfigManager: {unified_config_check['status']}")
        print(f"  统一日志系统: {logger_check['status']}")
    else:
        check_results["infrastructure_compliance"]["data_source_config_manager"] = {
            "status": "failed",
            "message": f"文件不存在: {config_manager_file}"
        }


def check_core_services_compliance():
    """检查核心服务层符合性"""
    print("\n=== 检查核心服务层符合性 ===")
    
    files_to_check = {
        "datasource_routes": "src/gateway/web/datasource_routes.py",
        "data_collectors": "src/gateway/web/data_collectors.py"
    }
    
    for module_name, file_path in files_to_check.items():
        print(f"\n检查 {module_name} ({file_path})...")
        
        if not check_file_exists(file_path):
            check_results["core_services_compliance"][module_name] = {
                "status": "failed",
                "message": f"文件不存在: {file_path}"
            }
            continue
        
        # 检查EventBus使用
        eventbus_patterns = [
            r"from src\.core\.event_bus\.core import EventBus",
            r"EventBus\s*\(|event_bus\s*=",
            r"event_bus\.(publish|subscribe)"
        ]
        
        eventbus_check = check_code_pattern(
            file_path,
            eventbus_patterns,
            f"{module_name} EventBus使用"
        )
        
        # 检查ServiceContainer使用
        service_container_patterns = [
            r"from src\.core\.(service_container|container)\.container import (ServiceContainer|DependencyContainer)",
            r"(ServiceContainer|DependencyContainer)\s*\(|container\s*=",
            r"container\.(register|get_service)"
        ]
        
        service_container_check = check_code_pattern(
            file_path,
            service_container_patterns,
            f"{module_name} ServiceContainer使用"
        )
        
        # 检查BusinessProcessOrchestrator使用
        orchestrator_patterns = [
            r"from src\.core\.orchestration\.orchestrator_refactored import BusinessProcessOrchestrator",
            r"BusinessProcessOrchestrator\s*\(|orchestrator\s*=",
            r"orchestrator\.(start_process|update_process)"
        ]
        
        orchestrator_check = check_code_pattern(
            file_path,
            orchestrator_patterns,
            f"{module_name} BusinessProcessOrchestrator使用"
        )
        
        check_results["core_services_compliance"][module_name] = {
            "eventbus": eventbus_check,
            "service_container": service_container_check,
            "orchestrator": orchestrator_check
        }
        
        print(f"  EventBus: {eventbus_check['status']}")
        print(f"  ServiceContainer: {service_container_check['status']}")
        print(f"  BusinessProcessOrchestrator: {orchestrator_check['status']}")


def check_data_layer_compliance():
    """检查数据管理层符合性"""
    print("\n=== 检查数据管理层符合性 ===")
    
    # 检查数据采集器是否使用适配器模式
    data_collectors_file = "src/gateway/web/data_collectors.py"
    
    if check_file_exists(data_collectors_file):
        print(f"\n检查 {data_collectors_file}...")
        
        # 检查统一适配器工厂使用
        adapter_factory_patterns = [
            r"from src\.core\.integration\.unified_business_adapters import (get_unified_adapter_factory|BusinessLayerType)",
            r"get_unified_adapter_factory\s*\(",
            r"BusinessLayerType\.DATA"
        ]
        
        adapter_factory_check = check_code_pattern(
            data_collectors_file,
            adapter_factory_patterns,
            "统一适配器工厂使用"
        )
        
        # 检查数据适配器使用（至少应该有适配器调用）
        adapter_patterns = [
            r"adapter\s*=|Adapter\s*\(",
            r"collect_from_\w+_adapter"
        ]
        
        adapter_check = check_code_pattern(
            data_collectors_file,
            adapter_patterns,
            "数据适配器使用"
        )
        
        check_results["data_layer_compliance"]["data_collectors"] = {
            "adapter_factory": adapter_factory_check,
            "adapter_usage": adapter_check
        }
        
        print(f"  统一适配器工厂: {adapter_factory_check['status']}")
        print(f"  数据适配器使用: {adapter_check['status']}")
    
    # 检查数据管理服务是否使用数据层组件
    data_management_service_file = "src/gateway/web/data_management_service.py"
    
    if check_file_exists(data_management_service_file):
        print(f"\n检查 {data_management_service_file}...")
        
        # 检查UnifiedQualityMonitor使用（支持多种导入方式）
        quality_monitor_patterns = [
            r"from src\.data\.quality\.unified_quality_monitor import.*UnifiedQualityMonitor",
            r"UnifiedQualityMonitor\s*\(|quality_monitor\s*="
        ]
        
        quality_monitor_check = check_code_pattern(
            data_management_service_file,
            quality_monitor_patterns,
            "UnifiedQualityMonitor使用"
        )
        
        # 检查PerformanceMonitor使用
        performance_monitor_patterns = [
            r"from src\.data\.monitoring\.performance_monitor import PerformanceMonitor",
            r"PerformanceMonitor\s*\(|performance_monitor\s*="
        ]
        
        performance_monitor_check = check_code_pattern(
            data_management_service_file,
            performance_monitor_patterns,
            "PerformanceMonitor使用"
        )
        
        # 检查DataLakeManager使用
        data_lake_patterns = [
            r"from src\.data\.lake\.data_lake_manager import DataLakeManager",
            r"DataLakeManager\s*\(|data_lake_manager\s*="
        ]
        
        data_lake_check = check_code_pattern(
            data_management_service_file,
            data_lake_patterns,
            "DataLakeManager使用"
        )
        
        check_results["data_layer_compliance"]["data_management_service"] = {
            "quality_monitor": quality_monitor_check,
            "performance_monitor": performance_monitor_check,
            "data_lake_manager": data_lake_check
        }
        
        print(f"  UnifiedQualityMonitor: {quality_monitor_check['status']}")
        print(f"  PerformanceMonitor: {performance_monitor_check['status']}")
        print(f"  DataLakeManager: {data_lake_check['status']}")


def check_websocket_integration():
    """检查WebSocket实时更新"""
    print("\n=== 检查WebSocket实时更新 ===")
    
    # 检查数据源路由中的WebSocket广播
    datasource_routes_file = "src/gateway/web/datasource_routes.py"
    
    if check_file_exists(datasource_routes_file):
        print(f"\n检查 {datasource_routes_file}...")
        
        # 检查WebSocket广播函数
        websocket_patterns = [
            r"broadcast_data_source_change",
            r"websocket_manager\.broadcast",
            r"WebSocket|websocket"
        ]
        
        websocket_check = check_code_pattern(
            datasource_routes_file,
            websocket_patterns,
            "WebSocket广播"
        )
        
        check_results["websocket_integration"]["datasource_routes"] = websocket_check
        print(f"  WebSocket广播: {websocket_check['status']}")
    
    # 检查前端WebSocket连接
    frontend_file = "web-static/data-sources-config.html"
    
    if check_file_exists(frontend_file):
        print(f"\n检查 {frontend_file}...")
        
        # 检查WebSocket连接代码
        frontend_websocket_patterns = [
            r"new WebSocket\s*\(",
            r"websocket\.(onmessage|onopen|onclose)",
            r"ws://|wss://"
        ]
        
        frontend_websocket_check = check_code_pattern(
            frontend_file,
            frontend_websocket_patterns,
            "前端WebSocket连接"
        )
        
        check_results["websocket_integration"]["frontend"] = frontend_websocket_check
        print(f"  前端WebSocket连接: {frontend_websocket_check['status']}")


def check_persistence_implementation():
    """检查持久化实现"""
    print("\n=== 检查持久化实现 ===")
    
    # 检查数据源配置持久化
    config_manager_file = "src/gateway/web/data_source_config_manager.py"
    
    if check_file_exists(config_manager_file):
        print(f"\n检查 {config_manager_file}...")
        
        # 检查配置保存和加载（文件系统）
        filesystem_patterns = [
            r"save_config\s*\(|load_config\s*\(",
            r"json\.(dump|load)",
            r"\.json"
        ]
        
        filesystem_check = check_code_pattern(
            config_manager_file,
            filesystem_patterns,
            "配置持久化（文件系统）"
        )
        
        # 检查PostgreSQL持久化（P3优化）
        postgresql_patterns = [
            r"_save_to_postgresql\s*\(",
            r"_load_from_postgresql\s*\(",
            r"postgresql_persistence|get_db_connection"
        ]
        
        postgresql_check = check_code_pattern(
            config_manager_file,
            postgresql_patterns,
            "配置持久化（PostgreSQL）"
        )
        
        # 合并检查结果
        if filesystem_check['status'] == 'passed' and postgresql_check['status'] in ['passed', 'warning']:
            persistence_check = {
                'status': 'passed',
                'message': '配置持久化: 文件系统 + PostgreSQL双重存储已实现',
                'filesystem': filesystem_check,
                'postgresql': postgresql_check
            }
        elif filesystem_check['status'] == 'passed':
            persistence_check = {
                'status': 'passed',
                'message': '配置持久化: 文件系统存储已实现',
                'filesystem': filesystem_check,
                'postgresql': postgresql_check
            }
        else:
            persistence_check = filesystem_check
        
        check_results["persistence_implementation"]["data_source_config"] = persistence_check
        print(f"  配置持久化: {persistence_check['status']}")
        if 'postgresql' in persistence_check:
            print(f"    - 文件系统: {filesystem_check['status']}")
            print(f"    - PostgreSQL: {postgresql_check['status']}")
    
    # 检查数据质量指标持久化（P3优化）
    quality_monitor_file = "src/data/quality/unified_quality_monitor.py"
    
    if check_file_exists(quality_monitor_file):
        print(f"\n检查 {quality_monitor_file}...")
        
        # 检查数据湖持久化
        data_lake_patterns = [
            r"_persist_quality_metrics_to_data_lake\s*\(",
            r"_load_quality_history_from_data_lake\s*\(",
            r"DataLakeManager|data_lake_manager"
        ]
        
        data_lake_check = check_code_pattern(
            quality_monitor_file,
            data_lake_patterns,
            "数据质量指标持久化（数据湖）"
        )
        
        check_results["persistence_implementation"]["quality_metrics"] = data_lake_check
        print(f"  数据质量指标持久化: {data_lake_check['status']}")
    
    # 检查业务流程编排器集成（P3优化）
    data_collectors_file = "src/gateway/web/data_collectors.py"
    
    if check_file_exists(data_collectors_file):
        print(f"\n检查 {data_collectors_file}...")
        
        # 检查DataCollectionWorkflow集成
        workflow_patterns = [
            r"DataCollectionWorkflow",
            r"start_collection_process",
            r"workflow\s*="
        ]
        
        workflow_check = check_code_pattern(
            data_collectors_file,
            workflow_patterns,
            "业务流程编排器集成"
        )
        
        check_results["persistence_implementation"]["business_process_orchestrator"] = workflow_check
        print(f"  业务流程编排器集成: {workflow_check['status']}")


def calculate_summary():
    """计算检查结果摘要"""
    def count_status(results_dict: Dict[str, Any], status_key: str = "status") -> tuple:
        """递归统计状态"""
        passed = 0
        failed = 0
        warnings = 0
        total = 0
        
        for key, value in results_dict.items():
            if isinstance(value, dict):
                if status_key in value:
                    total += 1
                    status = value[status_key]
                    if status == "passed":
                        passed += 1
                    elif status == "failed":
                        failed += 1
                    elif status == "warning":
                        warnings += 1
                else:
                    # 递归检查嵌套字典
                    p, f, w, t = count_status(value, status_key)
                    passed += p
                    failed += f
                    warnings += w
                    total += t
        
        return passed, failed, warnings, total
    
    # 统计所有检查结果
    all_results = {
        "frontend_checks": check_results["frontend_checks"],
        "backend_checks": check_results["backend_checks"],
        "infrastructure_compliance": check_results["infrastructure_compliance"],
        "core_services_compliance": check_results["core_services_compliance"],
        "data_layer_compliance": check_results["data_layer_compliance"],
        "websocket_integration": check_results["websocket_integration"],
        "persistence_implementation": check_results["persistence_implementation"]
    }
    
    passed, failed, warnings, total = count_status(all_results)
    
    check_results["summary"] = {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "0%"
    }


def main():
    """主函数"""
    print("=" * 80)
    print("数据收集仪表盘与数据源配置管理架构符合性检查")
    print("=" * 80)
    
    # 执行各项检查
    check_frontend_modules()
    check_backend_apis()
    check_infrastructure_compliance()
    check_core_services_compliance()
    check_data_layer_compliance()
    check_websocket_integration()
    check_persistence_implementation()
    
    # 计算摘要
    calculate_summary()
    
    # 保存检查结果
    output_file = "docs/data_collection_dashboard_architecture_compliance_check_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(check_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)
    print(f"\n检查结果摘要:")
    print(f"  总检查项: {check_results['summary']['total_checks']}")
    print(f"  通过: {check_results['summary']['passed']}")
    print(f"  失败: {check_results['summary']['failed']}")
    print(f"  警告: {check_results['summary']['warnings']}")
    print(f"  通过率: {check_results['summary']['pass_rate']}")
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()


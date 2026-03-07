#!/usr/bin/env python3
"""
量化策略开发流程数据收集仪表盘、数据源监控及数据源配置管理架构符合性检查脚本

检查范围：
1. 数据收集仪表盘dashboard功能实现
2. 数据源监控功能实现
3. 数据源配置管理data-sources-config功能实现
4. 与架构设计文档的符合性检查
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
    "dashboard_checks": {},
    "data_source_monitoring_checks": {},
    "data_source_config_checks": {},
    "architecture_compliance": {},
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


def check_dashboard_implementation():
    """检查数据收集仪表盘实现"""
    print("\n" + "="*80)
    print("检查数据收集仪表盘实现")
    print("="*80)
    
    dashboard_checks = {}
    
    # 1. 检查核心仪表盘文件
    dashboard_file = "src/data/monitoring/dashboard.py"
    dashboard_checks["core_dashboard_file"] = {
        "file": dashboard_file,
        "exists": check_file_exists(dashboard_file),
        "status": "passed" if check_file_exists(dashboard_file) else "failed"
    }
    
    if check_file_exists(dashboard_file):
        # 检查DataDashboard类
        dashboard_checks["DataDashboard_class"] = check_code_pattern(
            dashboard_file,
            [r"class DataDashboard", r"def __init__", r"def get_dashboard_data"],
            "DataDashboard类实现"
        )
        
        # 检查仪表盘配置
        dashboard_checks["DashboardConfig"] = check_code_pattern(
            dashboard_file,
            [r"class DashboardConfig", r"refresh_interval", r"enable_auto_refresh"],
            "DashboardConfig配置类"
        )
        
        # 检查指标组件
        dashboard_checks["MetricWidget"] = check_code_pattern(
            dashboard_file,
            [r"class MetricWidget", r"metric_type", r"data_source"],
            "MetricWidget指标组件"
        )
        
        # 检查告警规则
        dashboard_checks["AlertRule"] = check_code_pattern(
            dashboard_file,
            [r"class AlertRule", r"condition", r"threshold"],
            "AlertRule告警规则"
        )
        
        # 检查指标收集功能
        dashboard_checks["metrics_collection"] = check_code_pattern(
            dashboard_file,
            [r"_collect_metrics", r"get_performance_metrics", r"get_quality_report"],
            "指标收集功能"
        )
    
    # 2. 检查网关层仪表盘路由
    datasource_routes = "src/gateway/web/datasource_routes.py"
    dashboard_checks["dashboard_routes"] = {
        "file": datasource_routes,
        "exists": check_file_exists(datasource_routes),
        "status": "passed" if check_file_exists(datasource_routes) else "failed"
    }
    
    if check_file_exists(datasource_routes):
        dashboard_checks["dashboard_api_endpoints"] = check_code_pattern(
            datasource_routes,
            [r"/api/v1/data-sources/metrics", r"get_data_sources_metrics"],
            "仪表盘API端点"
        )
    
    # 3. 检查前端仪表盘页面（如果存在）
    frontend_dashboard = "web-static/data-sources-config.html"
    dashboard_checks["frontend_dashboard"] = {
        "file": frontend_dashboard,
        "exists": check_file_exists(frontend_dashboard),
        "status": "passed" if check_file_exists(frontend_dashboard) else "warning"
    }
    
    # 更新统计
    for check_name, check_result in dashboard_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
            check_results["summary"]["total_checks"] += 1
    
    check_results["dashboard_checks"] = dashboard_checks
    return dashboard_checks


def check_data_source_monitoring():
    """检查数据源监控实现"""
    print("\n" + "="*80)
    print("检查数据源监控实现")
    print("="*80)
    
    monitoring_checks = {}
    
    # 1. 检查数据源健康监控器
    source_manager_file = "src/data/sources/intelligent_source_manager.py"
    monitoring_checks["source_manager_file"] = {
        "file": source_manager_file,
        "exists": check_file_exists(source_manager_file),
        "status": "passed" if check_file_exists(source_manager_file) else "failed"
    }
    
    if check_file_exists(source_manager_file):
        # 检查DataSourceHealthMonitor类
        monitoring_checks["DataSourceHealthMonitor"] = check_code_pattern(
            source_manager_file,
            [r"class DataSourceHealthMonitor", r"record_request", r"get_health_report"],
            "DataSourceHealthMonitor健康监控器"
        )
        
        # 检查健康状态枚举
        monitoring_checks["DataSourceStatus"] = check_code_pattern(
            source_manager_file,
            [r"DataSourceStatus", r"HEALTHY", r"DEGRADED", r"UNHEALTHY"],
            "DataSourceStatus状态枚举"
        )
        
        # 检查监控循环
        monitoring_checks["monitoring_loop"] = check_code_pattern(
            source_manager_file,
            [r"start_monitoring", r"_monitor_loop", r"is_monitoring"],
            "监控循环实现"
        )
        
        # 检查IntelligentSourceManager
        monitoring_checks["IntelligentSourceManager"] = check_code_pattern(
            source_manager_file,
            [r"class IntelligentSourceManager", r"register_source", r"health_monitor"],
            "IntelligentSourceManager智能管理器"
        )
    
    # 2. 检查数据源监控API
    datasource_routes = "src/gateway/web/datasource_routes.py"
    if check_file_exists(datasource_routes):
        monitoring_checks["monitoring_api"] = check_code_pattern(
            datasource_routes,
            [r"/api/v1/data-sources/metrics", r"get_data_sources_metrics"],
            "数据源监控API端点"
        )
    
    # 3. 检查性能监控集成
    performance_monitor = "src/data/monitoring/performance_monitor.py"
    monitoring_checks["performance_monitor"] = {
        "file": performance_monitor,
        "exists": check_file_exists(performance_monitor),
        "status": "passed" if check_file_exists(performance_monitor) else "warning"
    }
    
    # 更新统计
    for check_name, check_result in monitoring_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
            check_results["summary"]["total_checks"] += 1
    
    check_results["data_source_monitoring_checks"] = monitoring_checks
    return monitoring_checks


def check_data_source_config():
    """检查数据源配置管理实现"""
    print("\n" + "="*80)
    print("检查数据源配置管理实现")
    print("="*80)
    
    config_checks = {}
    
    # 1. 检查数据源配置管理器
    config_manager_file = "src/gateway/web/data_source_config_manager.py"
    config_checks["config_manager_file"] = {
        "file": config_manager_file,
        "exists": check_file_exists(config_manager_file),
        "status": "passed" if check_file_exists(config_manager_file) else "failed"
    }
    
    if check_file_exists(config_manager_file):
        # 检查DataSourceConfigManager类
        config_checks["DataSourceConfigManager"] = check_code_pattern(
            config_manager_file,
            [r"class DataSourceConfigManager", r"load_config", r"save_config"],
            "DataSourceConfigManager配置管理器"
        )
        
        # 检查基础设施层配置管理器集成
        config_checks["UnifiedConfigManager_integration"] = check_code_pattern(
            config_manager_file,
            [r"UnifiedConfigManager", r"config_manager", r"get\("],
            "基础设施层配置管理器集成"
        )
        
        # 检查配置验证
        config_checks["config_validation"] = check_code_pattern(
            config_manager_file,
            [r"_validate", r"validation", r"validate_data_source"],
            "配置验证功能"
        )
        
        # 检查环境隔离
        config_checks["environment_isolation"] = check_code_pattern(
            config_manager_file,
            [r"RQA_ENV", r"production", r"development", r"environment"],
            "环境隔离支持"
        )
        
        # 检查配置CRUD操作
        config_checks["config_crud"] = check_code_pattern(
            config_manager_file,
            [r"add_data_source", r"update_data_source", r"delete_data_source", r"get_data_source"],
            "配置CRUD操作"
        )
    
    # 2. 检查配置管理路由
    config_manager_legacy = "src/gateway/web/config_manager.py"
    config_checks["config_manager_legacy"] = {
        "file": config_manager_legacy,
        "exists": check_file_exists(config_manager_legacy),
        "status": "passed" if check_file_exists(config_manager_legacy) else "warning"
    }
    
    if check_file_exists(config_manager_legacy):
        config_checks["load_save_functions"] = check_code_pattern(
            config_manager_legacy,
            [r"load_data_sources", r"save_data_sources"],
            "配置加载保存函数"
        )
    
    # 3. 检查配置API路由
    datasource_routes = "src/gateway/web/datasource_routes.py"
    if check_file_exists(datasource_routes):
        config_checks["config_api_routes"] = check_code_pattern(
            datasource_routes,
            [r"/api/v1/data/sources", r"get_data_sources", r"create_or_get_data_sources"],
            "配置管理API路由"
        )
    
    # 4. 检查前端配置页面
    frontend_config = "web-static/data-sources-config.html"
    config_checks["frontend_config_page"] = {
        "file": frontend_config,
        "exists": check_file_exists(frontend_config),
        "status": "passed" if check_file_exists(frontend_config) else "warning"
    }
    
    # 更新统计
    for check_name, check_result in config_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
            check_results["summary"]["total_checks"] += 1
    
    check_results["data_source_config_checks"] = config_checks
    return config_checks


def check_architecture_compliance():
    """检查架构设计符合性"""
    print("\n" + "="*80)
    print("检查架构设计符合性")
    print("="*80)
    
    compliance_checks = {}
    
    # 1. 检查数据层架构符合性
    data_layer_doc = "docs/architecture/data_layer_architecture_design.md"
    compliance_checks["data_layer_doc"] = {
        "file": data_layer_doc,
        "exists": check_file_exists(data_layer_doc),
        "status": "passed" if check_file_exists(data_layer_doc) else "warning"
    }
    
    # 2. 检查网关层架构符合性
    gateway_layer_doc = "docs/architecture/gateway_layer_architecture_design.md"
    compliance_checks["gateway_layer_doc"] = {
        "file": gateway_layer_doc,
        "exists": check_file_exists(gateway_layer_doc),
        "status": "passed" if check_file_exists(gateway_layer_doc) else "warning"
    }
    
    # 3. 检查监控层架构符合性
    monitoring_layer_doc = "docs/architecture/monitoring_layer_architecture_design.md"
    compliance_checks["monitoring_layer_doc"] = {
        "file": monitoring_layer_doc,
        "exists": check_file_exists(monitoring_layer_doc),
        "status": "passed" if check_file_exists(monitoring_layer_doc) else "warning"
    }
    
    # 4. 检查基础设施层集成
    dashboard_file = "src/data/monitoring/dashboard.py"
    if check_file_exists(dashboard_file):
        compliance_checks["infrastructure_logging"] = check_code_pattern(
            dashboard_file,
            [r"get_unified_logger", r"unified_logger"],
            "基础设施层统一日志集成"
        )
    
    # 5. 检查核心服务层集成（EventBus、ServiceContainer等）
    data_collectors = "src/gateway/web/data_collectors.py"
    if check_file_exists(data_collectors):
        # EventBus支持publish()和publish_event()两种方法，检查任一即可
        compliance_checks["event_bus_integration"] = check_code_pattern(
            data_collectors,
            [r"EventBus", r"event_bus", r"\.publish\(|publish_event\("],
            "事件总线集成"
        )
        
        compliance_checks["service_container"] = check_code_pattern(
            data_collectors,
            [r"DependencyContainer", r"ServiceContainer", r"container"],
            "服务容器集成"
        )
        
        compliance_checks["business_orchestrator"] = check_code_pattern(
            data_collectors,
            [r"BusinessProcessOrchestrator", r"orchestrator"],
            "业务流程编排器集成"
        )
    
    # 6. 检查数据适配器模式
    config_manager_file = "src/gateway/web/data_source_config_manager.py"
    if check_file_exists(config_manager_file):
        # 检查适配器模式：UnifiedConfigManager是适配器模式的实现，代码注释中应说明
        compliance_checks["adapter_pattern"] = check_code_pattern(
            config_manager_file,
            [r"UnifiedConfigManager", r"适配器模式|adapter.*pattern|integration"],
            "适配器模式使用"
        )
    
    # 更新统计
    for check_name, check_result in compliance_checks.items():
        if isinstance(check_result, dict):
            status = check_result.get("status", "unknown")
            if status == "passed":
                check_results["summary"]["passed"] += 1
            elif status == "failed":
                check_results["summary"]["failed"] += 1
            elif status == "warning":
                check_results["summary"]["warnings"] += 1
            check_results["summary"]["total_checks"] += 1
    
    check_results["architecture_compliance"] = compliance_checks
    return compliance_checks


def generate_report():
    """生成检查报告"""
    print("\n" + "="*80)
    print("生成检查报告")
    print("="*80)
    
    # 计算通过率
    total = check_results["summary"]["total_checks"]
    passed = check_results["summary"]["passed"]
    failed = check_results["summary"]["failed"]
    warnings = check_results["summary"]["warnings"]
    
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    # 生成报告文件
    report_file = project_root / "docs" / f"strategy_data_collection_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 量化策略开发流程数据收集功能架构符合性检查报告\n\n")
        f.write(f"**检查时间**: {check_results['timestamp']}\n\n")
        f.write("## 📊 检查摘要\n\n")
        f.write(f"- **总检查项**: {total}\n")
        f.write(f"- **通过**: {passed} ✅\n")
        f.write(f"- **失败**: {failed} ❌\n")
        f.write(f"- **警告**: {warnings} ⚠️\n")
        f.write(f"- **通过率**: {pass_rate:.2f}%\n\n")
        
        f.write("## 1. 数据收集仪表盘检查\n\n")
        for check_name, check_result in check_results["dashboard_checks"].items():
            status_icon = "✅" if isinstance(check_result, dict) and check_result.get("status") == "passed" else \
                         "❌" if isinstance(check_result, dict) and check_result.get("status") == "failed" else "⚠️"
            f.write(f"### {check_name} {status_icon}\n\n")
            if isinstance(check_result, dict):
                f.write(f"- **文件**: {check_result.get('file', 'N/A')}\n")
                f.write(f"- **状态**: {check_result.get('status', 'unknown')}\n")
                if 'message' in check_result:
                    f.write(f"- **消息**: {check_result['message']}\n")
            f.write("\n")
        
        f.write("## 2. 数据源监控检查\n\n")
        for check_name, check_result in check_results["data_source_monitoring_checks"].items():
            status_icon = "✅" if isinstance(check_result, dict) and check_result.get("status") == "passed" else \
                         "❌" if isinstance(check_result, dict) and check_result.get("status") == "failed" else "⚠️"
            f.write(f"### {check_name} {status_icon}\n\n")
            if isinstance(check_result, dict):
                f.write(f"- **文件**: {check_result.get('file', 'N/A')}\n")
                f.write(f"- **状态**: {check_result.get('status', 'unknown')}\n")
                if 'message' in check_result:
                    f.write(f"- **消息**: {check_result['message']}\n")
            f.write("\n")
        
        f.write("## 3. 数据源配置管理检查\n\n")
        for check_name, check_result in check_results["data_source_config_checks"].items():
            status_icon = "✅" if isinstance(check_result, dict) and check_result.get("status") == "passed" else \
                         "❌" if isinstance(check_result, dict) and check_result.get("status") == "failed" else "⚠️"
            f.write(f"### {check_name} {status_icon}\n\n")
            if isinstance(check_result, dict):
                f.write(f"- **文件**: {check_result.get('file', 'N/A')}\n")
                f.write(f"- **状态**: {check_result.get('status', 'unknown')}\n")
                if 'message' in check_result:
                    f.write(f"- **消息**: {check_result['message']}\n")
            f.write("\n")
        
        f.write("## 4. 架构设计符合性检查\n\n")
        for check_name, check_result in check_results["architecture_compliance"].items():
            status_icon = "✅" if isinstance(check_result, dict) and check_result.get("status") == "passed" else \
                         "❌" if isinstance(check_result, dict) and check_result.get("status") == "failed" else "⚠️"
            f.write(f"### {check_name} {status_icon}\n\n")
            if isinstance(check_result, dict):
                f.write(f"- **文件**: {check_result.get('file', 'N/A')}\n")
                f.write(f"- **状态**: {check_result.get('status', 'unknown')}\n")
                if 'message' in check_result:
                    f.write(f"- **消息**: {check_result['message']}\n")
            f.write("\n")
        
        f.write("## 📝 详细检查结果\n\n")
        f.write("```json\n")
        f.write(json.dumps(check_results, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    
    print(f"\n检查报告已生成: {report_file}")
    return report_file


def main():
    """主函数"""
    print("="*80)
    print("量化策略开发流程数据收集功能架构符合性检查")
    print("="*80)
    
    # 执行各项检查
    check_dashboard_implementation()
    check_data_source_monitoring()
    check_data_source_config()
    check_architecture_compliance()
    
    # 生成报告
    report_file = generate_report()
    
    # 打印摘要
    print("\n" + "="*80)
    print("检查摘要")
    print("="*80)
    print(f"总检查项: {check_results['summary']['total_checks']}")
    print(f"通过: {check_results['summary']['passed']} ✅")
    print(f"失败: {check_results['summary']['failed']} ❌")
    print(f"警告: {check_results['summary']['warnings']} ⚠️")
    print(f"通过率: {(check_results['summary']['passed'] / check_results['summary']['total_checks'] * 100) if check_results['summary']['total_checks'] > 0 else 0:.2f}%")
    print(f"\n详细报告: {report_file}")
    print("="*80)


if __name__ == "__main__":
    main()


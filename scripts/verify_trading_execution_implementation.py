#!/usr/bin/env python3
"""
验证交易执行流程仪表盘实施完整性
按照检查计划验证所有功能、持久化和修复
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_persistence_modules():
    """检查持久化模块是否存在"""
    print("\n=== 检查持久化模块 ===")
    
    modules = {
        "execution_persistence.py": "策略执行状态持久化",
        "signal_persistence.py": "交易信号持久化",
        "routing_persistence.py": "订单路由持久化",
        "trading_execution_persistence.py": "交易执行记录持久化"
    }
    
    results = {}
    for module, desc in modules.items():
        filepath = project_root / "src" / "gateway" / "web" / module
        exists = filepath.exists()
        results[module] = {
            "exists": exists,
            "description": desc,
            "path": str(filepath)
        }
        status = "✅" if exists else "❌"
        print(f"{status} {module} - {desc}")
    
    return results

def check_api_routes():
    """检查API路由是否实现"""
    print("\n=== 检查API路由 ===")
    
    routes = {
        "/api/v1/trading/execution/flow": "交易执行流程监控",
        "/api/v1/trading/overview": "交易概览"
    }
    
    results = {}
    routes_file = project_root / "src" / "gateway" / "web" / "trading_execution_routes.py"
    
    if routes_file.exists():
        content = routes_file.read_text(encoding='utf-8')
        for route, desc in routes.items():
            exists = route in content
            results[route] = {
                "exists": exists,
                "description": desc
            }
            status = "✅" if exists else "❌"
            print(f"{status} {route} - {desc}")
    else:
        print("❌ trading_execution_routes.py 不存在")
        for route, desc in routes.items():
            results[route] = {"exists": False, "description": desc}
    
    return results

def check_service_integration():
    """检查服务层集成"""
    print("\n=== 检查服务层集成 ===")
    
    services = {
        "strategy_execution_service.py": ["execution_persistence", "save_execution_state"],
        "trading_signal_service.py": ["signal_persistence", "save_signal"],
        "order_routing_service.py": ["routing_persistence", "save_routing_decision"]
    }
    
    results = {}
    for service_file, [persistence_module, save_function] in services.items():
        filepath = project_root / "src" / "gateway" / "web" / service_file
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            has_import = persistence_module in content
            has_save = save_function in content
            integrated = has_import and has_save
            
            results[service_file] = {
                "integrated": integrated,
                "has_import": has_import,
                "has_save": has_save
            }
            status = "✅" if integrated else "⚠️"
            print(f"{status} {service_file} - 集成持久化: {integrated}")
        else:
            results[service_file] = {"integrated": False}
            print(f"❌ {service_file} 不存在")
    
    return results

def check_hardcoded_values():
    """检查硬编码值是否已修复"""
    print("\n=== 检查硬编码值修复 ===")
    
    checks = {
        "web-static/trading-execution.html": {
            "hardcoded_fallback": ["15ms", "98.5%", "2.3/秒", "87.3%", "8.5ms", "2.1%", "1.8/秒"],
            "should_have": ["--", "数据不可用"]
        },
        "src/gateway/web/trading_signal_service.py": {
            "hardcoded_effectiveness": ["0.75", "0.68", "0.82"],
            "should_not_have": ["effectiveness = {", '"买入信号": 0.75']
        }
    }
    
    results = {}
    for file_path, patterns in checks.items():
        filepath = project_root / file_path
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            
            # 检查是否还有硬编码值
            has_hardcoded = False
            for pattern in patterns.get("hardcoded_fallback", []) + patterns.get("hardcoded_effectiveness", []):
                if pattern in content:
                    has_hardcoded = True
                    break
            
            # 检查是否已修复（有替代值）
            is_fixed = False
            for pattern in patterns.get("should_have", []):
                if pattern in content:
                    is_fixed = True
                    break
            
            # 对于有效性数据，检查是否已移除硬编码
            if "hardcoded_effectiveness" in patterns:
                # 检查是否还有硬编码的有效性字典
                has_old_pattern = '"买入信号": 0.75' in content or '"卖出信号": 0.68' in content or '"持有信号": 0.82' in content
                # 检查是否使用了动态计算（从持久化存储获取）
                has_dynamic_calc = ("list_signals" in content or "executed_signals" in content) and "signal_persistence" in content
                is_fixed = not has_old_pattern and has_dynamic_calc
            else:
                is_fixed = not has_hardcoded or is_fixed
            
            results[file_path] = {
                "has_hardcoded": has_hardcoded,
                "is_fixed": is_fixed
            }
            status = "✅" if is_fixed else "❌"
            print(f"{status} {file_path} - 硬编码修复: {is_fixed}")
        else:
            results[file_path] = {"is_fixed": False}
            print(f"❌ {file_path} 不存在")
    
    return results

def check_mock_functions():
    """检查模拟数据函数是否已删除"""
    print("\n=== 检查模拟数据函数 ===")
    
    files_to_check = {
        "src/gateway/web/trading_signal_service.py": ["_get_mock_signals"],
        "src/gateway/web/order_routing_service.py": ["_get_mock_routing_decisions"]
    }
    
    results = {}
    for file_path, functions in files_to_check.items():
        filepath = project_root / file_path
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            
            found_functions = []
            for func_name in functions:
                if func_name in content:
                    # 检查是否是函数定义（不是注释）
                    func_def = f"def {func_name}("
                    if func_def in content:
                        found_functions.append(func_name)
            
            is_removed = len(found_functions) == 0
            results[file_path] = {
                "removed": is_removed,
                "found_functions": found_functions
            }
            status = "✅" if is_removed else "❌"
            print(f"{status} {file_path} - 模拟函数已删除: {is_removed}")
            if found_functions:
                print(f"   仍存在的函数: {', '.join(found_functions)}")
        else:
            results[file_path] = {"removed": False}
            print(f"❌ {file_path} 不存在")
    
    return results

def check_chart_initialization():
    """检查图表初始化是否已修复"""
    print("\n=== 检查图表初始化 ===")
    
    filepath = project_root / "web-static" / "trading-execution.html"
    if filepath.exists():
        content = filepath.read_text(encoding='utf-8')
        
        # 检查执行性能图表
        has_hardcoded_perf = "[98.2, 97.8, 98.5" in content
        has_empty_perf = 'data: []' in content or 'datasets: []' in content
        
        # 检查风险指标图表
        has_hardcoded_risk = "[2.1, 1.8, 1.5" in content
        has_empty_risk = 'data: []' in content
        
        is_fixed = (not has_hardcoded_perf or has_empty_perf) and (not has_hardcoded_risk or has_empty_risk)
        
        results = {
            "has_hardcoded_perf": has_hardcoded_perf,
            "has_empty_perf": has_empty_perf,
            "has_hardcoded_risk": has_hardcoded_risk,
            "has_empty_risk": has_empty_risk,
            "is_fixed": is_fixed
        }
        
        status = "✅" if is_fixed else "⚠️"
        print(f"{status} 图表初始化修复: {is_fixed}")
        if has_hardcoded_perf:
            print("   ⚠️ 执行性能图表仍有硬编码数据")
        if has_hardcoded_risk:
            print("   ⚠️ 风险指标图表仍有硬编码数据")
    else:
        results = {"is_fixed": False}
        print("❌ trading-execution.html 不存在")
    
    return results

def generate_summary(persistence_results, api_results, service_results, hardcoded_results, mock_results, chart_results):
    """生成验证总结"""
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    # 持久化模块
    persistence_count = sum(1 for r in persistence_results.values() if r["exists"])
    print(f"\n持久化模块: {persistence_count}/4")
    
    # API路由
    api_count = sum(1 for r in api_results.values() if r["exists"])
    print(f"API路由: {api_count}/2")
    
    # 服务层集成
    service_count = sum(1 for r in service_results.values() if r.get("integrated", False))
    print(f"服务层集成: {service_count}/3")
    
    # 硬编码修复
    hardcoded_fixed = sum(1 for r in hardcoded_results.values() if r.get("is_fixed", False))
    print(f"硬编码修复: {hardcoded_fixed}/{len(hardcoded_results)}")
    
    # 模拟函数删除
    mock_removed = sum(1 for r in mock_results.values() if r.get("removed", False))
    print(f"模拟函数删除: {mock_removed}/{len(mock_results)}")
    
    # 图表初始化
    chart_fixed = chart_results.get("is_fixed", False)
    print(f"图表初始化修复: {'✅' if chart_fixed else '❌'}")
    
    # 总体状态
    all_persistence = persistence_count == 4
    all_api = api_count == 2
    all_service = service_count == 3
    all_hardcoded = hardcoded_fixed == len(hardcoded_results)
    all_mock = mock_removed == len(mock_results)
    
    overall = all_persistence and all_api and all_service and all_hardcoded and all_mock and chart_fixed
    
    print(f"\n总体状态: {'✅ 全部完成' if overall else '⚠️ 部分完成'}")
    
    return {
        "persistence": {"count": persistence_count, "total": 4, "complete": all_persistence},
        "api": {"count": api_count, "total": 2, "complete": all_api},
        "service": {"count": service_count, "total": 3, "complete": all_service},
        "hardcoded": {"count": hardcoded_fixed, "total": len(hardcoded_results), "complete": all_hardcoded},
        "mock": {"count": mock_removed, "total": len(mock_results), "complete": all_mock},
        "chart": {"fixed": chart_fixed},
        "overall": overall
    }

def main():
    """主函数"""
    print("="*60)
    print("交易执行流程仪表盘实施验证")
    print("="*60)
    
    # 执行各项检查
    persistence_results = check_persistence_modules()
    api_results = check_api_routes()
    service_results = check_service_integration()
    hardcoded_results = check_hardcoded_values()
    mock_results = check_mock_functions()
    chart_results = check_chart_initialization()
    
    # 生成总结
    summary = generate_summary(
        persistence_results, api_results, service_results,
        hardcoded_results, mock_results, chart_results
    )
    
    # 保存结果到JSON文件
    output_file = project_root / "docs" / "trading_execution_verification_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "persistence": persistence_results,
        "api": api_results,
        "service": service_results,
        "hardcoded": hardcoded_results,
        "mock": mock_results,
        "chart": chart_results,
        "summary": summary
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n验证结果已保存到: {output_file}")
    
    return 0 if summary["overall"] else 1

if __name__ == "__main__":
    sys.exit(main())


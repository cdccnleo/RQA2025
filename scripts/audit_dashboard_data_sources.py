"""
仪表盘数据来源审计脚本
扫描所有API路由和服务层，识别模拟数据和硬编码
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
GATEWAY_WEB_DIR = PROJECT_ROOT / "src" / "gateway" / "web"
WEB_STATIC_DIR = PROJECT_ROOT / "web-static"

# 审计结果
audit_results = {
    "timestamp": datetime.now().isoformat(),
    "mock_functions": [],
    "mock_imports": [],
    "mock_calls": [],
    "hardcoded_data": [],
    "todo_comments": [],
    "service_layer_status": {},
    "api_routes_status": {},
    "summary": {}
}


def find_mock_functions(file_path: Path) -> List[Dict[str, Any]]:
    """查找所有模拟数据函数定义"""
    mock_functions = []
    try:
        content = file_path.read_text(encoding='utf-8')
        # 查找 _get_mock_* 函数定义
        pattern = r'def\s+(_get_mock_\w+)\s*\([^)]*\)\s*:'
        matches = re.finditer(pattern, content)
        for match in matches:
            mock_functions.append({
                "file": str(file_path.relative_to(PROJECT_ROOT)),
                "function": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return mock_functions


def find_mock_imports(file_path: Path) -> List[Dict[str, Any]]:
    """查找所有模拟数据函数导入"""
    mock_imports = []
    try:
        content = file_path.read_text(encoding='utf-8')
        # 查找导入 _get_mock_* 的语句
        pattern = r'(_get_mock_\w+)'
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '_get_mock_' in line and ('import' in line or 'from' in line):
                match = re.search(pattern, line)
                if match:
                    mock_imports.append({
                        "file": str(file_path.relative_to(PROJECT_ROOT)),
                        "import": match.group(1),
                        "line": i,
                        "code": line.strip()
                    })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return mock_imports


def find_mock_calls(file_path: Path) -> List[Dict[str, Any]]:
    """查找所有模拟数据函数调用"""
    mock_calls = []
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # 查找调用 _get_mock_* 的语句
            if '_get_mock_' in line:
                match = re.search(r'(_get_mock_\w+)\s*\(', line)
                if match:
                    # 检查上下文，看是否有降级注释
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 2)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    mock_calls.append({
                        "file": str(file_path.relative_to(PROJECT_ROOT)),
                        "function": match.group(1),
                        "line": i,
                        "code": line.strip(),
                        "context": context,
                        "has_fallback_comment": "降级" in context or "fallback" in context.lower()
                    })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return mock_calls


def find_hardcoded_data(file_path: Path) -> List[Dict[str, Any]]:
    """查找硬编码数据"""
    hardcoded = []
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # 查找硬编码的性能估算值
        patterns = [
            (r'base_latency\s*=\s*(\d+)', "硬编码延迟值"),
            (r'base_throughput\s*=\s*(\d+)', "硬编码吞吐量值"),
            (r'random\.(uniform|choice|randint)', "随机生成数据"),
            (r'#\s*⚠️.*估算|#\s*警告.*估算', "性能估算注释"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line):
                    hardcoded.append({
                        "file": str(file_path.relative_to(PROJECT_ROOT)),
                        "line": i,
                        "code": line.strip(),
                        "type": description
                    })
                    break
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return hardcoded


def find_todo_comments(file_path: Path) -> List[Dict[str, Any]]:
    """查找TODO注释，特别是关于对接实际系统的"""
    todos = []
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'TODO' in line.upper() and ('对接' in line or '实际' in line or 'real' in line.lower()):
                todos.append({
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "line": i,
                    "code": line.strip()
                })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return todos


def check_service_layer_status(file_path: Path) -> Dict[str, Any]:
    """检查服务层对接状态"""
    status = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "imports_real_components": False,
        "returns_empty": False,
        "has_mock_fallback": False,
        "components": []
    }
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 检查是否导入真实组件
        real_component_patterns = [
            r'from\s+src\.(trading|data|ml|strategy|risk|features)',
            r'import\s+(\w+Generator|\w+Monitor|\w+Manager|\w+Engine)',
        ]
        
        for pattern in real_component_patterns:
            if re.search(pattern, content):
                status["imports_real_components"] = True
                matches = re.finditer(pattern, content)
                for match in matches:
                    status["components"].append(match.group(0))
        
        # 检查是否返回空数据
        if 'return []' in content or 'return {}' in content or 'return None' in content:
            status["returns_empty"] = True
        
        # 检查是否有模拟数据降级
        if '_get_mock_' in content:
            status["has_mock_fallback"] = True
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return status


def check_api_route_status(file_path: Path) -> Dict[str, Any]:
    """检查API路由状态"""
    status = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "endpoints": [],
        "has_mock_fallback": False,
        "uses_real_service": False
    }
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 查找所有API端点
        endpoint_pattern = r'@router\.(get|post|put|delete)\("([^"]+)"\)'
        endpoints = re.finditer(endpoint_pattern, content)
        for match in endpoints:
            method = match.group(1)
            path = match.group(2)
            status["endpoints"].append(f"{method.upper()} {path}")
        
        # 检查是否有模拟数据降级
        if '_get_mock_' in content:
            status["has_mock_fallback"] = True
        
        # 检查是否使用真实服务层
        if re.search(r'from\s+\.\w+_service\s+import', content):
            status["uses_real_service"] = True
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return status


def main():
    """主函数"""
    print("=" * 80)
    print("仪表盘数据来源审计")
    print("=" * 80)
    print()
    
    # 扫描所有Python文件
    python_files = []
    if GATEWAY_WEB_DIR.exists():
        python_files.extend(GATEWAY_WEB_DIR.rglob("*.py"))
    
    print(f"扫描 {len(python_files)} 个Python文件...")
    print()
    
    # 审计每个文件
    for file_path in python_files:
        if file_path.name.startswith('__'):
            continue
            
        print(f"检查: {file_path.relative_to(PROJECT_ROOT)}")
        
        # 查找模拟数据函数定义
        mock_funcs = find_mock_functions(file_path)
        audit_results["mock_functions"].extend(mock_funcs)
        
        # 查找模拟数据导入
        mock_imports = find_mock_imports(file_path)
        audit_results["mock_imports"].extend(mock_imports)
        
        # 查找模拟数据调用
        mock_calls = find_mock_calls(file_path)
        audit_results["mock_calls"].extend(mock_calls)
        
        # 查找硬编码数据
        hardcoded = find_hardcoded_data(file_path)
        audit_results["hardcoded_data"].extend(hardcoded)
        
        # 查找TODO注释
        todos = find_todo_comments(file_path)
        audit_results["todo_comments"].extend(todos)
        
        # 检查服务层状态
        if '_service.py' in file_path.name:
            service_status = check_service_layer_status(file_path)
            audit_results["service_layer_status"][file_path.name] = service_status
        
        # 检查API路由状态
        if '_routes.py' in file_path.name:
            route_status = check_api_route_status(file_path)
            audit_results["api_routes_status"][file_path.name] = route_status
    
    # 生成摘要
    audit_results["summary"] = {
        "total_mock_functions": len(audit_results["mock_functions"]),
        "total_mock_imports": len(audit_results["mock_imports"]),
        "total_mock_calls": len(audit_results["mock_calls"]),
        "total_hardcoded": len(audit_results["hardcoded_data"]),
        "total_todos": len(audit_results["todo_comments"]),
        "service_layers_checked": len(audit_results["service_layer_status"]),
        "api_routes_checked": len(audit_results["api_routes_status"]),
        "api_routes_with_mock_fallback": sum(
            1 for s in audit_results["api_routes_status"].values() 
            if s.get("has_mock_fallback", False)
        )
    }
    
    # 保存审计报告
    report_file = PROJECT_ROOT / "docs" / "dashboard_data_authenticity_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成Markdown报告
    report_content = generate_markdown_report(audit_results)
    report_file.write_text(report_content, encoding='utf-8')
    
    # 保存JSON报告
    json_file = PROJECT_ROOT / "docs" / "dashboard_data_authenticity_report.json"
    json_file.write_text(json.dumps(audit_results, indent=2, ensure_ascii=False), encoding='utf-8')
    
    # 打印摘要
    print()
    print("=" * 80)
    print("审计摘要")
    print("=" * 80)
    print(f"模拟数据函数定义: {audit_results['summary']['total_mock_functions']}")
    print(f"模拟数据函数导入: {audit_results['summary']['total_mock_imports']}")
    print(f"模拟数据函数调用: {audit_results['summary']['total_mock_calls']}")
    print(f"硬编码数据: {audit_results['summary']['total_hardcoded']}")
    print(f"TODO注释: {audit_results['summary']['total_todos']}")
    print(f"服务层检查: {audit_results['summary']['service_layers_checked']}")
    print(f"API路由检查: {audit_results['summary']['api_routes_checked']}")
    print(f"有模拟数据降级的API路由: {audit_results['summary']['api_routes_with_mock_fallback']}")
    print()
    print(f"详细报告已保存到: {report_file}")
    print(f"JSON报告已保存到: {json_file}")


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """生成Markdown格式的审计报告"""
    lines = []
    lines.append("# 仪表盘数据来源审计报告")
    lines.append("")
    lines.append(f"**生成时间**: {results['timestamp']}")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    summary = results["summary"]
    lines.append(f"- 模拟数据函数定义: {summary['total_mock_functions']}")
    lines.append(f"- 模拟数据函数导入: {summary['total_mock_imports']}")
    lines.append(f"- 模拟数据函数调用: {summary['total_mock_calls']}")
    lines.append(f"- 硬编码数据: {summary['total_hardcoded']}")
    lines.append(f"- TODO注释: {summary['total_todos']}")
    lines.append(f"- 服务层检查: {summary['service_layers_checked']}")
    lines.append(f"- API路由检查: {summary['api_routes_checked']}")
    lines.append(f"- 有模拟数据降级的API路由: {summary['api_routes_with_mock_fallback']}")
    lines.append("")
    
    # 模拟数据函数调用详情
    if results["mock_calls"]:
        lines.append("## 模拟数据函数调用详情")
        lines.append("")
        lines.append("| 文件 | 函数 | 行号 | 有降级注释 |")
        lines.append("|------|------|------|------------|")
        for call in results["mock_calls"]:
            has_fallback = "是" if call.get("has_fallback_comment") else "否"
            lines.append(f"| {call['file']} | {call['function']} | {call['line']} | {has_fallback} |")
        lines.append("")
    
    # API路由状态
    if results["api_routes_status"]:
        lines.append("## API路由状态")
        lines.append("")
        lines.append("| 文件 | 端点数量 | 使用真实服务 | 有模拟数据降级 |")
        lines.append("|------|----------|--------------|----------------|")
        for file_name, status in results["api_routes_status"].items():
            uses_real = "是" if status.get("uses_real_service") else "否"
            has_mock = "是" if status.get("has_mock_fallback") else "否"
            lines.append(f"| {file_name} | {len(status.get('endpoints', []))} | {uses_real} | {has_mock} |")
        lines.append("")
    
    # 服务层状态
    if results["service_layer_status"]:
        lines.append("## 服务层状态")
        lines.append("")
        lines.append("| 文件 | 导入真实组件 | 返回空数据 | 有模拟数据降级 |")
        lines.append("|------|--------------|------------|----------------|")
        for file_name, status in results["service_layer_status"].items():
            imports_real = "是" if status.get("imports_real_components") else "否"
            returns_empty = "是" if status.get("returns_empty") else "否"
            has_mock = "是" if status.get("has_mock_fallback") else "否"
            lines.append(f"| {file_name} | {imports_real} | {returns_empty} | {has_mock} |")
        lines.append("")
    
    # 硬编码数据
    if results["hardcoded_data"]:
        lines.append("## 硬编码数据")
        lines.append("")
        for item in results["hardcoded_data"]:
            lines.append(f"- **{item['file']}:{item['line']}** - {item['type']}")
            lines.append(f"  ```python")
            lines.append(f"  {item['code']}")
            lines.append(f"  ```")
        lines.append("")
    
    # TODO注释
    if results["todo_comments"]:
        lines.append("## TODO注释（需要对接实际系统）")
        lines.append("")
        for todo in results["todo_comments"]:
            lines.append(f"- **{todo['file']}:{todo['line']}**")
            lines.append(f"  ```python")
            lines.append(f"  {todo['code']}")
            lines.append(f"  ```")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()


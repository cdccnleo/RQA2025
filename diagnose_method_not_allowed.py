#!/usr/bin/env python3
"""
详细诊断Method Not Allowed错误
"""

def diagnose_method_not_allowed():
    """详细诊断Method Not Allowed错误的可能原因"""

    print("🔍 Method Not Allowed错误详细诊断")
    print("=" * 50)

    # 1. 检查nginx配置
    print("\n1. Nginx配置检查:")
    with open('web-static/nginx.conf', 'r', encoding='utf-8') as f:
        nginx_config = f.read()

    nginx_checks = [
        ('API代理路径正确', 'proxy_pass http://rqa2025-app-main:8000' in nginx_config),
        ('CORS方法支持', 'GET, POST, OPTIONS, PUT, DELETE' in nginx_config),
        ('OPTIONS特殊处理', 'if ($request_method = \'OPTIONS\')' in nginx_config),
        ('OPTIONS返回CORS头', 'add_header \'Access-Control-Allow-Origin\' \'*\' always;' in nginx_config),
    ]

    for check_name, result in nginx_checks:
        status = '✅' if result else '❌'
        print(f'   {status} {check_name}')

    # 2. 检查前端fetch调用
    print("\n2. 前端API调用检查:")
    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 查找所有fetch调用
    import re
    fetch_calls = re.findall(r'fetch\([^,]+,\s*\{[^}]*method:\s*[\'"]([^\'"]+)[\'"]', html_content)

    print(f"   发现 {len(fetch_calls)} 个fetch调用:")
    for method in fetch_calls:
        print(f"   - {method}")

    # 检查是否有错误的HTTP方法
    invalid_methods = [m for m in fetch_calls if m not in ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']]
    if invalid_methods:
        print(f"   ❌ 发现无效的HTTP方法: {invalid_methods}")
    else:
        print("   ✅ 所有HTTP方法都有效")

    # 3. 检查API路由定义
    print("\n3. 后端API路由检查:")
    with open('src/gateway/web/api.py', 'r', encoding='utf-8') as f:
        api_content = f.read()

    routes = re.findall(r'@app\.(get|post|put|delete|options)\([\'"]([^\'"]+)[\'"]', api_content)

    print(f"   发现 {len(routes)} 个API路由:")
    for method, path in routes:
        print(f"   - {method.upper():6} {path}")

    # 检查数据源相关的路由
    data_source_routes = [r for r in routes if '/data/sources' in r[1]]
    print(f"\n   数据源相关路由 ({len(data_source_routes)} 个):")
    for method, path in data_source_routes:
        print(f"   - {method.upper():6} {path}")

    # 4. 可能的错误原因
    print("\n4. 可能的错误原因分析:")

    issues = []

    # 检查是否有未定义的路由
    frontend_urls = re.findall(r'fetch\([\'"`]([^\'"`]+)[\'"`]', html_content)
    api_urls = [path for _, path in routes]

    for frontend_url in frontend_urls:
        # 移除查询参数和片段
        clean_url = frontend_url.split('?')[0].split('#')[0]
        if '/api/v1/data/sources' in clean_url:
            # 检查是否匹配任何API路由
            matched = False
            for api_url in api_urls:
                if api_url in clean_url or clean_url in api_url:
                    matched = True
                    break
            if not matched:
                issues.append(f"前端URL '{clean_url}' 没有匹配的后端路由")

    # 检查nginx配置问题
    if 'return 204;' not in nginx_config:
        issues.append("nginx OPTIONS处理不完整")

    if issues:
        print("   发现问题:")
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ 未发现明显的配置问题")

    # 5. 建议的调试步骤
    print("\n5. 建议的调试步骤:")
    debug_steps = [
        "1. 打开浏览器开发者工具(F12) → Network标签",
        "2. 尝试添加数据源，观察网络请求",
        "3. 检查是否有OPTIONS预检请求及其响应",
        "4. 检查POST请求的详细信息和响应",
        "5. 查看浏览器控制台是否有JavaScript错误",
        "6. 确认nginx服务器是否正在运行",
        "7. 检查API服务器日志是否有错误",
    ]

    for step in debug_steps:
        print(f"   {step}")

    # 6. 快速修复建议
    print("\n6. 快速修复建议:")
    if 'return 204;' not in nginx_config:
        print("   - 修复nginx OPTIONS处理")
    if invalid_methods:
        print("   - 检查前端的HTTP方法使用")
    if issues:
        print("   - 解决上述配置问题")
    else:
        print("   - 检查浏览器缓存和代理设置")
        print("   - 尝试重启nginx和API服务器")

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    diagnose_method_not_allowed()

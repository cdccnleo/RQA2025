#!/usr/bin/env python3
"""
诊断数据源配置404错误
"""

def diagnose_404_error():
    """诊断404错误的原因和解决方案"""

    print("🔍 诊断数据源配置404错误")
    print("=" * 50)

    # 1. 检查运行环境
    print("\n1. 环境检查:")

    import socket
    import requests

    # 检查本地API服务器
    try:
        response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=5)
        print(f"   ✅ 本地API服务器 (localhost:8000): {response.status_code}")
        local_api_ok = True
    except:
        print("   ❌ 本地API服务器 (localhost:8000): 无法连接")
        local_api_ok = False

    # 检查nginx代理
    try:
        response = requests.get('http://localhost:8080/api/v1/data/sources', timeout=5)
        print(f"   ✅ Nginx代理 (localhost:8080): {response.status_code}")
        nginx_ok = True
    except:
        print("   ❌ Nginx代理 (localhost:8080): 无法连接")
        nginx_ok = False

    # 2. 检查前端环境检测
    print("\n2. 前端环境检测检查:")
    with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('getApiBaseUrl函数', 'function getApiBaseUrl()' in content),
        ('文件协议检测', 'window.location.protocol === \'file:\'' in content),
        ('开发环境URL', 'http://localhost:8000/api/v1/data/sources' in content),
        ('生产环境URL', '/api/v1/data/sources' in content),
    ]

    for check_name, result in checks:
        status = '✅' if result else '❌'
        print(f"   {status} {check_name}")

    # 3. 可能的错误原因
    print("\n3. 可能的错误原因分析:")

    issues = []

    if not local_api_ok and not nginx_ok:
        issues.append("API服务器和nginx代理都无法访问")
    elif not local_api_ok:
        issues.append("开发环境缺少本地API服务器")
    elif not nginx_ok:
        issues.append("生产环境nginx服务未启动")

    if issues:
        print("   发现问题:")
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ 网络连接正常")

    # 4. 解决方案
    print("\n4. 解决方案:")

    solutions = [
        ("开发环境", [
            "1. 启动FastAPI服务器: python -m uvicorn src.gateway.web.api:app --reload --port 8000",
            "2. 直接打开HTML文件或使用本地开发服务器",
            "3. 前端会自动检测并使用 http://localhost:8000/api/v1/data/sources"
        ]),
        ("生产环境", [
            "1. 启动完整系统: docker-compose up",
            "2. 访问: http://localhost:8080/data-sources",
            "3. API通过nginx代理访问后端服务"
        ]),
        ("调试方法", [
            "1. 打开浏览器开发者工具(F12) → Console标签",
            "2. 查看 'API URL:' 日志，确认使用的API地址",
            "3. 检查Network标签中的请求状态",
            "4. 如果是CORS错误，检查nginx配置"
        ])
    ]

    for env, steps in solutions:
        print(f"\n   {env}:")
        for step in steps:
            print(f"      {step}")

    # 5. 快速修复建议
    print("\n5. 快速修复建议:")

    if not local_api_ok:
        print("   - 启动本地API服务器以支持开发环境")
    if not nginx_ok:
        print("   - 启动nginx服务以支持生产环境")

    print("   - 清除浏览器缓存 (Ctrl+F5)")
    print("   - 检查浏览器控制台错误信息")

    # 6. 改进建议
    print("\n6. 改进建议:")
    print("   - 添加更好的错误处理和用户提示")
    print("   - 在API不可用时显示离线模式")
    print("   - 添加连接重试机制")

if __name__ == "__main__":
    import os
    os.chdir('C:\\PythonProject\\RQA2025')
    diagnose_404_error()

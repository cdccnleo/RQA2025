#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json

def test_api_connection():
    """测试API连接功能"""
    print("🔌 开始API连接功能测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 1. 检查API基础URL配置
    api_base_checks = [
        ('getApiBaseUrl函数', 'function getApiBaseUrl' in content),
        ('本地开发环境URL', 'localhost:8000' in content),
        ('生产环境相对URL', "'/api/v1'" in content or '"/api/v1"' in content),
        ('协议检测逻辑', 'window.location.protocol' in content)
    ]

    print("\n🔗 API基础URL配置检查:")
    api_config_score = 0
    for check_name, result in api_base_checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            api_config_score += 1

    # 2. 检查API端点定义
    api_endpoints = [
        'getDataSourcesUrl',
        'getDataSourceUrl',
        'getDataSourceMetricsUrl'
    ]

    print(f"\n🎯 API端点定义检查:")
    endpoints_score = 0
    for endpoint in api_endpoints:
        found = endpoint in content
        status = "✅" if found else "❌"
        print(f"  {status} {endpoint}")
        if found:
            endpoints_score += 1

    # 3. 检查HTTP请求实现
    http_methods = ['fetch(', 'method:', 'headers:', 'body:']
    print(f"\n📡 HTTP请求实现检查:")
    http_score = 0
    for method in http_methods:
        found = method in content
        status = "✅" if found else "❌"
        print(f"  {status} {method.replace('(', '').replace(':', '')}")
        if found:
            http_score += 1

    # 4. 检查错误处理
    error_handling = [
        ('try-catch块', 'try {' in content and 'catch' in content),
        ('网络错误处理', 'NetworkError' in content or 'fetch' in content),
        ('超时处理', 'AbortController' in content or 'timeout' in content),
        ('HTTP状态码检查', 'response.ok' in content or 'response.status' in content)
    ]

    print(f"\n🛡️ 错误处理机制检查:")
    error_score = 0
    for check_name, condition in error_handling:
        status = "✅" if condition else "❌"
        print(f"  {status} {check_name}")
        if condition:
            error_score += 1

    # 5. 检查数据处理
    data_processing = [
        ('JSON解析', 'response.json()' in content),
        ('数据验证', 'data.data_sources' in content or 'data.data' in content),
        ('数据过滤', 'filter(' in content),
        ('数据渲染', 'renderDataSources' in content)
    ]

    print(f"\n📊 数据处理逻辑检查:")
    data_score = 0
    for check_name, condition in data_processing:
        status = "✅" if condition else "❌"
        print(f"  {status} {check_name}")
        if condition:
            data_score += 1

    # 6. 检查API调用模式
    api_call_patterns = [
        ('GET请求', 'GET' in content),
        ('POST请求', 'POST' in content),
        ('PUT请求', 'PUT' in content),
        ('DELETE请求', 'DELETE' in content)
    ]

    print(f"\n🔄 API调用模式检查:")
    call_score = 0
    for check_name, condition in api_call_patterns:
        status = "✅" if condition else "❌"
        print(f"  {status} {check_name}")
        if condition:
            call_score += 1

    # 7. 检查缓存和重试机制
    caching_mechanisms = [
        ('请求缓存控制', 'Cache-Control' in content),
        ('重试逻辑', 'retryCount' in content or 'maxRetries' in content),
        ('超时控制', 'setTimeout' in content and 'controller.abort' in content),
        ('加载状态管理', 'isLoadingDataSources' in content)
    ]

    print(f"\n💾 缓存和重试机制检查:")
    cache_score = 0
    for check_name, condition in caching_mechanisms:
        status = "✅" if condition else "❌"
        print(f"  {status} {check_name}")
        if condition:
            cache_score += 1

    # 计算总分
    total_score = api_config_score + endpoints_score + http_score + error_score + data_score + call_score + cache_score
    max_score = len(api_base_checks) + len(api_endpoints) + len(http_methods) + len(error_handling) + len(data_processing) + len(api_call_patterns) + len(caching_mechanisms)

    # 总结
    print(f"\n🎯 测试总结:")
    print(f"- API配置: {api_config_score}/{len(api_base_checks)}")
    print(f"- 端点定义: {endpoints_score}/{len(api_endpoints)}")
    print(f"- HTTP请求: {http_score}/{len(http_methods)}")
    print(f"- 错误处理: {error_score}/{len(error_handling)}")
    print(f"- 数据处理: {data_score}/{len(data_processing)}")
    print(f"- API调用: {call_score}/{len(api_call_patterns)}")
    print(f"- 缓存重试: {cache_score}/{len(caching_mechanisms)}")
    print(f"- 总分: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")

    # 判断是否通过
    essential_checks = [
        api_config_score >= 3,  # 至少3个API配置检查通过
        endpoints_score >= 2,   # 至少2个端点定义
        http_score >= 3,        # 至少3个HTTP功能
        error_score >= 2,       # 至少2个错误处理
        data_score >= 3,        # 至少3个数据处理
        call_score >= 2,        # 至少2个API调用模式
        cache_score >= 2        # 至少2个缓存机制
    ]

    success = all(essential_checks)

    if success:
        print("\n🎉 API连接功能测试通过！")
        return True
    else:
        print("\n⚠️ API连接功能测试失败！")
        return False

if __name__ == "__main__":
    import sys
    success = test_api_connection()
    sys.exit(0 if success else 1)

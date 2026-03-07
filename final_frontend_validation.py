#!/usr/bin/env python3
"""
最终验证前端数据源配置页面修复结果
"""

import requests
import json
import re

def validate_frontend_fixes():
    """验证前端修复的完整性"""
    print("🎯 最终验证前端数据源配置修复")
    print("=" * 60)

    # 1. 检查服务状态
    print("1. 检查服务状态...")
    services_ok = True

    try:
        backend_resp = requests.get('http://localhost:8000/health', timeout=5)
        frontend_resp = requests.get('http://localhost:8080/data-sources-config.html', timeout=5)
        api_resp = requests.get('http://localhost:8000/api/v1/data/sources', timeout=5)

        backend_ok = backend_resp.status_code == 200
        frontend_ok = frontend_resp.status_code == 200
        api_ok = api_resp.status_code == 200

        print(f"   {'✅' if backend_ok else '❌'} 后端API服务: {backend_resp.status_code}")
        print(f"   {'✅' if frontend_ok else '❌'} 前端页面服务: {frontend_resp.status_code}")
        print(f"   {'✅' if api_ok else '❌'} 数据源API: {api_resp.status_code}")

        services_ok = backend_ok and frontend_ok and api_ok

    except Exception as e:
        print(f"   ❌ 服务检查失败: {e}")
        services_ok = False

    if not services_ok:
        print("\n❌ 服务状态异常，无法进行完整验证")
        return False

    # 2. 验证前端页面内容
    print("\n2. 验证前端页面修复...")

    page_content = frontend_resp.text
    fixes_validated = []

    # 检查API URL生成函数
    api_functions = [
        'getApiBaseUrl',
        'getDataSourcesUrl',
        'getDataSourceUrl',
        'getDataSourceMetricsUrl'
    ]

    print("   API URL函数检查:")
    for func in api_functions:
        if func in page_content:
            print(f"   ✅ {func}")
            fixes_validated.append(f"{func}存在")
        else:
            print(f"   ❌ {func}缺失")
            fixes_validated.append(f"{func}缺失")

    # 检查环境检测逻辑
    env_checks = [
        'window.location.hostname === \'localhost\'',
        'window.location.protocol === \'file:\''
    ]

    print("\n   环境检测逻辑检查:")
    for check in env_checks:
        if check in page_content:
            print(f"   ✅ {check}")
            fixes_validated.append(f"环境检测:{check}")
        else:
            print(f"   ❌ {check}缺失")

    # 检查硬编码API路径是否已被替换
    hardcoded_patterns = [
        r'http://localhost:8000/api/v1/data/sources/[^/"]*',
        r'/api/v1/data/sources/[^/"]*'
    ]

    print("\n   硬编码API路径检查:")
    remaining_hardcoded = []

    for pattern in hardcoded_patterns:
        matches = re.findall(pattern, page_content)
        if matches:
            # 过滤掉函数定义中的示例
            real_hardcoded = [m for m in matches if 'function' not in page_content[page_content.find(m)-50:page_content.find(m)]]
            if real_hardcoded:
                remaining_hardcoded.extend(real_hardcoded)

    if remaining_hardcoded:
        print(f"   ⚠️ 发现 {len(remaining_hardcoded)} 个可能的硬编码路径")
        for hc in remaining_hardcoded[:3]:  # 只显示前3个
            print(f"      - {hc}")
    else:
        print("   ✅ 未发现硬编码API路径")
        fixes_validated.append("无硬编码API路径")

    # 3. 测试实际功能
    print("\n3. 测试实际功能操作...")

    api_data = api_resp.json()
    sources = api_data.get('data_sources', [])

    if sources:
        test_source = sources[0]
        source_id = test_source['id']

        operations = [
            ("数据源列表", "GET", f"http://localhost:8000/api/v1/data/sources"),
            ("数据源详情", "GET", f"http://localhost:8000/api/v1/data/sources/{source_id}"),
            ("数据源编辑", "PUT", f"http://localhost:8000/api/v1/data/sources/{source_id}"),
            ("连接测试", "POST", f"http://localhost:8000/api/v1/data/sources/{source_id}/test"),
        ]

        print("   API操作测试:")
        operations_ok = True

        for op_name, method, url in operations:
            try:
                if method == "GET":
                    resp = requests.get(url, timeout=5)
                elif method == "PUT":
                    # 发送简单的更新请求
                    update_data = test_source.copy()
                    update_data['name'] = test_source['name'] + '（验证）'
                    resp = requests.put(url, json=update_data, timeout=5)
                elif method == "POST":
                    resp = requests.post(url, timeout=10)

                status = "✅" if resp.status_code == 200 else "❌"
                print(f"   {status} {op_name}: {resp.status_code}")

                if resp.status_code != 200:
                    operations_ok = False

            except Exception as e:
                print(f"   ❌ {op_name}: 异常 - {e}")
                operations_ok = False

        if operations_ok:
            fixes_validated.append("所有API操作正常")

        # 4. 创建测试数据源进行删除测试
        print("\n   删除操作测试:")
        test_data = {
            'id': 'validation_test_source',
            'name': '验证测试数据源',
            'type': '财经新闻',
            'url': 'https://validation.test.com',
            'rate_limit': '15次/分钟',
            'enabled': False
        }

        # 创建
        create_resp = requests.post('http://localhost:8000/api/v1/data/sources', json=test_data, timeout=5)
        if create_resp.status_code == 200:
            print("   ✅ 创建测试数据源成功")

            # 删除
            delete_resp = requests.delete('http://localhost:8000/api/v1/data/sources/validation_test_source', timeout=5)
            if delete_resp.status_code == 200:
                print("   ✅ 删除测试数据源成功")
                fixes_validated.append("删除操作正常")
            else:
                print(f"   ❌ 删除测试失败: {delete_resp.status_code}")
        else:
            print(f"   ❌ 创建测试数据源失败: {create_resp.status_code}")

    else:
        print("   ⚠️ 没有数据源可用于功能测试")

    # 5. 总结
    print("\n" + "=" * 60)
    print("🎯 修复验证结果:")

    all_fixes = [
        "API URL函数存在",
        "环境检测逻辑存在",
        "无硬编码API路径",
        "所有API操作正常",
        "删除操作正常"
    ]

    validated_count = 0
    for fix in all_fixes:
        status = "✅" if any(fix in v for v in fixes_validated) else "❌"
        print(f"   {status} {fix}")
        if any(fix in v for v in fixes_validated):
            validated_count += 1

    total_fixes = len(all_fixes)
    score = validated_count / total_fixes * 100

    print("\n📊 总体评分: {:.1f}%".format(score))
    if score >= 80:
        print("\n🎉 前端数据源配置修复成功！")
        print("\n📋 用户现在可以正常使用:")
        print("   🌐 访问: http://localhost:8080/data-sources-config.html")
        print("   ✏️ 编辑数据源配置")
        print("   🔍 测试数据源连接")
        print("   ➕ 添加新的数据源")
        print("   🗑️ 删除不需要的数据源")
        print("   💾 保存所有修改")

        print("\n💡 如果仍有问题:")
        print("   1. 刷新浏览器页面 (Ctrl+F5)")
        print("   2. 清除浏览器缓存")
        print("   3. 检查浏览器开发者工具的控制台错误")

        return True
    else:
        print("\n⚠️ 修复不完整，可能仍存在问题")
        print("   请检查上述失败的项目")
        return False

if __name__ == "__main__":
    success = validate_frontend_fixes()
    exit(0 if success else 1)

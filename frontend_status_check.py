#!/usr/bin/env python3
"""
前端数据源配置状态检查脚本
快速验证前端功能是否正常工作
"""

import requests
import json

def check_frontend_status():
    """检查前端数据源配置状态"""
    print("🔍 前端数据源配置状态检查")
    print("=" * 40)

    # 1. 检查服务状态
    print("1. 检查服务状态...")
    services = {
        "后端API": "http://localhost:8000/health",
        "前端页面": "http://localhost:8080/data-sources-config.html",
        "数据源API": "http://localhost:8000/api/v1/data/sources"
    }

    service_status = {}
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            service_status[name] = response.status_code == 200
            status_icon = "✅" if service_status[name] else "❌"
            print(f"   {status_icon} {name}: {response.status_code}")
        except Exception as e:
            service_status[name] = False
            print(f"   ❌ {name}: 连接失败")

    all_services_ok = all(service_status.values())

    if not all_services_ok:
        print("\n❌ 服务状态异常，请先启动相关服务:")
        print("   python scripts/start_production.py")
        print("   cd web-static && python -m http.server 8080")
        return False

    # 2. 检查数据源状态
    print("\n2. 检查数据源状态...")
    try:
        response = requests.get("http://localhost:8000/api/v1/data/sources", timeout=5)
        data = response.json()
        sources = data.get('data_sources', [])

        print(f"   📊 数据源数量: {len(sources)}")
        if sources:
            for i, source in enumerate(sources[:3], 1):  # 只显示前3个
                name = source.get('name', 'Unknown')
                enabled = "启用" if source.get('enabled', False) else "禁用"
                print(f"      {i}. {name} ({enabled})")

            if len(sources) > 3:
                print(f"      ... 还有 {len(sources) - 3} 个数据源")
        else:
            print("   ⚠️ 没有配置数据源")
    except Exception as e:
        print(f"   ❌ 获取数据源失败: {e}")
        return False

    # 3. 快速功能测试
    print("\n3. 快速功能测试...")
    if sources:
        enabled_sources = [s for s in sources if s.get('enabled', False)]
        test_source = sources[0]
        source_id = test_source['id']

        # 测试编辑功能
        update_data = test_source.copy()
        update_data['name'] = test_source['name'] + '（状态检查）'

        try:
            edit_response = requests.put(
                f"http://localhost:8000/api/v1/data/sources/{source_id}",
                json=update_data,
                timeout=5
            )
            edit_ok = edit_response.status_code == 200
            print(f"   {'✅' if edit_ok else '❌'} 编辑功能: {edit_response.status_code}")
        except Exception as e:
            print(f"   ❌ 编辑功能: 异常 - {str(e)[:30]}")
            edit_ok = False

        # 测试删除功能（创建临时数据源）
        try:
            temp_data = {
                'id': 'status_check_temp',
                'name': '状态检查临时数据源',
                'type': '财经新闻',
                'url': 'https://status.check.com',
                'rate_limit': '5次/分钟',
                'enabled': False
            }

            # 创建
            create_resp = requests.post("http://localhost:8000/api/v1/data/sources", json=temp_data, timeout=5)

            if create_resp.status_code == 200:
                # 删除
                delete_resp = requests.delete("http://localhost:8000/api/v1/data/sources/status_check_temp", timeout=5)
                delete_ok = delete_resp.status_code == 200
                print(f"   {'✅' if delete_ok else '❌'} 删除功能: {delete_resp.status_code}")
            else:
                print(f"   ❌ 删除功能: 创建临时数据源失败 ({create_resp.status_code})")
                delete_ok = False

        except Exception as e:
            print(f"   ❌ 删除功能: 异常 - {str(e)[:30]}")
            delete_ok = False

        functions_ok = edit_ok and delete_ok

        # 测试禁用功能
        if enabled_sources:
            test_source = enabled_sources[0]
            source_id = test_source['id']

            # 禁用测试
            disable_data = test_source.copy()
            disable_data['enabled'] = False

            disable_resp = requests.put(
                f"http://localhost:8000/api/v1/data/sources/{source_id}",
                json=disable_data,
                timeout=5
            )

            disable_ok = disable_resp.status_code == 200
            print(f"   {'✅' if disable_ok else '❌'} 禁用功能: {disable_resp.status_code}")

            if disable_ok:
                functions_ok = functions_ok and disable_ok

                # 恢复启用状态
                enable_data = disable_data.copy()
                enable_data['enabled'] = True
                requests.put(
                    f"http://localhost:8000/api/v1/data/sources/{source_id}",
                    json=enable_data,
                    timeout=5
                )

    else:
        print("   ⚠️ 跳过功能测试（无数据源）")
        functions_ok = True

    # 4. 前端页面验证
    print("\n4. 前端页面验证...")
    try:
        page_response = requests.get("http://localhost:8080/data-sources-config.html", timeout=5)
        page_content = page_response.text

        checks = [
            ("API URL函数", "getDataSourcesUrl" in page_content),
            ("环境适配", "window.location.hostname === 'localhost'" in page_content),
            ("无硬编码路径", "http://localhost:8000/api/v1/data/sources/" not in page_content)
        ]

        page_ok = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                page_ok = False

        frontend_ok = page_ok

    except Exception as e:
        print(f"   ❌ 前端页面验证失败: {e}")
        frontend_ok = False

    # 5. 总结
    print("\n" + "=" * 40)
    print("🎯 检查结果:")

    checks = [
        ("服务运行状态", all_services_ok),
        ("数据源配置", len(sources) > 0),
        ("CRUD功能", functions_ok),
        ("前端页面修复", frontend_ok)
    ]

    all_ok = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\n🎉 前端数据源配置完全正常！")
        print("\n📋 当前状态:")
        print(f"   🌐 前端页面: http://localhost:8080/data-sources-config.html")
        print(f"   🔧 数据源数量: {len(sources)}")
        print("   ✅ 所有功能正常工作")
        print("\n💡 提示: 可以在浏览器中直接使用数据源配置功能")

        return True
    else:
        print("\n⚠️ 发现问题，请检查上述失败的项目")
        print("\n🔧 可能的解决方案:")
        print("   1. 确保后端服务正在运行 (端口8000)")
        print("   2. 确保前端服务正在运行 (端口8080)")
        print("   3. 刷新浏览器页面 (Ctrl+F5)")
        print("   4. 清除浏览器缓存")

        return False

if __name__ == "__main__":
    success = check_frontend_status()
    exit(0 if success else 1)

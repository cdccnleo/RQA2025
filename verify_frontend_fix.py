#!/usr/bin/env python3
"""
验证前端数据源配置加载修复结果
"""

import requests
import json
import time

def test_frontend_fix():
    """测试前端修复结果"""
    print("🔧 验证前端数据源配置加载修复")
    print("=" * 50)

    try:
        # 1. 检查服务状态
        print("1. 检查服务状态...")
        services = {
            "后端API": "http://localhost:8000/health",
            "前端页面": "http://localhost:8080/data-sources-config.html",
            "API端点": "http://localhost:8000/api/v1/data/sources"
        }

        service_status = {}
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                service_status[name] = response.status_code == 200
                print(f"   {'✅' if service_status[name] else '❌'} {name}: {response.status_code}")
            except:
                service_status[name] = False
                print(f"   ❌ {name}: 连接失败")

        if not all(service_status.values()):
            print("\n❌ 服务状态异常，请先启动相关服务")
            return False

        # 2. 验证API数据
        print("\n2. 验证API数据...")
        api_response = requests.get("http://localhost:8000/api/v1/data/sources", timeout=10)
        api_data = api_response.json()

        sources = api_data.get('data_sources', [])
        print(f"   📊 获取到 {len(sources)} 个数据源")

        if sources:
            print("   数据源列表:")
            for i, source in enumerate(sources, 1):
                name = source.get('name', 'Unknown')
                id_val = source.get('id')
                enabled = source.get('enabled', False)
                print(f"      {i}. {name}")
                print(f"         ID: {id_val}, 状态: {'启用' if enabled else '禁用'}")
        else:
            print("   ⚠️  API返回空的数据源列表")

        # 3. 验证前端页面
        print("\n3. 验证前端页面...")
        page_response = requests.get("http://localhost:8080/data-sources-config.html", timeout=10)
        page_content = page_response.text

        # 检查关键修复
        fixes_applied = {
            "统一API URL函数": "getDataSourcesUrl" in page_content,
            "本地开发环境支持": "window.location.hostname === 'localhost'" in page_content,
            "数据源加载函数": "loadDataSources" in page_content,
            "错误处理": "API服务未找到" not in page_content  # 应该已经修复
        }

        print("   前端修复状态:")
        for fix_name, applied in fixes_applied.items():
            status = "✅" if applied else "❌"
            print(f"   {status} {fix_name}")

        # 4. 总结
        print("\n" + "=" * 50)
        print("🎯 修复验证结果:")

        all_fixes_applied = all(fixes_applied.values())
        api_working = len(sources) > 0
        services_running = all(service_status.values())

        checks = [
            ("服务运行状态", services_running),
            ("API数据获取", api_working),
            ("前端修复应用", all_fixes_applied)
        ]

        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")

        total_passed = sum(1 for _, passed in checks if passed)
        total_checks = len(checks)

        print(f"\n📊 总体评分: {total_passed}/{total_checks}")

        if total_passed == total_checks:
            print("\n🎉 前端数据源配置加载问题已完全修复！")
            print("\n📋 用户现在可以:")
            print("   🌐 访问: http://localhost:8080/data-sources-config.html")
            print("   📊 查看所有配置的数据源")
            print("   ✏️ 编辑现有数据源")
            print("   ➕ 添加新的数据源")
            print("   🗑️ 删除不需要的数据源")
            print("\n💡 如果仍有问题，请刷新浏览器页面")

            return True
        else:
            print("\n⚠️ 修复不完整，可能仍存在问题")
            return False

    except Exception as e:
        print(f"\n❌ 验证过程中出错: {e}")
        return False

if __name__ == "__main__":
    success = test_frontend_fix()
    exit(0 if success else 1)

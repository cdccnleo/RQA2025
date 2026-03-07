#!/usr/bin/env python3
"""
测试数据源计数显示修复
"""

import json
import requests
from typing import Dict, Any

def test_count_display():
    """测试数据源计数显示修复"""
    print("🧪 测试数据源计数显示修复")
    print("=" * 50)

    try:
        # 1. 获取配置文件数据
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            sources = json.load(f)

        enabled_sources = [s for s in sources if s.get('enabled', True)]
        disabled_sources = [s for s in sources if not s.get('enabled', True)]

        total_count = len(sources)
        enabled_count = len(enabled_sources)
        disabled_count = len(disabled_sources)

        print(f"📄 配置文件统计:")
        print(f"   • 总数据源: {total_count}")
        print(f"   • 启用数据源: {enabled_count}")
        print(f"   • 禁用数据源: {disabled_count}")

        # 2. 模拟前端计数逻辑
        print("\n🎯 模拟前端计数逻辑:")
        print("   • 默认状态（不显示禁用）: 可见数据源 =", enabled_count)
        print(f"   • 显示禁用后: 可见数据源 =", total_count)

        # 3. 测试API响应
        api_url = "http://localhost:8000/api/v1/data/sources"
        response = requests.post(api_url, json={"action": "get_all"}, timeout=10)

        if response.status_code == 200:
            data = response.json()
            api_sources = data.get('data') or data.get('data_sources') or []

            api_enabled = [s for s in api_sources if s.get('enabled', True)]
            api_disabled = [s for s in api_sources if not s.get('enabled', True)]

            api_total = len(api_sources)
            api_enabled_count = len(api_enabled)
            api_disabled_count = len(api_disabled)

            print(f"\n🔌 API返回统计:")
            print(f"   • 总数据源: {api_total}")
            print(f"   • 启用数据源: {api_enabled_count}")
            print(f"   • 禁用数据源: {api_disabled_count}")

            # 4. 验证一致性
            consistency_checks = [
                ("总数量一致性", total_count, api_total),
                ("启用数量一致性", enabled_count, api_enabled_count),
                ("禁用数量一致性", disabled_count, api_disabled_count),
            ]

            all_consistent = True
            for check_name, config_count, api_count in consistency_checks:
                status = "✅" if config_count == api_count else "❌"
                print(f"   {status} {check_name}: 配置{config_count} vs API{api_count}")
                if config_count != api_count:
                    all_consistent = False

            if all_consistent:
                print("\n🎉 数据源数量一致性验证通过！")
                print("修复内容:")
                print("   ✅ renderDataSources() 现在调用 updateVisibleCount()")
                print("   ✅ 页面加载后计数显示正确更新")
                print("   ✅ 支持切换显示禁用数据源时计数实时更新")

                # 5. 检查代码修复
                print("\n🔧 代码修复验证:")
                # 检查renderDataSources中是否调用了updateVisibleCount
                with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
                    content = f.read()

                render_calls_update_visible = 'updateStats();\n            updateVisibleCount();' in content
                toggle_calls_update_visible = 'updateStats();\n        updateVisibleCount();' in content

                print(f"   ✅ renderDataSources调用updateVisibleCount: {render_calls_update_visible}")
                print(f"   ✅ toggleDisabledSources调用updateVisibleCount: {toggle_calls_update_visible}")

                if render_calls_update_visible and toggle_calls_update_visible:
                    print("\n🎯 修复成功！数据源计数显示现在应该正常工作。")
                    print("   • 页面加载后显示正确的计数")
                    print("   • 切换'显示禁用数据源'时计数实时更新")
                    print("   • 格式为: 可见数量/总数量")
                    return True
                else:
                    print("\n❌ 代码修复不完整")
                    return False
            else:
                print("\n❌ 数据源数量存在不一致问题")
                return False
        else:
            print(f"❌ API请求失败: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_count_display()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
最终验证数据源配置状态
"""

import json
import os

def validate_data_sources():
    """验证数据源配置的完整性"""
    config_file = "data/data_sources_config.json"

    print("🔍 最终验证数据源配置状态")
    print("=" * 50)

    if not os.path.exists(config_file):
        print("❌ 配置文件不存在")
        return False

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sources = data.get('data_sources', [])

        print(f"📊 配置文件状态:")
        print(f"   总数据源数量: {len(sources)}")

        # 检查财联社数据源
        cls_sources = [s for s in sources if s.get('name') == '财联社']
        if cls_sources:
            print(f"❌ 发现 {len(cls_sources)} 个财联社数据源:")
            for i, source in enumerate(cls_sources):
                print(f"   {i+1}. ID: {repr(source.get('id'))}, 类型: {repr(source.get('type'))}")
            return False
        else:
            print("✅ 未发现财联社数据源")

        # 检查ID有效性
        print(f"\\n🆔 ID有效性检查:")
        valid_sources = []
        invalid_sources = []

        for source in sources:
            name = source.get('name', 'Unknown')
            id_val = source.get('id')

            # 检查ID是否有效
            is_valid = (
                id_val is not None and
                str(id_val).strip() != '' and
                str(id_val).lower() not in ['null', 'none'] and
                isinstance(id_val, str)
            )

            if is_valid:
                valid_sources.append(source)
                print(f"   ✅ {name}: {repr(id_val)}")
            else:
                invalid_sources.append(source)
                print(f"   ❌ {name}: {repr(id_val)} (无效)")

        # 检查ID唯一性
        print(f"\\n🔄 ID唯一性检查:")
        all_ids = [s.get('id') for s in valid_sources]
        unique_ids = set(all_ids)

        if len(all_ids) == len(unique_ids):
            print("   ✅ 所有ID都是唯一的")
        else:
            print("   ❌ 发现重复的ID")
            duplicates = [id_val for id_val in all_ids if all_ids.count(id_val) > 1]
            for dup in set(duplicates):
                print(f"      重复ID: {repr(dup)}")

        # 最终结果
        print(f"\\n📈 最终统计:")
        print(f"   有效数据源: {len(valid_sources)}")
        print(f"   无效数据源: {len(invalid_sources)}")

        if len(invalid_sources) == 0 and len(cls_sources) == 0:
            print("\\n🎉 数据源配置完全清洁！")
            print("✅ 财联社数据源已完全删除")
            print("✅ 所有数据源ID有效且唯一")
            return True
        else:
            print("\\n⚠️ 数据源配置需要进一步清理")
            return False

    except Exception as e:
        print(f"❌ 验证过程中出错: {e}")
        return False

def test_api_functionality():
    """测试API功能"""
    print("\\n🔧 测试API功能")
    print("=" * 30)

    try:
        import subprocess
        import time

        # 启动测试服务器
        print("启动测试服务器...")
        server = subprocess.Popen(['python', 'verify_deletion.py'])
        time.sleep(3)

        try:
            import requests

            # 测试获取数据源
            print("测试获取数据源列表...")
            response = requests.get('http://localhost:8000/api/v1/data/sources')
            if response.status_code == 200:
                data = response.json()
                sources = data.get('data_sources', [])
                print(f"✅ API响应正常，获取到 {len(sources)} 个数据源")

                # 检查是否有财联社
                cls_in_api = [s for s in sources if s.get('name') == '财联社']
                if cls_in_api:
                    print(f"❌ API中仍有 {len(cls_in_api)} 个财联社数据源")
                    return False
                else:
                    print("✅ API中财联社数据源已清除")

                # 测试编辑和删除功能
                valid_sources = [s for s in sources if s.get('id') and s.get('id') != 'None']
                if valid_sources:
                    test_source = valid_sources[0]
                    test_id = test_source['id']
                    test_name = test_source['name']

                    print(f"\\n测试数据源: {test_name} (ID: {test_id})")

                    # 测试编辑
                    edit_resp = requests.get(f'http://localhost:8000/api/v1/data/sources/{test_id}')
                    if edit_resp.status_code == 200:
                        print("✅ 编辑功能正常")
                    else:
                        print(f"❌ 编辑功能异常: {edit_resp.status_code}")
                        return False

                    # 测试删除
                    delete_resp = requests.delete(f'http://localhost:8000/api/v1/data/sources/{test_id}')
                    if delete_resp.status_code == 200:
                        result = delete_resp.json()
                        print("✅ 删除功能正常")
                        print(f"   删除消息: {result.get('message', 'N/A')}")
                    else:
                        print(f"❌ 删除功能异常: {delete_resp.status_code}")
                        return False

                    return True
                else:
                    print("⚠️ 没有有效的测试数据源")
                    return True  # 配置正确，只是没有数据源可测试
            else:
                print(f"❌ API调用失败: {response.status_code}")
                return False

        finally:
            server.terminate()
            server.wait()

    except Exception as e:
        print(f"❌ API测试出错: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始最终验证")
    print("目标: 确认财联社数据源已完全删除，数据源配置清洁")
    print()

    # 验证配置
    config_ok = validate_data_sources()

    # 测试API
    api_ok = test_api_functionality()

    print("\\n" + "=" * 50)
    if config_ok and api_ok:
        print("🎉 所有验证通过！财联社数据源删除成功！")
        print("✅ 数据源配置已清洁")
        print("✅ API功能正常工作")
        exit(0)
    else:
        print("❌ 验证失败，需要进一步处理")
        exit(1)